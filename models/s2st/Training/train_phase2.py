import os
import sys

# Ensure SparkTTS directory is in sys.path before importing S2STModel or Dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
spark_tts_dir = os.path.abspath(os.path.join(current_dir, "../SparkTTS"))
if spark_tts_dir not in sys.path:
    sys.path.insert(0, spark_tts_dir)

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup
from tqdm import tqdm
import bitsandbytes as bnb

from Dataset.Phase2 import Phase2_S2ST, collate_fn_phase2
from models.s2st_model import S2STModel
from utils.audio_utils import download_augmentation
from utils.tracker import CheckpointTracker

def load_phase1_weights(model, phase1_ckpt_path, accelerator):
    
    sf_path = os.path.join(phase1_ckpt_path, "model.safetensors")
    pt_path = os.path.join(phase1_ckpt_path, "pytorch_model.bin")
    
    if os.path.exists(sf_path):
        import safetensors.torch
        state_dict = safetensors.torch.load_file(sf_path)
    elif os.path.exists(pt_path):
        state_dict = torch.load(pt_path, map_location="cpu")
    else:
        accelerator.print(f"Could not find model.safetensors or pytorch_model.bin in {phase1_ckpt_path}")
        return
        
    current_state_dict = model.state_dict()
    merged_state_dict = {}
    
    for k, v in state_dict.items():
        if k in current_state_dict:
            current_v = current_state_dict[k]
            if v.shape != current_v.shape:
                if len(v.shape) == 2 and v.shape[0] < current_v.shape[0] and v.shape[1] == current_v.shape[1]:
                    accelerator.print(f"Resizing mismatch for {k}: Phase 1 {v.shape} -> Phase 2 {current_v.shape}. Merging old weights and keeping new tokens initialized.")
                    new_v = current_v.clone()
                    new_v[:v.shape[0], :] = v
                    merged_state_dict[k] = new_v
                else:
                    accelerator.print(f"WARNING: Unhandled shape mismatch for {k}")
                    merged_state_dict[k] = v 
            else:
                merged_state_dict[k] = v
                
    model.load_state_dict(merged_state_dict, strict=False)
    accelerator.print("Phase 1 weights loaded successfully. Optimizer will start fresh for Phase 2.")


def main():
    # Load Config
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_config.yaml')
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg['training']['seed']))

    # Model
    s2st_model = S2STModel(
        envit5_path=cfg['paths'].get('envit5'),
        sparkTTS_path=cfg['paths'].get('sparktts'),
    )

    # Dataset & Audio Augmentation (RIR + MUSAN Noise)
    rir_recordings, musan_cuts = download_augmentation(root_dir="data/augmentation")

    train_dataset = Phase2_S2ST(
        files_csv=cfg['data']['train_csv'],
        tokenizer=s2st_model.envit5_tokenizer,
        split='train',
        sr=int(cfg['data']['sample_rate']),
        rir_recordings=rir_recordings,
        musan_cuts=musan_cuts,
    )

    # DataLoader
    batch_size = int(cfg['training']['batch_size'])
    num_workers = int(cfg['training'].get('num_workers', 8))

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn_phase2, num_workers=num_workers,
        pin_memory=True, persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # Optimizer (excluding frozen SparkTTS)
    opt_cfg = cfg['optimizer']
    trainable_params = [p for p in s2st_model.parameters() if p.requires_grad]
    optimizer = bnb.optim.AdamW8bit(
        trainable_params,
        lr=float(opt_cfg['lr']),
        weight_decay=float(opt_cfg['weight_decay']),
        betas=tuple(float(b) for b in opt_cfg['betas']),
        eps=float(opt_cfg['eps']),
    )

    # Loss
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=-100,
        label_smoothing=float(cfg['loss']['label_smoothing']),
    )

    # Scheduler
    total_steps = (
        len(train_dataloader)
        // int(cfg['training']['gradient_accumulation_steps'])
        * int(cfg['training']['epochs'])
    )
    scheduler = get_cosine_with_min_lr_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(float(cfg['scheduler']['warmup_ratio']) * total_steps),
        num_training_steps=total_steps,
        min_lr=float(cfg['scheduler']['min_lr'])
    )

    # Accelerator
    wandb_cfg = cfg.get('wandb', {})
    if 'mode' in wandb_cfg and wandb_cfg['mode']:
        os.environ['WANDB_MODE'] = str(wandb_cfg['mode'])

    mixed_precision = cfg['training'].get('mixed_precision', None)
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=cfg['training']['gradient_accumulation_steps'],
    )
    accelerator.init_trackers(
        project_name=cfg['wandb']['project'],
        config=cfg,
        init_kwargs={"wandb": {"name": cfg['wandb']['run_name']}},
    )

    s2st_model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        s2st_model, optimizer, train_dataloader, scheduler
    )

    unwrapped_model = accelerator.unwrap_model(s2st_model)
    unwrapped_model._move_sparkTTS_to_device()

    tracker = CheckpointTracker()
    accelerator.register_for_checkpointing(tracker)

    # Checkpoint Loading & Resuming Logic
    ckpt_dir = cfg['paths'].get('checkpoint_dir_2', 'models/checkpoints/phase2')
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_ckpt_path = os.path.join(ckpt_dir, "latest_checkpoint")

    if os.path.exists(latest_ckpt_path):
        accelerator.print(f"Found existing checkpoint at '{latest_ckpt_path}'. Loading model, optimizer, scheduler, and RNG state...")
        accelerator.load_state(latest_ckpt_path)

        tracker.start_epoch += 1

        accelerator.print(f"Resuming training from Epoch {tracker.start_epoch}")
        accelerator.print(f"Loaded best TTS_WER: {tracker.best_tts_wer:.4f}")
    else:
        # Fallback: load best checkpoint from Phase 1
        phase1_ckpt_dir = cfg['paths'].get('checkpoint_dir_1', 'models/checkpoints/phase1')
        best_ckpt_path = os.path.join(phase1_ckpt_dir, "best_checkpoint")
        if os.path.exists(best_ckpt_path):
            accelerator.print(f"No Phase 2 checkpoint found. Loading ONLY model weights from Phase 1 best checkpoint '{best_ckpt_path}'...")
            load_phase1_weights(unwrapped_model, best_ckpt_path, accelerator)
        else:
            accelerator.print(
                f"WARNING: No Phase 2 checkpoint and no Phase 1 best checkpoint found at '{best_ckpt_path}'.\n"
                f"Starting Phase 2 training from scratch (original EnViT5 & SparkTTS weights)."
            )

    epochs = cfg['training']['epochs']
    max_grad_norm = cfg['training']['max_grad_norm']

    # Calculate global_step for resumed training
    # After resume, tracker.start_epoch is already incremented, so completed epochs = start_epoch - 1
    steps_per_epoch = len(train_dataloader)
    global_step = (tracker.start_epoch - 1) * steps_per_epoch

    for epoch in range(tracker.start_epoch, epochs + 1):
        tracker.start_epoch = epoch
        # Train
        s2st_model.train()
        train_pbar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch}/{epochs} [Train]",
            disable=not accelerator.is_local_main_process,
        )

        for batch in train_pbar:
            with accelerator.accumulate(s2st_model):
                audio_prediction, audio_labels, _ = s2st_model(
                    A_encode=batch['audio_encode'],
                    A_decode=batch['audio_decode'],
                    special_tokens=batch['special_tokens'],
                    phase=2,
                )

                loss = loss_fn(
                    audio_prediction.reshape(-1, audio_prediction.size(-1)),
                    audio_labels.reshape(-1),
                )

                loss_val = loss.item()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(s2st_model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                del audio_prediction, audio_labels, loss

                global_step += 1
                current_lr = scheduler.get_last_lr()[0]
                train_pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{current_lr:.2e}")
                accelerator.log({
                    "train/loss": loss_val,
                    "train/lr": current_lr,
                    "train/epoch": epoch,
                    "train/step": global_step,
                })

        # Save latest_checkpoint
        accelerator.save_state(latest_ckpt_path)

        # Save separate checkpoint
        epoch_ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}")
        accelerator.save_state(epoch_ckpt_path)
        accelerator.print(f"Saved separate checkpoint for Epoch {epoch} at '{epoch_ckpt_path}'")

        # Clear GPU VRAM after epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    accelerator.end_training()
    accelerator.print("\nAll Training Epochs Completed Successfully!")


if __name__ == "__main__":
    main()