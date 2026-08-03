import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import get_cosine_with_min_lr_schedule_with_warmup
from tqdm import tqdm

from Dataset.Phase1 import Phase1_S2ST, collate_fn_phase1
from models.s2st_model import S2STModel
from utils.metric.Phase1 import Phase1Metrics
from utils.tracker import CheckpointTracker


def compute_loss(loss_fn, text_prediction, audio_prediction, labels, audio_labels, sub_tasks):
    """
    Compute combined text + audio loss with proper reshaping for CrossEntropyLoss.
    
    text_prediction: (B, L_model, text_vocab)
    audio_prediction: (B, L_model, audio_vocab)
    labels: (B, L_labels)
    audio_labels: (N_tts, L_audio) or None
    sub_tasks: (B,)
    """
    losses = []

    # Text loss (tasks 1, 2, 4: ASR, S2TT, MT)
    text_mask = sub_tasks != 3
    if text_mask.any():
        text_pred = text_prediction[text_mask]   # (N_text, L_model, vocab)
        text_tgt = labels[text_mask]              # (N_text, L_labels)

        # Reshape for CrossEntropyLoss: (N*L, V) and (N*L,)
        text_pred = text_pred.reshape(-1, text_pred.size(-1))
        text_tgt = text_tgt.reshape(-1)

        text_loss = loss_fn(text_pred, text_tgt)
        losses.append(text_loss)

    # Audio loss (task 3: TTS)
    if audio_labels is not None:
        tts_mask = sub_tasks == 3
        if tts_mask.any():
            audio_pred = audio_prediction[tts_mask]  # (N_tts, L_model, audio_vocab)

            # Reshape for CrossEntropyLoss: (N*L, V) and (N*L,)
            audio_pred = audio_pred.reshape(-1, audio_pred.size(-1))
            audio_tgt = audio_labels.reshape(-1)

            audio_loss = loss_fn(audio_pred, audio_tgt)
            losses.append(audio_loss)

    if len(losses) == 0:
        return torch.tensor(0.0, device=text_prediction.device, requires_grad=True)
    
    return sum(losses) / len(losses)


def main():
    # Load Config 
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_config.yaml')
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg['training']['seed'])

    # Model
    s2st_model = S2STModel(
        envit5_path=cfg['paths'].get('envit5'),
        sparkTTS_path=cfg['paths'].get('sparktts'),
    )

    # Dataset 
    train_dataset = Phase1_S2ST(
        files_csv=cfg['data']['train_csv'],
        tokenizer=s2st_model.envit5_tokenizer,
        split='train',
        sr=cfg['data']['sample_rate'],
    )
    valid_dataset = Phase1_S2ST(
        files_csv=cfg['data']['valid_csv'],
        tokenizer=s2st_model.envit5_tokenizer,
        split='valid',
        sr=cfg['data']['sample_rate'],
    )

    # DataLoader 
    batch_size = cfg['training']['batch_size']
    num_workers = cfg['training'].get('num_workers', 8)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn_phase1, num_workers=num_workers, 
        pin_memory=True, persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn_phase1, num_workers=num_workers, 
        pin_memory=True, persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # Optimizer (excluding frozen SparkTTS)
    opt_cfg = cfg['optimizer']
    trainable_params = [p for p in s2st_model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=opt_cfg['lr'],
        weight_decay=opt_cfg['weight_decay'],
        betas=tuple(opt_cfg['betas']),
        eps=opt_cfg['eps'],
    )

    # Loss
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=-100,
        label_smoothing=cfg['loss']['label_smoothing'],
    )

    # Scheduler 
    total_steps = (
        len(train_dataloader)
        // cfg['training']['gradient_accumulation_steps']
        * cfg['training']['epochs']
    )
    scheduler = get_cosine_with_min_lr_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg['scheduler']['warmup_ratio'] * total_steps),
        num_training_steps=total_steps,
        min_lr=cfg['scheduler']['min_lr']
    )

    # Accelerator 
    accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=cfg['training']['gradient_accumulation_steps'],
    )
    accelerator.init_trackers(
        project_name=cfg['wandb']['project'],
        config=cfg,
        init_kwargs={"wandb": {"name": cfg['wandb']['run_name']}},
    )

    s2st_model, optimizer, train_dataloader, valid_dataloader, scheduler = accelerator.prepare(
        s2st_model, optimizer, train_dataloader, valid_dataloader, scheduler
    )

    unwrapped_model = accelerator.unwrap_model(s2st_model)
    unwrapped_model._move_sparkTTS_to_device()

    # Metric (use the accelerator device)
    metric = Phase1Metrics(
        device=str(unwrapped_model.device),
        text_tokenizer=unwrapped_model.envit5_tokenizer,
        audio_tokenizer=unwrapped_model.sparkTTS,
    )

    tracker = CheckpointTracker()
    accelerator.register_for_checkpointing(tracker)

    # Checkpoint Loading & Resuming Logic 
    ckpt_dir = cfg['paths'].get('checkpoint_dir', 'models/checkpoints/phase1')
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_ckpt_path = os.path.join(ckpt_dir, "latest_checkpoint")
    
    if os.path.exists(latest_ckpt_path):
        accelerator.print(f"Found existing checkpoint at '{latest_ckpt_path}'. Loading model, optimizer, scheduler, and RNG state...")
        accelerator.load_state(latest_ckpt_path)
        
        tracker.start_epoch += 1
        
        accelerator.print(f"Resuming training from Epoch {tracker.start_epoch}")
        accelerator.print(f"Loaded best TTS_WER: {tracker.best_tts_wer:.4f}")
    else:
        accelerator.print("No checkpoint found. Starting training from original EnViT5 & SparkTTS weights.")

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
                text_prediction, audio_prediction, audio_labels = s2st_model(
                    T_encode=batch['text_encode'],
                    T_decode=batch['text_decode'],
                    A_encode=batch['audio_encode'],
                    A_decode=batch['audio_decode'],
                    phase=1,
                    sub_tasks=batch['sub_tasks'],
                )

                loss = compute_loss(
                    loss_fn, text_prediction, audio_prediction,
                    batch['labels'], audio_labels, batch['sub_tasks'],
                )

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(s2st_model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                current_lr = scheduler.get_last_lr()[0]
                train_pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")
                accelerator.log({
                    "train/loss": loss.item(),
                    "train/lr": current_lr,
                    "train/epoch": epoch,
                    "train/step": global_step,
                })

        # Validation 
        s2st_model.eval()
        val_losses = []
        val_pbar = tqdm(
            valid_dataloader,
            desc=f"Epoch {epoch}/{epochs} [Valid]",
            disable=not accelerator.is_local_main_process,
        )

        with torch.no_grad():
            for batch in val_pbar:
                text_prediction, audio_prediction, audio_labels = s2st_model(
                    T_encode=batch['text_encode'],
                    T_decode=batch['text_decode'],
                    A_encode=batch['audio_encode'],
                    A_decode=batch['audio_decode'],
                    phase=1,
                    sub_tasks=batch['sub_tasks'],
                )
                global_token = unwrapped_model.get_global_token_task_3()

                val_loss = compute_loss(
                    loss_fn, text_prediction, audio_prediction,
                    batch['labels'], audio_labels, batch['sub_tasks'],
                )
                val_losses.append(val_loss.item())
                val_pbar.set_postfix(val_loss=f"{val_loss.item():.4f}")

                # Build safe fallback tensors on the correct device
                model_device = unwrapped_model.device
                safe_audio_labels = audio_labels if audio_labels is not None else torch.empty(0, dtype=torch.long, device=model_device)
                safe_audio_pred = audio_prediction if audio_prediction is not None else torch.empty(0, device=model_device)

                metric.add_batch(
                    text_prediction=text_prediction,
                    text_target=batch['labels'],
                    text_input=batch['text_encode'],
                    global_tokens=global_token,
                    audio_prediction=safe_audio_pred,
                    audio_target=safe_audio_labels,
                    sub_tasks=batch['sub_tasks'],
                    task_names=batch['task_names'],
                )

        result = metric.compute()
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0

        # Log validation metrics to WandB
        accelerator.log({
            "val/loss": avg_val_loss,
            "val/asr_wer": result["asr_wer"],
            "val/mt_bleu": result["mt_bleu"],
            "val/utmos": result["utmos"],
            "val/spk_similarity": result["spk_similarity"],
            "val/tts_wer": result["tts_wer"],
            "val/epoch": epoch,
        })

        accelerator.print(
            f"[Epoch {epoch}/{epochs}] "
            f"val_loss={avg_val_loss:.4f} | "
            f"WER={result['asr_wer']:.4f} | "
            f"BLEU={result['mt_bleu']:.2f} | "
            f"UTMOS={result['utmos']:.3f} | "
            f"SPK={result['spk_similarity']:.3f} | "
            f"TTS_WER={result['tts_wer']:.4f}"
        )

        # Save latest_checkpoint 
        accelerator.save_state(latest_ckpt_path)

        # Save best_checkpoint 
        current_tts_wer = result["tts_wer"]
        if current_tts_wer < tracker.best_tts_wer:
            tracker.best_tts_wer = current_tts_wer
            best_ckpt_path = os.path.join(ckpt_dir, "best_checkpoint")
            accelerator.save_state(best_ckpt_path)
            accelerator.print(f"New best TTS_WER={tracker.best_tts_wer:.4f} — saved best_checkpoint")

    accelerator.end_training()

if __name__ == "__main__":
    main()
