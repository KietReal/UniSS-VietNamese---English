import os
import sys
import glob
import yaml
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed

# Ensure proper paths are in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
s2st_dir = os.path.dirname(current_dir)
spark_tts_dir = os.path.abspath(os.path.join(s2st_dir, "SparkTTS"))

if spark_tts_dir not in sys.path:
    sys.path.insert(0, spark_tts_dir)
if s2st_dir not in sys.path:
    sys.path.insert(0, s2st_dir)

from Dataset.Phase2 import Phase2_S2ST, collate_fn_phase2
from models.s2st_model import S2STModel
from utils.metric.Phase2 import Phase2Metrics
from utils.tracker import CheckpointTracker


def evaluate_task(accelerator, s2st_model, metric, cfg):
    """Evaluates the model on a specific task."""
    batch_size = 8
    num_workers = int(cfg['training'].get('num_workers', 4))
    
    # Initialize the eval dataset for this specific task
    eval_dataset = Phase2_S2ST(
        files_csv=cfg['data']['valid_csv'],
        tokenizer=s2st_model.envit5_tokenizer,
        split='valid',
        sr=int(cfg['data']['sample_rate']),
    )

    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn_phase2, num_workers=num_workers, 
        pin_memory=True, persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    eval_dataloader = accelerator.prepare(eval_dataloader)
    s2st_model.eval()
    
    unwrapped_model = accelerator.unwrap_model(s2st_model)
    
    val_pbar = tqdm(
        eval_dataloader,
        disable=not accelerator.is_local_main_process,
    )

    with torch.no_grad():
        for batch in val_pbar:
            audio_prediction, audio_labels, global_tokens = s2st_model(
                    A_encode=batch['audio_encode'],
                    A_decode=batch['audio_decode'],
                    special_tokens=batch['special_tokens'],
                    phase=2,
                )

            metric.add_batch(
                text_target=batch['VieEng_Text'],
                global_tokens=global_tokens,
                audio_prediction=audio_prediction,
                audio_target=audio_labels,
            )

            del audio_prediction, audio_labels, global_tokens
            
    result = metric.compute()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return result


def main():
    config_path = os.path.join(s2st_dir, 'train_config.yaml')
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    accelerator = Accelerator(mixed_precision=cfg['training'].get('mixed_precision', 'fp16'))
    
    # Model
    s2st_model = S2STModel(
        envit5_path=cfg['paths'].get('envit5'),
        sparkTTS_path=cfg['paths'].get('sparktts'),
    )
    
    s2st_model = accelerator.prepare(s2st_model)
    unwrapped_model = accelerator.unwrap_model(s2st_model)
    unwrapped_model._move_sparkTTS_to_device()

    # Metric
    metric = Phase2Metrics(
        device=str(unwrapped_model.device),
        text_tokenizer=unwrapped_model.envit5_tokenizer,
        audio_tokenizer=unwrapped_model.sparkTTS,
    )
    tracker = CheckpointTracker()
    accelerator.register_for_checkpointing(tracker)
    
    # Find checkpoints
    ckpt_dir = cfg['paths'].get('checkpoint_dir_2', 'models/checkpoints/phase2')
    checkpoint_folders = glob.glob(os.path.join(ckpt_dir, "epoch_*"))
    
    # Sort checkpoints by epoch number
    def get_epoch_num(path):
        basename = os.path.basename(path)
        try:
            return int(basename.split('_')[1])
        except:
            return -1
            
    checkpoint_folders.sort(key=get_epoch_num)
    
    if not checkpoint_folders:
        accelerator.print(f"No epoch checkpoints found in {ckpt_dir}")
        return
    
    csv_file = os.path.join('/content/drive/MyDrive/S2ST_data/S2ST/models/s2st', "Evaluation", "evaluation_results_phase2.csv")
    results_list = []

    accelerator.print("\n" + "=" * 60)
    accelerator.print(f" Starting Evaluation for {len(checkpoint_folders)} Checkpoints")
    accelerator.print("=" * 60 + "\n")

    for ckpt_path in checkpoint_folders:

        epoch_name = os.path.basename(ckpt_path)
        accelerator.print(f"\n--> [Evaluating Checkpoint: '{epoch_name}']")
        
        # Load weights
        accelerator.load_state(ckpt_path)
        
        res = evaluate_task(accelerator, s2st_model, metric, cfg)
        
        vi_spk_wer        = res["vi_spk_wer"]
        vi_utmos          = res["vi_utmos"]
        vi_spk_similarity = res["vi_spk_similarity"]
        

        en_spk_wer        = res["en_spk_wer"]
        en_utmos          = res["en_utmos"]
        en_spk_similarity = res["en_spk_similarity"]

        epoch_result = {
            "Checkpoint" : epoch_name,
            "VIE_SPK_WER": f"{vi_spk_wer:.4f}",
            "VIE_UTMOS"  : f"{vi_utmos:.4f}",
            "VIE_SIM_Speaker": f"{vi_spk_similarity:.4f}",

            "ENG_SPK_WER": f"{en_spk_wer:.4f}",
            "ENG_UTMOS"  : f"{en_utmos:.4f}",
            "ENG_SIM_Speaker": f"{en_spk_similarity:.4f}",
        }
        results_list.append(epoch_result)
        
        accelerator.print(f"Results for {epoch_name}:")
        accelerator.print(epoch_result)
        
        # Save incrementally
        if accelerator.is_local_main_process:
            df = pd.DataFrame(results_list)
            df.to_csv(csv_file, index=False)
            accelerator.print(f"Saved intermediate results to {csv_file}")

    accelerator.print("\n=== Evaluation Completed Successfully! ===")

if __name__ == "__main__":
    main()
