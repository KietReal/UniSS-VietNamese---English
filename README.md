# UNISS: Unified Expressive Speech-to-Speech Translation

An implementation of the **UNISS** (Unified Expressive Speech-to-Speech Translation) paper, adapted for **Vietnamese ↔ English** speech-to-speech translation using **EnViT5** as the backbone LLM and **Spark-TTS (BiCodec)** as the audio tokenizer.

## Project Overview

This project builds a unified model that handles four sub-tasks within a single encoder-decoder architecture:

| Sub-task | ID | Encoder Input | Decoder Output |
|----------|-----|---------------|----------------|
| **ASR** (Automatic Speech Recognition) | 1 | Audio + `<ASR>` `<LANG>` | Text transcription |
| **S2TT** (Speech-to-Text Translation) | 2 | Audio + `<S2TT>` `<LANG>` | Translated text |
| **TTS** (Text-to-Speech) | 3 | Text + `<TTS>` `<LANG>` + Speaker embedding | Semantic audio tokens |
| **MT** (Machine Translation) | 4 | Text + `<MT>` `<LANG>` | Translated text |

### Training Phases

The training follows a **3-phase curriculum**:

- **Phase 1** *(Current — Implemented)* — Multi-task pre-training: Train all 4 sub-tasks jointly to build shared representations.
- **Phase 2** *(Planned)* — End-to-end S2ST fine-tuning.
- **Phase 3** *(Planned)* — Expressive and speaker-preserving refinement.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         S2STModel                                   │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ Spark-TTS     │    │   EnViT5      │    │  Custom Heads       │  │
│  │ (BiCodec)     │    │ (T5 Enc-Dec)  │    │                     │  │
│  │               │    │               │    │  ┌───────────────┐  │  │
│  │ ┌───────────┐ │    │  ┌─────────┐  │    │  │  lm_head      │  │  │
│  │ │ Speaker   │ │───▶│  │ Encoder │  │───▶│  │  (text vocab) │  │  │
│  │ │ Encoder   │ │    │  └─────────┘  │    │  └───────────────┘  │  │
│  │ └───────────┘ │    │               │    │                     │  │
│  │ ┌───────────┐ │    │  ┌─────────┐  │    │  ┌───────────────┐  │  │
│  │ │ Semantic  │ │───▶│  │ Decoder │  │───▶│  │  AudioHeader  │  │  │
│  │ │ Quantizer │ │    │  └─────────┘  │    │  │  (8194 vocab) │  │  │
│  │ └───────────┘ │    │               │    │  └───────────────┘  │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐                        │
│  │ SemanticProjector │  │ SpeakerProjector │  (1024 → 768)         │
│  └──────────────────┘  └──────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Description | Trainable? |
|-----------|-------------|:----------:|
| **EnViT5** | Vietnamese-English T5 seq2seq backbone | ✅ Yes |
| **Spark-TTS (BiCodec)** | Audio tokenizer (speaker encoder + semantic quantizer) | ❌ Frozen |
| **SemanticProjector** | 3-layer MLP with residual: 1024 → 768 → 512 → 768 | ✅ Yes |
| **SpeakerProjector** | 3-layer MLP with residual: 1024 → 768 → 512 → 768 | ✅ Yes |
| **AudioHeader** | Linear head: 768 → 8194 (8192 codebook + BOS + EOS) | ✅ Yes |

### Special Tokens

**Text tokens** (added to EnViT5 tokenizer):
- `<S2TT>`, `<MT>`, `<TTS>`, `<ASR>` — Task identifiers
- `<ENG>`, `<VIE>` — Target language identifiers
- `<BOS>` — Beginning of sequence (decoder)

**Audio tokens** (custom IDs beyond codebook):
- `AUDIO_BOS_ID = 8193` — Start of audio sequence
- `AUDIO_EOS_ID = 8192` — End of audio sequence

---

## Directory Structure

```
S2ST/
├── requirements.txt              # Project dependencies
├── data/
│   ├── PhoST.py                  # PhoST dataset preprocessing script
│   └── processing.py             # Text normalization utilities
│
└── models/
    ├── checkpoints/
    │   ├── envit5-translation/    # Pre-trained EnViT5 model
    │   └── Spark-TTS-0.5B/       # Pre-trained Spark-TTS model
    │
    ├── SparkTTS/                  # SparkTTS source code (dependency)
    │   └── sparktts/              # BiCodec tokenizer, modules, etc.
    │
    ├── EnViT5/                    # EnViT5 standalone test environment
    │
    └── s2st/                      # ★ Main project code
        ├── config.yaml            # Model architecture config
        ├── train_config.yaml      # Training hyperparameters
        ├── train_phase1.py        # Phase 1 training entry point
        ├── __init__.py
        │
        ├── models/
        │   ├── __init__.py
        │   └── s2st_model.py      # Core S2STModel class
        │
        ├── head/
        │   ├── __init__.py
        │   └── audio_header.py    # Audio prediction head (768→8194)
        │
        ├── projector/
        │   ├── __init__.py
        │   ├── semantic_projector.py   # Semantic feature projector
        │   └── speaker_projector.py    # Speaker feature projector
        │
        ├── Dataset/
        │   ├── __init__.py
        │   └── Phase1.py          # Phase 1 dataset & collate function
        │
        └── utils/
            ├── __init__.py
            ├── audio_utils.py     # Audio tokenization & augmentation
            ├── loss.py            # Loss utilities (unused in Phase 1)
            ├── tracker.py         # Checkpoint state tracker
            └── metric/
                ├── __init__.py
                └── Phase1.py      # Phase 1 evaluation metrics
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended: 24GB+ VRAM)
- 48 CPU cores (optimized `num_workers=8`)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd S2ST

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Pre-trained Models

Download and place the following models in `models/checkpoints/`:

1. **EnViT5-Translation**: Vietnamese-English translation model
   ```bash
   # Download from Hugging Face
   huggingface-cli download VietAI/envit5-translation --local-dir models/checkpoints/envit5-translation
   ```

2. **Spark-TTS-0.5B**: BiCodec audio tokenizer
   ```bash
   # Download from Hugging Face
   huggingface-cli download SparkAudio/Spark-TTS-0.5B --local-dir models/checkpoints/Spark-TTS-0.5B
   ```

---

## Dataset Preparation

### CSV Format

Training and validation data should be provided as CSV files with the following columns:

| Column | Description |
|--------|-------------|
| `VieAudio` | Path to Vietnamese audio file (.wav, 16kHz) |
| `EngAudio` | Path to English audio file (.wav, 16kHz) |
| `VieText` | Path to Vietnamese text file (.txt) |
| `EngText` | Path to English text file (.txt) |

Example CSV:
```csv
VieAudio,EngAudio,VieText,EngText
/data/PhoST/train/001/001_0.wav,/data/PhoST/train/001/001_0_en.wav,/data/PhoST/train/001/001_0.vi,/data/PhoST/train/001/001_0.en
```

### PhoST Dataset

The project includes a preprocessing script for the [PhoST](https://huggingface.co/datasets/vinai/PhoST) dataset:

```bash
# Download PhoST data
python data/PhoST.py
```

> **Note**: Update the paths in `data/PhoST.py` to match your local directory structure before running.

### Audio Augmentation (Optional)

For training with audio augmentation, download MUSAN and RIR noise datasets:

```python
from utils.audio_utils import dowload_augmentaion

# This downloads MUSAN noise and RIR impulse responses
rir_recordings, musan_cuts = dowload_augmentaion("/path/to/augmentation/data")
```

Then pass them to the dataset:
```python
train_dataset = Phase1_S2ST(
    files_csv=["data/train.csv"],
    tokenizer=tokenizer,
    split='train',
    rir_recordings=rir_recordings,
    musan_cuts=musan_cuts,
)
```

---

## Training

### Configuration

Edit `models/s2st/train_config.yaml` to configure training:

```yaml
training:
  epochs: 10
  batch_size: 4
  gradient_accumulation_steps: 16  # Effective batch = 4 × 16 = 64
  max_grad_norm: 1.0
  num_workers: 8                   # Optimized for 48-core machine
  seed: 42

optimizer:
  lr: 3e-4
  weight_decay: 0.01

scheduler:
  warmup_ratio: 0.1
  min_lr: 1e-6

loss:
  label_smoothing: 0.1
```

### Running Training

```bash
cd models/s2st
python train_phase1.py
```

The training script uses [Accelerate](https://huggingface.co/docs/accelerate) for distributed training support:

```bash
# Multi-GPU training
accelerate launch train_phase1.py

# Custom accelerate config
accelerate config  # Set up distributed config first
accelerate launch train_phase1.py
```

### Training Pipeline

1. **Data Loading**: Each sample randomly receives one of 8 task types (ASR/S2TT/MT/TTS × EN/VI)
2. **Audio Tokenization**: Spark-TTS BiCodec extracts speaker embeddings and semantic tokens (frozen)
3. **Projection**: Projectors map 1024-dim audio features to 768-dim LLM space
4. **Encoder-Decoder**: EnViT5 processes task-specific input sequences
5. **Loss**: CrossEntropyLoss with label smoothing, averaged across text and audio losses
6. **Metrics**: WER (ASR), BLEU (MT/S2TT), UTMOS + Speaker Similarity + TTS-WER (TTS)

### Logging

Training logs are sent to [Weights & Biases](https://wandb.ai/):

```bash
# Login to WandB before training
wandb login
```

Tracked metrics:
- `train/loss`, `train/lr`, `train/step`
- `val/loss`, `val/asr_wer`, `val/mt_bleu`
- `val/utmos`, `val/spk_similarity`, `val/tts_wer`

---

## Checkpoint Management

### Automatic Saving

After each epoch, the training script saves:
- `models/checkpoints/phase1/latest_checkpoint/` — Latest model state (for resuming)
- `models/checkpoints/phase1/best_checkpoint/` — Best model by TTS WER

### Resuming Training

Training automatically resumes from `latest_checkpoint` if it exists:

```bash
# Just re-run — it will detect and load the checkpoint
cd models/s2st
python train_phase1.py
```

### Checkpoint Contents

Each checkpoint (saved via Accelerate) includes:
- Model weights (all trainable parameters)
- Optimizer state
- LR scheduler state
- RNG states
- `CheckpointTracker`: `start_epoch`, `best_tts_wer`

---

## Inference

> **Note**: Inference pipeline is planned for Phase 2/3. The current Phase 1 implementation focuses on multi-task pre-training.

Planned inference workflow:
1. Load trained model from best checkpoint
2. For S2ST: Audio → SparkTTS tokenization → Encoder → Decoder → AudioHeader → Detokenize → Waveform
3. For TTS: Text + Speaker reference → Encoder → Decoder → AudioHeader → Detokenize → Waveform

---

## Configuration Reference

### Model Architecture (`config.yaml`)

```yaml
semantic_projector:
  input_dim: 1024    # BiCodec semantic feature dim
  hidden_dim: 512    # Bottleneck dim
  output_dim: 768    # EnViT5 hidden dim
  layers: 3
  dropout: 0.1
  normalize: true
  activation: ReLU

speaker_projector:
  input_dim: 1024    # BiCodec speaker feature dim
  hidden_dim: 512
  output_dim: 768
  layers: 3
  dropout: 0.1
  normalize: true
  activation: ReLU

audio_header:
  input_dim: 768     # EnViT5 hidden dim
  output_dim: 8194   # 8192 codebook + BOS + EOS
```

---

## Current Status

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1** | ✅ Implemented | Multi-task pre-training (ASR, S2TT, MT, TTS) |
| **Phase 2** | 🔲 Planned | End-to-end S2ST fine-tuning |
| **Phase 3** | 🔲 Planned | Expressive and speaker-preserving refinement |

---

## Assumptions & Limitations

1. **Language pair**: Currently limited to Vietnamese ↔ English
2. **Audio format**: All audio must be 16kHz mono WAV
3. **Text format**: Each text file contains a single sentence/utterance
4. **Frozen audio tokenizer**: Spark-TTS BiCodec is never fine-tuned
5. **Speaker reference**: For TTS, speaker identity comes from the target audio's global tokens (teacher forcing during training)
6. **Task sampling**: Tasks are uniformly sampled at random; no curriculum or weighted sampling

## Future Work

- [ ] Implement Phase 2: End-to-end S2ST with cascaded ASR→MT→TTS
- [ ] Implement Phase 3: Expressive prosody transfer
- [ ] Add autoregressive inference loop for TTS decoding
- [ ] Support additional language pairs
- [ ] Add Gradio-based demo UI
- [ ] Implement weighted task sampling for curriculum learning
- [ ] Add mixed-precision training (fp16/bf16) configuration

---

## References

- **UNISS Paper**: *UNISS: Unified Expressive Speech-to-Speech Translation with Your Voice*
- **EnViT5**: [VietAI/envit5-translation](https://huggingface.co/VietAI/envit5-translation)
- **Spark-TTS**: [SparkAudio/Spark-TTS-0.5B](https://huggingface.co/SparkAudio/Spark-TTS-0.5B)
- **PhoST Dataset**: [vinai/PhoST](https://huggingface.co/datasets/vinai/PhoST)
