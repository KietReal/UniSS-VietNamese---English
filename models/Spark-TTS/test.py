from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.modules.vq.factorized_vector_quantize import FactorizedVectorQuantize
import torch
import soundfile as sf

device = torch.device("mps" if torch.mps.is_available() else "cpu")
tokenizer = BiCodecTokenizer(
    model_dir="../checkpoints/Spark-TTS-0.5B",
    device=device,
)

# Get the trained quantizer from the loaded BiCodec tokenizer model
semantic_quantizer = tokenizer.model.quantizer
speaker_encoder = tokenizer.model.speaker_encoder

wav_path = "./example/prompt_audio.wav"

# 1. Load two audio waveforms as PyTorch Tensors
wav1, ref_wav1 = tokenizer.process_audio(wav_path)
wav2, ref_wav2 = tokenizer.process_audio(wav_path)

wav1_tensor = torch.from_numpy(wav1).float()
wav2_tensor = torch.from_numpy(wav2).float()

# 2. Create 2D PyTorch Tensors for the batch
# wavs_2d: shape [2, max_audio_len]
wavs_2d = torch.nn.utils.rnn.pad_sequence([wav1_tensor, wav2_tensor], batch_first=True, padding_value=0.0).to(device)
# ref_wavs_2d: shape [2, 96000]
ref_wavs_2d = torch.cat([ref_wav1, ref_wav2], dim=0).to(device)

# 3. Pure PyTorch Tensor tokenization function (ideal for custom PyTorch models)
def tokenize_tensors_direct(tokenizer, wavs_2d, ref_wavs_2d):
    # Pass 2D tensor directly to Wav2Vec2Model (bypasses processor wrapper)
    feat_outputs = tokenizer.feature_extractor(wavs_2d)
    feats_mix = (
        feat_outputs.hidden_states[11] + feat_outputs.hidden_states[14] + feat_outputs.hidden_states[16]
    ) / 3

    batch = {
        "feat": feats_mix,
        "ref_wav": ref_wavs_2d
    }
    
    semantic_tokens, global_tokens = tokenizer.model.tokenize(batch)
    return global_tokens, semantic_tokens

# 4. Execute direct 2D Tensor tokenization
global_tokens, semantic_tokens = tokenize_tensors_direct(tokenizer, wavs_2d, ref_wavs_2d)

print("Direct 2D Tensor Global tokens shape:", global_tokens.shape)     # torch.Size([2, 1, 32])
print("Direct 2D Tensor Semantic tokens shape:", semantic_tokens.shape)   # torch.Size([2, max_seq_len])