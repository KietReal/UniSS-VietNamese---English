import torch
from lhotse import Recording, CutSet, RecordingSet
from lhotse.dataset import PerturbSpeed, PerturbVolume, CutMix, ReverbWithImpulseResponse
from lhotse.recipes import (
    download_musan,
    download_rir_noise,
    prepare_musan,
    prepare_rir_noise,
)

def dowload_augmentaion(root_dir: str):
    musan_cuts = CutSet()
    musan_raw_dir = download_musan(root_dir)
    musan = prepare_musan(musan_raw_dir, output_dir=root_dir)
    musan_cuts = CutSet.from_manifests(
        recordings=(
            musan["noise"]["recordings"]
        )
    ).to_eager()

    rir_recordings = RecordingSet()
    rir_raw_dir = download_rir_noise(root_dir)
    rirs = prepare_rir_noise(rir_raw_dir, output_dir=root_dir, parts=["sim_rir"])
    rir_recordings = rirs["sim_rir"]["recordings"]

    return rir_recordings, musan_cuts


def AugmentedAudio(rir_recordings, musan_cuts, 
    wav_path: str, transforms: list|None = None
):
    # Dummy implementation matching your logic
    recording = Recording.from_file(wav_path)
    cut = recording.to_cut()
    cuts = CutSet.from_cuts([cut])

    augmented_cuts = cuts

    if transforms is None:
        transforms = [
            PerturbVolume(scale_low=0.7, scale_high=1.5, p=0.5),
            ReverbWithImpulseResponse(rir_recordings, p=0.5),
            CutMix(musan_cuts, snr=[10, 20], p=0.5, seed='trng'),
        ]
    
    for transform in transforms:
        augmented_cuts = transform(augmented_cuts)

    augmented_cut = augmented_cuts[0]
    augmented_audio = augmented_cut.load_audio() # NumPy array [Channels, Samples]
    
    # Squeeze to ensure it's a 1D tensor (Samples,) for sequence padding
    audio_tensor = torch.from_numpy(augmented_audio).float().squeeze() 
    return audio_tensor

def get_ref_clip(
    wav: torch.Tensor, 
    sample_rate: int = 16000, 
    ref_segment_duration: float = 6.0, 
    latent_hop_length: int = 320
) -> torch.Tensor:
    """Get reference audio clip for speaker embedding."""
    ref_segment_length = (
        int(sample_rate * ref_segment_duration)
        // latent_hop_length
        * latent_hop_length
    )
    
    wav_length = wav.shape[-1]

    if ref_segment_length > wav_length:
        repeat_factor = ref_segment_length // wav_length + 1
        wav = torch.tile(wav, (repeat_factor,))

    return wav[..., :ref_segment_length]

@torch.no_grad() 
def audio2token(tokenizer, wavs_2d: torch.Tensor, ref_wavs_2d: torch.Tensor):
    """
    Args:
        tokenizer: Loaded BiCodecTokenizer instance.
        wavs_2d (torch.Tensor): Padded 2D audio tensor of shape (batch_size, max_audio_len) at 16kHz.
        ref_wavs_2d (torch.Tensor): 2D reference audio tensor of shape (batch_size, 96000) at 16kHz.
        
    Returns:
        global_tokens (torch.Tensor): Shape (batch_size, 1, 32)
        semantic_tokens (torch.Tensor): Shape (batch_size, seq_len)
    """
    device = tokenizer.device
    wavs_2d = wavs_2d.to(device)
    ref_wavs_2d = ref_wavs_2d.to(device)
    
    feat_outputs = tokenizer.feature_extractor(wavs_2d)
    
    feats_mix = (
        feat_outputs.hidden_states[11] + 
        feat_outputs.hidden_states[14] + 
        feat_outputs.hidden_states[16]
    ) / 3

    batch = {
        "feat": feats_mix,         
        "ref_wav": ref_wavs_2d     
    }
    
    semantic_tokens, global_tokens = tokenizer.model.tokenize(batch)
    return global_tokens, semantic_tokens