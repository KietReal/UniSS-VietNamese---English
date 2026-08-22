from typing import Any
import os
import sys

# Add SparkTTS directory to sys.path before importing sparktts
current_file_dir = os.path.dirname(os.path.abspath(__file__))
spark_tts_dir = os.path.abspath(os.path.join(current_file_dir, "../../SparkTTS"))
if spark_tts_dir not in sys.path:
    sys.path.insert(0, spark_tts_dir)

import yaml
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sparktts.models.audio_tokenizer import BiCodecTokenizer

from projector.semantic_projector import SemanticProjector
from projector.speaker_projector import SpeakerProjector
from head.audio_header import AudioHeader
from utils.audio_utils import get_ref_clip, audio2token

class S2STModel(nn.Module):
    def __init__(self, envit5_path=None, sparkTTS_path=None):
        super().__init__()
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.abspath(os.path.join(current_dir, '../config.yaml'))
        if not os.path.exists(config_path):
            config_path = os.path.abspath(os.path.join(current_dir, '../../config.yaml'))
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        ### 1. Initalize sub-modules
        self.semantic_projector = SemanticProjector()
        self.speaker_projector = SpeakerProjector()
        self.audio_header = AudioHeader()
        
        ### 2. Initalize LLM
        if envit5_path is None:
            envit5_path = os.path.abspath(os.path.join(current_dir, '../checkpoints/envit5-translation'))
        if sparkTTS_path is None:
            sparkTTS_path = os.path.abspath(os.path.join(current_dir, '../checkpoints/Spark-TTS-0.5B'))
        
        try:
            self.envit5 = AutoModelForSeq2SeqLM.from_pretrained(
                envit5_path,
                attn_implementation="sdpa" if torch.cuda.is_available() else None
            )
        except Exception:
            self.envit5 = AutoModelForSeq2SeqLM.from_pretrained(envit5_path)
        self.envit5_tokenizer = AutoTokenizer.from_pretrained(envit5_path)

        # Add special tokens to tokenizer and resize LLM embedding table
        special_tokens = ['<S2TT>', '<MT>', '<TTS>', '<ASR>', '<ENG>', '<VIE>', '<BOS>']
        num_added = self.envit5_tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        if num_added > 0:
            self.envit5.resize_token_embeddings(len(self.envit5_tokenizer))

        # Enable gradient checkpointing to save VRAM
        if hasattr(self.envit5, "gradient_checkpointing_enable"):
            self.envit5.gradient_checkpointing_enable()

        self.embedding = self.envit5.shared
        self.bos = self.envit5_tokenizer.convert_tokens_to_ids('<BOS>')
        self.eos = self.envit5_tokenizer.eos_token_id

        AUDIO_CODEBOOK_SIZE = 8192

        # New Special Audio Token IDs
        self.AUDIO_EOS_ID = 8192  # Index 8192
        self.AUDIO_BOS_ID = 8193  # Index 8193

        # Learnable 768-dim embedding vectors for special audio tokens
        # Shape (1, 768) — single-step embeddings to cat with (seq, 768) decoder tokens
        self.audio_bos_embed = nn.Parameter(torch.randn(1, 768) * 0.02)
        self.audio_eos_embed = nn.Parameter(torch.randn(1, 768) * 0.02)

        ### 3. Initalize Audio Tokenizer
        self.sparkTTS = BiCodecTokenizer(model_dir=sparkTTS_path, device=torch.device("cpu"))
        self.semantic_quantizer = self.sparkTTS.model.quantizer
        self.speaker_encoder = self.sparkTTS.model.speaker_encoder
        
        # Freeze SparkTTS nn.Module components
        self.sparkTTS.model.eval()
        for param in self.sparkTTS.model.parameters():
            param.requires_grad = False
        self.sparkTTS.feature_extractor.eval()
        for param in self.sparkTTS.feature_extractor.parameters():
            param.requires_grad = False

    @property
    def device(self):
        """Derive device from model parameters — always consistent with Accelerate."""
        return next(self.parameters()).device

    def train(self, mode: bool = True):
        """Ensure SparkTTS submodules stay in eval mode during training."""
        super().train(mode)
        if hasattr(self, 'sparkTTS') and hasattr(self.sparkTTS, 'model'):
            self.sparkTTS.model.eval()
            self.sparkTTS.feature_extractor.eval()
        return self

    def _move_sparkTTS_to_device(self):
        """Move non-nn.Module SparkTTS components to the correct device.
        
        BiCodecTokenizer is not an nn.Module, so Accelerate won't move it automatically.
        This must be called after accelerator.prepare() or manual .to(device).
        """
        device = self.device
        self.sparkTTS.model = self.sparkTTS.model.to(device)
        self.sparkTTS.feature_extractor = self.sparkTTS.feature_extractor.to(device)
        self.sparkTTS.device = device
        self.sparkTTS.model.eval()
        self.sparkTTS.feature_extractor.eval()
        self.semantic_quantizer = self.sparkTTS.model.quantizer
        self.speaker_encoder = self.sparkTTS.model.speaker_encoder

    def llm_forward(self, inputs_encoder: torch.Tensor, inputs_decoder: torch.Tensor, 
        att_mask: torch.Tensor, decoder_att_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        encoder_outputs = self.envit5.encoder(
            inputs_embeds=inputs_encoder,
            attention_mask=att_mask,
            return_dict=True,
        )

        decoder_outputs = self.envit5.decoder(
            inputs_embeds=inputs_decoder,
            attention_mask=decoder_att_mask,          
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=att_mask, 
            return_dict=True,
        )
        pre = decoder_outputs.last_hidden_state
        
        text_prediction = self.envit5.lm_head(pre)
        audio_prediction = self.audio_header(pre)

        return text_prediction, audio_prediction

    def get_audio_tokens(self, audio: torch.Tensor, tts_mask: torch.Tensor):
        audio = audio[tts_mask]
        ref_audio = get_ref_clip(wav=audio)
        global_tokens, semantic_tokens = audio2token(
            tokenizer=self.sparkTTS, wavs_2d=audio, ref_wavs_2d=ref_audio
        )
        return global_tokens, semantic_tokens         

    def get_global_token_task_3(self):
        return self.global_token_task_3

    def phase_1_forward(self, T_encode: torch.Tensor, T_decode: torch.Tensor, 
                        A_encode: torch.Tensor, A_decode: torch.Tensor, sub_tasks: torch.Tensor):
        
        batch_size = sub_tasks.size(0)
        device = self.device
        pad_id = 0 
        self.global_token_task_3 = None

        # 1. Look up Text Embeddings for the whole batch
        embed_T_encode = self.embedding(T_encode) 
        embed_T_decode = self.embedding(T_decode) 

        # 2. Extract Audio Features on GPU 
        enc_mask = (sub_tasks == 1) | (sub_tasks == 2)
        if enc_mask.any():
            with torch.no_grad():
                enc_global_toks, enc_sem_toks = self.get_audio_tokens(audio=A_encode, tts_mask=enc_mask)
                S_speaker_raw = self.speaker_encoder.detokenize(enc_global_toks).unsqueeze(1)
                S_Ling_raw = self.semantic_quantizer.detokenize(enc_sem_toks).transpose(1, 2)
            
            S_speaker_enc = self.speaker_projector(S_speaker_raw) 
            S_Ling_enc = self.semantic_projector(S_Ling_raw)

        dec_mask = (sub_tasks == 3)
        if dec_mask.any():
            with torch.no_grad():
                dec_global_toks, dec_sem_toks = self.get_audio_tokens(audio=A_decode, tts_mask=dec_mask)
                S_speaker_dec_raw = self.speaker_encoder.detokenize(dec_global_toks).unsqueeze(1)
                S_Sem_dec_raw = self.semantic_quantizer.detokenize(dec_sem_toks).transpose(1, 2)

                self.global_token_task_3 = dec_global_toks
                
            S_speaker_dec = self.speaker_projector(S_speaker_dec_raw)
            S_Sem_dec = self.semantic_projector(S_Sem_dec_raw)

        # 3. Dynamically build Encoder and Decoder sequences item by item
        inputs_encode_list, att_mask_list = [], []
        inputs_decode_list, dec_att_mask_list = [], []
        audio_labels_list = []

        enc_audio_idx, dec_audio_idx = 0, 0

        for i in range(batch_size):
            task = sub_tasks[i].item()
            valid_len_enc = (T_encode[i] != pad_id).sum().item()
            valid_len_dec = (T_decode[i] != pad_id).sum().item()

            # BUILD ENCODER SEQUENCE 
            if task in [1, 2]: # ASR (1), S2TT (2)
                prompt_embeds = embed_T_encode[i, :2] 
                spk_embeds = S_speaker_enc[enc_audio_idx]
                ling_embeds = S_Ling_enc[enc_audio_idx]
                enc_seq = torch.cat([prompt_embeds, spk_embeds, ling_embeds], dim=0)
                enc_audio_idx += 1
                
            elif task == 3: # TTS (3)
                prompt_embeds = embed_T_encode[i, :2]
                spk_embeds = S_speaker_dec[dec_audio_idx] 
                text_embeds = embed_T_encode[i, 2:valid_len_enc]
                enc_seq = torch.cat([prompt_embeds, spk_embeds, text_embeds], dim=0)
                
            else: # MT (4)
                enc_seq = embed_T_encode[i, :valid_len_enc]
                
            inputs_encode_list.append(enc_seq)
            att_mask_list.append(torch.ones(enc_seq.size(0), dtype=torch.bool, device=device))

            # BUILD DECODER SEQUENCE 
            if task in [1, 2, 4]: # ASR, S2TT, MT
                dec_seq = embed_T_decode[i, :valid_len_dec]
                inputs_decode_list.append(dec_seq)
                dec_att_mask_list.append(torch.ones(dec_seq.size(0), dtype=torch.bool, device=device))
                
            elif task == 3: # TTS
                sem_embeds = S_Sem_dec[dec_audio_idx]
                # audio_bos_embed is (1, 768), sem_embeds is (seq, 768) — both 2D, concat on dim=0
                dec_seq = torch.cat([self.audio_bos_embed, sem_embeds], dim=0)
                inputs_decode_list.append(dec_seq)
                dec_att_mask_list.append(torch.ones(dec_seq.size(0), dtype=torch.bool, device=device))
                
                # Build audio label: semantic tokens for this item + EOS token
                eos_token = torch.tensor([self.AUDIO_EOS_ID], dtype=dec_sem_toks.dtype, device=device)
                item_audio_labels = torch.cat([dec_sem_toks[dec_audio_idx], eos_token], dim=0)
                audio_labels_list.append(item_audio_labels)
                dec_audio_idx += 1

        # 4. Pad the sequences to form batch tensors
        inputs_encoder = pad_sequence(inputs_encode_list, batch_first=True, padding_value=0.0)
        inputs_decoder = pad_sequence(inputs_decode_list, batch_first=True, padding_value=0.0)
        
        att_mask = pad_sequence(att_mask_list, batch_first=True, padding_value=False)
        decoder_att_mask = pad_sequence(dec_att_mask_list, batch_first=True, padding_value=False)

        if len(audio_labels_list) > 0:
            audio_labels = pad_sequence(audio_labels_list, batch_first=True, padding_value=-100)
        else:
            audio_labels = None

        # 5. Forward through LLM
        text_prediction, audio_prediction = self.llm_forward(inputs_encoder, inputs_decoder, att_mask, decoder_att_mask)

        return text_prediction, audio_prediction, audio_labels

    def phase_2_forward(self):
        pass

    def phase_3_forward(self):
        pass

    def forward(self, T_encode: torch.Tensor, T_decode: torch.Tensor, 
                A_encode: torch.Tensor, A_decode: torch.Tensor, phase: int, sub_tasks: torch.Tensor):
        """
            Args:

            T_encode: idx of encoder token. shape: (batch, seq)
            T_decode: idx of decoder token. shape: (batch, seq)


            A_encode: audio encode. shape: (batch, wav_len)
            A_decode: audio decode. shape: (batch, wav_len)

            phase: int (1-2-3)
            sub_task: only for phas 1, 3.

        """

        if phase == 1:
            text_prediction, audio_prediction, audio_labels = self.phase_1_forward(
                T_encode, T_decode, A_encode, A_decode, sub_tasks
            )
            return text_prediction, audio_prediction, audio_labels
        elif phase == 2:
            pass
        elif phase == 3:
            pass
