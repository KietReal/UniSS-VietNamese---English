from jiwer import wer
import jiwer
from speechbrain.inference.speaker import SpeakerRecognition
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
from transformers import pipeline
from torch.nn.utils.rnn import pad_sequence

AUDIO_EOS_ID = 8192

class Phase2Metrics:
    """Evaluation metrics for Phase 2"""

    def __init__(self, device: str, text_tokenizer, audio_tokenizer):
        self.device = device
        self.text_tokenizer = text_tokenizer
        self.audio_tokenizer = audio_tokenizer

        if hasattr(self.audio_tokenizer, "model"):
            self.audio_tokenizer.model.eval()

        self.wer_transform = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(),
        ])

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.eval_dtype = dtype

        # TTS quality: UTMOS
        self.utmos_model = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
        ).to(device=device, dtype=dtype)
        self.utmos_model.eval()

        # TTS speaker similarity: ECAPA-TDNN
        self.spk_verifier = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )

        # TTS (English): Whisper ASR 
        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-base",
            torch_dtype=dtype,
        ).to(device)
        self.whisper_model.config.forced_decoder_ids = None
        self.whisper_model.eval()

        # TTS (Vietnamese): PhoWhisper ASR 
        self.pho_whisper = pipeline(
            "automatic-speech-recognition",
            model="vinai/PhoWhisper-base",
            torch_dtype=dtype,
            device=device,
        )

        # Accumulator lists
        self._vi_utmos_scores   = []
        self._vi_spk_sim_scores = []
        self._vi_spk_wer_scores = []

        self._en_utmos_scores   = []
        self._en_spk_sim_scores = []
        self._en_spk_wer_scores = []

    def _tokens_to_waveforms(
        self,
        semantic_tokens: list[torch.Tensor],
        global_tokens: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        1 token in Semantic tokens = 320 samples in audio
        """
        valid_token_lens = [t.size(0) for t in semantic_tokens]
        sem_padded = pad_sequence(semantic_tokens, batch_first=True, padding_value=0)
        
        wav_batch = self.audio_tokenizer.model.detokenize(sem_padded, global_tokens)

        clean_waveforms = []
        for i, T_i in enumerate(valid_token_lens):
            valid_samples = T_i * 320
            clean_wav = wav_batch[i, 0, :valid_samples]
            clean_waveforms.append(clean_wav)

        return clean_waveforms

    def _whisper_transcribe(self, waveforms: list[torch.Tensor]) -> list[str]:
        """Transcribe English waveforms via Whisper."""
        wav_list = [w.cpu().numpy() for w in waveforms]
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        inputs = self.whisper_processor(
            wav_list,
            sampling_rate=16000,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
        ).input_features.to(device=self.device, dtype=dtype)
        predicted_ids = self.whisper_model.generate(inputs, return_timestamps=True)
        return self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)

    def _phowhisper_transcribe(self, waveforms: list[torch.Tensor]) -> list[str]:
        """Transcribe Vietnamese waveforms via PhoWhisper."""
        wav_list = [w.cpu().numpy() for w in waveforms]
        results = self.pho_whisper(
            wav_list,
            batch_size=len(wav_list),
            chunk_length_s=30,
            return_timestamps=True,
        )
        return [r["text"] for r in results]

    @torch.no_grad()
    def add_batch(
        self,
        text_target: torch.Tensor,       # (B, L)
        global_tokens: torch.Tensor,     # (B, 1, 32)
        audio_prediction: torch.Tensor,  # (B, L, audio_vocab)
        audio_target: torch.Tensor,      # (B, L) 
    ):
        """
        Args:
            text_target: (B, L)
            global_tokens: (B, 1, 32)
            audio_prediction: (B, L, audio_vocab)
            audio_target: (B, L)
        """
        batch = global_tokens.shape[0]

        # Separate English vs Vietnamese
        en_pred_toks, en_tgt_toks, en_ref_texts, en_global = [], [], [], []
        vi_pred_toks, vi_tgt_toks, vi_ref_texts, vi_global = [], [], [], []

        
        for batch_idx in range(batch):

            valid_mask = (audio_target[batch_idx] != -100) & (audio_target[batch_idx] != AUDIO_EOS_ID)
            valid_len = valid_mask.sum().item()
            tgt_toks = audio_target[batch_idx, :valid_len]

            pred_toks = audio_prediction[batch_idx].argmax(dim=-1)
            pred_toks = pred_toks[:valid_len]
            pred_toks = pred_toks[(pred_toks >= 0) & (pred_toks < 8192)]

            valid_input = text_target[batch_idx][text_target[batch_idx] >= 0] #remove -100
            ref_text = self.text_tokenizer.decode(
                valid_input, skip_special_tokens=True
            )

            if batch_idx%2 == 0:
                en_pred_toks.append(pred_toks)
                en_tgt_toks.append(tgt_toks)
                en_ref_texts.append(ref_text)
                en_global.append(global_tokens[batch_idx])  # (1, 32)
            
            else:
                vi_pred_toks.append(pred_toks)
                vi_tgt_toks.append(tgt_toks)
                vi_ref_texts.append(ref_text)
                vi_global.append(global_tokens[batch_idx])  # (1, 32)

        # English evaluation 
        if en_pred_toks:
            en_global_batch = torch.stack(en_global, dim=0)  # (N, 1, 32)
            en_pred_wavs = self._tokens_to_waveforms(en_pred_toks, en_global_batch)
            en_tgt_wavs = self._tokens_to_waveforms(en_tgt_toks, en_global_batch)

            en_pred_tensor = pad_sequence(en_pred_wavs, batch_first=True, padding_value=0.0)
            en_tgt_tensor = pad_sequence(en_tgt_wavs, batch_first=True, padding_value=0.0)

            sim_scores, _ = self.spk_verifier.verify_batch(en_pred_tensor.float(), en_tgt_tensor.float())
            utmos_scores = self.utmos_model(en_pred_tensor.to(self.eval_dtype), 16000)
            transcriptions = self._whisper_transcribe(en_pred_wavs)

            self._en_spk_sim_scores.extend(sim_scores.view(-1).tolist())
            self._en_utmos_scores.extend(utmos_scores.view(-1).tolist())
            self._en_spk_wer_scores.append(
                wer(
                    reference=en_ref_texts,
                    hypothesis=transcriptions,
                    reference_transform=self.wer_transform,
                    hypothesis_transform=self.wer_transform,
                )
            )

        # Vietnamese evaluation
        if vi_pred_toks:
            vi_global_batch = torch.stack(vi_global, dim=0)  # (N, 1, 32)
            vi_pred_wavs = self._tokens_to_waveforms(vi_pred_toks, vi_global_batch)
            vi_tgt_wavs = self._tokens_to_waveforms(vi_tgt_toks, vi_global_batch)

            vi_pred_tensor = pad_sequence(vi_pred_wavs, batch_first=True, padding_value=0.0)
            vi_tgt_tensor = pad_sequence(vi_tgt_wavs, batch_first=True, padding_value=0.0)

            sim_scores, _ = self.spk_verifier.verify_batch(vi_pred_tensor.float(), vi_tgt_tensor.float())
            utmos_scores = self.utmos_model(vi_pred_tensor.to(self.eval_dtype), 16000)
            transcriptions = self._phowhisper_transcribe(vi_pred_wavs)

            self._vi_spk_sim_scores.extend(sim_scores.view(-1).tolist())
            self._vi_utmos_scores.extend(utmos_scores.view(-1).tolist())
            self._vi_spk_wer_scores.append(
                wer(
                    reference=vi_ref_texts,
                    hypothesis=transcriptions,
                    reference_transform=self.wer_transform,
                    hypothesis_transform=self.wer_transform,
                )
            )

    def compute(self) -> dict[str, float]:
        """Compute epoch-level metrics and reset accumulators."""
        results = {
            "vi_utmos": (
                sum(self._vi_utmos_scores) / len(self._vi_utmos_scores)
            ),
            "vi_spk_similarity": (
                sum(self._vi_spk_sim_scores) / len(self._vi_spk_sim_scores)
            ),
            "vi_spk_wer": (
                sum(self._vi_spk_wer_scores) / len(self._vi_spk_wer_scores)
            ),

            "en_utmos": (
                sum(self._en_utmos_scores) / len(self._en_utmos_scores)
            ),
            "en_spk_similarity": (
                sum(self._en_spk_sim_scores) / len(self._en_spk_sim_scores)
            ),
            "en_spk_wer": (
                sum(self._en_spk_wer_scores) / len(self._en_spk_wer_scores)
            ),
        }

        # Reset accumulators
        self._vi_utmos_scores.clear()
        self._vi_spk_sim_scores.clear()
        self._vi_spk_wer_scores.clear()

        self._en_utmos_scores.clear()
        self._en_spk_sim_scores.clear()
        self._en_spk_wer_scores.clear()

        return results