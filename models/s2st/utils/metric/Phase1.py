from jiwer import wer
import jiwer
import sacrebleu
from speechbrain.inference.speaker import SpeakerRecognition
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
from transformers import pipeline
from torch.nn.utils.rnn import pad_sequence

AUDIO_EOS_ID = 8192


class Phase1Metrics:
    """Evaluation metrics for Phase 1 multi-task S2ST training."""

    def __init__(self, device: str, text_tokenizer, audio_tokenizer):
        self.device = device
        self.text_tokenizer = text_tokenizer
        self.audio_tokenizer = audio_tokenizer

        if hasattr(self.audio_tokenizer, "model"):
            self.audio_tokenizer.model.eval()

        # WER text transformation pipeline (lowercasing, strip, remove punctuation & extra spaces)
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
        self._asr_wer_scores = []
        self._mt_bleu_scores = []
        self._utmos_scores = []
        self._spk_sim_scores = []
        self._tts_wer_scores = []

    def _decode_batch(self, token_ids: torch.Tensor) -> list[str]:
        """Decode a batch of token ID tensors to a list of strings."""
        sentences = []
        for row in token_ids:
            valid_ids = row[row >= 0]
            text = self.text_tokenizer.decode(valid_ids, skip_special_tokens=True)
            sentences.append(text)
        return sentences

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
            wav_list, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device=self.device, dtype=dtype)
        predicted_ids = self.whisper_model.generate(inputs, return_timestamps=True)
        return self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)

    def _phowhisper_transcribe(self, waveforms: list[torch.Tensor]) -> list[str]:
        """Transcribe Vietnamese waveforms via PhoWhisper."""
        wav_list = [w.cpu().numpy() for w in waveforms]
        results = self.pho_whisper(
            wav_list,
            batch_size=len(wav_list),
            return_timestamps=True,
        )
        return [r["text"] for r in results]

    @torch.no_grad()
    def add_batch(
        self,
        text_prediction: torch.Tensor,   # (B, L, text_vocab)
        text_target: torch.Tensor,       # (B, L)
        text_input: torch.Tensor,        # (B, L)
        global_tokens: torch.Tensor | None,  # (N_tts, 1, 32)
        audio_prediction: torch.Tensor,  # (B, L, audio_vocab)
        audio_target: torch.Tensor,      # (N_tts, L) — only TTS items, NOT full batch
        sub_tasks: torch.Tensor,         # (B,)
        task_names: list[str],
    ):
        # ── Task 1: ASR → WER ──
        asr_mask = sub_tasks == 1
        if asr_mask.any():
            pred_ids = text_prediction[asr_mask].argmax(dim=-1)
            tgt_ids = text_target[asr_mask]
            pred_texts = self._decode_batch(pred_ids)
            ref_texts = self._decode_batch(tgt_ids)
            self._asr_wer_scores.append(
                wer(
                    reference=ref_texts,
                    hypothesis=pred_texts,
                    reference_transform=self.wer_transform,
                    hypothesis_transform=self.wer_transform,
                )
            )

        # ── Task 2 & 4: S2TT / MT → BLEU ──
        mt_mask = (sub_tasks == 2) | (sub_tasks == 4)
        if mt_mask.any():
            pred_ids = text_prediction[mt_mask].argmax(dim=-1)
            tgt_ids = text_target[mt_mask]
            pred_texts = self._decode_batch(pred_ids)
            ref_texts = self._decode_batch(tgt_ids)
            bleu = sacrebleu.corpus_bleu(
                hypotheses=pred_texts,
                references=[ref_texts],
                lowercase=True,
            )
            self._mt_bleu_scores.append(bleu.score)

        # ── Task 3: TTS → UTMOS + Speaker Similarity + WER ──
        if global_tokens is None:
            return

        tts_indices = torch.where(sub_tasks == 3)[0]
        if tts_indices.numel() == 0:
            return

        # audio_target is (N_tts, L_audio) — indexed by TTS-local index (0, 1, 2, ...)
        # audio_prediction is (B, L, audio_vocab) — indexed by batch index
        # global_tokens is (N_tts, 1, 32) — indexed by TTS-local index

        # Separate English vs Vietnamese TTS items
        en_pred_toks, en_tgt_toks, en_ref_texts, en_global = [], [], [], []
        vi_pred_toks, vi_tgt_toks, vi_ref_texts, vi_global = [], [], [], []

        for local_idx, batch_idx in enumerate(tts_indices):
            i = batch_idx.item()

            # Extract valid target tokens (exclude padding=-100 and EOS=8192)
            # audio_target is indexed by local_idx (TTS-only tensor)
            valid_mask = (audio_target[local_idx] != -100) & (audio_target[local_idx] != AUDIO_EOS_ID)
            valid_len = valid_mask.sum().item()
            tgt_toks = audio_target[local_idx, :valid_len]

            # Extract predicted tokens from full-batch audio_prediction using batch index
            # Skip BOS at position 0
            pred_toks = audio_prediction[i].argmax(dim=-1)
            pred_toks = pred_toks[1 : valid_len + 1]
            
            # Filter out any accidentally predicted EOS tokens (8192) to prevent CUDA device-side assert
            pred_toks = pred_toks[(pred_toks >= 0) & (pred_toks < 8192)]

            # Decode source text for WER reference (filtering negative tokens)
            valid_input = text_input[i][text_input[i] >= 0]
            ref_text = self.text_tokenizer.decode(
                valid_input, skip_special_tokens=True
            )

            if task_names[i] == "tts_en":
                en_pred_toks.append(pred_toks)
                en_tgt_toks.append(tgt_toks)
                en_ref_texts.append(ref_text)
                en_global.append(global_tokens[local_idx])  # (1, 32)
            elif task_names[i] == "tts_vi":
                vi_pred_toks.append(pred_toks)
                vi_tgt_toks.append(tgt_toks)
                vi_ref_texts.append(ref_text)
                vi_global.append(global_tokens[local_idx])  # (1, 32)

        # ── English TTS evaluation ──
        if en_pred_toks:
            en_global_batch = torch.stack(en_global, dim=0)  # (N, 1, 32)
            en_pred_wavs = self._tokens_to_waveforms(en_pred_toks, en_global_batch)
            en_tgt_wavs = self._tokens_to_waveforms(en_tgt_toks, en_global_batch)

            en_pred_tensor = pad_sequence(en_pred_wavs, batch_first=True, padding_value=0.0)
            en_tgt_tensor = pad_sequence(en_tgt_wavs, batch_first=True, padding_value=0.0)

            sim_scores, _ = self.spk_verifier.verify_batch(en_pred_tensor.float(), en_tgt_tensor.float())
            utmos_scores = self.utmos_model(en_pred_tensor.to(self.eval_dtype), 16000)
            transcriptions = self._whisper_transcribe(en_pred_wavs)

            self._spk_sim_scores.extend(sim_scores.view(-1).tolist())
            self._utmos_scores.extend(utmos_scores.view(-1).tolist())
            self._tts_wer_scores.append(
                wer(
                    reference=en_ref_texts,
                    hypothesis=transcriptions,
                    reference_transform=self.wer_transform,
                    hypothesis_transform=self.wer_transform,
                )
            )

        # ── Vietnamese TTS evaluation ──
        if vi_pred_toks:
            vi_global_batch = torch.stack(vi_global, dim=0)  # (N, 1, 32)
            vi_pred_wavs = self._tokens_to_waveforms(vi_pred_toks, vi_global_batch)
            vi_tgt_wavs = self._tokens_to_waveforms(vi_tgt_toks, vi_global_batch)

            vi_pred_tensor = pad_sequence(vi_pred_wavs, batch_first=True, padding_value=0.0)
            vi_tgt_tensor = pad_sequence(vi_tgt_wavs, batch_first=True, padding_value=0.0)

            sim_scores, _ = self.spk_verifier.verify_batch(vi_pred_tensor.float(), vi_tgt_tensor.float())
            utmos_scores = self.utmos_model(vi_pred_tensor.to(self.eval_dtype), 16000)
            transcriptions = self._phowhisper_transcribe(vi_pred_wavs)

            self._spk_sim_scores.extend(sim_scores.view(-1).tolist())
            self._utmos_scores.extend(utmos_scores.view(-1).tolist())
            self._tts_wer_scores.append(
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
            "asr_wer": (
                sum(self._asr_wer_scores) / len(self._asr_wer_scores)
                if self._asr_wer_scores else 0.0
            ),
            "mt_bleu": (
                sum(self._mt_bleu_scores) / len(self._mt_bleu_scores)
                if self._mt_bleu_scores else 0.0
            ),
            "utmos": (
                sum(self._utmos_scores) / len(self._utmos_scores)
                if self._utmos_scores else 0.0
            ),
            "spk_similarity": (
                sum(self._spk_sim_scores) / len(self._spk_sim_scores)
                if self._spk_sim_scores else 0.0
            ),
            "tts_wer": (
                sum(self._tts_wer_scores) / len(self._tts_wer_scores)
                if self._tts_wer_scores else 0.0
            ),
        }

        # Reset accumulators
        self._asr_wer_scores.clear()
        self._mt_bleu_scores.clear()
        self._utmos_scores.clear()
        self._spk_sim_scores.clear()
        self._tts_wer_scores.clear()

        return results
