import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import librosa
from utils.audio_utils import AugmentedAudio, dowload_augmentaion

class Phase1_S2ST(Dataset):
    def __init__(self, files_csv: list, tokenizer, split: str, 
                 transforms: list|None = None, sr: int = 16000,
                 rir_recordings=None, musan_cuts=None):
        dfs = [pd.read_csv(f) for f in files_csv]
        self.datas = pd.concat(dfs, ignore_index=True)

        self.tokenizer = tokenizer
        self.transforms = transforms
        self.sr = sr
        self.split = split
        self.bos = self.tokenizer.convert_tokens_to_ids('<BOS>')
        
        self.rir_recordings = rir_recordings
        self.musan_cuts = musan_cuts
        
        self.task_types = [
            "s2tt_vi2en", "mt_vi2en", "tts_en", "asr_vi",
            "s2tt_en2vi", "mt_en2vi", "tts_vi", "asr_en"
        ]

    def shuffle_data(self, seed=None):
        self.datas = self.datas.sample(frac=1, random_state=seed).reset_index(drop=True)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        task = random.choice(self.task_types)

        ## Special tokens logic
        if task in ['s2tt_vi2en', 's2tt_en2vi']:
            c_task = self.tokenizer.convert_tokens_to_ids('<S2TT>')
            c_lang = self.tokenizer.convert_tokens_to_ids('<ENG>' if task == 's2tt_vi2en' else '<VIE>')
        elif task in ['mt_vi2en', 'mt_en2vi']:
            c_task = self.tokenizer.convert_tokens_to_ids('<MT>')
            c_lang = self.tokenizer.convert_tokens_to_ids('<ENG>' if task == 'mt_vi2en' else '<VIE>')
        elif task in ['tts_en', 'tts_vi']:
            c_task = self.tokenizer.convert_tokens_to_ids('<TTS>')
            c_lang = self.tokenizer.convert_tokens_to_ids('<VIE>' if task == 'tts_vi' else '<ENG>')
        else: # ASR
            c_task = self.tokenizer.convert_tokens_to_ids('<ASR>')
            c_lang = self.tokenizer.convert_tokens_to_ids('<ENG>' if task == 'asr_en' else '<VIE>')

        row = self.datas.iloc[idx] 

        ## Viet Text
        with open(row['VieText'], 'r', encoding='utf-8') as f:
            VieText = f.read().strip()
        ids_VieText = self.tokenizer.encode(VieText)

        ### English Text
        with open(row['EngText'], 'r', encoding='utf-8') as f:
            EngText = f.read().strip()
        ids_EngText = self.tokenizer.encode(EngText)

        ## Audio loading
        if self.split.lower() == 'train' and self.rir_recordings is not None and self.musan_cuts is not None:
            # Augmented audio for training
            augmented_VieAudio = AugmentedAudio(
                rir_recordings=self.rir_recordings, 
                musan_cuts=self.musan_cuts,
                wav_path=row['VieAudio'], 
                transforms=self.transforms,
            )
            augmented_EngAudio = AugmentedAudio(
                rir_recordings=self.rir_recordings, 
                musan_cuts=self.musan_cuts,
                wav_path=row['EngAudio'], 
                transforms=self.transforms,
            )
        else:
            vie_audio_np, _ = librosa.load(path=row['VieAudio'], sr=self.sr, mono=True)
            eng_audio_np, _ = librosa.load(path=row['EngAudio'], sr=self.sr, mono=True)
            augmented_VieAudio = torch.from_numpy(vie_audio_np).float()
            augmented_EngAudio = torch.from_numpy(eng_audio_np).float()

        return {
            'task': task, 
            'c_task': [c_task], 
            'c_lang': [c_lang], 
            'bos': [self.bos],
            'AugmentVieAudio': augmented_VieAudio, # 1D FloatTensor
            'VieText': ids_VieText,                # List of ints
            'AugmentEngAudio': augmented_EngAudio, # 1D FloatTensor
            'EngText': ids_EngText                 # List of ints
        }


def collate_fn_phase1(batch):
    text_encode   = []
    text_decode   = []
    audio_encode  = []
    audio_decode  = []
    sub_tasks     = []
    task_names    = []
    labels        = []

    #Dummies audio and text to make sure batch is alway same
    dummy_audio = torch.tensor([0.0], dtype=torch.float32)
    dummy_text  = torch.tensor([0], dtype=torch.long)

    for row in batch:
        task = row['task']
        
        # 1. S2TT (Audio Encoder -> Text Decoder)
        if task == 's2tt_vi2en' or task == 's2tt_en2vi':
            is_vi2en = (task == 's2tt_vi2en')
            audio_encode.append(row['AugmentVieAudio'] if is_vi2en else row['AugmentEngAudio'])
            audio_decode.append(dummy_audio)

            encode_inputs = row['c_task'] + row['c_lang']
            text_encode.append(torch.tensor(encode_inputs, dtype=torch.long))

            tgt_text = row['EngText'] if is_vi2en else row['VieText']
            decode_inputs = row['bos'] + tgt_text[:-1]
            text_decode.append(torch.tensor(decode_inputs, dtype=torch.long))

            sub_tasks.append(2)
            labels.append(torch.tensor(tgt_text, dtype=torch.long))
            task_names.append('s2tt_vi2en' if is_vi2en else 's2tt_en2vi')

        # 2. MT (Text Encoder -> Text Decoder)
        elif task == 'mt_vi2en' or task == 'mt_en2vi':
            is_vi2en = (task == 'mt_vi2en')
            audio_encode.append(dummy_audio)
            audio_decode.append(dummy_audio)

            src_text = row['VieText'] if is_vi2en else row['EngText']
            tgt_text = row['EngText'] if is_vi2en else row['VieText']

            encode_inputs = row['c_task'] + row['c_lang'] + src_text
            text_encode.append(torch.tensor(encode_inputs, dtype=torch.long))

            decode_inputs = row['bos'] + tgt_text[:-1]
            text_decode.append(torch.tensor(decode_inputs, dtype=torch.long))

            sub_tasks.append(4)
            labels.append(torch.tensor(tgt_text, dtype=torch.long))
            task_names.append('mt_vi2en' if is_vi2en else 'mt_en2vi')

        # 3. TTS (Text Encoder -> Audio Decoder)
        elif task == 'tts_en' or task == 'tts_vi':
            is_en = (task == 'tts_en')
            audio_encode.append(dummy_audio)
            audio_decode.append(row['AugmentEngAudio'] if is_en else row['AugmentVieAudio'])

            src_text = row['EngText'] if is_en else row['VieText']
            
            encode_inputs = row['c_task'] + row['c_lang'] + src_text
            text_encode.append(torch.tensor(encode_inputs, dtype=torch.long))

            text_decode.append(dummy_text)
            sub_tasks.append(3)
            labels.append(torch.tensor([-100], dtype=torch.long))
            task_names.append('tts_en' if is_en else 'tts_vi')


        # 4. ASR (Audio Encoder -> Text Decoder)
        elif task == 'asr_en' or task == 'asr_vi':
            is_en = (task == 'asr_en')
            audio_encode.append(row['AugmentEngAudio'] if is_en else row['AugmentVieAudio'])
            audio_decode.append(dummy_audio)

            encode_inputs = row['c_task'] + row['c_lang']
            text_encode.append(torch.tensor(encode_inputs, dtype=torch.long))

            tgt_text = row['EngText'] if is_en else row['VieText']
            decode_inputs = row['bos'] + tgt_text[:-1]
            text_decode.append(torch.tensor(decode_inputs, dtype=torch.long))

            sub_tasks.append(1)
            labels.append(torch.tensor(tgt_text, dtype=torch.long))
            task_names.append('asr_en' if is_en else 'asr_vi')


    # Padding step
    text_encode  = pad_sequence(text_encode, batch_first=True, padding_value=0)
    text_decode  = pad_sequence(text_decode, batch_first=True, padding_value=0)
    audio_encode = pad_sequence(audio_encode, batch_first=True, padding_value=0.0)
    audio_decode = pad_sequence(audio_decode, batch_first=True, padding_value=0.0)
    
    sub_tasks    = torch.tensor(sub_tasks, dtype=torch.long)
    labels       = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        'text_encode'  : text_encode,  # (Batch, Enc_Seq_Len)
        'text_decode'  : text_decode,  # (Batch, Dec_Seq_Len)
        'audio_encode' : audio_encode, # (Batch, Audio_Enc_Samples)
        'audio_decode' : audio_decode, # (Batch, Audio_Dec_Samples)
        'sub_tasks'    : sub_tasks,    # (Batch,)
        'task_names'   : task_names,   # List of string
        'labels'       : labels,       # (Batch, Dec_Seq_Len)
    }