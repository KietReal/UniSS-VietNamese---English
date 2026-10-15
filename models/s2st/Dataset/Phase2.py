from typing import Text
import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import librosa
from utils.audio_utils import AugmentedAudio, dowload_augmentaion

class Phase2_S2ST(Dataset):
    def __init__(self, files_csv: list, tokenizer, split: str,
                 transforms: list|None = None, sr: int = 16000,
                 rir_recordings=None, musan_cuts=None):
        dfs = [pd.read_csv(f) for f in files_csv]
        self.datas = pd.concat(dfs, ignore_index=True)

        self.tokenizer = tokenizer
        self.transforms = transforms
        self.sr = sr
        self.split = split

        self.c_task = self.tokenizer.convert_tokens_to_ids("<S2ST>")
        self.c_task = torch.tensor([self.c_task])

        self.vie    = self.tokenizer.convert_tokens_to_ids("<VIE>")
        self.vie    = torch.tensor([self.vie])

        self.eng    = self.tokenizer.convert_tokens_to_ids("<ENG>")
        self.eng    = torch.tensor([self.eng])

        speed_token_strs = ['<0.5>', '<0.6>', '<0.7>', '<0.8>', '<0.9>', '<1.0>',
                            '<1.1>', '<1.2>', '<1.3>', '<1.4>', '<1.5>', '<1.6>',
                            '<1.7>', '<1.8>', '<1.9>', '<2.0>']
        self.speed_token_ids = {
            tok: self.tokenizer.convert_tokens_to_ids(tok)
            for tok in speed_token_strs
        }

        self.rir_recordings = rir_recordings
        self.musan_cuts = musan_cuts

    def shuffle_data(self, seed=None):
        self.datas = self.datas.sample(frac=1, random_state=seed).reset_index(drop=True)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        row = self.datas.iloc[idx]

        c_speed = '<'+str(row['Ratio'])+'>'
        c_speed_id = self.speed_token_ids[c_speed]
        c_speed = torch.tensor([c_speed_id])

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
            augmented_VieAudio, ori_VieAudio = AugmentedAudio(
                rir_recordings=self.rir_recordings,
                musan_cuts=self.musan_cuts,
                wav_path=row['VieAudio'],
                transforms=self.transforms,
            )
            augmented_EngAudio, ori_EngAudio = AugmentedAudio(
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
            ori_VieAudio = augmented_VieAudio
            ori_EngAudio = augmented_EngAudio

        return {
            'c_task' : self.c_task,                # (1,) int64
            'c_speed': c_speed,                    # (1,) int64
            'vie'    : self.vie,                   # (1,) int64
            'VieText': ids_VieText,                # List int64
            'eng'    : self.eng,                   # (1,) int64
            'EngText': ids_EngText,                # List int64
            'AugmentVieAudio': augmented_VieAudio, # 1D FloatTensor
            'VieAudio': ori_VieAudio,              # 1D FloatTensor
            'AugmentEngAudio': augmented_EngAudio, # 1D FloatTensor
            'EngAudio': ori_EngAudio,              # 1D FloatTensor
        }


def collate_fn_phase2(batch):
    special_tokens = []
    audio_encode   = []
    audio_decode   = []
    VieEng_Text    = []

    for row in batch:
        vi2en_tokens = torch.cat([row['c_task'], row['eng'], row['c_speed']])   # (3,) int64
        en2vi_tokens = torch.cat([row['c_task'], row['vie'], row['c_speed']])   # (3,) int64

        # Vie -> Eng
        special_tokens.append(vi2en_tokens)
        audio_encode.append(row['AugmentVieAudio'])
        audio_decode.append(row['EngAudio'])
        VieEng_Text.append(torch.tensor(row['EngText'], dtype=torch.long))

        # Eng -> Vie
        special_tokens.append(en2vi_tokens)
        audio_encode.append(row['AugmentEngAudio'])
        audio_decode.append(row['VieAudio'])
        VieEng_Text.append(torch.tensor(row['VieText'], dtype=torch.long))

    special_tokens = torch.stack(special_tokens)                                 # (B, 3)

    audio_encode = pad_sequence(audio_encode, batch_first=True, padding_value=0.0)
    audio_decode = pad_sequence(audio_decode, batch_first=True, padding_value=0.0)
    VieEng_Text  = pad_sequence(VieEng_Text, batch_first=True, padding_value=-100)

    return {
        'special_tokens': special_tokens,  # (Batch, 3) int64
        'audio_encode'  : audio_encode,    # (Batch, Audio_Enc_Samples)
        'audio_decode'  : audio_decode,    # (Batch, Audio_Dec_Samples)
        'VieEng_Text'   : VieEng_Text,     # (Batch, Len)
    }