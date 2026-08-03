# pyrefly: ignore [missing-import]
from huggingface_hub import snapshot_download

#snapshot_download('vinai/PhoST', repo_type='dataset', allow_patterns=['text_data/*'])

import os
import yaml
import librosa
from tqdm import tqdm
import soundfile as sf
from .processing import processing_text

text_root_path = '/content/drive/MyDrive/PhoST/text_data'
audio_root_path= '/content/drive/MyDrive/PhoST/audio_data'
root_save_path = '/content/drive/MyDrive/PhoST/Processed'

number_of_err = []
PAD = 0.1

for split in os.listdir(text_root_path):
  split_text_path = os.path.join(text_root_path, split)

  for number in os.listdir(split_text_path):

    ### Text
    en_text_path = f"{split_text_path}/{number}/{number}.en"
    vi_text_path = f"{split_text_path}/{number}/{number}.vi"
    config_text_path = f"{split_text_path}/{number}/{number}.yaml"

    with open(en_text_path, 'r', encoding='utf-8') as f:
      en_text = f.readlines()

    with open(vi_text_path, 'r', encoding='utf-8') as f:
      vi_text = f.readlines()

    with open(config_text_path, 'r') as f:
      config_data = yaml.safe_load(f)

    ### Audio
    audio_path = f"{audio_root_path}/{split}/wav/{number}.wav"
    audio, sr = librosa.load(audio_path, sr=None, mono=True)

    ### Check missing text
    if len(en_text)!=len(vi_text) or len(en_text)!=len(config_data):
      number_of_err.append(f"{split}/{number}")
      continue

    ### Save folder
    folder_save = f"{root_save_path}/{split}/{number}"
    os.makedirs(folder_save, exist_ok=True)

    for idx, (en, vi, config) in tqdm(
      enumerate(zip(en_text, vi_text, config_data)),
      total=len(en_text)
    ):

      en = processing_text(en)
      vi = processing_text(vi)

      if not en or not vi:
          continue

      duration = config['duration']
      offset   = config['offset']

      start_sample = max(
          0,
          int(sr * (offset - PAD))
      )

      end_sample = min(
          len(audio),
          int(sr * (offset + duration + PAD))
      )

      if end_sample <= start_sample:
          continue

      audio_segment = audio[start_sample:end_sample]

      # ==========================
      # Folder của sample
      # ==========================

      sample_folder = os.path.join(
          folder_save,
          f"{number}_{idx}"
      )

      os.makedirs(sample_folder, exist_ok=True)

      # ==========================
      # Save audio
      # ==========================

      sf.write(
          os.path.join(sample_folder, f"{idx}.wav"),
          audio_segment,
          sr,
          subtype="PCM_16"
      )

      # ==========================
      # Save text
      # ==========================

      with open(
          os.path.join(sample_folder, f"{idx}.vi"),
          "w",
          encoding="utf-8"
      ) as f:
          f.write(vi)

      with open(
          os.path.join(sample_folder, f"{idx}.en"),
          "w",
          encoding="utf-8"
      ) as f:
          f.write(en)
