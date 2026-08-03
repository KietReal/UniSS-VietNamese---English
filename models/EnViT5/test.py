# pyrefly: ignore [missing-import]
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


import os
current_dir = os.path.dirname(os.path.abspath(__file__))
model_name = os.path.abspath(os.path.join(current_dir, "../checkpoints/envit5-translation"))
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)  
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# Tokenize
inputs = tokenizer(["Hello world", "how are you"], padding=True, return_tensors="pt")
targets = tokenizer("Xin chào", padding=True, return_tensors="pt")
print(inputs)
    
    # Optional: only run lm_head if you actually need logits
    # logits = model.lm_head(pre_head)
# ['en: VietAI is a non-profit organization with the mission of nurturing artificial intelligence talents and building an international - class community of artificial intelligence experts in Vietnam.',
#  'en: According to the latest LinkedIn report on the 2020 list of attractive and promising jobs, AI - related job titles such as AI Specialist, ML Engineer and ML Engineer all rank high.',
#  'vi: Nhóm chúng tôi khao khát tạo ra những khám phá có ảnh hưởng đến mọi người, và cốt lõi trong cách tiếp cận của chúng tôi là chia sẻ nghiên cứu và công cụ để thúc đẩy sự tiến bộ trong lĩnh vực này.',
#  'vi: Chúng ta đang trên hành trình tiến bộ và dân chủ hoá trí tuệ nhân tạo thông qua mã nguồn mở và khoa học mở.']

