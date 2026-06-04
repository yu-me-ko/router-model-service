from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

import torch

PATH = "./saved_models/router_distilbert"

tokenizer = AutoTokenizer.from_pretrained(PATH)

model = AutoModelForSequenceClassification.from_pretrained(PATH)

question = "帮我制定明天的日程"

inputs = tokenizer(
    question,
    return_tensors="pt"
)

with torch.no_grad():

    outputs = model(**inputs)

print("输出维度：")

print(outputs.logits.shape)