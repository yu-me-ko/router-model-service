from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

MODEL_NAME = "distilbert/distilbert-base-multilingual-cased"

print("开始下载 tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("开始下载 model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=5
)

print("保存到本地...")

SAVE_DIR = "./saved_models/router_distilbert"

tokenizer.save_pretrained(
    SAVE_DIR
)

model.save_pretrained(
    SAVE_DIR
)

print("================================")
print("DistilBERT 下载成功")
print("保存位置：", SAVE_DIR)
print("================================")