import json
import os
from collections import Counter

import pandas as pd
import torch

torch.set_num_threads(12)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


BASE_MODEL = "distilbert/distilbert-base-multilingual-cased"
DATA_PATH = "data/router_train.csv"
LABEL_MAP_PATH = "data/router_label_map.json"
SAVE_DIR = "saved_models/router_distilbert"
MAX_LENGTH = 64
MIN_CLASS_WARN_COUNT = 20

LEGACY_LABEL_MAP = {
    "NORMAL_KNOWLEDGE_QA": "KNOWLEDGE_QA",
    "DIRECT_AI_FALLBACK": "DIRECT_AI",
}


class RouterDataset(Dataset):
    def __init__(self, questions, labels, tokenizer):
        self.questions = questions
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.questions[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_training_frame(label_map: dict[str, int]) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    if "question" not in df.columns:
        raise ValueError("CSV must contain a question column.")

    label_column = "workflow" if "workflow" in df.columns else "route" if "route" in df.columns else None
    if label_column is None:
        raise ValueError("CSV must contain workflow or route column.")

    df = df.dropna(subset=["question", label_column]).copy()
    df["question"] = df["question"].astype(str).str.strip()
    df["workflow"] = df[label_column].astype(str).str.strip().replace(LEGACY_LABEL_MAP)
    df = df[df["question"] != ""]

    unknown_workflows = sorted(set(df["workflow"]) - set(label_map.keys()))
    if unknown_workflows:
        raise ValueError(f"Found workflows not defined in label map: {unknown_workflows}")

    return df


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Training data not found: {DATA_PATH}")

    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError(f"Label map not found: {LABEL_MAP_PATH}")

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    df = load_training_frame(label_map)
    questions = df["question"].tolist()
    workflows = df["workflow"].tolist()
    labels = [label_map[workflow] for workflow in workflows]
    distribution = Counter(workflows)

    for workflow in label_map:
        count = distribution.get(workflow, 0)
        if count < MIN_CLASS_WARN_COUNT:
            print(f"WARN: workflow {workflow} has only {count} samples.")

    stratify = labels if min(Counter(labels).values()) >= 2 else None
    train_questions, eval_questions, train_labels, eval_labels = train_test_split(
        questions,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    id_to_label = {idx: label for label, idx in label_map.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(label_map),
        id2label=id_to_label,
        label2id=label_map,
    )

    train_dataset = RouterDataset(train_questions, train_labels, tokenizer)
    eval_dataset = RouterDataset(eval_questions, eval_labels, tokenizer)

    training_args = TrainingArguments(
        output_dir="outputs/router_train",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_steps=5,
        load_best_model_at_end=True,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    os.makedirs(SAVE_DIR, exist_ok=True)
    tokenizer.save_pretrained(SAVE_DIR)
    model.save_pretrained(SAVE_DIR)

    with open(os.path.join(SAVE_DIR, "router_label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print("================================")
    print("Router workflow model training complete")
    print(f"Total samples: {len(df)}")
    print("Workflow distribution:")
    for workflow in label_map:
        print(f"  {workflow}: {distribution.get(workflow, 0)}")
    print(f"Model saved to: {SAVE_DIR}")
    print("================================")


if __name__ == "__main__":
    main()
