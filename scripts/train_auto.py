import argparse
import csv
import json
import math
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.evaluate_router_cases import evaluate_model_dir


BASE_MODEL = "distilbert/distilbert-base-multilingual-cased"
DATA_PATH = ROOT_DIR / "data" / "router_train.csv"
LABEL_MAP_PATH = ROOT_DIR / "data" / "router_label_map.json"
OUTPUT_ROOT = ROOT_DIR / "outputs" / "auto_train"
BEST_MODEL_DIR = OUTPUT_ROOT / "best_model"
AUTO_BEST_SAVED_MODEL_DIR = ROOT_DIR / "saved_models" / "router_distilbert_auto_best"

MAX_EPOCHS = 300
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 2e-5
BATCH_SIZE = 8
MAX_LENGTH = 64
RANDOM_SEED = 42
EFFECTIVE_NUMBER_BETA = 0.999

STRATEGIES = [
    "none",
    "inverse_frequency",
    "sqrt_inverse_frequency",
    "effective_number",
]

ALL_STRATEGIES = list(STRATEGIES)


class RouterDataset(Dataset):
    def __init__(self, questions: list[str], labels: list[int], tokenizer):
        self.questions = questions
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> dict:
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automatically train and compare router workflow models.")
    parser.add_argument("--data_path", default="data/router_train.csv", help="Training CSV path.")
    parser.add_argument("--label_map_path", default="data/router_label_map.json", help="Workflow label map JSON path.")
    parser.add_argument("--model_name", default=BASE_MODEL, help="Base HuggingFace model name or path.")
    parser.add_argument("--output_dir", default="outputs/auto_train", help="Auto-training output directory.")
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS, help="Maximum epochs per strategy.")
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE, help="Early stopping patience.")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Optimizer learning rate.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH, help="Tokenizer max sequence length.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=ALL_STRATEGIES,
        default=ALL_STRATEGIES,
        help="One or more class weight strategies to train.",
    )
    return parser.parse_args()


def configure_from_args(args: argparse.Namespace) -> None:
    global BASE_MODEL
    global DATA_PATH
    global LABEL_MAP_PATH
    global OUTPUT_ROOT
    global BEST_MODEL_DIR
    global AUTO_BEST_SAVED_MODEL_DIR
    global MAX_EPOCHS
    global EARLY_STOPPING_PATIENCE
    global LEARNING_RATE
    global BATCH_SIZE
    global MAX_LENGTH
    global RANDOM_SEED
    global STRATEGIES

    BASE_MODEL = args.model_name
    DATA_PATH = resolve_path(args.data_path)
    LABEL_MAP_PATH = resolve_path(args.label_map_path)
    OUTPUT_ROOT = resolve_path(args.output_dir)
    BEST_MODEL_DIR = OUTPUT_ROOT / "best_model"
    AUTO_BEST_SAVED_MODEL_DIR = ROOT_DIR / "saved_models" / "router_distilbert_auto_best"
    MAX_EPOCHS = args.max_epochs
    EARLY_STOPPING_PATIENCE = args.patience
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size
    MAX_LENGTH = args.max_length
    RANDOM_SEED = args.seed
    STRATEGIES = list(args.strategies)


def print_config() -> None:
    print("Auto train config:")
    print(f"data_path={DATA_PATH}")
    print(f"label_map_path={LABEL_MAP_PATH}")
    print(f"model_name={BASE_MODEL}")
    print(f"output_dir={OUTPUT_ROOT}")
    print(f"max_epochs={MAX_EPOCHS}")
    print(f"patience={EARLY_STOPPING_PATIENCE}")
    print(f"learning_rate={LEARNING_RATE}")
    print(f"batch_size={BATCH_SIZE}")
    print(f"max_length={MAX_LENGTH}")
    print(f"seed={RANDOM_SEED}")
    print(f"strategies={STRATEGIES}")


def load_label_map() -> dict[str, int]:
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(label_map: dict[str, int]) -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required_columns = {"question", "workflow"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"router_train.csv missing columns: {sorted(missing_columns)}")

    df = df.dropna(subset=["question", "workflow"]).copy()
    df["question"] = df["question"].astype(str).str.strip()
    df["workflow"] = df["workflow"].astype(str).str.strip()
    df = df[(df["question"] != "") & (df["workflow"] != "")]

    unknown_workflows = sorted(set(df["workflow"]) - set(label_map.keys()))
    if unknown_workflows:
        raise ValueError(f"Found workflows not defined in label map: {unknown_workflows}")

    return df


def compute_class_weights(strategy: str, train_labels: list[int], num_labels: int, device: torch.device) -> torch.Tensor | None:
    if strategy == "none":
        return None

    counts = Counter(train_labels)
    total = len(train_labels)
    weights = []

    for label_id in range(num_labels):
        count = counts.get(label_id, 0)
        if count <= 0:
            weight = 0.0
        elif strategy == "inverse_frequency":
            weight = total / count
        elif strategy == "sqrt_inverse_frequency":
            weight = math.sqrt(total / count)
        elif strategy == "effective_number":
            effective_num = 1.0 - (EFFECTIVE_NUMBER_BETA ** count)
            weight = (1.0 - EFFECTIVE_NUMBER_BETA) / effective_num
        else:
            raise ValueError(f"Unknown class weight strategy: {strategy}")
        weights.append(weight)

    mean_weight = sum(weights) / len(weights)
    if mean_weight > 0:
        weights = [weight / mean_weight for weight in weights]

    return torch.tensor(weights, dtype=torch.float, device=device)


def evaluate_epoch(model, dataloader: DataLoader, loss_fn, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return {
        "loss": total_loss / max(len(dataloader), 1),
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
    }


def train_epoch(model, dataloader: DataLoader, optimizer, scheduler, loss_fn, device: torch.device) -> float:
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def save_metrics_csv(metrics_rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_accuracy",
                "val_macro_f1",
                "val_weighted_f1",
                "current_best_macro_f1",
                "early_stop_counter",
            ],
        )
        writer.writeheader()
        writer.writerows(metrics_rows)


def plot_curve(metrics_rows: list[dict], metric_name: str, output_path: Path, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            raise RuntimeError("Missing matplotlib. Install dependencies with: pip install -r requirements.txt") from exc
        raise

    epochs = [row["epoch"] for row in metrics_rows]
    values = [row[metric_name] for row in metrics_rows]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, values, marker="o")
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(metric_name)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_curves(metrics_rows: list[dict], strategy_dir: Path, strategy: str) -> None:
    plot_curve(metrics_rows, "val_accuracy", strategy_dir / "val_accuracy_curve.png", f"{strategy} validation accuracy")
    plot_curve(metrics_rows, "val_loss", strategy_dir / "val_loss_curve.png", f"{strategy} validation loss")
    plot_curve(metrics_rows, "val_macro_f1", strategy_dir / "val_macro_f1_curve.png", f"{strategy} validation macro F1")


def save_model_artifacts(model, tokenizer, label_map: dict[str, int], model_dir: Path) -> None:
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    with open(model_dir / "router_label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)


def copy_model_dir(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def train_strategy(
    strategy: str,
    train_questions: list[str],
    val_questions: list[str],
    train_labels: list[int],
    val_labels: list[int],
    label_map: dict[str, int],
    device: torch.device,
) -> dict:
    print(f"Training strategy: {strategy}")
    strategy_dir = OUTPUT_ROOT / strategy
    strategy_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = strategy_dir / "best_model"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    id_to_label = {idx: label for label, idx in label_map.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(label_map),
        id2label=id_to_label,
        label2id=label_map,
    )
    model.to(device)

    train_dataset = RouterDataset(train_questions, train_labels, tokenizer)
    val_dataset = RouterDataset(val_questions, val_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class_weights = compute_class_weights(strategy, train_labels, len(label_map), device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * MAX_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_macro_f1 = -1.0
    best_epoch = 0
    best_metrics = {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}
    early_stop_counter = 0
    metrics_rows = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        val_metrics = evaluate_epoch(model, val_loader, loss_fn, device)

        improved = val_metrics["macro_f1"] > best_macro_f1
        if improved:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_metrics = val_metrics
            early_stop_counter = 0
            save_model_artifacts(model, tokenizer, label_map, best_model_dir)
        else:
            early_stop_counter += 1

        metrics_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "current_best_macro_f1": best_macro_f1,
            "early_stop_counter": early_stop_counter,
        })

        print(
            f"{strategy} epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} patience={early_stop_counter}"
        )

        if early_stop_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping strategy {strategy} at epoch {epoch}.")
            break

    save_metrics_csv(metrics_rows, strategy_dir / "metrics.csv")
    save_curves(metrics_rows, strategy_dir, strategy)

    eval_report = evaluate_model_dir(best_model_dir, print_predictions=False)
    final_score = (
        0.45 * best_metrics["macro_f1"]
        + 0.25 * best_metrics["accuracy"]
        + 0.20 * eval_report["workflow_accuracy"]
        + 0.10 * eval_report["tool_hints_satisfaction_rate"]
        - 0.20 * eval_report["critical_error_count"]
    )

    return {
        "best_epoch": best_epoch,
        "best_val_accuracy": best_metrics["accuracy"],
        "best_val_macro_f1": best_metrics["macro_f1"],
        "best_val_weighted_f1": best_metrics["weighted_f1"],
        "eval_cases_workflow_accuracy": eval_report["workflow_accuracy"],
        "eval_cases_tool_hints_satisfaction_rate": eval_report["tool_hints_satisfaction_rate"],
        "critical_error_count": eval_report["critical_error_count"],
        "critical_errors": eval_report["critical_errors"],
        "final_score": final_score,
        "model_path": str(best_model_dir.relative_to(ROOT_DIR)),
    }


def write_training_report(report: dict) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_ROOT / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    configure_from_args(args)
    print_config()

    set_seed(RANDOM_SEED)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    label_map = load_label_map()
    df = load_dataset(label_map)
    distribution = Counter(df["workflow"])
    labels = [label_map[workflow] for workflow in df["workflow"].tolist()]

    train_questions, val_questions, train_labels, val_labels = train_test_split(
        df["question"].tolist(),
        labels,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset total count: {len(df)}")
    print(f"Train count: {len(train_questions)}")
    print(f"Validation count: {len(val_questions)}")

    strategy_reports = {}
    for strategy in STRATEGIES:
        strategy_reports[strategy] = train_strategy(
            strategy,
            train_questions,
            val_questions,
            train_labels,
            val_labels,
            label_map,
            device,
        )

    best_strategy = max(strategy_reports, key=lambda name: strategy_reports[name]["final_score"])
    best_strategy_model = OUTPUT_ROOT / best_strategy / "best_model"
    copy_model_dir(best_strategy_model, BEST_MODEL_DIR)
    copy_model_dir(best_strategy_model, AUTO_BEST_SAVED_MODEL_DIR)

    report = {
        "dataset_total_count": len(df),
        "workflow_distribution": {workflow: distribution.get(workflow, 0) for workflow in label_map},
        "train_count": len(train_questions),
        "validation_count": len(val_questions),
        "strategies": strategy_reports,
        "best_strategy": best_strategy,
        "best_model_path": str(BEST_MODEL_DIR.relative_to(ROOT_DIR)),
    }
    write_training_report(report)

    best = strategy_reports[best_strategy]
    print("Auto training complete")
    print(f"Best strategy: {best_strategy}")
    print(f"Best epoch: {best['best_epoch']}")
    print(f"Best val_macro_f1: {best['best_val_macro_f1']:.4f}")
    print(f"Eval cases workflow accuracy: {best['eval_cases_workflow_accuracy']:.4f}")
    print(f"ToolHints satisfaction: {best['eval_cases_tool_hints_satisfaction_rate']:.4f}")
    print(f"Critical errors: {best['critical_error_count']}")
    print(f"Best model path: {BEST_MODEL_DIR.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
