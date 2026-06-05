import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.routing_rules import infer_tool_hints
from scripts.router_augmentations import get_augmented_rows


INPUT_PATH = Path("data/raw/BombGPT_\u8def\u7531\u8bad\u7ec3\u6570\u636e_3118\u6761.xlsx")
OUTPUT_PATH = Path("data/router_train.csv")
LABEL_MAP_PATH = Path("data/router_label_map.json")
HARD_TRAIN_PATH = Path("data/router_hard_train_cases.csv")
SHEET_NAME = "\u5168\u90e8\u6570\u636e"
FIELDNAMES = ["question", "workflow", "tools"]
REQUIRED_COLUMNS = {"question", "suggested_workflow"}


def load_known_workflows() -> set[str]:
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        return set(json.load(f).keys())


def read_excel_rows(path: Path) -> list[dict]:
    try:
        import pandas as pd
    except ModuleNotFoundError:
        print("缺少依赖 pandas，请先安装项目依赖。")
        sys.exit(1)

    try:
        df = pd.read_excel(path, sheet_name=SHEET_NAME, dtype=str, engine="openpyxl")
    except ModuleNotFoundError as exc:
        if exc.name == "openpyxl":
            print("缺少依赖 openpyxl，请先安装：pip install -r requirements.txt")
            sys.exit(1)
        raise
    except ImportError as exc:
        if "openpyxl" in str(exc):
            print("缺少依赖 openpyxl，请先安装：pip install -r requirements.txt")
            sys.exit(1)
        raise

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"Excel sheet {SHEET_NAME} 缺少列: {sorted(missing_columns)}")

    return df.fillna("").to_dict("records")


def normalize_row(row: dict, known_workflows: set[str]) -> tuple[dict[str, str] | None, str | None]:
    question = str(row.get("question") or "").strip()
    workflow = str(row.get("suggested_workflow") or "").strip()

    if not question:
        return None, "empty_question"
    if not workflow:
        return None, "empty_workflow"
    if workflow not in known_workflows:
        return None, workflow

    tools = "|".join(infer_tool_hints(question, workflow))
    return {"question": question, "workflow": workflow, "tools": tools}, None


def convert_rows(raw_rows: list[dict], known_workflows: set[str]) -> tuple[list[dict[str, str]], Counter]:
    output_rows = []
    seen_questions = set()
    skipped = Counter()

    for raw_row in raw_rows:
        row, skip_reason = normalize_row(raw_row, known_workflows)
        if row is None:
            skipped[skip_reason or "unknown"] += 1
            continue

        question = row["question"]
        if question in seen_questions:
            skipped["duplicate_question"] += 1
            continue

        seen_questions.add(question)
        output_rows.append(row)

    return output_rows, skipped


def read_training_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def normalize_training_row(row: dict, known_workflows: set[str]) -> tuple[dict[str, str] | None, str | None]:
    question = (row.get("question") or "").strip()
    workflow = (row.get("workflow") or row.get("route") or "").strip()
    tools = (row.get("tools") or "").strip()

    if not question:
        return None, "empty_question"
    if not workflow:
        return None, "empty_workflow"
    if workflow not in known_workflows:
        return None, workflow

    if not tools:
        tools = "|".join(infer_tool_hints(question, workflow))

    return {"question": question, "workflow": workflow, "tools": tools}, None


def merge_training_rows(
    rows: list[dict[str, str]],
    extra_rows: list[dict[str, str]],
    known_workflows: set[str],
) -> tuple[list[dict[str, str]], Counter]:
    merged_rows = list(rows)
    seen_questions = {row["question"] for row in merged_rows}
    skipped = Counter()

    for raw_row in extra_rows:
        row, skip_reason = normalize_training_row(raw_row, known_workflows)
        if row is None:
            skipped[skip_reason or "unknown"] += 1
            continue
        if row["question"] in seen_questions:
            skipped["duplicate_question"] += 1
            continue

        seen_questions.add(row["question"])
        merged_rows.append(row)

    return merged_rows, skipped


def merge_augmented_rows(rows: list[dict[str, str]], known_workflows: set[str]) -> tuple[list[dict[str, str]], Counter]:
    return merge_training_rows(rows, get_augmented_rows(), known_workflows)


def write_rows(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_path.with_suffix(".tmp")

    try:
        with open(temp_output, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

        temp_output.replace(output_path)
    except PermissionError:
        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import latest BombGPT Excel router dataset.")
    parser.add_argument("--input", default=str(INPUT_PATH), help="Input xlsx path.")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output CSV path.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print("请将 Excel 文件放入 data/raw/ 目录。")
        print(f"期望文件路径: {input_path}")
        sys.exit(1)

    known_workflows = load_known_workflows()
    raw_rows = read_excel_rows(input_path)
    rows, skipped = convert_rows(raw_rows, known_workflows)
    excel_import_count = len(rows)

    hard_rows = read_training_csv(HARD_TRAIN_PATH)
    rows, hard_skipped = merge_training_rows(rows, hard_rows, known_workflows)
    skipped.update({f"hard_train_{key}": value for key, value in hard_skipped.items()})

    rows, augment_skipped = merge_augmented_rows(rows, known_workflows)
    skipped.update({f"augmentation_{key}": value for key, value in augment_skipped.items()})

    non_workflow_skip_reasons = {
        "empty_question",
        "empty_workflow",
        "duplicate_question",
        "augmentation_empty_question",
        "augmentation_empty_workflow",
        "augmentation_duplicate_question",
        "hard_train_empty_question",
        "hard_train_empty_workflow",
        "hard_train_duplicate_question",
    }
    for reason, count in sorted(skipped.items()):
        if reason not in non_workflow_skip_reasons:
            print(f"WARN: 发现未知 workflow {reason}: {count} 行，已跳过。")

    write_rows(rows, output_path)

    workflow_counts = Counter(row["workflow"] for row in rows)
    dedupe_count = (
        skipped.get("duplicate_question", 0)
        + skipped.get("hard_train_duplicate_question", 0)
        + skipped.get("augmentation_duplicate_question", 0)
    )
    print("Excel 数据导入完成")
    print("输入文件:", input_path)
    print("输出文件:", output_path)
    print("Excel 原始导入数量:", excel_import_count)
    print("hard_train_cases 数量:", len(hard_rows))
    print("去重数量:", dedupe_count)
    print("总条数:", len(rows))
    print("跳过统计:")
    for reason, count in sorted(skipped.items()):
        print(f"  {reason}: {count}")
    print("workflow 分布:")
    for workflow in sorted(known_workflows):
        print(f"  {workflow}: {workflow_counts.get(workflow, 0)}")


if __name__ == "__main__":
    main()
