import argparse
import csv
import json
from collections import Counter
from pathlib import Path


DEFAULT_PATH = Path("data/router_train.csv")
LABEL_MAP_PATH = Path("data/router_label_map.json")
OLD_LABELS = {"NORMAL_KNOWLEDGE_QA", "DIRECT_AI_FALLBACK"}
HIGH_RATIO_WARN = 0.40
LOW_COUNT_WARN = 100


def load_known_workflows() -> set[str]:
    if not LABEL_MAP_PATH.exists():
        return set()

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        return set(json.load(f).keys())


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def count_tools(rows: list[dict[str, str]]) -> Counter:
    counts = Counter()
    for row in rows:
        for tool in (row.get("tools") or "").split("|"):
            tool = tool.strip()
            if tool:
                counts[tool] += 1
    return counts


def analyze(path: Path) -> None:
    rows = read_rows(path)
    known_workflows = load_known_workflows()
    total = len(rows)

    questions = [(row.get("question") or "").strip() for row in rows]
    workflows = [(row.get("workflow") or row.get("route") or "").strip() for row in rows]
    workflow_counts = Counter(workflows)
    question_counts = Counter(question for question in questions if question)
    tool_counts = count_tools(rows)

    duplicate_question_count = sum(count - 1 for count in question_counts.values() if count > 1)
    empty_question_count = sum(1 for question in questions if not question)
    unknown_workflow_count = sum(
        1 for workflow in workflows if workflow and known_workflows and workflow not in known_workflows
    )

    print(f"数据文件: {path}")
    print(f"总条数: {total}")
    print("workflow 分布:")

    for workflow, count in sorted(workflow_counts.items()):
        ratio = count / total if total else 0
        name = workflow or "<EMPTY>"
        print(f"  {name}: {count} ({ratio:.2%})")

    print(f"重复 question 数量: {duplicate_question_count}")
    print(f"空 question 数量: {empty_question_count}")
    print(f"未知 workflow 数量: {unknown_workflow_count}")
    print("tools 分布:")

    for tool, count in sorted(tool_counts.items()):
        print(f"  {tool}: {count}")

    for workflow, count in sorted(workflow_counts.items()):
        ratio = count / total if total else 0
        if ratio > HIGH_RATIO_WARN:
            print(f"WARN: {workflow} 占比过高，可能导致模型偏向该 workflow。")
        if workflow and count < LOW_COUNT_WARN:
            print(f"WARN: {workflow} 样本过少。")

    old_labels = sorted(OLD_LABELS & set(workflows))
    if old_labels:
        print(f"WARN: 发现旧标签: {old_labels}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze router workflow dataset quality.")
    parser.add_argument("path", nargs="?", default=str(DEFAULT_PATH), help="CSV path. Defaults to data/router_train.csv.")
    args = parser.parse_args()

    analyze(Path(args.path))


if __name__ == "__main__":
    main()
