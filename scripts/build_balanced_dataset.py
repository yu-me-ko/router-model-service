import csv
import argparse
from collections import Counter
from pathlib import Path


EXTRACTED_PATH = Path("data/extracted_public_qa.csv")
GENERATED_PATH = Path("data/generated_router_train.csv")
DEFAULT_OUTPUT_PATH = Path("data/router_train.csv")
FIELDNAMES = ["question", "workflow", "tools"]

WORKFLOW_LIMITS = {
    "KNOWLEDGE_QA": 600,
    "AGENT_TASK": 450,
    "FILE_QA": 400,
    "USER_KNOWLEDGE_QA": 400,
    "DIRECT_AI": 350,
    "SYSTEM_ACTION": 300,
    "UNKNOWN": 250,
}

REQUIRED_QUESTIONS = {
    "我饿了",
    "我现在饿了",
    "明天食堂开门吗",
    "明天有什么作业要交吗",
    "帮我制定明天的日程表",
    "总结我刚刚上传的文件",
    "查询我的全局知识库",
    "删除这个对话",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def normalize_row(row: dict[str, str]) -> dict[str, str] | None:
    question = (row.get("question") or "").strip()
    workflow = (row.get("workflow") or row.get("route") or "").strip()
    tools = (row.get("tools") or "").strip()

    if not question or not workflow:
        return None

    return {
        "question": question,
        "workflow": workflow,
        "tools": tools,
    }


def append_if_possible(row, output_rows, seen_questions, counts) -> bool:
    normalized = normalize_row(row)
    if normalized is None:
        return False

    question = normalized["question"]
    workflow = normalized["workflow"]
    limit = WORKFLOW_LIMITS.get(workflow)

    if limit is None or question in seen_questions or counts[workflow] >= limit:
        return False

    seen_questions.add(question)
    counts[workflow] += 1
    output_rows.append(normalized)
    return True


def build_rows(extracted_rows, generated_rows) -> list[dict[str, str]]:
    output_rows = []
    seen_questions = set()
    counts = Counter()

    for row in generated_rows:
        if (row.get("question") or "").strip() in REQUIRED_QUESTIONS:
            append_if_possible(row, output_rows, seen_questions, counts)

    for row in extracted_rows:
        append_if_possible(row, output_rows, seen_questions, counts)

    for row in generated_rows:
        append_if_possible(row, output_rows, seen_questions, counts)

    return output_rows


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
    parser = argparse.ArgumentParser(description="Build balanced router training dataset.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output CSV path.")
    args = parser.parse_args()
    output_path = Path(args.output)

    if not GENERATED_PATH.exists():
        raise FileNotFoundError("请先运行 python scripts/generate_seed_data.py")

    if not EXTRACTED_PATH.exists():
        print("WARN: 未找到 data/extracted_public_qa.csv，将仅使用脚本生成数据构建训练集。")
        print("      如需接入真实校园问题，请先运行 python scripts/extract_docx_questions.py")

    generated_rows = read_rows(GENERATED_PATH)
    extracted_rows = read_rows(EXTRACTED_PATH)
    rows = build_rows(extracted_rows, generated_rows)
    write_rows(rows, output_path)

    counts = Counter(row["workflow"] for row in rows)
    print("均衡训练集构建完成")
    print("总条数:", len(rows))
    for workflow, limit in WORKFLOW_LIMITS.items():
        count = counts.get(workflow, 0)
        print(f"{workflow}: {count}/{limit}")
        if count < limit:
            print(f"WARN: {workflow} 样本不足 {limit} 条，未复制重复数据。")
    print("输出文件:", output_path)


if __name__ == "__main__":
    main()
