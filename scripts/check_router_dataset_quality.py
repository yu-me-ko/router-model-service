import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


TRAIN_PATH = Path("data/router_train.csv")
HARD_PATH = Path("data/router_hard_train_cases.csv")
CHALLENGE_PATH = Path("data/router_challenge_eval_cases.csv")
LABEL_MAP_PATH = Path("data/router_label_map.json")

ALLOWED_WORKFLOWS = {
    "KNOWLEDGE_QA",
    "AGENT_TASK",
    "FILE_QA",
    "USER_KNOWLEDGE_QA",
    "DIRECT_AI",
    "SYSTEM_ACTION",
    "UNKNOWN",
}
OLD_LABELS = {"NORMAL_KNOWLEDGE_QA", "DIRECT_AI_FALLBACK"}
ALLOWED_TOOLS = {
    "GET_CURRENT_TIME",
    "SEARCH_PUBLIC_KNOWLEDGE",
    "SEARCH_CONVERSATION_KNOWLEDGE",
    "SEARCH_GLOBAL_KNOWLEDGE",
    "SEARCH_FILE_CONTENT",
    "SEARCH_CONVERSATION_HISTORY",
    "SEARCH_USER_PROFILE",
    "SEARCH_USER_SCHEDULE",
    "SEARCH_USER_TODO",
    "SEARCH_COLLEGE_EVENTS",
    "WRITE_USER_DATA",
    "DIRECT_LLM",
}

REQUIRED_BOUNDARIES = {
    "KNOWLEDGE_QA_TIME_LIFE": {
        "workflow": "KNOWLEDGE_QA",
        "any_question": ["我现在饿了", "现在还有饭吃吗", "现在还能去饭堂买饮料吗"],
        "all_tools": {"GET_CURRENT_TIME", "SEARCH_PUBLIC_KNOWLEDGE"},
    },
    "KNOWLEDGE_QA_TIME_CAMPUS": {
        "workflow": "KNOWLEDGE_QA",
        "any_question": ["明天食堂开门吗", "明天饭堂营业吗", "明天图书馆几点开门"],
        "all_tools": {"GET_CURRENT_TIME", "SEARCH_PUBLIC_KNOWLEDGE"},
    },
    "AGENT_TASK": {
        "workflow": "AGENT_TASK",
        "any_question": ["明天有什么作业要交吗", "明天有哪些作业要交", "明天还有哪些作业需要提交"],
        "all_tools": {"GET_CURRENT_TIME"},
    },
    "FILE_QA": {
        "workflow": "FILE_QA",
        "any_question": ["我上传的课程表里周三有什么课", "我上传的课表里明天有什么课", "我发的课表文件里明天排了什么课"],
        "all_tools": {"SEARCH_FILE_CONTENT"},
    },
    "DIRECT_AI": {
        "workflow": "DIRECT_AI",
        "any_question": ["给我讲个笑话", "讲个关于大学生的冷笑话", "以华工校园为主题写几句诗"],
        "all_tools": {"DIRECT_LLM"},
    },
    "SYSTEM_ACTION": {
        "workflow": "SYSTEM_ACTION",
        "any_question": ["删除这个对话", "删掉当前聊天记录", "请把刚传的资料存进全局知识库"],
        "all_tools": {"WRITE_USER_DATA"},
    },
    "UNKNOWN": {
        "workflow": "UNKNOWN",
        "any_question": ["刚才那个呢", "那这个呢", "刚刚这个要不要做"],
        "all_tools": {"SEARCH_CONVERSATION_HISTORY"},
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def split_tools(value: str) -> set[str]:
    return {tool.strip() for tool in (value or "").split("|") if tool.strip()}


def count_duplicate_questions(rows: list[dict[str, str]]) -> int:
    counts = Counter((row.get("question") or "").strip() for row in rows if (row.get("question") or "").strip())
    return sum(count - 1 for count in counts.values() if count > 1)


def count_empty(rows: list[dict[str, str]], column: str) -> int:
    return sum(1 for row in rows if not (row.get(column) or "").strip())


def count_by_column(rows: list[dict[str, str]], column: str) -> Counter:
    return Counter((row.get(column) or "").strip() for row in rows)


def invalid_labels(rows: list[dict[str, str]], column: str) -> set[str]:
    return {
        (row.get(column) or "").strip()
        for row in rows
        if (row.get(column) or "").strip() and (row.get(column) or "").strip() not in ALLOWED_WORKFLOWS
    }


def old_labels(rows: list[dict[str, str]], column: str) -> set[str]:
    return {
        (row.get(column) or "").strip()
        for row in rows
        if (row.get(column) or "").strip() in OLD_LABELS
    }


def invalid_tools(rows: list[dict[str, str]], column: str) -> set[str]:
    found = set()
    for row in rows:
        for tool in split_tools(row.get(column) or ""):
            if tool not in ALLOWED_TOOLS:
                found.add(tool)
    return found


def question_set(rows: list[dict[str, str]]) -> set[str]:
    return {(row.get("question") or "").strip() for row in rows if (row.get("question") or "").strip()}


def print_distribution(title: str, counts: Counter, allowed_order: list[str]) -> None:
    print(title)
    for key in allowed_order:
        print(f"  {key}: {counts.get(key, 0)}")


def preview_rows(rows: list[dict[str, str]], group_column: str, columns: list[str]) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get(group_column) or "").strip()].append(row)

    rng = random.Random(42)
    for workflow in sorted(ALLOWED_WORKFLOWS):
        sample = grouped.get(workflow, [])
        if not sample:
            continue
        print(f"Preview {group_column}={workflow}:")
        for row in rng.sample(sample, min(10, len(sample))):
            print("  " + ",".join((row.get(column) or "").strip() for column in columns))


def check_required_boundaries(train_rows: list[dict[str, str]]) -> list[str]:
    failures = []
    by_question = {(row.get("question") or "").strip(): row for row in train_rows}

    for name, rule in REQUIRED_BOUNDARIES.items():
        matched = False
        for question in rule["any_question"]:
            row = by_question.get(question)
            if not row:
                continue
            workflow = (row.get("workflow") or "").strip()
            tools = split_tools(row.get("tools") or "")
            if workflow == rule["workflow"] and rule["all_tools"].issubset(tools):
                matched = True
                break
        if not matched:
            failures.append(f"missing required boundary sample: {name}")

    return failures


def main() -> None:
    train_rows = read_csv(TRAIN_PATH)
    hard_rows = read_csv(HARD_PATH)
    challenge_rows = read_csv(CHALLENGE_PATH)

    train_questions = question_set(train_rows)
    hard_questions = question_set(hard_rows)
    challenge_questions = question_set(challenge_rows)

    hard_challenge_overlap = hard_questions & challenge_questions
    train_challenge_overlap = train_questions & challenge_questions

    failures = []

    print(f"router_train.csv 总条数: {len(train_rows)}")
    print_distribution("router_train.csv workflow 数量:", count_by_column(train_rows, "workflow"), sorted(ALLOWED_WORKFLOWS))
    print(f"router_hard_train_cases.csv 总条数: {len(hard_rows)}")
    print_distribution("router_hard_train_cases.csv workflow 数量:", count_by_column(hard_rows, "workflow"), sorted(ALLOWED_WORKFLOWS))
    print(f"router_challenge_eval_cases.csv 总条数: {len(challenge_rows)}")
    print_distribution("router_challenge_eval_cases.csv expected_workflow 数量:", count_by_column(challenge_rows, "expected_workflow"), sorted(ALLOWED_WORKFLOWS))

    hard_internal_dupes = count_duplicate_questions(hard_rows)
    challenge_internal_dupes = count_duplicate_questions(challenge_rows)
    print(f"hard_train_cases 内部重复 question 数量: {hard_internal_dupes}")
    print(f"challenge_eval_cases 内部重复 question 数量: {challenge_internal_dupes}")
    print(f"hard_train_cases 与 challenge_eval_cases 重复 question 数量: {len(hard_challenge_overlap)}")
    print(f"router_train.csv 与 challenge_eval_cases 重复 question 数量: {len(train_challenge_overlap)}")

    empty_counts = {
        "train_empty_question": count_empty(train_rows, "question"),
        "train_empty_workflow": count_empty(train_rows, "workflow"),
        "train_empty_tools": count_empty(train_rows, "tools"),
        "hard_empty_question": count_empty(hard_rows, "question"),
        "hard_empty_workflow": count_empty(hard_rows, "workflow"),
        "hard_empty_tools": count_empty(hard_rows, "tools"),
        "challenge_empty_question": count_empty(challenge_rows, "question"),
        "challenge_empty_expected_workflow": count_empty(challenge_rows, "expected_workflow"),
        "challenge_empty_expected_required_tools": count_empty(challenge_rows, "expected_required_tools"),
    }
    print("空值统计:")
    for key, value in empty_counts.items():
        print(f"  {key}: {value}")

    invalid_workflows = {
        "train": invalid_labels(train_rows, "workflow"),
        "hard": invalid_labels(hard_rows, "workflow"),
        "challenge": invalid_labels(challenge_rows, "expected_workflow"),
    }
    print("非法 workflow:")
    for key, value in invalid_workflows.items():
        print(f"  {key}: {sorted(value)}")

    found_old_labels = {
        "train": old_labels(train_rows, "workflow"),
        "hard": old_labels(hard_rows, "workflow"),
        "challenge": old_labels(challenge_rows, "expected_workflow"),
    }
    print("旧标签:")
    for key, value in found_old_labels.items():
        print(f"  {key}: {sorted(value)}")
        if value:
            print(f"WARN: {key} contains old labels {sorted(value)}")

    invalid_tool_values = {
        "train": invalid_tools(train_rows, "tools"),
        "hard": invalid_tools(hard_rows, "tools"),
        "challenge": invalid_tools(challenge_rows, "expected_required_tools"),
    }
    print("非法 tools:")
    for key, value in invalid_tool_values.items():
        print(f"  {key}: {sorted(value)}")

    boundary_failures = check_required_boundaries(train_rows)
    print("关键边界样本检查:")
    if boundary_failures:
        for failure in boundary_failures:
            print(f"  FAIL: {failure}")
    else:
        print("  OK")

    print("router_train.csv 每类随机预览:")
    preview_rows(train_rows, "workflow", ["question", "workflow", "tools"])
    print("router_challenge_eval_cases.csv 每类随机预览:")
    preview_rows(challenge_rows, "expected_workflow", ["question", "expected_workflow", "expected_required_tools"])

    if hard_internal_dupes:
        failures.append("hard_train_cases has duplicate questions")
    if challenge_internal_dupes:
        failures.append("challenge_eval_cases has duplicate questions")
    if hard_challenge_overlap:
        failures.append("hard_train_cases overlaps challenge_eval_cases")
    if train_challenge_overlap:
        failures.append("router_train.csv overlaps challenge_eval_cases")
    if any(empty_counts[key] for key in empty_counts if "question" in key):
        failures.append("empty question exists")
    if any(invalid_workflows.values()):
        failures.append("invalid workflow exists")
    if any(found_old_labels.values()):
        failures.append("old labels exist")
    if set(count_by_column(train_rows, "workflow")) & ALLOWED_WORKFLOWS != ALLOWED_WORKFLOWS:
        failures.append("router_train.csv does not contain all 7 workflows")
    if set(count_by_column(challenge_rows, "expected_workflow")) & ALLOWED_WORKFLOWS != ALLOWED_WORKFLOWS:
        failures.append("challenge_eval_cases does not contain all 7 expected workflows")

    if failures:
        print("DATA QUALITY CHECK FAILED")
        print("失败原因:")
        for failure in failures:
            print(f"  - {failure}")
    else:
        print("DATA QUALITY CHECK PASSED")


if __name__ == "__main__":
    main()
