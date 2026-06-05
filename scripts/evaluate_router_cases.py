import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.routing_rules import infer_tool_hints, post_process_route


DEFAULT_MODEL_DIR = ROOT_DIR / "saved_models" / "router_distilbert"
DEFAULT_EVAL_CASES_PATH = ROOT_DIR / "data" / "router_eval_cases.csv"
MAX_LENGTH = 64

LEGACY_LABEL_MAP = {
    "NORMAL_KNOWLEDGE_QA": "KNOWLEDGE_QA",
    "DIRECT_AI_FALLBACK": "DIRECT_AI",
}

CRITICAL_EXPECTATIONS = {
    "给我讲个笑话": "DIRECT_AI",
    "删除这个对话": "SYSTEM_ACTION",
    "把这个文件加入全局知识库": "SYSTEM_ACTION",
    "我上传的课程表里周三有什么课": "FILE_QA",
    "明天食堂开门吗": "KNOWLEDGE_QA",
    "明天有什么作业要交吗": "AGENT_TASK",
}


def normalize_workflow(label: str) -> str:
    return LEGACY_LABEL_MAP.get(label, label)


def load_label_map(model_dir: Path) -> dict[str, int]:
    model_label_map = model_dir / "router_label_map.json"
    default_label_map = ROOT_DIR / "data" / "router_label_map.json"
    label_map_path = model_label_map if model_label_map.exists() else default_label_map

    with open(label_map_path, "r", encoding="utf-8") as f:
        return json.load(f)


class RouterCaseEvaluator:
    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.eval()

        label_map = load_label_map(self.model_dir)
        self.id_to_label = {v: normalize_workflow(k) for k, v in label_map.items()}

    def predict(self, question: str) -> dict:
        inputs = self.tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            confidence, pred_id = torch.max(probs, dim=-1)

        raw_workflow = normalize_workflow(self.id_to_label[pred_id.item()])
        raw_confidence = confidence.item()
        route_result = post_process_route(question, raw_workflow, raw_confidence)
        workflow = route_result.workflow
        confidence_value = max(raw_confidence, 0.99) if route_result.corrected_by_rule else raw_confidence

        return {
            "question": question,
            "workflow": workflow,
            "confidence": round(confidence_value, 4),
            "toolHints": infer_tool_hints(question, workflow),
            "correctedByRule": route_result.corrected_by_rule,
            "ruleName": route_result.rule_name,
            "rawWorkflow": route_result.raw_workflow,
            "rawConfidence": round(raw_confidence, 4),
        }


def read_cases(path: Path) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def parse_required_tools(value: str) -> set[str]:
    return {tool.strip() for tool in (value or "").split("|") if tool.strip()}


def evaluate_model_dir(
    model_dir: Path = DEFAULT_MODEL_DIR,
    cases_path: Path = DEFAULT_EVAL_CASES_PATH,
    print_predictions: bool = True,
) -> dict:
    if not cases_path.exists():
        raise FileNotFoundError(f"Eval cases not found: {cases_path}")

    evaluator = RouterCaseEvaluator(model_dir)
    cases = read_cases(cases_path)

    workflow_errors = []
    tool_missing_errors = []
    raw_low_confidence_cases = []
    final_low_confidence_cases = []
    rule_corrected_samples = []
    critical_errors = []
    workflow_correct_count = 0
    tool_satisfied_count = 0

    for case in cases:
        question = case["question"].strip()
        expected_workflow = case["expected_workflow"].strip()
        required_tools = parse_required_tools(case.get("expected_required_tools", ""))
        result = evaluator.predict(question)
        actual_tools = set(result["toolHints"])

        workflow_ok = result["workflow"] == expected_workflow
        tools_ok = required_tools.issubset(actual_tools)

        if workflow_ok:
            workflow_correct_count += 1
        else:
            workflow_errors.append({"question": question, "expected": expected_workflow, "actual": result["workflow"]})

        if tools_ok:
            tool_satisfied_count += 1
        else:
            tool_missing_errors.append(
                {
                    "question": question,
                    "missing": sorted(required_tools - actual_tools),
                    "actual": result["toolHints"],
                }
            )

        if result["rawConfidence"] < 0.65:
            raw_low_confidence_cases.append(result)

        if result["correctedByRule"]:
            rule_corrected_samples.append(result)

        if not result["correctedByRule"] and result["confidence"] < 0.65:
            final_low_confidence_cases.append(result)

        critical_expected = CRITICAL_EXPECTATIONS.get(question)
        if critical_expected and result["workflow"] != critical_expected:
            critical_errors.append({"question": question, "expected": critical_expected, "actual": result["workflow"]})

        if print_predictions:
            print(
                f"{question} -> workflow={result['workflow']} "
                f"confidence={result['confidence']} toolHints={result['toolHints']} "
                f"correctedByRule={result['correctedByRule']} ruleName={result['ruleName']} "
                f"rawWorkflow={result['rawWorkflow']} rawConfidence={result['rawConfidence']}"
            )

    total = len(cases)
    workflow_accuracy = workflow_correct_count / total if total else 0.0
    tool_satisfaction_rate = tool_satisfied_count / total if total else 0.0

    return {
        "total": total,
        "workflow_correct_count": workflow_correct_count,
        "workflow_accuracy": workflow_accuracy,
        "tool_hints_satisfied_count": tool_satisfied_count,
        "tool_hints_satisfaction_rate": tool_satisfaction_rate,
        "workflow_errors": workflow_errors,
        "tool_missing_errors": tool_missing_errors,
        "low_confidence_cases": final_low_confidence_cases,
        "raw_low_confidence_cases": raw_low_confidence_cases,
        "final_low_confidence_cases": final_low_confidence_cases,
        "rule_corrected_samples": rule_corrected_samples,
        "rule_corrected_count": len(rule_corrected_samples),
        "critical_errors": critical_errors,
        "critical_error_count": len(critical_errors),
    }


def print_report(report: dict) -> None:
    print("================================")
    print(f"总测试数: {report['total']}")
    print(f"workflow 正确数: {report['workflow_correct_count']}")
    print(f"workflow 准确率: {report['workflow_accuracy']:.2%}")
    print(f"toolHints 满足数: {report['tool_hints_satisfied_count']}")
    print(f"toolHints 满足率: {report['tool_hints_satisfaction_rate']:.2%}")
    print(f"rule corrected count: {report['rule_corrected_count']}")
    print("workflow 错误样本:")
    for item in report["workflow_errors"]:
        print(f"  {item['question']} expected={item['expected']} actual={item['actual']}")
    print("toolHints 缺失样本:")
    for item in report["tool_missing_errors"]:
        print(f"  {item['question']} missing={item['missing']} actual={item['actual']}")
    print("rule corrected samples:")
    for item in report["rule_corrected_samples"]:
        print(
            f"  {item['question']} ruleName={item['ruleName']} "
            f"rawWorkflow={item['rawWorkflow']} rawConfidence={item['rawConfidence']} "
            f"workflow={item['workflow']} confidence={item['confidence']}"
        )
    print("raw low confidence samples:")
    for item in report["raw_low_confidence_cases"]:
        print(
            f"  {item['question']} rawWorkflow={item['rawWorkflow']} "
            f"rawConfidence={item['rawConfidence']} workflow={item['workflow']} "
            f"confidence={item['confidence']} correctedByRule={item['correctedByRule']}"
        )
    print("final low confidence samples:")
    for item in report["final_low_confidence_cases"]:
        print(f"  {item['question']} workflow={item['workflow']} confidence={item['confidence']}")
    print("critical errors:")
    for item in report["critical_errors"]:
        print(f"  {item['question']} expected={item['expected']} actual={item['actual']}")
    print("================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate router model on fixed regression cases.")
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR), help="Model directory to evaluate.")
    parser.add_argument("--cases", default=None, help="Evaluation cases CSV path.")
    parser.add_argument("--eval_path", default=None, help="Evaluation cases CSV path.")
    args = parser.parse_args()

    cases_path = args.eval_path or args.cases or str(DEFAULT_EVAL_CASES_PATH)
    report = evaluate_model_dir(Path(args.model_dir), Path(cases_path), print_predictions=True)
    print_report(report)


if __name__ == "__main__":
    main()
