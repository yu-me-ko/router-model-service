import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.evaluate_router_cases import evaluate_model_dir


MODEL_DIR = ROOT_DIR / "saved_models" / "router_distilbert"
ORDINARY_EVAL_PATH = ROOT_DIR / "data" / "router_eval_cases.csv"
CHALLENGE_EVAL_PATH = ROOT_DIR / "data" / "router_challenge_eval_cases.csv"


def report_passed(report: dict) -> bool:
    return (
        report["workflow_accuracy"] == 1.0
        and report["tool_hints_satisfaction_rate"] == 1.0
        and report["critical_error_count"] == 0
        and len(report["final_low_confidence_cases"]) == 0
    )


def print_summary(name: str, report: dict) -> None:
    print(f"{name}:")
    print(f"  total={report['total']}")
    print(f"  workflow_accuracy={report['workflow_accuracy']:.2%}")
    print(f"  tool_hints_satisfaction_rate={report['tool_hints_satisfaction_rate']:.2%}")
    print(f"  critical_error_count={report['critical_error_count']}")
    print(f"  rule_corrected_count={report['rule_corrected_count']}")
    print(f"  raw_low_confidence_count={len(report['raw_low_confidence_cases'])}")
    print(f"  final_low_confidence_count={len(report['final_low_confidence_cases'])}")


def check_fastapi_predictor_loads() -> bool:
    try:
        from app.predictor import router_predictor

        result = router_predictor.predict("给我讲个笑话")
        return result.get("workflow") == "DIRECT_AI" and "DIRECT_LLM" in result.get("toolHints", [])
    except Exception as exc:
        print(f"FastAPI predictor load failed: {exc}")
        return False


def main() -> int:
    failures = []

    if not MODEL_DIR.exists():
        failures.append(f"model directory missing: {MODEL_DIR}")
        print("ROUTER RELEASE CHECK FAILED")
        for failure in failures:
            print(f"  {failure}")
        return 1

    ordinary_report = evaluate_model_dir(MODEL_DIR, ORDINARY_EVAL_PATH, print_predictions=False)
    challenge_report = evaluate_model_dir(MODEL_DIR, CHALLENGE_EVAL_PATH, print_predictions=False)

    print_summary("ordinary eval cases", ordinary_report)
    print_summary("challenge eval cases", challenge_report)

    if not report_passed(ordinary_report):
        failures.append("ordinary eval cases did not meet release criteria")
    if not report_passed(challenge_report):
        failures.append("challenge eval cases did not meet release criteria")
    if not check_fastapi_predictor_loads():
        failures.append("FastAPI predictor could not load or predict expected DIRECT_AI case")

    if failures:
        print("ROUTER RELEASE CHECK FAILED")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("ROUTER RELEASE CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
