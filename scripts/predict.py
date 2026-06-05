import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.routing_rules import infer_tool_hints, post_process_route


MODEL_DIR = ROOT_DIR / "saved_models" / "router_distilbert"
LABEL_MAP_PATH = MODEL_DIR / "router_label_map.json"
MAX_LENGTH = 64

LEGACY_LABEL_MAP = {
    "NORMAL_KNOWLEDGE_QA": "KNOWLEDGE_QA",
    "DIRECT_AI_FALLBACK": "DIRECT_AI",
}


def normalize_workflow(label: str) -> str:
    return LEGACY_LABEL_MAP.get(label, label)


def load_label_names():
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    return {v: normalize_workflow(k) for k, v in label_map.items()}


def predict(question: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    id_to_label = load_label_names()

    inputs = tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        confidence, pred_id = torch.max(probs, dim=-1)

    raw_workflow = normalize_workflow(id_to_label[pred_id.item()])
    raw_confidence = confidence.item()
    route_result = post_process_route(question, raw_workflow, raw_confidence)
    workflow = route_result.workflow
    final_confidence = max(raw_confidence, 0.99) if route_result.corrected_by_rule else raw_confidence

    return {
        "question": question,
        "route": workflow,
        "workflow": workflow,
        "confidence": round(final_confidence, 4),
        "toolHints": infer_tool_hints(question, workflow),
        "correctedByRule": route_result.corrected_by_rule,
        "ruleName": route_result.rule_name,
        "rawWorkflow": route_result.raw_workflow,
        "rawConfidence": round(raw_confidence, 4),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python scripts\\predict.py "你的问题"')
        sys.exit(1)

    result = predict(sys.argv[1])
    print(json.dumps(result, ensure_ascii=False, indent=2))
