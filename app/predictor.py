import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.routing_rules import infer_tool_hints, post_process_route


MODEL_DIR = Path(__file__).resolve().parent.parent / "saved_models" / "router_distilbert"
LABEL_MAP_PATH = MODEL_DIR / "router_label_map.json"
MAX_LENGTH = 64
EMPTY_WORKFLOW = "UNKNOWN"

LEGACY_LABEL_MAP = {
    "NORMAL_KNOWLEDGE_QA": "KNOWLEDGE_QA",
    "DIRECT_AI_FALLBACK": "DIRECT_AI",
}


def normalize_workflow(label: str) -> str:
    return LEGACY_LABEL_MAP.get(label, label)


class RouterPredictor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        self.model.eval()

        with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            label_map = json.load(f)

        self.id_to_label = {v: normalize_workflow(k) for k, v in label_map.items()}
        print("Router model loaded")

    def predict(self, question: str):
        if question is None or not question.strip():
            return {
                "question": question,
                "route": EMPTY_WORKFLOW,
                "workflow": EMPTY_WORKFLOW,
                "confidence": 0.0,
                "toolHints": infer_tool_hints(question, EMPTY_WORKFLOW),
                "correctedByRule": True,
                "ruleName": "UNKNOWN_EMPTY_RULE",
                "rawWorkflow": EMPTY_WORKFLOW,
                "rawConfidence": 0.0,
            }

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


router_predictor = RouterPredictor()
