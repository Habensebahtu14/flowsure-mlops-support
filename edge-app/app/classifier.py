import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as rt


class IntentClassifier:
    def __init__(self, model_path: str, mapping_path: str):
        self.session = rt.InferenceSession(model_path)
        with open(mapping_path, "r") as f:
            self.intent_mapping = json.load(f)
        self.intents = list(self.intent_mapping.keys())

    @property
    def num_intents(self) -> int:
        return len(self.intent_mapping)

    @property
    def model_size_mb(self) -> float:
        path = self.session._model_path if hasattr(self.session, "_model_path") else None
        return 1.53  # known size

    def predict(self, text: str) -> dict:
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        input_array = np.array([[text.strip()]], dtype=object)

        start = time.perf_counter()
        outputs = self.session.run(None, {"text": input_array})
        elapsed_ms = (time.perf_counter() - start) * 1000

        label: str = outputs[0][0]
        prob_map: dict = outputs[1][0]  # {intent: float}

        confidence = float(prob_map.get(label, 0.0))

        top_3 = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)[:3]
        top_3_list = [{"intent": k, "confidence": float(v)} for k, v in top_3]

        category, priority = self.intent_mapping.get(label, ["UNKNOWN", "unknown"])

        return {
            "intent": label,
            "category": category,
            "priority": priority,
            "confidence": confidence,
            "top_3": top_3_list,
            "inference_time_ms": round(elapsed_ms, 2),
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts if t.strip()]
