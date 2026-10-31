"""Local inference for a fine-tuned sequence classifier, with sliding windows for long text."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lca.eval.metrics import apply_thresholds
from lca.types import Detection, RiskLabel


class TransformerDetector:
    def __init__(self, model_dir: Path, name: str, device: str | None = None, max_length: int = 256,
                 stride: int = 64, batch_size: int = 8):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "thresholds.json").read_text())
        self._setup(AutoModelForSequenceClassification.from_pretrained(model_dir),
                    AutoTokenizer.from_pretrained(model_dir), meta["classes"], meta["thresholds"],
                    name, device, max_length, stride, batch_size)

    @classmethod
    def from_components(cls, model, tokenizer, classes: list[str], thresholds: dict[str, float], name: str,
                        device: str | None = None, max_length: int = 256, stride: int = 64,
                        batch_size: int = 8) -> "TransformerDetector":
        self = cls.__new__(cls)
        self._setup(model, tokenizer, classes, thresholds, name, device, max_length, stride, batch_size)
        return self

    def _setup(self, model, tokenizer, classes, thresholds, name, device, max_length, stride, batch_size):
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval() if hasattr(model, "to") else model
        self.tokenizer = tokenizer
        self.classes = list(classes)
        self.thresholds = dict(thresholds)
        self.name = name
        self.max_length, self.stride, self.batch_size = max_length, stride, batch_size

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Per text, the per-class maximum probability over all of its windows."""
        import torch

        out = np.zeros((len(texts), len(self.classes)), dtype=np.float32)
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            enc = self.tokenizer(batch, truncation=True, max_length=self.max_length, stride=self.stride,
                                 return_overflowing_tokens=True, padding=True, return_tensors="pt")
            owner = enc.pop("overflow_to_sample_mapping").tolist()
            inputs = {k: v.to(self.device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
            with torch.inference_mode():
                probs = torch.softmax(self.model(**inputs).logits.float(), dim=-1).cpu().numpy()
            for row, i in zip(probs, owner):
                out[start + i] = np.maximum(out[start + i], row)
        return out

    def predict(self, texts: list[str]) -> list[list[Detection]]:
        if not texts:
            return []
        hits = apply_thresholds(self.predict_proba(texts), self.classes, self.thresholds)
        return [[Detection(self.name, RiskLabel(c), s) for c, s in h] for h in hits]
