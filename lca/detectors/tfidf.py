"""TF-IDF + linear classifier detector (loads a bundle from lca.train.tfidf_train)."""
from __future__ import annotations

from pathlib import Path

import joblib

from lca.eval.metrics import apply_thresholds
from lca.types import Detection, RiskLabel


class TfidfDetector:
    name = "tfidf"

    def __init__(self, path: Path):
        bundle = joblib.load(path)
        self.pipeline = bundle["pipeline"]
        self.classes: list[str] = list(bundle["classes"])
        self.thresholds: dict[str, float] = bundle["thresholds"]

    def predict(self, texts: list[str]) -> list[list[Detection]]:
        if not texts:
            return []
        proba = self.pipeline.predict_proba(texts)
        return [[Detection(self.name, RiskLabel(c), s) for c, s in hits]
                for hits in apply_thresholds(proba, self.classes, self.thresholds)]
