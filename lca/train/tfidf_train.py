"""Train TF-IDF + LogisticRegression / calibrated LinearSVC on data/processed.

Run: python -m lca.train.tfidf_train
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from lca.eval.metrics import predict_labels, report, tune_thresholds

SEED = 42
MAX_FEATURES = 500_000


def _classifier(kind: str):
    if kind == "lr":
        return LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    if kind == "svm":
        return CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=SEED), cv=3)
    raise ValueError(f"unknown model kind {kind!r}")


def fit_model(kind: Literal["lr", "svm"], train: pd.DataFrame, val: pd.DataFrame, min_df: int = 2) -> dict:
    """Fit on train, tune recall-first thresholds on val; returns the saved bundle."""
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True,
                                  max_features=MAX_FEATURES)),
        ("clf", _classifier(kind)),
    ])
    pipe.fit(train.text.tolist(), train.label.tolist())
    classes = [str(c) for c in pipe.classes_]
    thresholds = tune_thresholds(val.label.tolist(), pipe.predict_proba(val.text.tolist()), classes)
    return {"pipeline": pipe, "classes": classes, "thresholds": thresholds}


def evaluate(bundle: dict, test: pd.DataFrame) -> dict:
    proba = bundle["pipeline"].predict_proba(test.text.tolist())
    pred = predict_labels(proba, bundle["classes"], bundle["thresholds"])
    return report(test.label.tolist(), pred, bundle["classes"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--models", type=Path, default=Path("artifacts/models"))
    ap.add_argument("--report", type=Path, default=Path("reports/tfidf_metrics.json"))
    args = ap.parse_args()
    train, val, test = (pd.read_parquet(args.data / f"{s}.parquet") for s in ("train", "val", "test"))
    args.models.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    for kind in ("lr", "svm"):
        bundle = fit_model(kind, train, val)
        joblib.dump(bundle, args.models / f"tfidf_{kind}.joblib")
        results[kind] = {"thresholds": bundle["thresholds"], "test": evaluate(bundle, test)}
        print(kind, "test macro F1 (risk classes):", round(results[kind]["test"]["macro_risk"]["f1"], 4))
    best = max(results, key=lambda k: results[k]["test"]["macro_risk"]["f1"])
    results["best"] = best
    joblib.dump(joblib.load(args.models / f"tfidf_{best}.joblib"), args.models / "tfidf_best.joblib")
    args.report.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
