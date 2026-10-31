"""Recall-first threshold tuning and classification metrics."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, fbeta_score, precision_recall_fscore_support

NONE = "none"
DEFAULT_THRESHOLD = 0.5
GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)
BETA = 2.0  # F2 weights recall twice as much as precision


def tune_thresholds(y_true: Sequence[str], proba: np.ndarray, classes: list[str]) -> dict[str, float]:
    """Per risk class, the grid threshold maximising one-vs-rest F2 (ties → lower threshold, recall-first)."""
    y = np.asarray(y_true)
    out = {}
    for k, cls in enumerate(classes):
        if cls == NONE:
            continue
        target = y == cls
        if not target.any():
            out[cls] = DEFAULT_THRESHOLD
            continue
        best_t, best_f = DEFAULT_THRESHOLD, -1.0
        for t in GRID:
            f = fbeta_score(target, proba[:, k] >= t, beta=BETA, zero_division=0)
            if f > best_f:
                best_t, best_f = float(t), f
        out[cls] = best_t
    return out


def apply_thresholds(proba: np.ndarray, classes: list[str],
                     thresholds: dict[str, float]) -> list[list[tuple[str, float]]]:
    """Every risk class whose probability meets its threshold, per row, highest first."""
    return [sorted(((c, float(row[k])) for k, c in enumerate(classes)
                    if c in thresholds and row[k] >= thresholds[c]), key=lambda h: -h[1])
            for row in proba]


def predict_labels(proba: np.ndarray, classes: list[str], thresholds: dict[str, float]) -> list[str]:
    """Single label per row: the highest-probability passing class, else 'none'."""
    return [max(hits, key=lambda h: h[1])[0] if hits else NONE
            for hits in apply_thresholds(proba, classes, thresholds)]


def report(y_true: Sequence[str], y_pred: Sequence[str], labels: list[str]) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per = {l: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
           for i, l in enumerate(labels)}
    macro = {m: float(np.mean([per[l][m] for l in labels])) for m in ("precision", "recall", "f1")}
    risk = [l for l in labels if l != NONE]
    macro_risk = {m: float(np.mean([per[l][m] for l in risk])) for m in ("precision", "recall", "f1")} if risk else {}
    return {"per_label": per, "macro": macro, "macro_risk": macro_risk,
            "labels": list(labels), "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist()}
