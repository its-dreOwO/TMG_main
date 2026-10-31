"""Semantic similarity to prototype risk clauses (multilingual sentence embeddings).

Run `python -m lca.detectors.sts` to tune per-group thresholds on the validation split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from lca.eval.metrics import tune_thresholds
from lca.types import Detection, RiskLabel

STS_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PROTOTYPES = Path(__file__).resolve().parent.parent / "resources" / "prototypes.yaml"
SEED = 42


def load_prototypes(path: Path = PROTOTYPES) -> dict[str, list[str]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {str(k): [str(t) for t in v] for k, v in data.items()}


def _load_encoder(model: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model)


class StsDetector:
    name = "sts"

    def __init__(self, prototypes: dict[str, list[str]], thresholds: dict[str, float],
                 model: str = STS_MODEL, encoder=None):
        self.encoder = encoder if encoder is not None else _load_encoder(model)
        self.groups = list(prototypes)
        self.thresholds = dict(thresholds)
        self._protos = [self._encode(prototypes[g]) for g in self.groups]

    def _encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.encoder.encode(texts, normalize_embeddings=True))

    def scores(self, texts: list[str]) -> np.ndarray:
        """(n_texts, n_groups) max cosine similarity to each group's prototypes."""
        emb = self._encode(texts)
        return np.stack([(emb @ p.T).max(axis=1) for p in self._protos], axis=1)

    def predict(self, texts: list[str]) -> list[list[Detection]]:
        if not texts:
            return []
        out = []
        for row in self.scores(texts):
            hits = [Detection(self.name, RiskLabel(g), float(s)) for g, s in zip(self.groups, row)
                    if s >= self.thresholds.get(g, 1.01)]
            out.append(sorted(hits, key=lambda d: -d.score))
        return out


def tune_sts(encoder, prototypes: dict[str, list[str]], val_frame: pd.DataFrame) -> dict[str, float]:
    det = StsDetector(prototypes, {}, encoder=encoder)
    return tune_thresholds(val_frame.label.tolist(), det.scores(val_frame.text.tolist()), det.groups)


def _sample_val(val: pd.DataFrame, per_class: int = 600, n_none: int = 3000) -> pd.DataFrame:
    parts = [g.sample(min(len(g), n_none if label == "none" else per_class), random_state=SEED)
             for label, g in val.groupby("label")]
    return pd.concat(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val", type=Path, default=Path("data/processed/val.parquet"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/models/sts_thresholds.json"))
    args = ap.parse_args()
    val = _sample_val(pd.read_parquet(args.val))
    thresholds = tune_sts(_load_encoder(STS_MODEL), load_prototypes(), val)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(thresholds, indent=2))
    print(json.dumps(thresholds, indent=2))


if __name__ == "__main__":
    main()
