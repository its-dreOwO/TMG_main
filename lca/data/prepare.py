"""Build leak-free train/val/test parquet files from CUAD and LEDGAR.

Run: python -m lca.data.prepare [--raw data] [--out data/processed]
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from lca.data.mapping import CUAD_MAP, map_ledgar_label
from lca.segmenter import normalize
from lca.types import RiskLabel

SEED = 42
NONE_RATIO = 3
LEDGAR_NEGATIVES = 200_000
MIN_POSITIVE_CHARS = 20
NEG_CHARS = (40, 2000)
CUAD_JSON = "cuad/CUAD_v1/CUAD_v1.json"
LEDGAR_JSONL = "ledgar/LEDGAR_2016-2019_clean.jsonl"
COLUMNS = ["text", "label", "source", "group"]

_QUESTION_RE = re.compile(r'related to "(.+?)"')
# CUAD has no indemnity/confidentiality categories: never teach such text as "none".
_UNLABELLED_RISK_RE = re.compile(r"indemnif|hold harmless|confidential|non-disclosure|penalt", re.I)
_SENT_BOUNDARY_RE = re.compile(r"(?<=[.;])\s+(?=[A-Z0-9(\"“])|\n\s*\n")


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in _SENT_BOUNDARY_RE.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return spans


def _cuad_rows(path: Path) -> list[dict]:
    rows = []
    for contract in json.loads(path.read_text())["data"]:
        group = f"cuad:{contract['title']}"
        for para in contract["paragraphs"]:
            ctx = para["context"]
            risky: list[tuple[int, int]] = []
            for qa in para["qas"]:
                m = _QUESTION_RE.search(qa["question"])
                label = CUAD_MAP.get(m.group(1)) if m else None
                if label is None or qa.get("is_impossible"):
                    continue
                for ans in qa["answers"]:
                    s = ans["answer_start"]
                    risky.append((s, s + len(ans["text"])))
                    text = normalize(ans["text"])
                    if len(text) >= MIN_POSITIVE_CHARS:
                        rows.append({"text": text, "label": label.value, "source": "cuad", "group": group})
            for s, e in _sentence_spans(ctx):
                if any(s < re_ and rs < e for rs, re_ in risky):
                    continue
                text = normalize(ctx[s:e])
                if NEG_CHARS[0] <= len(text) <= NEG_CHARS[1] and not _UNLABELLED_RISK_RE.search(text):
                    rows.append({"text": text, "label": RiskLabel.NONE.value, "source": "cuad", "group": group})
    return rows


def _ledgar_rows(path: Path, n_negatives: int, stats: dict) -> list[dict]:
    rng = random.Random(SEED)
    positives, reservoir, seen_neg = [], [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            labels = rec["label"] if isinstance(rec["label"], list) else [rec["label"]]
            mapped = [map_ledgar_label(l) for l in labels]
            if any(m is None for m in mapped):
                stats["dropped_uncertain"] += 1
                continue
            risks = {m for m in mapped if m is not RiskLabel.NONE}
            if len(risks) > 1:
                stats["dropped_multi_risk"] += 1
                continue
            group = "ledgar:" + "/".join(rec["source"].split("/")[:3])
            if risks:
                positives.append({"text": normalize(rec["provision"]), "label": risks.pop().value,
                                  "source": "ledgar", "group": group})
                continue
            seen_neg += 1
            row = (rec["provision"], group)
            if len(reservoir) < n_negatives:
                reservoir.append(row)
            else:
                j = rng.randrange(seen_neg)
                if j < n_negatives:
                    reservoir[j] = row
    negatives = [{"text": normalize(t), "label": RiskLabel.NONE.value, "source": "ledgar", "group": g}
                 for t, g in reservoir]
    return positives + negatives


def _group_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, rest = next(outer.split(df, groups=df.group))
    train, rest_df = df.iloc[tr], df.iloc[rest]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    va, te = next(inner.split(rest_df, groups=rest_df.group))
    return train, rest_df.iloc[va], rest_df.iloc[te]


def prepare(raw_dir: Path, out_dir: Path, ledgar_negatives: int = LEDGAR_NEGATIVES) -> dict:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    stats = {"dropped_uncertain": 0, "dropped_multi_risk": 0}
    rows = _cuad_rows(raw_dir / CUAD_JSON) + _ledgar_rows(raw_dir / LEDGAR_JSONL, ledgar_negatives, stats)
    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df[df.text.str.len() > 0]

    is_none = df.label == RiskLabel.NONE.value
    cap = NONE_RATIO * int((~is_none).sum())
    stats["dropped_none_cap"] = max(0, int(is_none.sum()) - cap)
    if stats["dropped_none_cap"]:
        df = pd.concat([df[~is_none], df[is_none].sample(n=cap, random_state=SEED)])

    train, val, test = _group_split(df.reset_index(drop=True))
    before = len(val) + len(test)
    val = val[~val.text.isin(set(train.text))]
    test = test[~test.text.isin(set(train.text) | set(val.text))]
    stats["dropped_duplicate"] = before - len(val) - len(test)

    out_dir.mkdir(parents=True, exist_ok=True)
    stats["splits"] = {}
    for name, frame in (("train", train), ("val", val), ("test", test)):
        frame = frame.sample(frac=1, random_state=SEED).reset_index(drop=True)
        frame.to_parquet(out_dir / f"{name}.parquet", index=False)
        stats["splits"][name] = frame.label.value_counts().to_dict()
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    args = ap.parse_args()
    print(json.dumps(prepare(args.raw, args.out), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
