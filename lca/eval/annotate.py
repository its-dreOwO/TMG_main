"""Sentence-level annotation templates and inter-annotator consensus (Cohen's Kappa).

  python -m lca.eval.annotate template data/contracts/hd1.docx data/annotations/hd1_khanh.csv
  python -m lca.eval.annotate consensus data/annotations/hd1_*.csv --out data/ground_truth/hd1.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from lca.reader import read_document
from lca.segmenter import segment
from lca.types import RiskLabel

FIELDS = ["seg_id", "location", "text", "label"]
VALID = {l.value for l in RiskLabel}


def _read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write(path: Path, rows: list[dict], fields: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def make_template(path: Path, out: Path) -> None:
    """One row per sentence segment with a blank label column for an annotator to fill."""
    segs = segment(read_document(Path(path)), "sentence")
    _write(out, [{"seg_id": s.seg_id, "location": s.location, "text": s.text, "label": ""} for s in segs], FIELDS)


def _labels(path: Path) -> list[str]:
    out = []
    for row in _read(path):
        label = row["label"].strip()
        if not label:
            raise ValueError(f"{Path(path).name}: seg_id {row['seg_id']} has no label")
        if label not in VALID:
            raise ValueError(f"{Path(path).name}: seg_id {row['seg_id']} has unknown label '{label}' "
                             f"(use one of {sorted(VALID)})")
        out.append(label)
    return out


def _kappa(a: list[str], b: list[str]) -> float:
    if a == b:
        return 1.0
    return float(cohen_kappa_score(a, b))


def consensus(paths: list[Path], resolved_out: Path) -> dict:
    """Kappa for every annotator pair; writes majority labels (blank = disagreement to resolve)."""
    paths = [Path(p) for p in paths]
    labels = {p.name: _labels(p) for p in paths}
    lengths = {len(v) for v in labels.values()}
    if len(lengths) != 1:
        raise ValueError(f"annotation files have different numbers of rows: { {k: len(v) for k, v in labels.items()} }")
    kappa = {f"{a}|{b}": _kappa(labels[a], labels[b]) for a, b in combinations(labels, 2)}
    base = _read(paths[0])
    rows, disagreements = [], 0
    for i, row in enumerate(base):
        votes = Counter(labels[p.name][i] for p in paths)
        (top, n), *rest = votes.most_common()
        agreed = len(votes) == 1
        majority = top if (n > len(paths) / 2) else ""
        disagreements += not agreed
        rows.append({"seg_id": row["seg_id"], "location": row["location"], "text": row["text"], "label": majority}
                    | {p.stem: labels[p.name][i] for p in paths})
    _write(resolved_out, rows, FIELDS + [p.stem for p in paths])
    result = {"kappa": kappa, "n_disagreements": disagreements, "n_segments": len(rows)}
    Path(resolved_out).with_suffix(".kappa.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template")
    t.add_argument("contract", type=Path)
    t.add_argument("out", type=Path)
    c = sub.add_parser("consensus")
    c.add_argument("annotations", type=Path, nargs="+")
    c.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "template":
        make_template(args.contract, args.out)
    else:
        print(json.dumps(consensus(args.annotations, args.out), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
