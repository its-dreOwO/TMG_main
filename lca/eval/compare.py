"""Compare detection methods on the labelled sample contracts and answer RQ1–RQ3.

  python -m lca.eval.compare --contracts data/contracts --truth data/ground_truth --out reports
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from lca.eval.metrics import report
from lca.pipeline import DEFAULT_DETECTORS, DETECTOR_NAMES, analyze, build_detectors
from lca.reader import read_document
from lca.segmenter import segment
from lca.types import AnalysisContext, Finding, RiskLabel, Segment

LABELS = [l.value for l in RiskLabel]
MODES = ("sentence", "clause")
ROOT = Path(__file__).resolve().parents[2]
TESTSET_METRICS = {
    "tfidf": ROOT / "reports" / "tfidf_metrics.json",
    "legal-bert": ROOT / "artifacts" / "models" / "legal-bert" / "final" / "metrics.json",
    "xlmr": ROOT / "artifacts" / "models" / "xlmr" / "final" / "metrics.json",
}


def _overlap(a: Segment, b: Segment) -> bool:
    return any(x[0] == y[0] and x[1] < y[2] and y[1] < x[2] for x in a.spans for y in b.spans)


def sentence_predictions(sentences: list[Segment], findings: list[Finding]) -> list[str]:
    """Project findings of any granularity onto sentences: overlapping, highest-level finding wins."""
    out = []
    for sent in sentences:
        hits = [f for f in findings if _overlap(sent, f.segment)]
        if not hits:
            out.append(RiskLabel.NONE.value)
            continue
        best = max(hits, key=lambda f: (f.level, sum(d.score for d in f.detections)))
        out.append(best.label.value)
    return out


def _contracts(contracts_dir: Path) -> list[Path]:
    return sorted(p for p in Path(contracts_dir).iterdir() if p.suffix.lower() in (".docx", ".pdf"))


def _truth(contract: Path, truth_dir: Path, sentences: list[Segment]) -> list[str]:
    path = Path(truth_dir) / f"{contract.stem}.csv"
    if not path.exists():
        raise ValueError(f"{contract.stem}: missing ground truth {path}")
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if [r["text"] for r in rows] != [s.text for s in sentences]:
        raise ValueError(f"{contract.stem}: ground truth rows do not match the current sentence segmentation "
                         f"({len(rows)} rows vs {len(sentences)} sentences); regenerate the template")
    labels = [r["label"].strip() for r in rows]
    bad = [r["seg_id"] for r, l in zip(rows, labels) if l not in LABELS]
    if bad:
        raise ValueError(f"{contract.stem}: unresolved or invalid labels at seg_id {bad[:10]}")
    return labels


def _table(results: dict[str, dict], key: str = "macro_risk") -> list[str]:
    lines = ["| Method | Precision | Recall | F1 |", "|---|---|---|---|"]
    for name, r in results.items():
        m = r[key]
        lines.append(f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |")
    return lines


def _per_label_table(r: dict) -> list[str]:
    lines = ["| Label | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for label, m in r["per_label"].items():
        lines.append(f"| {label} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |")
    return lines


def _plot_confusion(r: dict, title: str, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [RiskLabel(l).vi for l in r["labels"]]
    cm = r["confusion"]
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=150)
    ax.imshow(cm, cmap="Blues")
    peak = max(max(row) for row in cm) or 1
    for i, row in enumerate(cm):
        for j, v in enumerate(row):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                    color="white" if v > peak * 0.6 else "#1f2937")
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("Dự đoán (predicted)")
    ax.set_ylabel("Đáp án (ground truth)")
    ax.set_title(title, fontsize=10, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def compare(contracts_dir: Path, ground_truth_dir: Path, out: Path, detectors: dict | None = None,
            combined: list[str] | None = None) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    detectors = detectors if detectors is not None else _available_detectors()
    combo = [n for n in (combined or DEFAULT_DETECTORS) if n in detectors]
    methods = {name: [det] for name, det in detectors.items()} | {"combined": [detectors[n] for n in combo]}
    ctx = AnalysisContext()

    truth: dict[Path, tuple[list[Segment], list[str]]] = {}
    for contract in _contracts(contracts_dir):
        sentences = segment(read_document(contract), "sentence")
        truth[contract] = (sentences, _truth(contract, ground_truth_dir, sentences))
    if not truth:
        raise ValueError(f"no .docx/.pdf contracts found in {contracts_dir}")

    preds: dict[str, dict[str, dict[Path, list[str]]]] = {m: {} for m in MODES}
    for mode in MODES:
        for name, dets in methods.items():
            preds[mode][name] = {c: sentence_predictions(sents, analyze(c, ctx, dets, mode).findings)
                                 for c, (sents, _) in truth.items()}

    y_true = [l for _, labels in truth.values() for l in labels]
    results = {mode: {name: report(y_true, [l for c in truth for l in preds[mode][name][c]], LABELS)
                      for name in methods} for mode in MODES}

    vi_idx = {c: [i for i, s in enumerate(sents) if s.lang == "vi"] for c, (sents, _) in truth.items()}
    vi_true = [truth[c][1][i] for c in truth for i in vi_idx[c]]
    routes = {}
    for name in ("xlmr", "legal-bert+translate"):
        if name in methods and vi_true:
            routes[name] = report(vi_true, [preds["sentence"][name][c][i] for c in truth for i in vi_idx[c]], LABELS)

    _write_errors(out / "error_analysis.csv", truth, preds["sentence"]["combined"])
    for mode in MODES:
        _plot_confusion(results[mode]["combined"], f"Kết hợp ({', '.join(combo)}) – {mode}",
                        out / f"confusion_combined_{mode}.png")

    singles = {n: r for n, r in results["sentence"].items() if n != "combined"}
    best_single = max(singles, key=lambda n: singles[n]["macro_risk"]["f1"])
    rq = {
        "rq1_regex": results["sentence"].get("regex", {}).get("macro_risk"),
        "rq2": {mode: {n: results[mode][n]["macro_risk"]["f1"] for n in methods} for mode in MODES},
        "rq3": {"best_single": best_single, "best_single_macro": singles[best_single]["macro_risk"],
                "combined_macro": results["sentence"]["combined"]["macro_risk"]},
        "routes": {n: r["macro_risk"] for n, r in routes.items()},
    }
    kappa = {p.name: json.loads(p.read_text()) for p in sorted(Path(ground_truth_dir).glob("*.kappa.json"))}
    testset = {n: json.loads(p.read_text()) for n, p in TESTSET_METRICS.items() if p.exists()}
    (out / "results.md").write_text(_markdown(results, routes, rq, combo, kappa, testset, truth), encoding="utf-8")
    summary = {"methods": list(methods), "modes": results, "rq": rq, "combined": combo}
    (out / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def _write_errors(path: Path, truth: dict, preds: dict[Path, list[str]]) -> None:
    rows = []
    for contract, (sents, labels) in truth.items():
        for s, t, p in zip(sents, labels, preds[contract]):
            if t == p:
                continue
            kind = "FP" if t == "none" else "FN" if p == "none" else "WRONG_LABEL"
            rows.append({"contract": contract.name, "seg_id": s.seg_id, "location": s.location,
                         "text": s.text, "truth": t, "predicted": p, "error": kind})
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["contract", "seg_id", "location", "text", "truth", "predicted", "error"])
        w.writeheader()
        w.writerows(rows)


def _markdown(results, routes, rq, combo, kappa, testset, truth) -> str:
    n_sent = sum(len(v[1]) for v in truth.values())
    n_risk = sum(l != "none" for v in truth.values() for l in v[1])
    L = ["# Legal Clause Analyzer — evaluation results", ""]
    L += ["## 1. Public test set (CUAD + LEDGAR, English)", ""]
    if testset:
        L += ["| Model | Precision | Recall | F1 (macro over risk groups) |", "|---|---|---|---|"]
        for name, data in testset.items():
            m = data[data["best"]]["test"]["macro_risk"] if name == "tfidf" else data["macro_risk"]
            label = f"tfidf ({data['best']})" if name == "tfidf" else name
            L.append(f"| {label} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |")
    else:
        L.append("No trained-model metrics found.")
    L += ["", f"## 2. Sample contracts ({len(truth)} contracts, {n_sent} sentences, {n_risk} risky)", ""]
    L += ["This is a **case study**: with so few risky sentences the numbers are not statistically "
          "significant and should be read as indicative only.", ""]
    for mode in MODES:
        L += [f"### Mode: {mode}", ""] + _table(results[mode]) + [""]
    L += [f"Combined = {' + '.join(combo)} (flag if any method fires).", ""]
    L += ["### Per-group results, combined, sentence mode", ""] + _per_label_table(results["sentence"]["combined"])
    L += ["", "## RQ1 — How well does regex alone do on real contracts?", ""]
    if rq["rq1_regex"]:
        m = rq["rq1_regex"]
        L.append(f"Regex: precision {m['precision']:.3f}, recall {m['recall']:.3f}, F1 {m['f1']:.3f} (sentence mode).")
    L += ["", "## RQ2 — Sentence vs clause segmentation", "", "| Method | F1 sentence | F1 clause |", "|---|---|---|"]
    for n in rq["rq2"]["sentence"]:
        L.append(f"| {n} | {rq['rq2']['sentence'][n]:.3f} | {rq['rq2']['clause'][n]:.3f} |")
    b, c = rq["rq3"]["best_single_macro"], rq["rq3"]["combined_macro"]
    L += ["", "## RQ3 — Does combining methods raise recall without losing too much precision?", "",
          f"Best single method: **{rq['rq3']['best_single']}** (P {b['precision']:.3f}, R {b['recall']:.3f}).  ",
          f"Combined: P {c['precision']:.3f}, R {c['recall']:.3f} → recall change {c['recall'] - b['recall']:+.3f}, "
          f"precision change {c['precision'] - b['precision']:+.3f}.", ""]
    L += ["## Route (a) vs (b) on Vietnamese sentences", ""]
    if routes:
        L += _table(routes)
    else:
        L.append("No Vietnamese sentences or route models available.")
    L += ["", "## Inter-annotator agreement (Cohen's Kappa)", ""]
    if kappa:
        for name, k in kappa.items():
            pairs = ", ".join(f"{p}: {v:.3f}" for p, v in k["kappa"].items())
            L.append(f"- {name}: {pairs} ({k['n_disagreements']} disagreements resolved)")
    else:
        L.append("No kappa files found next to the ground truth.")
    L += ["", "Confusion matrices: `confusion_combined_sentence.png`, `confusion_combined_clause.png`. "
          "Errors: `error_analysis.csv`.", "",
          "_Kết quả chỉ mang tính hỗ trợ, không thay thế tư vấn pháp lý._", ""]
    return "\n".join(L)


def _available_detectors() -> dict:
    out = {}
    for name in DETECTOR_NAMES:
        try:
            out[name] = build_detectors([name])[0]
        except (FileNotFoundError, OSError) as exc:
            print(f"skipping {name}: {exc}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contracts", type=Path, default=Path("data/contracts"))
    ap.add_argument("--truth", type=Path, default=Path("data/ground_truth"))
    ap.add_argument("--out", type=Path, default=Path("reports"))
    args = ap.parse_args()
    res = compare(args.contracts, args.truth, args.out)
    print(json.dumps(res["rq"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
