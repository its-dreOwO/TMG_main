import csv
import json
from dataclasses import replace
from pathlib import Path

import docx
import pytest

from lca.detectors.regex_rules import RegexDetector
from lca.eval.annotate import consensus, make_template
from lca.eval.compare import compare, sentence_predictions
from lca.types import Finding, Level, RiskLabel, Segment

LABELS = {  # sentence text fragment -> true label
    "chịu phạt 12%": "penalty",
    "bồi thường toàn bộ": "indemnity",
    "hiệu lực": "none",
    "Bảo mật": "none",
    "giữ kín": "confidentiality",
}


def _write(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seg_id", "location", "text", "label"])
        w.writeheader()
        w.writerows(rows)


def _read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _label(text):
    return next((v for k, v in LABELS.items() if k in text), "none")


@pytest.fixture
def contracts(tmp_path):
    cdir, gdir = tmp_path / "contracts", tmp_path / "gt"
    cdir.mkdir()
    gdir.mkdir()
    for name, paras in {
        "hd1": ["Điều 1. Phạt", "Bên B phải chịu phạt 12% giá trị hợp đồng.", "Hợp đồng có hiệu lực từ ngày ký."],
        "hd2": ["Điều 2. Bảo mật", "Bên B bồi thường toàn bộ thiệt hại.",
                "Bên B phải giữ kín mọi tài liệu của Bên A."],
    }.items():
        d = docx.Document()
        for p in paras:
            d.add_paragraph(p)
        d.save(cdir / f"{name}.docx")
        tmpl = gdir / f"{name}.template.csv"
        make_template(cdir / f"{name}.docx", tmpl)
        rows = _read(tmpl)
        _write(gdir / f"{name}.csv", [r | {"label": _label(r["text"])} for r in rows])
        tmpl.unlink()
    return cdir, gdir


def _renamed(name):
    class Renamed:
        def __init__(self):
            self.name = name

        def predict(self, texts):
            return [[replace(d, method=name) for d in ds] for ds in RegexDetector().predict(texts)]
    return Renamed()


@pytest.fixture
def detectors():
    names = ["regex", "tfidf", "legal-bert", "legal-bert+translate", "xlmr", "sts"]
    return {n: _renamed(n) for n in names}


def test_template_has_blank_labels_per_sentence(tmp_path):
    d = docx.Document()
    d.add_paragraph("Bên B phải chịu phạt. Bên A trả tiền.")
    d.save(tmp_path / "a.docx")
    make_template(tmp_path / "a.docx", tmp_path / "a.csv")
    rows = _read(tmp_path / "a.csv")
    assert list(rows[0]) == ["seg_id", "location", "text", "label"]
    assert len(rows) == 2 and all(r["label"] == "" for r in rows)


def _annot(path, labels):
    _write(path, [{"seg_id": i, "location": "x", "text": f"t{i}", "label": l} for i, l in enumerate(labels)])
    return path


def test_kappa_perfect_agreement_is_one(tmp_path):
    a = _annot(tmp_path / "a.csv", ["penalty", "none", "indemnity"])
    b = _annot(tmp_path / "b.csv", ["penalty", "none", "indemnity"])
    res = consensus([a, b], tmp_path / "gt.csv")
    assert list(res["kappa"].values()) == [pytest.approx(1.0)] and res["n_disagreements"] == 0


def test_kappa_reported_per_pair_for_three_annotators(tmp_path):
    paths = [_annot(tmp_path / f"{n}.csv", ["penalty", "none", l]) for n, l in zip("abc", ["none", "none", "penalty"])]
    res = consensus(paths, tmp_path / "gt.csv")
    assert len(res["kappa"]) == 3


def test_consensus_writes_majority_and_blanks_ties(tmp_path):
    a = _annot(tmp_path / "a.csv", ["penalty", "none"])
    b = _annot(tmp_path / "b.csv", ["penalty", "indemnity"])
    consensus([a, b], tmp_path / "gt.csv")
    rows = _read(tmp_path / "gt.csv")
    assert [r["label"] for r in rows] == ["penalty", ""]
    assert json.loads((tmp_path / "gt.kappa.json").read_text())["n_disagreements"] == 1


def test_blank_labels_rejected(tmp_path):
    a = _annot(tmp_path / "a.csv", ["penalty", ""])
    b = _annot(tmp_path / "b.csv", ["penalty", "none"])
    with pytest.raises(ValueError, match="seg_id"):
        consensus([a, b], tmp_path / "gt.csv")


def test_unknown_label_rejected(tmp_path):
    a = _annot(tmp_path / "a.csv", ["phạt"])
    b = _annot(tmp_path / "b.csv", ["penalty"])
    with pytest.raises(ValueError, match="phạt"):
        consensus([a, b], tmp_path / "gt.csv")


def _sent(i, start, end):
    return Segment(i, f"s{i}", "Điều 1", [(0, start, end)], "vi")


def test_clause_finding_projects_onto_contained_sentences():
    sentences = [_sent(0, 0, 10), _sent(1, 11, 20), _sent(2, 0, 5)]
    sentences[2] = Segment(2, "s2", "Điều 2", [(1, 0, 5)], "vi")
    clause = Segment(0, "c", "Điều 1", [(0, 0, 20)], "vi")
    clause_finding = Finding(clause, RiskLabel.PENALTY, [], [], Level.LOW, "")
    assert sentence_predictions(sentences, [clause_finding]) == ["penalty", "penalty", "none"]


def test_projection_prefers_higher_level_finding():
    s = [_sent(0, 0, 10)]
    low = Finding(_sent(0, 0, 10), RiskLabel.PENALTY, [], [], Level.LOW, "")
    high = Finding(_sent(0, 0, 10), RiskLabel.INDEMNITY, [], [], Level.HIGH, "")
    assert sentence_predictions(s, [low, high]) == ["indemnity"]


def test_compare_reports_every_method(contracts, detectors, tmp_path):
    res = compare(*contracts, tmp_path / "rep", detectors=detectors)
    assert {"regex", "tfidf", "legal-bert+translate", "xlmr", "sts", "combined"} <= set(res["methods"])
    assert {"sentence", "clause"} <= set(res["modes"])
    assert res["modes"]["sentence"]["regex"]["per_label"]["penalty"]["recall"] == 1.0


def test_compare_writes_report_answering_research_questions(contracts, detectors, tmp_path):
    compare(*contracts, tmp_path / "rep", detectors=detectors)
    md = (tmp_path / "rep" / "results.md").read_text()
    for heading in ("RQ1", "RQ2", "RQ3", "Route (a) vs (b)", "case study"):
        assert heading in md
    assert list((tmp_path / "rep").glob("confusion_*.png"))


def test_error_analysis_lists_false_positives_and_negatives(contracts, detectors, tmp_path):
    compare(*contracts, tmp_path / "rep", detectors=detectors)
    rows = _read(tmp_path / "rep" / "error_analysis.csv")
    kinds = {r["error"] for r in rows}
    assert kinds <= {"FP", "FN", "WRONG_LABEL"} and rows
    assert any("giữ kín" in r["text"] and r["error"] == "FN" for r in rows)


def test_compare_rejects_ground_truth_that_does_not_match_segmentation(contracts, detectors, tmp_path):
    cdir, gdir = contracts
    rows = _read(gdir / "hd1.csv")
    _write(gdir / "hd1.csv", rows[:-1])
    with pytest.raises(ValueError, match="hd1"):
        compare(cdir, gdir, tmp_path / "rep", detectors=detectors)
