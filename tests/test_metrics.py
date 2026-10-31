import numpy as np
import pytest

from lca.eval.metrics import apply_thresholds, predict_labels, report, tune_thresholds


def test_threshold_favours_recall():
    t = tune_thresholds(["a", "none", "none"], np.array([[.3, .7], [.1, .9], [.05, .95]]), ["a", "none"])
    assert 0.1 < t["a"] <= 0.3


def test_threshold_only_for_risk_classes():
    t = tune_thresholds(["a", "none"], np.array([[.9, .1], [.1, .9]]), ["a", "none"])
    assert set(t) == {"a"}


def test_threshold_default_when_class_absent_from_val():
    t = tune_thresholds(["none"], np.array([[.1, .9]]), ["a", "none"])
    assert t["a"] == 0.5


def test_threshold_prefers_recall_over_precision():
    # 2 positives at .4/.8, 2 negatives at .5/.6: F2 picks .4 (R=1) over .8 (R=.5, P=1)
    y = ["a", "a", "none", "none"]
    p = np.array([[.4, .6], [.8, .2], [.5, .5], [.6, .4]])
    assert tune_thresholds(y, p, ["a", "none"])["a"] <= 0.4


def test_apply_thresholds_can_return_several_and_none():
    out = apply_thresholds(np.array([[.6, .5, .1]]), ["a", "b", "none"], {"a": .5, "b": .4})
    assert {c for c, _ in out[0]} == {"a", "b"}
    assert apply_thresholds(np.array([[.1, .1, .8]]), ["a", "b", "none"], {"a": .5, "b": .4}) == [[]]


def test_predict_labels_picks_highest_passing_else_none():
    p = np.array([[.6, .7, 0], [.1, .1, .8]])
    assert predict_labels(p, ["a", "b", "none"], {"a": .5, "b": .5}) == ["b", "none"]


def test_report_macro_and_per_label():
    r = report(["a", "b", "none"], ["a", "none", "none"], ["a", "b", "none"])
    assert r["per_label"]["a"]["recall"] == 1.0 and r["per_label"]["b"]["recall"] == 0.0
    assert r["macro"]["f1"] == pytest.approx(np.mean([r["per_label"][l]["f1"] for l in "a b none".split()]))
    assert r["confusion"] == [[1, 0, 0], [0, 0, 1], [0, 0, 1]]
