from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lca.detectors.sts import StsDetector, load_prototypes, tune_sts
from lca.segmenter import detect_lang
from lca.types import RiskLabel

PATH = Path("lca/resources/prototypes.yaml")


class FakeEncoder:
    """Deterministic 2-D embedding: x-axis 'penalty-ness', y-axis 'other'."""

    def encode(self, texts, normalize_embeddings=True, **_):
        v = np.array([[1.0, 0.0] if "phạt" in t else [0.6, 0.8] if "half" in t else [0.0, 1.0] for t in texts])
        return v / np.linalg.norm(v, axis=1, keepdims=True)


PROTOS = {"penalty": ["tiền phạt"]}


def test_flags_above_threshold():
    out = StsDetector(PROTOS, {"penalty": 0.9}, encoder=FakeEncoder()).predict(["bị phạt nặng"])
    assert out[0][0].label == RiskLabel.PENALTY and out[0][0].score == pytest.approx(1.0)
    assert out[0][0].method == "sts"


def test_silent_below_threshold():
    assert StsDetector(PROTOS, {"penalty": 0.9}, encoder=FakeEncoder()).predict(["half match"]) == [[]]


def test_one_output_per_input_and_empty():
    det = StsDetector(PROTOS, {"penalty": 0.9}, encoder=FakeEncoder())
    assert det.predict([]) == [] and len(det.predict(["a", "b"])) == 2


def test_tune_sts_returns_threshold_per_group():
    val = pd.DataFrame({"text": ["phạt a", "phạt b", "half x", "other"], "label": ["penalty", "penalty", "none", "none"]})
    t = tune_sts(FakeEncoder(), PROTOS, val)
    assert set(t) == {"penalty"} and 0.6 < t["penalty"] <= 1.0


def test_prototypes_cover_all_groups_in_both_languages():
    p = load_prototypes(PATH)
    assert set(p) == {"penalty", "indemnity", "confidentiality", "termination", "non_compete"}
    assert all(len(v) >= 10 for v in p.values())
    assert all(sum(detect_lang(t) == "vi" for t in v) >= 5 and sum(detect_lang(t) == "en" for t in v) >= 5
               for v in p.values())
    assert "Bên vi phạm phải chịu phạt 8% giá trị phần nghĩa vụ bị vi phạm." in p["penalty"]


@pytest.mark.slow
def test_paraphrase_without_keyword_matches_real_model():
    import json
    thresholds = json.loads(Path("artifacts/models/sts_thresholds.json").read_text())
    det = StsDetector(load_prototypes(PATH), thresholds)
    out = det.predict(["Bên B phải trả thêm một khoản bằng một phần mười giá trị đơn hàng khi giao trễ"])
    assert RiskLabel.PENALTY in {d.label for d in out[0]}
