import joblib
import pandas as pd
import pytest

from lca.detectors.tfidf import TfidfDetector
from lca.train.tfidf_train import fit_model
from lca.types import RiskLabel

ROWS = (
    [("the vendor shall indemnify the buyer against claims", "indemnity")] * 20
    + [("each party shall keep all information confidential", "confidentiality")] * 20
    + [("this agreement is governed by the laws of new york", "none")] * 40
)


@pytest.fixture(scope="module", params=["lr", "svm"])
def tiny_model_path(request, tmp_path_factory):
    df = pd.DataFrame(ROWS, columns=["text", "label"])
    path = tmp_path_factory.mktemp("m") / f"{request.param}.joblib"
    joblib.dump(fit_model(request.param, df, df, min_df=1), path)
    return path


def test_saved_bundle_keys(tiny_model_path):
    assert set(joblib.load(tiny_model_path)) == {"pipeline", "classes", "thresholds"}


def test_tfidf_detector_never_returns_none(tiny_model_path):
    out = TfidfDetector(tiny_model_path).predict(["governed by the laws of new york", "x"])
    assert all(d.label != RiskLabel.NONE for ds in out for d in ds)


def test_tfidf_detects_trained_phrase(tiny_model_path):
    out = TfidfDetector(tiny_model_path).predict(["shall indemnify the buyer"])
    assert out[0][0].label == RiskLabel.INDEMNITY and out[0][0].method == "tfidf"


def test_one_output_per_input(tiny_model_path):
    assert len(TfidfDetector(tiny_model_path).predict(["a", "b", "c"])) == 3
