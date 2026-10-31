import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from lca.detectors.transformer import TransformerDetector
from lca.detectors.translate import TranslatingDetector
from lca.types import Detection, RiskLabel

CLASSES = ["indemnity", "none"]
VOCAB = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "filler": 4, "MARKER": 5, "plain": 6}


def _tokenizer():
    tk = Tokenizer(models.WordLevel(VOCAB, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    return PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", unk_token="[UNK]",
                                   cls_token="[CLS]", sep_token="[SEP]")


class MarkerModel(torch.nn.Module):
    """Logits favour 'indemnity' only when the MARKER token is inside the window."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, input_ids=None, attention_mask=None, **_):
        self.calls.append(input_ids.shape)
        hit = (input_ids == VOCAB["MARKER"]).any(dim=1).float()
        logits = torch.stack([hit * 10 - 5, 5 - hit * 10], dim=1)
        return type("Out", (), {"logits": logits})()


@pytest.fixture
def stub_detector():
    return TransformerDetector.from_components(
        MarkerModel(), _tokenizer(), CLASSES, {"indemnity": 0.5}, name="stub",
        device="cpu", max_length=32, stride=8, batch_size=2)


def test_long_text_risk_at_end_detected(stub_detector):
    text = "filler " * 1000 + "MARKER"
    out = stub_detector.predict([text])[0]
    assert out and out[0].label == RiskLabel.INDEMNITY and out[0].method == "stub"


def test_windows_respect_max_length(stub_detector):
    stub_detector.predict(["filler " * 200])
    assert all(shape[1] <= 32 for shape in stub_detector.model.calls)


def test_order_preserved_with_batches(stub_detector):
    out = stub_detector.predict(["plain", "MARKER", "plain"])
    assert [bool(x) for x in out] == [False, True, False]


def test_empty_input_list(stub_detector):
    assert stub_detector.predict([]) == []


class SpyTranslator:
    def __init__(self):
        self.seen = []

    def translate(self, texts):
        self.seen.extend(texts)
        return [f"EN:{t}" for t in texts]


class RecordingDetector:
    name = "inner"

    def __init__(self):
        self.seen = []

    def predict(self, texts):
        self.seen.extend(texts)
        return [[Detection("inner", RiskLabel.INDEMNITY, 0.9)] for _ in texts]


def test_translating_detector_only_translates_vietnamese():
    spy, inner = SpyTranslator(), RecordingDetector()
    TranslatingDetector(inner, spy, "t").predict(["Bên B bồi thường toàn bộ thiệt hại", "The Vendor pays"])
    assert spy.seen == ["Bên B bồi thường toàn bộ thiệt hại"]
    assert inner.seen == ["EN:Bên B bồi thường toàn bộ thiệt hại", "The Vendor pays"]


def test_translating_detector_renames_method():
    out = TranslatingDetector(RecordingDetector(), SpyTranslator(), "legal-bert+translate").predict(["x"])
    assert out[0][0].method == "legal-bert+translate"
