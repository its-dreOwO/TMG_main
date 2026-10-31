import pytest

from lca.detectors.regex_rules import RegexDetector
from lca.types import RiskLabel


def scores(text):
    return {d.label: d.score for d in RegexDetector().predict([text])[0]}


@pytest.mark.parametrize("text,label,score", [
    ("Bên B phải chịu phạt 12% giá trị hợp đồng", RiskLabel.PENALTY, 1.0),
    ("Bên B bồi thường toàn bộ thiệt hại phát sinh", RiskLabel.INDEMNITY, 1.0),
    ("Nghĩa vụ bảo mật thông tin trong 02 năm", RiskLabel.CONFIDENTIALITY, 1.0),
    ("Vendor shall indemnify and hold harmless Client", RiskLabel.INDEMNITY, 0.6),
    ("Company may terminate for convenience at any time", RiskLabel.TERMINATION, 0.6),
    ("Employee shall not engage in competing business (non-compete)", RiskLabel.NON_COMPETE, 0.6),
    ("Bên A được đơn phương chấm dứt hợp đồng không cần báo trước", RiskLabel.TERMINATION, 0.6),
    ("Liquidated damages of $5,000 per day", RiskLabel.PENALTY, 0.6),
    ("PENALTY equal to 10% of the contract price", RiskLabel.PENALTY, 1.0),
    ("Bên B không được làm việc cho đối thủ cạnh tranh trong 12 tháng", RiskLabel.NON_COMPETE, 0.6),
])
def test_positive(text, label, score):
    assert scores(text).get(label) == score


@pytest.mark.parametrize("text", [
    "Bên B không bị phạt trong trường hợp bất khả kháng",
    "Thông tin mật là mọi thông tin được đánh dấu mật",
    '"Confidential Information" means any non-public information',
    "Hợp đồng này có hiệu lực từ ngày ký",
    "Hàng hóa bao gồm nhưng không giới hạn ở máy móc và thiết bị",
    "Bên A được miễn bồi thường trong trường hợp bất khả kháng",
])
def test_negated_definition_and_plain_text_not_flagged(text):
    assert RegexDetector().predict([text]) == [[]]


def test_one_output_per_input():
    assert len(RegexDetector().predict(["a", "b"])) == 2


def test_one_detection_per_label():
    out = RegexDetector().predict(["Bên B chịu phạt vi phạm 8% giá trị hợp đồng và tiền phạt khác"])[0]
    assert [d.label for d in out] == [RiskLabel.PENALTY]


def test_method_name():
    det = RegexDetector()
    assert det.name == "regex" and det.predict(["tiền phạt"])[0][0].method == "regex"


def test_never_returns_none_label():
    out = RegexDetector().predict(["bồi thường toàn bộ, bảo mật 2 năm, tiền phạt"])[0]
    assert RiskLabel.NONE not in {d.label for d in out} and len(out) == 3
