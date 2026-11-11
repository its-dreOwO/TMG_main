import pytest

from lca.detectors.regex_rules import RegexDetector, is_definition
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


@pytest.mark.parametrize("text", [
    "“Fees” has the meaning set forth in Section 5.1.",
    "1.14 “Program Data” means the data set forth on Exhibit A in the possession of University.",
    "DEFINITIONS 1.1 “Affiliates” of a person or entity means any other entity which controls it.",
    "1.15 “Project” shall mean the compilation of the Program Data.",
    "The term “control” means the possession of the power to direct the management of a person.",
    '"Confidential Information" means any non-public information',
    "Commencement Date: means the date of this agreement.",
    "Inactive Territories means a country in which the Distributor has not sold the Product.",
    "Initial Term has the meaning given to it in clause 12.1.",
    "1.2. “Thông tin mật” là mọi thông tin do Bên A cung cấp cho Bên B.",
    "“Sự kiện bất khả kháng” có nghĩa là sự kiện xảy ra một cách khách quan.",
])
def test_definition_sentences_recognised(text):
    assert is_definition(text)


@pytest.mark.parametrize("text", [
    "Mức phạt là 12% giá trị hợp đồng.",
    "Bên B là bên chịu phạt 12% giá trị hợp đồng nếu giao hàng chậm.",
    "This means the Supplier shall pay a penalty of 12% of the contract value.",
    "Consultation by either Party shall be by means of personal visits and telephone calls.",
    "Each Party shall keep the “Confidential Information” secret, which means no disclosure for 5 years.",
    "The Receiving Party shall maintain in confidence all Confidential Information for seven (7) years.",
])
def test_obligations_are_not_definitions(text):
    assert not is_definition(text)
