import pytest

from lca.ner import extract_entities
from lca.scoring import score
from lca.types import AnalysisContext, Detection, Level, RiskLabel, Segment


def seg(text, lang="vi"):
    return Segment(0, text, "Điều 8", [(0, 0, len(text))], lang)


def run(text, label, ctx=AnalysisContext(), methods=("regex",), lang="vi"):
    dets = [Detection(m, label, 0.9) for m in methods]
    return score(seg(text, lang), dets, extract_entities(text, lang), ctx)


def test_no_detections_returns_none():
    assert score(seg("x"), [], [], AnalysisContext()) is None


def test_label_is_highest_summed_score():
    dets = [Detection("regex", RiskLabel.PENALTY, 0.6), Detection("sts", RiskLabel.INDEMNITY, 0.5),
            Detection("tfidf", RiskLabel.INDEMNITY, 0.4)]
    assert score(seg("x"), dets, [], AnalysisContext()).label == RiskLabel.INDEMNITY


@pytest.mark.parametrize("methods,level", [(("regex",), Level.LOW), (("regex", "sts"), Level.MEDIUM),
                                           (("regex", "sts", "xlmr"), Level.HIGH)])
def test_base_level_from_method_agreement(methods, level):
    assert run("Bên B có nghĩa vụ liên quan", RiskLabel.TERMINATION, methods=methods).level == level


@pytest.mark.parametrize("pct,level", [("8", Level.MEDIUM), ("8,5", Level.HIGH), ("12", Level.HIGH)])
def test_commercial_penalty_boundary(pct, level):
    assert run(f"Bên B phải chịu phạt {pct}% giá trị phần nghĩa vụ bị vi phạm", RiskLabel.PENALTY).level == level


def test_8_percent_rule_not_applied_to_labour_contract():
    f = run("Người lao động phải chịu phạt 12% tiền lương", RiskLabel.PENALTY, AnalysisContext("labour"))
    assert f.level == Level.MEDIUM


def test_uncapped_indemnity_is_high():
    assert run("Bên B bồi thường toàn bộ thiệt hại phát sinh", RiskLabel.INDEMNITY).level == Level.HIGH


def test_uncapped_indemnity_english_is_high():
    f = run("Vendor shall indemnify Client without limit.", RiskLabel.INDEMNITY, lang="en")
    assert f.level == Level.HIGH


def test_two_year_confidentiality_is_medium():
    assert run("Người lao động có nghĩa vụ bảo mật thông tin trong 02 năm", RiskLabel.CONFIDENTIALITY).level == Level.MEDIUM


def test_indefinite_confidentiality_is_high():
    assert run("Bên B phải bảo mật thông tin vô thời hạn", RiskLabel.CONFIDENTIALITY).level == Level.HIGH


def test_other_party_obligation_lowered():
    f = run("Bên B phải chịu phạt 12% giá trị hợp đồng", RiskLabel.PENALTY, AnalysisContext(my_party="Bên A"))
    assert f.level == Level.MEDIUM


def test_my_party_obligation_not_lowered():
    f = run("Bên B phải chịu phạt 12% giá trị hợp đồng", RiskLabel.PENALTY, AnalysisContext(my_party="Bên B"))
    assert f.level == Level.HIGH


def test_unknown_obligor_not_lowered():
    f = run("Bên A có quyền phạt Bên B 12% giá trị hợp đồng", RiskLabel.PENALTY, AnalysisContext(my_party="Bên A"))
    assert f.level == Level.HIGH


def test_lowering_never_below_low():
    f = run("Bên B phải thông báo trước", RiskLabel.TERMINATION, AnalysisContext(my_party="Bên A"))
    assert f.level == Level.LOW


def test_reason_names_entities():
    f = run("Bên B phải chịu phạt 12% giá trị hợp đồng", RiskLabel.PENALTY)
    assert "12%" in f.reason and "Bên B" in f.reason and "Phạt vi phạm" in f.reason


def test_over_cap_penalty_inside_indemnity_sentence_is_high():
    text = "Bên vi phạm phải bồi thường thiệt hại và chịu phạt vi phạm bằng 10% giá trị hợp đồng."
    dets = [Detection("regex", RiskLabel.PENALTY, 1.0), Detection("regex", RiskLabel.INDEMNITY, 0.6),
            Detection("sts", RiskLabel.INDEMNITY, 0.8)]
    f = score(seg(text), dets, extract_entities(text, "vi"), AnalysisContext())
    assert f.label == RiskLabel.INDEMNITY and f.level == Level.HIGH
    assert "phạt" in f.reason.lower() and "8%" in f.reason


@pytest.mark.parametrize("stray", [Detection("regex", RiskLabel.PENALTY, 0.6),   # keyword "% giá trị hợp đồng"
                                   Detection("sts", RiskLabel.PENALTY, 0.7)])    # STS false alarm
def test_payment_share_is_not_an_over_cap_penalty(stray):
    text = "Bên A thanh toán 30% giá trị hợp đồng và bồi thường thiệt hại nếu chậm trả."
    dets = [stray, Detection("xlmr", RiskLabel.INDEMNITY, 0.9)]
    f = score(seg(text), dets, extract_entities(text, "vi"), AnalysisContext())
    assert f.label == RiskLabel.INDEMNITY and f.level == Level.MEDIUM
    assert "8%" not in f.reason


def test_weak_similarity_alone_is_not_reported():
    dets = [Detection("sts", RiskLabel.CONFIDENTIALITY, 0.52), Detection("sts", RiskLabel.INDEMNITY, 0.45)]
    assert score(seg("Hai bên thống nhất ký kết hợp đồng với các điều khoản sau:"), dets, [],
                 AnalysisContext()) is None


def test_strong_similarity_alone_is_reported_low():
    text = "Bên B phải trả thêm một khoản bằng một phần mười giá trị đơn hàng khi giao trễ"
    f = score(seg(text), [Detection("sts", RiskLabel.PENALTY, 0.73)], [], AnalysisContext())
    assert f is not None and f.label == RiskLabel.PENALTY and f.level == Level.LOW


def test_moderate_similarity_alone_does_not_trigger_penalty_cap():
    text = "Bên A thanh toán bằng chuyển khoản thành ba đợt: 30% sau khi ký, 40% sau khi nghiệm thu."
    f = score(seg(text), [Detection("sts", RiskLabel.PENALTY, 0.64)], extract_entities(text, "vi"),
              AnalysisContext())
    assert f.level == Level.LOW and "8%" not in f.reason


def test_full_repayment_of_training_cost_is_high():
    text = "Nhân viên phải hoàn trả lại 100% phí đào tạo và các khoản chế độ đã được nhận."
    assert run(text, RiskLabel.INDEMNITY, AnalysisContext("labour")).level == Level.HIGH
