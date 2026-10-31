from lca.types import Level, RiskLabel


def test_level_vietnamese_names():
    assert Level.HIGH.vi == "Cao" and Level.MEDIUM.vi == "Trung bình" and Level.LOW.vi == "Thấp"


def test_levels_are_ordered():
    assert Level.LOW < Level.MEDIUM < Level.HIGH


def test_risk_label_set():
    assert {l.value for l in RiskLabel} == {
        "penalty", "indemnity", "confidentiality", "termination", "non_compete", "none"}
