import math

import pytest

from lca.ner import extract_entities, obligor


def values(text, lang, kind):
    return [e.value for e in extract_entities(text, lang) if e.kind == kind]


@pytest.mark.parametrize("text,lang,kind,value", [
    ("phạt 12% giá trị hợp đồng", "vi", "PERCENT", 12.0),
    ("phạt 8,5 % giá trị", "vi", "PERCENT", 8.5),
    ("tám phần trăm giá trị", "vi", "PERCENT", 8.0),
    ("a penalty of 2.5% of the price", "en", "PERCENT", 2.5),
    ("ten percent of the fees", "en", "PERCENT", 10.0),
    ("500.000.000 đồng", "vi", "MONEY", 5e8),
    ("1,5 tỷ đồng", "vi", "MONEY", 1.5e9),
    ("$1,000,000", "en", "MONEY", 1e6),
    ("trong 02 năm", "vi", "DURATION", 24),
    ("30 ngày", "vi", "DURATION", 1),
    ("vô thời hạn", "vi", "DURATION", math.inf),
    ("for two (2) years", "en", "DURATION", 24),
    ("in perpetuity", "en", "DURATION", math.inf),
    ("for twelve months", "en", "DURATION", 12),
])
def test_values(text, lang, kind, value):
    assert value in values(text, lang, kind)


def test_vietnamese_date_and_party():
    kinds = {e.kind for e in extract_entities("Bên B thanh toán trước ngày 15/03/2026", "vi")}
    assert {"PARTY", "DATE"} <= kinds


def test_written_vietnamese_date_is_not_a_duration():
    ents = extract_entities("Hợp đồng ký ngày 15 tháng 03 năm 2026", "vi")
    assert "DATE" in {e.kind for e in ents} and "DURATION" not in {e.kind for e in ents}


def test_english_party_and_org():
    ents = extract_entities("Party A shall pay Acme Corporation within 30 days.", "en")
    parties = {e.text for e in ents if e.kind == "PARTY"}
    assert "Party A" in parties


def test_labour_parties():
    parties = {e.text for e in extract_entities("Người lao động phải bồi thường cho Người sử dụng lao động", "vi")
               if e.kind == "PARTY"}
    assert {"Người lao động", "Người sử dụng lao động"} <= parties


def test_obligor_nearest_party_before_obligation_word():
    text = "Theo Điều 3, Bên A và Bên B đồng ý rằng Bên B phải chịu phạt 12%."
    assert obligor(text, extract_entities(text, "vi")) == "Bên B"


def test_obligor_english_shall():
    text = "Party B shall indemnify Party A."
    assert obligor(text, extract_entities(text, "en")) == "Party B"


def test_obligor_unknown_without_obligation_word():
    text = "Bên A có quyền phạt Bên B 12% giá trị hợp đồng"
    assert obligor(text, extract_entities(text, "vi")) is None
