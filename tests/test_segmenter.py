import unicodedata
from pathlib import Path

from lca.segmenter import detect_lang, normalize, segment
from lca.types import Block, Document

VI = ["Điều 8. Phạt vi phạm",
      "8.2. Bên B phải chịu phạt 12% giá trị hợp đồng nếu giao hàng chậm.",
      "Điều 9. Bồi thường"]


def doc(texts):
    return Document(Path("x.docx"), "docx", [Block(i, t) for i, t in enumerate(texts)])


def raw_of(d, seg):
    return " ".join(d.blocks[b].text[s:e] for b, s, e in seg.spans)


def test_normalize_nfc_and_whitespace():
    assert normalize("  Bên  B\n" + unicodedata.normalize("NFD", "bồi") + "  ") == "Bên B bồi"


def test_detect_lang():
    assert detect_lang("Bên B phải bồi thường toàn bộ thiệt hại") == "vi"
    assert detect_lang("The Vendor shall indemnify the Client") == "en"


def test_clause_mode_groups_until_next_heading():
    segs = segment(doc(VI), "clause")
    assert len(segs) == 2 and "12%" in segs[0].text and segs[0].location == "Điều 8"


def test_clause_spans_cover_multiple_blocks():
    assert [s[0] for s in segment(doc(VI), "clause")[0].spans] == [0, 1]


def test_text_before_first_heading_is_preamble():
    segs = segment(doc(["HỢP ĐỒNG MUA BÁN", "Điều 1. Đối tượng"]), "clause")
    assert segs[0].location == "Mở đầu" and segs[1].location == "Điều 1"


def test_sentence_location_uses_subclause():
    assert any(s.location == "Điều 8.2" for s in segment(doc(VI), "sentence"))


def test_english_article_heading():
    segs = segment(doc(["Article 7 Indemnification", "Vendor shall indemnify."]), "clause")
    assert segs[0].location == "Article 7"


def test_enumerated_items_split():
    segs = segment(doc(["Bên B phải: a) trả phí; b) bồi thường; c) bảo mật."]), "sentence")
    assert len(segs) == 3


def test_enumerated_item_keeps_lead_in_as_context_only():
    d = doc(["Bên B phải: a) trả phí; b) bồi thường toàn bộ."])
    seg = segment(d, "sentence")[1]
    assert seg.text.startswith("Bên B phải:") and "bồi thường" in seg.text
    assert raw_of(d, seg).startswith("b)")


def test_semicolons_split_without_markers():
    assert len(segment(doc(["Bên A trả phí; Bên B giao hàng."]), "sentence")) == 2


def test_abbreviation_does_not_split():
    d = doc(["See Sec. 4 of the Agreement between Acme Inc. and Beta."])
    assert len(segment(d, "sentence")) == 1


def test_two_sentences_split():
    d = doc(["Bên B phải chịu phạt 12% giá trị hợp đồng. Bên A có quyền chấm dứt theo Điều 7.2."])
    assert len(segment(d, "sentence")) == 2


def test_nfd_input_normalised_with_raw_offsets():
    raw = unicodedata.normalize("NFD", "Bên B bồi thường toàn bộ.")
    d = doc([raw])
    s = segment(d, "sentence")[0]
    assert "bồi thường" in s.text
    assert raw_of(d, s) == raw


def test_spans_reproduce_sentence_text():
    d = doc(["  Bên B   bồi thường.  Bên A trả phí.  "])
    segs = segment(d, "sentence")
    assert [normalize(raw_of(d, s)) for s in segs] == [s.text for s in segs]


def test_segment_language_detected():
    assert segment(doc(VI), "sentence")[0].lang == "vi"


def test_short_segment_uses_document_language():
    d = doc(["Bên B phải bồi thường toàn bộ thiệt hại cho Bên A như sau:", "a) 5%."])
    assert {s.lang for s in segment(d, "sentence")} == {"vi"}


def test_seg_ids_sequential():
    segs = segment(doc(VI), "sentence")
    assert [s.seg_id for s in segs] == list(range(len(segs)))
