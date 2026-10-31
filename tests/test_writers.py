import docx
import pymupdf
import pytest
from docx.enum.text import WD_COLOR_INDEX

from lca.reader import iter_docx_paragraphs, read_document
from lca.types import Finding, Level, RiskLabel, Segment
from lca.writers import write_docx, write_pdf


def finding(block, start, end, text="x", label=RiskLabel.PENALTY, level=Level.HIGH):
    return Finding(Segment(0, text, "Điều 8", [(block, start, end)], "vi"), label, [], [], level,
                   "Phạt vi phạm; mức 12%.")


def red_text(paragraph):
    return "".join(r.text for r in paragraph.runs if r.font.highlight_color == WD_COLOR_INDEX.RED)


def comments(path):
    return [c.text for c in docx.Document(path).comments]


@pytest.fixture
def multi_run_docx(tmp_path):
    d = docx.Document()
    d.add_paragraph("Mở đầu.")
    p = d.add_paragraph("Bên B ")
    p.add_run("bồi thường").bold = True
    p.add_run(" toàn bộ. Câu khác.")
    path = tmp_path / "src.docx"
    d.save(path)
    return path


def test_docx_sentence_across_runs_fully_highlighted(multi_run_docx, tmp_path):
    out = tmp_path / "out.docx"
    write_docx(multi_run_docx, [finding(1, 0, 25)], out)
    para = docx.Document(out).paragraphs[1]
    assert red_text(para) == "Bên B bồi thường toàn bộ."
    assert para.text == "Bên B bồi thường toàn bộ. Câu khác."


def test_docx_partial_highlight_splits_run_keeps_bold(multi_run_docx, tmp_path):
    out = tmp_path / "out.docx"
    write_docx(multi_run_docx, [finding(1, 6, 11)], out)  # "bồi t" inside the bold run
    para = docx.Document(out).paragraphs[1]
    assert red_text(para) == "bồi t"
    assert all(r.bold for r in para.runs if r.text in ("bồi t", "hường"))


def test_docx_table_cell_highlighted(tmp_path):
    d = docx.Document()
    d.add_paragraph("A")
    d.add_table(rows=1, cols=1).cell(0, 0).text = "Bên B chịu phạt 12%."
    src, out = tmp_path / "t.docx", tmp_path / "o.docx"
    d.save(src)
    write_docx(src, [finding(1, 0, 20)], out)
    cell_para = list(iter_docx_paragraphs(docx.Document(out)))[1]
    assert red_text(cell_para) == "Bên B chịu phạt 12%."


def test_docx_comment_names_risk_and_level(multi_run_docx, tmp_path):
    out = tmp_path / "out.docx"
    write_docx(multi_run_docx, [finding(1, 0, 25)], out)
    note = comments(out)[0]
    assert "Phạt vi phạm" in note and "Cao" in note


def test_docx_unflagged_text_unchanged(multi_run_docx, tmp_path):
    out = tmp_path / "out.docx"
    write_docx(multi_run_docx, [finding(1, 0, 25)], out)
    first = docx.Document(out).paragraphs[0]
    assert first.text == "Mở đầu." and red_text(first) == ""


def test_docx_no_findings_is_a_valid_copy(multi_run_docx, tmp_path):
    out = tmp_path / "out.docx"
    write_docx(multi_run_docx, [], out)
    assert [p.text for p in docx.Document(out).paragraphs] == [p.text for p in docx.Document(multi_run_docx).paragraphs]


def _pdf(path, pages):
    pdf = pymupdf.open()
    for lines in pages:
        page = pdf.new_page()
        y = 72
        for line in lines:
            page.insert_text((72, y), line, fontsize=11)
            y += 40
    pdf.save(path)
    return path


def test_pdf_highlight_annotation_on_right_page(tmp_path):
    src = _pdf(tmp_path / "s.pdf", [["Page one has nothing risky at all."],
                                    ["The Vendor shall pay a penalty of 12% of the price."]])
    doc = read_document(src)
    block = next(b for b in doc.blocks if "penalty" in b.text)
    out = tmp_path / "o.pdf"
    write_pdf(src, doc, [finding(block.index, 0, len(block.text))], out)
    pdf = pymupdf.open(out)
    assert not list(pdf[0].annots())
    page = pdf[1]  # keep the page alive while reading its annotations
    annot = list(page.annots())[0]
    assert annot.type[1] == "Highlight" and "Cao" in annot.info["content"]


def test_pdf_sentence_wrapping_lines_gets_annotation(tmp_path):
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_textbox(pymupdf.Rect(72, 72, 250, 300),
                        "The Vendor shall indemnify and hold harmless the Client from any and all claims "
                        "arising out of this Agreement.", fontsize=11)
    src = tmp_path / "w.pdf"
    pdf.save(src)
    doc = read_document(src)
    block = doc.blocks[0]
    out = tmp_path / "o.pdf"
    write_pdf(src, doc, [finding(block.index, 0, len(block.text), label=RiskLabel.INDEMNITY)], out)
    assert list(pymupdf.open(out)[0].annots())


def test_output_never_overwrites_source(multi_run_docx, tmp_path):
    with pytest.raises(ValueError):
        write_docx(multi_run_docx, [], multi_run_docx)
    src = _pdf(tmp_path / "s.pdf", [["Text that is long enough to read."]])
    with pytest.raises(ValueError):
        write_pdf(src, read_document(src), [], src)
