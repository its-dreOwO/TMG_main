import docx
import pymupdf
import pytest

from lca.reader import UnreadableDocumentError, read_document


def _pdf(path, pages, **save_kw):
    pdf = pymupdf.open()
    for lines in pages:
        page = pdf.new_page()
        y = 72
        for line in lines:
            page.insert_text((72, y), line, fontsize=11)
            y += 40
    pdf.save(path, **save_kw)
    return path


def test_docx_includes_table_cells_in_order(tmp_path):
    d = docx.Document()
    d.add_paragraph("A")
    d.add_table(rows=1, cols=1).cell(0, 0).text = "B"
    d.add_paragraph("C")
    path = tmp_path / "a.docx"
    d.save(path)
    assert [b.text for b in read_document(path).blocks] == ["A", "B", "C"]


def test_docx_empty_paragraph_still_counted_in_index(tmp_path):
    d = docx.Document()
    d.add_paragraph("A")
    d.add_paragraph("")
    d.add_paragraph("C")
    path = tmp_path / "a.docx"
    d.save(path)
    assert [(b.index, b.text) for b in read_document(path).blocks] == [(0, "A"), (2, "C")]


def test_pdf_blocks_have_pages(tmp_path):
    path = _pdf(tmp_path / "a.pdf", [["Article 1 The Vendor shall deliver the goods on time."]])
    doc = read_document(path)
    assert doc.kind == "pdf" and doc.blocks[0].page == 0


def test_pdf_repeated_header_removed(tmp_path):
    pages = [["CONFIDENTIAL DRAFT", f"Body text of page number {i} with enough words."] for i in range(3)]
    path = _pdf(tmp_path / "a.pdf", pages)
    blocks = read_document(path).blocks
    assert blocks and all("CONFIDENTIAL DRAFT" not in b.text for b in blocks)


def test_image_only_pdf_raises(tmp_path):
    pdf = pymupdf.open()
    pdf.new_page().draw_rect(pymupdf.Rect(50, 50, 300, 300), fill=(0, 0, 0))
    path = tmp_path / "scan.pdf"
    pdf.save(path)
    with pytest.raises(UnreadableDocumentError, match="scan"):
        read_document(path)


def test_corrupt_file_raises(tmp_path):
    path = tmp_path / "broken.docx"
    path.write_bytes(b"not a zip file at all")
    with pytest.raises(UnreadableDocumentError):
        read_document(path)


def test_corrupt_pdf_raises(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-garbage")
    with pytest.raises(UnreadableDocumentError):
        read_document(path)


def test_encrypted_pdf_raises(tmp_path):
    path = _pdf(tmp_path / "locked.pdf", [["Secret text body here."]],
                encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="u", owner_pw="o")
    with pytest.raises(UnreadableDocumentError, match="mật khẩu"):
        read_document(path)


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("x")
    with pytest.raises(UnreadableDocumentError):
        read_document(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(UnreadableDocumentError):
        read_document(tmp_path / "nope.docx")


def test_word_file_named_pdf_is_read_as_docx(tmp_path):
    d = docx.Document()
    d.add_paragraph("Điều 1. Hiệu lực")
    path = tmp_path / "mau.pdf"
    d.save(path)
    doc = read_document(path)
    assert doc.kind == "docx" and [b.text for b in doc.blocks] == ["Điều 1. Hiệu lực"]


def test_pdf_file_named_docx_is_read_as_pdf(tmp_path):
    path = _pdf(tmp_path / "a.docx", [["Article 1 The Vendor shall deliver the goods on time."]])
    doc = read_document(path)
    assert doc.kind == "pdf" and "Vendor shall deliver" in doc.blocks[0].text
