"""Read .docx and text-based .pdf contracts into a Document of text blocks."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterator

import docx
import pymupdf
from docx.table import Table
from docx.text.paragraph import Paragraph

from lca.types import Block, Document

MIN_CHARS_PER_PAGE = 20
HEADER_PAGE_RATIO = 0.6
HEADER_MIN_PAGES = 3


class UnreadableDocumentError(Exception):
    """Raised with a message that can be shown directly to the end user."""


def iter_docx_paragraphs(document) -> Iterator[Paragraph]:
    """Body and table-cell paragraphs in document order (nested tables included)."""
    yield from _iter_container(document)


def _iter_container(container) -> Iterator[Paragraph]:
    for item in container.iter_inner_content():
        if isinstance(item, Paragraph):
            yield item
        elif isinstance(item, Table):
            seen = set()
            for row in item.rows:
                for cell in row.cells:
                    if id(cell._tc) in seen:  # merged cells repeat the same element
                        continue
                    seen.add(id(cell._tc))
                    yield from _iter_container(cell)


def read_document(path: Path) -> Document:
    path = Path(path)
    if not path.is_file():
        raise UnreadableDocumentError(f"Không tìm thấy tệp: {path.name}")
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return _read_docx(path)
    if suffix == ".pdf":
        return _read_pdf(path)
    raise UnreadableDocumentError(
        f"Định dạng '{suffix or path.name}' không được hỗ trợ. Chỉ nhận .docx hoặc .pdf.")


def _read_docx(path: Path) -> Document:
    try:
        d = docx.Document(str(path))
    except Exception as exc:
        raise UnreadableDocumentError(
            f"Không đọc được tệp Word '{path.name}' (tệp hỏng hoặc không phải .docx).") from exc
    blocks = [Block(i, p.text) for i, p in enumerate(iter_docx_paragraphs(d)) if p.text.strip()]
    return Document(path, "docx", blocks)


def _read_pdf(path: Path) -> Document:
    try:
        pdf = pymupdf.open(str(path))
    except Exception as exc:
        raise UnreadableDocumentError(
            f"Không đọc được tệp PDF '{path.name}' (tệp hỏng).") from exc
    with pdf:
        if pdf.needs_pass:
            raise UnreadableDocumentError(
                f"Tệp PDF '{path.name}' được bảo vệ bằng mật khẩu. Hãy gỡ mật khẩu rồi thử lại.")
        pages = [[b[4] for b in page.get_text("blocks") if b[6] == 0] for page in pdf]
        n_pages = len(pages)
    total_chars = sum(len(t.strip()) for page in pages for t in page)
    if n_pages == 0 or total_chars / n_pages < MIN_CHARS_PER_PAGE:
        raise UnreadableDocumentError(
            f"Tệp PDF '{path.name}' có vẻ là bản scan (ảnh), không có lớp văn bản. "
            "Hệ thống chỉ hỗ trợ PDF dạng văn bản; OCR chưa được hỗ trợ.")
    headers = _repeated_lines(pages)
    blocks: list[Block] = []
    for page_no, page_blocks in enumerate(pages):
        for raw in page_blocks:
            lines = [l.strip() for l in raw.splitlines()]
            text = " ".join(l for l in lines if l and l not in headers)
            if text:
                blocks.append(Block(len(blocks), text, page_no))
    return Document(path, "pdf", blocks)


def _repeated_lines(pages: list[list[str]]) -> set[str]:
    if len(pages) < HEADER_MIN_PAGES:
        return set()
    counts = Counter()
    for page in pages:
        counts.update({l.strip() for t in page for l in t.splitlines() if l.strip()})
    return {l for l, c in counts.items() if c / len(pages) >= HEADER_PAGE_RATIO}
