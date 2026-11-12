"""Write the analysed contract back out: red highlights + notes in .docx/.pdf, and a CSV report."""
from __future__ import annotations

import copy
import unicodedata
from pathlib import Path

import docx
import pymupdf
from docx.enum.text import WD_COLOR_INDEX
from docx.text.hyperlink import Hyperlink
from docx.text.run import Run

from lca.reader import iter_docx_paragraphs
from lca.types import Document, Finding

AUTHOR = "Legal Clause Analyzer"
RED = (1, 0, 0)


def note(f: Finding) -> str:
    return f"{f.label.vi} – Mức {f.level.vi}: {f.reason}"


def _check_paths(source: Path, out: Path) -> None:
    if Path(out).resolve() == Path(source).resolve():
        raise ValueError("Output path must differ from the source file; the original is never overwritten.")


# ---------------------------------------------------------------- docx
def _runs(paragraph) -> list[Run]:
    out = []
    for item in paragraph.iter_inner_content():
        out.extend(item.runs if isinstance(item, Hyperlink) else [item])
    return out


def _split(run: Run, k: int) -> tuple[Run, Run]:
    """Split a run at character k, both halves keeping the original formatting."""
    new_r = copy.deepcopy(run._r)
    run._r.addnext(new_r)
    right = Run(new_r, run._parent)
    text = run.text
    run.text, right.text = text[:k], text[k:]
    return run, right


def _highlight_range(paragraph, start: int, end: int) -> list[Run]:
    runs = _runs(paragraph)
    if "".join(r.text for r in runs) != paragraph.text:
        return []  # unusual structure (e.g. fields); skip rather than mis-highlight
    out, pos = [], 0
    for run in runs:
        text = run.text
        a, b = pos, pos + len(text)
        pos = b
        if not text or b <= start or a >= end:
            continue
        if a < start:
            _, run = _split(run, start - a)
            a = start
        if b > end:
            run, _ = _split(run, end - a)
        run.font.highlight_color = WD_COLOR_INDEX.RED
        out.append(run)
    return out


def write_docx(source: Path, findings: list[Finding], out: Path) -> None:
    _check_paths(source, out)
    document = docx.Document(str(source))
    paragraphs = list(iter_docx_paragraphs(document))
    for f in findings:
        marked: list[Run] = []
        for block, start, end in f.segment.spans:
            if 0 <= block < len(paragraphs):
                marked.extend(_highlight_range(paragraphs[block], start, end))
        if marked:
            anchor = marked[0] if len(marked) == 1 else [marked[0], marked[-1]]
            document.add_comment(anchor, text=note(f), author=AUTHOR, initials="LCA")
    document.save(str(out))


# ---------------------------------------------------------------- pdf
def _find_rects(page, text: str) -> list:
    """Rects of the words of `text` on the page, matched as a word sequence (handles line wraps)."""
    target = [unicodedata.normalize("NFC", w) for w in text.split()]
    if not target:
        return []
    words = page.get_text("words")
    tokens = [unicodedata.normalize("NFC", w[4]) for w in words]
    n = len(target)
    for i in range(len(tokens) - n + 1):
        if tokens[i:i + n] == target:
            return [pymupdf.Rect(w[:4]) for w in words[i:i + n]]
    return page.search_for(text)


def write_pdf(source: Path, doc: Document, findings: list[Finding], out: Path) -> None:
    _check_paths(source, out)
    blocks = {b.index: b for b in doc.blocks}
    with pymupdf.open(str(source)) as pdf:
        for f in findings:
            for index, start, end in f.segment.spans:
                block = blocks.get(index)
                if block is None or block.page is None:
                    continue
                page = pdf[block.page]
                rects = _find_rects(page, block.text[start:end])
                if not rects:
                    continue
                annot = page.add_highlight_annot(rects)
                annot.set_colors(stroke=RED)
                annot.set_info(content=note(f), title=AUTHOR)
                annot.update()
        pdf.save(str(out))


# ---------------------------------------------------------------- report
REPORT_COLUMNS = ["Vị trí", "Câu trích xuất", "Loại rủi ro", "Thực thể", "Mức", "Giải thích"]
DISCLAIMER = "Lưu ý: kết quả chỉ mang tính hỗ trợ sàng lọc, không thay thế tư vấn pháp lý."
NO_FINDINGS = "Không phát hiện điều khoản rủi ro nào."


def sort_findings(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda f: (-f.level, f.segment.seg_id))


def report_rows(findings: list[Finding]) -> list[dict[str, str]]:
    return [{
        "Vị trí": f.segment.location,
        "Câu trích xuất": f.segment.text,
        "Loại rủi ro": f.label.vi,
        "Thực thể": ", ".join(f"{e.text} ({e.kind})" for e in f.entities),
        "Mức": f.level.vi,
        "Giải thích": f.reason,
    } for f in sort_findings(findings)]


def write_report(findings: list[Finding], out: Path) -> None:
    """UTF-8 with BOM so Excel shows Vietnamese correctly."""
    import csv

    rows = report_rows(findings) or [dict.fromkeys(REPORT_COLUMNS, "") | {"Câu trích xuất": NO_FINDINGS}]
    rows.append(dict.fromkeys(REPORT_COLUMNS, "") | {"Câu trích xuất": DISCLAIMER})
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
