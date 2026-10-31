import csv
import subprocess
import sys

import docx
import pymupdf
import pytest

from lca.cli import main as cli_main
from lca.detectors.regex_rules import RegexDetector
from lca.pipeline import analyze, build_detectors, export
from lca.reader import UnreadableDocumentError
from lca.types import AnalysisContext, Level


def _docx(path, paragraphs):
    d = docx.Document()
    for p in paragraphs:
        d.add_paragraph(p)
    d.save(path)
    return path


@pytest.fixture
def sample_docx(tmp_path):
    return _docx(tmp_path / "hd.docx", [
        "Điều 8. Phạt vi phạm",
        "8.2. Bên B phải chịu phạt 12% giá trị hợp đồng nếu giao hàng chậm.",
        "Điều 9. Hiệu lực",
        "Bên A có quyền đơn phương chấm dứt hợp đồng không cần báo trước.",
    ])


@pytest.fixture
def clean_docx(tmp_path):
    return _docx(tmp_path / "clean.docx", ["Điều 1. Hiệu lực", "Hợp đồng này có hiệu lực từ ngày ký."])


def rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_end_to_end_docx_with_stub_detectors(sample_docx, tmp_path):
    res = analyze(sample_docx, AnalysisContext(), [RegexDetector()])
    paths = export(res, tmp_path / "out")
    assert paths["document"].exists() and paths["document"] != sample_docx
    assert "12%" in paths["report"].read_text("utf-8-sig")


def test_report_columns_match_spec(sample_docx, tmp_path):
    paths = export(analyze(sample_docx, AnalysisContext(), [RegexDetector()]), tmp_path / "out")
    assert list(rows(paths["report"])[0]) == ["Vị trí", "Câu trích xuất", "Loại rủi ro", "Thực thể", "Mức"]


def test_report_sorted_high_first(sample_docx, tmp_path):
    paths = export(analyze(sample_docx, AnalysisContext(), [RegexDetector()]), tmp_path / "out")
    order = {"Cao": 3, "Trung bình": 2, "Thấp": 1}
    levels = [order[r["Mức"]] for r in rows(paths["report"]) if r["Mức"]]
    assert levels == sorted(levels, reverse=True) and levels[0] == 3


def test_report_has_legal_disclaimer(sample_docx, tmp_path):
    paths = export(analyze(sample_docx, AnalysisContext(), [RegexDetector()]), tmp_path / "out")
    assert "không thay thế tư vấn pháp lý" in paths["report"].read_text("utf-8-sig")


def test_no_findings_still_exports_valid_files(clean_docx, tmp_path):
    res = analyze(clean_docx, AnalysisContext(), [RegexDetector()])
    paths = export(res, tmp_path / "out")
    assert res.findings == [] and docx.Document(paths["document"])
    assert "Không phát hiện" in paths["report"].read_text("utf-8-sig")


def test_empty_docx_raises_clear_error(tmp_path):
    empty = _docx(tmp_path / "empty.docx", [])
    with pytest.raises(UnreadableDocumentError):
        analyze(empty, AnalysisContext(), [RegexDetector()])


def test_pdf_end_to_end(tmp_path):
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), "The Vendor shall pay a penalty of 12% of the contract price.")
    src = tmp_path / "c.pdf"
    pdf.save(src)
    res = analyze(src, AnalysisContext(), [RegexDetector()])
    paths = export(res, tmp_path / "out")
    page = pymupdf.open(paths["document"])[0]
    assert res.findings and list(page.annots())


def test_findings_combine_detectors(sample_docx):
    class Second:
        name = "second"

        def predict(self, texts):
            from dataclasses import replace
            return [[replace(d, method="second") for d in ds] for ds in RegexDetector().predict(texts)]

    res = analyze(sample_docx, AnalysisContext(), [RegexDetector(), Second()])
    assert max(f.level for f in res.findings) == Level.HIGH
    assert all({d.method for d in f.detections} == {"regex", "second"} for f in res.findings)


def test_build_detectors_rejects_unknown_name():
    with pytest.raises(ValueError):
        build_detectors(["nope"])


def test_pipeline_does_not_import_modal():
    code = "import lca.pipeline, sys; assert 'modal' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_cli_writes_outputs(sample_docx, tmp_path, capsys):
    code = cli_main([str(sample_docx), "--detectors", "regex", "--out", str(tmp_path / "o")])
    assert code == 0 and "Cao" in capsys.readouterr().out
    assert len(list((tmp_path / "o").iterdir())) == 2


def test_cli_returns_nonzero_with_message_on_scanned_pdf(tmp_path, capsys):
    pdf = pymupdf.open()
    pdf.new_page().draw_rect(pymupdf.Rect(50, 50, 300, 300), fill=(0, 0, 0))
    src = tmp_path / "scan.pdf"
    pdf.save(src)
    assert cli_main([str(src), "--detectors", "regex", "--out", str(tmp_path / "o")]) != 0
    assert "scan" in capsys.readouterr().err


def test_app_shows_disclaimer_before_any_upload():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py").run(timeout=30)
    assert not at.exception
    assert any("không thay thế tư vấn pháp lý" in w.value for w in at.warning)
