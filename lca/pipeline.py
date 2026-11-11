"""End-to-end analysis: read → segment → detect → entities → score → export.

Everything here runs locally; models must already be cached (HF_HUB_OFFLINE=1),
so contract text can never be sent to a remote service.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from lca.detectors.base import Detector  # noqa: E402
from lca.detectors.regex_rules import is_definition  # noqa: E402
from lca.ner import extract_entities  # noqa: E402
from lca.reader import UnreadableDocumentError, read_document  # noqa: E402
from lca.scoring import score  # noqa: E402
from lca.segmenter import segment  # noqa: E402
from lca.types import AnalysisContext, Document, Finding, Segment  # noqa: E402
from lca.writers import write_docx, write_pdf, write_report  # noqa: E402

MODELS = Path(__file__).resolve().parent.parent / "artifacts" / "models"
DETECTOR_NAMES = ["regex", "tfidf", "legal-bert", "xlmr", "legal-bert+translate", "sts"]
DEFAULT_DETECTORS = ["regex", "xlmr", "sts"]


@dataclass(frozen=True)
class AnalysisResult:
    document: Document
    findings: list[Finding]
    segments: list[Segment]


def build_detectors(names: list[str]) -> list[Detector]:
    unknown = [n for n in names if n not in DETECTOR_NAMES]
    if unknown:
        raise ValueError(f"unknown detectors {unknown}; choose from {DETECTOR_NAMES}")
    out: list[Detector] = []
    for name in names:
        if name == "regex":
            from lca.detectors.regex_rules import RegexDetector
            out.append(RegexDetector())
        elif name == "tfidf":
            from lca.detectors.tfidf import TfidfDetector
            out.append(TfidfDetector(MODELS / "tfidf_best.joblib"))
        elif name in ("legal-bert", "xlmr"):
            from lca.detectors.transformer import TransformerDetector
            out.append(TransformerDetector(MODELS / name / "final", name))
        elif name == "legal-bert+translate":
            from lca.detectors.transformer import TransformerDetector
            from lca.detectors.translate import Translator, TranslatingDetector
            inner = TransformerDetector(MODELS / "legal-bert" / "final", "legal-bert")
            out.append(TranslatingDetector(inner, Translator(), name))
        elif name == "sts":
            from lca.detectors.sts import StsDetector, load_prototypes
            thresholds = json.loads((MODELS / "sts_thresholds.json").read_text())
            out.append(StsDetector(load_prototypes(), thresholds))
    return out


def analyze(path: Path, ctx: AnalysisContext, detectors: list[Detector],
            mode: Literal["clause", "sentence"] = "sentence") -> AnalysisResult:
    doc = read_document(Path(path))
    if not doc.blocks:
        raise UnreadableDocumentError(f"Tệp '{Path(path).name}' không có nội dung văn bản.")
    segments = segment(doc, mode)
    texts = [s.text for s in segments]
    merged: list[list] = [[] for _ in segments]
    for det in detectors:
        for i, found in enumerate(det.predict(texts)):
            merged[i].extend(found)
    findings = []
    for seg, dets in zip(segments, merged):
        if dets and not is_definition(seg.text):
            finding = score(seg, dets, extract_entities(seg.text, seg.lang), ctx)
            if finding:
                findings.append(finding)
    return AnalysisResult(doc, findings, segments)


def export(result: AnalysisResult, out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = result.document.path
    doc_out = out_dir / f"{src.stem}_danh_dau.{result.document.kind}"
    if result.document.kind == "docx":
        write_docx(src, result.findings, doc_out)
    else:
        write_pdf(src, result.document, result.findings, doc_out)
    report = out_dir / f"{src.stem}_bao_cao.csv"
    write_report(result.findings, report)
    return {"document": doc_out, "report": report}
