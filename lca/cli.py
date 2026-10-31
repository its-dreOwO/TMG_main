"""Command line: python -m lca.cli FILE [--type commercial] [--party "Bên B"] [--detectors regex,xlmr,sts]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lca.pipeline import DEFAULT_DETECTORS, analyze, build_detectors, export
from lca.reader import UnreadableDocumentError
from lca.types import AnalysisContext
from lca.writers import DISCLAIMER, NO_FINDINGS, report_rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lca", description="Legal Clause Analyzer")
    ap.add_argument("file", type=Path)
    ap.add_argument("--type", choices=["commercial", "labour", "other"], default="commercial")
    ap.add_argument("--party", default=None, help='Your side, e.g. "Bên B"')
    ap.add_argument("--mode", choices=["sentence", "clause"], default="sentence")
    ap.add_argument("--detectors", default=",".join(DEFAULT_DETECTORS))
    ap.add_argument("--out", type=Path, default=Path("reports/contracts"))
    args = ap.parse_args(argv)
    try:
        detectors = build_detectors([d.strip() for d in args.detectors.split(",") if d.strip()])
        result = analyze(args.file, AnalysisContext(args.type, args.party), detectors, args.mode)
    except UnreadableDocumentError as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 2
    paths = export(result, args.out)
    rows = report_rows(result.findings)
    for r in rows:
        print(f"[{r['Mức']}] {r['Vị trí']} | {r['Loại rủi ro']} | {r['Câu trích xuất'][:100]}")
    if not rows:
        print(NO_FINDINGS)
    print(f"\nĐã lưu: {paths['document']}\n        {paths['report']}\n{DISCLAIMER}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
