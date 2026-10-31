"""Build-time warm-up: download every lazily fetched resource into the image, then run
one real analysis so nothing needs the network at run time."""
import os
import tempfile
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

import pymupdf  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from lca.detectors.sts import STS_MODEL  # noqa: E402

SentenceTransformer(STS_MODEL)  # caches into $HF_HOME

from lca.ner import extract_entities  # noqa: E402
from lca.pipeline import DEFAULT_DETECTORS, analyze, build_detectors  # noqa: E402
from lca.types import AnalysisContext  # noqa: E402

extract_entities("Công ty TNHH ABC phải bồi thường cho Bên B.", "vi")
extract_entities("Acme Corporation shall indemnify the Client.", "en")
with tempfile.TemporaryDirectory() as tmp:
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), "Ben B phai chiu phat 12% gia tri hop dong.")
    path = Path(tmp) / "x.pdf"
    pdf.save(path)
    res = analyze(path, AnalysisContext(), build_detectors(DEFAULT_DETECTORS))
    print("prefetch ok, findings:", len(res.findings))
