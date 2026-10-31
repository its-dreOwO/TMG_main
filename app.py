"""Streamlit demo: streamlit run app.py (served on localhost only)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from lca.pipeline import DEFAULT_DETECTORS, DETECTOR_NAMES, analyze, build_detectors, export
from lca.reader import UnreadableDocumentError
from lca.types import AnalysisContext
from lca.writers import DISCLAIMER, NO_FINDINGS, report_rows

CONTRACT_TYPES = {"Hợp đồng thương mại": "commercial", "Hợp đồng lao động": "labour", "Khác": "other"}
PARTIES = {"Không chọn": None, "Bên A": "Bên A", "Bên B": "Bên B"}
LEVELS = ["Cao", "Trung bình", "Thấp"]

st.set_page_config(page_title="Legal Clause Analyzer", layout="wide")
st.title("Trợ lý nhận diện rủi ro hợp đồng")
st.warning(DISCLAIMER + " Tài liệu được xử lý hoàn toàn trên máy này.")


@st.cache_resource(show_spinner="Đang tải mô hình…")
def detectors_for(names: tuple[str, ...]):
    return build_detectors(list(names))


with st.sidebar:
    contract_type = st.selectbox("Loại hợp đồng", list(CONTRACT_TYPES))
    party = st.radio("Tôi là", list(PARTIES), horizontal=True)
    mode = st.selectbox("Đơn vị phân tích", ["sentence", "clause"],
                        format_func=lambda m: "Câu" if m == "sentence" else "Điều khoản")
    chosen = st.multiselect("Phương pháp", DETECTOR_NAMES, default=DEFAULT_DETECTORS)

upload = st.file_uploader("Tải lên hợp đồng (.docx hoặc .pdf)", type=["docx", "pdf"])
if upload and st.button("Phân tích", type="primary", disabled=not chosen):
    with tempfile.TemporaryDirectory() as tmp:  # nothing is kept on disk afterwards
        src = Path(tmp) / Path(upload.name).name
        src.write_bytes(upload.getvalue())
        try:
            dets = detectors_for(tuple(chosen))
            with st.spinner("Đang phân tích…"):
                result = analyze(src, AnalysisContext(CONTRACT_TYPES[contract_type], PARTIES[party]), dets, mode)
                paths = export(result, Path(tmp) / "out")
            st.session_state["result"] = {
                "rows": report_rows(result.findings),
                "files": {k: (p.name, p.read_bytes()) for k, p in paths.items()},
            }
        except UnreadableDocumentError as exc:
            st.error(str(exc))
            st.session_state.pop("result", None)
        except FileNotFoundError as exc:
            st.error(f"Thiếu mô hình đã huấn luyện: {exc}")

res = st.session_state.get("result")
if res:
    rows = res["rows"]
    if not rows:
        st.success(NO_FINDINGS)
    else:
        counts = pd.Series([r["Mức"] for r in rows]).value_counts()
        cols = st.columns(3)
        for col, level in zip(cols, LEVELS):
            col.metric(f"Mức {level}", int(counts.get(level, 0)))
        shown = st.multiselect("Lọc theo mức", LEVELS, default=LEVELS)
        st.dataframe(pd.DataFrame([r for r in rows if r["Mức"] in shown]), use_container_width=True,
                     hide_index=True)
    name, data = res["files"]["document"]
    st.download_button("Tải hợp đồng đã đánh dấu", data, file_name=name)
    name, data = res["files"]["report"]
    st.download_button("Tải báo cáo (.csv)", data, file_name=name, mime="text/csv")
