# Legal Clause Analyzer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local prototype that takes a Word/PDF contract, flags risky sentences/clauses, and outputs the original file with them highlighted red plus a summary report.

**Architecture:** A six-stage Python pipeline (read → segment → detect → extract entities → score → write). Three detectors (regex, classifier, STS) share one interface and are combined recall-first. GPU training runs on Modal against public data only; all contract processing runs locally.

**Tech Stack:** Python 3.11, scikit-learn, transformers, sentence-transformers, spaCy, underthesea, python-docx, PyMuPDF, Streamlit, Modal, pytest.

**Spec:** `Proposal_Legal_Clause_Analyzer.docx`

## Global Constraints

- All commands run from `/opt/study/TMG/project`. Interpreter: `/opt/study/.venv/bin/python` (shared venv, Python 3.11; also reachable through the `/home/dre/Desktop/study` symlink).
- Do not upgrade or downgrade packages other projects use in the shared venv.
- GPU training runs on Modal (profile `its-dreowo`), never locally.
- Only `data/processed/` (public CUAD/LEDGAR) may be uploaded to Modal. Contracts run fully on-premise; no contract text may leave the machine, including for translation.
- Risk groups are exactly: `penalty`, `indemnity`, `confidentiality` (required); `termination`, `non_compete` (extended); plus `none`.
- Combination is recall-first: a segment is flagged if at least one method fires with sufficient confidence.
- Levels are Cao / Trung bình / Thấp (`HIGH` / `MEDIUM` / `LOW`).
- PDF scope is text PDFs only. OCR is out of scope.
- The UI and report must state that the tool does not replace legal advice ("không thay thế tư vấn pháp lý").
- Random seed is `42` everywhere.
- `data/`, `artifacts/` and `reports/contracts/` stay out of git.

## Human Prerequisites

Group members must supply these before Task 13 can finish; code tasks do not depend on them:

1. Two real sample contracts in `data/contracts/`: one labour contract and one sales/service contract or NDA.
2. Independent sentence-level labels from at least two members, using the template from Task 13.

## Review Focus

1. **Scanned, corrupt or password-protected file:** the user gets a clear error, never an empty "no risk" report. Pinned in Task 2.
2. **Vietnamese text in decomposed Unicode (NFD), common in PDFs:** patterns still match and highlight offsets stay correct. Pinned in Task 3.
3. **Clause longer than the model's `max_length`:** a risk phrase at the end is still detected. Pinned in Task 8.
4. **Sentence spanning several Word runs or sitting in a table cell:** the whole sentence is highlighted and other formatting survives. Pinned in Task 11.
5. **Contract with no findings:** valid output files and an explicit "no risk found" message, not a crash or blank table. Pinned in Task 12.

---

## File Structure

```
lca/
  types.py            shared dataclasses and enums
  reader.py           .docx/.pdf → Document
  segmenter.py        Document → Segments (clause or sentence mode)
  detectors/
    base.py           Detector protocol
    regex_rules.py    baseline
    tfidf.py          scikit-learn classifier
    transformer.py    local inference, sliding window
    translate.py      local vi→en wrapper
    sts.py            prototype similarity
  resources/
    prototypes.yaml   reference clauses (VI + EN)
  data/
    mapping.py        CUAD/LEDGAR label → RiskLabel
    prepare.py        build data/processed/*.parquet
  train/
    tfidf_train.py
    core.py           class weights, metrics for the HF Trainer
  ner.py              entities
  scoring.py          fusion and risk level
  writers.py          highlighted docx/pdf, CSV report
  pipeline.py         analyze()
  cli.py
  eval/
    metrics.py        threshold tuning, P/R/F1, kappa
    annotate.py       templates and consensus
    compare.py        method comparison, RQ1–RQ3, error analysis
modal_train.py        Modal app
app.py                Streamlit demo
tests/
```

---

### Task 1: Environment, project skeleton and shared types

**Files:**
- Create: `.gitignore`, `pyproject.toml`, `lca/__init__.py`, `lca/types.py`, `.streamlit/config.toml`
- Modify: `/opt/study/requirements-global.txt`
- Test: `tests/test_types.py`

**Interfaces:**
- Produces, in `lca/types.py`:
  - `RiskLabel(str, Enum)`: `PENALTY="penalty"`, `INDEMNITY="indemnity"`, `CONFIDENTIALITY="confidentiality"`, `TERMINATION="termination"`, `NON_COMPETE="non_compete"`, `NONE="none"`
  - `Level(IntEnum)`: `LOW=1`, `MEDIUM=2`, `HIGH=3`, with property `vi` returning `"Thấp"`, `"Trung bình"`, `"Cao"`
  - `Block(index: int, text: str, page: int | None)`
  - `Document(path: Path, kind: Literal["docx","pdf"], blocks: list[Block])`
  - `Segment(seg_id: int, text: str, location: str, spans: list[tuple[int,int,int]], lang: Literal["vi","en"])`. Each span is `(block_index, start, end)` into the raw `Block.text`; `text` is normalised and is used for detection only.
  - `Detection(method: str, label: RiskLabel, score: float)`
  - `Entity(kind: Literal["PARTY","MONEY","PERCENT","DURATION","DATE"], text: str, value: float | None)`. PERCENT value is the percentage; DURATION value is months (`math.inf` for indefinite); MONEY value is the number.
  - `Finding(segment: Segment, label: RiskLabel, detections: list[Detection], entities: list[Entity], level: Level, reason: str)`
  - `AnalysisContext(contract_type: Literal["commercial","labour","other"]="commercial", my_party: str | None=None)`

- [ ] **Step 1: Check the new dependencies against the shared venv**

Run: `uv pip install --dry-run --python /opt/study/.venv/bin/python sentence-transformers spacy underthesea pymupdf "python-docx>=1.2" sentencepiece sacremoses`
Expected: the output shows no change to `torch`, `transformers`, `numpy`, `pandas`, `scikit-learn`. If one would change, stop and ask the user.

- [ ] **Step 2: Install them and record them**

Run the same command without `--dry-run`, then append the seven packages to `/opt/study/requirements-global.txt`. Install the spaCy English model with `uv pip install` from the `en_core_web_sm` release wheel matching the installed spaCy version (the venv may have no `pip` for `spacy download`).

- [ ] **Step 3: Initialise the project**

Run `git init` in the project. `.gitignore` must cover `data/`, `artifacts/`, `reports/contracts/`, `__pycache__/`, `.pytest_cache/`. `pyproject.toml` sets `[tool.pytest.ini_options] pythonpath=["."]`, `testpaths=["tests"]`. `.streamlit/config.toml` sets `server.address="localhost"` and `browser.gatherUsageStats=false`.

- [ ] **Step 4: Write the failing test**

```python
def test_level_vietnamese_names():
    assert Level.HIGH.vi == "Cao" and Level.MEDIUM.vi == "Trung bình" and Level.LOW.vi == "Thấp"

def test_levels_are_ordered():
    assert Level.LOW < Level.MEDIUM < Level.HIGH

def test_risk_label_set():
    assert {l.value for l in RiskLabel} == {"penalty","indemnity","confidentiality","termination","non_compete","none"}
```

- [ ] **Step 5: Run it to verify it fails**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_types.py -v`
Expected: FAIL with an import error.

- [ ] **Step 6: Implement `lca/types.py` as listed above, using frozen dataclasses**

- [ ] **Step 7: Run it to verify it passes**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_types.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add .gitignore pyproject.toml .streamlit lca tests docs
git commit -m "chore: project skeleton and shared types"
```

---

### Task 2: Document reader

**Files:**
- Create: `lca/reader.py`
- Test: `tests/test_reader.py` (builds fixtures in `tmp_path` with python-docx and PyMuPDF)

**Interfaces:**
- Consumes: `Document`, `Block`
- Produces:
  - `read_document(path: Path) -> Document`
  - `iter_docx_paragraphs(doc: docx.Document) -> Iterator[Paragraph]`: body paragraphs and table-cell paragraphs in document order. `Block.index` is the position in this iterator, and Task 11 relies on that.
  - `class UnreadableDocumentError(Exception)`: its message is shown to the end user.

- [ ] **Step 1: Write the failing tests**

```python
def test_docx_includes_table_cells_in_order(tmp_path):
    # paragraph "A", table cell "B", paragraph "C"
    assert [b.text for b in read_document(path).blocks] == ["A", "B", "C"]

def test_pdf_blocks_have_pages(tmp_path):
    assert read_document(path).blocks[0].page == 0

def test_pdf_repeated_header_removed(tmp_path):
    # same "CONFIDENTIAL DRAFT" line on all 3 pages
    assert all("CONFIDENTIAL DRAFT" not in b.text for b in read_document(path).blocks)

def test_image_only_pdf_raises(tmp_path):
    with pytest.raises(UnreadableDocumentError, match="scan"): read_document(path)

def test_corrupt_file_raises(tmp_path):
    with pytest.raises(UnreadableDocumentError): read_document(path)

def test_encrypted_pdf_raises(tmp_path):
    with pytest.raises(UnreadableDocumentError, match="mật khẩu"): read_document(path)

def test_unsupported_extension_raises(tmp_path):
    with pytest.raises(UnreadableDocumentError): read_document(tmp_path / "a.txt")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_reader.py -v`
Expected: FAIL with an import error.

- [ ] **Step 3: Implement `read_document`**

For PDF use `page.get_text("blocks")`. Treat a PDF as scanned when it averages under 20 extracted characters per page. Remove a line as header/footer when it appears on at least 60% of pages in a document with 3+ pages. Skip empty paragraphs in text output but still count them in `Block.index`.

- [ ] **Step 4: Run them to verify they pass**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_reader.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lca/reader.py tests/test_reader.py
git commit -m "feat: read docx and text pdf with clear failure errors"
```

---

### Task 3: Segmenter

**Files:**
- Create: `lca/segmenter.py`
- Test: `tests/test_segmenter.py`

**Interfaces:**
- Consumes: `Document`, `Segment`
- Produces:
  - `normalize(text: str) -> str`: NFC, collapse whitespace
  - `detect_lang(text: str) -> Literal["vi","en"]`: `vi` when Vietnamese-specific letters (after NFC) are at least 3% of alphabetic characters
  - `segment(doc: Document, mode: Literal["clause","sentence"]) -> list[Segment]`

- [ ] **Step 1: Write the failing tests**

```python
VI = ["Điều 8. Phạt vi phạm", "8.2. Bên B phải chịu phạt 12% giá trị hợp đồng nếu giao hàng chậm.", "Điều 9. Bồi thường"]

def test_clause_mode_groups_until_next_heading():
    segs = segment(doc(VI), "clause")
    assert len(segs) == 2 and "12%" in segs[0].text and segs[0].location == "Điều 8"

def test_clause_spans_cover_multiple_blocks():
    assert [s[0] for s in segment(doc(VI), "clause")[0].spans] == [0, 1]

def test_sentence_location_uses_subclause():
    assert any(s.location == "Điều 8.2" for s in segment(doc(VI), "sentence"))

def test_english_article_heading():
    assert segment(doc(["Article 7 Indemnification", "Vendor shall indemnify."]), "clause")[0].location == "Article 7"

def test_enumerated_items_split():
    segs = segment(doc(["Bên B phải: a) trả phí; b) bồi thường; c) bảo mật."]), "sentence")
    assert len(segs) == 3

def test_abbreviation_does_not_split():
    assert len(segment(doc(["See Sec. 4 of the Agreement between Acme Inc. and Beta."]), "sentence")) == 1

def test_nfd_input_normalised_with_raw_offsets():
    raw = unicodedata.normalize("NFD", "Bên B bồi thường toàn bộ.")
    s = segment(doc([raw]), "sentence")[0]
    assert "bồi thường" in s.text
    b, start, end = s.spans[0]
    assert raw[start:end].strip().rstrip(".") == raw.rstrip(".")

def test_segment_language_detected():
    assert segment(doc(VI), "sentence")[0].lang == "vi"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_segmenter.py -v`
Expected: FAIL with an import error.

- [ ] **Step 3: Implement `segment`**

Compute all offsets on raw block text and normalise only `Segment.text`. Heading pattern: `^(Điều|Khoản|Article|Section|Clause)\s+\d+(\.\d+)*` (case-insensitive). Sub-clause pattern: `^\d+(\.\d+)+\.?\s`. Text before the first heading has location `"Mở đầu"`. Sentence mode splits structure first (`;`, `a)`/`b)` markers), then uses `underthesea.sent_tokenize` for `vi` and spaCy's `en_core_web_sm` sentencizer for `en`, locating results in the raw block with sequential `str.find`. Keep the enumerator lead-in as context only if the split item would otherwise lack a subject: append nothing to spans, but prefix the lead-in to `Segment.text`.

- [ ] **Step 4: Run them to verify they pass**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_segmenter.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lca/segmenter.py tests/test_segmenter.py
git commit -m "feat: structure-aware clause and sentence segmentation"
```

---

### Task 4: Regex baseline detector

**Files:**
- Create: `lca/detectors/__init__.py`, `lca/detectors/base.py`, `lca/detectors/regex_rules.py`
- Test: `tests/test_regex_rules.py`

**Interfaces:**
- Consumes: `Detection`, `RiskLabel`
- Produces:
  - `class Detector(Protocol)`: `name: str`; `predict(texts: list[str]) -> list[list[Detection]]`. Output has one inner list per input and never contains `RiskLabel.NONE`. All later detectors implement this.
  - `class RegexDetector(Detector)` with `name="regex"`. Contextual pattern match scores `1.0`, keyword-only match scores `0.6`.

Patterns must include the spec's three verbatim (case-insensitive):
- `(phạt|tiền phạt).{0,40}(\d+\s?%|giá trị hợp đồng)`
- `bồi thường.{0,60}(toàn bộ|mọi|không giới hạn)`
- `(bảo mật|thông tin mật|không được tiết lộ).{0,80}(\d+\s?(năm|tháng)|vô thời hạn)`

Keywords come from spec Table 1 for all five groups, in both languages.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("text,label,score", [
    ("Bên B phải chịu phạt 12% giá trị hợp đồng", RiskLabel.PENALTY, 1.0),
    ("Bên B bồi thường toàn bộ thiệt hại phát sinh", RiskLabel.INDEMNITY, 1.0),
    ("Nghĩa vụ bảo mật thông tin trong 02 năm", RiskLabel.CONFIDENTIALITY, 1.0),
    ("Vendor shall indemnify and hold harmless Client", RiskLabel.INDEMNITY, 0.6),
    ("Company may terminate for convenience at any time", RiskLabel.TERMINATION, 0.6),
    ("Employee shall not engage in competing business (non-compete)", RiskLabel.NON_COMPETE, 0.6),
    ("Bên A được đơn phương chấm dứt hợp đồng không cần báo trước", RiskLabel.TERMINATION, 0.6),
    ("Liquidated damages of $5,000 per day", RiskLabel.PENALTY, 0.6),
])
def test_positive(text, label, score): ...

@pytest.mark.parametrize("text", [
    "Bên B không bị phạt trong trường hợp bất khả kháng",
    "Thông tin mật là mọi thông tin được đánh dấu mật",
    '"Confidential Information" means any non-public information',
    "Hợp đồng này có hiệu lực từ ngày ký",
])
def test_negated_definition_and_plain_text_not_flagged(text):
    assert RegexDetector().predict([text]) == [[]]

def test_one_output_per_input():
    assert len(RegexDetector().predict(["a", "b"])) == 2
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_regex_rules.py -v`
Expected: FAIL with an import error.

- [ ] **Step 3: Implement the detector**

Check negation only within the 15 characters before a match (`không bị`, `miễn`, `không phải chịu`, `not be subject to`). Skip definition sentences starting with an optionally quoted term followed by `là`, `có nghĩa là` or `means`.

- [ ] **Step 4: Run them to verify they pass**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_regex_rules.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lca/detectors tests/test_regex_rules.py
git commit -m "feat: regex baseline detector with negation and definition rules"
```

---

### Task 5: Label mapping and dataset preparation

**Files:**
- Create: `lca/data/__init__.py`, `lca/data/mapping.py`, `lca/data/prepare.py`
- Test: `tests/test_mapping.py`, `tests/test_prepare.py`

**Interfaces:**
- Consumes: `RiskLabel`, `segmenter.normalize`
- Produces:
  - `CUAD_MAP: dict[str, RiskLabel]`
  - `map_ledgar_label(label: str) -> RiskLabel | None`: `None` means uncertain, so the row is dropped
  - `parquets data/processed/{train,val,test}.parquet` with columns `text: str`, `label: str`, `source: "cuad"|"ledgar"`, `group: str`
  - `data/processed/stats.json`: per-split, per-label counts and rows dropped by each rule
  - `prepare(raw_dir: Path, out_dir: Path) -> dict`; CLI `python -m lca.data.prepare`

`CUAD_MAP`: Liquidated Damages → penalty; Uncapped Liability, Cap On Liability → indemnity; Non-Compete, No-Solicit Of Employees, No-Solicit Of Customers → non_compete; Termination For Convenience → termination.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("raw,expected", [
    ("indemnifications", RiskLabel.INDEMNITY),
    ("confidential information", RiskLabel.CONFIDENTIALITY),
    ("termination for cause", RiskLabel.TERMINATION),
    ("noncompetition", RiskLabel.NON_COMPETE),
    ("non-solicitation of employees", RiskLabel.NON_COMPETE),
    ("liquidated damages", RiskLabel.PENALTY),
    ("governing laws", RiskLabel.NONE),
    ("no premium or penalty", None),
])
def test_ledgar_mapping(raw, expected): ...

def test_cuad_negative_excludes_unlabelled_risk_topics(fixture_cuad):
    # CUAD has no indemnity/confidentiality labels, so such unlabelled sentences must not become "none"
    out = prepare(...)
    none_texts = out_frame.query("label=='none' and source=='cuad'").text
    assert not none_texts.str.contains("indemnif|confidential", case=False).any()

def test_no_group_in_two_splits(prepared):
    assert not set(train.group) & set(test.group) and not set(train.group) & set(val.group)

def test_no_duplicate_text_across_splits(prepared):
    assert not set(train.text) & set(test.text)

def test_conflicting_risk_labels_dropped(fixture_ledgar):
    # row labelled ["indemnity", "confidentiality"]
    assert stats["dropped_multi_risk"] == 1

def test_none_capped_at_three_times_positives(prepared):
    assert (all_rows.label == "none").sum() <= 3 * (all_rows.label != "none").sum()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_mapping.py tests/test_prepare.py -v`
Expected: FAIL with an import error.

- [ ] **Step 3: Implement the mapping and preparation**

CUAD: `CUAD_v1.json` question names come from `related to "(.+?)"`. Positives are the answer spans. Negatives are sentences from each `context` that overlap no mapped answer and do not match `indemnif|hold harmless|confidential|non-disclosure|penalt`.

LEDGAR: stream the JSONL line by line (the machine has 14 GB RAM). Keep all positives. Hold negatives in a seeded reservoir of 200,000. For the group, use the first three `/`-separated parts of `source` (the filing). Drop rows with two or more distinct risk labels.

Split by group with `GroupShuffleSplit` 80/10/10, seed 42, then drop from val/test any normalised text present in train. Keep Legal-Clause-Dataset out: its overlap with LEDGAR is unknown and could leak across splits.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_mapping.py tests/test_prepare.py -v`
Expected: PASS.

- [ ] **Step 5: Build the real data and inspect it**

Run: `/opt/study/.venv/bin/python -m lca.data.prepare`
Expected: `data/processed/stats.json` exists; every risk label has at least one row in val and test; `penalty` has at least 300 rows overall.

- [ ] **Step 6: Commit**

```bash
git add lca/data tests/test_mapping.py tests/test_prepare.py
git commit -m "feat: map CUAD and LEDGAR to risk groups with leak-free splits"
```

---

### Task 6: Metrics and TF-IDF classifiers

**Files:**
- Create: `lca/eval/__init__.py`, `lca/eval/metrics.py`, `lca/train/__init__.py`, `lca/train/tfidf_train.py`, `lca/detectors/tfidf.py`
- Test: `tests/test_metrics.py`, `tests/test_tfidf.py`

**Interfaces:**
- Consumes: `Detector`, parquet schema from Task 5
- Produces:
  - `tune_thresholds(y_true: Sequence[str], proba: np.ndarray, classes: list[str]) -> dict[str, float]`: per-risk-class threshold maximising F2 (recall-first), searching `0.05..0.95` in steps of `0.05`
  - `report(y_true, y_pred, labels) -> dict`: per-label and macro P/R/F1 plus confusion matrix
  - `apply_thresholds(proba, classes, thresholds) -> list[list[tuple[str, float]]]`: all risk classes meeting their threshold
  - `class TfidfDetector(Detector)` with `name="tfidf"`: `__init__(path: Path)`
  - `artifacts/models/tfidf_lr.joblib` and `tfidf_svm.joblib`, each storing `{"pipeline", "classes", "thresholds"}`
  - `reports/tfidf_metrics.json`

- [ ] **Step 1: Write the failing tests**

```python
def test_threshold_favours_recall():
    # one positive at p=0.3, negatives at p<=0.1
    t = tune_thresholds(["a","none","none"], np.array([[.3,.7],[.1,.9],[.05,.95]]), ["a","none"])
    assert t["a"] <= 0.3

def test_apply_thresholds_can_return_several_and_none():
    out = apply_thresholds(np.array([[.6,.5,.1]]), ["a","b","none"], {"a":.5,"b":.4})
    assert {c for c,_ in out[0]} == {"a","b"}

def test_tfidf_detector_never_returns_none(tiny_model_path):
    assert all(d.label != RiskLabel.NONE for ds in TfidfDetector(tiny_model_path).predict(["x"]) for d in ds)

def test_tfidf_detects_trained_phrase(tiny_model_path):
    assert TfidfDetector(tiny_model_path).predict(["shall indemnify the buyer"])[0][0].label == RiskLabel.INDEMNITY
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_metrics.py tests/test_tfidf.py -v`
Expected: FAIL with an import error.

- [ ] **Step 3: Implement metrics, training and detector**

Train `TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True)` with `LogisticRegression(class_weight="balanced", max_iter=1000)` and with `CalibratedClassifierCV(LinearSVC(class_weight="balanced"))`. Tune thresholds on val, report on test. `TfidfDetector` loads whichever file it is given; default to the model with the higher test macro F1.

- [ ] **Step 4: Run tests, then train**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_metrics.py tests/test_tfidf.py -v`
Expected: PASS.

Run: `/opt/study/.venv/bin/python -m lca.train.tfidf_train`
Expected: both joblib files and `reports/tfidf_metrics.json` exist.

- [ ] **Step 5: Commit**

```bash
git add lca/eval lca/train lca/detectors/tfidf.py tests reports/tfidf_metrics.json
git commit -m "feat: TF-IDF classifiers with recall-first thresholds"
```

---

### Task 7: Transformer training on Modal

**Files:**
- Create: `lca/train/core.py`, `modal_train.py`
- Test: `tests/test_train_core.py`

**Interfaces:**
- Consumes: parquet schema from Task 5, `tune_thresholds`
- Produces:
  - `class_weights(labels: list[str], classes: list[str]) -> list[float]`: inverse frequency, `sum/len` normalised to mean 1
  - `WeightedTrainer(transformers.Trainer)`: overrides `compute_loss` with weighted cross-entropy
  - Modal app `legal-clause-train` with `train(model_name: str, run_name: str, epochs: int=3, lr: float=2e-5, batch_size: int=32, max_length: int=256, max_samples: int|None=None) -> dict`
  - On volume `legal-clause-models` at `/<run_name>/final/`: HF model + tokenizer + `thresholds.json` + `metrics.json`
  - Locally after download: `artifacts/models/<run_name>/final/`

Modal configuration:
- Volumes: `legal-clause-data` at `/data`, `legal-clause-models` at `/models`, both with `create_if_missing=True`.
- Image: `modal.Image.debian_slim(python_version="3.11").uv_pip_install(...)` pinning `torch==2.12.0`, `transformers==4.57.6` to match the local venv, plus `accelerate`, `sentencepiece`, `scikit-learn`, `pandas`, `pyarrow`, then `.add_local_python_source("lca")`.
- Function: `gpu="T4"`, `timeout=4*3600`, `retries=2`, `HF_HOME=/models/hf`.
- Runs: `legal-bert` = `nlpaueb/legal-bert-base-uncased`; `xlmr` = `xlm-roberta-base`.

- [ ] **Step 1: Write the failing tests**

```python
def test_class_weights_boost_rare_class():
    w = class_weights(["a","a","a","b"], ["a","b"])
    assert w[1] > w[0] and pytest.approx(sum(w)/2) == 1.0

def test_class_weights_handles_absent_class():
    assert class_weights(["a"], ["a","b"])[1] == 0.0
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_train_core.py -v`
Expected: FAIL with an import error.

- [ ] **Step 3: Implement `lca/train/core.py` and `modal_train.py`**

The Trainer evaluates and saves every epoch to `/models/<run_name>/ckpt`, with `load_best_model_at_end=True`, `metric_for_best_model="eval_macro_f1"`, `fp16=True`. A `TrainerCallback.on_save` calls `volume.commit()`. On start, resume from the last checkpoint if one exists, because retries or preemption can restart the function. After training, compute val probabilities, tune thresholds, score test, write the `final/` directory, then commit. Import torch and transformers inside the function so a local `modal run` needs no GPU.

- [ ] **Step 4: Run the unit tests**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_train_core.py -v`
Expected: PASS.

- [ ] **Step 5: Upload public data only**

Run: `modal volume create legal-clause-data` (ignore "already exists"), then `modal volume put legal-clause-data data/processed /processed`.
Expected: `modal volume ls legal-clause-data /processed` lists the three parquet files.

- [ ] **Step 6: Smoke-test on Modal**

Run: `/opt/study/.venv/bin/modal run modal_train.py::train --model-name xlm-roberta-base --run-name smoke --epochs 1 --max-samples 500`
Expected: exits 0; `modal volume ls legal-clause-models /smoke/final` shows model weights and `thresholds.json`. Note the runtime and extrapolate the full cost; if a full run looks over about 2 hours, reduce before continuing.

- [ ] **Step 7: Run both full trainings detached**

Run:
```bash
/opt/study/.venv/bin/modal run --detach modal_train.py::train --model-name nlpaueb/legal-bert-base-uncased --run-name legal-bert
/opt/study/.venv/bin/modal run --detach modal_train.py::train --model-name xlm-roberta-base --run-name xlmr
```
Expected: both finish; `metrics.json` shows a test macro F1 above the best TF-IDF model. If not, record that honestly in the final report rather than retraining indefinitely.

- [ ] **Step 8: Download the models**

Run: `mkdir -p artifacts/models && modal volume get legal-clause-models /legal-bert/final artifacts/models/legal-bert/ && modal volume get legal-clause-models /xlmr/final artifacts/models/xlmr/`
Expected: `artifacts/models/{legal-bert,xlmr}/final/thresholds.json` exist locally.

- [ ] **Step 9: Commit**

```bash
git add lca/train/core.py modal_train.py tests/test_train_core.py
git commit -m "feat: train transformer classifiers on Modal"
```

---

### Task 8: Local transformer inference and Vietnamese routes

**Files:**
- Create: `lca/detectors/transformer.py`, `lca/detectors/translate.py`
- Test: `tests/test_transformer.py`

**Interfaces:**
- Consumes: `Detector`, `apply_thresholds`, models from Task 7
- Produces:
  - `class TransformerDetector(Detector)`: `__init__(model_dir: Path, name: str, device: str|None=None, max_length: int=256, stride: int=64, batch_size: int=8)`
  - `class Translator`: `translate(texts: list[str]) -> list[str]` using `Helsinki-NLP/opus-mt-vi-en` locally
  - `class TranslatingDetector(Detector)`: `__init__(inner: Detector, translator: Translator, name: str)`; translates Vietnamese inputs only and passes English through

These implement the proposal's two Vietnamese routes: (a) `xlmr` directly, (b) translate + `legal-bert`. Task 13 compares them.

- [ ] **Step 1: Write the failing tests**

Use a stub model/tokenizer returning a high `indemnity` score only when a marker token is in the window.

```python
def test_long_text_risk_at_end_detected(stub_detector):
    text = "filler " * 1000 + "MARKER"
    assert stub_detector.predict([text])[0][0].label == RiskLabel.INDEMNITY

def test_order_preserved_with_batches(stub_detector):
    out = stub_detector.predict(["plain", "MARKER", "plain"])
    assert [bool(x) for x in out] == [False, True, False]

def test_translating_detector_only_translates_vietnamese():
    spy = SpyTranslator()
    TranslatingDetector(EchoDetector(), spy, "t").predict(["Bên B bồi thường", "The Vendor pays"])
    assert spy.seen == ["Bên B bồi thường"]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_transformer.py -v`
Expected: FAIL with an import error.

- [ ] **Step 3: Implement both modules**

Tokenise with `return_overflowing_tokens=True`, `stride=stride`. Take the per-class maximum across a text's windows. Device defaults to `"cuda"` if available, else `"cpu"`. Use `torch.inference_mode()`.

- [ ] **Step 4: Run them to verify they pass**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_transformer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lca/detectors/transformer.py lca/detectors/translate.py tests/test_transformer.py
git commit -m "feat: local sliding-window inference and Vietnamese routes"
```

---

### Task 9: Semantic similarity detector

**Files:**
- Create: `lca/detectors/sts.py`, `lca/resources/prototypes.yaml`
- Test: `tests/test_sts.py`

**Interfaces:**
- Consumes: `Detector`, `tune_thresholds`
- Produces:
  - `class StsDetector(Detector)` with `name="sts"`: `__init__(prototypes: dict[str, list[str]], thresholds: dict[str, float], model: str="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")`
  - `load_prototypes(path: Path) -> dict[str, list[str]]`
  - `tune_sts(detector_model, prototypes, val_frame) -> dict[str, float]`, saved to `artifacts/models/sts_thresholds.json` by `python -m lca.detectors.sts`

`prototypes.yaml` holds at least 10 clauses per risk group, half Vietnamese and half English, drawn from Vietnamese contract practice. It must include "Bên vi phạm phải chịu phạt 8% giá trị phần nghĩa vụ bị vi phạm".

- [ ] **Step 1: Write the failing tests**

Use a fixed fake encoder so these run offline, plus one test marked `slow` with the real model.

```python
def test_flags_above_threshold(fake_encoder): ...
def test_silent_below_threshold(fake_encoder): ...

def test_prototypes_cover_all_groups_in_both_languages():
    p = load_prototypes(PATH)
    assert set(p) == {"penalty","indemnity","confidentiality","termination","non_compete"}
    assert all(len(v) >= 10 and any(detect_lang(t) == "vi" for t in v) and any(detect_lang(t) == "en" for t in v) for v in p.values())

@pytest.mark.slow
def test_paraphrase_without_keyword_matches_real_model():
    # no "phạt" or "bồi thường" keyword
    out = detector.predict(["Bên B phải trả thêm một khoản bằng một phần mười giá trị đơn hàng khi giao trễ"])
    assert out[0]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_sts.py -v`
Expected: FAIL with an import error.

- [ ] **Step 3: Implement the detector**

Score per class is the maximum cosine similarity to that class's prototypes. Encode prototypes once in the constructor.

- [ ] **Step 4: Run tests, then tune thresholds**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_sts.py -v`
Expected: PASS, including the `slow` test. If the paraphrase test fails, add prototypes rather than loosening the assertion.

Run: `/opt/study/.venv/bin/python -m lca.detectors.sts`
Expected: `artifacts/models/sts_thresholds.json` has one value per risk group.

- [ ] **Step 5: Commit**

```bash
git add lca/detectors/sts.py lca/resources tests/test_sts.py
git commit -m "feat: multilingual semantic similarity detector"
```

---

### Task 10: Entity extraction and risk scoring

**Files:**
- Create: `lca/ner.py`, `lca/scoring.py`
- Test: `tests/test_ner.py`, `tests/test_scoring.py`

**Interfaces:**
- Consumes: `Entity`, `Detection`, `Segment`, `Finding`, `Level`, `AnalysisContext`
- Produces:
  - `extract_entities(text: str, lang: str) -> list[Entity]`
  - `obligor(text: str, entities: list[Entity]) -> str | None`: the PARTY nearest before an obligation word (`phải`, `chịu`, `có nghĩa vụ`, `shall`, `must`), else `None`
  - `score(segment: Segment, detections: list[Detection], entities: list[Entity], ctx: AnalysisContext) -> Finding | None`

Level rules, applied in order:
1. No detections → `None`.
2. Label = the label with the highest summed detection score.
3. Base level: `LOW` for one method, `MEDIUM` for two, `HIGH` for three.
4. Raise to `HIGH` when any applies: `penalty`, `ctx.contract_type=="commercial"`, and a PERCENT above 8 (Art. 301 Luật Thương mại 2005); `indemnity` with `toàn bộ thiệt hại|không giới hạn|uncapped|unlimited|without limit`; `confidentiality` with an infinite DURATION.
5. Raise to at least `MEDIUM`: `confidentiality` with a finite DURATION; `penalty` with any PERCENT.
6. When `ctx.my_party` is set and `obligor` returns a different party, lower one level, minimum `LOW`.

`reason` is one Vietnamese sentence naming the entities, e.g. "Phạt 12% giá trị hợp đồng, Bên B chịu".

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("text,kind,value", [
    ("phạt 12% giá trị hợp đồng", "PERCENT", 12.0),
    ("phạt 8,5 % giá trị", "PERCENT", 8.5),
    ("tám phần trăm giá trị", "PERCENT", 8.0),
    ("500.000.000 đồng", "MONEY", 5e8),
    ("1,5 tỷ đồng", "MONEY", 1.5e9),
    ("$1,000,000", "MONEY", 1e6),
    ("trong 02 năm", "DURATION", 24),
    ("30 ngày", "DURATION", 1),
    ("vô thời hạn", "DURATION", math.inf),
    ("for two (2) years", "DURATION", 24),
])
def test_values(text, kind, value): ...

def test_vietnamese_date_and_party():
    kinds = {e.kind for e in extract_entities("Bên B thanh toán trước ngày 15/03/2026", "vi")}
    assert {"PARTY","DATE"} <= kinds

@pytest.mark.parametrize("pct,level", [(8, Level.MEDIUM), (8.5, Level.HIGH), (12, Level.HIGH)])
def test_commercial_penalty_boundary(pct, level): ...

def test_8_percent_rule_not_applied_to_labour_contract(): ...
def test_uncapped_indemnity_is_high(): ...
def test_two_year_confidentiality_is_medium(): ...
def test_indefinite_confidentiality_is_high(): ...
def test_other_party_obligation_lowered(): ...
def test_my_party_obligation_not_lowered(): ...
def test_unknown_obligor_not_lowered():  # "Bên A có quyền phạt Bên B"
    ...
def test_reason_names_entities():
    assert "12%" in finding.reason and "Bên B" in finding.reason
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_ner.py tests/test_scoring.py -v`
Expected: FAIL with an import error.

- [ ] **Step 3: Implement both modules**

Use regex for MONEY, PERCENT, DURATION, DATE and `Bên [A-Z]` / `Party [A-Z]` / `Người lao động` / `Người sử dụng lao động` in both languages. Add ORG from spaCy `en_core_web_sm` (en) and `underthesea.ner` (vi) as PARTY. Parse `.` as thousands separator and `,` as decimal in Vietnamese money and percents; the reverse in English. Convert days to months with `round(days/30)`, minimum 1.

- [ ] **Step 4: Run them to verify they pass**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_ner.py tests/test_scoring.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lca/ner.py lca/scoring.py tests/test_ner.py tests/test_scoring.py
git commit -m "feat: entity extraction and rule-based risk levels"
```

---

### Task 11: Highlighted Word and PDF output

**Files:**
- Create: `lca/writers.py`
- Test: `tests/test_writers.py`

**Interfaces:**
- Consumes: `Finding`, `Document`, `iter_docx_paragraphs`
- Produces:
  - `write_docx(source: Path, findings: list[Finding], out: Path) -> None`
  - `write_pdf(source: Path, doc: Document, findings: list[Finding], out: Path) -> None`
  - `write_report(findings: list[Finding], out: Path) -> None`: added in Task 12 (same file)

- [ ] **Step 1: Write the failing tests**

```python
def test_docx_sentence_across_runs_fully_highlighted(tmp_path):
    # "Bên B " + bold "bồi thường" + " toàn bộ." as three runs
    highlighted = "".join(r.text for r in para(out).runs if r.font.highlight_color == WD_COLOR_INDEX.RED)
    assert highlighted == "Bên B bồi thường toàn bộ."

def test_docx_partial_highlight_splits_run_keeps_bold(tmp_path): ...

def test_docx_table_cell_highlighted(tmp_path): ...

def test_docx_comment_names_risk_and_level(tmp_path):
    assert "Phạt vi phạm" in comment_text(out) and "Cao" in comment_text(out)

def test_docx_unflagged_text_unchanged(tmp_path): ...

def test_pdf_highlight_annotation_on_right_page(tmp_path):
    assert list(fitz.open(out)[1].annots())[0].type[1] == "Highlight"

def test_pdf_sentence_wrapping_lines_gets_annotation(tmp_path): ...

def test_output_never_overwrites_source(tmp_path):
    with pytest.raises(ValueError): write_docx(src, [], src)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_writers.py -v`
Expected: FAIL with an import error.

- [ ] **Step 3: Implement the writers**

Docx: map each `(block, start, end)` span to run offsets, split runs at the boundaries by copying the run's `rPr`, set `font.highlight_color = WD_COLOR_INDEX.RED`, and add a comment with `Document.add_comment` (python-docx ≥ 1.2) giving the risk name, level and reason. PDF: use `page.search_for` with a text prefix when the full span wraps, restricted to the block's page, add a red highlight annotation, and set its content to the same note.

- [ ] **Step 4: Run them to verify they pass**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_writers.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lca/writers.py tests/test_writers.py
git commit -m "feat: highlight risky text in docx and pdf output"
```

---

### Task 12: Pipeline, report, CLI and Streamlit demo

**Files:**
- Create: `lca/pipeline.py`, `lca/cli.py`, `app.py`
- Modify: `lca/writers.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: all earlier modules
- Produces:
  - `build_detectors(names: list[str]) -> list[Detector]`, names from `regex, tfidf, legal-bert, xlmr, legal-bert+translate, sts`
  - `AnalysisResult(document: Document, findings: list[Finding], segments: list[Segment])`
  - `analyze(path: Path, ctx: AnalysisContext, detectors: list[Detector], mode: Literal["clause","sentence"]="sentence") -> AnalysisResult`
  - `export(result: AnalysisResult, out_dir: Path) -> dict[str, Path]`: keys `document`, `report`
  - `write_report(findings, out)`: UTF-8-with-BOM CSV (so Excel shows Vietnamese) with the spec's columns `Vị trí, Câu trích xuất, Loại rủi ro, Thực thể, Mức`, sorted High first
  - CLI: `python -m lca.cli FILE --type commercial --party "Bên B" --mode sentence --detectors regex,xlmr,sts --out reports/contracts/`

Default detectors are `regex,xlmr,sts`; Task 13 replaces this with the best route. Set `HF_HUB_OFFLINE=1` inside `pipeline.py` so inference cannot reach the network after models are cached. `lca/pipeline.py` must not import `modal`.

- [ ] **Step 1: Write the failing tests**

```python
def test_end_to_end_docx_with_stub_detectors(tmp_path):
    res = analyze(sample_docx, AnalysisContext(), [RegexDetector()])
    paths = export(res, tmp_path)
    assert paths["document"].exists() and "12%" in paths["report"].read_text("utf-8-sig")

def test_no_findings_still_exports_valid_files(tmp_path):
    res = analyze(clean_docx, AnalysisContext(), [RegexDetector()])
    paths = export(res, tmp_path)
    assert res.findings == [] and docx.Document(paths["document"])
    assert "Không phát hiện" in paths["report"].read_text("utf-8-sig")

def test_empty_docx_raises_clear_error(tmp_path):
    with pytest.raises(UnreadableDocumentError): analyze(empty_docx, AnalysisContext(), [RegexDetector()])

def test_report_sorted_high_first(tmp_path): ...

def test_pipeline_does_not_import_modal():
    code = "import lca.pipeline, sys; assert 'modal' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)

def test_cli_returns_nonzero_with_message_on_scanned_pdf(tmp_path, capsys): ...
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_pipeline.py -v`
Expected: FAIL with an import error.

- [ ] **Step 3: Implement `analyze`, `export`, `write_report`, the CLI and `app.py`**

`app.py` lets the user upload a file; pick contract type and "Tôi là Bên A/Bên B/Không chọn"; shows the summary table with level filter and download buttons; and always shows the "không thay thế tư vấn pháp lý" notice. Cache detectors with `st.cache_resource`. Show `UnreadableDocumentError` messages with `st.error`. Write uploads to a `tempfile.TemporaryDirectory` so contracts are not kept on disk.

- [ ] **Step 4: Run them to verify they pass**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_pipeline.py -v`
Expected: PASS.

- [ ] **Step 5: Run the app**

Run: `/opt/study/.venv/bin/streamlit run app.py`, upload a generated sample docx, and download both outputs.
Expected: the red highlights and table appear, and the page shows the notice.

- [ ] **Step 6: Commit**

```bash
git add lca/pipeline.py lca/cli.py lca/writers.py app.py tests/test_pipeline.py
git commit -m "feat: end-to-end pipeline, CLI and Streamlit demo"
```

---

### Task 13: Evaluation and research questions

**Files:**
- Create: `lca/eval/annotate.py`, `lca/eval/compare.py`
- Modify: `lca/pipeline.py` (final default detectors)
- Test: `tests/test_eval.py`
- Output: `reports/results.md`, `reports/error_analysis.csv`, `reports/confusion_*.png`

**Interfaces:**
- Consumes: `analyze`, detectors, `report`
- Produces:
  - `make_template(path: Path, out: Path) -> None`: CSV with `seg_id, location, text, label` (sentence segments, blank `label`), one copy per annotator
  - `consensus(paths: list[Path], resolved_out: Path) -> dict`: Cohen's Kappa per pair, and writes disagreements to resolve
  - `sentence_predictions(sentences: list[Segment], findings: list[Finding]) -> list[str]`: projects any granularity onto sentences by span overlap, so clause and sentence modes are scored on the same ground truth
  - `compare(contracts_dir: Path, ground_truth_dir: Path, out: Path) -> dict`
  - CLI: `python -m lca.eval.compare`

- [ ] **Step 1: Write the failing tests**

```python
def test_kappa_perfect_agreement_is_one(tmp_path): ...
def test_kappa_reported_per_pair_for_three_annotators(tmp_path): ...

def test_blank_labels_rejected(tmp_path):
    with pytest.raises(ValueError, match="seg_id"): consensus(...)

def test_clause_finding_projects_onto_contained_sentences():
    assert sentence_predictions(sentences, [clause_finding]) == ["penalty", "penalty", "none"]

def test_compare_reports_every_method(fixture_contracts, tmp_path):
    res = compare(...)
    assert {"regex","tfidf","legal-bert+translate","xlmr","sts","combined"} <= set(res["methods"])
    assert {"sentence","clause"} <= set(res["modes"])

def test_error_analysis_lists_false_positives_and_negatives(fixture_contracts, tmp_path): ...
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_eval.py -v`
Expected: FAIL with an import error.

- [ ] **Step 3: Implement annotation tools and `compare`**

`compare` writes these sections to `reports/results.md`:
- Test-set P/R/F1 (macro and per group) and confusion matrices for regex, TF-IDF, both transformers.
- Sample-contract results for the same methods plus the combination, for each mode.
- RQ1: regex performance on real contracts.
- RQ2: clause vs sentence mode.
- RQ3: combination recall gain versus precision loss against the best single method.
- Route (a) vs (b) on Vietnamese text.
- Kappa values and a note that the two contracts are a case study, not statistically significant.

Generate the plots following the `dataviz` skill.

- [ ] **Step 4: Run them to verify they pass**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_eval.py -v`
Expected: PASS.

- [ ] **Step 5: Create annotation templates**

Run `make_template` for each file in `data/contracts/` and hand the CSVs to the group. Pause here until the Human Prerequisites are met.

- [ ] **Step 6: Run the real evaluation**

Run: `/opt/study/.venv/bin/python -m lca.eval.compare`
Expected: `reports/results.md` answers RQ1–RQ3 with real numbers.

- [ ] **Step 7: Set the final defaults**

Set `build_detectors` defaults to the Vietnamese route and detector combination that won in `results.md`, recording the reason there.

- [ ] **Step 8: Run the full test suite**

Run: `/opt/study/.venv/bin/python -m pytest -v`
Expected: all tests PASS.

- [ ] **Step 9: Commit**

```bash
git add lca tests reports/results.md reports/error_analysis.csv reports/*.png
git commit -m "feat: evaluate methods and answer research questions"
```
