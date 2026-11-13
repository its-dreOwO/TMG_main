# Prompt: slide deck for the Legal Clause Analyzer

Copy everything below the line into a slide generator (Gamma, Claude, ChatGPT, Canva Magic Design, …).

---

Create a 12-slide presentation **entirely in English** for a university project demo. Every word on the
slides and in the speaker notes must be English. The tool's own interface and the sample contracts are
Vietnamese, so show the English translations given below, never the Vietnamese text.

**Project:** *Legal Clause Analyzer: a contract risk assistant*. Group 5: Nguyen Hoai Khanh, Le Tuan Kiet,
Bui Ngoc Phuong Uyen, Duong Gia Bao.
An internal tool for HR and Sales staff. They upload a contract (Word or PDF) and get back the same file with
the risky sentences highlighted red, a comment explaining each one, and a summary report.

**Focus:** follow one contract through the system: what the user does when uploading it, what happens
during processing, and what results come back. Show results with charts, not tables of text. Use only the
numbers given below; do not invent figures.

**Style:** clean and professional, white background, dark navy text, one accent colour per risk level:
red = High, amber = Medium, yellow = Low. Few words per slide, one idea per slide, large diagrams. Add
speaker notes (2–4 sentences) to every slide.

## Slide plan

**1. Title.** Project name, subtitle "Automatically highlight unfavourable clauses in Word/PDF contracts",
group members, live demo URL `kodingkat.xyz/TMG`.

**2. The problem.** HR/Sales staff sign contracts without legal training, and the risky clauses (penalties,
compensation, confidentiality) are buried in long documents. The goal is a first-pass screen in seconds, not
legal advice. Footer: *"A screening aid only; it does not replace legal advice."*

**3. Step 1, Upload (user's view).** A mock-up of the web page, with English labels:
- title "Contract Risk Detection";
- drop zone for a `.docx` / `.pdf` file (max 20 MB, 60 pages);
- dropdown **Contract type**: Commercial / Labour / Other;
- dropdown **I am**: Not specified / Party A / Party B;
- button **Analyze**.

Note: the contract type turns on legal rules (e.g. the 8% penalty cap for commercial contracts), and the
party choice lowers the level of obligations that fall on the other side.

**4. Step 2, What happens after "Analyze" (system flow).** Draw this as a horizontal flowchart:
Browser → `POST /analyze` (returns a job id at once) → cloud worker (Modal, 4 CPU) runs the pipeline →
browser asks `GET /result` every 2.5 s → result is ready → download.
Timing callouts: 1 page ≈ 13 s on a cold start; 60 pages ≈ 80 s when warm.

**5. Step 3, Processing pipeline (the core diagram).** Flowchart with 6 stages:
1. **Read**: python-docx / PyMuPDF; rejects scanned, encrypted or corrupt files with a clear message.
2. **Segment**: split into articles, clauses and sentences (Vietnamese and English); article titles are
   skipped.
3. **Detect**: three methods run in parallel:
   - **Regex + keywords** (Vietnamese + English, with negation handling: "shall not be penalised" is not
     flagged);
   - **Classifier**: XLM-R (multilingual Transformer), fine-tuned on CUAD + LEDGAR;
   - **Semantic similarity (STS)** against 50 bilingual reference clauses.
4. **Combine**: recall-first; a sentence is flagged if any method fires, and the level rises when the
   methods agree.
5. **Entities (NER)**: parties, money, percentages, durations, dates.
6. **Score**: High / Medium / Low.

**6. Five risk groups and the scoring rules.** Icons for Penalty, Indemnity, Confidentiality, Termination
and Non-compete. Rules that force **High**:
- a penalty above **8%** in a commercial contract (Article 301, Vietnam Commercial Law 2005);
- uncapped compensation ("all damages", "without limit", "100%");
- confidentiality with no time limit ("indefinitely").

**7. Step 4, Results the user gets.** Show three outputs side by side:
- (a) the original Word/PDF with a red-highlighted sentence and a margin comment, e.g. *"Indemnity – High:
  rate 10%; exceeds the 8% penalty cap (Article 301, Commercial Law 2005) (detected by: regex, sts,
  xlmr)."*;
- (b) the on-screen findings list, filterable by level;
- (c) the CSV report with columns: Location, Extracted sentence, Risk type, Entities, Level, Explanation.

**8. Sample contract 1: fixed-term labour contract.**
- **Chart 1:** donut, findings by level: High 1, Medium 6, Low 12 (total 19).
- **Chart 2:** horizontal bar, findings by risk group: Termination 9, Indemnity 6, Non-compete 2, Penalty 1,
  Confidentiality 1.
- Callout for the top finding (High): *"…the employee must repay 100% of the training costs…"* if they
  leave after the training course.

**9. Sample contract 2: business cooperation agreement.**
- **Chart:** donut, findings by level: High 1, Medium 2, Low 1 (total 4).
- Callout (High): *"…shall compensate for all damages incurred and pay a penalty of 10% of the agreement's
  value"*: uncapped damages, and the 10% penalty exceeds the legal 8% cap.
- Also flagged: confidentiality (Article 8.3) and termination (Article 9.1), both Medium.

**10. Model results on the public test set (18,369 English clauses).**
- **Chart:** grouped bar chart of Precision / Recall / F1 (macro over the 5 risk groups):

  | Model | Precision | Recall | F1 |
  |---|---|---|---|
  | TF-IDF + Logistic Regression | 0.853 | 0.918 | 0.883 |
  | TF-IDF + Linear SVM | 0.875 | 0.925 | 0.898 |
  | Legal-BERT | 0.857 | 0.929 | 0.891 |
  | XLM-R (multilingual) | 0.842 | 0.926 | 0.882 |

- Takeaway: every model has recall above 0.91, which matches the recall-first goal.
- Optional second chart, F1 per risk group for SVM vs XLM-R: Penalty 0.851 vs 0.735, Indemnity 0.929 vs
  0.925, Confidentiality 0.943 vs 0.941, Termination 0.872 vs 0.888, Non-compete 0.896 vs 0.920. The
  Transformers are weak on penalty because it has only 314 training examples.

**11. Cutting false alarms.**
- **Chart:** before/after bar chart of the number of findings:
  - business cooperation contract: 27 → 4;
  - demo service contract: 27 → 15.
- What changed: article titles are skipped, a sentence flagged by similarity alone needs a score ≥ 0.60,
  and PDF ligatures are expanded. No real risk was lost.

**12. Summary and limitations.**
- Done: Word/PDF in → highlighted file + report out, Vietnamese and English contracts, three combined
  methods, live website.
- Limitations: text PDFs only (no OCR); some Low false alarms remain; the English-trained models are weaker
  on Vietnamese penalty wording, which is why regex and STS are combined with them.
- Closing line: *"A screening aid only; it does not replace legal advice."*

Training data facts, if a slide needs them: CUAD + LEDGAR, 136,478 training / 17,426 validation / 18,369
test rows, split by source contract so no contract appears in two splits.
