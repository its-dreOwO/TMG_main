# Prompt: slide deck for the Legal Clause Analyzer

Copy everything below the line into a slide generator (Gamma, Claude, ChatGPT, Canva Magic Design, …).

---

Create a 12-slide presentation in **Vietnamese** (keep technical terms such as Regex, TF-IDF, XLM-R, NER and
STS in English) for a university project demo.

**Project:** *Trợ lý nhận diện rủi ro hợp đồng pháp lý (Legal Clause Analyzer)*, Group 5: Nguyễn Hoài
Khánh, Lê Tuấn Kiệt, Bùi Ngọc Phương Uyên, Dương Gia Bảo.
An internal tool for HR and Sales staff. They upload a contract (Word or PDF) and get back the same file with
the risky sentences highlighted red, a comment explaining each one, and a summary report.

**Focus:** follow one contract through the system: what the user does when uploading it, what happens
during processing, and what results come back. Show results with charts, not tables of text. Use only the
numbers given below; do not invent figures.

**Style:** clean and professional, white background, dark navy text, one accent colour for the risk levels:
red = Cao, amber = Trung bình, yellow = Thấp. Few words per slide, one idea per slide, large diagrams. Add
speaker notes (2–4 sentences) to every slide.

## Slide plan

**1. Title.** Project name, subtitle "Tự động bôi đỏ điều khoản bất lợi trong hợp đồng Word/PDF", group
members, live demo URL `kodingkat.xyz/TMG`.

**2. The problem.** HR/Sales staff sign contracts without legal training, and the risky clauses (penalties,
compensation, confidentiality) are buried in long documents. The goal is a first-pass screen in seconds, not
legal advice. Footer: *"Công cụ chỉ mang tính hỗ trợ sàng lọc, không thay thế tư vấn pháp lý."*

**3. Step 1, Upload (user's view).** A mock-up of the web page:
- title "Nhận diện rủi ro hợp đồng";
- drop zone for a `.docx` / `.pdf` file (max 20 MB, 60 pages);
- dropdown **Loại hợp đồng**: Thương mại / Lao động / Khác;
- dropdown **Tôi là**: Không chọn / Bên A / Bên B;
- button **Phân tích**.

Note: the contract type turns on legal rules (e.g. the 8% penalty cap for commercial contracts), and the
party choice lowers the level of obligations that fall on the other side.

**4. Step 2, What happens after "Phân tích" (system flow).** Draw this as a horizontal flowchart:
Browser → `POST /analyze` (returns a job id at once) → cloud worker (Modal, 4 CPU) runs the pipeline →
browser asks `GET /result` every 2.5 s → result is ready → download.
Timing callouts: 1 page ≈ 13 s on a cold start; 60 pages ≈ 80 s when warm.

**5. Step 3, Processing pipeline (the core diagram).** Flowchart with 6 stages:
1. **Read**: python-docx / PyMuPDF; rejects scanned, encrypted or corrupt files with a clear message.
2. **Segment**: split into Điều / Khoản / sentence, Vietnamese and English; article titles are skipped.
3. **Detect**: three methods run in parallel:
   - **Regex + keywords** (VI + EN, with negation handling: "không bị phạt" is not flagged);
   - **Classifier** XLM-R (multilingual Transformer), fine-tuned on CUAD + LEDGAR;
   - **Semantic similarity (STS)** against 50 bilingual reference clauses.
4. **Combine**: recall-first; a sentence is flagged if any method fires, and the level rises when the
   methods agree.
5. **Entities (NER)**: parties, money, percentages, durations, dates.
6. **Score**: Cao / Trung bình / Thấp.

**6. Five risk groups and the scoring rules.** Icons for Phạt vi phạm, Bồi thường thiệt hại, Bảo mật,
Chấm dứt hợp đồng, Hạn chế cạnh tranh. Rules that force **Cao**:
- penalty above **8%** in a commercial contract (Điều 301 Luật Thương mại 2005);
- uncapped compensation ("toàn bộ thiệt hại", "không giới hạn", "100%");
- confidentiality with no time limit ("vô thời hạn", "indefinitely").

**7. Step 4, Results the user gets.** Show three outputs side by side:
- (a) the original Word/PDF with a red-highlighted sentence and a margin comment, e.g. *"Bồi thường thiệt
  hại – Mức Cao: Bồi thường thiệt hại; mức 10%; vượt trần phạt 8% (Điều 301 Luật Thương mại 2005)
  (phát hiện bởi: regex, sts, xlmr)."*;
- (b) the on-screen findings list, filterable by level;
- (c) the CSV report with columns: Vị trí, Câu trích xuất, Loại rủi ro, Thực thể, Mức, Giải thích.

**8. Sample contract 1: labour contract (Hợp đồng lao động xác định thời hạn).**
- **Chart 1:** donut, findings by level: Cao 1, Trung bình 6, Thấp 12 (total 19).
- **Chart 2:** horizontal bar, findings by risk group: Chấm dứt hợp đồng 9, Bồi thường thiệt hại 6, Hạn chế
  cạnh tranh 2, Phạt vi phạm 1, Bảo mật 1.
- Callout for the top finding (Cao): *"…nhân viên phải hoàn trả lại 100% phí đào tạo…"* (full
  repayment of training costs if the employee leaves).

**9. Sample contract 2: business cooperation agreement (Hợp đồng hợp tác kinh doanh).**
- **Chart:** donut, findings by level: Cao 1, Trung bình 2, Thấp 1 (total 4).
- Callout (Cao): *"…phải bồi thường toàn bộ thiệt hại xảy ra và chịu phạt vi phạm bằng 10% giá trị của
  thỏa thuận"*: uncapped damages, and the 10% penalty exceeds the legal 8% cap.
- Also flagged: confidentiality (Điều 8.3) and termination (Điều 9.1), both Trung bình.

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
- Done: Word/PDF in → highlighted file + report out, Vietnamese and English, three combined methods, live
  website.
- Limitations: text PDFs only (no OCR); some Thấp false alarms remain; the English-trained models are
  weaker on Vietnamese penalty wording, which is why regex and STS are combined with them.
- Closing line: *"Công cụ chỉ mang tính hỗ trợ sàng lọc, không thay thế tư vấn pháp lý."*

Training data facts, if a slide needs them: CUAD + LEDGAR, 136,478 training / 17,426 validation / 18,369
test rows, split by source contract so no contract appears in two splits.
