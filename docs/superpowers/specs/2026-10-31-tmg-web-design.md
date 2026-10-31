# Legal Clause Analyzer website (kodingkat.xyz/TMG): design

## Intent

A public demo page where anyone (the course teacher, classmates, a member of HR) uploads a PDF contract and gets the same PDF back with risky text highlighted red. The highlights carry notes such as "Phạt vi phạm – Mức Cao: …". The person sees the result in the browser's own PDF viewer and can download it. The site exists so the project can be tried from a link, without installing anything. Expected use is tiny: fewer than 10 analyses, mostly by the teacher while grading. A ready-to-run local download is on pause (decided 2026-10-31), so the page makes no promise about running it locally.

## Goals

- Upload one PDF, analyse it with the project's pipeline, return the highlighted PDF.
- After processing, the highlighted PDF opens in a browser PDF viewer tab, and a download button appears.
- Minimal, calm design: no gradients, no decorative effects.
- Served at `https://kodingkat.xyz/TMG`, behind the existing Cloudflare + Caddy setup.
- Costs stay inside the AWS credit and free tier. The account is on the AWS Free plan and must stay there.

## Non-goals

- `.docx` upload (the local Streamlit app covers it). Scanned PDFs / OCR.
- Accounts, history, saved results, editing results.
- Showing the CSV report on the page (a short count of findings by level is enough).

## Anti-goals (a failure even if it works)

- Storing or logging contract content anywhere (disk, S3, CloudWatch log lines with text).
- Gradient or flashy UI.
- Charges outside AWS credits, any change to the AWS Paid plan, or anything always-on that bills around the clock (no provisioned concurrency, no Fargate).
- Breaking or slowing the existing `kodingkat.xyz` site or `shop.kodingkat.xyz`.

## Constraints

- **VPS** (AWS EC2 t4g.small, 2 GB RAM, Debian 13, Singapore, `ssh -i ~/.ssh/aws-vps.pem admin@13.251.153.23`) only serves static files and proxies. It must not run the models.
- **Caddy** config is `/etc/caddy/Caddyfile`. The `kodingkat.xyz` block serves a React SPA with `try_files {path} /index.html`, so `/TMG` needs its own `handle` placed before that fallback. That block's CSP is `script-src 'self' 'nonce-…'`, `connect-src 'self'`, `frame-src 'self'`, `object-src 'none'`, `frame-ancestors 'none'`. The page therefore uses external JS/CSS files only and calls the API on the same origin. The `/TMG` handle may set its own CSP if it needs to (e.g. to allow `blob:` for the viewer), without loosening the rest of the site.
- **Cloudflare** proxies the domain and cuts a request off after **100 seconds**, so a cold start plus processing must finish inside that.
- **AWS**: region `ap-southeast-1`. CLI user `claude` (roughly PowerUser; whether it may create IAM roles is unknown, see Open items). Account Lambda concurrency limit is **5**. Docker 29 (x86_64) is available on the laptop for building the image.
- **Lambda request/response limit** is 6 MB, so the PDF upload cap is **4 MB**, checked in the browser and again on the server.
- Reuse the existing project code (`lca/`), not a rewrite: `read_document → segment → detectors → extract_entities → score → write_pdf`, as `lca.pipeline.analyze` + `lca.writers.write_pdf` already do.

## Decided

### Compute: AWS Lambda container image

- One Lambda function, `tmg-lca`, built from a container image pushed to ECR in `ap-southeast-1`. Architecture: whichever builds reliably on the x86_64 laptop (x86_64 by default).
- The image holds Python 3.11, the `lca` package, CPU-only torch 2.12.0, transformers 4.57.6 and sentence-transformers 5.7.0 (the versions the models were trained and tested with), spaCy + `en_core_web_sm`, underthesea, pymupdf, scikit-learn, and the model files baked in: `artifacts/models/xlmr/final`, the multilingual MiniLM, `sts_thresholds.json`, `lca/resources/prototypes.yaml`. All caches point inside the image, nothing downloads at run time (`HF_HUB_OFFLINE=1`), and anything that must be writable goes under `/tmp`.
- Detectors: the pipeline default `regex + xlmr + sts`, sentence mode.
- Memory around 6 GB (Lambda gives proportionally more CPU), timeout 90 s. Models load once per container and stay warm between requests.
- No provisioned concurrency. Cold starts are accepted (expected 15–40 s).
- **Warm-up ping:** when the page loads, the browser fires a cheap request (`GET /TMG/api/warm`) so the container starts loading models while the person picks a file.

### API

- The browser only ever talks to `https://kodingkat.xyz/TMG/api/...` (same origin).
- Caddy reverse-proxies `/TMG/api/*` to the Lambda **Function URL** over HTTPS, setting the upstream Host header and adding a shared secret header. Lambda rejects requests without the secret, so the Function URL isn't usable directly. The secret lives in the Caddyfile and the Lambda env var only, never in git.
- `POST /TMG/api/analyze`: PDF bytes plus optional `contract_type` (`commercial` | `labour` | `other`, default `commercial`) and `party` (`Bên A` | `Bên B` | none). Returns the highlighted PDF plus counts of findings per level (Cao / Trung bình / Thấp). Errors return a clear Vietnamese message, reusing `UnreadableDocumentError` text: scanned PDF, encrypted, corrupt, too large, not a PDF.
- `GET /TMG/api/warm`: loads the models if they aren't loaded yet, returns quickly.

### Page (static files on the VPS)

- Files live in their own folder on the VPS, e.g. `/var/www/tmg`, served by a `handle /TMG*` block in the `kodingkat.xyz` site. Plain HTML + one CSS + one JS file, no framework, no build step.
- **Look:** white background, near-black text, one accent colour (the same red as the highlights, used sparingly), system font stack, generous whitespace, one narrow centred column, thin 1px borders, no gradients, no heavy shadows, no animation beyond a simple progress indicator. Works at phone width. Respects dark mode only if it costs nothing; light is enough.
- **Copy in Vietnamese** (the users are Vietnamese), with the project title and group name.
- **Flow:**
  1. Title, one-line description, the privacy note (*"File được xử lý trong bộ nhớ và không được lưu lại."*), a small link to the source code (`github.com/its-dreOwO/TMG_main`), and the legal disclaimer (*"Kết quả chỉ mang tính hỗ trợ sàng lọc, không thay thế tư vấn pháp lý."*). No instructions or promise about running it locally.
  2. One drop zone / file picker (PDF only), plus two small selects: contract type and "Tôi là" (Bên A / Bên B / Không chọn).
  3. Analyse button. While working, a status line that is honest about the cold start ("Đang khởi động mô hình, lần đầu có thể mất khoảng 30 giây…" → "Đang phân tích…").
  4. Done: the highlighted PDF opens in a new browser tab (browser PDF viewer), a **Tải xuống** button appears, plus a one-line count ("3 rủi ro Cao · 2 Trung bình · 1 Thấp"), or "Không phát hiện điều khoản rủi ro nào". Browsers block pop-ups opened after an asynchronous wait, so if the new tab can't open automatically, a **Xem PDF** button opens it on click.
  5. Errors show inline in plain words, and the person can try again without reloading.

### Privacy

- Lambda processes in memory (`/tmp` only if a library insists, deleted before returning). No S3. CloudWatch logs carry only request id, page count, timing and finding counts, never text.
- Caddy access logs for `/TMG/api` must not record request bodies (they don't by default).

### Testing

- Unit tests for the Lambda handler: valid PDF round-trip, missing/invalid secret, not-a-PDF, oversized, scanned PDF message, warm endpoint, counts by level.
- The image is tested locally with the Lambda runtime interface emulator before pushing.
- End-to-end: `curl` through `https://kodingkat.xyz/TMG/api/analyze` with a sample PDF, and the page checked in a browser.
- Existing `kodingkat.xyz` routes and `shop.kodingkat.xyz` still work after the Caddy change (`caddy validate` before reload).

### Limits and abuse protection

- **Same AWS account as the VPS.** The Lambda function lives in the same account as the EC2 VPS behind `kodingkat.xyz` and `shop.kodingkat.xyz`. On the Free plan, running out of credit closes the account and takes those sites down too, so protecting the credit is a hard requirement, not a nicety.
- Reserved concurrency can't serve as a cap (AWS keeps at least 10 unreserved, and the account limit is 5), so protection comes from:
  - **hard usage caps** inside the handler, counted in a tiny DynamoDB table (date → count and month → count, no content): **20 analyses per day and 100 per month** (the person expects fewer than 10 in total). Once reached, the page shows a polite Vietnamese message. Warm-up pings don't count;
  - **a Cloudflare rate-limiting rule** on `/TMG/api/analyze` (per IP), if the stored Cloudflare token is allowed to create one; if not, the person adds it in the dashboard;
  - **a dedicated AWS Budget of $1/month** for this project (Lambda + ECR + DynamoDB), with an email alert to drenguyen288@gmail.com at 50% and 100% actual spend, alongside the existing "My Monthly Cost Budget". Expected spend is cents: Lambda and DynamoDB stay inside the free tier at this volume; ECR image storage is the only real cost.
- **Page limit: 30 pages** (builder's call, adjustable). Longer PDFs are refused with a clear message pointing to the local app; no partial "first N pages" output. This keeps a cold start plus processing well inside the 90 s timeout and Cloudflare's 100 s limit.

### Memory fallback

The ~6 GB target assumes the account allows it. New accounts are sometimes capped lower, and that can only be confirmed by creating the function. If 6 GB is refused, the builder uses the highest allowed size, measures cold start and a 30-page contract, and if that doesn't fit within ~90 s, lowers the page limit before dropping any detector (builder's call).

## Left to the builder

- Exact request/response encoding (raw body vs multipart, PDF as base64 JSON vs binary with counts in a header), as long as the 6 MB limit and the 4 MB cap hold.
- Lambda handler framework (plain handler vs an adapter), image base, layer layout, how models are copied in.
- Exact memory size within ~4–8 GB, chosen from measured cold-start and per-contract times.
- File names, CSS details, icon/favicon, exact Vietnamese wording beyond the lines given above.
- Deploy scripts (build, push, update function, upload static files) and where they live in the repo.

## Open items

- **IAM role for Lambda:** the `claude` user may not be allowed to create roles. If `iam:CreateRole` is denied, the person creates a basic Lambda execution role (`AWSLambdaBasicExecutionRole` only) in the console and gives its ARN.
- **ECR storage** for an image of roughly 3–4 GB costs about $0.30–0.40 a month from AWS credit; acceptable.
