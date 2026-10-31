# TMG Website (kodingkat.xyz/TMG) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Public page at `https://kodingkat.xyz/TMG` that uploads a PDF, runs the pipeline on AWS Lambda, and returns the red-highlighted PDF for viewing and download.

**Architecture:** Static HTML/CSS/JS served by Caddy on the VPS; Caddy proxies `/TMG/api/*` (same origin) to a Lambda Function URL with a shared secret header; the Lambda container image runs `lca.pipeline.analyze` + `lca.writers.write_pdf` in memory and counts usage in DynamoDB.

**Tech Stack:** Python 3.12 Lambda container (AL2023 base, glibc new enough for torch 2.12 wheels), boto3, DynamoDB, ECR, Caddy, Cloudflare, vanilla JS.

**Spec:** `docs/superpowers/specs/2026-10-31-tmg-web-design.md`

## Global Constraints

- Region `ap-southeast-1`; AWS account stays on the Free plan; nothing always-on (no provisioned concurrency).
- Upload cap 4 MB (checked in browser and server); PDF only; max 30 pages.
- Caps: 20 analyses/day, 100/month (UTC dates); warm pings don't count.
- Never store or log contract text; logs carry request id, pages, seconds, counts only.
- Detectors `regex + xlmr + sts`, sentence mode; torch 2.12.0 CPU, transformers 4.57.6, sentence-transformers 5.7.0; `HF_HUB_OFFLINE=1` at run time.
- Copy: Vietnamese; privacy line "File được xử lý trong bộ nhớ và không được lưu lại."; disclaimer "Kết quả chỉ mang tính hỗ trợ sàng lọc, không thay thế tư vấn pháp lý."; no local-run promise; source link `https://github.com/its-dreOwO/TMG_main`.
- No gradients; one accent red; system fonts; external JS/CSS only (site CSP).
- Shared secret header `x-tmg-secret`; secret only in Lambda env and the VPS Caddyfile, never in git.
- Budget: AWS Budget `tmg-lca` $1/month, alerts at 50% and 100% actual to drenguyen288@gmail.com.

## Review Focus

1. Cold start + 30-page PDF must answer within 100 s (Cloudflare) — measured in Task 5.
2. A request straight to the Function URL without the secret must get 403 — pinned in Task 1 tests and checked live in Task 5.
3. Usage cap race between concurrent containers — DynamoDB atomic `ADD` with a condition, pinned in Task 1.
4. Browser blocks the new tab after an async wait — "Xem PDF" button fallback, checked in Task 4.
5. Existing `kodingkat.xyz` SPA routes and `shop.kodingkat.xyz` keep working after the Caddy change — checked in Task 5.

---

### Task 1: Lambda handler

**Files:** Create `lca/web/__init__.py`, `lca/web/handler.py`; Test `tests/test_web_handler.py`

**Interfaces:**
- Consumes: `lca.pipeline.analyze`, `build_detectors`, `DEFAULT_DETECTORS`; `lca.writers.write_pdf`; `lca.reader.UnreadableDocumentError`; `AnalysisContext`.
- Produces:
  - `handler(event: dict, context) -> dict` — Lambda Function URL event (payload v2). Routes on `event["rawPath"]` suffix: `/warm` (GET) and `/analyze` (POST). Response `{"statusCode", "headers": {"content-type": "application/json"}, "body": json}`.
  - `/analyze` request: raw PDF body (`isBase64Encoded` true), query `contract_type` ∈ commercial|labour|other (default commercial), `party` ∈ "Bên A"|"Bên B"|absent.
  - `/analyze` 200 body: `{"pdf": <base64>, "counts": {"Cao": n, "Trung bình": n, "Thấp": n}, "filename": "<stem>_danh_dau.pdf"}`.
  - Errors: `{"error": "<Vietnamese message>"}` with 400 (not PDF / empty / bad params), 413 (>4 MB or >30 pages), 422 (`UnreadableDocumentError` message), 429 (cap reached), 403 (secret missing/wrong).
  - `class UsageCounter(Protocol): def try_consume(self, day: str, month: str) -> bool`; `DynamoUsageCounter(table: str)` uses `UpdateItem ADD n :1` with `ConditionExpression attribute_not_exists(n) OR n < :limit` on items `day#YYYY-MM-DD` (limit 20) and `month#YYYY-MM` (limit 100); day is consumed only if the month check passes.
  - Module-level lazy detector cache so `/warm` loads models once per container.

- [ ] Step 1: tests (fakes for counter and detectors via `handler` injection points `_COUNTER`, `_DETECTORS`): `test_rejects_missing_secret` (403), `test_rejects_wrong_secret`, `test_warm_does_not_consume_quota`, `test_rejects_non_pdf` (400), `test_rejects_over_4mb` (413), `test_rejects_over_30_pages` (413), `test_scanned_pdf_returns_422_message` (contains "scan"), `test_analyze_returns_highlighted_pdf_and_counts` (decoded PDF has a Highlight annot; counts keys exactly Cao/Trung bình/Thấp), `test_quota_exhausted_returns_429`, `test_invalid_contract_type_400`, `test_no_text_logged` (caplog contains no sentence text), `test_dynamo_counter_conditional_update` (stubbed boto3 client via botocore Stubber: ConditionalCheckFailedException → False).
- [ ] Step 2: run, see FAIL.
- [ ] Step 3: implement. Secret from env `TMG_SECRET` (compare with `hmac.compare_digest`); table from env `TMG_TABLE`. Process in a `tempfile.TemporaryDirectory()` under `/tmp`, deleted before returning.
- [ ] Step 4: run, see PASS; full suite green.
- [ ] Step 5: commit.

### Task 2: Container image

**Files:** Create `deploy/lambda/Dockerfile`, `deploy/lambda/requirements.txt`, `deploy/lambda/prefetch.py`, `deploy/lambda/smoke.sh`

- Base `public.ecr.aws/lambda/python:3.12` (x86_64). Install pinned deps with the CPU torch index. Copy `lca/` and `artifacts/models/xlmr/final`, `artifacts/models/sts_thresholds.json`. `prefetch.py` runs at build time (network on) to cache MiniLM, spaCy, underthesea resources under `/opt/cache` and runs one analysis on a generated Vietnamese PDF so every lazy download happens at build. Runtime env: `HF_HOME=/opt/cache/hf`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, plus whatever underthesea needs pointed at `/opt/cache`; `CMD ["lca.web.handler.handler"]`.
- [ ] Build; `smoke.sh` runs the image with the Lambda runtime interface emulator and `--network none`, posts `/warm` then a sample PDF; expected 200 with a highlight, and timings printed.
- [ ] Commit.

### Task 3: AWS resources

**Files:** Create `deploy/lambda/deploy.sh` (idempotent: create-or-update)

- ECR repo `tmg-lca`; DynamoDB table `tmg-lca-usage` (PK `pk` string, on-demand); IAM role `tmg-lca-lambda` with `AWSLambdaBasicExecutionRole` + inline `dynamodb:UpdateItem` on that table (if `iam:CreateRole` is denied, stop and ask the person); Lambda `tmg-lca` image, memory 6144 (fallback: highest accepted), timeout 90, env `TMG_SECRET` (generated, stored only in `~/.config/tmg/secret`), `TMG_TABLE`; Function URL auth NONE; Budget `tmg-lca` per Global Constraints.
- [ ] Verify: `curl` Function URL `/warm` with secret → 200; without → 403. Commit.

### Task 4: Static page

**Files:** Create `web/index.html`, `web/style.css`, `web/app.js`

- Behaviour per spec Flow 1–5: on load `fetch('api/warm')` (relative, fire-and-forget); file check (PDF, ≤4 MB) before upload; POST raw bytes to `api/analyze?contract_type=…&party=…`; status texts; on success build a Blob URL, try `window.open` in a tab pre-opened at click time, else show "Xem PDF"; show "Tải xuống" (download attribute = filename) and the counts line; errors inline; retry without reload.
- [ ] Verify locally with a tiny static server + a mocked API (Python script returning a fixture), check at 375 px and desktop width. Commit.

### Task 5: VPS, Cloudflare, end-to-end

**Files:** Create `deploy/vps/caddy-tmg.snippet` (secret as placeholder `{env.TMG_SECRET}` or written on the server only)

- Upload `web/` to `/var/www/tmg`. In the `kodingkat.xyz` block add, before the SPA fallback: `redir /TMG /TMG/`; `handle_path /TMG/api/* { reverse_proxy https://<function-url-host> { header_up Host {upstream_hostport}; header_up x-tmg-secret <secret> } }`; `handle_path /TMG/* { root * /var/www/tmg; file_server }`; a `/TMG` CSP that adds `blob:` to `frame-src`/`object-src`. Back up Caddyfile, `caddy validate`, reload.
- Cloudflare rate-limit rule on `/TMG/api/analyze` (if the token allows; else tell the person).
- [ ] Verify: page loads; real contract PDF round-trip through `https://kodingkat.xyz/TMG/api/analyze` with cold-start timing < 100 s; 30-page PDF timing; Function URL without secret → 403; `https://kodingkat.xyz/`, a deep SPA route and `https://shop.kodingkat.xyz/` still 200. Commit and push.
