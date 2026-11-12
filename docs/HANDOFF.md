# Handoff: Legal Clause Analyzer (TMG, Group 5)

Last updated: 2026-11-12. Read this first when resuming.

## Where things are

| What | Where |
|---|---|
| Code (git) | `/opt/study/TMG/project` (also reachable via `/home/dre/Desktop/study/TMG/project`, a symlink) |
| GitHub (public) | https://github.com/its-dreOwO/TMG_main. `main` = Checkpoint 2 README + all pipeline code; work continues on `feat/legal-clause-analyzer` |
| Python | shared venv `/opt/study/.venv/bin/python` (3.11). Install only with the current versions pinned as constraints (`uv pip freeze \| grep == > cons.txt; uv pip install -c cons.txt …`) and record new packages in `/opt/study/requirements-global.txt` |
| Data (git-ignored) | `data/cuad`, `data/ledgar`, `data/legal_clause`, `data/processed/{train,val,test}.parquet` |
| Models (git-ignored) | `artifacts/models/`: `tfidf_{lr,svm,best}.joblib`, `legal-bert/final`, `xlmr/final` (1.1 GB), `sts_thresholds.json` |
| Plans and specs | `docs/superpowers/plans/`, `docs/superpowers/specs/` |
| Execution ledgers (git-ignored) | `.superpowers/sdd/<plan-name>/progress.md`: every task line and every `Ruling:` made so far |
| Checkpoint 2 report | `README.md` |

## Status

### 1. Pipeline (plan `2026-10-31-legal-clause-analyzer.md`): Tasks 1–12 done, Task 13 partly done

- **Working:** read docx/pdf → segment → regex + TF-IDF + Legal-BERT/XLM-R + STS → entities → Cao/Trung bình/Thấp → highlighted docx/pdf + CSV; CLI `python -m lca.cli`; Streamlit `app.py`.
- **Tests:** 229 passing (`/opt/study/.venv/bin/python -m pytest -q`).
- **Fixed 2026-11-11** after running six real documents (3 English PDFs, 3 Vietnamese Word files):
  - the reader goes by file content, so a Word file saved as `.pdf` (or the reverse) is read
    correctly and the highlighted output takes the real type's extension;
  - the 8% penalty cap also fires when the regex context pattern finds a penalty inside a sentence
    whose main label is another group ("bồi thường ... và chịu phạt 10%"); the note says so;
  - definition sentences ("“Fees” has the meaning ...", "“X” là ...") are dropped for every
    detector (`is_definition` in `lca/detectors/regex_rules.py`, applied in `lca/pipeline.py`).
- **Fixed 2026-11-12** after cross-checking the written report (`TMG.pdf`) against the code:
  - the CSV report has a sixth column, `Giải thích`, holding the same explanation as the note;
  - a negation counts when it *ends* within 15 characters before the match, so
    "not be subject to any penalty" and "không phải chịu tiền phạt" are no longer flagged.
- **Test-set macro F1 over the risk groups:** TF-IDF+SVM 0.898 (best), Legal-BERT 0.891, XLM-R 0.882. The Transformers win 4 of 5 groups but are weak on penalty (only 314 training rows).
- **Task 13 is blocked on the team:** 2 real contracts in `data/contracts/` plus at least 2 independent labelings, using `python -m lca.eval.annotate template|consensus`, then `python -m lca.eval.compare`. The evaluation code is done and tested.
- **Final whole-branch code review:** not done yet.
- **Known issues:**
  - heading-only lines ("Điều 8. Phạt vi phạm") get flagged;
  - STS gives LOW false alarms on neutral sentences (81% of the findings on those six documents
    were Thấp from STS alone);
  - section locations go wrong on some English PDFs (clause 5.1 reported as "Section 3.6");
  - the entity step tags non-parties as PARTY ("Confidential Information", "the Commencement Date");
  - a finding holds one label, so a penalty inside an indemnity sentence shows only in the note;
  - the opus-mt vi→en translation loses legal meaning (route b);
  - transformers prints a spurious-looking "incorrect regex pattern / fix_mistral_regex" warning when loading the xlmr tokenizer (unverified).

### 2. Website at kodingkat.xyz/TMG: **live** (backend on Modal, not Lambda)

Redeployed 2026-11-12 from `main` at `a15d2c7` (build, smoke and deploy all passed; a live analysis
returned the same notes as the local pipeline). The page files did not change.

Spec `2026-10-31-tmg-web-design.md`, plan `2026-10-31-tmg-website.md`. The plan's Lambda backend
was dropped on 2026-11-09: this AWS account (Free plan) caps Lambda memory at **512 MB**, and the
app needs ~4–6 GB (XLM-R alone is 1.1 GB). The person chose Modal. All AWS resources made for it
(ECR `tmg-lca`, DynamoDB `tmg-lca-usage`, IAM role `tmg-lca-lambda`, Budget `tmg-lca`) were deleted.

How it fits together:
- **Page:** `web/` served by Caddy on the VPS from `/var/www/tmg` at `https://kodingkat.xyz/TMG/`.
- **API:** Caddy `handle_path /TMG/api/*` → `reverse_proxy https://its-dreowo--tmg-lca-web.us-east.modal.run`,
  adding `x-tmg-secret`, `Modal-Key`, `Modal-Secret` from `/etc/caddy/tmg-secrets.caddy`
  (`root:caddy 0640`, not in git). The main site moved into a `handle {}` block; backup of the old
  Caddyfile is `/etc/caddy/Caddyfile.bak-20261109-042708`. Secrets are not in env vars because the
  caddy unit runs with `--environ`, which logs them.
- **Backend:** `deploy/modal/app.py` (Modal app `tmg-lca`):
  - `web`: light WSGI front end (0.25 CPU), `requires_proxy_auth=True`, so direct hits get 401
    before any container starts. Per-IP limiter (3 analyses + 6 warm-ups per 10 min, keyed IP
    hashes in Modal Dict `tmg-lca-ratelimit`).
  - Job flow, so Cloudflare's 100 s limit never applies: `POST /analyze` → 202 `{job}` (Modal
    call id + HMAC); the page polls `GET /result?job=` every 2.5 s. `web/app.js` does the polling.
  - `Analyzer`: runs `lca.web.handler` unchanged on Lambda-style events (4 CPU, 6 GB limit,
    scale to zero after 60 s). 1 page ≈ 13 s cold, 30 pages ≈ 45–90 s end to end.
  - Upload limits (raised 2026-11-11): 20 MB and 60 pages, `Analyzer` timeout 600 s. They are set in
    `lca/web/handler.py`, `deploy/modal/app.py`, `web/app.js`, `web/index.html`, and Caddy's
    `request_body max_size 21MB` on the API route. 60 pages ≈ 80 s warm; 45 pages ≈ 125 s cold.
  - `consume`: the 20/day + 100/month caps in `counts.json` on Volume `tmg-lca-usage`
    (single container, so atomic).
- **Image:** built without Docker by `deploy/lambda/modal_build.py` (`--step build|smoke|push`):
  pip into `/var/task` inside `public.ecr.aws/lambda/python:3.12`, layer tarball on Volume
  `tmg-lca-build`, baked into the Modal image. `--step smoke` runs the handler with the network
  blocked. The XLM-R weights come from `/xlmr-final` on `tmg-lca-build`, uploaded from the local
  copy (sha256 prefix `049a181eefb110e3`). **Do not use `legal-clause-models:/xlmr/final`**: two
  reads of it gave two different hashes, so the copy on that volume isn't trustworthy.
  `--step push` (ECR) is unused now. `Dockerfile`/`smoke.sh` are kept for reference.
- **Redeploy after code changes:** `modal run deploy/lambda/modal_build.py --step build`, then
  `/opt/study/.venv/bin/python -m modal run deploy/lambda/modal_build.py --step smoke` (the pipx `modal`
  has no `pymupdf`), then `MODAL_FORCE_BUILD=1 modal deploy deploy/modal/app.py` (without the force
  flag the cached image with the old layer is reused). Page: copy `web/*` to `/var/www/tmg/`.
- **Secrets:** `~/.config/tmg/secret` (also Modal secret `tmg-lca`), `~/.config/tmg/modal-proxy.json`
  (Modal proxy token `tmg-lca-caddy`).
- **Not done:**
  - Cloudflare rate-limit rule: the token lacks the rulesets permission.
  - A Modal spending limit: no CLI setting; check the dashboard.
  - A real-browser test of the polling page (the person is testing it).

## Environment facts the next session needs

- **Modal** profile `its-dreowo`, about $25 of credit.
  - Volumes: `legal-clause-data` (raw + processed data, downloaded by Modal from Kaggle) and `legal-clause-models` (trained runs).
  - Secret `kaggle`.
  - Training GPU: `RTX-PRO-6000` (bf16, batch 64, ~7 min of training per model).
- **AWS** CLI user `claude`, Free plan; never upgrade to Paid. Lambda memory is capped at 512 MB and concurrency at 5, which is why TMG runs on Modal. VPS `ssh -i ~/.ssh/aws-vps.pem admin@13.251.153.23` (t4g.small, Caddy, also serves `shop.kodingkat.xyz`).
- **Cloudflare** token in `~/.config/cloudflare/token` (strip a possible BOM).
- **Kaggle** token in `~/.kaggle/access_token`.
- **Docker** 29 works locally, but only about 7 GB of disk was free on 2026-11-09; image builds run on Modal instead.
- **Local DNS** fails intermittently (`Could not resolve host`); retry before suspecting the server. Cloudflare's bot check rejects Python-urllib's default user agent (error 1010).
- **Commit trailers:** `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` plus the current session's `Claude-Session:` line.

## Suggested order next session

1. Confirm the person's browser test of kodingkat.xyz/TMG (new tab with the PDF, counts, download, 375 px layout).
2. Final whole-branch review of both plans (none has run yet).
3. When the team supplies the contracts: pipeline Task 13 (labels → Kappa → `compare` → `reports/results.md`), then fix the heading and STS false alarms.
