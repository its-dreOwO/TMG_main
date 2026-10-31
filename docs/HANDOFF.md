# Handoff: Legal Clause Analyzer (TMG, Group 5)

Last updated: 2026-10-31. Read this first when resuming.

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
- **Tests:** 200 passing (`/opt/study/.venv/bin/python -m pytest -q`).
- **Test-set macro F1 over the risk groups:** TF-IDF+SVM 0.898 (best), Legal-BERT 0.891, XLM-R 0.882. The Transformers win 4 of 5 groups but are weak on penalty (only 314 training rows).
- **Task 13 is blocked on the team:** 2 real contracts in `data/contracts/` plus at least 2 independent labelings, using `python -m lca.eval.annotate template|consensus`, then `python -m lca.eval.compare`. The evaluation code is done and tested.
- **Final whole-branch code review:** not done yet.
- **Known issues:**
  - heading-only lines ("Điều 8. Phạt vi phạm") get flagged;
  - STS gives LOW false alarms on neutral sentences;
  - the opus-mt vi→en translation loses legal meaning (route b);
  - transformers prints a spurious-looking "incorrect regex pattern / fix_mistral_regex" warning when loading the xlmr tokenizer (unverified).

### 2. Website at kodingkat.xyz/TMG (spec `2026-10-31-tmg-web-design.md`, plan `2026-10-31-tmg-website.md`)

The person approved the design ("ship it"). Choices already made:
- AWS Lambda (container image, ap-southeast-1) rather than Modal or Daytona;
- caps of 20 analyses a day and 100 a month;
- a $1/month AWS Budget;
- no local-run promise on the page;
- page copy in Vietnamese;
- PDFs only, max 4 MB / 30 pages.

| Task | State |
|---|---|
| 1. Lambda handler `lca/web/handler.py` | ✅ done, 20 tests (`tests/test_web_handler.py`), committed |
| 2. Container image `deploy/lambda/` | ⚠️ **First build failed** at prefetch (`No module named 'lca'`). Fixed in the Dockerfile (`PYTHONPATH=${LAMBDA_TASK_ROOT}`) but **not rebuilt yet**. Next: `docker build --platform linux/amd64 -f deploy/lambda/Dockerfile -t tmg-lca .` then `PYTHON=/opt/study/.venv/bin/python deploy/lambda/smoke.sh` (runs with `--network none`, prints cold start, 1-page and 30-page timings, and checks 403 without the secret). Base is `public.ecr.aws/lambda/python:3.12` (AL2023; python 3.11's AL2 glibc is too old for torch 2.12 wheels). |
| 3. AWS resources | ⏳ not started. Plan: ECR `tmg-lca`, DynamoDB `tmg-lca-usage` (pk string, on-demand), IAM role `tmg-lca-lambda` (may be denied for the `claude` IAM user; then ask the person to create it), Lambda `tmg-lca` 6144 MB / 90 s, Function URL auth NONE, env `TMG_SECRET` (keep in `~/.config/tmg/secret`, never in git) + `TMG_TABLE`, Budget `tmg-lca` $1 (alerts to drenguyen288@gmail.com at 50%/100%). Never set `TMG_QUOTA=off` on the deployed function (it exists for local smoke tests only). |
| 4. Static page `web/` | 🟡 written (`index.html`, `style.css`, `app.js`, minimal, no gradients). The API flow is verified through `deploy/web/mock_server.py` (serves `/TMG/`, real handler with the regex detector). **Not yet checked in a real browser:** layout at 375 px and desktop, the pop-up tab/"Xem PDF" fallback, download. The headless Chromium download into the scratchpad was interrupted; just open `http://localhost:8765/TMG/` in a browser after `python deploy/web/mock_server.py 8765`. |
| 5. VPS + Cloudflare + end-to-end | ⏳ not started. Add to the `kodingkat.xyz` block in `/etc/caddy/Caddyfile` **before** the SPA `try_files`: `redir /TMG /TMG/`, `handle_path /TMG/api/*` → `reverse_proxy https://<function-url-host>` with `header_up Host {upstream_hostport}` + `header_up x-tmg-secret <secret>`, `handle_path /TMG/*` → `root * /var/www/tmg` + `file_server`, and a `/TMG`-only CSP that allows `blob:` in `frame-src`/`object-src`. Back up the Caddyfile, run `caddy validate`, then reload. Add a Cloudflare rate-limit rule on `/TMG/api/analyze` if the token allows. Verify that kodingkat.xyz, a deep SPA route and shop.kodingkat.xyz still return 200. |

## Environment facts the next session needs

- **Modal** profile `its-dreowo`, about $25 of credit.
  - Volumes: `legal-clause-data` (raw + processed data, downloaded by Modal from Kaggle) and `legal-clause-models` (trained runs).
  - Secret `kaggle`.
  - Training GPU: `RTX-PRO-6000` (bf16, batch 64, ~7 min of training per model).
- **AWS** CLI user `claude`, Free plan; never upgrade to Paid. Account Lambda concurrency limit is 5. VPS `ssh -i ~/.ssh/aws-vps.pem admin@13.251.153.23` (t4g.small, Caddy, also serves `shop.kodingkat.xyz`).
- **Cloudflare** token in `~/.config/cloudflare/token` (strip a possible BOM).
- **Kaggle** token in `~/.kaggle/access_token`.
- **Docker** 29 works locally; about 15 GB of disk was free on 2026-10-31.
- **Commit trailers:** `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` plus the current session's `Claude-Session:` line.

## Suggested order next session

1. Rebuild the image, run `smoke.sh`, record cold-start and 30-page timings (they must stay under Cloudflare's 100 s).
2. Website Task 3 (AWS), then Task 5 (Caddy/Cloudflare), then an end-to-end test from a real browser.
3. Final whole-branch review of both plans (none has run yet).
4. When the team supplies the contracts: pipeline Task 13 (labels → Kappa → `compare` → `reports/results.md`), then fix the heading and STS false alarms.
