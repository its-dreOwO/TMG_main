"""AWS Lambda (Function URL) entry point for kodingkat.xyz/TMG.

Routes: GET .../warm loads the models; POST .../analyze takes raw PDF bytes and returns
the highlighted PDF (base64) plus finding counts. Contract content is never stored or logged.
"""
from __future__ import annotations

import base64
import hmac
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import pymupdf

from lca.pipeline import DEFAULT_DETECTORS, analyze, build_detectors
from lca.reader import UnreadableDocumentError
from lca.types import AnalysisContext, Level
from lca.writers import write_pdf

MAX_BYTES = 4 * 1024 * 1024
MAX_PAGES = 30
DAY_LIMIT = 20
MONTH_LIMIT = 100
CONTRACT_TYPES = {"commercial", "labour", "other"}
PARTIES = {"Bên A", "Bên B"}

log = logging.getLogger("tmg")
log.setLevel(logging.INFO)

_COUNTER = None
_DETECTORS = None


class UsageCounter(Protocol):
    def try_consume(self, day: str, month: str) -> bool: ...


class DynamoUsageCounter:
    """Atomic daily/monthly caps; stores only counters, never content."""

    def __init__(self, table: str, client=None):
        import boto3
        self.table = table
        self.client = client or boto3.client("dynamodb")

    def _add(self, key: str, delta: int, limit: int | None) -> bool:
        from botocore.exceptions import ClientError
        kwargs = {"TableName": self.table, "Key": {"pk": {"S": key}},
                  "UpdateExpression": "ADD n :d",
                  "ExpressionAttributeValues": {":d": {"N": str(delta)}}}
        if limit is not None:
            kwargs["ConditionExpression"] = "attribute_not_exists(n) OR n < :lim"
            kwargs["ExpressionAttributeValues"][":lim"] = {"N": str(limit)}
        try:
            self.client.update_item(**kwargs)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                return False
            raise

    def try_consume(self, day: str, month: str) -> bool:
        if not self._add(f"month#{month}", 1, MONTH_LIMIT):
            return False
        if not self._add(f"day#{day}", 1, DAY_LIMIT):
            self._add(f"month#{month}", -1, None)
            return False
        return True


def _counter() -> UsageCounter:
    global _COUNTER
    if _COUNTER is None:
        _COUNTER = DynamoUsageCounter(os.environ["TMG_TABLE"])
    return _COUNTER


def _detectors():
    global _DETECTORS
    if _DETECTORS is None:
        _DETECTORS = build_detectors(DEFAULT_DETECTORS)
    return _DETECTORS


def _reply(status: int, payload: dict) -> dict:
    return {"statusCode": status, "headers": {"content-type": "application/json; charset=utf-8",
                                              "cache-control": "no-store"},
            "body": json.dumps(payload, ensure_ascii=False)}


def _error(status: int, message: str) -> dict:
    return _reply(status, {"error": message})


def _authorized(event: dict) -> bool:
    expected = os.environ.get("TMG_SECRET", "")
    given = (event.get("headers") or {}).get("x-tmg-secret", "")
    return bool(expected) and hmac.compare_digest(given.encode(), expected.encode())


def _body(event: dict) -> bytes:
    raw = event.get("body") or ""
    return base64.b64decode(raw) if event.get("isBase64Encoded") else raw.encode("latin-1")


def handler(event: dict, context) -> dict:
    if not _authorized(event):
        return _error(403, "Forbidden")
    path = event.get("rawPath", "")
    method = event.get("requestContext", {}).get("http", {}).get("method", "GET")
    if path.endswith("/warm"):
        _detectors()
        return _reply(200, {"ready": True})
    if path.endswith("/analyze") and method == "POST":
        return _analyze(event)
    return _error(404, "Not found")


def _analyze(event: dict) -> dict:
    started = time.monotonic()
    request_id = event.get("requestContext", {}).get("requestId", "-")
    query = event.get("queryStringParameters") or {}
    contract_type = query.get("contract_type", "commercial")
    party = query.get("party") or None
    if contract_type not in CONTRACT_TYPES or (party is not None and party not in PARTIES):
        return _error(400, "Tham số không hợp lệ.")

    data = _body(event)
    if len(data) > MAX_BYTES:
        return _error(413, "File quá lớn. Giới hạn là 4 MB.")
    if not data.startswith(b"%PDF-"):
        return _error(400, "Vui lòng chọn một file PDF.")
    try:
        with pymupdf.open(stream=data, filetype="pdf") as probe:
            pages = probe.page_count
    except Exception:
        return _error(422, "Không đọc được file PDF (file hỏng).")
    if pages > MAX_PAGES:
        return _error(413, f"File có {pages} trang. Giới hạn là {MAX_PAGES} trang.")

    now = datetime.now(timezone.utc)
    if not _counter().try_consume(now.strftime("%Y-%m-%d"), now.strftime("%Y-%m")):
        return _error(429, "Đã đạt giới hạn sử dụng của bản demo. Vui lòng thử lại sau.")

    ctx = AnalysisContext(contract_type, party)
    with tempfile.TemporaryDirectory() as tmp:  # removed before returning
        src = Path(tmp) / "hop_dong.pdf"
        out = Path(tmp) / "hop_dong_danh_dau.pdf"
        src.write_bytes(data)
        try:
            result = analyze(src, ctx, _detectors(), "sentence")
        except UnreadableDocumentError as exc:
            return _error(422, str(exc))
        write_pdf(src, result.document, result.findings, out)
        highlighted = out.read_bytes()

    counts = {lvl.vi: 0 for lvl in (Level.HIGH, Level.MEDIUM, Level.LOW)}
    for f in result.findings:
        counts[f.level.vi] += 1
    log.info(json.dumps({"request": request_id, "pages": pages, "seconds": round(time.monotonic() - started, 2),
                         "counts": counts}, ensure_ascii=False))
    return _reply(200, {"pdf": base64.b64encode(highlighted).decode(), "counts": counts,
                        "filename": "hop_dong_danh_dau.pdf"})
