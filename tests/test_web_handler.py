import base64
import json
import logging

import boto3
import pymupdf
import pytest
from botocore.stub import Stubber

from lca.detectors.regex_rules import RegexDetector
from lca.web import handler as h

SECRET = "s3cret"
SENTENCE = "The Vendor shall pay a penalty of 12% of the contract price."


class FakeCounter:
    def __init__(self, allow=True):
        self.allow, self.calls = allow, 0

    def try_consume(self, day, month):
        self.calls += 1
        return self.allow


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    monkeypatch.setenv("TMG_SECRET", SECRET)
    monkeypatch.setattr(h, "_COUNTER", FakeCounter())
    monkeypatch.setattr(h, "_DETECTORS", [RegexDetector()])


def pdf_bytes(pages=1, text=SENTENCE, image_only=False):
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page()
        if image_only:
            page.draw_rect(pymupdf.Rect(50, 50, 300, 300), fill=(0, 0, 0))
        else:
            page.insert_text((72, 72), text)
    return doc.tobytes()


def event(path="/analyze", body=b"", secret=SECRET, method="POST", query=None):
    headers = {"content-type": "application/pdf"}
    if secret is not None:
        headers["x-tmg-secret"] = secret
    return {"rawPath": path, "headers": headers, "queryStringParameters": query or {},
            "body": base64.b64encode(body).decode(), "isBase64Encoded": True,
            "requestContext": {"http": {"method": method}, "requestId": "r1"}}


def call(ev):
    res = h.handler(ev, None)
    return res["statusCode"], json.loads(res["body"])


def test_rejects_missing_secret():
    assert call(event(secret=None, body=pdf_bytes()))[0] == 403


def test_rejects_wrong_secret():
    assert call(event(secret="nope", body=pdf_bytes()))[0] == 403


def test_warm_does_not_consume_quota():
    status, body = call(event("/warm", method="GET"))
    assert status == 200 and body["ready"] is True and h._COUNTER.calls == 0


def test_rejects_non_pdf():
    status, body = call(event(body=b"hello world"))
    assert status == 400 and "PDF" in body["error"]


def test_rejects_empty_body():
    assert call(event(body=b""))[0] == 400


def test_rejects_over_4mb():
    status, body = call(event(body=b"%PDF-1.7\n" + b"0" * (4 * 1024 * 1024)))
    assert status == 413 and "4 MB" in body["error"]


def test_rejects_over_30_pages():
    status, body = call(event(body=pdf_bytes(pages=31)))
    assert status == 413 and "30" in body["error"]


def test_scanned_pdf_returns_422_message():
    status, body = call(event(body=pdf_bytes(image_only=True)))
    assert status == 422 and "scan" in body["error"]


def test_analyze_returns_highlighted_pdf_and_counts():
    status, body = call(event(body=pdf_bytes(), query={"contract_type": "commercial", "party": "Bên A"}))
    assert status == 200
    assert set(body["counts"]) == {"Cao", "Trung bình", "Thấp"} and sum(body["counts"].values()) >= 1
    doc = pymupdf.open(stream=base64.b64decode(body["pdf"]), filetype="pdf")
    page = doc[0]
    assert [a.type[1] for a in page.annots()] == ["Highlight"]
    assert body["filename"].endswith("_danh_dau.pdf")


def test_clean_pdf_has_zero_counts():
    status, body = call(event(body=pdf_bytes(text="This Agreement is signed by both parties today.")))
    assert status == 200 and sum(body["counts"].values()) == 0


def test_quota_exhausted_returns_429(monkeypatch):
    monkeypatch.setattr(h, "_COUNTER", FakeCounter(allow=False))
    status, body = call(event(body=pdf_bytes()))
    assert status == 429 and body["error"]


@pytest.mark.parametrize("query", [{"contract_type": "x"}, {"party": "Bên C"}])
def test_invalid_params_400(query):
    assert call(event(body=pdf_bytes(), query=query))[0] == 400


def test_unknown_path_404():
    assert call(event("/nope", method="GET"))[0] == 404


def test_no_text_logged(caplog):
    with caplog.at_level(logging.DEBUG):
        call(event(body=pdf_bytes()))
    assert "penalty" not in caplog.text and "Vendor" not in caplog.text


def _stubbed_counter():
    client = boto3.client("dynamodb", region_name="ap-southeast-1",
                          aws_access_key_id="x", aws_secret_access_key="x")
    return h.DynamoUsageCounter("tbl", client=client), Stubber(client)


def test_dynamo_counter_consumes_month_then_day():
    counter, stub = _stubbed_counter()
    stub.add_response("update_item", {})
    stub.add_response("update_item", {})
    with stub:
        assert counter.try_consume("2026-11-01", "2026-11") is True
    stub.assert_no_pending_responses()


def test_dynamo_counter_month_limit_reached():
    counter, stub = _stubbed_counter()
    stub.add_client_error("update_item", "ConditionalCheckFailedException")
    with stub:
        assert counter.try_consume("2026-11-01", "2026-11") is False


def test_dynamo_counter_day_limit_refunds_month():
    counter, stub = _stubbed_counter()
    stub.add_response("update_item", {})
    stub.add_client_error("update_item", "ConditionalCheckFailedException")
    stub.add_response("update_item", {})  # month refund
    with stub:
        assert counter.try_consume("2026-11-01", "2026-11") is False
    stub.assert_no_pending_responses()
