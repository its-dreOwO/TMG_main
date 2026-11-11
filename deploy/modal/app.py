"""Backend for kodingkat.xyz/TMG on Modal (the AWS account caps Lambda memory at 512 MB).

It serves the image built by deploy/lambda/modal_build.py (`--step build`): the layer on the
`tmg-lca-build` volume is baked into a Modal image. `Analyzer` runs lca.web.handler (already
tested) on Lambda-style events in that image; `web` is a light front end that starts each
analysis as a background job and lets the page poll for the result.

Caps (20 a day, 100 a month) live in a JSON file on the `tmg-lca-usage` volume. Only one
`consume` container ever runs and it takes one call at a time, so the read-modify-write is atomic.
A per-IP limiter (3 analyses and 6 warm-ups per 10 minutes) sits in front of those caps.
Only counters are stored (IPs as keyed hashes), never contract content.

Deploy from the project root:
    modal secret create --force tmg-lca TMG_SECRET="$(cat ~/.config/tmg/secret)"
    modal deploy deploy/modal/app.py
Callers must send Modal-Key / Modal-Secret from a workspace proxy token (Caddy does).
"""
import hashlib
import hmac
import json
import os
import re
import sys
import time
from urllib.parse import parse_qsl

import modal

BASE = "public.ecr.aws/lambda/python:3.12"
ENV = {
    "HOME": "/opt/home",
    "HF_HOME": "/opt/cache/hf",
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "4",
    "PYTHONDONTWRITEBYTECODE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
build_volume = modal.Volume.from_name("tmg-lca-build")
# Per-visitor rate limits: keys hold a keyed hash of the IP and a time window, values a count.
# Modal drops Dict entries that go unused for a while, so old windows clean themselves up.
rate_limits = modal.Dict.from_name("tmg-lca-ratelimit", create_if_missing=True)
usage_volume = modal.Volume.from_name("tmg-lca-usage", create_if_missing=True)
app = modal.App("tmg-lca")


def _bake() -> None:
    import tarfile

    with tarfile.open("/out/layer.tar.gz") as tar:
        tar.extractall("/", filter="fully_trusted")


web_image = (
    modal.Image.from_registry(BASE)
    .entrypoint([])
    .run_function(_bake, volumes={"/out": build_volume})
    .env(ENV)
)
counter_image = modal.Image.debian_slim("3.12")


@app.function(image=counter_image, volumes={"/usage": usage_volume}, max_containers=1, timeout=30,
              scaledown_window=300)
def consume(day: str, month: str, day_limit: int, month_limit: int) -> bool:
    """Take one analysis from both caps, or none if either is used up."""
    path = "/usage/counts.json"
    usage_volume.reload()
    try:
        with open(path) as f:
            counts = json.load(f)
    except FileNotFoundError:
        counts = {}
    d, m = counts.get(f"day#{day}", 0), counts.get(f"month#{month}", 0)
    if d >= day_limit or m >= month_limit:
        return False
    # Keep only the current day and month: nothing older is ever needed.
    counts = {f"day#{day}": d + 1, f"month#{month}": m + 1}
    with open(path + ".tmp", "w") as f:
        json.dump(counts, f)
    os.replace(path + ".tmp", path)
    usage_volume.commit()
    return True


class ModalCounter:
    """Drop-in for the handler's DynamoUsageCounter."""

    def __init__(self, day_limit: int, month_limit: int):
        self.day_limit, self.month_limit = day_limit, month_limit

    def try_consume(self, day: str, month: str) -> bool:
        return consume.remote(day, month, self.day_limit, self.month_limit)


# A full 4-CPU reservation: with 0.25 CPU reserved the container did not get to burst and a
# 30-page contract took 139 s. Memory is not throttled, so its low reservation keeps idle cost down.
@app.cls(image=web_image, secrets=[modal.Secret.from_name("tmg-lca")], cpu=4.0, memory=(1024, 6144),
         timeout=600, max_containers=2, scaledown_window=60)
class Analyzer:
    """Runs the Lambda handler unchanged; models load once per container."""

    @modal.enter()
    def load(self):
        os.chdir("/var/task")  # the Lambda runtime's working directory
        sys.path.insert(0, "/var/task")  # like Lambda: the deps in /var/task come first
        from lca.web import handler

        handler._COUNTER = ModalCounter(handler.DAY_LIMIT, handler.MONTH_LIMIT)
        self.handler = handler

    @modal.method()
    def run(self, event: dict) -> dict:
        return self.handler.handler(event, None)


MAX_BYTES = 20 * 1024 * 1024  # same as lca.web.handler.MAX_BYTES; checked here before reading
RATE_WINDOW_S = 600
RATE_LIMITS = {"analyze": 3, "warm": 6}  # per visitor IP per 10 minutes
JOB = re.compile(r"^(fc-[A-Za-z0-9]{10,40})\.([0-9a-f]{32})$")
REASONS = {200: "OK", 202: "Accepted", 400: "Bad Request", 403: "Forbidden", 404: "Not Found",
           413: "Payload Too Large", 422: "Unprocessable Entity", 429: "Too Many Requests",
           500: "Internal Server Error"}


def _json(status: int, payload: dict) -> dict:
    return {"statusCode": status, "headers": {"content-type": "application/json; charset=utf-8",
                                              "cache-control": "no-store"},
            "body": json.dumps(payload, ensure_ascii=False)}


def _allow(kind: str, ip: str, secret: str) -> bool:
    """Approximate fixed-window limit per IP (two front containers may race; good enough)."""
    who = hmac.new(secret.encode(), ip.encode(), hashlib.sha256).hexdigest()[:16]
    key = f"{kind}:{who}:{int(time.time() // RATE_WINDOW_S)}"
    n = rate_limits.get(key, 0)
    if n >= RATE_LIMITS[kind]:
        return False
    rate_limits[key] = n + 1
    return True


def _sign(call_id: str, secret: str) -> str:
    return hmac.new(secret.encode(), call_id.encode(), hashlib.sha256).hexdigest()[:32]


# Requests return within a second or two whatever the contract length, so Cloudflare's 100 s
# limit never applies: POST /analyze starts a background job and answers 202 with a signed job id,
# then the page polls GET /result?job=... until the job's response is ready.
@app.function(image=counter_image, secrets=[modal.Secret.from_name("tmg-lca")], cpu=0.25, memory=512,
              timeout=60, max_containers=2, scaledown_window=120)
# Modal refuses requests without the proxy-token headers before starting any container, so
# only Caddy (which holds the token) can wake this up.
@modal.wsgi_app(requires_proxy_auth=True)
def web():
    import base64

    secret = os.environ["TMG_SECRET"]
    analyzer = Analyzer()

    def route(environ) -> dict:
        given = environ.get("HTTP_X_TMG_SECRET", "")
        if not hmac.compare_digest(given.encode(), secret.encode()):
            return _json(403, {"error": "Forbidden"})
        path, method = environ.get("PATH_INFO", ""), environ.get("REQUEST_METHOD", "GET")
        query = dict(parse_qsl(environ.get("QUERY_STRING", ""), keep_blank_values=True))
        # Cloudflare sets CF-Connecting-IP to the visitor's address; Caddy passes it through.
        ip = environ.get("HTTP_CF_CONNECTING_IP") or environ.get("REMOTE_ADDR", "-")

        if path.endswith("/warm"):
            if not _allow("warm", ip, secret):  # page reloads must not keep the big container up
                return _json(200, {"ready": False, "warming": False})
            analyzer.run.spawn({"rawPath": "/warm", "headers": {"x-tmg-secret": given},
                                "requestContext": {"http": {"method": "GET"}, "requestId": "warm"}})
            return _json(200, {"ready": False, "warming": True})

        if path.endswith("/analyze") and method == "POST":
            length = int(environ.get("CONTENT_LENGTH") or 0)
            if length > MAX_BYTES:
                return _json(413, {"error": "File quá lớn. Giới hạn là 20 MB."})
            if not _allow("analyze", ip, secret):
                return _json(429, {"error": "Bạn gửi quá nhiều yêu cầu. Vui lòng đợi vài phút rồi thử lại."})
            body = environ["wsgi.input"].read(length) if length else b""
            call = analyzer.run.spawn({
                "rawPath": "/analyze", "headers": {"x-tmg-secret": given},
                "queryStringParameters": query,
                "body": base64.b64encode(body).decode(), "isBase64Encoded": True,
                "requestContext": {"http": {"method": "POST"},
                                   "requestId": environ.get("HTTP_CF_RAY", "-")}})
            return _json(202, {"job": f"{call.object_id}.{_sign(call.object_id, secret)}"})

        if path.endswith("/result") and method == "GET":
            m = JOB.match(query.get("job", ""))
            if not m or not hmac.compare_digest(m.group(2), _sign(m.group(1), secret)):
                return _json(404, {"error": "Không tìm thấy kết quả. Vui lòng phân tích lại."})
            try:
                return modal.FunctionCall.from_id(m.group(1)).get(timeout=0)
            except TimeoutError:
                return _json(202, {"status": "pending"})
            except modal.exception.OutputExpiredError:
                return _json(404, {"error": "Kết quả đã hết hạn. Vui lòng phân tích lại."})
            except Exception:
                return _json(500, {"error": "Có lỗi khi phân tích. Vui lòng thử lại."})

        return _json(404, {"error": "Not found"})

    def wsgi(environ, start_response):
        res = route(environ)
        out = res["body"].encode()
        start_response(f"{res['statusCode']} {REASONS.get(res['statusCode'], '')}",
                       list(res["headers"].items()) + [("content-length", str(len(out)))])
        return [out]

    return wsgi
