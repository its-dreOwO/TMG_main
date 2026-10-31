"""Local preview of the TMG page: serves web/ at /TMG/ and routes /TMG/api/* to the real
Lambda handler with the regex detector only (no large models, no AWS).

  python deploy/web/mock_server.py 8765   →  http://localhost:8765/TMG/
"""
import base64
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TMG_SECRET", "local")
os.environ["TMG_QUOTA"] = "off"

from lca.detectors.regex_rules import RegexDetector  # noqa: E402
from lca.web import handler as h  # noqa: E402

h._DETECTORS = [RegexDetector()]
WEB = ROOT / "web"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB), **kw)

    def _lambda(self, method):
        url = urlsplit(self.path)
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n) if n else b""
        event = {"rawPath": url.path, "headers": {"x-tmg-secret": os.environ["TMG_SECRET"]},
                 "queryStringParameters": dict(parse_qsl(url.query)),
                 "body": base64.b64encode(body).decode(), "isBase64Encoded": True,
                 "requestContext": {"http": {"method": method}, "requestId": "local"}}
        res = h.handler(event, None)
        out = res["body"].encode()
        self.send_response(res["statusCode"])
        for k, v in res["headers"].items():
            self.send_header(k, v)
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        if self.path.startswith("/TMG/api/"):
            return self._lambda("GET")
        if self.path == "/TMG":
            self.send_response(308)
            self.send_header("location", "/TMG/")
            return self.end_headers()
        if self.path.startswith("/TMG/"):
            self.path = self.path[len("/TMG"):]
            return super().do_GET()
        self.send_error(404)

    def do_POST(self):
        if self.path.startswith("/TMG/api/"):
            return self._lambda("POST")
        self.send_error(404)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
