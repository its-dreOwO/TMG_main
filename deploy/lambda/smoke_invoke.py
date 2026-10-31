"""Runs inside the container: invoke the local Lambda emulator like a Function URL would."""
import base64
import json
import sys
import time
import urllib.request


URL = "http://localhost:8080/2015-03-31/functions/function/invocations"
SECRET = sys.argv[1]


def invoke(path, body=b"", method="POST", query=None):
    event = {"rawPath": path, "headers": {"x-tmg-secret": SECRET}, "queryStringParameters": query or {},
             "body": base64.b64encode(body).decode(), "isBase64Encoded": True,
             "requestContext": {"http": {"method": method}, "requestId": "smoke"}}
    t = time.monotonic()
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(event).encode()), timeout=300) as r:
        res = json.loads(r.read())
    return res, time.monotonic() - t


def contract(pages):
    with open(f"/tmp/smoke_{pages}.pdf", "rb") as f:
        return f.read()


for name, args in [("warm (cold start)", ("/warm", b"", "GET")),
                   ("analyze 1 page", ("/analyze", contract(1))),
                   ("analyze 30 pages", ("/analyze", contract(30))),
                   ("no secret", None)]:
    if args is None:
        SECRET, saved = "wrong", SECRET
        res, sec = invoke("/warm", method="GET")
        SECRET = saved
    else:
        res, sec = invoke(*args)
    body = json.loads(res.get("body", "{}"))
    body.pop("pdf", None)
    print(f"{name}: status {res.get('statusCode')} in {sec:.1f}s -> {body}")
