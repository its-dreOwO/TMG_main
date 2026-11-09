"""Build the Lambda image on Modal without Docker, then push it to ECR.

Modal cannot run a Docker daemon, so this assembles the same image the Dockerfile describes:
1. `build` runs inside the Lambda base image itself, pip-installs the pinned deps into /var/task
   (AWS's documented layout), adds lca/ and the models, runs prefetch.py, and packs
   /var/task + /opt/home + /opt/cache into one layer tarball on a Modal volume.
   The model comes from the local copy uploaded with
   `modal volume put tmg-lca-build artifacts/models/xlmr/final /xlmr-final`.
2. `smoke` extracts that layer into a fresh base-image container with the network blocked and
   calls the handler directly (cold start, 1 page, 30 pages, wrong secret).
3. `push` stacks the layer on the base image with crane, sets env and CMD, and pushes to ECR.

Run from the project root:
    ECR_TOKEN=$(aws ecr get-login-password) modal run deploy/lambda/modal_build.py
The ECR token lives only in an ephemeral Modal secret for this run.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

BASE = "public.ecr.aws/lambda/python:3.12"
REGISTRY = "455959165143.dkr.ecr.ap-southeast-1.amazonaws.com"
REPO = f"{REGISTRY}/tmg-lca"
CRANE = "https://github.com/google/go-containerregistry/releases/download/v0.20.6/go-containerregistry_Linux_x86_64.tar.gz"
MODEL_SHA_PREFIX = "049a181eefb110e3"  # sha256 of the local artifacts/models/xlmr/final/model.safetensors
LAYER = "/out/layer.tar.gz"

RUNTIME_ENV = {
    "HOME": "/opt/home",
    "HF_HOME": "/opt/cache/hf",
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "4",
    "PYTHONDONTWRITEBYTECODE": "1",
}

here = Path(__file__).resolve().parent
root = here.parent.parent
image = (
    modal.Image.from_registry(BASE)
    .entrypoint([])
    .add_local_dir(root / "lca", "/src/lca", ignore=["**/__pycache__"])
    .add_local_file(here / "requirements.txt", "/src/requirements.txt")
    .add_local_file(here / "prefetch.py", "/src/prefetch.py")
    .add_local_file(root / "artifacts/models/sts_thresholds.json", "/src/sts_thresholds.json")
)
out = modal.Volume.from_name("tmg-lca-build", create_if_missing=True)
ecr = modal.Secret.from_dict({"ECR_TOKEN": os.environ.get("ECR_TOKEN", "")})
app = modal.App("tmg-lca-build")


def sh(cmd: list[str], **kw) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def clean_env(extra: dict) -> dict:
    """Environment for child processes, free of Modal's own PYTHONPATH tweaks."""
    keep = {k: os.environ[k] for k in ("PATH", "LANG", "LAMBDA_TASK_ROOT") if k in os.environ}
    return {**keep, **RUNTIME_ENV, **extra}


@app.function(image=image, volumes={"/out": out}, cpu=4, memory=8192, timeout=3600)
def build() -> str:
    import hashlib
    import shutil
    import tarfile

    task = Path("/var/task")
    shutil.copytree("/src/lca", task / "lca")
    dest = task / "artifacts/models/xlmr/final"
    shutil.copytree("/out/xlmr-final", dest)  # uploaded from the local, tested copy
    shutil.copy("/src/sts_thresholds.json", task / "artifacts/models/sts_thresholds.json")
    for path in (Path("/out/xlmr-final/model.safetensors"), dest / "model.safetensors"):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        assert h.hexdigest().startswith(MODEL_SHA_PREFIX), f"{path} differs from local: {h.hexdigest()}"

    sh([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--disable-pip-version-check",
        "-r", "/src/requirements.txt", "--target", str(task)])
    for d in ("/opt/home", "/opt/cache/hf"):
        Path(d).mkdir(parents=True, exist_ok=True)
    # Run from /tmp like the Dockerfile: Python puts the script's folder first on sys.path,
    # and /src/lca must not shadow /var/task/lca.
    shutil.copy("/src/prefetch.py", "/tmp/prefetch.py")
    sh([sys.executable, "/tmp/prefetch.py"], cwd=task,
       env=clean_env({"PYTHONPATH": str(task), "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}))

    def readable(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        info.uid = info.gid = 0
        info.uname = info.gname = "root"
        info.mode |= 0o555 if info.isdir() or info.mode & 0o111 else 0o444
        return info

    t = time.monotonic()
    with tarfile.open(LAYER, "w:gz", compresslevel=6) as tar:
        for d in ("var/task", "opt/home", "opt/cache"):
            tar.add("/" + d, arcname=d, filter=readable)
    out.commit()
    size = Path(LAYER).stat().st_size / 2**20
    usage = subprocess.run(["du", "-sh", "/var/task", "/opt/cache", "/opt/home"], capture_output=True, text=True)
    print(usage.stdout)
    return f"layer {size:.0f} MiB (gzip) in {time.monotonic() - t:.0f}s"


@app.function(image=image, volumes={"/out": out}, cpu=4, memory=6144, timeout=1800, block_network=True)
def smoke(pdf1: bytes, pdf30: bytes) -> str:
    import tarfile

    with tarfile.open(LAYER) as tar:
        tar.extractall("/", filter="fully_trusted")
    Path("/tmp/smoke_1.pdf").write_bytes(pdf1)
    Path("/tmp/smoke_30.pdf").write_bytes(pdf30)
    code = r'''
import base64, json, time
t0 = time.monotonic()
from lca.web.handler import handler
print(f"import: {time.monotonic() - t0:.1f}s", flush=True)
def call(path, body=b"", method="POST", secret="smoke-secret"):
    ev = {"rawPath": path, "headers": {"x-tmg-secret": secret}, "queryStringParameters": {},
          "body": base64.b64encode(body).decode(), "isBase64Encoded": True,
          "requestContext": {"http": {"method": method}, "requestId": "smoke"}}
    t = time.monotonic(); res = handler(ev, None); sec = time.monotonic() - t
    body = json.loads(res["body"]); pdf = body.pop("pdf", "")
    print(f"{path} {method}: {res['statusCode']} in {sec:.1f}s, pdf {len(pdf) * 3 // 4} bytes -> {body}", flush=True)
call("/warm", method="GET")
call("/analyze", open("/tmp/smoke_1.pdf", "rb").read())
call("/analyze", open("/tmp/smoke_30.pdf", "rb").read())
call("/warm", method="GET", secret="wrong")
# The base image's boto3/botocore must still work next to the deps in /var/task.
import boto3
from botocore.stub import Stubber
from lca.web.handler import DynamoUsageCounter
client = boto3.client("dynamodb", region_name="ap-southeast-1", aws_access_key_id="x", aws_secret_access_key="x")
with Stubber(client) as stub:
    stub.add_response("update_item", {})  # 1st: month +1
    stub.add_response("update_item", {})  # 1st: day +1
    stub.add_response("update_item", {})  # 2nd: month +1
    stub.add_client_error("update_item", "ConditionalCheckFailedException")  # 2nd: day cap hit
    stub.add_response("update_item", {})  # 2nd: month rolled back
    c = DynamoUsageCounter("tmg-lca-usage", client)
    ok, capped = c.try_consume("2026-11-09", "2026-11"), c.try_consume("2026-11-09", "2026-11")
    stub.assert_no_pending_responses()
print(f"boto3 {boto3.__version__} from {boto3.__file__}: consume={ok}, at day cap={capped}", flush=True)
assert ok and not capped
'''
    env = clean_env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TMG_SECRET": "smoke-secret",
                     "TMG_QUOTA": "off", "PYTHONPATH": "/var/task"})
    # Mirror Lambda: the runtime puts /var/task first on sys.path and runs from there.
    res = subprocess.run([sys.executable, "-I", "-c", "import sys; sys.path.insert(0, '/var/task'); exec(sys.stdin.read())"],
                         input=code, text=True, capture_output=True, cwd="/var/task", env=env)
    return res.stdout + res.stderr[-4000:] + f"\nexit {res.returncode}"


@app.function(image=image, volumes={"/out": out}, secrets=[ecr], cpu=2, memory=4096, timeout=3600)
def push(tag: str) -> str:
    import io
    import tarfile
    import urllib.request

    with urllib.request.urlopen(CRANE) as r, tarfile.open(fileobj=io.BytesIO(r.read())) as tar:
        tar.extract("crane", "/usr/local/bin", filter="fully_trusted")
    crane = "/usr/local/bin/crane"
    token = os.environ["ECR_TOKEN"]
    assert token, "ECR_TOKEN missing: run with ECR_TOKEN=$(aws ecr get-login-password)"
    subprocess.run([crane, "auth", "login", REGISTRY, "-u", "AWS", "--password-stdin"], input=token, text=True, check=True)
    digest = subprocess.run([crane, "digest", "--platform", "linux/amd64", BASE],
                            capture_output=True, text=True, check=True).stdout.strip()
    staging = f"{REPO}:staging-{tag}"
    sh([crane, "append", "-b", f"{BASE}@{digest}", "-f", LAYER, "-t", staging])
    env = RUNTIME_ENV | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    args = [crane, "mutate", staging, "-t", f"{REPO}:{tag}", "--cmd", "lca.web.handler.handler"]
    for k, v in env.items():
        args += ["--env", f"{k}={v}"]
    sh(args)
    subprocess.run([crane, "delete", staging])  # best effort; the ECR lifecycle drops leftovers
    final = subprocess.run([crane, "digest", f"{REPO}:{tag}"], capture_output=True, text=True, check=True).stdout.strip()
    config = subprocess.run([crane, "config", f"{REPO}:{tag}"], capture_output=True, text=True, check=True).stdout
    return f"{REPO}@{final}\n{config}"


@app.local_entrypoint()
def main(step: str = "all", tag: str = ""):
    tag = tag or time.strftime("%Y%m%d-%H%M")
    if step in ("all", "build"):
        print(build.remote())
    if step in ("all", "smoke"):
        sys.path.insert(0, str(here))
        import tempfile

        from make_sample import make

        with tempfile.TemporaryDirectory() as tmp:
            make(1, Path(tmp) / "1.pdf")
            make(30, Path(tmp) / "30.pdf")
            print(smoke.remote((Path(tmp) / "1.pdf").read_bytes(), (Path(tmp) / "30.pdf").read_bytes()))
    if step in ("all", "push"):
        print(push.remote(tag))
