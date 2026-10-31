#!/usr/bin/env bash
# Smoke-test the image locally with networking disabled (proves no run-time downloads).
set -euo pipefail
IMAGE=${1:-tmg-lca}
NAME=tmg-lca-smoke
docker rm -f $NAME >/dev/null 2>&1 || true
docker run -d --name $NAME --network none --memory 6g --cpus 4 \
  -e TMG_SECRET=smoke-secret -e TMG_QUOTA=off "$IMAGE" >/dev/null
trap 'docker rm -f $NAME >/dev/null' EXIT
TMP=$(mktemp -d)
"${PYTHON:-python3}" "$(dirname "$0")/make_sample.py" "$TMP"
docker cp "$(dirname "$0")/smoke_invoke.py" $NAME:/tmp/smoke_invoke.py
docker cp "$TMP/smoke_1.pdf" $NAME:/tmp/smoke_1.pdf
docker cp "$TMP/smoke_30.pdf" $NAME:/tmp/smoke_30.pdf
rm -rf "$TMP"
sleep 2
docker exec -e AWS_LAMBDA_FUNCTION_TIMEOUT=90 $NAME python /tmp/smoke_invoke.py smoke-secret
