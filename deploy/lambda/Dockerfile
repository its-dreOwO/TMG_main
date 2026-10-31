# Lambda container for kodingkat.xyz/TMG. Build from the project root:
#   docker build -f deploy/lambda/Dockerfile -t tmg-lca .
FROM public.ecr.aws/lambda/python:3.12

ENV HOME=/opt/home \
    HF_HOME=/opt/cache/hf \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4 \
    PYTHONDONTWRITEBYTECODE=1

COPY deploy/lambda/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY lca ${LAMBDA_TASK_ROOT}/lca
COPY artifacts/models/xlmr/final ${LAMBDA_TASK_ROOT}/artifacts/models/xlmr/final
COPY artifacts/models/sts_thresholds.json ${LAMBDA_TASK_ROOT}/artifacts/models/sts_thresholds.json
COPY deploy/lambda/prefetch.py /tmp/prefetch.py

RUN mkdir -p /opt/home /opt/cache/hf && cd ${LAMBDA_TASK_ROOT} && PYTHONPATH=${LAMBDA_TASK_ROOT} python /tmp/prefetch.py \
    && rm /tmp/prefetch.py && chmod -R a+rX /opt/home /opt/cache

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

CMD ["lca.web.handler.handler"]
