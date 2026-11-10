# SatQuery AI - production image (CPU). Runs on Hugging Face Spaces, any Docker host, or a VPS.
#   docker build -t satquery .   &&   docker run -p 8000:7860 satquery   ->  http://localhost:8000
# Trained models are baked in from checkpoints/ when present in the build context (they are not in git;
# scripts/deploy_hf_space.py uploads them with the code). Without them the app runs on measurement tools only.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/opt/torch MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# CPU-only PyTorch (about 200 MB instead of several GB of CUDA libraries), then the app's dependencies
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && pip install -r requirements.txt

# ImageNet weights used by the high-resolution models: fetched at build time, not on the first request
RUN python -c "import torchvision.models as m; m.resnet18(weights=m.ResNet18_Weights.IMAGENET1K_V1)" \
 && chmod -R a+rX /opt/torch

COPY . .
RUN python -c "from satquery.demo.make import load_manifest; load_manifest()" \
 && test -f satquery/web/dist/index.html \
 && python -c "from pathlib import Path; m = sorted(p.name for p in Path('checkpoints').glob('*.pt')); print('trained models in image:', m or 'none (measurement tools only)')" \
 && useradd -m -u 1000 app && mkdir -p runs && chown app runs

USER app

# Public-deployment defaults; override any of them with environment variables on the host.
ENV PORT=7860 \
    SATQUERY_PERSIST_RUNS=0 \
    SATQUERY_RATE_LIMIT=20 \
    SATQUERY_MAX_CONCURRENT=2 \
    SATQUERY_MAX_UPLOAD_MB=100
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT', '7860'), timeout=4)"

# one process: the rate limiter, demo cache and report store live in memory; analyses run on threads.
# --proxy-headers makes the rate limiter see real client addresses behind a hosting proxy.
CMD ["sh", "-c", "exec uvicorn satquery.api.app:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips=\"${FORWARDED_ALLOW_IPS:-*}\" --timeout-keep-alive 30"]
