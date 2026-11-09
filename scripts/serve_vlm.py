"""GPU inference server for SatQuery-VLM (run on Colab/Kaggle/any CUDA box).

  pip install -r requirements-train.txt
  python scripts/serve_vlm.py --adapter checkpoints/satquery-vlm-lora --port 8100
  # expose it, e.g.:  cloudflared tunnel --url http://localhost:8100

Then on the machine running the web app:
  SATQUERY_VLM_BACKEND=remote SATQUERY_VLM_ENDPOINT=https://<tunnel-host>  uvicorn satquery.api.app:app

API: GET /health -> {"model": ...};  POST /generate {"prompt", "images": [base64 PNG], "max_new_tokens"}
     -> {"text", "confidence"}
"""
from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_app(base: str, adapter: str | None):
    from fastapi import FastAPI
    from PIL import Image
    from pydantic import BaseModel, Field

    from satquery.models.vlm_backend import HFBackend

    backend = HFBackend(base, adapter)
    app = FastAPI(title="SatQuery-VLM server")

    class Req(BaseModel):
        prompt: str
        images: list[str] = Field(default_factory=list, max_length=4)
        max_new_tokens: int = Field(160, ge=1, le=1024)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": backend.name}

    @app.post("/generate")
    def generate(req: Req) -> dict:
        imgs = [Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB") for s in req.images]
        text, conf = backend.generate(imgs, req.prompt, req.max_new_tokens)
        return {"text": text, "confidence": conf}

    return app


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    a = ap.parse_args()
    uvicorn.run(build_app(a.base, a.adapter), host=a.host, port=a.port)
