"""Inference backends for the RS-adapted VLM.

``HFBackend``      loads the base VLM with transformers and applies the LoRA adapter
                   trained on BigEarthNet.txt (training/finetune_vlm.py).
``RemoteBackend``  calls a GPU server running scripts/serve_vlm.py (e.g. on Colab/Kaggle),
                   so the web app can run on a CPU-only machine.

Both return ``(text, confidence)`` where confidence is the geometric-mean token
probability of the generated answer (None if unavailable).
"""
from __future__ import annotations

import base64
import io
import threading
from typing import Protocol

from PIL import Image

from satquery.models.vlm_prompts import chat_messages


class VLMBackend(Protocol):
    name: str

    def generate(self, images: list[Image.Image], prompt: str, max_new_tokens: int = 128) -> tuple[str, float | None]: ...


class HFBackend:
    def __init__(self, base_model: str, adapter: str | None = None, device: str | None = None):
        self.base_model, self.adapter = base_model, adapter
        self.name = f"{base_model}" + (f" + LoRA({adapter})" if adapter else " (no RS adapter!)")
        self._device = device
        self._model = None
        self._processor = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (
            torch.float16 if device == "cuda" else torch.float32)
        self._processor = AutoProcessor.from_pretrained(self.base_model)
        model = AutoModelForImageTextToText.from_pretrained(self.base_model, dtype=dtype)
        if self.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.adapter)
            model = model.merge_and_unload()
        self._model = model.to(device).eval()
        self._device = device

    def generate(self, images: list[Image.Image], prompt: str, max_new_tokens: int = 128) -> tuple[str, float | None]:
        import torch

        with self._lock:
            if self._model is None:
                self._load()
            inputs = self._processor.apply_chat_template(
                chat_messages(images, prompt), tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt").to(self._device)
            with torch.inference_mode():
                out = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                           output_scores=True, return_dict_in_generate=True)
            gen = out.sequences[:, inputs["input_ids"].shape[1]:]
            text = self._processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
            conf = None
            try:
                ts = self._model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
                conf = float(torch.exp(ts[0].float().mean()).item())
            except Exception:
                pass
            return text, conf


class RemoteBackend:
    def __init__(self, endpoint: str, timeout_s: float = 180.0):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout_s
        self.name = f"remote({self.endpoint})"
        self._info: dict | None = None

    def info(self) -> dict:
        import httpx
        if self._info is None:
            self._info = httpx.get(f"{self.endpoint}/health", timeout=10).json()
            self.name = f"remote:{self._info.get('model', '?')}"
        return self._info

    def generate(self, images: list[Image.Image], prompt: str, max_new_tokens: int = 128) -> tuple[str, float | None]:
        import httpx

        enc = []
        for im in images:
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            enc.append(base64.b64encode(buf.getvalue()).decode("ascii"))
        r = httpx.post(f"{self.endpoint}/generate", json={"prompt": prompt, "images": enc,
                                                          "max_new_tokens": max_new_tokens}, timeout=self.timeout)
        r.raise_for_status()
        d = r.json()
        return d["text"], d.get("confidence")
