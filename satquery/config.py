"""Runtime configuration, read once from environment variables.

Every knob has a safe default so the system runs out of the box on a CPU-only
laptop: the rule-based planner is used when no Anthropic credentials exist and
the adapted VLM tool reports itself unavailable when no checkpoint/endpoint is
configured (the deterministic remote-sensing tools still run).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- Planner -----------------------------------------------------------
    # "auto": use Claude when credentials resolve, else rules. "llm" | "rules" force one.
    planner: str = field(default_factory=lambda: os.getenv("SATQUERY_PLANNER", "auto"))
    planner_model: str = field(default_factory=lambda: os.getenv("SATQUERY_PLANNER_MODEL", "claude-opus-5-5"))
    planner_effort: str = field(default_factory=lambda: os.getenv("SATQUERY_PLANNER_EFFORT", "low"))

    # --- Adapted VLM -------------------------------------------------------
    # "hf": load locally with transformers (+ optional LoRA adapter)
    # "remote": POST to SATQUERY_VLM_ENDPOINT (see scripts/serve_vlm.py)
    # "off": tool reports unavailable
    vlm_backend: str = field(default_factory=lambda: os.getenv("SATQUERY_VLM_BACKEND", "off"))
    vlm_base_model: str = field(default_factory=lambda: os.getenv("SATQUERY_VLM_BASE", "Qwen/Qwen3-VL-2B-Instruct"))
    vlm_adapter: str | None = field(default_factory=lambda: os.getenv("SATQUERY_VLM_ADAPTER") or None)
    vlm_endpoint: str | None = field(default_factory=lambda: os.getenv("SATQUERY_VLM_ENDPOINT") or None)
    vlm_timeout_s: float = field(default_factory=lambda: float(os.getenv("SATQUERY_VLM_TIMEOUT", "180")))

    # --- EuroSAT-adapted scene classifier (trained by training/train_scene_classifier.py) -----
    scene_checkpoint: str = field(default_factory=lambda: os.getenv("SATQUERY_SCENE_CKPT", "checkpoints/scene_resnet18.pt"))

    vrsbench_ground_checkpoint: str = field(default_factory=lambda: os.getenv("SATQUERY_VRSBENCH_GROUND_CKPT", "checkpoints/vrsbench_ground.pt"))
    vrsbench_checkpoint: str = field(default_factory=lambda: os.getenv("SATQUERY_VRSBENCH_CKPT", "checkpoints/vrsbench_vqa.pt"))
    cdvqa_checkpoint: str = field(default_factory=lambda: os.getenv("SATQUERY_CDVQA_CKPT", "checkpoints/cdvqa.pt"))
    rsvqa_checkpoint: str = field(default_factory=lambda: os.getenv("SATQUERY_RSVQA_CKPT", "checkpoints/rsvqa_lr.pt"))

    # --- Learned change detector (optional) ---------------------------------
    change_checkpoint: str | None = field(default_factory=lambda: os.getenv("SATQUERY_CHANGE_CKPT") or None)

    # --- Limits --------------------------------------------------------------
    max_upload_mb: int = field(default_factory=lambda: int(os.getenv("SATQUERY_MAX_UPLOAD_MB", "200")))
    max_pixels: int = field(default_factory=lambda: int(os.getenv("SATQUERY_MAX_PIXELS", str(4096 * 4096))))
    allow_benchmark_png: bool = field(default_factory=lambda: _env_bool("SATQUERY_ALLOW_PNG", True))
    max_query_chars: int = field(default_factory=lambda: int(os.getenv("SATQUERY_MAX_QUERY_CHARS", "1000")))

    # --- Public deployment ------------------------------------------------------
    # analyses running at once (each uses a full CPU core and some memory); further requests wait up to
    # queue_timeout_s, then get "busy, try again" instead of piling up
    max_concurrent: int = field(default_factory=lambda: int(os.getenv("SATQUERY_MAX_CONCURRENT", "2")))
    queue_timeout_s: float = field(default_factory=lambda: float(os.getenv("SATQUERY_QUEUE_TIMEOUT", "60")))
    # analyses per client IP per minute (0 = unlimited)
    rate_limit_per_min: int = field(default_factory=lambda: int(os.getenv("SATQUERY_RATE_LIMIT", "0")))
    # reverse proxies in front of the app that each append the caller's address to X-Forwarded-For
    # (Azure Container Apps ingress: 1). 0 = use the TCP peer address. The client is identified by the
    # entry this many places from the right, which a caller cannot forge (the leftmost entries are theirs).
    trusted_proxy_hops: int = field(default_factory=lambda: int(os.getenv("SATQUERY_TRUSTED_PROXY_HOPS", "0")))

    # --- Storage ---------------------------------------------------------------
    run_dir: Path = field(default_factory=lambda: Path(os.getenv("SATQUERY_RUN_DIR", "runs")))
    persist_runs: bool = field(default_factory=lambda: _env_bool("SATQUERY_PERSIST_RUNS", True))
    # load the trained models in the background at startup, so the first request is fast
    warm_up: bool = field(default_factory=lambda: _env_bool("SATQUERY_WARM_UP", True))


settings = Settings()
