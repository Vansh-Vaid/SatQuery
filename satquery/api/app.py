"""FastAPI backend + web UI for SatQuery AI.

Run:  uvicorn satquery.api.app:app --port 8000   then open http://localhost:8000
The UI is the React/Three.js app in ``frontend/`` built into ``satquery/web/dist``
(``cd frontend && npm install && npm run build``).
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import threading
import time
from collections import OrderedDict, defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from satquery import __version__
from satquery.agent.controller import AgentResponse, SatQueryAgent
from satquery.agent.planner import has_anthropic_credentials
from satquery.agent.viewer import image_entry
from satquery.config import settings
from satquery.demo.make import DEFAULT_DIR, load_manifest
from satquery.imagery.io import SUPPORTED_EXT, UnsupportedInput, load_image
from satquery.imagery.raster import Raster
from satquery.imagery.validate import validate_inputs
from satquery.reports.bundle import render_geojson, render_markdown, render_pdf, render_zip
from satquery.reports.report import render_html, render_json

DIST_DIR = Path(__file__).resolve().parents[1] / "web" / "dist"

log = logging.getLogger("satquery")
agent = SatQueryAgent()
_runs: "OrderedDict[str, AgentResponse]" = OrderedDict()
_runs_lock = threading.Lock()
MAX_RUNS = 100


# Demo scenes give the same answer to everyone, so their results are computed once and shared. This keeps
# the landing page (which runs these three demos for every visitor) from loading the server.
_demo_cache: "OrderedDict[tuple, AgentResponse]" = OrderedDict()
_DEMO_CACHE_MAX = 64
# keep in sync with DEMO_RUNS in frontend/src/landing/useDemoRuns.ts (pre-computed at startup)
LANDING_DEMOS = [
    ("single_s2", "Highlight the water body referred to in the query."),
    ("cross_modal", "Use the optical and SAR images together to identify built-up and water-covered regions."),
    ("bitemporal_s2", "What changed between these two dates, and where did the change occur?"),
]


def _demo_key(demo: str, query: str, pair_type: str, mods: tuple, dates: tuple) -> tuple:
    return (demo, " ".join(query.lower().split()), pair_type or "auto", mods, dates)


def _run_demo(demo: str, query: str, pair_type: str = "auto", mods: tuple = ("auto", "auto"),
              dates: tuple = ("", "")) -> AgentResponse:
    key = _demo_key(demo, query, pair_type, mods, dates)
    with _runs_lock:
        hit = _demo_cache.get(key)
    if hit is not None:
        return hit
    rasters = _collect(Path(tempfile.gettempdir()), demo, [(None, mods[0], dates[0]), (None, mods[1], dates[1])])
    resp = agent.run(query, rasters, None if pair_type in ("", "auto") else pair_type)
    with _runs_lock:
        _demo_cache[key] = resp
        while len(_demo_cache) > _DEMO_CACHE_MAX:
            _demo_cache.popitem(last=False)
    return resp


def _warm_up() -> None:
    """Load every trained model and pre-compute the landing-page demos in the background."""
    try:
        t0 = time.perf_counter()
        for demo, q in LANDING_DEMOS:
            _store(_run_demo(demo, q))
        for q in ("Is there water in this image?", "Describe this image."):
            _run_demo("single_s2", q)
        log.warning("SatQuery models warmed up in %.1f s", time.perf_counter() - t0)
    except Exception:  # never block startup on this
        log.exception("warm-up failed (the app still works; the first request will be slower)")


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.warm_up:
        threading.Thread(target=_warm_up, daemon=True).start()
    yield


app = FastAPI(title="SatQuery AI", version=__version__, lifespan=lifespan,
              description="Agentic vision-language assistant for multimodal remote-sensing image analysis.")


# ------------------------------------------------------------------ public-deployment guards
class RateLimiter:
    """Sliding one-minute window per client IP (in memory, for a single server process)."""

    def __init__(self, per_min: int):
        self.per_min = per_min
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if self.per_min <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_min:
                return False
            q.append(now)
            if len(self._hits) > 10_000:  # forget idle clients
                for k in [k for k, v in self._hits.items() if not v]:
                    del self._hits[k]
            return True


limiter = RateLimiter(settings.rate_limit_per_min)
slots = threading.BoundedSemaphore(max(1, settings.max_concurrent))


@contextmanager
def _guard(request: Request):
    """Per-client rate limit, then a bounded number of analyses running at once."""
    if not limiter.allow(request.client.host if request.client else "unknown"):
        raise HTTPException(429, "Too many requests from your address. Please wait a minute and try again.",
                            headers={"Retry-After": "60"})
    if not slots.acquire(timeout=settings.queue_timeout_s):
        raise HTTPException(503, "The server is busy with other analyses. Please try again in a minute.",
                            headers={"Retry-After": "30"})
    try:
        yield
    finally:
        slots.release()


@app.middleware("http")
async def _edge(request: Request, call_next):
    cl = request.headers.get("content-length", "")
    if cl.isdigit() and int(cl) > (2 * settings.max_upload_mb + 1) * 1024 * 1024:  # two images + form fields
        return JSONResponse({"detail": f"Upload too large (limit {settings.max_upload_mb} MB per image)."}, 413)
    t0 = time.perf_counter()
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    if request.url.path.startswith("/api/") and request.method == "POST":
        log.warning("%s %s %d %.0f ms", request.method, request.url.path, resp.status_code,
                    (time.perf_counter() - t0) * 1e3)
    return resp


def _store(resp: AgentResponse) -> None:
    with _runs_lock:
        seen = resp.request_id in _runs
        _runs[resp.request_id] = resp
        _runs.move_to_end(resp.request_id)
        while len(_runs) > MAX_RUNS:
            _runs.popitem(last=False)
    if seen or not settings.persist_runs:  # a shared demo result is written once
        return
    try:
        d = settings.run_dir / resp.request_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "trace.json").write_text(render_json(resp), encoding="utf-8")
    except OSError:
        pass  # persistence is best-effort; the in-memory copy serves downloads


def _get(rid: str) -> AgentResponse:
    with _runs_lock:
        resp = _runs.get(rid)
    if resp is None:
        raise HTTPException(404, f"Unknown or expired request id '{rid}'")
    return resp


def _collect(tmp: Path, demo: str, uploads: list[tuple[Optional[UploadFile], str, str]]) -> list[Raster]:
    """Resolve demo scenario or uploaded files into validated-format rasters (raises HTTPException)."""
    specs: list[tuple[Path, str, str, str]] = []
    if demo:
        sc = next((s for s in load_manifest()["scenarios"] if s["id"] == demo), None)
        if sc is None:
            raise HTTPException(400, f"unknown demo '{demo}'")
        for i, f in enumerate(sc["files"]):
            _, mod, dt = uploads[i] if i < len(uploads) else (None, "auto", "")
            specs.append((DEFAULT_DIR / f, f, mod, dt))
    else:
        for i, (up, mod, dt) in enumerate(uploads, start=1):
            if up is None or not up.filename:
                continue
            ext = Path(up.filename).suffix.lower()
            if ext not in SUPPORTED_EXT:
                raise HTTPException(415, f"{up.filename}: unsupported format '{ext}'. "
                                         f"Use GeoTIFF/TIFF (PNG/JPEG for benchmark images only).")
            dst = tmp / f"img{i}{ext}"
            limit, n = settings.max_upload_mb * 1024 * 1024, 0
            with dst.open("wb") as fh:
                while chunk := up.file.read(1 << 20):
                    n += len(chunk)
                    if n > limit:
                        raise HTTPException(413, f"{up.filename} exceeds {settings.max_upload_mb} MB")
                    fh.write(chunk)
            specs.append((dst, up.filename, mod, dt))
    if not specs:
        raise HTTPException(400, "Upload at least one image (or choose a demo scenario).")
    rasters = []
    for path, name, mod, dt in specs:
        try:
            rasters.append(load_image(path, None if mod in ("", "auto") else mod, dt or None, display_name=name))
        except UnsupportedInput as e:
            raise HTTPException(415, str(e)) from e
    return rasters


@app.get("/api/health")
def health() -> dict:
    vlm_ok, vlm_status = agent.registry.available("rs_vlm")
    planner = settings.planner
    if planner == "auto":
        planner = f"claude ({settings.planner_model})" if has_anthropic_credentials() else "rules (no Claude credentials)"
    return {"status": "ok", "version": __version__, "planner": planner,
            "vlm": {"available": vlm_ok, "status": vlm_status}}


@app.get("/api/tools")
def tools() -> dict:
    return {"tasks": agent.registry.tasks(), "tools": agent.registry.describe()}


@app.get("/api/demos")
def demos() -> dict:
    return load_manifest()


@app.post("/api/inspect")
def inspect(
    request: Request,
    image1: Optional[UploadFile] = File(None), image2: Optional[UploadFile] = File(None),
    modality1: str = Form("auto"), modality2: str = Form("auto"), date1: str = Form(""), date2: str = Form(""),
    pair_type: str = Form("auto"), demo: str = Form(""),
) -> JSONResponse:
    """Compatibility gate: validate inputs and return previews + footprints, without running any tool."""
    tmp = Path(tempfile.mkdtemp(prefix="satquery_"))
    try:
        with _guard(request):
            rasters = _collect(tmp, demo, [(image1, modality1, date1), (image2, modality2, date2)])
            rep = validate_inputs(rasters, None if pair_type in ("", "auto") else pair_type)
        return JSONResponse({**rep.to_dict(), "images": [image_entry(i, r) for i, r in enumerate(rep.rasters)]})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.post("/api/analyze")
def analyze(
    request: Request,
    query: str = Form(...),
    image1: Optional[UploadFile] = File(None), image2: Optional[UploadFile] = File(None),
    modality1: str = Form("auto"), modality2: str = Form("auto"), date1: str = Form(""), date2: str = Form(""),
    pair_type: str = Form("auto"), demo: str = Form(""),
) -> JSONResponse:
    query = query.strip()
    if not query or len(query) > settings.max_query_chars:
        raise HTTPException(400, f"Please ask a question of 1-{settings.max_query_chars} characters.")
    if demo:
        key = _demo_key(demo, query, pair_type, (modality1, modality2), (date1, date2))
        with _runs_lock:
            cached = key in _demo_cache
        if cached:  # shared result: free, so not rate limited
            resp = _run_demo(demo, query, pair_type, (modality1, modality2), (date1, date2))
        else:
            with _guard(request):
                resp = _run_demo(demo, query, pair_type, (modality1, modality2), (date1, date2))
        _store(resp)
        return JSONResponse(resp.to_dict())
    tmp = Path(tempfile.mkdtemp(prefix="satquery_"))
    try:
        with _guard(request):
            rasters = _collect(tmp, demo, [(image1, modality1, date1), (image2, modality2, date2)])
            resp = agent.run(query, rasters, None if pair_type in ("", "auto") else pair_type)
        _store(resp)
        return JSONResponse(resp.to_dict())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _dl(rid: str, ext: str) -> dict[str, str]:
    return {"Content-Disposition": f'attachment; filename="satquery_{rid}.{ext}"'}


@app.get("/api/report/{rid}.html", response_class=HTMLResponse)
def report_html(rid: str) -> HTMLResponse:
    return HTMLResponse(render_html(_get(rid)), headers=_dl(rid, "html"))


@app.get("/api/report/{rid}.json")
def report_json(rid: str) -> PlainTextResponse:
    return PlainTextResponse(render_json(_get(rid)), media_type="application/json", headers=_dl(rid, "json"))


@app.get("/api/report/{rid}.md")
def report_md(rid: str) -> PlainTextResponse:
    return PlainTextResponse(render_markdown(_get(rid)), media_type="text/markdown", headers=_dl(rid, "md"))


@app.get("/api/report/{rid}.geojson")
def report_geojson(rid: str) -> PlainTextResponse:
    return PlainTextResponse(render_geojson(_get(rid)), media_type="application/geo+json", headers=_dl(rid, "geojson"))


@app.get("/api/report/{rid}.pdf")
def report_pdf(rid: str) -> Response:
    return Response(render_pdf(_get(rid)), media_type="application/pdf", headers=_dl(rid, "pdf"))


@app.get("/api/report/{rid}.zip")
def report_zip(rid: str) -> Response:
    return Response(render_zip(_get(rid)), media_type="application/zip", headers=_dl(rid, "zip"))


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
@app.api_route("/console", methods=["GET", "HEAD"], response_class=HTMLResponse)
def index():
    page = DIST_DIR / "index.html"
    if not page.exists():
        return HTMLResponse("<h1>SatQuery AI</h1><p>Web UI not built. Run <code>cd frontend && npm install && "
                            "npm run build</code>, or use the API at <a href='/docs'>/docs</a>.</p>", status_code=503)
    # the HTML shell must revalidate so a rebuilt UI (new hashed asset names) is picked up immediately
    return FileResponse(page, headers={"Cache-Control": "no-cache"})


if (DIST_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="assets")
