"""FastAPI backend + web UI for SatQuery AI.

Run:  uvicorn satquery.api.app:app --port 8000   then open http://localhost:8000
The UI is the React/Three.js app in ``frontend/`` built into ``satquery/web/dist``
(``cd frontend && npm install && npm run build``).
"""
from __future__ import annotations

import shutil
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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

app = FastAPI(title="SatQuery AI", version=__version__,
              description="Agentic vision-language assistant for multimodal remote-sensing image analysis.")
agent = SatQueryAgent()
_runs: "OrderedDict[str, AgentResponse]" = OrderedDict()
_runs_lock = threading.Lock()
MAX_RUNS = 100


def _store(resp: AgentResponse) -> None:
    with _runs_lock:
        _runs[resp.request_id] = resp
        while len(_runs) > MAX_RUNS:
            _runs.popitem(last=False)
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
            with dst.open("wb") as fh:
                shutil.copyfileobj(up.file, fh, length=1 << 20)
            if dst.stat().st_size > settings.max_upload_mb * 1024 * 1024:
                raise HTTPException(413, f"{up.filename} exceeds {settings.max_upload_mb} MB")
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
    image1: Optional[UploadFile] = File(None), image2: Optional[UploadFile] = File(None),
    modality1: str = Form("auto"), modality2: str = Form("auto"), date1: str = Form(""), date2: str = Form(""),
    pair_type: str = Form("auto"), demo: str = Form(""),
) -> JSONResponse:
    """Compatibility gate: validate inputs and return previews + footprints, without running any tool."""
    tmp = Path(tempfile.mkdtemp(prefix="satquery_"))
    try:
        rasters = _collect(tmp, demo, [(image1, modality1, date1), (image2, modality2, date2)])
        rep = validate_inputs(rasters, None if pair_type in ("", "auto") else pair_type)
        return JSONResponse({**rep.to_dict(), "images": [image_entry(i, r) for i, r in enumerate(rep.rasters)]})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.post("/api/analyze")
def analyze(
    query: str = Form(...),
    image1: Optional[UploadFile] = File(None), image2: Optional[UploadFile] = File(None),
    modality1: str = Form("auto"), modality2: str = Form("auto"), date1: str = Form(""), date2: str = Form(""),
    pair_type: str = Form("auto"), demo: str = Form(""),
) -> JSONResponse:
    tmp = Path(tempfile.mkdtemp(prefix="satquery_"))
    try:
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


@app.get("/", response_class=HTMLResponse)
@app.get("/console", response_class=HTMLResponse)
def index():
    page = DIST_DIR / "index.html"
    if not page.exists():
        return HTMLResponse("<h1>SatQuery AI</h1><p>Web UI not built. Run <code>cd frontend && npm install && "
                            "npm run build</code>, or use the API at <a href='/docs'>/docs</a>.</p>", status_code=503)
    # the HTML shell must revalidate so a rebuilt UI (new hashed asset names) is picked up immediately
    return FileResponse(page, headers={"Cache-Control": "no-cache"})


if (DIST_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="assets")
