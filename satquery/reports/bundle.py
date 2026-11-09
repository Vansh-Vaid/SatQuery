"""Auditable report bundle: Markdown, GeoJSON, PDF and a ZIP with everything (incl. HTML + JSON trace)."""
from __future__ import annotations

import base64
import io
import json
import zipfile
from typing import Any

from satquery.agent.controller import AgentResponse
from satquery.reports.report import render_html, render_json


# ------------------------------------------------------------------ markdown
def render_markdown(resp: AgentResponse) -> str:
    t = resp.trace
    conf = f"{resp.confidence:.1%}" if resp.confidence is not None else "n/a"
    lines = [
        "# SatQuery AI - auditable analysis report", "",
        f"- **Request:** `{resp.request_id}` · {t.get('started_utc')} · {t.get('duration_ms', 0):.0f} ms",
        f"- **Query:** {resp.query}",
        f"- **Task:** `{resp.task}` · **Input:** `{t.get('input', {}).get('configuration')}` · "
        f"**Planner:** `{t.get('planner', {}).get('name', '-')}`",
        f"- **Confidence:** {conf} ({resp.confidence_label})", "",
        "## Answer", "", resp.answer, "",
    ]
    if resp.warnings:
        lines += ["## Input warnings", ""] + [f"- {w}" for w in resp.warnings] + [""]
    lines += ["## Inputs", "", "| # | name | modality | acquired | size | CRS | WGS84 bbox |", "|---|---|---|---|---|---|---|"]
    for i, im in enumerate(t.get("input", {}).get("images", []), 1):
        lines.append(f"| {i} | {im['name']} | {im['modality']} | {im.get('acquired') or '-'} | "
                     f"{im['size'][0]}x{im['size'][1]} | {im.get('crs') or '-'} | {im.get('bbox_wgs84') or '-'} |")
    lines += ["", "## Execution trace", "", "| # | tool | kind | status | conf | ms | parameters |",
              "|---|---|---|---|---|---|---|"]
    for s in t.get("steps", []):
        lines.append(f"| {s['step']} | `{s['tool']}` | {s['kind']} | {s['status']} | {s['confidence']} | "
                     f"{s['duration_ms']} | `{json.dumps(s['params'])}` |")
    corr = t.get("plan", {}).get("corrections", [])
    if corr:
        lines += ["", "Plan corrections by the registry validator:", ""] + [f"- {c}" for c in corr]
    lines += ["", "## Evidence by tool", ""]
    lines += [f"- **{e['tool']}**{' (primary)' if e['primary'] else ''}: {e['text']}" for e in resp.evidence]
    lines += ["", "## Input validation", "", "| check | status | detail |", "|---|---|---|"]
    lines += [f"| {c['id']} | {c['status']} | {c['message']} |" for c in t.get("validation", [])]
    lines += ["", "_Visual evidence: see report.html / report.pdf; spatial outputs: evidence.geojson._", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------ geojson
def _pix_to_lonlat(geo: dict, pts: list[tuple[float, float]]) -> list[list[float]] | None:
    if not geo.get("georeferenced") or not geo.get("transform") or not geo.get("crs"):
        return None
    from rasterio.warp import transform
    a, b, c, d, e, f = geo["transform"][:6]
    xs = [a * x + b * y + c for x, y in pts]
    ys = [d * x + e * y + f for x, y in pts]
    lon, lat = transform(geo["crs"], "EPSG:4326", xs, ys)
    return [[round(lo, 7), round(la, 7)] for lo, la in zip(lon, lat)]


def render_geojson(resp: AgentResponse) -> str:
    geo = resp.viewer.get("geo", {}) if resp.viewer else {}
    feats: list[dict[str, Any]] = []
    for i, im in enumerate(resp.trace.get("input", {}).get("images", []), 1):
        bb = im.get("bbox_wgs84")
        if bb:
            w, s, e, n = bb
            feats.append({"type": "Feature", "properties": {"kind": "input_footprint", "image": i, "name": im["name"],
                                                            "modality": im["modality"], "acquired": im.get("acquired")},
                          "geometry": {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}})
    for b in resp.boxes:
        ring_px = [(b["x0"], b["y0"]), (b["x1"], b["y0"]), (b["x1"], b["y1"]), (b["x0"], b["y1"]), (b["x0"], b["y0"])]
        ring = _pix_to_lonlat(geo, ring_px)
        props = {"kind": "evidence_box", "label": b["label"], "source": b["source"], "score": b["score"],
                 "pixel_bbox": [b["x0"], b["y0"], b["x1"], b["y1"]]}
        if ring is None:
            props["note"] = "input not georeferenced; geometry in pixel coordinates"
            ring = [list(p) for p in ring_px]
        feats.append({"type": "Feature", "properties": props, "geometry": {"type": "Polygon", "coordinates": [ring]}})
    fc = {"type": "FeatureCollection", "name": f"satquery_{resp.request_id}",
          "properties": {"query": resp.query, "task": resp.task, "answer": resp.answer, "confidence": resp.confidence},
          "features": feats}
    return json.dumps(fc, indent=2, ensure_ascii=False)


# ------------------------------------------------------------------ pdf
_ASCII = {"→": "->", "²": "2", "≥": ">=", "≤": "<=", "·": "-", "σ": "sigma", "…": "...", "—": "-", "–": "-"}


def _a(s: Any) -> str:
    s = "" if s is None else str(s)
    for k, v in _ASCII.items():
        s = s.replace(k, v)
    s = s.encode("latin-1", "replace").decode("latin-1")
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_pdf(resp: AgentResponse) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm,
                            bottomMargin=14 * mm, title=f"SatQuery report {resp.request_id}")
    ss = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=ss["BodyText"], fontSize=8, leading=10)
    mono = ParagraphStyle("mono", parent=small, fontName="Courier", fontSize=7, leading=9)
    t = resp.trace
    conf = f"{resp.confidence:.1%}" if resp.confidence is not None else "n/a"
    story: list = [
        Paragraph("SatQuery AI - auditable analysis report", ss["Title"]),
        Paragraph(_a(f"Request {resp.request_id} - {t.get('started_utc')} - {t.get('duration_ms', 0):.0f} ms - "
                     f"planner {t.get('planner', {}).get('name', '-')}"), small),
        Spacer(1, 4 * mm),
        Paragraph(f"<b>Query:</b> {_a(resp.query)}", ss["BodyText"]),
        Paragraph(f"<b>Task:</b> {_a(resp.task)} &nbsp; <b>Confidence:</b> {conf} ({_a(resp.confidence_label)})",
                  ss["BodyText"]),
        Spacer(1, 2 * mm),
        Paragraph("Answer", ss["Heading2"]),
        Paragraph(_a(resp.answer).replace("\n", "<br/>"), ss["BodyText"]),
    ]
    imgs = [v for v in resp.get_visuals() if v.get("png", "").startswith("data:image/png;base64,")][:8]
    if imgs:
        story += [Paragraph("Visual evidence", ss["Heading2"])]
        cells, row = [], []
        for v in imgs:
            raw = base64.b64decode(v["png"].split(",", 1)[1])
            row.append([Image(io.BytesIO(raw), width=80 * mm, height=80 * mm, kind="proportional"),
                        Paragraph(_a(v["title"]), small)])
            if len(row) == 2:
                cells.append(row)
                row = []
        if row:
            cells.append(row + [""])
        tbl = Table(cells, colWidths=[88 * mm, 88 * mm])
        tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(tbl)
    story += [Paragraph("Execution trace", ss["Heading2"])]
    rows = [["#", "tool", "status", "conf", "ms", "parameters"]]
    for s in t.get("steps", []):
        rows.append([str(s["step"]), Paragraph(_a(s["tool"]), small), s["status"], str(s["confidence"]),
                     str(s["duration_ms"]), Paragraph(_a(json.dumps(s["params"])), mono)])
    tbl = Table(rows, colWidths=[8 * mm, 38 * mm, 20 * mm, 14 * mm, 14 * mm, 82 * mm], repeatRows=1)
    tbl.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b1a33")),
                             ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 8),
                             ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9fb3c8")),
                             ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(tbl)
    story += [Paragraph("Evidence by tool", ss["Heading2"])]
    story += [Paragraph(f"<b>{_a(e['tool'])}</b>{' (primary)' if e['primary'] else ''}: {_a(e['text'])}", small)
              for e in resp.evidence]
    story += [Paragraph("Input validation", ss["Heading2"])]
    vr = [["check", "status", "detail"]] + [[c["id"], c["status"], Paragraph(_a(c["message"]), small)]
                                            for c in t.get("validation", [])]
    vt = Table(vr, colWidths=[34 * mm, 14 * mm, 128 * mm], repeatRows=1)
    vt.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9fb3c8"))]))
    story.append(vt)
    doc.build(story)
    return buf.getvalue()


def render_zip(resp: AgentResponse) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("report.pdf", render_pdf(resp))
        z.writestr("report.md", render_markdown(resp))
        z.writestr("report.html", render_html(resp))
        z.writestr("evidence.geojson", render_geojson(resp))
        z.writestr("trace.json", render_json(resp))
    return buf.getvalue()
