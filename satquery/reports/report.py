"""Downloadable, self-contained analysis reports (HTML with embedded images, and JSON)."""
from __future__ import annotations

import html
import json
from typing import Any

from satquery.agent.controller import AgentResponse

_CSS = """
:root{--bg:#f7f8fa;--fg:#1d2433;--muted:#5b6578;--card:#fff;--line:#e2e6ee;--accent:#1f6feb;
--ok:#1a7f37;--warn:#9a6700;--fail:#cf222e}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
background:var(--bg);color:var(--fg)}main{max-width:1100px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 10px;border-bottom:1px solid var(--line);
padding-bottom:6px}.muted{color:var(--muted)}.card{background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:16px;margin:10px 0}.answer{white-space:pre-wrap;font-size:15px}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:600;
background:#eef2ff;color:var(--accent);margin-right:6px}.high{background:#dafbe1;color:var(--ok)}
.medium{background:#fff8c5;color:var(--warn)}.low{background:#ffebe9;color:var(--fail)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px}
figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px}
figure img{width:100%;image-rendering:pixelated;border-radius:4px}figcaption{font-size:12px;color:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{border-bottom:1px solid var(--line);
padding:6px 8px;text-align:left;vertical-align:top}th{background:#f0f2f6}
.s-pass,.s-ok{color:var(--ok)}.s-warn{color:var(--warn)}.s-fail,.s-error{color:var(--fail)}
.s-unavailable,.s-unsupported{color:var(--muted)}code,pre{font:12px ui-monospace,Consolas,monospace}
pre{background:#0d1117;color:#e6edf3;padding:12px;border-radius:8px;overflow:auto;max-height:480px}
"""


def _e(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def render_html(resp: AgentResponse) -> str:
    t = resp.trace
    conf = f"{resp.confidence:.0%}" if resp.confidence is not None else "n/a"
    rows_val = "".join(f"<tr><td>{_e(c['id'])}</td><td class='s-{_e(c['status'])}'>{_e(c['status'])}</td>"
                       f"<td>{_e(c['message'])}</td></tr>" for c in t.get("validation", []))
    rows_steps = "".join(
        f"<tr><td>{s['step']}</td><td><b>{_e(s['tool'])}</b><br><span class='muted'>{_e(s['kind'])}</span></td>"
        f"<td><code>{_e(json.dumps(s['params']))}</code></td><td class='s-{_e(s['status'])}'>{_e(s['status'])}</td>"
        f"<td>{_e(s['confidence'])}</td><td>{s['duration_ms']}</td><td>{_e(s['output'] or '; '.join(s['notes']))}</td></tr>"
        for s in t.get("steps", []))
    plan = t.get("plan", {})
    corr = "".join(f"<li>{_e(c)}</li>" for c in plan.get("corrections", [])) or "<li>none</li>"
    figs = "".join(f"<figure><img alt='{_e(v['title'])}' src='{v['png']}'><figcaption><b>{_e(v['title'])}</b>"
                   f"{(' - ' + _e(v.get('legend'))) if v.get('legend') else ''}</figcaption></figure>"
                   for v in resp.get_visuals())
    ev = "".join(f"<li><b>{_e(e['tool'])}</b>{' (primary)' if e['primary'] else ''}: {_e(e['text'])} "
                 f"<span class='muted'>conf {_e(e['confidence'])}</span></li>" for e in resp.evidence)
    warns = "".join(f"<li>{_e(w)}</li>" for w in resp.warnings)
    trace_json = _e(json.dumps(t, indent=2, ensure_ascii=False))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SatQuery report {resp.request_id}</title>
<style>{_CSS}</style></head><body><main>
<h1>SatQuery AI analysis report</h1>
<div class="muted">Request {resp.request_id} · {_e(t.get('started_utc'))} · SatQuery v{_e(t.get('satquery_version'))}
 · {t.get('duration_ms', 0):.0f} ms</div>
<div class="card"><div class="muted">Query</div><div style="font-size:16px"><b>{_e(resp.query)}</b></div></div>
<div class="card"><span class="pill">{_e(resp.task or 'rejected')}</span>
<span class="pill {resp.confidence_label}">confidence {conf} ({_e(resp.confidence_label)})</span>
<span class="pill">input: {_e(t.get('input', {}).get('configuration'))}</span>
<span class="pill">planner: {_e(t.get('planner', {}).get('name', '-'))}</span>
<p class="answer">{_e(resp.answer)}</p>
{('<div class="muted">Input warnings:</div><ul>' + warns + '</ul>') if warns else ''}</div>
<h2>Visual evidence</h2><div class="grid">{figs}</div>
<h2>Supporting evidence by tool</h2><div class="card"><ul>{ev or '<li>none</li>'}</ul></div>
<h2>Execution summary</h2>
<div class="card"><p><b>Task:</b> {_e(plan.get('task'))} · <b>Intent:</b> {_e(plan.get('intent'))} ·
<b>Rationale:</b> {_e(plan.get('rationale'))}</p><p><b>Plan corrections by the registry validator:</b></p><ul>{corr}</ul>
<table><tr><th>#</th><th>Tool</th><th>Parameters</th><th>Status</th><th>Conf.</th><th>ms</th><th>Output</th></tr>
{rows_steps}</table></div>
<h2>Input validation</h2><div class="card"><table><tr><th>Check</th><th>Status</th><th>Detail</th></tr>{rows_val}</table></div>
<h2>Machine-readable trace</h2><pre>{trace_json}</pre>
</main></body></html>"""


def render_json(resp: AgentResponse) -> str:
    return json.dumps(resp.to_dict(include_visuals=False), indent=2, ensure_ascii=False, default=str)
