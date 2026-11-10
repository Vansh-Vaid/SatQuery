// Evidence-grounded result flashcard: tilt, flip to audit log, entity spans that drive hover beams (spec §6.1, §7.1)
import { useRef, useState } from "react";
import type { AnalyzeResult, Box } from "../api";
import { fmtLat, fmtLon, pixelToLatLon } from "../api";

interface Props {
  result: AnalyzeResult;
  reduced: boolean;
  onEntity: (ids: string[], el: HTMLElement | null) => void;
}

const CLASS_PATTERNS: [RegExp, (b: Box) => boolean][] = [
  [/^(water( bod(y|ies))?|lakes?|rivers?|water-covered)$/i, (b) => /water|lake/i.test(b.label)],
  [/^(built-up( areas?)?|urban|town|buildings?)$/i, (b) => /built/i.test(b.label)],
  [/^(vegetation|forest|fields?)$/i, (b) => /vegetation/i.test(b.label)],
  [/^(bare soil|quarry)$/i, (b) => /bare/i.test(b.label)],
  [/^(changed?( regions?| areas?)?|changes)$/i, (b) => b.source === "change_detector"],
];
const SECTORS = /^(north-west|north-east|south-west|south-east|north|south|east|west|centre)$/i;
const TOKEN = /(water bod(?:y|ies)|water-covered|built-up(?: areas?)?|bare soil|changed? (?:regions?|areas?)|north-west|north-east|south-west|south-east|\b(?:water|lakes?|rivers?|urban|town|buildings?|vegetation|forest|fields?|quarry|changes?|changed|north|south|east|west|centre)\b)/gi;

function sectorOf(b: Box, w: number, h: number): string {
  const cy = (b.y0 + b.y1) / 2 / h, cx = (b.x0 + b.x1) / 2 / w;
  const v = cy < 1 / 3 ? "north" : cy > 2 / 3 ? "south" : "";
  const hz = cx < 1 / 3 ? "west" : cx > 2 / 3 ? "east" : "";
  return v && hz ? `${v}-${hz}` : v || hz || "centre";
}

function linkify(text: string, boxes: Box[], w: number, h: number) {
  const parts: { t: string; ids: string[] }[] = [];
  let last = 0;
  for (const m of text.matchAll(TOKEN)) {
    const word = m[0];
    let ids: string[] = [];
    if (SECTORS.test(word)) ids = boxes.filter((b) => sectorOf(b, w, h).includes(word.toLowerCase())).map((b) => b.id);
    else {
      const rule = CLASS_PATTERNS.find(([re]) => re.test(word));
      if (rule) ids = boxes.filter(rule[1]).map((b) => b.id);
    }
    if (!ids.length) continue;
    parts.push({ t: text.slice(last, m.index), ids: [] }, { t: word, ids });
    last = (m.index ?? 0) + word.length;
  }
  parts.push({ t: text.slice(last), ids: [] });
  return parts;
}

export default function EvidenceCard({ result, reduced, onEntity }: Props) {
  const [flipped, setFlipped] = useState(false);
  const [tilt, setTilt] = useState({ x: 0, y: 0 });
  const card = useRef<HTMLDivElement>(null);
  const v = result.viewer;
  const steps = result.trace.steps ?? [];
  const primary = steps.find((s) => s.tool === result.trace.result?.primary_tool);
  const vlm = steps.find((s) => s.tool === "rs_vlm" && s.status === "ok");
  const backend = (vlm?.facts?.backend as string | undefined) ?? "";
  const adapted = /LoRA/.test(backend);
  const sceneUsed = steps.some((s) => s.tool === "rs_scene_classifier" && s.status === "ok");
  const provenance = adapted ? "BigEarthNet.txt-adapted model"
    : vlm ? "Base model, no remote-sensing adapter"
    : primary?.tool === "rs_scene_classifier" ? "EuroSAT-adapted ResNet-18"
    : sceneUsed ? "Algorithmic evidence + EuroSAT-adapted classifier" : "Algorithmic evidence";
  const conf = result.confidence;
  const parts = linkify(result.answer, v.boxes ?? [], v.width, v.height);
  const audit = {
    request_id: result.request_id,
    task: result.task,
    primary_tool: primary?.tool ?? null,
    model: vlm ? backend : primary?.kind ?? null,
    adaptation_source: adapted ? "BigEarthNet.txt (LoRA)"
      : primary?.tool === "rs_scene_classifier" ? "EuroSAT Sentinel-2 (fine-tuned ResNet-18)" : "n/a — algorithmic specialist",
    inputs: v.images.map((im) => ({ name: im.name, modality: im.modality, shape: [im.bands.length, ...im.size],
                                    crs: im.crs, acquired: im.acquired })),
    latency_ms: { total: result.trace.duration_ms, primary: primary?.duration_ms ?? null },
    parameters: primary?.params ?? {},
    regions: (v.boxes ?? []).map((b) => {
      const c = pixelToLatLon(v.geo.bbox_wgs84, (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2, v.width, v.height);
      return { label: b.label, source: b.source, pixel_bbox: [b.x0, b.y0, b.x1, b.y1],
               centre: c ? `${fmtLat(c.lat)} ${fmtLon(c.lon)}` : null };
    }),
  };
  const onMove = (e: React.MouseEvent) => {
    if (reduced || !card.current) return;
    const r = card.current.getBoundingClientRect();
    setTilt({ x: ((e.clientY - r.top) / r.height - 0.5) * -6, y: ((e.clientX - r.left) / r.width - 0.5) * 8 });
  };
  const allIds = (v.boxes ?? []).map((b) => b.id);
  // verdict: the short answer, when it is a real answer and not a placeholder like "description"
  const short = (result.short_answer ?? "").trim();
  const verdict = short && short.length <= 48 && !/^(description|caption|unsupported|n\/a)$/i.test(short)
    ? short[0].toUpperCase() + short.slice(1) : null;
  const nBoxes = v.boxes?.length ?? 0;
  const regions = `${nBoxes} region${nBoxes === 1 ? "" : "s"}`;
  const fallback = result.status !== "ok" ? "Partial answer"
    : result.task === "grounding" ? (nBoxes ? `${regions} highlighted` : "Nothing found to highlight")
    : result.task === "cross_modal_analysis" ? (nBoxes ? `${regions} found with both sensors` : "Fused reading ready")
    : result.task === "change_description" ? (nBoxes ? `${regions} of change` : "No significant change")
    : result.task === "caption" ? "Scene described" : "Answer ready";
  const notes = result.trace.result?.integration_notes ?? [];
  const crossCheck = notes.some((n) => /disagree/.test(n)) ? { cls: "warn", text: "Sources disagree" }
    : notes.some((n) => /confirmed/.test(n)) ? { cls: "ok", text: "Cross-checked" } : null;
  const secs = result.trace.duration_ms != null ? `${(result.trace.duration_ms / 1000).toFixed(1)} s` : null;
  const learned = /^learned/.test(primary?.kind ?? "");
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard?.writeText(result.answer).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1600); })
      .catch(() => undefined);
  };
  const pct = conf != null ? Math.round(conf * 100) : null;
  const R = 22, C = 2 * Math.PI * R;

  return (
    <div className="card-wrap" onMouseMove={onMove} onMouseLeave={() => { setTilt({ x: 0, y: 0 }); onEntity([], null); }}>
      <div ref={card} className={`evidence-card ${flipped ? "flipped" : ""}`}
        style={{ transform: `rotateX(${tilt.x}deg) rotateY(${tilt.y + (flipped ? 180 : 0)}deg)` }}>
        <div className="face front" aria-hidden={flipped}>
          <div className="verdict">
            <div className="verdict-main">
              <span className="verdict-kicker">{(result.task ?? "rejected").replace(/_/g, " ")}</span>
              {verdict ? <p className="verdict-answer">{verdict}</p>
                : <p className="verdict-answer soft">{fallback}</p>}
            </div>
            <div className={`ring ${result.confidence_label}`} role="img"
              aria-label={pct != null ? `Confidence ${pct} percent, ${result.confidence_label}` : "Confidence not available"}>
              <svg viewBox="0 0 56 56" aria-hidden="true">
                <circle cx="28" cy="28" r={R} className="ring-track" />
                <circle cx="28" cy="28" r={R} className="ring-fill" strokeDasharray={C}
                  strokeDashoffset={C * (1 - (conf ?? 0))} transform="rotate(-90 28 28)" />
              </svg>
              <span className="ring-val">{pct != null ? pct : "–"}<small>%</small></span>
              <span className="ring-label">{result.confidence_label} confidence</span>
            </div>
          </div>
          <ul className="verdict-meta" aria-label="How this answer was produced">
            {primary && <li title={primary.purpose}><i className={learned ? "dot-learned" : "dot-measure"} />
              {primary.title}<span>{learned ? "trained model" : "measurement"}</span></li>}
            {crossCheck && <li className={crossCheck.cls}>{crossCheck.cls === "ok" ? "✓" : "!"} {crossCheck.text}</li>}
            {secs && <li className="mono">{secs}</li>}
          </ul>
          <p className="provenance">{provenance}</p>
          <p className="answer">
            {parts.map((p, i) => p.ids.length ? (
              <span key={i} className="entity" tabIndex={0}
                onMouseEnter={(e) => onEntity(p.ids, e.currentTarget)} onMouseLeave={() => onEntity([], null)}
                onFocus={(e) => onEntity(p.ids, e.currentTarget)} onBlur={() => onEntity([], null)}>{p.t}</span>
            ) : <span key={i}>{p.t}</span>)}
          </p>
          {result.warnings.length > 0 && <ul className="warn-list">{result.warnings.map((w) => <li key={w}>⚠ {w}</li>)}</ul>}
          <div className="telemetry-bar">
            <div className="card-actions">
              {allIds.length > 0 && (
                <button className="ghost-btn" onMouseEnter={(e) => onEntity(allIds, e.currentTarget)}
                  onFocus={(e) => onEntity(allIds, e.currentTarget)} onBlur={() => onEntity([], null)}
                  onMouseLeave={() => onEntity([], null)}>◎ Show all regions</button>
              )}
              <button className="ghost-btn" onClick={copy} aria-live="polite">{copied ? "✓ Copied" : "Copy answer"}</button>
              <button className="ghost-btn" onClick={() => setFlipped(true)}>Audit log →</button>
            </div>
          </div>
        </div>
        <div className="face back" aria-hidden={!flipped}>
          <div className="card-badge"><b>Audit log</b> · {result.request_id}</div>
          <pre className="code">{JSON.stringify(audit, null, 2)}</pre>
          <button className="link-btn" onClick={() => setFlipped(false)}>← Back to answer</button>
        </div>
      </div>
    </div>
  );
}
