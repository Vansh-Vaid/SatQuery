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

  return (
    <div className="card-wrap" onMouseMove={onMove} onMouseLeave={() => { setTilt({ x: 0, y: 0 }); onEntity([], null); }}>
      <div ref={card} className={`evidence-card ${flipped ? "flipped" : ""}`}
        style={{ transform: `rotateX(${tilt.x}deg) rotateY(${tilt.y + (flipped ? 180 : 0)}deg)` }}>
        <div className="face front" aria-hidden={flipped}>
          <div className="card-badge"><b>{(result.task ?? "rejected").replace(/_/g, " ")}</b> · {provenance}</div>
          <p className="answer">
            {parts.map((p, i) => p.ids.length ? (
              <span key={i} className="entity" tabIndex={0}
                onMouseEnter={(e) => onEntity(p.ids, e.currentTarget)} onMouseLeave={() => onEntity([], null)}
                onFocus={(e) => onEntity(p.ids, e.currentTarget)} onBlur={() => onEntity([], null)}>{p.t}</span>
            ) : <span key={i}>{p.t}</span>)}
          </p>
          {result.warnings.length > 0 && <ul className="warn-list">{result.warnings.map((w) => <li key={w}>⚠ {w}</li>)}</ul>}
          <div className="telemetry-bar">
            <div className="conf">
              <span className="micro">Confidence <b>{conf != null ? `${(conf * 100).toFixed(1)}%` : "n/a"}</b> · {result.confidence_label}</span>
              <div className="meter"><span className={result.confidence_label} style={{ width: `${(conf ?? 0) * 100}%` }} /></div>
            </div>
            <div className="card-actions">
              {allIds.length > 0 && (
                <button className="link-btn" onMouseEnter={(e) => onEntity(allIds, e.currentTarget)}
                  onMouseLeave={() => onEntity([], null)}>Hover to highlight regions</button>
              )}
              <button className="link-btn" onClick={() => setFlipped(true)}>Show audit log →</button>
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
