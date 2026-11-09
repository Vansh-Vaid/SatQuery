// PANEL // 02 // GEO-CANVAS - layered viewer with split-swipe curtain, band/layer pill bar,
// grounding boxes with coordinate callouts, change masks, orbital targeting reticle and radar sweep
// loading state (spec §5.2.2, §7.2, §7.3).
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { Box, ImageEntry, Overlay } from "../api";
import { fmtLat, fmtLon, pixelToLatLon } from "../api";
import { useScramble } from "../hooks/motion";

interface Props {
  images: ImageEntry[];
  overlays: Overlay[];
  boxes: Box[];
  width: number;
  height: number;
  config: string | null;
  busy: boolean;
  highlighted: Set<string>;
  onHoverBox: (id: string | null) => void;
}

const BOX_COLOR: Record<string, string> = {
  region_grounder: "var(--box-ground)", rs_vlm: "var(--box-vlm)", change_detector: "var(--box-change)",
};

function defaultOn(o: Overlay): boolean {
  return /^grounded|^change mask|^fused: (water|built)/.test(o.title);
}

export default function GeoCanvas(p: Props) {
  const two = p.images.length === 2;
  const [baseA, setBaseA] = useState("");
  const [baseB, setBaseB] = useState("");
  const [split, setSplit] = useState(0.5);
  const [orient, setOrient] = useState<"v" | "h">("v");
  const [vis, setVis] = useState<Record<string, boolean>>({});
  const [showBoxes, setShowBoxes] = useState(true);
  const [showLayers, setShowLayers] = useState(false);
  const [cursor, setCursor] = useState<{ x: number; y: number; px: number; py: number } | null>(null);
  const stage = useRef<HTMLDivElement>(null);
  const frame = useRef<HTMLDivElement>(null);
  const [fit, setFit] = useState({ w: 0, h: 0 });
  const status = useScramble(p.busy ? "Agent running · selecting and executing specialists" : "", 300);

  // reset layer selection whenever the image set changes
  const sig = p.images.map((i) => i.name + i.layers.length).join("|");
  useEffect(() => {
    setBaseA(p.images[0]?.layers[0]?.id ?? "");
    setBaseB(p.images[1]?.layers[0]?.id ?? "");
    setSplit(0.5);
  }, [sig]); // eslint-disable-line react-hooks/exhaustive-deps
  const osig = p.overlays.map((o) => o.id).join("|");
  useEffect(() => {
    setVis(Object.fromEntries(p.overlays.map((o) => [o.id, defaultOn(o)])));
  }, [osig]); // eslint-disable-line react-hooks/exhaustive-deps

  // contain-fit the frame inside the stage, preserving the raster aspect ratio
  useLayoutEffect(() => {
    const el = stage.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const W = el.clientWidth - 24, H = el.clientHeight - 24;
      const ar = p.width / p.height || 1;
      const w = Math.max(0, Math.min(W, H * ar));
      setFit({ w, h: w / ar });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [p.width, p.height]);

  const layerA = p.images[0]?.layers.find((l) => l.id === baseA) ?? p.images[0]?.layers[0];
  const layerB = p.images[1]?.layers.find((l) => l.id === baseB) ?? p.images[1]?.layers[0];
  const bbox = p.images[0]?.bbox_wgs84 ?? null;

  const pills = useMemo(() => {
    const out: { key: string; label: string; active: boolean; onClick: () => void }[] = [];
    const names: Record<string, string> = { true: "True colour", false: "False-colour infrared", sar: "Radar (SAR)" };
    p.images.forEach((im, idx) => {
      im.layers.forEach((l) => {
        const tag = two ? (p.config === "bitemporal" ? (idx === 0 ? "Before · " : "After · ") : "") : "";
        out.push({ key: `${idx}-${l.id}`, label: tag + (names[l.id] ?? l.id),
          active: (idx === 0 ? baseA : baseB) === l.id, onClick: () => (idx === 0 ? setBaseA : setBaseB)(l.id) });
      });
    });
    return out;
  }, [p.images, p.config, two, baseA, baseB]);
  const changeIds = p.overlays.filter((o) => /^change|^transition/.test(o.title)).map((o) => o.id);
  const changeOn = changeIds.length > 0 && changeIds.some((id) => vis[id]);

  const dragTo = (clientX: number, clientY: number) => {
    const r = frame.current?.getBoundingClientRect();
    if (!r) return;
    const v = orient === "v" ? (clientX - r.left) / r.width : (clientY - r.top) / r.height;
    setSplit(Math.min(1, Math.max(0, v)));
  };
  const startDrag = (e: React.PointerEvent) => {
    e.preventDefault();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    const move = (ev: PointerEvent) => dragTo(ev.clientX, ev.clientY);
    const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };
  const onMove = (e: React.MouseEvent) => {
    const r = frame.current?.getBoundingClientRect();
    if (!r) return;
    const px = e.clientX - r.left, py = e.clientY - r.top;
    setCursor({ px, py, x: Math.floor((px / r.width) * p.width), y: Math.floor((py / r.height) * p.height) });
  };
  const clipB = orient === "v" ? `inset(0 0 0 ${split * 100}%)` : `inset(${split * 100}% 0 0 0)`;
  const geo = cursor ? pixelToLatLon(bbox, cursor.x + 0.5, cursor.y + 0.5, p.width, p.height) : null;
  const empty = p.images.length === 0;

  return (
    <section className="hud-panel geo" aria-label="Multimodal geospatial canvas">
      <header className="panel-head row">
        <div>
          <span className="panel-code">02 · Viewer</span>
          <h2>Image viewer</h2>
        </div>
        {two && (
          <div className="seg" role="group" aria-label="Split orientation">
            <button className={orient === "v" ? "active" : ""} onClick={() => setOrient("v")} aria-label="Vertical split">⇹</button>
            <button className={orient === "h" ? "active" : ""} onClick={() => setOrient("h")} aria-label="Horizontal split">⇳</button>
          </div>
        )}
      </header>

      {!empty && (
        <div className="pillbar" role="toolbar" aria-label="Band and layer switcher">
          {pills.map((pl) => (
            <button key={pl.key} className={`pill ${pl.active ? "active" : ""}`} onClick={pl.onClick}>{pl.label}</button>
          ))}
          {p.boxes.length > 0 && (
            <button className={`pill amber ${showBoxes ? "active" : ""}`} onClick={() => setShowBoxes(!showBoxes)}>
              {p.boxes.every((b) => b.source === "change_detector") ? "Change regions" : "Boxes"}
            </button>
          )}
          {changeIds.length > 0 && (
            <button className={`pill crimson ${changeOn ? "active" : ""}`}
              onClick={() => setVis((v) => ({ ...v, ...Object.fromEntries(changeIds.map((id) => [id, !changeOn])) }))}>
              Change masks
            </button>
          )}
          {p.overlays.length > 0 && (
            <button className={`pill ghost ${showLayers ? "active" : ""}`} onClick={() => setShowLayers(!showLayers)}
              aria-expanded={showLayers}>Layers {p.overlays.filter((o) => vis[o.id]).length}/{p.overlays.length}</button>
          )}
        </div>
      )}
      {showLayers && (
        <div className="layer-list">
          {p.overlays.map((o) => (
            <label key={o.id} className="layer-row">
              <input type="checkbox" checked={!!vis[o.id]} onChange={(e) => setVis((v) => ({ ...v, [o.id]: e.target.checked }))} />
              <span className="swatch" style={{ background: o.color ?? "linear-gradient(90deg,#2a2d1e,#d2651f,#f3e6c8)" }} />
              <span>{o.title}</span><span className="mono-muted">{o.tool}</span>
            </label>
          ))}
        </div>
      )}

      <div className="stage" ref={stage}>
        {empty ? (
          <div className="no-signal">
            <div className="ns-title">No imagery yet</div>
            <div className="mono-muted">Choose a demo scene or add your own images on the left.</div>
          </div>
        ) : (
          <div className="frame" ref={frame} style={{ width: fit.w, height: fit.h }} onMouseMove={onMove}
            onMouseLeave={() => setCursor(null)}>
            {layerA && <img className="layer" src={layerA.png} alt={`${p.images[0].name} ${layerA.title}`} draggable={false} />}
            {two && layerB && (
              <img className="layer" src={layerB.png} alt={`${p.images[1].name} ${layerB.title}`} style={{ clipPath: clipB }} draggable={false} />
            )}
            {p.overlays.filter((o) => vis[o.id]).map((o) => (
              <img key={o.id} className={`layer overlay ${o.kind}`} src={o.png} alt={o.title} draggable={false} />
            ))}
            {showBoxes && p.boxes.length > 0 && (
              <svg className="boxes" viewBox={`0 0 ${p.width} ${p.height}`} preserveAspectRatio="none">
                {p.boxes.map((b) => {
                  const hi = p.highlighted.has(b.id);
                  const c = BOX_COLOR[b.source] ?? "var(--box-ground)";
                  return (
                    <g key={b.id} onMouseEnter={() => p.onHoverBox(b.id)} onMouseLeave={() => p.onHoverBox(null)}>
                      <rect id={`box-${b.id}`} x={b.x0} y={b.y0} width={b.x1 - b.x0} height={b.y1 - b.y0}
                        className={`gbox ${hi ? "hi" : ""}`} style={{ stroke: c, color: c }} vectorEffect="non-scaling-stroke" />
                    </g>
                  );
                })}
              </svg>
            )}
            {showBoxes && p.boxes.map((b) => {
              const c = pixelToLatLon(bbox, (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2, p.width, p.height);
              return (
                <div key={b.id} className={`callout ${p.highlighted.has(b.id) ? "hi" : ""}`}
                  style={{ left: `${(b.x0 / p.width) * 100}%`, top: `${(b.y0 / p.height) * 100}%`,
                           borderColor: BOX_COLOR[b.source] ?? "var(--box-ground)" }}>
                  <b>{b.label}</b>
                  <span>{c ? `${fmtLat(c.lat)} ${fmtLon(c.lon)}` : `x${Math.round(b.x0)} y${Math.round(b.y0)}`}</span>
                </div>
              );
            })}
            {two && (
              <div className={`curtain ${orient}`} style={orient === "v" ? { left: `${split * 100}%` } : { top: `${split * 100}%` }}
                role="slider" tabIndex={0} aria-label="Split-swipe position" aria-valuemin={0} aria-valuemax={100}
                aria-valuenow={Math.round(split * 100)} onPointerDown={startDrag}
                onKeyDown={(e) => {
                  const d = e.shiftKey ? 0.1 : 0.02;
                  if (["ArrowLeft", "ArrowUp"].includes(e.key)) { e.preventDefault(); setSplit((s) => Math.max(0, s - d)); }
                  if (["ArrowRight", "ArrowDown"].includes(e.key)) { e.preventDefault(); setSplit((s) => Math.min(1, s + d)); }
                }}>
                <span className="curtain-grip">{orient === "v" ? "‹ | ›" : "⌃ ⌄"}</span>
              </div>
            )}
            {two && (
              <>
                <span className="corner-tag tl">{p.config === "bitemporal" ? `Before · ${p.images[0].acquired ?? ""}` : p.images[0].modality.replace("_", " ")}</span>
                <span className="corner-tag tr">{p.config === "bitemporal" ? `After · ${p.images[1].acquired ?? ""}` : p.images[1].modality.replace("_", " ")}</span>
              </>
            )}
            {cursor && (
              <div className="reticle" style={{ left: cursor.px, top: cursor.py }} aria-hidden="true">
                <span className="rh" /><span className="rv" /><span className="ring" />
                <span className="rlabel">
                  X {String(cursor.x).padStart(4, "0")} · Y {String(cursor.y).padStart(4, "0")}
                  {geo && <><br />{fmtLat(geo.lat)} {fmtLon(geo.lon)}</>}
                </span>
              </div>
            )}
            {p.busy && <div className="sweep" aria-hidden="true" />}
            <span className="bracket b1" /><span className="bracket b2" /><span className="bracket b3" /><span className="bracket b4" />
          </div>
        )}
        {p.busy && <div className="stage-status mono">{status}</div>}
      </div>
      {!empty && (
        <footer className="geo-foot mono-muted">
          <span>{p.images[0].crs ?? "NO CRS"}</span>
          {p.images[0].gsd_m && <span>GSD: {p.images[0].gsd_m}m</span>}
          <span>{p.width}×{p.height}px</span>
          {bbox && <span>BBOX {bbox.map((v) => v.toFixed(3)).join(", ")}</span>}
        </footer>
      )}
    </section>
  );
}
