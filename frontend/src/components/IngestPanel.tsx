// PANEL // 01 // INGEST - multimodal ingestion & compatibility gate (spec §5.2.1)
import { useRef, useState, type ReactElement } from "react";
import type { Check, ImageEntry, InspectResult, Mode, Scenario, Slot } from "../api";

interface Props {
  mode: Mode;
  onMode: (m: Mode) => void;
  scenarios: Scenario[];
  demo: string;
  onDemo: (id: string) => void;
  slots: Slot[];
  onSlot: (i: number, patch: Partial<Slot>) => void;
  inspect: InspectResult | null;
  inspecting: boolean;
  inspectError: string;
}

const MODES: { id: Mode; label: string; hint: string }[] = [
  { id: "single", label: "Single image", hint: "One optical, multispectral or radar image" },
  { id: "cross_modal", label: "Optical + radar", hint: "A co-registered Sentinel-2 and Sentinel-1 pair" },
  { id: "bitemporal", label: "Before & after", hint: "Two images of the same place on different dates" },
];
const SLOT_LABELS: Record<Mode, string[]> = {
  single: ["Image"],
  cross_modal: ["Optical / multispectral", "Radar (SAR)"],
  bitemporal: ["Before (T₁)", "After (T₂)"],
};

function sensorTag(im: ImageEntry): string {
  if (im.modality === "sar") return `${im.bands.length}-CH SAR (${im.bands.join(", ")})`;
  if (im.modality === "multispectral") return `${im.bands.length}-CH MSI`;
  return "3-CH RGB";
}

function statusOf(checks: Check[], id: string) {
  return checks.find((c) => c.id === id);
}

function Badge({ k, v, s }: { k: string; v: string; s?: "pass" | "warn" | "fail" | "info" }) {
  return (
    <div className={`badge badge-${s ?? "info"}`}>
      <span className="badge-k">{k}:</span> <span className="badge-v">{v}</span>
    </div>
  );
}

function CompatibilityGate({ r }: { r: InspectResult }) {
  const [open, setOpen] = useState(false);
  const badges: ReactElement[] = [];
  r.images.forEach((im, i) => {
    const f = statusOf(r.checks, `image${i + 1}.format`);
    const tag = r.images.length > 1 ? (r.config === "bitemporal" ? (i === 0 ? "Before " : "After ") : `Image ${i + 1} `) : "";
    badges.push(<Badge key={`f${i}`} k={`${tag}format`} v={`${im.crs ? "GeoTIFF" : im.name.split(".").pop()?.toUpperCase()} · ${f?.status ?? "pass"}`}
      s={f?.status} />);
  });
  const ref = r.images[0];
  const grid = statusOf(r.checks, "pair.grid");
  if (r.images.length === 2) {
    const state = !grid ? "unverified" : grid.status === "fail" ? "misaligned"
      : /reprojected|resized/i.test(grid.message) ? "reprojected" : "co-registered";
    badges.push(<Badge key="crs" k="Alignment" v={`${ref.crs ?? "no CRS"} · ${state}`} s={grid?.status ?? "warn"} />);
  } else {
    badges.push(<Badge key="crs" k="CRS" v={`${ref.crs ?? "not georeferenced"}${ref.gsd_m ? ` · ${ref.gsd_m} m pixels` : ""}`}
      s={ref.crs ? "pass" : "warn"} />);
  }
  badges.push(<Badge key="dim" k="Size" v={`${ref.size[1]} × ${ref.size[0]} px · ${r.images.map(sensorTag).join(" + ")}`} />);
  if (r.config === "bitemporal" && r.images[0].acquired && r.images[1]?.acquired) {
    const d = (Date.parse(r.images[1].acquired) - Date.parse(r.images[0].acquired)) / 864e5;
    badges.push(<Badge key="t" k="Dates" v={`${r.images[0].acquired} → ${r.images[1].acquired} (${Math.round(d)} days)`}
      s={statusOf(r.checks, "pair.dates")?.status} />);
  }
  const fails = r.checks.filter((c) => c.status === "fail").length;
  const warns = r.checks.filter((c) => c.status === "warn").length;
  return (
    <div className="gate">
      <div className={`gate-head ${r.ok ? "open" : "blocked"}`}>
        <span>{r.ok ? "Ready to analyse" : "Blocked"}</span>
        <span className="mono-muted">{r.checks.length - fails - warns} passed · {warns} warnings · {fails} failed</span>
      </div>
      <div className="badges">{badges}</div>
      <button className="link-btn" onClick={() => setOpen(!open)} aria-expanded={open}>
        {open ? "− hide" : "+ show"} all {r.checks.length} checks
      </button>
      {open && (
        <ul className="checks">
          {r.checks.map((c) => (
            <li key={c.id} className={`check-${c.status}`}>
              <span className="mono">{c.status}</span> {c.message}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function DropSlot({ label, slot, onSlot, preview, disabled }:
  { label: string; slot: Slot; onSlot: (p: Partial<Slot>) => void; preview?: string; disabled: boolean }) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  return (
    <div className="slot">
      <div className={`drop ${over ? "over" : ""} ${slot.file || preview ? "filled" : ""} ${disabled ? "disabled" : ""}`}
        role="button" tabIndex={disabled ? -1 : 0} aria-label={`Upload ${label}`}
        onClick={() => !disabled && input.current?.click()}
        onKeyDown={(e) => { if (!disabled && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); input.current?.click(); } }}
        onDragOver={(e) => { e.preventDefault(); if (!disabled) setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); if (!disabled && e.dataTransfer.files[0]) onSlot({ file: e.dataTransfer.files[0] }); }}>
        <input ref={input} type="file" hidden accept=".tif,.tiff,.png,.jpg,.jpeg"
          onChange={(e) => onSlot({ file: e.target.files?.[0] ?? null })} />
        {preview ? <img src={preview} alt="" className="drop-thumb" /> : <div className="drop-thumb empty">+</div>}
        <div className="drop-text">
          <span className="micro">{label}</span>
          <span className="drop-name">{slot.file ? slot.file.name : disabled ? "provided by demo scenario" : "Drop GeoTIFF / TIFF or click"}</span>
          {slot.file && <span className="mono-muted">{(slot.file.size / 1048576).toFixed(2)} MB</span>}
        </div>
      </div>
      {!disabled && (
        <div className="slot-opts">
          <select value={slot.modality} onChange={(e) => onSlot({ modality: e.target.value })} aria-label={`${label} sensor`}>
            <option value="auto">Sensor: detect automatically</option>
            <option value="multispectral">Sentinel-2 multispectral</option>
            <option value="optical_rgb">Optical RGB</option>
            <option value="sar">Sentinel-1 radar</option>
          </select>
          <input type="date" value={slot.date} onChange={(e) => onSlot({ date: e.target.value })} aria-label={`${label} acquisition date`} />
        </div>
      )}
    </div>
  );
}

export default function IngestPanel(p: Props) {
  const labels = SLOT_LABELS[p.mode];
  const demoMode = Boolean(p.demo);
  return (
    <section className="hud-panel ingest" aria-label="Data ingestion and sensor validation">
      <header className="panel-head">
        <span className="panel-code">01 · Inputs</span>
        <h2>Imagery</h2>
      </header>
      <div className="panel-scroll">
        <div className="mode-tabs" role="tablist" aria-label="Input mode">
          {MODES.map((m) => (
            <button key={m.id} role="tab" aria-selected={p.mode === m.id} title={m.hint}
              className={p.mode === m.id ? "active" : ""} onClick={() => p.onMode(m.id)}>
              {m.label}
            </button>
          ))}
        </div>
        <p className="mode-hint">{MODES.find((m) => m.id === p.mode)?.hint}</p>

        <label className="micro" htmlFor="demo">Demo scene</label>
        <select id="demo" value={p.demo} onChange={(e) => p.onDemo(e.target.value)}>
          <option value="">Use my own images</option>
          {p.scenarios.filter((s) => modeOf(s.id) === p.mode).map((s) => <option key={s.id} value={s.id}>{s.title}</option>)}
        </select>

        <div className="slots">
          {labels.map((label, i) => (
            <DropSlot key={`${p.mode}-${i}`} label={label} slot={p.slots[i]} disabled={demoMode}
              onSlot={(patch) => p.onSlot(i, patch)} preview={p.inspect?.images[i]?.layers[0]?.png} />
          ))}
        </div>

        <h3 className="sub">Input checks</h3>
        {p.inspecting && <div className="mono-muted pulse">Checking the images…</div>}
        {p.inspectError && <div className="error" role="alert">{p.inspectError}</div>}
        {p.inspect && !p.inspecting && <CompatibilityGate r={p.inspect} />}
        {!p.inspect && !p.inspecting && !p.inspectError && (
          <div className="mono-muted">Choose a demo scene or add GeoTIFF images. PNG and JPEG are accepted for benchmark images.</div>
        )}
      </div>
    </section>
  );
}

export function modeOf(scenarioId: string): Mode {
  if (scenarioId.startsWith("cross")) return "cross_modal";
  if (scenarioId.startsWith("bitemporal")) return "bitemporal";
  return "single";
}
