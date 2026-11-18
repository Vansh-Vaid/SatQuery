import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { AnalyzeResult, Health, InspectResult, Mode, Scenario, Slot, ToolSpec } from "./api";
import { api } from "./api";
import EvidenceCard from "./components/EvidenceCard";
import GeoCanvas from "./components/GeoCanvas";
import HoverBeam from "./components/HoverBeam";
import IngestPanel, { modeOf } from "./components/IngestPanel";
import OrbitalBackdrop from "./components/OrbitalBackdrop";
import QueryDeck from "./components/QueryDeck";
import TracePanel from "./components/TracePanel";
import { useReducedMotion } from "./hooks/motion";

const REPRESENTATIVE = [
  "Describe the land-cover and major objects visible in this image.",
  "Highlight the water body referred to in the query.",
  "What changed between these two dates, and where did the change occur?",
  "Use the optical and SAR images together to identify built-up and water-covered regions.",
  "Has the built-up area increased, decreased, or remained unchanged?",
];
const emptySlot = (): Slot => ({ file: null, modality: "auto", date: "" });

function useUtcClock() {
  const [now, set] = useState(() => new Date());
  useEffect(() => {
    const id = setInterval(() => set(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return now.toISOString().slice(11, 19) + "Z";
}

const STAGES = ["Checking the inputs", "Planning which tools to use", "Running the specialists", "Cross-checking the evidence"];

function WorkingSteps({ reduced }: { reduced: boolean }) {
  // the four stages the backend runs, in order; the stage shown advances with elapsed time
  const [i, setI] = useState(0);
  useEffect(() => {
    const t = [350, 900, 2200].map((ms, k) => setTimeout(() => setI(k + 1), reduced ? 0 : ms));
    return () => t.forEach(clearTimeout);
  }, [reduced]);
  return (
    <div className="working" role="status" aria-live="polite">
      <p className="working-title">Working on it…</p>
      <ol>
        {STAGES.map((s, k) => (
          <li key={s} className={k < i ? "done" : k === i ? "now" : ""}><i aria-hidden="true" />{s}</li>
        ))}
      </ol>
      <div className="working-skel"><span /><span /><span /></div>
    </div>
  );
}

export default function App() {
  const reduced = useReducedMotion();
  const clock = useUtcClock();
  const [health, setHealth] = useState<Health | null>(null);
  const [tools, setTools] = useState<ToolSpec[]>([]);
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [mode, setMode] = useState<Mode>("cross_modal");
  const [demo, setDemo] = useState("");
  const [slots, setSlots] = useState<Slot[]>([emptySlot(), emptySlot()]);
  const [inspect, setInspect] = useState<InspectResult | null>(null);
  const [inspecting, setInspecting] = useState(false);
  const [inspectError, setInspectError] = useState("");
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [scanKey, setScanKey] = useState(0);
  const [beam, setBeam] = useState<{ ids: string[]; el: HTMLElement | null }>({ ids: [], el: null });
  const [hoverBox, setHoverBox] = useState<string | null>(null);
  const feed = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
    api.tools().then((r) => setTools(r.tools)).catch(() => undefined);
    // deep links from the landing page: /console?mode=bitemporal&demo=bitemporal_s2&q=...
    const params = new URLSearchParams(window.location.search);
    const wantMode = params.get("mode") as Mode | null;
    if (wantMode && ["single", "cross_modal", "bitemporal"].includes(wantMode)) setMode(wantMode);
    api.demos().then((r) => {
      setScenarios(r.scenarios);
      const asked = r.scenarios.find((s) => s.id === params.get("demo"));
      const first = asked ?? r.scenarios.find((s) => modeOf(s.id) === (wantMode ?? "cross_modal"));
      if (first) { setMode(modeOf(first.id)); setDemo(first.id); setQuery(params.get("q") ?? first.queries[0]); }
    }).catch(() => undefined);
  }, []);

  const activeSlots = useMemo(() => (mode === "single" ? slots.slice(0, 1) : slots), [mode, slots]);
  const hasInput = Boolean(demo) || activeSlots.some((s) => s.file);

  // compatibility gate: re-validate whenever the inputs change (debounced)
  useEffect(() => {
    setResult(null);
    if (!hasInput) { setInspect(null); setInspectError(""); return; }
    const id = setTimeout(() => {
      setInspecting(true);
      setInspectError("");
      api.inspect(mode, demo, activeSlots)
        .then(setInspect)
        .catch((e: Error) => { setInspect(null); setInspectError(e.message); })
        .finally(() => setInspecting(false));
    }, 250);
    return () => clearTimeout(id);
  }, [mode, demo, activeSlots, hasInput]);

  const onMode = (m: Mode) => {
    setMode(m);
    const ownFiles = slots.some((s) => s.file);
    if (!ownFiles) {
      const first = scenarios.find((s) => modeOf(s.id) === m);
      setDemo(first?.id ?? "");
      if (first) setQuery(first.queries[0]);
    } else setDemo("");
  };
  const onDemo = (id: string) => {
    setDemo(id);
    const sc = scenarios.find((s) => s.id === id);
    if (sc) setQuery(sc.queries[0]);
  };
  const onSlot = (i: number, patch: Partial<Slot>) => {
    if (patch.file) setDemo("");
    setSlots((s) => s.map((x, j) => (j === i ? { ...x, ...patch } : x)));
  };

  const run = useCallback(async (override?: string) => {
    const q = (override ?? query).trim();
    if (!q || !hasInput || busy) return;
    if (override) setQuery(override);
    setBusy(true);
    setError("");
    setResult(null);
    if (mode !== "single") setScanKey((k) => k + 1); // SAR C-band pulse for cross-modal / change workflows
    try {
      setResult(await api.analyze(mode, demo, activeSlots, q));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }, [query, hasInput, busy, mode, demo, activeSlots]);

  // on stacked (narrow) layouts the answer is below the fold: bring it into view when work starts and when it lands
  useEffect(() => {
    if (!(busy || result) || !window.matchMedia("(max-width: 1023px)").matches) return;
    const id = requestAnimationFrame(() => {
      const el = document.querySelector(".col-right");
      if (el) window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY - 8, behavior: reduced ? "auto" : "smooth" });
    });
    return () => cancelAnimationFrame(id);
  }, [busy, result, reduced]);

  const viewer = result?.viewer;
  const images = viewer?.images?.length ? viewer.images : inspect?.images ?? [];
  const width = viewer?.width ?? inspect?.images[0]?.size[1] ?? 1;
  const height = viewer?.height ?? inspect?.images[0]?.size[0] ?? 1;
  const focus = images[0]?.bbox_wgs84 ?? null;
  const chips = scenarios.find((s) => s.id === demo)?.queries ?? REPRESENTATIVE;
  const highlighted = useMemo(() => new Set([...beam.ids, ...(hoverBox ? [hoverBox] : [])]), [beam.ids, hoverBox]);
  const supporting = (result?.evidence ?? []).filter((e) => !e.primary);
  const available = tools.filter((t) => t.available).length;
  const trained = tools.filter((t) => t.name !== "rs_vlm" && /^learned/.test(t.kind));
  const trainedReady = trained.filter((t) => t.available).length;
  const blocked = inspect !== null && !inspect.ok;

  return (
    <div className="app">
      <OrbitalBackdrop busy={busy} focus={focus} reduced={reduced} />
      {scanKey > 0 && !reduced && <div key={scanKey} className="radar-pulse" aria-hidden="true" />}
      <a className="skip" href="#query">Skip to query</a>

      <header className="topbar">
        {busy && <span className="busy-bar" role="progressbar" aria-label="Analysis running" />}
        <div className="brand">
          <h1 className="wordmark">
            <a href="/" title="Back to the SatQuery AI home page">
              <svg className="logo" viewBox="0 0 32 32" aria-hidden="true">
                <circle cx="16" cy="16" r="8.5" fill="none" stroke="currentColor" strokeWidth="2.2" />
                <ellipse cx="16" cy="16" rx="14" ry="5" fill="none" stroke="#D2651F" strokeWidth="1.8" transform="rotate(-24 16 16)" />
                <circle cx="27.5" cy="10.6" r="2.2" fill="#D2651F" />
              </svg>
              SatQuery <em>AI</em>
            </a>
          </h1>
          <span className="console-chip">Console</span>
        </div>
        <dl className="telemetry">
          <div title={health?.planner}><dt>Planner</dt><dd><i className={`sdot ${health ? "ok" : ""}`} />
            {health ? (/claude/i.test(health.planner) && !/no claude/i.test(health.planner) ? "Claude" : "Rule planner") : "—"}</dd></div>
          <div title={trained.map((t) => `${t.title}: ${t.available ? "ready" : t.status}`).join("\n")}>
            <dt>Trained models</dt><dd><i className={`sdot ${trainedReady ? "ok" : "warn"}`} />{trainedReady} of {trained.length} ready</dd></div>
          <div title={health?.vlm.status}><dt>Free-form VLM</dt><dd><i className={`sdot ${health?.vlm.available ? "ok" : "off"}`} />
            {health ? (health.vlm.available ? "Online" : "Offline") : "—"}</dd></div>
          <div><dt>Tools</dt><dd className="mono">{available}/{tools.length}</dd></div>
          <div className="hide-sm"><dt>UTC</dt><dd className="mono">{clock}</dd></div>
        </dl>
      </header>

      <main className="grid">
        <div className="col-left">
          <IngestPanel mode={mode} onMode={onMode} scenarios={scenarios} demo={demo} onDemo={onDemo} slots={slots}
            onSlot={onSlot} inspect={inspect} inspecting={inspecting} inspectError={inspectError} />
        </div>

        <div className="col-center">
          <GeoCanvas images={images} overlays={viewer?.overlays ?? []} boxes={viewer?.boxes ?? []} width={width} height={height}
            config={result?.trace.input.configuration ?? inspect?.config ?? null} busy={busy} highlighted={highlighted}
            onHoverBox={setHoverBox} />
          <div id="query">
            <QueryDeck query={query} onQuery={setQuery} onRun={() => run()} chips={chips} busy={busy}
              disabled={!hasInput || blocked} />
          </div>
        </div>

        <aside className="col-right hud-panel" aria-label="Evidence synthesis feed">
          <div className="feed" ref={feed} onScroll={() => { if (beam.ids.length) setBeam({ ids: [], el: null }); }}>
            <div className="feed-content">
              <header className="panel-head">
                <span className="panel-code">04 · Answer</span>
                <h2>Answer</h2>
              </header>
              {error && <div className="error" role="alert">{error}</div>}
              {!result && !busy && !error && (
                <div className="empty-answer">
                  <p className="ea-title">{hasInput ? "Ask a question to get a checked answer." : "Choose a demo scene or add an image to begin."}</p>
                  <ul className="ea-list">
                    <li><b>A direct answer</b> with a confidence score</li>
                    <li><b>Regions highlighted</b> on the image</li>
                    <li><b>Every step</b> the agent took, in order</li>
                    <li><b>A report</b> as PDF, GeoJSON or JSON</li>
                  </ul>
                  {hasInput && !blocked && (
                    <div className="ea-try">
                      <span>Try one</span>
                      {chips.slice(0, 3).map((c) => (
                        <button key={c} className="ea-q" onClick={() => run(c)}>{c}<i aria-hidden="true">→</i></button>
                      ))}
                    </div>
                  )}
                </div>
              )}
              {busy && <WorkingSteps reduced={reduced} />}
              {result && (
                <>
                  <EvidenceCard result={result} reduced={reduced} onEntity={(ids, el) => setBeam({ ids, el })} />
                  <div className="export">
                    <a className="export-main" href={api.reportUrl(result.request_id, "zip")} download>
                      Download report bundle (.zip)
                    </a>
                    <div className="export-links mono">
                      <a href={api.reportUrl(result.request_id, "pdf")} download>PDF</a>
                      <a href={api.reportUrl(result.request_id, "md")} download>Markdown</a>
                      <a href={api.reportUrl(result.request_id, "geojson")} download>GeoJSON</a>
                      <a href={api.reportUrl(result.request_id, "html")} download>HTML</a>
                      <a href={api.reportUrl(result.request_id, "json")} download>JSON trace</a>
                    </div>
                  </div>
                  {supporting.length > 0 && (
                    <details className="supporting">
                      <summary>Supporting evidence · {supporting.length} tools</summary>
                      <ul>
                        {supporting.map((e) => (
                          <li key={e.tool}><span className="mono cyan">{e.tool}</span>
                            {e.confidence != null && <span className="mono-muted"> C={e.confidence.toFixed(2)}</span>}<br />{e.text}</li>
                        ))}
                      </ul>
                    </details>
                  )}
                </>
              )}
              <TracePanel result={result} busy={busy} tools={tools} />
            </div>
          </div>
        </aside>
      </main>
      <HoverBeam source={beam.el} targets={beam.ids} />
    </div>
  );
}
