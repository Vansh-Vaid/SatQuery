// SatQuery AI landing page: a short front door to the console, built from real output.
// Hero (live before/after frame + "try asking") -> layer viewer (live output) -> how it works (live facts)
// -> measured results -> call to action. Off-white / orange / olive theme; no decorative 3D.
import { memo, useEffect, useState } from "react";
import type { AnalyzeResult, Mode } from "../api";
import { useReducedMotion } from "../hooks/motion";
import { useReveal } from "./interactions";
import { DEMO_RUNS, type RunState, useDemoRuns, useTools } from "./useDemoRuns";

type StackLayer = { key: string; title: string; sub: string; png?: string; tint: string; overlay?: boolean };

const consoleUrl = (mode: Mode, q?: string) => {
  const p = new URLSearchParams({ mode, demo: DEMO_RUNS[mode].demo });
  if (q) p.set("q", q);
  return `/console?${p.toString()}`;
};

const HERO_QUERIES: { q: string; mode: Mode }[] = [
  { q: "What changed between these two dates, and where did the change occur?", mode: "bitemporal" },
  { q: "Use the optical and SAR images together to identify built-up and water-covered regions.", mode: "cross_modal" },
  { q: "Highlight the water body referred to in the query.", mode: "single" },
  { q: "Has the built-up area increased, decreased, or remained unchanged?", mode: "bitemporal" },
  { q: "Describe the land-cover and major objects visible in this image.", mode: "single" },
];
// Held-out benchmark results (docs/TECHNICAL.md, "Measured results")
const PROOF: { score: number; title: string; source: string; before?: number; ref?: number; note?: string }[] = [
  { score: 95.7, title: "Recognising land use", source: "EuroSAT · 5,399 Sentinel-2 test images · 10 classes",
    note: "93% on raw satellite files, loaded the way your uploads are" },
  { score: 84.6, title: "Answering questions about one image", source: "RSVQA-LR · 10,004 test questions", before: 41.8 },
  { score: 65.1, title: "Answering questions about change", source: "CDVQA · 6,000 questions on real aerial image pairs",
    before: 18.7, ref: 67.7 },
  { score: 56.0, title: "Answering questions about sharp aerial photos", source: "VRSBench · 37,409 questions about planes, ships, vehicles…",
    before: 23.8 },
  { score: 38.5, title: "Finding a described object", source: "VRSBench · 16,159 referring expressions (box overlap ≥ 50%)",
    before: 0 },
];

function useTypewriter(items: string[], reduced: boolean) {
  const [i, setI] = useState(0);
  const [n, setN] = useState(reduced ? items[0].length : 0);
  useEffect(() => {
    if (reduced) { setN(items[i].length); return; }
    const full = items[i].length;
    const id = setTimeout(() => {
      if (n < full) setN(n + 1);
      else { setI((i + 1) % items.length); setN(0); }
    }, n < full ? 26 : 2800);
    return () => clearTimeout(id);
  }, [i, n, items, reduced]);
  return { index: i, text: items[i].slice(0, n), done: n >= items[i].length };
}

/** The cycling "Try asking" prompt. Isolated so its 26 ms typing ticks re-render only this element. */
const TryAsking = memo(function TryAsking({ reduced }: { reduced: boolean }) {
  const typed = useTypewriter(HERO_QUERY_TEXT, reduced);
  const h = HERO_QUERIES[typed.index];
  return (
    <a className="lp-prompt" href={consoleUrl(h.mode, h.q)} aria-label={`Try this question: ${h.q}`}>
      <span className="lp-prompt-label" aria-hidden="true">Try asking</span>
      <span className="lp-prompt-text" aria-hidden="true">{typed.text}<i className={typed.done ? "blink" : ""} /></span>
      <span className="lp-prompt-go" aria-hidden="true">Ask <span>→</span></span>
    </a>
  );
});
const HERO_QUERY_TEXT = HERO_QUERIES.map((h) => h.q);

function layersFor(mode: Mode, r?: AnalyzeResult): StackLayer[] {
  const v = r?.viewer;
  const img = (k: number, l = 0) => v?.images[k]?.layers[l]?.png;
  const ov = (title: string) => v?.overlays.find((o) => o.title === title)?.png;
  if (mode === "single") return [
    { key: "true", title: "True colour", sub: "Sentinel-2 · bands 4, 3, 2", png: img(0, 0), tint: "#8C8370" },
    { key: "nir", title: "False-colour infrared", sub: "Bands 8, 4, 3 · vegetation shows red", png: img(0, 1), tint: "#8A9A3C" },
    { key: "g", title: "Grounded water body", sub: "Mask and box returned for the phrase “the water body”", png: ov("grounded: water"), tint: "#D2651F", overlay: true },
  ];
  if (mode === "cross_modal") return [
    { key: "opt", title: "Optical, partly cloudy", sub: "Sentinel-2 · 18% of the scene hidden by cloud", png: img(0, 0), tint: "#8C8370" },
    { key: "sar", title: "Radar backscatter", sub: "Sentinel-1 VV / VH · unaffected by cloud", png: img(1, 0), tint: "#4F5B22" },
    { key: "bu", title: "Fused built-up area", sub: "Radar double-bounce confirmed by the built-up index", png: ov("fused: built-up area"), tint: "#D2651F", overlay: true },
    { key: "w", title: "Fused water", sub: "Water index plus low radar return; cloud gaps filled", png: ov("fused: water"), tint: "#4F86A6", overlay: true },
  ];
  return [
    { key: "t1", title: "Before · 1 June 2024", sub: "Sentinel-2", png: img(0, 0), tint: "#8C8370" },
    { key: "t2", title: "After · 5 June 2025", sub: "Sentinel-2", png: img(1, 0), tint: "#6E7A74" },
    { key: "c", title: "Detected change", sub: "Normalised change-vector analysis with a physical floor", png: ov("change mask"), tint: "#D2651F", overlay: true },
  ];
}

const MODES: { id: Mode; label: string; claim: string }[] = [
  { id: "single", label: "Single image", claim: "Ask about one optical, multispectral or radar scene, and point at what you mean." },
  { id: "cross_modal", label: "Optical + radar", claim: "Radar sees through the cloud that hides optical pixels. Combined, each covers the other's blind spots." },
  { id: "bitemporal", label: "Before & after", claim: "Two dates on one grid: where it changed, what it changed into, and by how much." },
];

/** Live before/after frame: the real demo pair, a swipe slider, the change overlay and the backend's answer. */
function EvidenceFrame({ run }: { run: RunState }) {
  const [split, setSplit] = useState(50);
  const [showChange, setShowChange] = useState(true);
  const v = run.result?.viewer;
  const before = v?.images[0]?.layers[0]?.png, after = v?.images[1]?.layers[0]?.png;
  const change = v?.overlays.find((o) => o.title === "change mask")?.png;
  const r = run.result;
  return (
    <figure className="lp-frame" aria-label="Live analysis of a before-and-after Sentinel-2 pair">
      <div className="lp-frame-img">
        {before && after ? (
          <>
            <img src={after} alt="After: 5 June 2025, Sentinel-2 true colour" />
            <img src={before} alt="Before: 1 June 2024, Sentinel-2 true colour" style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }} />
            {showChange && change && <img className="lp-frame-overlay" src={change} alt="" />}
            <span className="lp-frame-tag left">Before · 2024</span><span className="lp-frame-tag right">After · 2025</span>
            <i className="lp-frame-rule" style={{ left: `${split}%` }} aria-hidden="true" />
          </>
        ) : <div className="lp-frame-wait" role="status">{run.status === "error" ? "Demo unavailable: the backend is offline" : "Running the agent on a demo pair…"}</div>}
      </div>
      <div className="lp-frame-controls">
        <label className="lp-frame-slider">
          <span className="sr-only">Swipe between before and after</span>
          <input type="range" min={0} max={100} value={split} onChange={(e) => setSplit(+e.target.value)}
            aria-label="Swipe between before and after" disabled={!before} />
        </label>
        <label className="lp-frame-toggle"><input type="checkbox" checked={showChange} onChange={(e) => setShowChange(e.target.checked)} /> Show change</label>
      </div>
      <figcaption className="lp-frame-answer">
        <span className="lp-frame-q">“{DEMO_RUNS.bitemporal.query}”</span>
        {r ? <><b>{r.answer.split(". ")[0]}.</b><em>{Math.round((r.confidence ?? 0) * 100)}% confidence · {r.confidence_label} · {Math.round(r.trace.duration_ms)} ms</em></>
          : <em>{run.status === "error" ? "Open the console to try it." : "Answering…"}</em>}
      </figcaption>
    </figure>
  );
}

/** Flat layer viewer: the base image with the selected (or all) evidence layers on top. */
function LayerView({ layers, selected }: { layers: StackLayer[]; selected: number | null }) {
  const base = layers.find((l) => !l.overlay && l.png);
  const pick = selected != null ? layers[selected] : null;
  const shown = pick && !pick.overlay ? pick : base;
  return (
    <div className="lp-layers">
      {shown?.png ? <img src={shown.png} alt={shown.title} /> : <div className="lp-frame-wait" role="status">Loading the layers…</div>}
      {layers.filter((l) => l.overlay && l.png && (pick == null || pick === l)).map((l) => (
        <img key={l.key} className="lp-frame-overlay" src={l.png} alt={l.title} />
      ))}
      <span className="lp-layers-tag">{pick ? pick.title : "All layers"}</span>
    </div>
  );
}

function Logo() {
  return (
    <svg className="lp-logo" viewBox="0 0 32 32" aria-hidden="true">
      <circle cx="16" cy="16" r="8.5" fill="none" stroke="currentColor" strokeWidth="2.2" />
      <ellipse cx="16" cy="16" rx="14" ry="5" fill="none" stroke="#D2651F" strokeWidth="1.8" transform="rotate(-24 16 16)" />
      <circle cx="27.5" cy="10.6" r="2.2" fill="#D2651F" />
    </svg>
  );
}

export default function Landing() {
  const reduced = useReducedMotion();
  const { runs, load } = useDemoRuns();
  const tools = useTools();
  const [mode, setMode] = useState<Mode>("cross_modal");
  const [hovered, setHovered] = useState<number | null>(null);
  const [scrolled, setScrolled] = useState(false);
  useReveal([runs.bitemporal.status, runs.cross_modal.status, tools.length, mode]);

  useEffect(() => { load("bitemporal"); load("cross_modal"); }, [load]);
  useEffect(() => { load(mode); }, [mode, load]);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 16);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);
  const onLayerHover = (i: number | null) => setHovered(i);
  const lifted = hovered;

  const run = runs[mode];
  const layers = layersFor(mode, run.result);
  const scrollTo = (id: string) => document.getElementById(id)?.scrollIntoView({ behavior: reduced ? "auto" : "smooth" });

  // live facts for the three steps, from the cross-modal demo run
  const live = runs.cross_modal.result;
  const checks = live?.trace.validation ?? [];
  const passed = checks.filter((c) => c.status === "pass").length;
  const ran = live?.trace.steps?.length ?? 0;
  const facts = [
    live ? `${passed} of ${checks.length} input checks passed` : "Checking the demo scene…",
    live ? `${ran} of ${tools.length || 9} registered tools ran for this question` : "Planning…",
    live ? `${Math.round((live.confidence ?? 0) * 100)}% confidence · answered in ${Math.round(live.trace.duration_ms)} ms` : "Answering…",
  ];

  return (
    <div className="lp">
      <a className="lp-skip" href="#main">Skip to content</a>
      <header className={`lp-nav ${scrolled ? "solid" : ""}`}>
        <a className="lp-brand" href="/" aria-label="SatQuery AI home"><Logo /><span>SatQuery</span><em>AI</em></a>
        <nav className="lp-nav-links" aria-label="Primary">
          <a href="#seeing">Product</a><a href="#how">How it works</a><a href="#proof">Results</a><a href="/console">Console</a>
        </nav>
        <a className="lp-btn lp-btn-sm" href="/console">Open console</a>
      </header>

      <main id="main">
        <section className="lp-hero" aria-labelledby="hero-title">
          <div className="lp-wrap lp-hero-grid">
            <div className="lp-hero-copy">
              <p className="lp-kicker" data-reveal><i />Earth observation · Agentic AI</p>
              <h1 id="hero-title" data-reveal style={{ ["--d" as string]: "60ms" }}>Ask the <em>planet</em><br />a question.</h1>
              <p className="lp-lead" data-reveal style={{ ["--d" as string]: "120ms" }}>
                SatQuery reads optical, radar and before-and-after satellite imagery. It picks the right
                remote-sensing specialists for your question, runs them, and answers with maps, a confidence
                score and a full record of what it did.
              </p>
              <div data-reveal style={{ ["--d" as string]: "180ms" }}><TryAsking reduced={reduced} /></div>
              <div className="lp-cta-row" data-reveal style={{ ["--d" as string]: "240ms" }}>
                <a className="lp-btn" href="/console">Open the console</a>
                <button className="lp-btn-text" onClick={() => scrollTo("seeing")}>See it work <span aria-hidden="true">↓</span></button>
              </div>
            </div>

            <div data-reveal style={{ ["--d" as string]: "120ms" }}><EvidenceFrame run={runs.bitemporal} /></div>
          </div>
        </section>

        <section id="seeing" className="lp-band" aria-labelledby="seeing-title">
          <div className="lp-wrap">
            <div className="lp-seeing">
              <div className="lp-seeing-stage" data-reveal>
                <LayerView layers={layers} selected={lifted} />
                <p className={`lp-source ${run.status === "ok" ? "live" : ""}`} role="status">
                  {run.status === "ok" ? "Live output from the running backend · demo Sentinel scenes"
                    : run.status === "loading" ? "Running the agent…" : "Illustrative render · backend offline"}
                </p>
              </div>
              <div className="lp-seeing-side">
                <p className="lp-label" data-reveal>One question, three ways of seeing</p>
                <h2 id="seeing-title" data-reveal>Every input becomes layers you can check.</h2>
                <div className="lp-tabs" role="tablist" aria-label="Input type">
                  {MODES.map((m) => (
                    <button key={m.id} role="tab" aria-selected={mode === m.id} className={mode === m.id ? "on" : ""}
                      onClick={() => { setMode(m.id); setHovered(null); }}>{m.label}</button>
                  ))}
                </div>
                <p className="lp-claim">{MODES.find((m) => m.id === mode)?.claim}</p>
                <ol className="lp-legend" aria-label="Layers, top to bottom">
                  {[...layers].reverse().map((l) => {
                    const i = layers.indexOf(l);
                    return (
                      <li key={l.key} className={lifted === i ? "on" : ""} tabIndex={0}
                        onMouseEnter={() => onLayerHover(i)} onMouseLeave={() => onLayerHover(null)}
                        onFocus={() => onLayerHover(i)} onBlur={() => onLayerHover(null)}>
                        <span className="lp-swatch" style={{ background: l.tint }} />
                        <span><b>{l.title}</b><em>{l.sub}</em></span>
                      </li>
                    );
                  })}
                </ol>
                <a className="lp-link" href={consoleUrl(mode, DEMO_RUNS[mode].query)}>Open this scene in the console <span aria-hidden="true">→</span></a>
              </div>
            </div>
          </div>
        </section>

        <section id="how" className="lp-band lp-band-sand" aria-labelledby="how-title">
          <div className="lp-wrap">
            <header className="lp-head lp-head-split">
              <div>
                <p className="lp-label" data-reveal>How it works</p>
                <h2 id="how-title" data-reveal>From question to a checked answer in three steps.</h2>
              </div>
              <p className="lp-sub" data-reveal>Each step shows a live fact from the demo run above, so you can see the agent working, not just read about it.</p>
            </header>
            <ol className="lp-steps">
              {[
                ["Checks your images", "Format, sensor type, map projection, overlap and dates are verified before anything runs. Bad inputs are stopped with the reason."],
                ["Picks the right specialists", "Claude, or a rule planner when offline, routes the question to registered tools: models trained on satellite imagery plus physical measurement tools."],
                ["Answers with evidence", "Maps, boxes, a confidence score and a full trace of every tool and setting, exportable as PDF, GeoJSON or JSON."],
              ].map(([t, b], i) => (
                <li key={t} data-reveal style={{ ["--d" as string]: `${i * 90}ms` }}>
                  <span className="lp-step-n">{i + 1}</span>
                  <h3>{t}</h3>
                  <p>{b}</p>
                  <span className={`lp-step-fact ${live ? "live" : ""}`}>{facts[i]}</span>
                </li>
              ))}
            </ol>
          </div>
        </section>

        <section id="proof" className="lp-band" aria-labelledby="proof-title">
          <div className="lp-wrap">
            <header className="lp-head lp-head-split">
              <div>
                <p className="lp-label" data-reveal>Measured, not claimed</p>
                <h2 id="proof-title" data-reveal>Scored on test questions it never saw in training.</h2>
              </div>
              <p className="lp-sub" data-reveal>Every score comes from held-out test data. The grey bar shows the same
                system before its trained models were added.</p>
            </header>
            <ol className="lp-proof">
              {PROOF.map((p, i) => (
                <li key={p.title} data-reveal style={{ ["--d" as string]: `${i * 90}ms` }}>
                  <p className="lp-proof-n">{p.score}<span>%</span></p>
                  <h3>{p.title}</h3>
                  <p className="lp-proof-src">{p.source}</p>
                  <div className="lp-bars" aria-hidden="true">
                    <div className="lp-bar now"><span style={{ ["--w" as string]: `${p.score}%` }} /></div>
                    {p.before != null && <div className="lp-bar before"><span style={{ ["--w" as string]: `${p.before}%` }} /></div>}
                    {p.ref != null && <i className="lp-ref" style={{ left: `${p.ref}%` }} />}
                  </div>
                  <p className="lp-proof-foot">
                    {p.before != null ? <>Before: <b>{p.before}%</b></> : p.note}
                    {p.ref != null && <> · Benchmark authors' model: <b>{p.ref}%</b></>}
                  </p>
                </li>
              ))}
            </ol>
            <p className="lp-proof-note" data-reveal>
              Change questions are scored on 6,000 of the benchmark's test questions; the "before" scores for change and
              aerial-photo questions come from stratified samples. Full method and per-question results are in
              the <a href="https://github.com/Vansh-Vaid/SatQuery/blob/main/docs/TECHNICAL.md">technical reference</a>.
            </p>
          </div>
        </section>

        <section className="lp-band lp-band-ink lp-final" aria-labelledby="final-title">
          <div className="lp-wrap lp-final-grid">
            <div>
              <h2 id="final-title" data-reveal>Ask your first question.</h2>
              <p className="lp-sub" data-reveal>The console comes loaded with demo scenes, so you can try every capability without uploading anything.</p>
            </div>
            <div className="lp-cta-row" data-reveal>
              <a className="lp-btn lp-btn-light" href="/console">Open the console</a>
              <a className="lp-btn-text light" href={consoleUrl("bitemporal", HERO_QUERIES[0].q)}>Try a change question <span aria-hidden="true">→</span></a>
            </div>
          </div>
        </section>
      </main>

      <footer className="lp-foot">
        <div className="lp-wrap lp-foot-grid">
          <a className="lp-brand" href="/"><Logo /><span>SatQuery</span><em>AI</em></a>
          <p>An agentic vision-language assistant for multimodal remote-sensing analysis.</p>
          <p className="lp-data">Trained on EuroSAT, RSVQA-LR, CDVQA and VRSBench · Coastlines: Natural Earth</p>
        </div>
      </footer>
    </div>
  );
}
