// SatQuery AI landing page: a short front door to the console.
// Hero (draggable 3D globe + "try asking") -> ticker -> 3D multimodal layer stack (live output)
// -> three-step "how it works" with live facts -> call to action. Off-white / orange / olive theme.
import { memo, useEffect, useRef, useState } from "react";
import type { AnalyzeResult, Mode } from "../api";
import { fmtLat, fmtLon } from "../api";
import { useReducedMotion } from "../hooks/motion";
import { Contours, Ticker } from "./Flourishes";
import HeroGlobe from "./HeroGlobe";
import { useReveal } from "./interactions";
import LayerStack, { type StackLayer } from "./LayerStack";
import { scene, TARGET } from "./store";
import { DEMO_RUNS, useDemoRuns, useTools } from "./useDemoRuns";

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
const MORE_QUERIES: { q: string; mode: Mode }[] = [
  { q: "How much of the area changed?", mode: "bitemporal" },
  { q: "Is this a rural or urban area?", mode: "single" },
  { q: "What did the water turn into?", mode: "bitemporal" },
  { q: "How much of the scene is water?", mode: "cross_modal" },
];
const TAG: Record<Mode, string> = { single: "Single image", cross_modal: "Optical + radar", bitemporal: "Before & after" };

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
  const marker = useRef<HTMLButtonElement>(null);
  const tele = [useRef<HTMLSpanElement>(null), useRef<HTMLSpanElement>(null)];
  const globe = useRef<HTMLDivElement>(null);
  useReveal([runs.cross_modal.status, tools.length, mode]);

  useEffect(() => { load("cross_modal"); }, [load]);
  useEffect(() => { load(mode); }, [mode, load]);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 16);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // drag the globe to rotate it
  useEffect(() => {
    const el = globe.current;
    if (!el) return;
    let x = 0, y = 0;
    const down = (e: PointerEvent) => {
      if ((e.target as Element).closest("button,a")) return;
      scene.dragging = true; x = e.clientX; y = e.clientY;
      el.setPointerCapture(e.pointerId); el.classList.add("grabbing");
    };
    const move = (e: PointerEvent) => {
      if (!scene.dragging) return;
      scene.dragVel = (e.clientX - x) * 0.005;
      scene.dragTilt = Math.max(-0.7, Math.min(0.7, scene.dragTilt + (e.clientY - y) * 0.004));
      x = e.clientX; y = e.clientY;
    };
    const up = () => { scene.dragging = false; el.classList.remove("grabbing"); };
    el.addEventListener("pointerdown", down); el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", up); el.addEventListener("pointercancel", up);
    return () => { el.removeEventListener("pointerdown", down); el.removeEventListener("pointermove", move);
      el.removeEventListener("pointerup", up); el.removeEventListener("pointercancel", up); };
  }, []);

  // when nobody is interacting with the stack, lift each layer in turn so the section never sits still
  const userHover = useRef(false);
  const [autoLayer, setAutoLayer] = useState<number | null>(null);
  useEffect(() => {
    if (reduced) return;
    const n = layersFor(mode, runs[mode].result).length;
    let i = 0;
    const id = setInterval(() => {
      if (userHover.current) return;
      setAutoLayer(i % (n + 1) === n ? null : i % (n + 1));
      i++;
    }, 1700);
    return () => clearInterval(id);
  }, [mode, runs, reduced]);
  const onLayerHover = (i: number | null) => { userHover.current = i !== null; setHovered(i); };
  const lifted = hovered ?? autoLayer;

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
          <a href="#seeing">Product</a><a href="#how">How it works</a><a href="/console">Console</a>
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

            <div className="lp-globe" data-reveal style={{ ["--d" as string]: "120ms" }}>
              <div className="lp-globe-stage" ref={globe}>
                <Contours />
                <HeroGlobe reduced={reduced} marker={marker} telemetry={tele} />
                <button ref={marker} className="lp-marker" onClick={() => scrollTo("seeing")}
                  aria-label="Demo scene near Lake Constance. View its analysis layers.">
                  <span className="lp-marker-label"><b>Demo scene</b>{fmtLat(TARGET.lat)}, {fmtLon(TARGET.lon)}<em>View its layers ↓</em></span>
                </button>
              </div>
              <dl className="lp-telemetry" aria-label="Live satellite positions (illustrative orbits)">
                <div><dt><i className="dot sar" />Sentinel-1 · radar</dt><dd><span ref={tele[0]}>—</span></dd></div>
                <div><dt><i className="dot msi" />Sentinel-2 · optical</dt><dd><span ref={tele[1]}>—</span></dd></div>
              </dl>
            </div>
          </div>
        </section>

        <Ticker items={[...HERO_QUERIES, ...MORE_QUERIES].map((h) => ({ q: h.q, href: consoleUrl(h.mode, h.q), tag: TAG[h.mode] }))} />

        <section id="seeing" className="lp-band" aria-labelledby="seeing-title">
          <div className="lp-wrap">
            <div className="lp-seeing">
              <div className="lp-seeing-stage" data-reveal>
                <LayerStack layers={layers} hovered={lifted} onHover={onLayerHover} reduced={reduced} />
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
                ["Picks the right specialists", "Claude, or a rule planner when offline, routes the question to registered tools: an adapted vision-language model plus physical measurement tools."],
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
          <p className="lp-data">Coastlines: Natural Earth · Training data: BigEarthNet.txt (CDLA-Permissive-1.0)</p>
        </div>
      </footer>
    </div>
  );
}
