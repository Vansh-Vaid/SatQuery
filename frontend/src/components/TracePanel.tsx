// PANEL // 04 // AGENT-TRACE - deterministic DAG + telemetry log of the observable execution (spec §5.2.4)
import { useLayoutEffect, useRef, useState } from "react";
import type { AnalyzeResult, Step, ToolSpec } from "../api";
import { useScramble } from "../hooks/motion";

interface NodeProps {
  id: string; code: string; title: string; status: string; delay: number; children?: React.ReactNode;
  meta?: React.ReactNode; detail?: unknown;
}

function TraceNode({ id, code, title, status, delay, children, meta, detail }: NodeProps) {
  const [open, setOpen] = useState(false);
  const t = useScramble(title, 300, delay);
  return (
    <li className={`node node-${status}`} data-node={id}>
      <div className="node-head">
        <span className="node-code">{code}</span>
        <span className={`node-dot ${status}`} aria-label={status} />
      </div>
      <h3 className="node-title">{t || " "}</h3>
      {meta && <div className="node-meta mono-muted">{meta}</div>}
      {children}
      {detail !== undefined && (
        <>
          <button className="link-btn" onClick={() => setOpen(!open)} aria-expanded={open}>
            {open ? "Hide parameters" : "Show parameters & facts"}
          </button>
          {open && <pre className="code">{JSON.stringify(detail, null, 2)}</pre>}
        </>
      )}
    </li>
  );
}

interface Props { result: AnalyzeResult | null; busy: boolean; tools: ToolSpec[] }

export default function TracePanel({ result, busy, tools }: Props) {
  const list = useRef<HTMLOListElement>(null);
  const [edges, setEdges] = useState<{ d: string; dep: boolean }[]>([]);
  const t = result?.trace;
  const steps: Step[] = t?.steps ?? [];
  const reqs = Object.fromEntries(tools.map((x) => [x.name, x.requires]));
  const titles = Object.fromEntries(tools.map((x) => [x.name, x.title]));

  // draw the DAG: spine through every node + curved dependency edges between tool nodes
  useLayoutEffect(() => {
    const ol = list.current;
    if (!ol) return;
    const draw = () => {
      const nodes = Array.from(ol.querySelectorAll<HTMLLIElement>("li.node"));
      const y = (el: HTMLElement) => el.offsetTop + 22;
      const out: { d: string; dep: boolean }[] = [];
      for (let i = 0; i < nodes.length - 1; i++) out.push({ d: `M 22 ${y(nodes[i])} L 22 ${y(nodes[i + 1])}`, dep: false });
      const pos: Record<string, number> = {};
      nodes.forEach((n) => { pos[n.dataset.node ?? ""] = y(n); });
      steps.forEach((s) => {
        (reqs[s.tool] ?? []).forEach((dep) => {
          const a = pos[`tool:${dep}`], b = pos[`tool:${s.tool}`];
          if (a !== undefined && b !== undefined && b - a > 1) {
            const bulge = Math.min(18, 6 + (b - a) / 40);
            out.push({ d: `M 22 ${a} C ${22 - bulge} ${a + 20}, ${22 - bulge} ${b - 20}, 18 ${b - 3}`, dep: true });
          }
        });
      });
      setEdges(out);
    };
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(ol);
    return () => ro.disconnect();
  }, [result, tools]); // eslint-disable-line react-hooks/exhaustive-deps

  const checks = t?.validation ?? [];
  const grid = checks.find((c) => c.id === "pair.grid");
  const fails = checks.filter((c) => c.status === "fail").length;
  const corrections = t?.plan?.corrections ?? [];

  return (
    <section className="trace" aria-label="Agentic execution trace">
      <header className="panel-head">
        <span className="panel-code">What the agent did</span>
        <h2>Execution trace</h2>
      </header>
      {!t && !busy && <p className="mono-muted">No execution yet. The observable trace (task, verified inputs, selected tools,
        permitted parameters, outputs) renders here as a directed graph.</p>}
      {busy && <p className="mono pulse cyan">Planning and running specialists…</p>}
      {t && (
        <div className="dag">
          <svg className="dag-edges" aria-hidden="true">
            <defs>
              <marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
                <path d="M0,0 L8,4 L0,8 z" fill="#d2651f" />
              </marker>
            </defs>
            {edges.map((e, i) => (
              <path key={i} d={e.d} className={e.dep ? "edge dep" : "edge"} markerEnd={e.dep ? "url(#arrow)" : undefined} />
            ))}
          </svg>
          <ol className="nodes" ref={list}>
            <TraceNode id="task" code="1 · Task" status={result?.status === "rejected" ? "fail" : "ok"} delay={0}
              title={(result?.task ?? "rejected").replace(/_/g, " ")}
              meta={<>intent {t.plan?.intent ?? "—"} · planner {t.planner?.name ?? "—"}
                {t.planner ? ` · ${t.planner.duration_ms} ms` : ""}</>}>
              {t.plan?.rationale && <p className="node-body">{t.plan.rationale}</p>}
            </TraceNode>
            <TraceNode id="input" code="2 · Input checks" status={fails ? "fail" : checks.some((c) => c.status === "warn") ? "warn" : "ok"}
              delay={60} title={`${t.input.images.length} image${t.input.images.length > 1 ? "s" : ""} · ${t.input.configuration.replace("_", " ")}`}
              meta={<>{t.input.images.map((i) => String(i.modality).replace("_", " ")).join(" + ")}
                {grid ? ` · ${grid.status === "pass" ? "co-registered" : grid.status}` : ""}</>}>
              <p className="node-body">{checks.length - fails} / {checks.length} checks passed{fails ? ` · ${fails} blocking` : ""}</p>
            </TraceNode>
            {corrections.length > 0 && (
              <TraceNode id="validator" code="3 · Plan validator" status="warn" delay={120}
                title={`${corrections.length} plan correction${corrections.length > 1 ? "s" : ""}`}>
                <ul className="node-list">{corrections.map((c) => <li key={c}>{c}</li>)}</ul>
              </TraceNode>
            )}
            {steps.map((s, i) => (
              <TraceNode key={s.tool} id={`tool:${s.tool}`} code={`${i + (corrections.length ? 4 : 3)} · ${s.kind}`}
                status={s.status === "ok" ? "ok" : s.status === "error" ? "fail" : "idle"} delay={180 + i * 70}
                title={titles[s.tool] ?? s.title}
                meta={<><span className="mono">{s.tool}</span> · {s.status} · {s.duration_ms} ms
                  {s.confidence != null ? ` · C = ${s.confidence.toFixed(3)}` : ""}</>}
                detail={{ params: s.params, facts: s.facts, notes: s.notes }}>
                <p className="node-body">{s.output ?? s.notes.join("; ")}</p>
              </TraceNode>
            ))}
            {t.result && (
              <TraceNode id="integrate" code="Result" status="ok" delay={200 + steps.length * 70}
                title={`Answer from ${t.result.primary_tool ?? "—"}`}
                meta={<>confidence {t.result.confidence?.toFixed(3) ?? "n/a"} · {t.result.confidence_label} · {t.duration_ms} ms total</>}>
                {t.result.integration_notes.length > 0 && (
                  <ul className="node-list">{t.result.integration_notes.map((n) => <li key={n}>{n}</li>)}</ul>
                )}
              </TraceNode>
            )}
          </ol>
        </div>
      )}
    </section>
  );
}
