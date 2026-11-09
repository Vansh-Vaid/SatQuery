// Live runs of the real backend on the bundled demo scenes. The landing page shows genuine outputs,
// and degrades to illustrative renders (clearly labelled) when the backend is unreachable.
import { useCallback, useEffect, useRef, useState } from "react";
import type { AnalyzeResult, Mode, ToolSpec } from "../api";
import { api } from "../api";

export const DEMO_RUNS: Record<Mode, { demo: string; query: string }> = {
  single: { demo: "single_s2", query: "Highlight the water body referred to in the query." },
  cross_modal: { demo: "cross_modal", query: "Use the optical and SAR images together to identify built-up and water-covered regions." },
  bitemporal: { demo: "bitemporal_s2", query: "What changed between these two dates, and where did the change occur?" },
};

export type RunState = { status: "idle" | "loading" | "ok" | "error"; result?: AnalyzeResult; error?: string };

export function useDemoRuns() {
  const [runs, setRuns] = useState<Record<Mode, RunState>>({
    single: { status: "idle" }, cross_modal: { status: "idle" }, bitemporal: { status: "idle" },
  });
  const started = useRef(new Set<Mode>());
  const load = useCallback((m: Mode) => {
    if (started.current.has(m)) return;
    started.current.add(m);
    setRuns((r) => ({ ...r, [m]: { status: "loading" } }));
    const { demo, query } = DEMO_RUNS[m];
    api.analyze(m, demo, [{ file: null, modality: "auto", date: "" }, { file: null, modality: "auto", date: "" }], query)
      .then((result) => setRuns((r) => ({ ...r, [m]: { status: "ok", result } })))
      .catch((e: Error) => {
        started.current.delete(m);
        setRuns((r) => ({ ...r, [m]: { status: "error", error: e.message } }));
      });
  }, []);
  return { runs, load };
}

export function useTools() {
  const [tools, set] = useState<ToolSpec[]>([]);
  useEffect(() => { api.tools().then((r) => set(r.tools)).catch(() => undefined); }, []);
  return tools;
}
