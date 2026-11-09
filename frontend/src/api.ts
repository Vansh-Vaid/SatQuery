// Typed client for the SatQuery FastAPI backend (satquery/api/app.py).

export type Mode = "single" | "cross_modal" | "bitemporal";
export type CheckStatus = "pass" | "warn" | "fail";

export interface Check { id: string; status: CheckStatus; message: string }
export interface BaseLayer { id: string; title: string; png: string }
export interface ImageEntry {
  index: number; name: string; modality: "multispectral" | "optical_rgb" | "sar"; bands: string[];
  acquired: string | null; bbox_wgs84: [number, number, number, number] | null; gsd_m: number | null;
  crs: string | null; size: [number, number]; layers: BaseLayer[];
}
export interface InspectResult { config: Mode | "invalid"; ok: boolean; checks: Check[]; images: ImageEntry[] }
export interface Overlay {
  id: string; title: string; kind: "mask" | "heatmap"; tool: string; legend: string; color: string | null; png: string;
}
export interface Box {
  id: string; x0: number; y0: number; x1: number; y1: number; label: string; score: number | null; source: string;
}
export interface Geo {
  crs: string | null; transform: number[] | null; bbox_wgs84: [number, number, number, number] | null;
  gsd_m: number | null; georeferenced: boolean;
}
export interface Viewer {
  width: number; height: number; images: ImageEntry[]; overlays: Overlay[]; boxes: Box[]; geo: Geo;
}
export interface Step {
  step: number; tool: string; title: string; kind: string; purpose: string; params: Record<string, unknown>;
  status: "ok" | "error" | "unavailable" | "unsupported"; duration_ms: number; confidence: number | null;
  output: string | null; facts: Record<string, unknown>; notes: string[]; n_boxes: number; layers: string[];
}
export interface Trace {
  request_id: string; started_utc: string; satquery_version: string; query: string;
  input: { configuration: string; pair_hint: string | null; images: Record<string, unknown>[] };
  validation: Check[];
  planner?: { name: string; duration_ms: number };
  plan?: { task: string; intent: string; planner: string; rationale: string; corrections: string[];
           steps: { tool: string; params: Record<string, unknown>; purpose: string }[] };
  steps?: Step[];
  result?: { task: string; intent: string; primary_tool: string | null; answer: string; short_answer: string | null;
             confidence: number | null; confidence_label: string; integration_notes: string[] };
  duration_ms: number; outcome: string;
}
export interface Evidence { tool: string; primary: boolean; text: string; confidence: number | null }
export interface AnalyzeResult {
  request_id: string; status: "ok" | "partial" | "rejected"; query: string; task: string | null; answer: string;
  short_answer: string | null; confidence: number | null; confidence_label: string; evidence: Evidence[];
  boxes: Box[]; warnings: string[]; trace: Trace; viewer: Viewer;
}
export interface ToolSpec {
  name: string; title: string; description: string; tasks: string[]; input_configs: string[]; kind: string;
  params: { name: string; type: string; default: unknown; description: string }[]; requires: string[];
  available: boolean; status: string;
}
export interface Health { status: string; version: string; planner: string; vlm: { available: boolean; status: string } }
export interface Scenario { id: string; title: string; files: string[]; queries: string[] }

export interface Slot { file: File | null; modality: string; date: string }

async function json<T>(res: Response): Promise<T> {
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error((body as { detail?: string }).detail || `${res.status} ${res.statusText}`);
  return body as T;
}

function form(mode: Mode, demo: string, slots: Slot[], query?: string): FormData {
  const fd = new FormData();
  if (query !== undefined) fd.append("query", query);
  if (demo) fd.append("demo", demo);
  slots.forEach((s, i) => {
    const n = i + 1;
    if (!demo && s.file) fd.append(`image${n}`, s.file);
    fd.append(`modality${n}`, s.modality);
    fd.append(`date${n}`, s.date);
  });
  fd.append("pair_type", mode === "single" ? "auto" : mode);
  return fd;
}

export const api = {
  health: () => fetch("/api/health").then((r) => json<Health>(r)),
  tools: () => fetch("/api/tools").then((r) => json<{ tools: ToolSpec[] }>(r)),
  demos: () => fetch("/api/demos").then((r) => json<{ scenarios: Scenario[] }>(r)),
  inspect: (mode: Mode, demo: string, slots: Slot[]) =>
    fetch("/api/inspect", { method: "POST", body: form(mode, demo, slots) }).then((r) => json<InspectResult>(r)),
  analyze: (mode: Mode, demo: string, slots: Slot[], query: string) =>
    fetch("/api/analyze", { method: "POST", body: form(mode, demo, slots, query) }).then((r) => json<AnalyzeResult>(r)),
  reportUrl: (rid: string, ext: "zip" | "pdf" | "md" | "geojson" | "html" | "json") => `/api/report/${rid}.${ext}`,
};

/** Approximate pixel -> lat/lon by bilinear interpolation inside the WGS84 footprint (fine for scene-sized tiles). */
export function pixelToLatLon(bbox: [number, number, number, number] | null, x: number, y: number, w: number, h: number) {
  if (!bbox) return null;
  const [west, south, east, north] = bbox;
  return { lat: north - (y / h) * (north - south), lon: west + (x / w) * (east - west) };
}

export const fmtLat = (v: number) => `${Math.abs(v).toFixed(4)}°${v >= 0 ? "N" : "S"}`;
export const fmtLon = (v: number) => `${Math.abs(v).toFixed(4)}°${v >= 0 ? "E" : "W"}`;
