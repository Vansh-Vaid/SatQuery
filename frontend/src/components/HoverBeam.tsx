// Spatial cross-linking: glowing cyan bezier beams from a hovered answer entity to its boxes (spec §7.1)
import { useEffect, useState } from "react";

interface Props { source: HTMLElement | null; targets: string[] }

export default function HoverBeam({ source, targets }: Props) {
  const [paths, setPaths] = useState<string[]>([]);
  useEffect(() => {
    if (!source || !targets.length) {
      setPaths([]);
      return;
    }
    let raf = 0;
    const tick = () => {
      const s = source.getBoundingClientRect();
      const out: string[] = [];
      for (const id of targets) {
        const el = document.getElementById(`box-${id}`);
        if (!el) continue;
        const t = el.getBoundingClientRect();
        const x1 = s.left, y1 = s.top + s.height / 2;
        const x2 = t.left + t.width / 2, y2 = t.top + t.height / 2;
        const dx = Math.max(80, Math.abs(x2 - x1) * 0.45);
        out.push(`M ${x1} ${y1} C ${x1 - dx} ${y1}, ${x2 + dx} ${y2}, ${x2} ${y2}`);
      }
      setPaths(out);
      raf = requestAnimationFrame(tick);
    };
    tick();
    return () => cancelAnimationFrame(raf);
  }, [source, targets]);
  if (!paths.length) return null;
  return (
    <svg className="beams" aria-hidden="true">
      <defs>
        <filter id="beam-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3" result="b" />
          <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>
      {paths.map((d, i) => (
        <g key={i} filter="url(#beam-glow)">
          <path d={d} className="beam" />
          <path d={d} className="beam-core" />
        </g>
      ))}
    </svg>
  );
}
