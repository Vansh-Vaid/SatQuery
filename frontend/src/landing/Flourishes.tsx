// Motion & colour details for the landing page. Each one carries domain meaning:
// topographic contours and a ticker of real questions.
import { useMemo } from "react";

/** Drifting topographic contour rings (deterministic; slow rotation via CSS). */
export function Contours({ rings = 16, seed = 3 }: { rings?: number; seed?: number }) {
  const paths = useMemo(() => {
    const out: string[] = [];
    let s = seed;
    const rnd = () => ((s = (s * 9301 + 49297) % 233280) / 233280);
    for (let i = 0; i < rings; i++) {
      const r = 40 + i * 19;
      const a1 = rnd() * 6.28, a2 = rnd() * 6.28, k1 = 0.05 + rnd() * 0.05, k2 = 0.02 + rnd() * 0.04;
      let d = "";
      for (let t = 0; t <= 96; t++) {
        const th = (t / 96) * Math.PI * 2;
        const rr = r * (1 + k1 * Math.sin(3 * th + a1) + k2 * Math.sin(5 * th + a2));
        d += `${t ? "L" : "M"}${(400 + rr * Math.cos(th)).toFixed(1)},${(400 + rr * Math.sin(th) * 0.86).toFixed(1)}`;
      }
      out.push(d + "Z");
    }
    return out;
  }, [rings, seed]);
  return (
    <svg className="lp-contours" viewBox="0 0 800 800" aria-hidden="true">
      {paths.map((d, i) => (
        <path key={i} d={d} className={i % 4 === 3 ? "major" : ""} style={{ animationDelay: `${-i * 1.7}s` }} />
      ))}
    </svg>
  );
}

/** Infinite, pausable ticker of example questions (each links into the console). */
export function Ticker({ items }: { items: { q: string; href: string; tag: string }[] }) {
  const row = [...items, ...items];
  return (
    <div className="lp-ticker" aria-label="Example questions">
      <div className="lp-ticker-track">
        {row.map((it, i) => (
          <a key={i} href={it.href} className="lp-ticker-item" aria-hidden={i >= items.length} tabIndex={i >= items.length ? -1 : 0}>
            <span className="lp-ticker-tag">{it.tag}</span>{it.q}
          </a>
        ))}
      </div>
    </div>
  );
}
