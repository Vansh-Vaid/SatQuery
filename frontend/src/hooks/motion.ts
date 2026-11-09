import { useEffect, useRef, useState } from "react";

/** True when the user asked the OS for reduced motion; all decorative animation keys off this. */
export function useReducedMotion(): boolean {
  const q = "(prefers-reduced-motion: reduce)";
  const [reduced, set] = useState(() => typeof window !== "undefined" && window.matchMedia(q).matches);
  useEffect(() => {
    const m = window.matchMedia(q);
    const on = () => set(m.matches);
    m.addEventListener("change", on);
    return () => m.removeEventListener("change", on);
  }, []);
  return reduced;
}

const GLYPHS = "01X#ΔΣ/<>[]=+*";

/** 300 ms glyph-decoding scramble from random telemetry glyphs to `text` (spec §7.2). */
export function useScramble(text: string, durationMs = 300, delayMs = 0): string {
  const reduced = useReducedMotion();
  const [out, setOut] = useState(reduced ? text : "");
  const raf = useRef(0);
  useEffect(() => {
    if (reduced) {
      setOut(text);
      return;
    }
    let start = 0;
    const tick = (t: number) => {
      if (!start) start = t;
      const p = (t - start - delayMs) / durationMs;
      if (p >= 1) {
        setOut(text);
        return;
      }
      const reveal = Math.max(0, Math.floor(p * text.length));
      let s = text.slice(0, reveal);
      for (let i = reveal; i < text.length; i++) {
        s += text[i] === " " ? " " : GLYPHS[(Math.random() * GLYPHS.length) | 0];
      }
      setOut(p < 0 ? "" : s);
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [text, durationMs, delayMs, reduced]);
  return out;
}
