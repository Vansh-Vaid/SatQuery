// Mutable, render-free shared state between DOM (scroll, drag) and the WebGL scene.
export const scene = {
  hero: 0,        // 0 at top -> 1 after the first viewport has scrolled past
  dragVel: 0,     // radians / frame added to globe spin by pointer drag (decays)
  dragTilt: 0,    // vertical drag offset (radians, clamped)
  dragging: false,
};

/** The demo scene ships georeferenced in UTM 32N near Lake Constance. */
export const TARGET = { lat: 47.8418, lon: 9.0171, label: "DEMO SCENE · LAKE CONSTANCE REGION" };
