// Shared globe geometry for the console backdrop and the landing hero.
import * as THREE from "three";
import { feature } from "topojson-client";
import land110 from "world-atlas/land-110m.json";

export const D2R = Math.PI / 180;
export const CYAN = new THREE.Color("#00F0FF");

/** lon=0, lat=0 faces +z (the camera). */
export function ll(lat: number, lon: number, r: number): THREE.Vector3 {
  return new THREE.Vector3(r * Math.cos(lat * D2R) * Math.sin(lon * D2R), r * Math.sin(lat * D2R),
    r * Math.cos(lat * D2R) * Math.cos(lon * D2R));
}

export function graticule(r: number, step: number): THREE.BufferGeometry {
  const pts: number[] = [];
  for (let lon = -180; lon < 180; lon += step)
    for (let lat = -90; lat < 90; lat += 3) pts.push(...ll(lat, lon, r).toArray(), ...ll(lat + 3, lon, r).toArray());
  for (let lat = -90 + step; lat < 90; lat += step)
    for (let lon = -180; lon < 180; lon += 3) pts.push(...ll(lat, lon, r).toArray(), ...ll(lat, lon + 3, r).toArray());
  return new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
}

let coastCache: Float32Array | null = null;

/** Natural Earth 110m land outlines as line segments on a sphere of radius r. */
export function coastlines(r: number): THREE.BufferGeometry {
  if (!coastCache) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const topo = land110 as any;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const fc = feature(topo, topo.objects.land) as unknown as { features: { geometry: { type: string; coordinates: any } }[] };
    const pts: number[] = [];
    const ring = (coords: number[][]) => {
      for (let i = 0; i < coords.length - 1; i++) {
        const [lo1, la1] = coords[i];
        const [lo2, la2] = coords[i + 1];
        pts.push(...ll(la1, lo1, 1).toArray(), ...ll(la2, lo2, 1).toArray());
      }
    };
    for (const f of fc.features) {
      const g = f.geometry;
      if (g.type === "Polygon") (g.coordinates as number[][][]).forEach(ring);
      else if (g.type === "MultiPolygon") (g.coordinates as number[][][][]).forEach((p) => p.forEach(ring));
    }
    coastCache = new Float32Array(pts);
  }
  const scaled = coastCache.map((v) => v * r);
  return new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(scaled, 3));
}

/** Equirectangular "atlas" texture: tinted ocean, land shaded by latitude band (ice, boreal, dry, tropical). */
export function atlasTexture(width = 2048): THREE.CanvasTexture {
  const h = width / 2;
  const c = document.createElement("canvas");
  c.width = width;
  c.height = h;
  const g = c.getContext("2d")!;
  const ocean = g.createLinearGradient(0, 0, 0, h);
  ocean.addColorStop(0, "#ece9dc");
  ocean.addColorStop(0.5, "#cdd7c4");
  ocean.addColorStop(1, "#ece9dc");
  g.fillStyle = ocean;
  g.fillRect(0, 0, width, h);
  const land = g.createLinearGradient(0, 0, 0, h);
  const y = (lat: number) => (90 - lat) / 180;
  land.addColorStop(y(90), "#f4f2ec");
  land.addColorStop(y(66), "#e9ece4");
  land.addColorStop(y(58), "#a3ad68");   // boreal
  land.addColorStop(y(42), "#bfc27e");   // temperate
  land.addColorStop(y(28), "#ebb877");   // dry belt
  land.addColorStop(y(12), "#8f9c4a");   // tropics
  land.addColorStop(y(-12), "#8a9945");
  land.addColorStop(y(-28), "#e8b674");
  land.addColorStop(y(-45), "#b9bf7c");
  land.addColorStop(y(-60), "#e9ece4");
  land.addColorStop(y(-90), "#f6f6f2");
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const topo = land110 as any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const fc = feature(topo, topo.objects.land) as unknown as { features: { geometry: { type: string; coordinates: any } }[] };
  const px = (lon: number) => ((lon + 180) / 360) * width;
  const py = (lat: number) => ((90 - lat) / 180) * h;
  g.beginPath();
  const ring = (coords: number[][]) => {
    coords.forEach(([lo, la], i) => {
      // break the path where a ring crosses the antimeridian to avoid horizontal streaks
      if (i === 0 || Math.abs(lo - coords[i - 1][0]) > 180) g.moveTo(px(lo), py(la));
      else g.lineTo(px(lo), py(la));
    });
    g.closePath();
  };
  for (const f of fc.features) {
    const geo = f.geometry;
    if (geo.type === "Polygon") (geo.coordinates as number[][][]).forEach(ring);
    else if (geo.type === "MultiPolygon") (geo.coordinates as number[][][][]).forEach((p) => p.forEach(ring));
  }
  g.fillStyle = land;
  g.fill("evenodd");
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

export function orbitRing(radius: number, segments = 256): THREE.BufferGeometry {
  const pts: number[] = [];
  for (let i = 0; i <= segments; i++) {
    const t = (i / segments) * Math.PI * 2;
    pts.push(Math.cos(t) * radius, 0, Math.sin(t) * radius);
  }
  return new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
}

export const rimVert = /* glsl */ `
  varying vec3 vN; varying vec3 vV;
  void main(){ vec4 mv = modelViewMatrix * vec4(position,1.0); vN = normalMatrix * normal; vV = -mv.xyz;
               gl_Position = projectionMatrix * mv; }`;
export const rimFrag = /* glsl */ `
  varying vec3 vN; varying vec3 vV; uniform vec3 uColor; uniform float uPow;
  void main(){ float f = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), uPow);
               gl_FragColor = vec4(uColor * f, f); }`;
