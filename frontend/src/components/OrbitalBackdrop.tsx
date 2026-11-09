// Console backdrop in the shared "survey atlas" style: an off-white field with a lit atlas globe in the
// lower-left that rotates and zooms to lock onto the loaded scene's WGS84 footprint (orange outline).
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { atlasTexture, coastlines, D2R, graticule, ll, orbitRing, rimFrag, rimVert } from "../three/geo";

type BBox = [number, number, number, number] | null;
interface Props { busy: boolean; focus: BBox; reduced: boolean }

const INK = "#2A2D1E";
const ORANGE = "#D2651F";
const OLIVE = "#6B7A2E";

function Satellite({ radius, tilt, phase, speed, color, reduced }:
  { radius: number; tilt: [number, number]; phase: number; speed: number; color: string; reduced: boolean }) {
  const sat = useRef<THREE.Mesh>(null);
  const ring = useMemo(() => {
    const l = new THREE.Line(orbitRing(radius), new THREE.LineDashedMaterial({ color: INK, transparent: true, opacity: 0.22, dashSize: 0.03, gapSize: 0.025 }));
    l.computeLineDistances();
    return l;
  }, [radius]);
  useFrame(({ clock }) => {
    if (!sat.current) return;
    const t = phase + (reduced ? 0 : clock.elapsedTime * speed);
    sat.current.position.set(Math.cos(t) * radius, 0, Math.sin(t) * radius);
  });
  return (
    <group rotation={[tilt[0], 0, tilt[1]]}>
      <primitive object={ring} />
      <mesh ref={sat}><sphereGeometry args={[0.028, 12, 12]} /><meshBasicMaterial color={color} /></mesh>
    </group>
  );
}

function Footprint({ bbox }: { bbox: [number, number, number, number] }) {
  const geo = useMemo(() => {
    const [w, s, e, n] = bbox;
    const cx = (w + e) / 2, cy = (s + n) / 2;
    const hw = Math.max((e - w) / 2, 0.9), hh = Math.max((n - s) / 2, 0.9);
    const c = [[cx - hw, cy - hh], [cx + hw, cy - hh], [cx + hw, cy + hh], [cx - hw, cy + hh], [cx - hw, cy - hh]];
    const pts: number[] = [];
    for (let i = 0; i < 4; i++) pts.push(...ll(c[i][1], c[i][0], 1.006).toArray(), ...ll(c[i + 1][1], c[i + 1][0], 1.006).toArray());
    return new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  }, [bbox]);
  const centre = useMemo(() => ll((bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2, 1.008), [bbox]);
  const pulse = useRef<THREE.Mesh>(null);
  useFrame(({ clock }) => { if (pulse.current) pulse.current.scale.setScalar(1 + 0.8 * ((clock.elapsedTime * 0.8) % 1)); });
  return (
    <group>
      <lineSegments geometry={geo}><lineBasicMaterial color={ORANGE} /></lineSegments>
      <mesh position={centre} ref={pulse}><sphereGeometry args={[0.02, 12, 12]} /><meshBasicMaterial color={ORANGE} transparent opacity={0.85} /></mesh>
    </group>
  );
}

function Earth({ focus, busy, reduced }: { focus: BBox; busy: boolean; reduced: boolean }) {
  const { viewport, invalidate } = useThree();
  useEffect(() => { invalidate(); }, [focus, busy, invalidate]);
  const spin = useRef<THREE.Group>(null);
  const coast = useMemo(() => coastlines(1.002), []);
  const grid = useMemo(() => graticule(1.001, 20), []);
  const atlas = useMemo(() => atlasTexture(1024), []);
  useEffect(() => () => atlas.dispose(), [atlas]);
  const haloU = useMemo(() => ({ uColor: { value: new THREE.Color("#c9d08f") }, uPow: { value: 3.2 } }), []);
  const idleLon = useRef(-20);
  const scaleRef = useRef(1);
  useFrame((_, dt) => {
    const g = spin.current;
    if (!g) return;
    let lat = 22, lon: number;
    if (focus) {
      lat = (focus[1] + focus[3]) / 2;
      lon = (focus[0] + focus[2]) / 2;
    } else {
      if (!reduced) idleLon.current += dt * (busy ? 14 : 4);
      lon = idleLon.current;
    }
    const k = reduced ? 1 : Math.min(1, dt * 1.8);
    let dy = -lon * D2R - g.rotation.y;
    dy = Math.atan2(Math.sin(dy), Math.cos(dy));
    g.rotation.x += (lat * D2R - g.rotation.x) * k;
    g.rotation.y += dy * k;
    scaleRef.current += ((focus ? 1.18 : 1) - scaleRef.current) * k;
    g.scale.setScalar(scaleRef.current);
    const moving = Math.abs(dy) > 1e-3 || Math.abs(lat * D2R - g.rotation.x) > 1e-3 || Math.abs((focus ? 1.18 : 1) - scaleRef.current) > 1e-3;
    if ((moving || (busy && !focus)) && !reduced) invalidate();
  });
  const r = Math.min(viewport.width, viewport.height) * 0.36;
  return (
    <group position={[-viewport.width * 0.3, -viewport.height * 0.28, 0]} scale={r}>
      <group ref={spin} rotation-order="XYZ">
        <mesh rotation={[0, -Math.PI / 2, 0]}>
          <sphereGeometry args={[1, 64, 64]} />
          <meshStandardMaterial map={atlas} roughness={0.9} />
        </mesh>
        <lineSegments geometry={grid}><lineBasicMaterial color={INK} transparent opacity={0.07} /></lineSegments>
        <lineSegments geometry={coast}><lineBasicMaterial color={INK} transparent opacity={0.45} /></lineSegments>
        {focus && <Footprint bbox={focus} />}
      </group>
      <mesh scale={1.07}>
        <sphereGeometry args={[1, 64, 64]} />
        <shaderMaterial vertexShader={rimVert} fragmentShader={rimFrag} uniforms={haloU} transparent depthWrite={false} side={THREE.BackSide} />
      </mesh>
      <Satellite radius={1.32} tilt={[1.2, 0.35]} phase={0} speed={0} color={OLIVE} reduced />
      <Satellite radius={1.45} tilt={[1.05, -0.5]} phase={2.1} speed={0} color={ORANGE} reduced />
    </group>
  );
}

export default function OrbitalBackdrop({ busy, focus, reduced }: Props) {
  return (
    <div className="backdrop" aria-hidden="true">
      <Canvas dpr={[1, 1.25]} camera={{ position: [0, 0, 10], fov: 45 }} frameloop="demand"
        gl={{ antialias: true, alpha: false }} onCreated={({ gl }) => gl.setClearColor("#faf7f0")}>
        <ambientLight intensity={1.1} />
        <directionalLight position={[-3, 2.5, 4]} intensity={1.7} color="#fffaf0" />
        <Earth focus={focus} busy={busy} reduced={reduced} />
      </Canvas>
    </div>
  );
}
