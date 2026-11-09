// Hero globe in the "survey atlas" style: lit ivory sphere, ink coastlines and graticule, Sentinel-1/2
// orbits (S1 casts its radar swath) and a terracotta target on the demo scene. Drag to rotate.
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { fmtLat, fmtLon } from "../api";
import { atlasTexture, coastlines, D2R, graticule, ll, orbitRing, rimFrag, rimVert } from "../three/geo";
import { scene, TARGET } from "./store";

const INK = "#2A2D1E";
const TERRA = "#D2651F";
const LAKE = "#6B7A2E";

interface Props {
  reduced: boolean;
  marker: React.RefObject<HTMLButtonElement>;
  telemetry: React.RefObject<HTMLSpanElement>[];
}

const _v = new THREE.Vector3();
const _n = new THREE.Vector3();
const _cam = new THREE.Vector3();

function Sat({ radius, tilt, phase, speed, color, swath, earth, readout, reduced }: {
  radius: number; tilt: [number, number]; phase: number; speed: number; color: string; swath?: boolean;
  earth: React.RefObject<THREE.Group>; readout?: React.RefObject<HTMLSpanElement>; reduced: boolean;
}) {
  const sat = useRef<THREE.Group>(null);
  const ring = useMemo(() => new THREE.Line(orbitRing(radius),
    new THREE.LineDashedMaterial({ color: INK, transparent: true, opacity: 0.28, dashSize: 0.03, gapSize: 0.025 })), [radius]);
  useEffect(() => { ring.computeLineDistances(); }, [ring]);
  const tick = useRef(0);
  useFrame(({ clock }) => {
    const s = sat.current;
    if (!s) return;
    const t = phase + (reduced ? 0 : clock.elapsedTime * speed);
    s.position.set(Math.cos(t) * radius, 0, Math.sin(t) * radius);
    s.lookAt(0, 0, 0);
    if (readout?.current && earth.current && ++tick.current % 6 === 0) {
      s.getWorldPosition(_v);
      earth.current.worldToLocal(_v);
      const r = _v.length();
      readout.current.textContent = `${fmtLat(Math.asin(_v.y / r) / D2R)}  ${fmtLon(Math.atan2(_v.x, _v.z) / D2R)}`;
    }
  });
  return (
    <group rotation={[tilt[0], 0, tilt[1]]}>
      <primitive object={ring} />
      <group ref={sat}>
        <mesh><boxGeometry args={[0.045, 0.022, 0.022]} /><meshStandardMaterial color={INK} roughness={0.6} /></mesh>
        <mesh position={[0.055, 0, 0]}><planeGeometry args={[0.065, 0.026]} /><meshStandardMaterial color={color} side={THREE.DoubleSide} /></mesh>
        <mesh position={[-0.055, 0, 0]}><planeGeometry args={[0.065, 0.026]} /><meshStandardMaterial color={color} side={THREE.DoubleSide} /></mesh>
        {swath && (
          <mesh position={[0, 0, -(radius - 1) / 2]} rotation={[-Math.PI / 2, 0, 0]}>
            <coneGeometry args={[0.15, radius - 1, 40, 1, true]} />
            <meshBasicMaterial color={color} transparent opacity={0.13} side={THREE.DoubleSide} depthWrite={false} />
          </mesh>
        )}
      </group>
    </group>
  );
}

function Globe({ reduced, marker, telemetry }: Props) {
  const { camera, size } = useThree();
  const earth = useRef<THREE.Group>(null);
  const coast = useMemo(() => coastlines(1.002), []);
  const grid = useMemo(() => graticule(1.001, 15), []);
  const atlas = useMemo(() => atlasTexture(1024), []);
  useEffect(() => () => atlas.dispose(), [atlas]);
  const haloU = useMemo(() => ({ uColor: { value: new THREE.Color("#c9d08f") }, uPow: { value: 3.4 } }), []);
  const target = useMemo(() => ll(TARGET.lat, TARGET.lon, 1.003), []);
  const ring = useRef<THREE.Mesh>(null);
  const spin = useRef(-TARGET.lon * D2R + 0.55);

  useFrame(({ clock }, dt) => {
    const e = earth.current;
    if (!e) return;
    if (!reduced && !scene.dragging) spin.current += dt * 0.045;
    spin.current += scene.dragVel;
    scene.dragVel *= 0.93;
    e.rotation.set(0.5 + scene.dragTilt, spin.current, 0, "XYZ");
    if (ring.current) {
      const p = (clock.elapsedTime * 0.7) % 1;
      ring.current.scale.setScalar(1 + p * 2.2);
      (ring.current.material as THREE.MeshBasicMaterial).opacity = 0.6 * (1 - p);
    }
    const m = marker.current;
    if (m) {
      _v.copy(target).applyMatrix4(e.matrixWorld);
      _n.copy(target).normalize().transformDirection(e.matrixWorld);
      camera.getWorldPosition(_cam);
      const facing = _n.dot(_cam.sub(_v).normalize());
      _v.project(camera);
      const visible = facing > 0.2;
      m.style.transform = `translate(${(_v.x * 0.5 + 0.5) * size.width}px, ${(-_v.y * 0.5 + 0.5) * size.height}px)`;
      m.style.opacity = visible ? String(Math.min(1, (facing - 0.2) * 4)) : "0";
      m.style.pointerEvents = visible ? "auto" : "none";
      m.tabIndex = visible ? 0 : -1;
    }
  });

  const quat = useMemo(() => new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), target.clone().normalize()), [target]);
  return (
    <group scale={1.55}>
      <group ref={earth}>
        <mesh rotation={[0, -Math.PI / 2, 0]}>
          <sphereGeometry args={[1, 64, 64]} />
          <meshStandardMaterial map={atlas} roughness={0.9} metalness={0} />
        </mesh>
        <lineSegments geometry={grid}>
          <lineBasicMaterial color={INK} transparent opacity={0.08} />
        </lineSegments>
        <lineSegments geometry={coast}>
          <lineBasicMaterial color={INK} transparent opacity={0.55} />
        </lineSegments>
        <mesh position={target}><sphereGeometry args={[0.016, 16, 16]} /><meshBasicMaterial color={TERRA} /></mesh>
        <mesh ref={ring} position={target} quaternion={quat}>
          <ringGeometry args={[0.02, 0.026, 40]} />
          <meshBasicMaterial color={TERRA} transparent side={THREE.DoubleSide} depthWrite={false} />
        </mesh>
      </group>
      <mesh scale={1.06}>
        <sphereGeometry args={[1, 64, 64]} />
        <shaderMaterial vertexShader={rimVert} fragmentShader={rimFrag} uniforms={haloU} transparent depthWrite={false} side={THREE.BackSide} />
      </mesh>
      <Sat radius={1.32} tilt={[1.2, 0.32]} phase={0.6} speed={0.28} color={LAKE} swath earth={earth} readout={telemetry[0]} reduced={reduced} />
      <Sat radius={1.46} tilt={[1.04, -0.48]} phase={2.5} speed={0.22} color={TERRA} earth={earth} readout={telemetry[1]} reduced={reduced} />
    </group>
  );
}

export default function HeroGlobe(p: Props) {
  const host = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(true);
  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const io = new IntersectionObserver(([e]) => setVisible(e.isIntersecting));
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return (
    <div ref={host} className="lp-globe-canvas" aria-hidden="true">
      <Canvas dpr={[1, 1.5]} camera={{ position: [0, 0, 6.4], fov: 40 }} frameloop={visible && !p.reduced ? "always" : "demand"}
        gl={{ antialias: true, alpha: true }}>
        <ambientLight intensity={1.1} />
        <directionalLight position={[-3, 2.5, 4]} intensity={1.8} color="#fffaf0" />
        <directionalLight position={[4, -2, -2]} intensity={0.3} color="#f0c08a" />
        <Globe {...p} />
      </Canvas>
    </div>
  );
}
