// Exploded, draggable 3D stack of the actual analysis layers for one demo scene.
import { Canvas, useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

export interface StackLayer {
  key: string;
  title: string;
  sub: string;
  png?: string;            // data URI from the backend; procedural texture when missing
  tint: string;
  overlay?: boolean;       // transparent mask that floats above its base
}

function proceduralTexture(tint: string, seed: number): THREE.Texture {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d")!;
  g.fillStyle = "#ECE6DA";
  g.fillRect(0, 0, 128, 128);
  g.strokeStyle = tint;
  g.globalAlpha = 0.35;
  for (let i = 0; i <= 128; i += 16) {
    g.beginPath(); g.moveTo(i, 0); g.lineTo(i, 128); g.stroke();
    g.beginPath(); g.moveTo(0, i); g.lineTo(128, i); g.stroke();
  }
  g.globalAlpha = 0.8;
  g.fillStyle = tint;
  let s = seed * 9301 + 49297;
  for (let i = 0; i < 6; i++) {
    s = (s * 9301 + 49297) % 233280;
    const x = (s / 233280) * 128;
    s = (s * 9301 + 49297) % 233280;
    g.beginPath();
    g.arc(x, (s / 233280) * 128, 6 + (i % 3) * 6, 0, Math.PI * 2);
    g.fill();
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function Plane({ layer, index, count, hovered, explode, onHover }: {
  layer: StackLayer; index: number; count: number; hovered: number | null; explode: number;
  onHover: (i: number | null) => void;
}) {
  const group = useRef<THREE.Group>(null);
  const mat = useRef<THREE.MeshBasicMaterial>(null);
  const tex = useMemo(() => {
    let t: THREE.Texture;
    if (layer.png) {
      t = new THREE.TextureLoader().load(layer.png);
      t.colorSpace = THREE.SRGBColorSpace;
    } else t = proceduralTexture(layer.tint, index + 1);
    t.magFilter = THREE.NearestFilter;
    t.minFilter = THREE.LinearMipMapLinearFilter;
    return t;
  }, [layer.png, layer.tint, index]);
  useEffect(() => () => tex.dispose(), [tex]);
  const edges = useMemo(() => new THREE.EdgesGeometry(new THREE.PlaneGeometry(2, 2)), []);
  useFrame((_, dt) => {
    const g = group.current;
    if (!g) return;
    const gap = 0.32 + 0.62 * explode;
    const lift = hovered === index ? 0.28 : 0;
    const target = (index - (count - 1) / 2) * gap + lift;
    g.position.y += (target - g.position.y) * Math.min(1, dt * 8);
    if (mat.current) {
      const dim = hovered !== null && hovered !== index ? 0.35 : 1;
      mat.current.opacity += ((layer.overlay ? 0.95 : 1) * dim - mat.current.opacity) * Math.min(1, dt * 8);
    }
  });
  return (
    <group ref={group}>
      <mesh rotation={[-Math.PI / 2, 0, 0]} onPointerOver={(e) => { e.stopPropagation(); onHover(index); }}
        onPointerOut={() => onHover(null)}>
        <planeGeometry args={[2, 2]} />
        <meshBasicMaterial ref={mat} map={tex} transparent side={THREE.DoubleSide} depthWrite={!layer.overlay} />
      </mesh>
      <lineSegments geometry={edges} rotation={[-Math.PI / 2, 0, 0]}>
        <lineBasicMaterial color={hovered === index ? "#D2651F" : "#2A2D1E"} transparent opacity={hovered === index ? 1 : 0.55} />
      </lineSegments>
    </group>
  );
}

function Rig({ children, reduced }: { children: React.ReactNode; reduced: boolean }) {
  const ref = useRef<THREE.Group>(null);
  const drag = useRef({ active: false, x: 0, vel: 0, yaw: -0.6 });
  useFrame(({ pointer }, dt) => {
    const g = ref.current;
    if (!g) return;
    const d = drag.current;
    if (!d.active && !reduced) d.yaw += dt * 0.12;
    d.yaw += d.vel;
    d.vel *= 0.92;
    g.rotation.y = d.yaw;
    g.rotation.x += ((reduced ? 0 : pointer.y * 0.08) - g.rotation.x) * 0.05;
  });
  return (
    <group ref={ref}
      onPointerDown={(e) => { drag.current.active = true; drag.current.x = e.clientX; (e.target as Element).setPointerCapture?.(e.pointerId); }}
      onPointerMove={(e) => { if (drag.current.active) { drag.current.vel = (e.clientX - drag.current.x) * 0.004; drag.current.x = e.clientX; } }}
      onPointerUp={() => { drag.current.active = false; }}>
      {children}
    </group>
  );
}

export default function LayerStack({ layers, hovered, onHover, reduced }: {
  layers: StackLayer[]; hovered: number | null; onHover: (i: number | null) => void; reduced: boolean;
}) {
  const host = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  const [explode, setExplode] = useState(0);
  // only render while on screen; explode the stack as the section scrolls into view
  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const io = new IntersectionObserver(([e]) => setVisible(e.isIntersecting), { rootMargin: "100px" });
    io.observe(el);
    const onScroll = () => {
      const r = el.getBoundingClientRect();
      const p = 1 - Math.min(1, Math.max(0, (r.top + r.height * 0.3) / window.innerHeight));
      setExplode(reduced ? 1 : Math.min(1, p * 1.6));
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => { io.disconnect(); window.removeEventListener("scroll", onScroll); };
  }, [reduced]);
  return (
    <div ref={host} className="lp-stack" role="img"
      aria-label={`3D exploded view of ${layers.length} analysis layers: ${layers.map((l) => l.title).join(", ")}. Drag to rotate.`}>
      <Canvas dpr={[1, 1.25]} camera={{ position: [3.9, 2.7, 3.9], fov: 38 }} frameloop={visible ? "always" : "never"}
        gl={{ antialias: true, alpha: true }} onCreated={({ camera }) => camera.lookAt(0, -0.2, 0)}>
        <Rig reduced={reduced}>
          {layers.map((l, i) => (
            <Plane key={l.key + (l.png ? "1" : "0")} layer={l} index={i} count={layers.length} hovered={hovered}
              explode={explode} onHover={onHover} />
          ))}
          <gridHelper args={[4, 16, "#bdb48f", "#e4dcc6"]} position={[0, -1.4, 0]} />
        </Rig>
      </Canvas>
      <span className="lp-stack-hint">Drag to rotate · hover a layer</span>
    </div>
  );
}
