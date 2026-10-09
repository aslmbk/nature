/**
 * The orb (frames 08 / 09, details 04): everything here lives in orb units (radius 1)
 * under a group scaled to the world radius.
 *
 *  - shell: slightly irregular sphere drawn twice — back faces first as the dark
 *    green body (replaces what is behind the orb), front faces last as a thin glossy
 *    film (fresnel rim, faint specular, granular mottling, low opacity in the middle
 *    so the interior stays visible). Premultiplied alpha, no transmission.
 *  - nodes: billboards in one draw — clear bubbles with rims, lime "berries", small
 *    teal / green glowing nodes, the bright core of the streams episode; pulses with
 *    their own phases.
 *  - threads: faint pale 1 px lines radiating from the nodes (additive).
 *  - filament: a dark twisted cable of thin tubes (an S / J curve) between the lights.
 *
 * Interior parts rotate with `uSpin` (computed in the shaders, no object rotation).
 *
 * The shell and the filament are step generators (vegetation/slices.ts): checkpoints
 * between vertex batches / tubes, the same rng draws in the same order.
 */
import {
  BackSide,
  BufferAttribute,
  BufferGeometry,
  CatmullRomCurve3,
  Color,
  CustomBlending,
  FrontSide,
  LineSegments,
  Mesh,
  OneFactor,
  OneMinusSrcAlphaFactor,
  ShaderMaterial,
  SphereGeometry,
  TubeGeometry,
  Vector3,
  type IUniform,
} from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import type { Rng } from "../../core/rng";
import type { Steps } from "../../vegetation/slices";
import { NOISE_GLSL, OUTPUT_TAIL } from "./glsl";
import { additiveKeepAlpha } from "./layer";
import { makeNoise3, type Palette } from "./particles";

// ---------------------------------------------------------------------------
// Node layout (orb units; x right, y up, z towards the camera) — frame 08
// ---------------------------------------------------------------------------

export const NODE_GLOW = 0;
export const NODE_BUBBLE = 1;
export const NODE_BERRY = 2;
export const NODE_CORE = 3;

export interface OrbNode {
  type: number;
  position: Vector3;
  radius: number;
  color: Color;
  brightness: number;
  /** Bubbles: strength of a glowing core inside. */
  inner: number;
}

export function layoutNodes(rng: Rng, palette: Palette): OrbNode[] {
  const j = (v: number) => v + rng.range(-0.05, 0.05);
  const node = (type: number, x: number, y: number, z: number, radius: number, color: Color, brightness: number, inner = 0): OrbNode => ({
    type,
    position: new Vector3(j(x), j(y), j(z)),
    radius: radius * rng.range(0.9, 1.1),
    color: color.clone(),
    brightness,
    inner,
  });
  return [
    node(NODE_BUBBLE, -0.23, 0.19, 0.52, 0.11, palette.paleTeal, 1, 0.9),
    node(NODE_BUBBLE, -0.48, -0.08, 0.45, 0.085, palette.paleLime, 0.8, 0.15),
    node(NODE_BUBBLE, -0.54, -0.37, 0.35, 0.065, palette.paleLime, 0.7),
    node(NODE_BUBBLE, 0.5, -0.3, 0.32, 0.055, palette.paleLime, 0.6),
    node(NODE_BUBBLE, 0.08, -0.5, -0.35, 0.07, palette.paleTeal, 0.5),
    node(NODE_BERRY, 0.27, -0.09, 0.5, 0.12, palette.lime, 1.25),
    node(NODE_BERRY, 0.24, -0.66, 0.42, 0.1, palette.lime, 0.95),
    node(NODE_BERRY, -0.12, 0.4, -0.3, 0.075, palette.lime, 0.6),
    node(NODE_GLOW, -0.5, 0.55, 0.3, 0.038, palette.teal, 1.6),
    node(NODE_GLOW, 0.38, 0.54, 0.35, 0.036, palette.teal, 1.5),
    node(NODE_GLOW, 0.25, 0.64, 0.2, 0.03, palette.green, 1.4),
    node(NODE_GLOW, 0.06, 0.06, 0.62, 0.032, palette.teal, 1.3),
    node(NODE_GLOW, -0.16, -0.3, 0.56, 0.03, palette.green, 1.3),
    node(NODE_GLOW, -0.3, 0.62, -0.2, 0.028, palette.lime, 1.0),
    // the yellow-lime heart that lights up in the streams episode
    node(NODE_CORE, 0.05, 0.02, 0.06, 0.085, new Color(0.6, 1.0, 0.1), 1.0),
  ];
}

// ---------------------------------------------------------------------------
// Shell
// ---------------------------------------------------------------------------

const SHELL_VERTEX = /* glsl */ `
varying vec3 vN;
varying vec3 vV;
varying vec3 vObj;
void main() {
  vObj = position;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vN = normalize(normalMatrix * normal);
  vV = -mv.xyz;
  gl_Position = projectionMatrix * mv;
}
`;

const SHELL_FRAGMENT = /* glsl */ `
uniform vec3 uBody;
uniform vec3 uFleck;
uniform vec3 uSpeck;
uniform vec3 uRim;
uniform vec3 uSpec;
uniform vec3 uLight;     // view space, towards the light
uniform float uOpacity;
uniform float uBack;
uniform float uRot;
uniform float uExit;
varying vec3 vN;
varying vec3 vV;
varying vec3 vObj;
${NOISE_GLSL}
void main() {
  vec3 n = normalize(vN);
  vec3 v = normalize(vV);
  if (uBack > 0.5) n = -n;
  // the blend layer is single-sample: fade the last pixel before the silhouette
  // (computed before any discard so the derivatives stay defined)
  float ndvRaw = dot(n, v);
  float silhouetteAA = clamp(ndvRaw / max(0.8 * fwidth(ndvRaw), 1e-4), 0.0, 1.0);
  float ndv = clamp(ndvRaw, 0.0, 1.0);
  vec3 q = oRotY(uRot) * vObj;
  float mottle = oNoise(q * 3.1) * 0.6 + oNoise(q * 7.3 + 4.0) * 0.4;
  float grain = oNoise(q * 34.0) * 0.55 + oNoise(q * 83.0) * 0.45;
  // exit: the shell breaks up from the bottom
  float holes = oNoise(q * 5.0 + 9.0) * 0.7 + grain * 0.3;
  float cut = uExit * (1.15 - 0.6 * (q.y * 0.5 + 0.5));
  if (holes < cut) discard;
  // lichen-like texture: soft blotches, fine grain, a few light flecks
  float fleck = smoothstep(0.7, 0.92, oNoise(q * 61.0 + 3.0)) * smoothstep(0.35, 0.75, mottle);
  float tex = (0.25 + 1.15 * mottle) * (0.4 + 1.1 * grain);
  vec4 outc;
  if (uBack > 0.5) {
    float a = 0.88;
    vec3 col = uBody * tex + uFleck * fleck * 0.8;
    outc = vec4(col * a, a);
  } else {
    float edge = 1.0 - ndv;
    // crust: dark lichen patches over a thin film, denser towards the edge and on the
    // left (frames 08 / 09: the interior shows mainly through the right half)
    float crust = smoothstep(0.38, 0.7, mottle + 0.16 * (-n.x) + 0.3 * edge * edge);
    float a = mix(0.07, 0.86, crust) * (0.8 + 0.4 * grain);
    a = max(a, 0.6 * pow(edge, 2.2));
    vec3 col = uBody * tex * 1.1 + uFleck * fleck;
    // fine mineral specks on the crust, brightest near the edge
    float speck = smoothstep(0.8, 0.96, oNoise(q * 74.0 + 1.7)) * crust * (0.3 + 1.2 * edge * edge);
    col += uSpeck * speck;
    vec3 l = normalize(uLight);
    vec3 h = normalize(l + v);
    float spec = pow(max(dot(n, h), 0.0), 120.0);
    float sheen = pow(max(dot(n, h), 0.0), 12.0) * 0.05;
    // thin rim, mostly on the lower right
    vec2 nd = normalize(n.xy + vec2(1e-4));
    float side = 0.2 + 0.8 * smoothstep(-0.3, 0.9, dot(nd, normalize(vec2(0.75, -0.55))));
    vec3 em = uRim * pow(edge, 7.0) * side * (0.5 + 0.9 * grain) + uSpec * (spec * 0.5 + sheen);
    outc = vec4(col * a + em, a);
  }
  // the shell forms in patches (crust first), not as a uniform fade (clip 08, 22.4 s)
  float form = smoothstep(1.0 - 1.3 * uOpacity, 1.15 - 1.3 * uOpacity, 0.65 * mottle + 0.35 * grain);
  gl_FragColor = outc * form * silhouetteAA;
  ${OUTPUT_TAIL}
}
`;

export interface Shell {
  back: Mesh<BufferGeometry, ShaderMaterial>;
  front: Mesh<BufferGeometry, ShaderMaterial>;
  uniforms: {
    uOpacity: IUniform<number>;
    uRot: IUniform<number>;
    uExit: IUniform<number>;
    uLight: IUniform<Vector3>;
  };
}

function premultipliedMaterial(m: ShaderMaterial): ShaderMaterial {
  m.transparent = true;
  m.depthWrite = false;
  m.blending = CustomBlending;
  m.blendSrc = OneFactor;
  m.blendDst = OneMinusSrcAlphaFactor;
  m.blendSrcAlpha = OneFactor;
  m.blendDstAlpha = OneMinusSrcAlphaFactor;
  return m;
}

export function* buildShell(rng: Rng): Steps<Shell> {
  const noise = makeNoise3(rng);
  const geometry = new SphereGeometry(1, 112, 72);
  yield;
  const pos = geometry.getAttribute("position") as BufferAttribute;
  const v = new Vector3();
  for (let i = 0; i < pos.count; i++) {
    if ((i & 1023) === 1023) yield;
    v.fromBufferAttribute(pos, i);
    const d = (noise(v.x * 1.3 + 5, v.y * 1.3, v.z * 1.3) - 0.5) * 0.05 + (noise(v.x * 3.1, v.y * 3.1 + 2, v.z * 3.1) - 0.5) * 0.014;
    v.multiplyScalar(1 + d);
    pos.setXYZ(i, v.x, v.y, v.z);
  }
  yield;
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();

  const uniforms = {
    uOpacity: { value: 1 },
    uRot: { value: 0 },
    uExit: { value: 0 },
    uLight: { value: new Vector3(-0.45, 0.6, 0.66) },
  };
  const make = (back: boolean) =>
    premultipliedMaterial(
      new ShaderMaterial({
        name: back ? "OracleShellBack" : "OracleShellFront",
        uniforms: {
          ...uniforms,
          uBody: { value: new Color(0.021, 0.038, 0.026) },
          uFleck: { value: new Color(0.035, 0.06, 0.045) },
          uSpeck: { value: new Color(0.22, 0.3, 0.23) },
          uRim: { value: new Color(0.1, 0.15, 0.11) },
          uSpec: { value: new Color(0.35, 0.45, 0.38) },
          uBack: { value: back ? 1 : 0 },
        },
        vertexShader: SHELL_VERTEX,
        fragmentShader: SHELL_FRAGMENT,
        side: back ? BackSide : FrontSide,
      }),
    );
  const back = new Mesh(geometry, make(true));
  back.name = "oracle_shell_back";
  const front = new Mesh(geometry, make(false));
  front.name = "oracle_shell_front";
  return { back, front, uniforms };
}

// ---------------------------------------------------------------------------
// Nodes
// ---------------------------------------------------------------------------

const NODE_VERTEX = /* glsl */ `
attribute vec3 aCenter;
attribute vec2 aCorner;
attribute vec4 aNode;    // x: radius, y: type, z: phase, w: brightness
attribute vec4 aColor;   // rgb, a: inner glow (bubbles)
uniform float uSpin;
uniform float uTime;
uniform float uGrow;
uniform float uCore;
uniform float uExit;
varying vec2 vUv;
varying vec3 vColor;
varying float vType;
varying float vInner;
varying float vSeed;
varying float vGrow;
varying float vExtent;
${NOISE_GLSL}
void main() {
  vec3 c = oRotY(uSpin) * aCenter;
  c += (normalize(c + vec3(1e-4)) * 0.2 + vec3(0.0, -0.3, 0.0)) * uExit;
  float type = aNode.y;
  float extent = type < 0.5 ? 3.2 : (type < 1.5 ? 1.12 : (type < 2.5 ? 1.9 : 6.0));
  float pulse = 0.72 + 0.28 * sin(uTime * (0.55 + 0.9 * aNode.z) + aNode.z * 40.0);
  float b = aNode.w * pulse;
  if (type > 2.5) b *= uCore;
  vec4 mv = modelViewMatrix * vec4(c, 1.0);
  float scale = length(modelViewMatrix[0].xyz);
  mv.xy += aCorner * aNode.x * extent * scale * uGrow;
  gl_Position = projectionMatrix * mv;
  vUv = aCorner * extent;
  vColor = aColor.rgb * b * uGrow;
  vType = type;
  vInner = aColor.a;
  vSeed = aNode.z;
  vGrow = uGrow;
  vExtent = extent;
}
`;

const NODE_FRAGMENT = /* glsl */ `
varying vec2 vUv;
varying vec3 vColor;
varying float vType;
varying float vInner;
varying float vSeed;
varying float vGrow;
varying float vExtent;
${NOISE_GLSL}
void main() {
  float r = length(vUv);
  // every glow ends inside its quad (no square edges)
  float window = 1.0 - smoothstep(max(1.1, 0.7 * vExtent), vExtent, r);
  vec3 col;
  float alpha = 0.0;
  if (vType < 0.5) {
    // small glowing node: lit sphere + soft halo
    float nz = sqrt(max(0.0, 1.0 - r * r));
    float sphere = (1.0 - smoothstep(0.85, 1.0, r)) * (0.55 + 0.45 * nz);
    float halo = exp(-r * r * 0.55) * 0.32 + exp(-r * r * 3.0) * 0.4;
    col = vColor * (sphere * 1.6 + halo);
  } else if (vType < 1.5) {
    // clear bubble: thin bright rim, faint fill, a specular dot, optional glowing core
    float rim = smoothstep(0.8, 0.96, r) * (1.0 - smoothstep(0.96, 1.08, r));
    float fill = (1.0 - smoothstep(0.85, 1.0, r)) * 0.05;
    vec2 sp = vUv - vec2(-0.36, 0.4);
    float spec = exp(-dot(sp, sp) * 70.0) * 1.4 + exp(-dot(sp, sp) * 9.0) * 0.12;
    vec2 sp2 = vUv - vec2(0.42, -0.5);
    float spec2 = exp(-dot(sp2, sp2) * 120.0) * 0.4;
    float core = exp(-r * r * 9.0) * vInner * 1.1 + exp(-r * r * 3.0) * vInner * 0.12;
    col = vColor * (rim * 0.9 + fill + core) + vec3(0.75, 0.95, 0.85) * (spec + spec2) * length(vColor) * 0.4;
    // glass bead: dims what is behind it a little
    alpha = (1.0 - smoothstep(0.9, 1.0, r)) * 0.38 * vGrow;
  } else if (vType < 2.5) {
    // lime berry: glowing sphere with a cellular pattern and a soft halo
    float nz = sqrt(max(0.0, 1.0 - r * r));
    vec3 sp = vec3(vUv, nz) * 4.2 + vSeed * 17.0;
    float cells = oNoise(sp) * 0.65 + oNoise(sp * 2.3) * 0.35;
    float dots = smoothstep(0.58, 0.8, cells);
    float disc = 1.0 - smoothstep(0.9, 1.0, r);
    float body = disc * ((0.3 + 0.55 * nz) + dots * 2.0 * nz);
    float edge = disc * smoothstep(0.55, 0.95, r) * 0.45;
    float halo = (1.0 - disc) * exp(-(r - 0.9) * (r - 0.9) * 4.0) * 0.3;
    float hot = exp(-r * r * 9.0) * 1.3;
    col = vColor * (body + edge + halo + hot);
  } else {
    // heart of the streams episode: bright centre and a soft glow
    col = vColor * (exp(-r * r * 2.2) * 1.6 + exp(-r * r * 0.25) * 0.3 + exp(-r * r * 0.06) * 0.06);
  }
  gl_FragColor = vec4(col * window, alpha * window);
  ${OUTPUT_TAIL}
}
`;

export interface Nodes {
  mesh: Mesh<BufferGeometry, ShaderMaterial>;
  uniforms: {
    uSpin: IUniform<number>;
    uTime: IUniform<number>;
    uGrow: IUniform<number>;
    uCore: IUniform<number>;
    uExit: IUniform<number>;
  };
}

export function buildNodes(nodes: OrbNode[], rng: Rng): Nodes {
  const n = nodes.length;
  const center = new Float32Array(n * 4 * 3);
  const corner = new Float32Array(n * 4 * 2);
  const data = new Float32Array(n * 4 * 4);
  const color = new Float32Array(n * 4 * 4);
  const index: number[] = [];
  const corners = [
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, 1],
  ];
  nodes.forEach((nd, i) => {
    const phase = rng.next();
    for (let k = 0; k < 4; k++) {
      const vi = i * 4 + k;
      center.set([nd.position.x, nd.position.y, nd.position.z], vi * 3);
      corner.set(corners[k], vi * 2);
      data.set([nd.radius, nd.type, phase, nd.brightness], vi * 4);
      color.set([nd.color.r, nd.color.g, nd.color.b, nd.inner], vi * 4);
    }
    const b = i * 4;
    index.push(b, b + 1, b + 2, b, b + 2, b + 3);
  });
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new BufferAttribute(center.slice(), 3));
  geometry.setAttribute("aCenter", new BufferAttribute(center, 3));
  geometry.setAttribute("aCorner", new BufferAttribute(corner, 2));
  geometry.setAttribute("aNode", new BufferAttribute(data, 4));
  geometry.setAttribute("aColor", new BufferAttribute(color, 4));
  geometry.setIndex(index);
  const uniforms = {
    uSpin: { value: 0 },
    uTime: { value: 0 },
    uGrow: { value: 1 },
    uCore: { value: 0 },
    uExit: { value: 0 },
  };
  const material = premultipliedMaterial(
    new ShaderMaterial({
      name: "OracleNodes",
      uniforms,
      vertexShader: NODE_VERTEX,
      fragmentShader: NODE_FRAGMENT,
      depthTest: true,
    }),
  );
  const mesh = new Mesh(geometry, material);
  mesh.name = "oracle_nodes";
  mesh.frustumCulled = false;
  return { mesh, uniforms };
}

// ---------------------------------------------------------------------------
// Threads (faint 1 px lines)
// ---------------------------------------------------------------------------

const THREAD_VERTEX = /* glsl */ `
attribute float aFade;
uniform float uSpin;
uniform float uExit;
varying float vFade;
${NOISE_GLSL}
void main() {
  vec3 p = oRotY(uSpin) * position;
  p += (normalize(p + vec3(1e-4)) * 0.2 + vec3(0.0, -0.3, 0.0)) * uExit;
  vFade = aFade;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
}
`;

const THREAD_FRAGMENT = /* glsl */ `
uniform vec3 uColor;
uniform float uOpacity;
varying float vFade;
void main() {
  gl_FragColor = vec4(uColor * vFade * uOpacity, 1.0);
  ${OUTPUT_TAIL}
}
`;

export interface Threads {
  lines: LineSegments<BufferGeometry, ShaderMaterial>;
  uniforms: { uSpin: IUniform<number>; uOpacity: IUniform<number>; uExit: IUniform<number> };
}

export function buildThreads(nodes: OrbNode[], rng: Rng): Threads {
  const pos: number[] = [];
  const fade: number[] = [];
  const dir = new Vector3();
  const a = new Vector3();
  const b = new Vector3();
  const segment = (from: Vector3, to: Vector3, f0: number, f1: number) => {
    // split into a few pieces so the fade reads along the line
    const steps = 4;
    for (let s = 0; s < steps; s++) {
      const k0 = s / steps;
      const k1 = (s + 1) / steps;
      a.copy(from).lerp(to, k0);
      b.copy(from).lerp(to, k1);
      pos.push(a.x, a.y, a.z, b.x, b.y, b.z);
      fade.push(f0 + (f1 - f0) * k0, f0 + (f1 - f0) * k1);
    }
  };
  // short fine fibres around the berries and the core (frame 08), a few on the glow nodes
  for (const nd of nodes) {
    const count = nd.type === NODE_CORE ? 26 : nd.type === NODE_BERRY ? 14 : nd.type === NODE_GLOW ? 3 : 0;
    for (let i = 0; i < count; i++) {
      const z = rng.range(-1, 1);
      const ang = rng.range(0, Math.PI * 2);
      const s = Math.sqrt(1 - z * z);
      dir.set(s * Math.cos(ang), z, s * Math.sin(ang));
      const len = nd.type === NODE_CORE ? rng.range(0.1, 0.32) : nd.type === NODE_BERRY ? rng.range(0.03, 0.12) : rng.range(0.03, 0.08);
      const from = nd.position.clone().addScaledVector(dir, nd.radius * 0.9);
      const to = nd.position.clone().addScaledVector(dir, nd.radius + len);
      if (to.length() > 0.95) to.setLength(0.95);
      segment(from, to, nd.type === NODE_CORE ? 0.6 : nd.type === NODE_BERRY ? 0.8 : 0.4, 0);
    }
  }
  // a few long faint links between nodes
  for (let i = 0; i < 5; i++) {
    const n0 = rng.pick(nodes);
    const n1 = rng.pick(nodes);
    if (n0 === n1) continue;
    segment(n0.position, n1.position, 0.18, 0.18);
  }
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new BufferAttribute(Float32Array.from(pos), 3));
  geometry.setAttribute("aFade", new BufferAttribute(Float32Array.from(fade), 1));
  const uniforms = { uSpin: { value: 0 }, uOpacity: { value: 1 }, uExit: { value: 0 } };
  const material = additiveKeepAlpha(
    new ShaderMaterial({
      name: "OracleThreads",
      uniforms: { ...uniforms, uColor: { value: new Color(0.13, 0.22, 0.12) } },
      vertexShader: THREAD_VERTEX,
      fragmentShader: THREAD_FRAGMENT,
      transparent: true,
      depthWrite: false,
      depthTest: true,
    }),
  );
  const lines = new LineSegments(geometry, material);
  lines.name = "oracle_threads";
  lines.frustumCulled = false;
  return { lines, uniforms };
}

// ---------------------------------------------------------------------------
// Dark filament (twisted cable of thin tubes)
// ---------------------------------------------------------------------------

const FILAMENT_VERTEX = /* glsl */ `
uniform float uSpin;
uniform float uExit;
varying vec3 vN;
varying vec3 vV;
varying float vAlong;
${NOISE_GLSL}
void main() {
  mat3 r = oRotY(uSpin);
  vec3 p = r * position;
  p += (normalize(p + vec3(1e-4)) * 0.2 + vec3(0.0, -0.3, 0.0)) * uExit;
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  vN = normalize(normalMatrix * (r * normal));
  vV = -mv.xyz;
  vAlong = uv.x;
  gl_Position = projectionMatrix * mv;
}
`;

const FILAMENT_FRAGMENT = /* glsl */ `
uniform float uGrow;
uniform vec3 uLight;
varying vec3 vN;
varying vec3 vV;
varying float vAlong;
void main() {
  if (vAlong > uGrow) discard;
  vec3 n = normalize(vN);
  vec3 v = normalize(vV);
  vec3 h = normalize(normalize(uLight) + v);
  float spec = pow(max(dot(n, h), 0.0), 30.0);
  float rim = pow(1.0 - abs(dot(n, v)), 3.0);
  vec3 col = vec3(0.003, 0.007, 0.005) + vec3(0.1, 0.16, 0.12) * spec + vec3(0.03, 0.05, 0.04) * rim;
  gl_FragColor = vec4(col, 1.0);
  ${OUTPUT_TAIL}
}
`;

export interface Filament {
  mesh: Mesh<BufferGeometry, ShaderMaterial>;
  uniforms: { uSpin: IUniform<number>; uGrow: IUniform<number>; uExit: IUniform<number>; uLight: IUniform<Vector3> };
}

export function* buildFilament(rng: Rng): Steps<Filament> {
  const j = (v: number) => v + rng.range(-0.03, 0.03);
  const paths: Vector3[][] = [
    // the main S / J cable (frame 08: from the upper middle, right, down, back to the lower left)
    [
      new Vector3(j(-0.18), j(0.64), 0.26),
      new Vector3(j(0.08), j(0.55), 0.38),
      new Vector3(j(0.3), j(0.3), 0.43),
      new Vector3(j(0.4), j(0.02), 0.4),
      new Vector3(j(0.3), j(-0.24), 0.42),
      new Vector3(j(0.06), j(-0.42), 0.45),
      new Vector3(j(-0.22), j(-0.5), 0.36),
    ],
    // a short curl at the back
    [new Vector3(-0.45, 0.2, -0.3), new Vector3(-0.3, 0.32, -0.42), new Vector3(-0.1, 0.25, -0.5), new Vector3(0.05, 0.05, -0.45)],
  ];
  const parts: BufferGeometry[] = [];
  for (let pi = 0; pi < paths.length; pi++) {
    const pts = paths[pi];
    const centre = new CatmullRomCurve3(pts, false, "centripetal");
    const samples = 90;
    const frames = centre.computeFrenetFrames(samples, false);
    const strands = pi === 0 ? 6 : 3;
    const twists = pi === 0 ? 3.0 : 1.5;
    const spread = pi === 0 ? 0.045 : 0.022;
    for (let s = 0; s < strands; s++) {
      yield;
      const off = (s / strands) * Math.PI * 2;
      const r = spread * (0.75 + 0.5 * ((s * 7) % 3) / 2);
      const strandPts: Vector3[] = [];
      for (let i = 0; i <= samples; i++) {
        const u = i / samples;
        const th = u * twists * Math.PI * 2 + off;
        const c = centre.getPointAt(u);
        c.addScaledVector(frames.normals[i], Math.cos(th) * r).addScaledVector(frames.binormals[i], Math.sin(th) * r);
        strandPts.push(c);
      }
      const curve = new CatmullRomCurve3(strandPts, false, "centripetal");
      const radius = pi === 0 ? 0.0075 : 0.005;
      parts.push(new TubeGeometry(curve, 160, radius, 5, false));
    }
  }
  yield;
  const geometry = mergeGeometries(parts, false) ?? parts[0];
  for (const p of parts) if (p !== geometry) p.dispose();
  const uniforms = {
    uSpin: { value: 0 },
    uGrow: { value: 1 },
    uExit: { value: 0 },
    uLight: { value: new Vector3(-0.45, 0.6, 0.66) },
  };
  const material = new ShaderMaterial({
    name: "OracleFilament",
    uniforms,
    vertexShader: FILAMENT_VERTEX,
    fragmentShader: FILAMENT_FRAGMENT,
  });
  const mesh = new Mesh(geometry, material);
  mesh.name = "oracle_filament";
  mesh.frustumCulled = false;
  return { mesh, uniforms };
}
