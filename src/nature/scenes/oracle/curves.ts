/**
 * Thin curves of the oracle scene set (orb units, radius 1):
 *
 *  - orbit: a thin tube on a circle (tilted by its parent group), drawn on during the
 *    assembly, with a softly lit segment; opaque, never a GL line.
 *  - tag: a small blank pill riding the orbit, camera-facing and aligned with the
 *    projected tangent (as in frame 08), opaque with an SDF outline.
 *  - loop: the long twisted figure-8 of the streams episode (frame 09). One closed
 *    curve whose two sides are the two strands; a flat ribbon cross-section twisting
 *    along it, dark glassy with lime specular from a key light and from the orb.
 *    Revealed from its top down to the orb; the part inside the orb is cut away.
 *
 * `buildLoop` is a step generator (vegetation/slices.ts): checkpoints between batches of
 * ring segments, the same rng draws in the same order.
 */
import {
  BufferAttribute,
  BufferGeometry,
  CatmullRomCurve3,
  Color,
  Mesh,
  ShaderMaterial,
  TubeGeometry,
  Vector2,
  Vector3,
  type IUniform,
} from "three";
import type { Rng } from "../../core/rng";
import type { Steps } from "../../vegetation/slices";
import { OUTPUT_TAIL } from "./glsl";

// ---------------------------------------------------------------------------
// Orbit
// ---------------------------------------------------------------------------

const ORBIT_VERTEX = /* glsl */ `
varying vec3 vObj;
varying vec3 vN;
varying vec3 vV;
void main() {
  vObj = position;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vN = normalize(normalMatrix * normal);
  vV = -mv.xyz;
  gl_Position = projectionMatrix * mv;
}
`;

const ORBIT_FRAGMENT = /* glsl */ `
uniform float uDraw;
uniform float uStart;
uniform float uLit;
uniform vec3 uColor;
uniform vec3 uLitColor;
uniform vec3 uLight;
varying vec3 vObj;
varying vec3 vN;
varying vec3 vV;
void main() {
  float phi = atan(vObj.z, vObj.x);
  float u = mod(uStart - phi, 6.2831853) / 6.2831853;
  if (u > uDraw) discard;
  float d = abs(mod(phi - uLit + 3.14159265, 6.2831853) - 3.14159265);
  float lit = exp(-d * d * 1.6);
  float head = (1.0 - step(0.999, uDraw)) * exp(-pow((uDraw - u) * 24.0, 2.0));
  vec3 n = normalize(vN);
  vec3 v = normalize(vV);
  float shade = 0.65 + 0.35 * max(dot(n, normalize(uLight)), 0.0);
  float facing = 0.75 + 0.25 * abs(dot(n, v));
  vec3 col = uColor * shade * facing + uLitColor * (lit + head * 2.5);
  gl_FragColor = vec4(col, 1.0);
  ${OUTPUT_TAIL}
}
`;

export interface Orbit {
  mesh: Mesh<BufferGeometry, ShaderMaterial>;
  uniforms: { uDraw: IUniform<number>; uStart: IUniform<number>; uLit: IUniform<number>; uLight: IUniform<Vector3> };
  radius: number;
}

export function buildOrbit(radius: number, tubeRadius: number): Orbit {
  const pts = Array.from({ length: 96 }, (_, i) => {
    const a = (i / 96) * Math.PI * 2;
    return new Vector3(Math.cos(a) * radius, 0, Math.sin(a) * radius);
  });
  const curve = new CatmullRomCurve3(pts, true, "centripetal");
  const geometry = new TubeGeometry(curve, 512, tubeRadius, 6, true);
  const uniforms = {
    uDraw: { value: 1 },
    uStart: { value: 2.3 },
    uLit: { value: 0.2 },
    uLight: { value: new Vector3(-0.45, 0.6, 0.66) },
  };
  const material = new ShaderMaterial({
    name: "OracleOrbit",
    uniforms: {
      ...uniforms,
      uColor: { value: new Color(0.11, 0.13, 0.115) },
      uLitColor: { value: new Color(0.12, 0.17, 0.13) },
    },
    vertexShader: ORBIT_VERTEX,
    fragmentShader: ORBIT_FRAGMENT,
  });
  const mesh = new Mesh(geometry, material);
  mesh.name = "oracle_orbit";
  return { mesh, uniforms, radius };
}

// ---------------------------------------------------------------------------
// Tag (blank pill riding the orbit)
// ---------------------------------------------------------------------------

const TAG_VERTEX = /* glsl */ `
attribute vec2 aCorner;
uniform vec3 uCenter;    // orbit-local position on the ring
uniform vec3 uTangent;   // orbit-local tangent
uniform vec2 uHalf;      // half length / half height (orbit units)
varying vec2 vP;
void main() {
  vec4 c = modelViewMatrix * vec4(uCenter, 1.0);
  vec4 t = modelViewMatrix * vec4(uCenter + uTangent * 0.05, 1.0);
  vec2 sc = c.xy / max(-c.z, 1e-3);
  vec2 dir = t.xy / max(-t.z, 1e-3) - sc;
  dir = length(dir) > 1e-6 ? normalize(dir) : vec2(1.0, 0.0);
  vec2 nrm = vec2(-dir.y, dir.x);
  float scale = length(modelViewMatrix[0].xyz);
  // depth: just in front of every part of the ring that projects under the pill (near
  // the ends of the ellipse the ring runs almost along the view ray, so a fixed nudge
  // is not enough)
  float rr = length(uCenter.xz);
  float phi0 = atan(uCenter.z, uCenter.x);
  float reach = (uHalf.x + uHalf.y) * scale / max(-c.z, 1e-3);
  float zFront = c.z;
  for (int i = -16; i <= 16; i++) {
    float a = phi0 + float(i) * 0.05;
    vec4 q = modelViewMatrix * vec4(cos(a) * rr, 0.0, sin(a) * rr, 1.0);
    vec2 d = q.xy / max(-q.z, 1e-3) - sc;
    if (abs(dot(d, dir)) < reach && abs(dot(d, nrm)) < reach) zFront = max(zFront, q.z);
  }
  vec4 mv = c;
  mv.xy += (dir * aCorner.x * uHalf.x + nrm * aCorner.y * uHalf.y) * scale;
  // slide the corner along its own view ray: same picture, nearer depth
  mv.xyz *= (zFront + 0.02 * scale) / mv.z;
  vP = aCorner * uHalf;
  gl_Position = projectionMatrix * mv;
}
`;

const TAG_FRAGMENT = /* glsl */ `
uniform vec2 uHalf;
uniform float uVisible;
uniform vec3 uFill;
uniform vec3 uLine;
varying vec2 vP;
void main() {
  float r = uHalf.y;
  vec2 q = abs(vP) - vec2(uHalf.x - r, 0.0);
  float d = length(max(q, 0.0)) - r;       // < 0 inside the pill
  float px = fwidth(d);
  if (d > 0.0 || uVisible < 0.5) discard;
  float line = 1.0 - smoothstep(px * 1.0, px * 2.4, -d);
  // blank: no label, no glyph-like marks
  vec3 col = mix(uFill, uLine, line);
  gl_FragColor = vec4(col, 1.0);
  ${OUTPUT_TAIL}
}
`;

export interface Tag {
  mesh: Mesh<BufferGeometry, ShaderMaterial>;
  uniforms: { uCenter: IUniform<Vector3>; uTangent: IUniform<Vector3>; uVisible: IUniform<number> };
}

export function buildTag(halfLength: number, halfHeight: number): Tag {
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new BufferAttribute(new Float32Array(12), 3));
  geometry.setAttribute("aCorner", new BufferAttribute(Float32Array.from([-1, -1, 1, -1, 1, 1, -1, 1]), 2));
  geometry.setIndex([0, 1, 2, 0, 2, 3]);
  const uniforms = {
    uCenter: { value: new Vector3() },
    uTangent: { value: new Vector3(1, 0, 0) },
    uVisible: { value: 1 },
  };
  const material = new ShaderMaterial({
    name: "OracleTag",
    uniforms: {
      ...uniforms,
      uHalf: { value: new Vector2(halfLength, halfHeight) },
      uFill: { value: new Color(0.03, 0.034, 0.031) },
      uLine: { value: new Color(0.26, 0.29, 0.26) },
    },
    vertexShader: TAG_VERTEX,
    fragmentShader: TAG_FRAGMENT,
  });
  const mesh = new Mesh(geometry, material);
  mesh.name = "oracle_tag";
  mesh.frustumCulled = false;
  return { mesh, uniforms };
}

// ---------------------------------------------------------------------------
// Loop (twisted figure-8 ribbon)
// ---------------------------------------------------------------------------

/**
 * Closed control polygon (orb units, orb centre at the origin; frame 09 measured with
 * the orb radius = 160 px): lower lobe through / around the orb, crossing 2.55 above
 * the centre, upper lobe leaving the frame. The strand from the lower right to the
 * upper left passes behind at the crossing, the other one in front.
 */
const LOOP_POINTS: [number, number, number][] = [
  [0.0, -0.35, 0.15],
  [0.48, -0.12, 0.62],
  [0.82, 0.32, 0.58],
  [0.98, 0.82, 0.36],
  [0.9, 1.38, 0.18],
  [0.52, 2.02, 0.0],
  [0.0, 2.55, -0.16],
  [-0.55, 3.15, -0.06],
  [-0.92, 3.92, 0.14],
  [-1.02, 4.85, 0.34],
  [-0.86, 5.9, 0.5],
  [0.0, 6.75, 0.56],
  [0.86, 5.9, 0.5],
  [1.02, 4.85, 0.34],
  [0.92, 3.92, 0.24],
  [0.55, 3.15, 0.2],
  [0.0, 2.55, 0.17],
  [-0.52, 2.0, 0.0],
  [-0.9, 1.36, -0.2],
  [-0.98, 0.8, -0.42],
  [-0.8, 0.3, -0.62],
  [-0.42, -0.1, -0.6],
];

const LOOP_VERTEX = /* glsl */ `
attribute float aReveal;
varying vec3 vN;
varying vec3 vPosV;
varying vec3 vObj;
varying float vReveal;
void main() {
  vObj = position;
  vReveal = aReveal;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vPosV = mv.xyz;
  vN = normalize(normalMatrix * normal);
  gl_Position = projectionMatrix * mv;
}
`;

const LOOP_FRAGMENT = /* glsl */ `
uniform float uReveal;     // grows from the bottom of the loop (inside the orb) to its top
uniform float uLight;      // 0: a dark shape, 1: glassy highlights
uniform float uCut;        // object-space radius inside which the loop is hidden
uniform vec3 uKey;         // view space, towards the key light
uniform vec3 uBackLight;   // view space, towards the back light (top glow)
uniform vec3 uOrbPos;      // view space
uniform float uOrbRange;
uniform vec3 uBase;
uniform vec3 uSpecColor;
uniform vec3 uOrbColor;
uniform vec3 uRimColor;
varying vec3 vN;
varying vec3 vPosV;
varying vec3 vObj;
varying float vReveal;
void main() {
  float grown = 1.0 - vReveal;
  if (grown > uReveal) discard;
  if (length(vObj) < uCut) discard;
  vec3 n = normalize(vN);
  vec3 v = normalize(-vPosV);
  if (dot(n, v) < 0.0) n = -n;
  vec3 l = normalize(uKey);
  float diff = max(dot(n, l), 0.0);
  float spec = pow(max(dot(n, normalize(l + v)), 0.0), 90.0);
  vec3 lb = normalize(uBackLight);
  float back = pow(max(dot(n, normalize(lb + v)), 0.0), 24.0);
  vec3 lo = uOrbPos - vPosV;
  float dist = length(lo);
  lo /= max(dist, 1e-4);
  float fall = 1.0 / (1.0 + pow(dist / uOrbRange, 2.0));
  float specO = pow(max(dot(n, normalize(lo + v)), 0.0), 36.0) * fall;
  float diffO = max(dot(n, lo), 0.0) * fall;
  float rim = pow(1.0 - abs(dot(n, v)), 3.0);
  // dark glass: broad lighter edges, stronger high up towards the glow at the top
  float glass = pow(1.0 - abs(dot(n, v)), 1.6) * (0.35 + 0.65 * smoothstep(1.0, 6.0, vObj.y));
  vec3 col = uBase * (0.35 + 0.65 * diff)
    + (uSpecColor * spec * 2.2
    + uRimColor * (rim * 0.35 + back * 0.6 + glass * 0.55)
    + uOrbColor * (specO * 2.6 + diffO * 0.22)) * mix(0.05, 1.0, uLight);
  // a faint lit tip while it grows
  float head = (1.0 - step(0.999, uReveal)) * exp(-pow((uReveal - grown) * 22.0, 2.0));
  col += uOrbColor * head * 0.8;
  gl_FragColor = vec4(col, 1.0);
  ${OUTPUT_TAIL}
}
`;

export interface Loop {
  mesh: Mesh<BufferGeometry, ShaderMaterial>;
  uniforms: {
    uReveal: IUniform<number>;
    uLight: IUniform<number>;
    uCut: IUniform<number>;
    uKey: IUniform<Vector3>;
    uBackLight: IUniform<Vector3>;
    uOrbPos: IUniform<Vector3>;
    uOrbRange: IUniform<number>;
  };
}

export interface LoopOptions {
  /** Ribbon half width / half thickness (orb units). */
  halfWidth: number;
  halfThickness: number;
  /** Full turns of the ribbon around its own axis over the whole loop. */
  twists: number;
  segments: number;
  radial: number;
}

export function* buildLoop(rng: Rng, opts: LoopOptions): Steps<Loop> {
  const pts = LOOP_POINTS.map(([x, y, z]) => new Vector3(x + rng.range(-0.03, 0.03), y, z + rng.range(-0.03, 0.03)));
  const curve = new CatmullRomCurve3(pts, true, "centripetal");
  const segs = opts.segments;
  const radial = opts.radial;
  const frames = curve.computeFrenetFrames(segs, true);
  yield;
  // reveal parameter: arc distance from the topmost sample, normalised to 1 at the far side
  const samples: Vector3[] = [];
  for (let i = 0; i <= segs; i++) {
    if ((i & 255) === 255) yield;
    samples.push(curve.getPointAt(i / segs));
  }
  let top = 0;
  samples.forEach((p, i) => {
    if (p.y > samples[top].y) top = i;
  });
  const positions = new Float32Array((segs + 1) * (radial + 1) * 3);
  const normals = new Float32Array((segs + 1) * (radial + 1) * 3);
  const reveal = new Float32Array((segs + 1) * (radial + 1));
  const uvs = new Float32Array((segs + 1) * (radial + 1) * 2);
  const a = opts.halfWidth;
  const b = opts.halfThickness;
  const nrm = new Vector3();
  const off = new Vector3();
  const N = new Vector3();
  const B = new Vector3();
  for (let i = 0; i <= segs; i++) {
    if ((i & 31) === 31) yield;
    const u = i / segs;
    const p = samples[i];
    const tw = u * opts.twists * Math.PI * 2;
    const ct = Math.cos(tw);
    const st = Math.sin(tw);
    // rotate the frame around the tangent by the twist angle
    N.copy(frames.normals[i]).multiplyScalar(ct).addScaledVector(frames.binormals[i], st);
    B.copy(frames.binormals[i]).multiplyScalar(ct).addScaledVector(frames.normals[i], -st);
    let d = Math.abs(i - top) / segs;
    d = Math.min(d, 1 - d) * 2; // 0 at the top, 1 at the opposite point of the closed curve
    // slightly thicker high up (frame 09: the strands widen towards the top of the frame)
    const k0 = Math.min(1, Math.max(0, (p.y - 2.5) / 4));
    const swell = 1 + 0.3 * k0 * k0 * (3 - 2 * k0);
    const aw = a * swell;
    const bw = b * swell;
    for (let j = 0; j <= radial; j++) {
      const th = (j / radial) * Math.PI * 2;
      const c = Math.cos(th);
      const s = Math.sin(th);
      off.copy(N).multiplyScalar(c * aw).addScaledVector(B, s * bw);
      nrm.copy(N).multiplyScalar(c / aw).addScaledVector(B, s / bw).normalize();
      const k = i * (radial + 1) + j;
      positions.set([p.x + off.x, p.y + off.y, p.z + off.z], k * 3);
      normals.set([nrm.x, nrm.y, nrm.z], k * 3);
      reveal[k] = d;
      uvs.set([u, j / radial], k * 2);
    }
  }
  const index: number[] = [];
  for (let i = 0; i < segs; i++) {
    if ((i & 127) === 127) yield;
    for (let j = 0; j < radial; j++) {
      const a0 = i * (radial + 1) + j;
      const b0 = (i + 1) * (radial + 1) + j;
      index.push(a0, b0, a0 + 1, b0, b0 + 1, a0 + 1);
    }
  }
  yield;
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new BufferAttribute(positions, 3));
  geometry.setAttribute("normal", new BufferAttribute(normals, 3));
  geometry.setAttribute("uv", new BufferAttribute(uvs, 2));
  geometry.setAttribute("aReveal", new BufferAttribute(reveal, 1));
  geometry.setIndex(index);
  geometry.computeBoundingSphere();

  const uniforms = {
    uReveal: { value: 1 },
    uLight: { value: 1 },
    uCut: { value: 0.985 },
    uKey: { value: new Vector3(-0.5, 0.55, 0.65) },
    uBackLight: { value: new Vector3(0.1, 0.75, -0.65) },
    uOrbPos: { value: new Vector3() },
    uOrbRange: { value: 1 },
  };
  const material = new ShaderMaterial({
    name: "OracleLoop",
    uniforms: {
      ...uniforms,
      uBase: { value: new Color(0.008, 0.026, 0.011) },
      uSpecColor: { value: new Color(0.16, 0.9, 0.05) },
      uOrbColor: { value: new Color(0.12, 0.85, 0.06) },
      uRimColor: { value: new Color(0.03, 0.16, 0.04) },
    },
    vertexShader: LOOP_VERTEX,
    fragmentShader: LOOP_FRAGMENT,
  });
  const mesh = new Mesh(geometry, material);
  mesh.name = "oracle_loop";
  mesh.frustumCulled = false;
  return { mesh, uniforms };
}
