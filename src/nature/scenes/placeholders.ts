/**
 * PLACEHOLDER geometry kit. Procedural stand-ins used until the Blender GLBs exist
 * (and as the fail-soft fallback afterwards). Nothing here is final art: lumpy
 * blobs for moss masses, variable-radius tubes for wood, faceted shards for rock,
 * additive point clouds, a flat seed emblem. Every geometry carries COLOR_0 per the
 * asset contract (R density, G length, B baked AO) so the real material path is
 * exercised.
 */
import {
  AdditiveBlending,
  BufferAttribute,
  BufferGeometry,
  CatmullRomCurve3,
  Color,
  Float32BufferAttribute,
  IcosahedronGeometry,
  Mesh,
  Points,
  ShaderMaterial,
  Shape,
  Vector2,
  Vector3,
  type IUniform,
  type Material,
  type Object3D,
} from "three";
import { mergeVertices } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import type { Rng } from "../core/rng";
import type { SceneContext } from "../types";

// ---------------------------------------------------------------------------
// Seeded 3D value noise
// ---------------------------------------------------------------------------

export type Noise3 = (x: number, y: number, z: number) => number;

/** Value noise in [-1, 1], fully determined by the rng stream. */
export function createNoise3(rng: Rng): Noise3 {
  const p = Array.from({ length: 256 }, (_, i) => i);
  for (let i = 255; i > 0; i--) {
    const j = Math.floor(rng.next() * (i + 1));
    [p[i], p[j]] = [p[j], p[i]];
  }
  const perm = new Uint16Array(512);
  for (let i = 0; i < 512; i++) perm[i] = p[i & 255];
  const vals = new Float32Array(256);
  for (let i = 0; i < 256; i++) vals[i] = rng.next() * 2 - 1;
  const h = (x: number, y: number, z: number) => vals[perm[perm[perm[x & 255] + (y & 255)] + (z & 255)]];
  const fade = (t: number) => t * t * (3 - 2 * t);
  return (x, y, z) => {
    const xi = Math.floor(x);
    const yi = Math.floor(y);
    const zi = Math.floor(z);
    const u = fade(x - xi);
    const v = fade(y - yi);
    const w = fade(z - zi);
    const l = (a: number, b: number, k: number) => a + (b - a) * k;
    return l(
      l(l(h(xi, yi, zi), h(xi + 1, yi, zi), u), l(h(xi, yi + 1, zi), h(xi + 1, yi + 1, zi), u), v),
      l(l(h(xi, yi, zi + 1), h(xi + 1, yi, zi + 1), u), l(h(xi, yi + 1, zi + 1), h(xi + 1, yi + 1, zi + 1), u), v),
      w,
    );
  };
}

export function fbm3(noise: Noise3, x: number, y: number, z: number, octaves = 4): number {
  let s = 0;
  let a = 0.5;
  let f = 1;
  let norm = 0;
  for (let i = 0; i < octaves; i++) {
    s += a * noise(x * f + i * 13.1, y * f + i * 7.7, z * f + i * 3.3);
    norm += a;
    a *= 0.5;
    f *= 2.03;
  }
  return s / norm;
}

function setContractColors(geometry: BufferGeometry, ao: Float32Array, length?: Float32Array): void {
  const n = ao.length;
  const c = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    c[i * 4] = 1;
    c[i * 4 + 1] = length ? length[i] : 0.6;
    c[i * 4 + 2] = Math.min(1, Math.max(0, ao[i]));
    c[i * 4 + 3] = 1;
  }
  geometry.setAttribute("color", new BufferAttribute(c, 4));
}

// ---------------------------------------------------------------------------
// Blobs, shards, tubes
// ---------------------------------------------------------------------------

export interface BlobOptions {
  radius: number;
  detail?: number;
  /** Displacement amplitude as a fraction of the radius. */
  roughness?: number;
  frequency?: number;
  scale?: [number, number, number];
  /** 0–1: squash the lower part flat (moss cushions sitting on something). */
  flattenBottom?: number;
}

/** PLACEHOLDER lumpy organic mass (moss cushion / overgrown volume). */
export function blobGeometry(rng: Rng, opts: BlobOptions): BufferGeometry {
  const noise = createNoise3(rng);
  let g: BufferGeometry = new IcosahedronGeometry(opts.radius, opts.detail ?? 14);
  g.deleteAttribute("normal");
  g.deleteAttribute("uv");
  g = mergeVertices(g);
  const pos = g.getAttribute("position") as BufferAttribute;
  const ao = new Float32Array(pos.count);
  const len = new Float32Array(pos.count);
  const v = new Vector3();
  const amp = (opts.roughness ?? 0.28) * opts.radius;
  const freq = (opts.frequency ?? 1.6) / opts.radius;
  const [sx, sy, sz] = opts.scale ?? [1, 1, 1];
  const off = rng.range(0, 100);
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i);
    const dir = v.clone().normalize();
    const n = fbm3(noise, v.x * freq + off, v.y * freq, v.z * freq, 5);
    const d = n * amp;
    v.addScaledVector(dir, d);
    if (opts.flattenBottom && v.y < 0) v.y *= 1 - opts.flattenBottom;
    pos.setXYZ(i, v.x * sx, v.y * sy, v.z * sz);
    ao[i] = 0.62 + (n + 0.35) * 0.9;
    len[i] = 0.5 + 0.5 * fbm3(noise, v.x * freq * 2 + 40, v.y * freq * 2, v.z * freq * 2, 2);
  }
  pos.needsUpdate = true;
  g.computeVertexNormals();
  setContractColors(g, ao, len);
  g.computeBoundingSphere();
  return g;
}

/** PLACEHOLDER faceted rock fragment. */
export function rockGeometry(rng: Rng, size: [number, number, number], detail = 6): BufferGeometry {
  const noise = createNoise3(rng);
  let g: BufferGeometry = new IcosahedronGeometry(1, detail);
  g.deleteAttribute("normal");
  g.deleteAttribute("uv");
  g = mergeVertices(g);
  const pos = g.getAttribute("position") as BufferAttribute;
  const v = new Vector3();
  const off = rng.range(0, 100);
  // a few cutting planes give flat facets
  const planes = Array.from({ length: 7 }, () => ({
    n: new Vector3(rng.range(-1, 1), rng.range(-1, 1), rng.range(-1, 1)).normalize(),
    d: rng.range(0.62, 0.85),
  }));
  const ao = new Float32Array(pos.count);
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i);
    const n = fbm3(noise, v.x * 2.2 + off, v.y * 2.2, v.z * 2.2, 5);
    v.multiplyScalar(1 + n * 0.22);
    for (const pl of planes) {
      const k = v.dot(pl.n);
      if (k > pl.d) v.addScaledVector(pl.n, pl.d - k);
    }
    pos.setXYZ(i, v.x * size[0] * 0.5, v.y * size[1] * 0.5, v.z * size[2] * 0.5);
    ao[i] = 0.7 + n * 0.8;
  }
  pos.needsUpdate = true;
  g.computeVertexNormals();
  setContractColors(g, ao);
  g.computeBoundingSphere();
  return g;
}

export interface TubeOptions {
  tubular?: number;
  radial?: number;
  /** Radial bumpiness as a fraction of the radius. */
  bumps?: number;
  rng?: Rng;
  /** Integer number of UV wraps around the trunk. */
  wraps?: number;
}

/**
 * PLACEHOLDER wood: tube along a Catmull-Rom spline with variable radius.
 * UVs follow the asset contract: U around the trunk, V along the grain, 1 UV unit = 0.5 m.
 */
export function tubeGeometry(points: Vector3[], radius: (u: number) => number, opts: TubeOptions = {}): BufferGeometry {
  const curve = new CatmullRomCurve3(points, false, "centripetal");
  const tubular = opts.tubular ?? 96;
  const radial = opts.radial ?? 24;
  const frames = curve.computeFrenetFrames(tubular, false);
  const noise = opts.rng ? createNoise3(opts.rng) : null;
  const bumps = opts.bumps ?? 0.12;
  const wraps = opts.wraps ?? 2;
  const length = curve.getLength();
  const positions: number[] = [];
  const normals: number[] = [];
  const uvs: number[] = [];
  const ao: number[] = [];
  const p = new Vector3();
  const n = new Vector3();
  for (let i = 0; i <= tubular; i++) {
    const u = i / tubular;
    curve.getPointAt(u, p);
    const N = frames.normals[i];
    const B = frames.binormals[i];
    const r0 = radius(u);
    for (let j = 0; j <= radial; j++) {
      const a = (j / radial) * Math.PI * 2;
      n.set(0, 0, 0).addScaledVector(N, Math.cos(a)).addScaledVector(B, Math.sin(a)).normalize();
      const bump = noise ? fbm3(noise, Math.cos(a) * 1.3, Math.sin(a) * 1.3, u * length * 3.0, 4) : 0;
      const r = r0 * (1 + bump * bumps);
      positions.push(p.x + n.x * r, p.y + n.y * r, p.z + n.z * r);
      normals.push(n.x, n.y, n.z);
      uvs.push((j / radial) * wraps, (u * length) / 0.5);
      ao.push(0.78 + bump * 0.9);
    }
  }
  const indices: number[] = [];
  for (let i = 0; i < tubular; i++) {
    for (let j = 0; j < radial; j++) {
      const a = i * (radial + 1) + j;
      const b = (i + 1) * (radial + 1) + j;
      indices.push(a, b, a + 1, b, b + 1, a + 1);
    }
  }
  const g = new BufferGeometry();
  g.setAttribute("position", new Float32BufferAttribute(positions, 3));
  g.setAttribute("normal", new Float32BufferAttribute(normals, 3));
  g.setAttribute("uv", new Float32BufferAttribute(uvs, 2));
  g.setIndex(indices);
  g.computeVertexNormals();
  setContractColors(g, Float32Array.from(ao));
  g.computeBoundingSphere();
  return g;
}

/** Point at u (0–1) along the same spline a tube was built from. */
export function curvePoint(points: Vector3[], u: number): Vector3 {
  return new CatmullRomCurve3(points, false, "centripetal").getPointAt(Math.min(1, Math.max(0, u)));
}

// ---------------------------------------------------------------------------
// Particles
// ---------------------------------------------------------------------------

export interface ParticleOptions {
  count: number;
  /** Fill `out` with the position of particle i. */
  place: (i: number, out: Vector3) => void;
  /** World-space size range (metres). */
  size: [number, number];
  colors: string[];
  /** Linear brightness range; a few particles get `hotChance` × `hot` boost. */
  brightness: [number, number];
  hot?: number;
  hotChance?: number;
  drift?: number;
}

const PARTICLE_VERTEX = /* glsl */ `
uniform float uTime;
uniform float uScale;
uniform float uDrift;
attribute float aSize;
attribute vec3 aColor;
attribute float aPhase;
varying vec3 vColor;
varying float vPulse;
void main() {
  vec3 p = position;
  float ph = aPhase * 6.2831853;
  p += uDrift * vec3(sin(uTime * 0.31 + ph), sin(uTime * 0.23 + ph * 1.7), cos(uTime * 0.27 + ph * 0.6));
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = max(1.0, aSize * uScale / max(0.05, -mv.z));
  vPulse = 0.75 + 0.25 * sin(uTime * (0.8 + aPhase) + ph);
  vColor = aColor;
}
`;

const PARTICLE_FRAGMENT = /* glsl */ `
uniform float uOpacity;
varying vec3 vColor;
varying float vPulse;
void main() {
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(c, c);
  if (r2 > 1.0) discard;
  float core = exp(-r2 * 9.0);
  float halo = exp(-r2 * 2.5) * 0.35;
  gl_FragColor = vec4(vColor * (core + halo) * vPulse * uOpacity, 1.0);
  // no-ops in the linear HDR pass; applied only when rendering straight to the canvas (post=0)
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

/** PLACEHOLDER additive glowing point cloud. Update `uTime` / `uScale` (/ `uOpacity`) per frame. */
export function particleCloud(rng: Rng, opts: ParticleOptions): Points<BufferGeometry, ShaderMaterial> {
  const n = opts.count;
  const pos = new Float32Array(n * 3);
  const size = new Float32Array(n);
  const color = new Float32Array(n * 3);
  const phase = new Float32Array(n);
  const v = new Vector3();
  const palette = opts.colors.map((c) => new Color(c));
  for (let i = 0; i < n; i++) {
    opts.place(i, v);
    pos.set([v.x, v.y, v.z], i * 3);
    const s = rng.range(opts.size[0], opts.size[1]);
    size[i] = s * (rng.chance(0.06) ? 2.2 : 1);
    let b = rng.range(opts.brightness[0], opts.brightness[1]);
    b = b * b;
    if (rng.chance(opts.hotChance ?? 0.03)) b *= opts.hot ?? 6;
    const c = rng.pick(palette);
    color.set([c.r * b, c.g * b, c.b * b], i * 3);
    phase[i] = rng.next();
  }
  const g = new BufferGeometry();
  g.setAttribute("position", new BufferAttribute(pos, 3));
  g.setAttribute("aSize", new BufferAttribute(size, 1));
  g.setAttribute("aColor", new BufferAttribute(color, 3));
  g.setAttribute("aPhase", new BufferAttribute(phase, 1));
  g.computeBoundingSphere();
  const uniforms: Record<string, IUniform> = {
    uTime: { value: 0 },
    uScale: { value: 500 },
    uDrift: { value: opts.drift ?? 0.01 },
    uOpacity: { value: 1 },
  };
  const m = new ShaderMaterial({
    name: "PlaceholderParticles",
    uniforms,
    vertexShader: PARTICLE_VERTEX,
    fragmentShader: PARTICLE_FRAGMENT,
    blending: AdditiveBlending,
    transparent: true,
    depthWrite: false,
    depthTest: true,
  });
  const points = new Points(g, m);
  points.frustumCulled = false;
  return points;
}

// ---------------------------------------------------------------------------
// Placement helper
// ---------------------------------------------------------------------------

export interface PlaceOptions {
  position?: [number, number, number];
  rotation?: [number, number, number];
  scale?: number | [number, number, number];
  name?: string;
}

/** Create a PLACEHOLDER mesh, put it under `parent` and on debug layer `layer`. */
export function placeholderMesh(
  ctx: SceneContext,
  parent: Object3D,
  geometry: BufferGeometry,
  material: Material,
  layer: string,
  opts: PlaceOptions = {},
): Mesh {
  const mesh = new Mesh(geometry, material);
  mesh.name = `PLACEHOLDER_${opts.name ?? layer}`;
  if (opts.position) mesh.position.set(...opts.position);
  if (opts.rotation) mesh.rotation.set(...opts.rotation);
  if (opts.scale !== undefined) {
    if (typeof opts.scale === "number") mesh.scale.setScalar(opts.scale);
    else mesh.scale.set(...opts.scale);
  }
  parent.add(mesh);
  ctx.layers.assign(mesh, layer);
  return mesh;
}

/** Pixels per metre at distance 1 for point sizes (drawing-buffer height / (2 tan(fov/2))). */
export function pointScale(viewportHeightPx: number, dpr: number, fovDeg: number): number {
  return (viewportHeightPx * dpr) / (2 * Math.tan((fovDeg * Math.PI) / 360));
}

// ---------------------------------------------------------------------------
// Seed emblem (own symbol of Silva)
// ---------------------------------------------------------------------------

/**
 * Seed emblem pieces following `assets-src/emblem/seed.svg` (construction in units
 * where the almond is 100 high, y up): almond = intersection of two discs of radius
 * 62.5 centred at (±37.5, 0); channel = band 6 wide, tilted 18° clockwise from
 * vertical through (4, 0). The left piece keeps the top tip, the right piece the
 * bottom tip. Returns [left, right] shapes in the XY plane, centred on the origin,
 * scaled to `height`. Corner rounding (radius 1) is left out of this PLACEHOLDER.
 */
export function seedEmblemShapes(height: number, steps = 24): [Shape, Shape] {
  const s = height / 100;
  const R = 62.5;
  const angle = (x: number, y: number, cx: number) => Math.atan2(y, x - cx);
  const arc = (cx: number, a0: number, a1: number, n: number, out: Vector2[], skipFirst: boolean) => {
    for (let i = skipFirst ? 1 : 0; i <= n; i++) {
      const a = a0 + ((a1 - a0) * i) / n;
      out.push(new Vector2((cx + R * Math.cos(a)) * s, R * Math.sin(a) * s));
    }
  };
  // right boundary of the almond = circle centred at (-37.5, 0); left boundary = centred at (+37.5, 0)
  const left: Vector2[] = [];
  arc(-37.5, angle(12.87, 37, -37.5), angle(0, 50, -37.5), steps, left, false);
  let a1 = angle(-11.68, -38.56, 37.5);
  const a0 = angle(0, 50, 37.5);
  if (a1 < a0) a1 += Math.PI * 2;
  arc(37.5, a0, a1, steps * 2, left, true);

  const right: Vector2[] = [];
  arc(-37.5, angle(17.06, 30.49, -37.5), angle(0, -50, -37.5), steps * 2, right, false);
  arc(37.5, angle(0, -50, 37.5), angle(-7.08, -43.81, 37.5), Math.max(2, steps >> 2), right, true);
  return [new Shape(left), new Shape(right)];
}
