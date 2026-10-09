/**
 * Canopy foliage, built once per seed / quality (pure CPU, deterministic).
 *
 * Leaves are the kit's `leaf_canopy_a/b/c` (2.2–2.9 cm, 10 triangles each), scaled to
 * ≈ 2.4–4 cm and grouped in small sprays: a short virtual stem (4–10 cm, not drawn — at
 * 2.7 mm per pixel it would be sub-pixel) leaves a twig of `canopy.glb` and carries
 * 2–8 leaflets, alternating left / right, opening 30–65° from the stem, tilted every
 * way around the spray plane, plus a terminal leaflet. Spray roots are scattered over
 * the twig surfaces with area × COLOR_0.R^p (R: 0 at every cluster base, rising along
 * the outer part of each twig, reduced in the crown's central hollow); spray length and
 * leaflet size follow COLOR_0.G (smaller at the tips). Sprays point away from the twig
 * and from the crown's hidden centre H, their leaves turn towards the key light.
 *
 * Seen where it matters: twice the budget is grown as candidates, their leaf area is
 * splatted into a 5 cm density grid, and the kept leaves are drawn by weight
 * (floor + visibility from the story's cameras, weighted reservoir sampling) — leaves
 * buried behind the front of the crown are mostly not drawn, the ones that are keep the
 * inside dark. Inside the hollow (an ellipse in the close-up view, ragged edge) the front
 * leaves are mostly left out, so the dark middle of frame 17 opens up. The full candidate
 * set still shades the crown (as if all leaves were there), so the light does not change
 * with the budget.
 *
 * Dim leaves fill the space behind the crown: on `far_canopy_outer` (density = its
 * COLOR_0.R) and on the dark cap `far_canopy_inner`.
 *
 * Lighting data baked per leaf:
 *  - key transmittance: each leaf marches towards the key light through the density
 *    grid (extinction σ × leaf area density): the outer leaves facing the light are
 *    fully lit, the inside of the crown falls dark;
 *  - fill AO: COLOR_0.B of the twig at the spray root (volumetric AO baked in Blender).
 * Exit data: a lumpy per-cluster (± 0.15) and per-leaf (± 0.05) jitter of the hole
 * radius (leafMaterial.ts), so the crown parts in clumps, not along a clean circle.
 *
 * The build is one generator (`foliageSteps`, vegetation/slices.ts) with checkpoints
 * between triangles, samples, sprays, leaves, grid rows and grid cells (typically < 1 ms
 * of work between two of them): `buildFoliage` runs it in ≈ 8 ms slices with a
 * macrotask yield in between (the set prepares while the current episode keeps
 * rendering), `buildFoliageSync` in one go. Both execute the same code in the same
 * order — same random draws, byte-identical instance data. All scratch state lives in
 * the generator.
 */
import { Quaternion, Vector3, type Mesh, type Object3D } from "three";
import { smoothstep } from "../../core/math";
import type { Rng } from "../../core/rng";
import { fbm3, hash3i } from "../../vegetation/noise";
import { runSliced, runSync, type Steps } from "../../vegetation/slices";
import { SurfaceSet } from "../../vegetation/SurfaceScatter";

export const LEAF_ITEMS = ["leaf_canopy_a", "leaf_canopy_b", "leaf_canopy_c"] as const;
/** Relative frequency of the three kit leaves (b darker, c yellow-green). */
const ITEM_WEIGHTS = [0.42, 0.34, 0.24];

/**
 * Per-instance layout (floats):
 *  aRel  (3) leaf root relative to its cluster base
 *  aAxis (3) growth axis (kit +Y)          aSide (3) blade width axis (kit +X)
 *  aMisc (4) size, cluster index, wind phase, spray sway arm (m)
 *  aLook (4) albedo tint rgb (linear, × brightness), key transmittance
 *  aAnim (4) fill AO, unfurl delay 0–1 (base → tip of the cluster), exit jitter, translucency
 *  aHollow (1) membership of the close-up's dark hollow 0–1 (darkened in the shader)
 */
export const LEAF_STRIDE = 22;
export const LEAF_ATTRIBUTES: Record<string, [number, number]> = {
  aRel: [0, 3],
  aAxis: [3, 3],
  aSide: [6, 3],
  aMisc: [9, 4],
  aLook: [13, 4],
  aAnim: [17, 4],
  aHollow: [21, 1],
};

export interface FoliageCluster {
  mesh: Mesh;
  base: Vector3;
  radius: number;
}

/**
 * The crown's dark hollow, as seen from one camera: an ellipse in that view's frame
 * (u right, v down, 0–1) whose front leaves are mostly not drawn, so the dark inside
 * of the crown shows (frame 17: the close-up is a ring of lit foliage around a dark
 * middle; frame 16 shows the same hollow from further away).
 */
export interface FoliageHollow {
  position: Vector3;
  quaternion: Quaternion;
  /** Vertical fov (degrees) and aspect of that view. */
  fov: number;
  aspect: number;
  /** Centre (u, v) and semi-axes (fractions of the frame width / height). */
  centre: [number, number];
  radii: [number, number];
  /** Softness of the edge (in units of the radius) and its raggedness. */
  soft: number;
  ragged: number;
}

export interface FoliageInput {
  /** Cluster twig meshes, index = cluster id (their world transforms are respected). */
  clusters: FoliageCluster[];
  farInner: Mesh | null;
  farOuter: Mesh | null;
  /** Cluster ids used for the leaves of the two far shells. */
  farInnerId: number;
  farOuterId: number;
  /** Pivots ("bases") of the far shells. */
  farInnerBase: Vector3;
  farOuterBase: Vector3;
  /** The crown root (space of the plants, identity world transform expected). */
  space: Object3D;
  /** Hidden centre the clusters radiate from. */
  hidden: Vector3;
  /** Unit vector towards the key light. */
  toLight: Vector3;
  /** Optional dark hollow (see FoliageHollow). */
  hollow: FoliageHollow | null;
  /**
   * Camera positions of the close-up (a polyline): crown leaves closer than ≈ 0.6–1.25 m
   * are mostly left out — at a few dozen cm a 3 cm leaf would cover a fifth of the
   * frame as a pale out-of-focus blob (the reference close-up has none).
   */
  nearPath: Vector3[];
  /** Modelled length (m) and one-sided blade area (m²) of each kit item. */
  itemLength: number[];
  itemArea: number[];
  /** Total leaves drawn (crown + far). */
  count: number;
  /** Leaf size multiplier (quality compensation, 1 at high). */
  sizeScale: number;
  /**
   * Leaf area of the high-quality crown over this build's (fewer, larger leaves on lower
   * qualities): the extinction towards the key and the camera is scaled by it, so the
   * crown is shaded and selected the same way on every quality.
   */
  densityScale: number;
  rng(label: string): Rng;
  seed: number;
}

export interface FoliageBuild {
  /** Packed instance data per kit item (LEAF_STRIDE floats per leaf). */
  data: Float32Array[];
  counts: number[];
  total: number;
  crown: number;
  far: number;
  /** Crown candidates grown before the visibility selection. */
  candidates: number;
  sprays: number;
  /** Per crown cluster: centroid of its leaves (crown space) and leaf count. */
  clusterCentroid: Vector3[];
  clusterLeaves: number[];
  /** Key transmittance and view visibility of the drawn crown leaves (debug). */
  keyT: { p10: number; p50: number; p90: number; mean: number };
  visibility: { p10: number; p50: number; p90: number; mean: number };
  /** Hollow membership of the drawn crown leaves (debug). */
  hollow: { p10: number; p50: number; p90: number; mean: number };
  /** Milliseconds from start to end, total and per phase (debug; wall clock: a sliced build includes the browser's turns). */
  ms: number;
  phases: string;
}

/** Leaf size: the kit leaves (2.2–2.9 cm) → ≈ 2.6–3.5 cm before variation. */
const BASE_SCALE = 1.45;
/** Shares of the leaf budget. */
const SHARE_FAR_OUTER = 0.05;
const SHARE_FAR_INNER = 0.025;
/** Crown candidates per drawn crown leaf. */
const CANDIDATES = 2;
/** Selection weight of a fully hidden candidate (visible ones: + visibility). */
const HIDDEN_WEIGHT = 0.03;
/** Extinction per unit leaf area density: towards the key light / towards the camera. */
const SIGMA_KEY = 0.42;
const SIGMA_VIEW = 0.5;

// ---------------------------------------------------------------------------
// growable per-item float lists
// ---------------------------------------------------------------------------

class FloatList {
  data: Float32Array;
  length = 0;
  constructor(capacity: number) {
    this.data = new Float32Array(Math.max(64, capacity));
  }
  reserve(extra: number): void {
    if (this.length + extra <= this.data.length) return;
    const next = new Float32Array(Math.max(this.data.length * 2, this.length + extra));
    next.set(this.data.subarray(0, this.length));
    this.data = next;
  }
}

// ---------------------------------------------------------------------------
// small vector helpers (hot loops; scratch vectors live in the build)
// ---------------------------------------------------------------------------

function randomUnit(rng: Rng, out: Vector3): Vector3 {
  // Marsaglia: uniform on the sphere
  let x = 0;
  let y = 0;
  let s = 2;
  while (s >= 1 || s < 1e-6) {
    x = rng.range(-1, 1);
    y = rng.range(-1, 1);
    s = x * x + y * y;
  }
  const k = 2 * Math.sqrt(1 - s);
  return out.set(x * k, y * k, 1 - 2 * s);
}

function perpendicular(n: Vector3, out: Vector3): Vector3 {
  if (Math.abs(n.y) < 0.9) out.set(0, 1, 0);
  else out.set(1, 0, 0);
  return out.addScaledVector(n, -out.dot(n)).normalize();
}

function pickWeighted(r: number, weights: readonly number[]): number {
  let total = 0;
  for (const w of weights) total += w;
  let x = r * total;
  for (let i = 0; i < weights.length; i++) {
    x -= weights[i];
    if (x <= 0) return i;
  }
  return weights.length - 1;
}

/** Quantiles / mean of values in [0, 1] (40-bin histogram). */
function summary(values: Float32Array, n: number): { p10: number; p50: number; p90: number; mean: number } {
  const hist = new Float64Array(40);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const v = Math.min(1, Math.max(0, values[i]));
    hist[Math.min(39, Math.floor(v * 40))]++;
    sum += v;
  }
  const q = (p: number) => {
    let acc = 0;
    for (let i = 0; i < hist.length; i++) {
      acc += hist[i];
      if (acc >= p * n) return (i + 0.5) / hist.length;
    }
    return 1;
  };
  return n ? { p10: q(0.1), p50: q(0.5), p90: q(0.9), mean: sum / n } : { p10: 0, p50: 0, p90: 0, mean: 0 };
}

// ---------------------------------------------------------------------------
// leaf area density grid
// ---------------------------------------------------------------------------

class DensityGrid {
  readonly nx: number;
  readonly ny: number;
  readonly nz: number;
  readonly data: Float32Array;

  constructor(
    readonly min: Vector3,
    max: Vector3,
    readonly cell: number,
  ) {
    this.nx = Math.max(1, Math.ceil((max.x - min.x) / cell));
    this.ny = Math.max(1, Math.ceil((max.y - min.y) / cell));
    this.nz = Math.max(1, Math.ceil((max.z - min.z) / cell));
    this.data = new Float32Array(this.nx * this.ny * this.nz);
  }

  /** Add `w` (m² of leaf area) to the cell of (x, y, z). */
  splat(x: number, y: number, z: number, w: number): void {
    const i = Math.floor((x - this.min.x) / this.cell);
    const j = Math.floor((y - this.min.y) / this.cell);
    const k = Math.floor((z - this.min.z) / this.cell);
    if (i < 0 || j < 0 || k < 0 || i >= this.nx || j >= this.ny || k >= this.nz) return;
    this.data[(k * this.ny + j) * this.nx + i] += w;
  }

  /** Leaf area → leaf area density (m²/m³), then a separable [1 2 1] blur per axis (pausable steps). */
  *finishSteps(): Steps<void> {
    const v = 1 / (this.cell * this.cell * this.cell);
    for (let i = 0; i < this.data.length; i++) {
      if ((i & 65535) === 65535) yield;
      this.data[i] *= v;
    }
    const tmp = new Float32Array(this.data.length);
    const { nx, ny, nz } = this;
    yield* blurPass(this.data, tmp, 1, nx, (idx) => idx % nx);
    yield* blurPass(tmp, this.data, nx, ny, (idx) => Math.floor(idx / nx) % ny);
    yield* blurPass(this.data, tmp, nx * ny, nz, (idx) => Math.floor(idx / (nx * ny)));
    this.data.set(tmp);
  }

  /**
   * Optical depth (∫ density, m²/m²) in front of every cell along +z (towards the
   * cameras, which all look down −z at the crown), own cell excluded (pausable steps).
   */
  *frontDepthSteps(): Steps<Float32Array> {
    const { nx, ny, nz, cell, data } = this;
    const out = new Float32Array(data.length);
    for (let j = 0; j < ny; j++) {
      yield;
      for (let i = 0; i < nx; i++) {
        let acc = 0;
        for (let k = nz - 1; k >= 0; k--) {
          const idx = (k * ny + j) * nx + i;
          out[idx] = acc;
          acc += data[idx] * cell;
        }
      }
    }
    return out;
  }

  /** Index of the cell of (x, y, z), −1 outside. */
  index(x: number, y: number, z: number): number {
    const i = Math.floor((x - this.min.x) / this.cell);
    const j = Math.floor((y - this.min.y) / this.cell);
    const k = Math.floor((z - this.min.z) / this.cell);
    if (i < 0 || j < 0 || k < 0 || i >= this.nx || j >= this.ny || k >= this.nz) return -1;
    return (k * this.ny + j) * this.nx + i;
  }

  /** exp(−σ ∫ density) from (x, y, z) along `dir` (unit), from `start` to `maxDist` m. */
  transmittance(x: number, y: number, z: number, dir: Vector3, sigma: number, step: number, start: number, maxDist: number): number {
    let tau = 0;
    const { nx, ny, nz, cell, min, data } = this;
    for (let d = start; d < maxDist; d += step) {
      const i = Math.floor((x + dir.x * d - min.x) / cell);
      const j = Math.floor((y + dir.y * d - min.y) / cell);
      const k = Math.floor((z + dir.z * d - min.z) / cell);
      if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) break;
      tau += data[(k * ny + j) * nx + i];
    }
    return Math.exp(-sigma * tau * step);
  }
}

/** One [1 2 1] blur pass along an axis of the grid (`coord`: index → coordinate along it). */
function* blurPass(src: Float32Array, dst: Float32Array, stride: number, n: number, coord: (idx: number) => number): Steps<void> {
  for (let idx = 0; idx < src.length; idx++) {
    if ((idx & 32767) === 32767) yield;
    const c = coord(idx);
    const a = c > 0 ? src[idx - stride] : src[idx];
    const b = c < n - 1 ? src[idx + stride] : src[idx];
    dst[idx] = 0.25 * a + 0.5 * src[idx] + 0.25 * b;
  }
}

// ---------------------------------------------------------------------------
// build
// ---------------------------------------------------------------------------

/**
 * Build the foliage time-sliced: `foliageSteps` run in slices of ≈ `yieldEveryMs` (8 ms)
 * with a macrotask yield in between (vegetation/slices.ts), so the set prepares while the
 * current episode keeps rendering. Same random draws and byte-identical data as
 * `buildFoliageSync`.
 */
export function buildFoliage(input: FoliageInput, yieldEveryMs = 8, owner: object | null = null): Promise<FoliageBuild> {
  return runSliced(foliageSteps(input), yieldEveryMs, owner);
}

/** The same build in one go (synchronous). */
export function buildFoliageSync(input: FoliageInput): FoliageBuild {
  return runSync(foliageSteps(input));
}

/**
 * The foliage build as pausable steps (`yield` = checkpoint, slices.ts): checkpoints sit
 * between triangles, samples, sprays, leaves, grid rows and cells. Its scratch vectors are
 * its own, so another build may run while this one waits.
 */
export function* foliageSteps(input: FoliageInput): Steps<FoliageBuild> {
  const now = () => (typeof performance !== "undefined" ? performance.now() : 0);
  const t0 = now();
  const phases: string[] = [];
  let tPhase = t0;
  /** Phase times are wall clock: a sliced build includes the time the browser had in between. */
  const mark = (label: string) => {
    const t = now();
    phases.push(`${label} ${Math.round(t - tPhase)}`);
    tPhase = t;
  };
  // scratch vectors: this build's own (it may pause while another build uses its own)
  const tA = new Vector3();
  const tB = new Vector3();
  const tC = new Vector3();
  const tN = new Vector3();
  const tR = new Vector3();
  const tU = new Vector3();
  const tF = new Vector3();
  const tS = new Vector3();
  const tP = new Vector3();
  const tQ = new Vector3();
  const nClusters = input.clusters.length;
  const toL = input.toLight;
  const H = input.hidden;
  const sizeQ = input.sizeScale;
  const crownTarget = Math.round(input.count * (1 - SHARE_FAR_OUTER - SHARE_FAR_INNER));
  const crownCandidates = Math.round(crownTarget * CANDIDATES);
  // capacity per item from its expected share (crown candidates by ITEM_WEIGHTS, far leaves
  // are items 0 / 1) + 8 %: no regrowth in practice; a list that overflows still grows
  const weightSum = ITEM_WEIGHTS.reduce((a, b) => a + b, 0);
  const farPerItem = input.count * (SHARE_FAR_OUTER + SHARE_FAR_INNER) * 0.5;
  const lists = LEAF_ITEMS.map((_, i) => new FloatList((Math.ceil((crownCandidates * (ITEM_WEIGHTS[i] / weightSum) + (i < 2 ? farPerItem : 0)) * 1.08) + 64) * LEAF_STRIDE));
  yield;

  // per-cluster character: wind phase, hue shift, share of yellow young leaves, exit lumps
  const clRng = input.rng("canopy-clusters");
  const clPhase = input.clusters.map(() => clRng.next());
  const clHue = input.clusters.map(() => clRng.range(-1, 1));
  const clYoung = input.clusters.map(() => clRng.range(0.06, 0.22));
  const clExitJitter = input.clusters.map(() => clRng.range(-0.5, 0.5));

  const writeLeaf = (
    item: number,
    root: Vector3,
    cluster: number,
    base: Vector3,
    axis: Vector3,
    side: Vector3,
    size: number,
    phase: number,
    swayArm: number,
    tint: [number, number, number],
    ao: number,
    unfurl: number,
    transl: number,
  ) => {
    const L = lists[item];
    L.reserve(LEAF_STRIDE);
    const d = L.data;
    const o = L.length;
    d[o] = root.x - base.x;
    d[o + 1] = root.y - base.y;
    d[o + 2] = root.z - base.z;
    d[o + 3] = axis.x;
    d[o + 4] = axis.y;
    d[o + 5] = axis.z;
    d[o + 6] = side.x;
    d[o + 7] = side.y;
    d[o + 8] = side.z;
    d[o + 9] = size;
    d[o + 10] = cluster;
    d[o + 11] = phase;
    d[o + 12] = swayArm;
    d[o + 13] = tint[0];
    d[o + 14] = tint[1];
    d[o + 15] = tint[2];
    d[o + 16] = 1; // key transmittance, baked below
    d[o + 17] = ao;
    d[o + 18] = unfurl;
    d[o + 19] = 0; // exit jitter, baked below
    d[o + 20] = transl;
    d[o + 21] = 0; // hollow membership, set in the selection
    L.length += LEAF_STRIDE;
  };

  /** Leaf albedo tint: per-leaf brightness, young yellow leaves, deep blue-green ones, per-cluster hue. */
  const tint: [number, number, number] = [1, 1, 1];
  const makeTint = (rng: Rng, hue: number, young: number, bright: [number, number]) => {
    let r = 1;
    let g = 1;
    let b = 1;
    const u = rng.next();
    if (u < young) {
      // young leaves at the tips: yellower, a little lighter
      r = 1.16;
      g = 1.07;
      b = 0.66;
    } else if (u < young + 0.2) {
      // deep, bluish green
      r = 0.8;
      g = 0.9;
      b = 0.98;
    }
    const k = rng.range(bright[0], bright[1]);
    tint[0] = r * k * (1 + 0.07 * hue);
    tint[1] = g * k;
    tint[2] = b * k * (1 - 0.07 * hue);
  };

  // ---------------------------------------------------------------------------
  // crown candidates: sprays on the twigs
  // ---------------------------------------------------------------------------
  let candidates = 0;
  let sprays = 0;
  if (nClusters > 0) {
    const set = yield* SurfaceSet.steps(
      input.clusters.map((c) => c.mesh),
      input.space,
    );
    mark("surface");
    yield;
    const R = set.channel(0);
    const weights = new Float32Array(set.count);
    for (let t = 0; t < set.count; t++) {
      if ((t & 4095) === 4095) yield;
      const r = R[t];
      weights[t] = r > 0.02 ? set.areas[t] * Math.pow(r, 1.25) : 0;
    }
    const rng = input.rng("canopy-sprays");
    const meanPerSpray = 4.5;
    const samples = yield* set.sampleSteps(rng, Math.ceil((crownCandidates / meanPerSpray) * 1.3), weights);
    // samples come in surface order: visit them shuffled so the cut at the target is even
    const order = new Uint32Array(samples.count);
    for (let i = 0; i < samples.count; i++) order[i] = i;
    for (let i = samples.count - 1; i > 0; i--) {
      if ((i & 8191) === 0) yield;
      const j = Math.floor(rng.next() * (i + 1));
      const tmp = order[i];
      order[i] = order[j];
      order[j] = tmp;
    }
    mark("samples");
    yield;
    const lr = input.rng("canopy-leaflets");
    for (let oi = 0; oi < samples.count && candidates < crownCandidates; oi++) {
      if ((oi & 63) === 63) yield;
      const i = order[oi];
      const c = set.meshOf[samples.tri[i]];
      const cl = input.clusters[c];
      tP.set(samples.position[i * 3], samples.position[i * 3 + 1], samples.position[i * 3 + 2]);
      tN.set(samples.normal[i * 3], samples.normal[i * 3 + 1], samples.normal[i * 3 + 2]);
      const G = samples.color[i * 4 + 1];
      const B = samples.color[i * 4 + 2];
      // spray axis: away from the twig and from the hidden centre, a little towards the light
      tR.copy(tP).sub(H).normalize();
      randomUnit(lr, tA);
      tU.copy(tN).multiplyScalar(0.5).addScaledVector(tR, 0.85).addScaledVector(toL, 0.3).addScaledVector(tA, 0.6).normalize();
      const len = lr.range(0.04, 0.1) * (0.7 + 0.3 * G) * Math.sqrt(sizeQ);
      const item = pickWeighted(lr.next(), ITEM_WEIGHTS);
      // spray plane: its normal turned outwards and towards the light
      randomUnit(lr, tB);
      tS.copy(tR).multiplyScalar(0.55).addScaledVector(toL, 0.75).addScaledVector(tB, 0.55);
      tF.crossVectors(tS, tU);
      if (tF.lengthSq() < 1e-6) perpendicular(tU, tF);
      tF.normalize(); // lateral axis
      tS.crossVectors(tU, tF).normalize(); // plane normal
      const bend = lr.range(-0.3, 0.4);
      const spacing = lr.range(0.009, 0.014) * Math.sqrt(sizeQ);
      const m = Math.max(1, Math.floor((len * 0.72) / spacing));
      const sideSign = lr.sign();
      const cellPhase = hash3i(Math.floor(tP.x / 0.06), Math.floor(tP.y / 0.06), Math.floor(tP.z / 0.06), input.seed) * 0.3;
      const phaseSpray = clPhase[c] * 0.6 + cellPhase;
      const baseDist = Math.min(1, tP.distanceTo(cl.base) / Math.max(0.2, cl.radius));
      sprays++;
      for (let k = 0; k <= m && candidates < crownCandidates; k++) {
        const terminal = k === m;
        if (!terminal && lr.chance(0.12)) continue;
        const u = terminal ? 1 : 0.28 + (0.72 * (k + lr.range(-0.15, 0.15))) / m;
        const s = len * u;
        // point and direction of the (bent) stem at s
        const ang = bend * u;
        tQ.copy(tU).multiplyScalar(Math.cos(ang)).addScaledVector(tS, -Math.sin(ang));
        tC.copy(tP)
          .addScaledVector(tU, s * Math.cos(ang * 0.5))
          .addScaledVector(tS, -s * Math.sin(ang * 0.5));
        const side = (k % 2 === 0 ? 1 : -1) * sideSign;
        const theta = terminal ? lr.range(-0.15, 0.15) : 1.1 + (0.55 - 1.1) * u + lr.range(-0.22, 0.22);
        // leaflet axis: opens from the stem to its side, sags a little below the plane
        tA.copy(tQ)
          .multiplyScalar(Math.cos(theta))
          .addScaledVector(tF, side * Math.sin(theta))
          .addScaledVector(tS, -0.18 + lr.range(-0.22, 0.22))
          .normalize();
        // blade normal: the plane normal, tilted every way, a little towards the light
        randomUnit(lr, tB);
        tN.copy(tS).addScaledVector(tB, 0.5).addScaledVector(toL, 0.15);
        // width axis = axis × normal (right-handed: kit +X × +Y = +Z)
        tR.crossVectors(tA, tN);
        if (tR.lengthSq() < 1e-6) perpendicular(tA, tR);
        tR.normalize();
        const size = BASE_SCALE * sizeQ * lr.range(0.8, 1.2) * (1 - 0.25 * u) * (0.75 + 0.25 * G) * (terminal ? 0.95 : 1);
        // the leaflet sits on a short petiole beside the stem
        tC.addScaledVector(tF, side * 0.0025);
        makeTint(lr, clHue[c], clYoung[c] * (0.6 + 0.8 * u), [0.78, 1.18]);
        const unfurl = Math.min(1, baseDist * 0.85 + 0.15 * u);
        const phase = phaseSpray + lr.next() * 0.08;
        writeLeaf(item, tC, c, cl.base, tA, tR, size, phase, len * u * u, tint, Math.pow(Math.max(0.05, B), 1.15), unfurl, lr.range(0.55, 1));
        candidates++;
      }
    }
  }
  /** Floats of crown candidates per list (far leaves follow them). */
  const crownEnd = lists.map((L) => L.length);
  mark("sprays");
  yield;

  // ---------------------------------------------------------------------------
  // dim far leaves (single leaves on the far shells, always drawn)
  // ---------------------------------------------------------------------------
  let far = 0;
  const farLeaves = function* (mesh: Mesh | null, id: number, base: Vector3, target: number, label: string, useR: boolean, bright: [number, number], radial: boolean): Steps<void> {
    if (!mesh || target <= 0) return;
    const set = yield* SurfaceSet.steps([mesh], input.space);
    const R = set.channel(0);
    const weights = new Float32Array(set.count);
    for (let t = 0; t < set.count; t++) {
      if ((t & 2047) === 2047) yield;
      const n = set.meanNormal(t, tN);
      // only the faces turned to the camera side (+z) carry leaves
      const facing = Math.max(0, Math.min(1, (n.z + 0.35) / 0.7));
      weights[t] = set.areas[t] * facing * (useR ? Math.max(0, R[t]) : 1);
    }
    const rng = input.rng(label);
    const samples = yield* set.sampleSteps(rng, target, weights);
    for (let i = 0; i < samples.count; i++) {
      if ((i & 255) === 255) yield;
      tP.set(samples.position[i * 3], samples.position[i * 3 + 1], samples.position[i * 3 + 2]);
      // a loose layer around the shell
      randomUnit(rng, tA);
      tP.addScaledVector(tA, 0.06);
      randomUnit(rng, tB);
      tN.set(0, 0.45, 1).addScaledVector(tB, 0.8).normalize();
      randomUnit(rng, tA);
      tA.addScaledVector(tN, -tA.dot(tN));
      if (tA.lengthSq() < 1e-6) perpendicular(tN, tA);
      tA.normalize();
      tR.crossVectors(tA, tN).normalize();
      const size = BASE_SCALE * sizeQ * rng.range(0.9, 1.35);
      makeTint(rng, rng.range(-1, 1), 0.08, bright);
      // unfurl delay: inner leaves of the ring first
      const rr = Math.hypot(tP.x - base.x, tP.y - base.y);
      const unfurl = radial ? Math.min(1, Math.max(0, (rr - 1.0) / 3.0)) : rng.next() * 0.5;
      writeLeaf(rng.next() < 0.5 ? 0 : 1, tP, id, base, tA, tR, size, rng.next(), 0.03, tint, 0.35, unfurl, 0.4);
      far++;
    }
  };
  yield* farLeaves(input.farOuter, input.farOuterId, input.farOuterBase, Math.round(input.count * SHARE_FAR_OUTER), "canopy-far-outer", true, [0.35, 0.75], true);
  yield* farLeaves(input.farInner, input.farInnerId, input.farInnerBase, Math.round(input.count * SHARE_FAR_INNER), "canopy-far-inner", false, [0.4, 0.8], false);
  mark("far");
  yield;

  // ---------------------------------------------------------------------------
  // density grid of the crown candidates
  // ---------------------------------------------------------------------------
  const bases: Vector3[] = input.clusters.map((c) => c.base);
  const baseOf = (id: number): Vector3 => (id === input.farInnerId ? input.farInnerBase : id === input.farOuterId ? input.farOuterBase : bases[id]);
  const leafCentre = (d: Float32Array, o: number, item: number, out: Vector3) => {
    const b = baseOf(d[o + 10]);
    const half = 0.5 * input.itemLength[item] * d[o + 9];
    return out.set(b.x + d[o] + d[o + 3] * half, b.y + d[o + 1] + d[o + 4] * half, b.z + d[o + 2] + d[o + 5] * half);
  };
  const lo = new Vector3(Infinity, Infinity, Infinity);
  const hi = new Vector3(-Infinity, -Infinity, -Infinity);
  for (let item = 0; item < lists.length; item++) {
    const L = lists[item];
    for (let o = 0, n = 0; o < L.length; o += LEAF_STRIDE, n++) {
      if ((n & 4095) === 4095) yield;
      if (L.data[o + 10] === input.farOuterId) continue;
      leafCentre(L.data, o, item, tC);
      lo.min(tC);
      hi.max(tC);
    }
  }
  let grid: DensityGrid | null = null;
  if (Number.isFinite(lo.x)) {
    lo.subScalar(0.15);
    hi.addScalar(0.15);
    const g = new DensityGrid(lo, hi, 0.05);
    grid = g;
    for (let item = 0; item < lists.length; item++) {
      const L = lists[item];
      for (let o = 0, n = 0; o < crownEnd[item]; o += LEAF_STRIDE, n++) {
        if ((n & 4095) === 4095) yield;
        const s = L.data[o + 9];
        leafCentre(L.data, o, item, tC);
        g.splat(tC.x, tC.y, tC.z, input.itemArea[item] * s * s);
      }
    }
    yield* g.finishSteps();
  }
  mark("grid");
  yield;

  // ---------------------------------------------------------------------------
  // visibility-driven selection of the crown leaves (weighted reservoir sampling:
  // key = ln(u) / w, the `crownTarget` largest keys are drawn)
  // ---------------------------------------------------------------------------
  const keep = lists.map((L) => new Uint8Array(L.length / LEAF_STRIDE));
  const visOf = lists.map((L) => new Float32Array(L.length / LEAF_STRIDE));
  {
    let front: Float32Array | null = null;
    if (grid) front = yield* grid.frontDepthSteps();
    const srng = input.rng("canopy-select");
    const keys = new Float64Array(candidates);
    const hollowOf = hollowMask(input.hollow, input.seed);
    const nearOf = nearDistance(input.nearPath);
    let n = 0;
    for (let item = 0; item < lists.length; item++) {
      const L = lists[item];
      for (let o = 0; o < crownEnd[item]; o += LEAF_STRIDE) {
        if ((n & 255) === 255) yield;
        leafCentre(L.data, o, item, tC);
        let v = 1;
        if (grid && front) {
          const cellIdx = grid.index(tC.x, tC.y, tC.z);
          v = cellIdx >= 0 ? Math.exp(-SIGMA_VIEW * input.densityScale * front[cellIdx]) : 1;
        }
        visOf[item][o / LEAF_STRIDE] = v;
        let w = HIDDEN_WEIGHT + Math.pow(v, 0.8);
        // inside the hollow most leaves go, whole clumps of them (≈ 20 cm noise), so the
        // ones left are tufts, not confetti; the shader darkens them in the close-up
        const h = hollowOf(tC);
        L.data[o + 21] = h;
        if (h > 0) {
          const clump = smoothstep(0.38, 0.62, fbm3(tC.x * 4.5, tC.y * 4.5, tC.z * 4.5, input.seed + 91));
          w *= 1 - h * (0.3 + 0.65 * clump);
        }
        w *= 0.02 + 0.98 * smoothstep(0.6, 1.25, nearOf(tC));
        keys[n++] = Math.log(Math.max(1e-12, srng.next())) / w;
      }
    }
    yield;
    const take = Math.min(n, crownTarget);
    // the `take` largest keys are drawn: the threshold is the (n − take)-th smallest key
    // (the value a full sort would put there), found by selection in pausable steps
    let threshold = Infinity;
    if (take > 0) threshold = yield* kthSmallest(keys.slice(0, n), n - take);
    let j = 0;
    let kept = 0;
    for (let item = 0; item < lists.length; item++) {
      const L = lists[item];
      for (let o = 0; o < crownEnd[item]; o += LEAF_STRIDE) {
        if ((j & 4095) === 4095) yield;
        if (keys[j++] >= threshold && kept < take) {
          keep[item][o / LEAF_STRIDE] = 1;
          kept++;
        }
      }
      // far leaves are always drawn (the ones in the hollow darken with it)
      for (let o = crownEnd[item], m = 0; o < L.length; o += LEAF_STRIDE, m++) {
        if ((m & 255) === 255) yield;
        keep[item][o / LEAF_STRIDE] = 1;
        L.data[o + 21] = hollowOf(leafCentre(L.data, o, item, tC));
      }
    }
  }

  mark("select");
  yield;

  // ---------------------------------------------------------------------------
  // key transmittance, exit jitter, cluster centroids (drawn leaves only)
  // ---------------------------------------------------------------------------
  const centroid = Array.from({ length: nClusters }, () => new Vector3());
  const clusterLeaves = new Array<number>(nClusters).fill(0);
  const trng = input.rng("canopy-light");
  const erng = input.rng("canopy-exit");
  const keyVals = new Float32Array(crownTarget + 16);
  const visVals = new Float32Array(crownTarget + 16);
  const hollowVals = new Float32Array(crownTarget + 16);
  let crown = 0;
  for (let item = 0; item < lists.length; item++) {
    const L = lists[item];
    for (let o = 0, n = 0; o < L.length; o += LEAF_STRIDE, n++) {
      if ((n & 255) === 255) yield;
      if (!keep[item][o / LEAF_STRIDE]) continue;
      const id = L.data[o + 10];
      let T: number;
      if (id === input.farOuterId) T = trng.range(0.25, 0.6);
      else {
        leafCentre(L.data, o, item, tC);
        T = grid ? grid.transmittance(tC.x, tC.y, tC.z, toL, SIGMA_KEY * input.densityScale, 0.04, 0.045, 1.8) : 1;
        // a little per-leaf scatter (flecks of light, leaves turned into the shade)
        T = Math.min(1, T * trng.range(0.8, 1.2));
        if (id === input.farInnerId) T *= 0.6;
      }
      L.data[o + 16] = T;
      const jitter = id < nClusters ? 0.3 * clExitJitter[id] : erng.range(-0.08, 0.08);
      L.data[o + 19] = jitter + erng.range(-0.05, 0.05);
      if (id < nClusters) {
        tC.set(bases[id].x + L.data[o], bases[id].y + L.data[o + 1], bases[id].z + L.data[o + 2]);
        centroid[id].add(tC);
        clusterLeaves[id]++;
        if (crown < keyVals.length) {
          keyVals[crown] = T;
          visVals[crown] = visOf[item][o / LEAF_STRIDE];
          hollowVals[crown] = L.data[o + 21];
        }
        crown++;
      }
    }
  }
  for (let c = 0; c < nClusters; c++) {
    if (clusterLeaves[c] > 0) centroid[c].multiplyScalar(1 / clusterLeaves[c]);
    else centroid[c].copy(input.clusters[c].base);
  }

  mark("light");
  yield;

  // ---------------------------------------------------------------------------
  // pack: per item, drawn leaves only, nearest first (early depth rejection)
  // ---------------------------------------------------------------------------
  const data: Float32Array[] = [];
  const counts: number[] = [];
  for (let item = 0; item < lists.length; item++) {
    const L = lists[item];
    const k = keep[item];
    let m = 0;
    for (let i = 0; i < k.length; i++) if (k[i]) m++;
    // key = quantised (10 m − z) × 2^20 + index: ascending = nearest (largest z) first
    const order = new Float64Array(m);
    m = 0;
    for (let i = 0; i < k.length; i++) {
      if ((i & 4095) === 4095) yield;
      if (!k[i]) continue;
      const z = baseOf(L.data[i * LEAF_STRIDE + 10]).z + L.data[i * LEAF_STRIDE + 2];
      order[m++] = Math.round((10 - z) * 1e4) * 1048576 + i;
    }
    yield* sortSteps(order);
    const out = new Float32Array(m * LEAF_STRIDE);
    // bit-exact copy through 32-bit integer views (no view object per leaf)
    const from = new Uint32Array(L.data.buffer, L.data.byteOffset, L.length);
    const to = new Uint32Array(out.buffer);
    for (let j = 0; j < m; j++) {
      if ((j & 1023) === 1023) yield;
      const src = (order[j] % 1048576) * LEAF_STRIDE;
      const dst = j * LEAF_STRIDE;
      for (let c = 0; c < LEAF_STRIDE; c++) to[dst + c] = from[src + c];
    }
    data.push(out);
    counts.push(m);
  }

  mark("pack");
  // debug summaries: one pass over up to `crown` values each (cold code, a few ms)
  const total = counts.reduce((s, n) => s + n, 0);
  const shown = Math.min(crown, keyVals.length);
  yield;
  const keyT = summary(keyVals, shown);
  yield;
  const visibility = summary(visVals, shown);
  yield;
  const hollow = summary(hollowVals, shown);
  return {
    data,
    counts,
    total,
    crown,
    far,
    candidates,
    sprays,
    clusterCentroid: centroid,
    clusterLeaves,
    keyT,
    visibility,
    hollow,
    ms: now() - t0,
    phases: phases.join(", "),
  };
}

/**
 * The k-th smallest value (0-based) of `a` — the value `a.slice().sort()[k]` holds —
 * without a full sort, in pausable steps: Wirth's selection (Hoare partitioning around
 * the current k-th element), `a` is reordered in place. No NaN in `a`.
 */
function* kthSmallest(a: Float64Array, k: number): Steps<number> {
  let l = 0;
  let r = a.length - 1;
  while (l < r) {
    const x = a[k];
    let i = l;
    let j = r;
    let work = 0;
    do {
      while (a[i] < x) i++;
      while (x < a[j]) j--;
      if (i <= j) {
        const w = a[i];
        a[i] = a[j];
        a[j] = w;
        i++;
        j--;
      }
      if (++work === 16384) {
        // i, j, x are this generator's own: the partition resumes where it stopped
        work = 0;
        yield;
      }
    } while (i <= j);
    if (j < k) l = i;
    if (k < i) r = j;
    yield;
  }
  return a[k];
}

/**
 * `a.sort()` (ascending) in pausable steps: runs of 4096 sorted natively, then merged
 * bottom-up. The same array as the plain call for distinct values (the pack keys are).
 */
function* sortSteps(a: Float64Array): Steps<Float64Array> {
  const n = a.length;
  const RUN = 4096;
  for (let i = 0; i < n; i += RUN) {
    yield;
    a.subarray(i, Math.min(n, i + RUN)).sort();
  }
  let src: Float64Array = a;
  let dst: Float64Array = new Float64Array(n);
  for (let width = RUN; width < n; width *= 2) {
    let work = 0;
    for (let lo = 0; lo < n; lo += 2 * width) {
      const mid = Math.min(n, lo + width);
      const hi = Math.min(n, lo + 2 * width);
      let i = lo;
      let j = mid;
      let k = lo;
      while (i < mid && j < hi) {
        dst[k++] = src[i] <= src[j] ? src[i++] : src[j++];
        if (++work === 32768) {
          work = 0;
          yield;
        }
      }
      while (i < mid) dst[k++] = src[i++];
      while (j < hi) dst[k++] = src[j++];
    }
    const t = src;
    src = dst;
    dst = t;
    yield;
  }
  if (src !== a) a.set(src);
  return a;
}

/** Distance (m) from a point to a polyline of camera positions (Infinity without one). */
function nearDistance(path: Vector3[]): (p: Vector3) => number {
  if (!path.length) return () => Infinity;
  const ab = new Vector3();
  const ap = new Vector3();
  return (p: Vector3) => {
    let best = Infinity;
    for (let i = 0; i < path.length; i++) {
      const a = path[i];
      const b = path[Math.min(path.length - 1, i + 1)];
      ab.subVectors(b, a);
      ap.subVectors(p, a);
      const L2 = ab.lengthSq();
      const k = L2 > 1e-9 ? Math.min(1, Math.max(0, ap.dot(ab) / L2)) : 0;
      best = Math.min(best, ap.addScaledVector(ab, -k).length());
    }
    return best;
  };
}

/** 0–1 membership of a point in the hollow (1 = inside), as a function of position. */
function hollowMask(hollow: FoliageHollow | null, seed: number): (p: Vector3) => number {
  if (!hollow) return () => 0;
  const inv = hollow.quaternion.clone().invert();
  const tanV = Math.tan((hollow.fov * Math.PI) / 360);
  const local = new Vector3();
  const [cu, cv] = hollow.centre;
  const [rx, ry] = hollow.radii;
  return (p: Vector3) => {
    local.copy(p).sub(hollow.position).applyQuaternion(inv);
    const depth = -local.z;
    if (depth < 0.05) return 0;
    const u = 0.5 + (0.5 * local.x) / (depth * tanV * hollow.aspect);
    const v = 0.5 - (0.5 * local.y) / (depth * tanV);
    const d = Math.hypot((u - cu) / rx, (v - cv) / ry) + hollow.ragged * (fbm3(p.x * 1.4, p.y * 1.4, p.z * 1.4, seed + 77) - 0.5);
    const k = Math.min(1, Math.max(0, (1 + hollow.soft - d) / (2 * hollow.soft)));
    return k * k * (3 - 2 * k);
  };
}
