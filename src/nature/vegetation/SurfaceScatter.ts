/**
 * Stable distribution of plants over mesh surfaces.
 *
 * `SurfaceSet` gathers the triangles of some meshes (positions / normals in the space of
 * a chosen parent object, `COLOR_0` as floats: R density, G length, B baked AO) and
 * computes per-triangle areas. A layer then builds a per-triangle weight
 * (area × density mask × view importance × anything else) and draws samples from it:
 * stratified along the cumulative weight (fewer clumps and holes than independent
 * draws), uniform barycentric inside each triangle, every number from the given rng.
 * The same inputs always give the same samples.
 *
 * View importance (`viewImportance`) keeps the budget where the story's cameras look:
 * for each triangle, the best of the given camera poses counts — inside the frustum
 * (plus a margin), facing the camera (grazing faces still count, their plants stick
 * out over the silhouette), within a useful distance. Plants are scattered once at
 * build time; nothing here runs per frame.
 *
 * The heavy parts also exist as pausable steps (`SurfaceSet.steps`, the `*Steps` methods,
 * `spatialChunksSteps`; slices.ts) with exactly the results of the plain calls.
 */
import { MathUtils, Matrix3, Matrix4, Quaternion, Vector3, type BufferAttribute, type InterleavedBufferAttribute, type Mesh, type Object3D } from "three";
import type { Rng } from "../core/rng";
import { runSync, type Steps } from "./slices";

type Attr = BufferAttribute | InterleavedBufferAttribute;

/** A camera pose the scatter should care about. */
export interface ScatterView {
  position: Vector3;
  quaternion: Quaternion;
  /** Vertical fov in degrees. */
  fov: number;
  aspect: number;
  /** Relative importance of this view (default 1). */
  weight?: number;
  /** Extra frustum margin as a fraction of the half extents (default 0.12). */
  margin?: number;
  /** Beyond this distance the importance fades out (default ∞). */
  maxDistance?: number;
  /** Label of the story beat this view belongs to (layers may scatter for some tags only). */
  tag?: string;
}

export interface SurfaceSamples {
  count: number;
  /** xyz per sample, in the space of the set. */
  position: Float32Array;
  /** Unit surface normal per sample. */
  normal: Float32Array;
  /** COLOR_0 per sample (R density, G length, B AO, A). */
  color: Float32Array;
  /** Source triangle per sample. */
  tri: Uint32Array;
}

export interface ViewImportanceOptions {
  /** dot(normal, toCamera) below which a face is ignored (default −0.3: grazing back faces still count). */
  facingMin?: number;
  /** dot(normal, toCamera) from which a face counts fully (default 0.15). */
  facingFull?: number;
}

/** The triangles of a SurfaceSet (see its fields). */
export interface SurfaceSetData {
  count: number;
  positions: Float32Array;
  normals: Float32Array;
  colors: Float32Array;
  areas: Float32Array;
  meshOf: Uint16Array;
}

const tmpQ = new Quaternion();

export class SurfaceSet {
  /** Number of triangles. */
  readonly count: number;
  /** 9 floats per triangle (3 vertices). */
  readonly positions: Float32Array;
  /** 9 floats per triangle (unit vertex normals). */
  readonly normals: Float32Array;
  /** 12 floats per triangle (RGBA per vertex). */
  readonly colors: Float32Array;
  /** Area (m², in the space of the set) per triangle. */
  readonly areas: Float32Array;
  /** Source mesh index per triangle. */
  readonly meshOf: Uint16Array;
  readonly meshes: readonly Mesh[];

  /**
   * @param meshes source meshes (their world transforms are respected)
   * @param space  the object the plants will be parented to (null = world)
   * @param data   the gathered triangles (`SurfaceSet.steps` passes them); gathered now when omitted
   */
  constructor(meshes: readonly Mesh[], space: Object3D | null = null, data?: SurfaceSetData) {
    this.meshes = meshes;
    const d = data ?? runSync(gatherTriangles(meshes, space));
    this.count = d.count;
    this.positions = d.positions;
    this.normals = d.normals;
    this.colors = d.colors;
    this.areas = d.areas;
    this.meshOf = d.meshOf;
  }

  /** The constructor as pausable steps (slices.ts): `const set = yield* SurfaceSet.steps(meshes, space)`. */
  static *steps(meshes: readonly Mesh[], space: Object3D | null = null): Steps<SurfaceSet> {
    const data = yield* gatherTriangles(meshes, space);
    return new SurfaceSet(meshes, space, data);
  }

  /** Mean of a COLOR_0 channel (0 R, 1 G, 2 B) over each triangle's vertices. */
  channel(c: 0 | 1 | 2 | 3): Float32Array {
    const out = new Float32Array(this.count);
    for (let t = 0; t < this.count; t++) {
      const o = t * 12 + c;
      out[t] = (this.colors[o] + this.colors[o + 4] + this.colors[o + 8]) / 3;
    }
    return out;
  }

  /** Centroid of triangle t. */
  centroid(t: number, out: Vector3): Vector3 {
    const p = this.positions;
    const o = t * 9;
    return out.set((p[o] + p[o + 3] + p[o + 6]) / 3, (p[o + 1] + p[o + 4] + p[o + 7]) / 3, (p[o + 2] + p[o + 5] + p[o + 8]) / 3);
  }

  /** Mean unit normal of triangle t. */
  meanNormal(t: number, out: Vector3): Vector3 {
    const n = this.normals;
    const o = t * 9;
    out.set(n[o] + n[o + 3] + n[o + 6], n[o + 1] + n[o + 4] + n[o + 7], n[o + 2] + n[o + 5] + n[o + 8]);
    const l = out.length();
    return l > 1e-8 ? out.multiplyScalar(1 / l) : out.set(0, 1, 0);
  }

  /**
   * Per-triangle importance 0–1: max over the views of
   * weight × inside-frustum × facing × distance fade. `views` are in the space of the set.
   */
  viewImportance(views: readonly ScatterView[], opts: ViewImportanceOptions = {}): Float32Array {
    return runSync(this.viewImportanceSteps(views, opts));
  }

  /** `viewImportance` as pausable steps (slices.ts). */
  *viewImportanceSteps(views: readonly ScatterView[], opts: ViewImportanceOptions = {}): Steps<Float32Array> {
    const out = new Float32Array(this.count);
    const fMin = opts.facingMin ?? -0.3;
    const fFull = opts.facingFull ?? 0.15;
    const prepared = views.map((v) => {
      const inv = tmpQ.copy(v.quaternion).invert().clone();
      const tanV = Math.tan(MathUtils.degToRad(v.fov) / 2);
      const m = 1 + (v.margin ?? 0.12);
      return { v, inv, tanV: tanV * m, tanH: tanV * v.aspect * m, weight: v.weight ?? 1, maxD: v.maxDistance ?? Infinity };
    });
    const c = new Vector3();
    const n = new Vector3();
    const local = new Vector3();
    const toCam = new Vector3();
    for (let t = 0; t < this.count; t++) {
      this.centroid(t, c);
      this.meanNormal(t, n);
      let best = 0;
      for (const p of prepared) {
        if (p.weight <= best) continue;
        local.copy(c).sub(p.v.position).applyQuaternion(p.inv);
        const z = -local.z;
        if (z <= 0.02) continue;
        if (Math.abs(local.x) > z * p.tanH || Math.abs(local.y) > z * p.tanV) continue;
        toCam.copy(p.v.position).sub(c);
        const d = toCam.length();
        toCam.multiplyScalar(1 / Math.max(d, 1e-6));
        const facing = smooth(fMin, fFull, n.dot(toCam));
        if (facing <= 0) continue;
        const fade = Number.isFinite(p.maxD) ? 1 - smooth(p.maxD * 0.7, p.maxD, d) : 1;
        best = Math.max(best, p.weight * facing * fade);
      }
      out[t] = best;
      if ((t & 1023) === 1023) yield;
    }
    return out;
  }

  /**
   * Per-triangle silhouette factor 0–1 for the given views: 1 where the face is seen
   * edge-on (plants there stick out over the outline), 0 where it faces the camera
   * (|cos| between the normal and the view ray ≥ `width`, default 0.55).
   */
  rimFactor(views: readonly ScatterView[], width = 0.55): Float32Array {
    return runSync(this.rimFactorSteps(views, width));
  }

  /** `rimFactor` as pausable steps (slices.ts). */
  *rimFactorSteps(views: readonly ScatterView[], width = 0.55): Steps<Float32Array> {
    const out = new Float32Array(this.count);
    const c = new Vector3();
    const n = new Vector3();
    const toCam = new Vector3();
    const w = Math.max(1e-3, width);
    for (let t = 0; t < this.count; t++) {
      this.centroid(t, c);
      this.meanNormal(t, n);
      let best = 0;
      for (const v of views) {
        toCam.copy(v.position).sub(c).normalize();
        best = Math.max(best, 1 - Math.min(1, Math.abs(n.dot(toCam)) / w));
      }
      out[t] = best;
      if ((t & 2047) === 2047) yield;
    }
    return out;
  }

  /**
   * Draw `count` samples with probability ∝ weights[t] (callers include the area).
   * Stratified along the cumulative weight; deterministic for a given rng.
   */
  sample(rng: Rng, count: number, weights: Float32Array): SurfaceSamples {
    return runSync(this.sampleSteps(rng, count, weights));
  }

  /** `sample` as pausable steps (slices.ts). */
  *sampleSteps(rng: Rng, count: number, weights: Float32Array): Steps<SurfaceSamples> {
    const n = this.count;
    const cdf = new Float64Array(n);
    let acc = 0;
    for (let t = 0; t < n; t++) {
      const w = weights[t];
      acc += w > 0 && Number.isFinite(w) ? w : 0;
      cdf[t] = acc;
      if ((t & 16383) === 16383) yield;
    }
    const total = acc;
    const k = total > 0 ? Math.max(0, Math.floor(count)) : 0;
    const out: SurfaceSamples = {
      count: k,
      position: new Float32Array(k * 3),
      normal: new Float32Array(k * 3),
      color: new Float32Array(k * 4),
      tri: new Uint32Array(k),
    };
    const P = this.positions;
    const N = this.normals;
    const C = this.colors;
    let lo = 0;
    for (let i = 0; i < k; i++) {
      const target = ((i + rng.next()) / k) * total;
      // targets increase monotonically: search forward from the last hit
      let a = lo;
      let b = n - 1;
      while (a < b) {
        const mid = (a + b) >> 1;
        if (cdf[mid] < target) a = mid + 1;
        else b = mid;
      }
      lo = a;
      const t = a;
      // uniform barycentric
      let u = rng.next();
      let v = rng.next();
      if (u + v > 1) {
        u = 1 - u;
        v = 1 - v;
      }
      const w0 = 1 - u - v;
      const o = t * 9;
      for (let d = 0; d < 3; d++) {
        out.position[i * 3 + d] = P[o + d] * w0 + P[o + 3 + d] * u + P[o + 6 + d] * v;
      }
      let nx = N[o] * w0 + N[o + 3] * u + N[o + 6] * v;
      let ny = N[o + 1] * w0 + N[o + 4] * u + N[o + 7] * v;
      let nz = N[o + 2] * w0 + N[o + 5] * u + N[o + 8] * v;
      const l = Math.hypot(nx, ny, nz) || 1;
      nx /= l;
      ny /= l;
      nz /= l;
      out.normal[i * 3] = nx;
      out.normal[i * 3 + 1] = ny;
      out.normal[i * 3 + 2] = nz;
      const co = t * 12;
      for (let d = 0; d < 4; d++) out.color[i * 4 + d] = C[co + d] * w0 + C[co + 4 + d] * u + C[co + 8 + d] * v;
      out.tri[i] = t;
      if ((i & 2047) === 2047) yield;
    }
    return out;
  }
}

/**
 * The triangles of `meshes` in the space of `space` (checkpoint every 1024 triangles).
 * Its scratch vectors / matrices are its own: it may pause while another build runs.
 */
function* gatherTriangles(meshes: readonly Mesh[], space: Object3D | null): Steps<SurfaceSetData> {
  let total = 0;
  for (const m of meshes) {
    const g = m.geometry;
    total += g.index ? g.index.count / 3 : g.getAttribute("position").count / 3;
  }
  const positions = new Float32Array(total * 9);
  const normals = new Float32Array(total * 9);
  const colors = new Float32Array(total * 12);
  const areas = new Float32Array(total);
  const meshOf = new Uint16Array(total);
  const inv = new Matrix4();
  const toSpace = new Matrix4();
  const normalMatrix = new Matrix3();
  const a = new Vector3();
  const b = new Vector3();
  const c = new Vector3();
  const nv = new Vector3();

  if (space) {
    space.updateWorldMatrix(true, false);
    inv.copy(space.matrixWorld).invert();
  } else inv.identity();

  let t = 0;
  for (let mi = 0; mi < meshes.length; mi++) {
    const mesh = meshes[mi];
    mesh.updateWorldMatrix(true, false);
    toSpace.multiplyMatrices(inv, mesh.matrixWorld);
    normalMatrix.getNormalMatrix(toSpace);
    const g = mesh.geometry;
    const pos = g.getAttribute("position") as Attr;
    const nrm = g.getAttribute("normal") as Attr | undefined;
    const col = g.getAttribute("color") as Attr | undefined;
    const index = g.index;
    const triCount = index ? index.count / 3 : pos.count / 3;
    for (let i = 0; i < triCount; i++, t++) {
      meshOf[t] = mi;
      for (let k = 0; k < 3; k++) {
        const v = index ? index.getX(i * 3 + k) : i * 3 + k;
        const o = t * 9 + k * 3;
        a.fromBufferAttribute(pos, v).applyMatrix4(toSpace);
        positions[o] = a.x;
        positions[o + 1] = a.y;
        positions[o + 2] = a.z;
        if (nrm) nv.fromBufferAttribute(nrm, v).applyMatrix3(normalMatrix).normalize();
        else nv.set(0, 0, 0);
        normals[o] = nv.x;
        normals[o + 1] = nv.y;
        normals[o + 2] = nv.z;
        const co = t * 12 + k * 4;
        if (col) {
          colors[co] = col.getX(v);
          colors[co + 1] = col.getY(v);
          colors[co + 2] = col.getZ(v);
          colors[co + 3] = col.itemSize > 3 ? col.getW(v) : 1;
        } else {
          colors[co] = 1;
          colors[co + 1] = 0.5;
          colors[co + 2] = 1;
          colors[co + 3] = 1;
        }
      }
      const o = t * 9;
      a.set(positions[o], positions[o + 1], positions[o + 2]);
      b.set(positions[o + 3], positions[o + 4], positions[o + 5]).sub(a);
      c.set(positions[o + 6], positions[o + 7], positions[o + 8]).sub(a);
      areas[t] = b.cross(c).length() * 0.5;
      if (!nrm) {
        // flat normal when the mesh has none
        b.normalize();
        for (let k = 0; k < 3; k++) {
          normals[o + k * 3] = b.x;
          normals[o + k * 3 + 1] = b.y;
          normals[o + k * 3 + 2] = b.z;
        }
      }
      if ((t & 1023) === 1023) yield;
    }
  }
  return { count: total, positions, normals, colors, areas, meshOf };
}

function smooth(e0: number, e1: number, x: number): number {
  const k = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return k * k * (3 - 2 * k);
}

/**
 * Split points into ≤ `maxChunks` spatially compact groups (median splits along the
 * longest axis). Returns index lists; deterministic.
 */
export function spatialChunks(positions: Float32Array, count: number, maxChunks: number): Uint32Array[] {
  return runSync(spatialChunksSteps(positions, count, maxChunks));
}

/** `spatialChunks` as pausable steps (slices.ts). */
export function* spatialChunksSteps(positions: Float32Array, count: number, maxChunks: number): Steps<Uint32Array[]> {
  const all = new Uint32Array(count);
  for (let i = 0; i < count; i++) all[i] = i;
  const out: Uint32Array[] = [];
  const target = Math.max(1, Math.ceil(count / Math.max(1, maxChunks)));
  const split = function* (idx: Uint32Array, depth: number): Steps<void> {
    if (idx.length <= target || depth > 12) {
      if (idx.length) out.push(idx);
      return;
    }
    let minX = Infinity, minY = Infinity, minZ = Infinity, maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (const i of idx) {
      const x = positions[i * 3], y = positions[i * 3 + 1], z = positions[i * 3 + 2];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      if (z < minZ) minZ = z;
      if (z > maxZ) maxZ = z;
    }
    const ex = maxX - minX, ey = maxY - minY, ez = maxZ - minZ;
    const axis = ex >= ey && ex >= ez ? 0 : ey >= ez ? 1 : 2;
    const sorted = yield* sortAlongAxis(idx, positions, axis);
    // split so that the left part holds a whole number of target-sized chunks
    const leftChunks = Math.max(1, Math.round(Math.ceil(idx.length / target) / 2));
    const mid = Math.min(sorted.length - 1, leftChunks * target);
    yield* split(sorted.slice(0, mid), depth + 1);
    yield* split(sorted.slice(mid), depth + 1);
  };
  yield* split(all, 0);
  return out;
}

/**
 * `idx` ordered by (positions[i * 3 + axis], i): the order of the comparator
 * `(a, b) => key(a) - key(b) || a - b`, in which no two indices tie, so the result equals
 * `Array.prototype.sort` with it. Insertion-sorted runs of 32, then bottom-up merges on
 * typed arrays, pausable between blocks of merges.
 */
function* sortAlongAxis(idx: Uint32Array, positions: Float32Array, axis: number): Steps<Uint32Array> {
  const n = idx.length;
  let ids = idx.slice();
  let keys = new Float64Array(n);
  for (let i = 0; i < n; i++) keys[i] = positions[ids[i] * 3 + axis];
  sortRuns(ids, keys, SORT_RUN);
  yield;
  let ids2 = new Uint32Array(n);
  let keys2 = new Float64Array(n);
  for (let width = SORT_RUN; width < n; width *= 2) {
    const block = Math.max(2 * width, 32768);
    for (let lo = 0; lo < n; lo += block) {
      mergeRuns(ids, keys, ids2, keys2, width, lo, Math.min(n, lo + block));
      yield;
    }
    const ti = ids;
    ids = ids2;
    ids2 = ti;
    const tk = keys;
    keys = keys2;
    keys2 = tk;
  }
  return ids;
}

const SORT_RUN = 32;

/** b orders strictly before a: smaller key, or the same key (incl. ±0, NaN difference) and a smaller index. */
function before(kb: number, ib: number, ka: number, ia: number): boolean {
  const d = kb - ka;
  return d < 0 || (!(d > 0) && ib < ia);
}

/** Insertion sort of every run of `run` elements. */
function sortRuns(ids: Uint32Array, keys: Float64Array, run: number): void {
  const n = ids.length;
  for (let lo = 0; lo < n; lo += run) {
    const hi = Math.min(lo + run, n);
    for (let a = lo + 1; a < hi; a++) {
      const ka = keys[a];
      const ia = ids[a];
      let b = a - 1;
      while (b >= lo && before(ka, ia, keys[b], ids[b])) {
        keys[b + 1] = keys[b];
        ids[b + 1] = ids[b];
        b--;
      }
      keys[b + 1] = ka;
      ids[b + 1] = ia;
    }
  }
}

/** Merge the sorted runs of `width` in [lo0, lo1) pairwise into ids2 / keys2. */
function mergeRuns(ids: Uint32Array, keys: Float64Array, ids2: Uint32Array, keys2: Float64Array, width: number, lo0: number, lo1: number): void {
  const n = ids.length;
  for (let lo = lo0; lo < lo1; lo += 2 * width) {
    const mid = Math.min(lo + width, n);
    const hi = Math.min(lo + 2 * width, n);
    let i = lo;
    let j = mid;
    let k = lo;
    while (i < mid && j < hi) {
      if (before(keys[j], ids[j], keys[i], ids[i])) {
        ids2[k] = ids[j];
        keys2[k++] = keys[j++];
      } else {
        ids2[k] = ids[i];
        keys2[k++] = keys[i++];
      }
    }
    while (i < mid) {
      ids2[k] = ids[i];
      keys2[k++] = keys[i++];
    }
    while (j < hi) {
      ids2[k] = ids[j];
      keys2[k++] = keys[j++];
    }
  }
}
