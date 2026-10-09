/**
 * Where the canyon's plants grow beyond the GLB's own mask (scatter only).
 *
 * canyon.glb marks plant density (COLOR_0.R) only on the upward faces of the ledges
 * inside its plant zones: 6.5 % of the right face's vertices, most of them on ledge
 * tops above eye level that face away from the lens. Frame 06 shows the upper part of
 * the right face overgrown in patches — leaves along the cracks and hollows, on the
 * flanks next to them, a band along its upper-left outline, the clump at the right
 * border — and sprigs dotted along the lit left edge. So inside the same plant zones
 * (screen rectangles of `cam_canyon_main`, as in build_canyon.py ZONES) the scene adds
 * growth that prefers the cracks and hollows (low baked AO, COLOR_0.B), keeps off the
 * undersides and comes in noise patches — except in the zones marked solid (the dense
 * clumps of frame 06), where it covers the surface:
 *
 *   R' = max(R, zone(u, v) · (flank + (1 − flank) · cavity(B)) · facing(n.y) · mix(patch(p), 1, solid(u, v)))
 *
 * Much of a fluted face is hidden from the lens (the far walls of the cracks, the backs
 * of the slabs): plants there would only use up the budget. A depth buffer of the
 * mesh's own triangles seen from the main camera gives every vertex a visibility, and
 * the whole density (the GLB's and the added) is scaled by `floor + (1 − floor) · vis`.
 *
 * The result is a scatter-only geometry: it shares position / normal / index with the
 * GLB geometry (registry-owned, never mutated) and carries its own COLOR_0; it is never
 * added to the scene or rendered. Deterministic (noise seeded from the scene seed).
 *
 * `growthMeshSteps` is the build as pausable steps (vegetation/slices.ts: checkpoints
 * between vertices and between triangles of the depth buffer, typically < 0.5 ms of work
 * between two), so the canyon set can prepare in ≈ 8 ms slices while another episode
 * renders; `growthMesh` runs the same steps in one go. All scratch state is per call.
 */
import { BufferAttribute, BufferGeometry, Mesh, Quaternion, Vector3, type InterleavedBufferAttribute } from "three";
import type { CameraPose } from "../../CameraRig";
import { REFERENCE_ASPECT } from "../../SceneConfig";
import { fbm3 } from "../../vegetation/noise";
import { runSync, type Steps } from "../../vegetation/slices";

/**
 * Screen rectangle of the main camera (u right, v down, 0–1) with a weight and an
 * optional `solid` 0–1 (how far the patch noise is ignored: a clump covering the zone).
 */
export type GrowthZone = [u0: number, v0: number, u1: number, v1: number, weight: number, solid?: number];

export interface GrowthOptions {
  zones: GrowthZone[];
  /** Soft edge of the zones (screen units). */
  edge: number;
  /** Share of growth on open flanks (B high) relative to the cracks (B low). */
  flank: number;
  /** Baked AO range: cavity weight goes 0 → 1 from `cavity[0]` down to `cavity[1]`. */
  cavity: [number, number];
  /** Patch noise: cells per metre, threshold, softness. */
  patch: { scale: number; threshold: number; softness: number };
  /** Overall strength of the added growth (R units). */
  strength: number;
  /** Density kept where the surface is hidden from the main camera (0–1). */
  hiddenFloor: number;
  seed: number;
}

/** Self-occlusion depth buffer (cells across the main camera's frame). */
const ZW = 480;
const ZH = 340;
/** A vertex this far (m) behind the front-most surface of its cell starts / ends being hidden. */
const HIDE_NEAR = 0.03;
const HIDE_FAR = 0.12;

function smooth(e0: number, e1: number, x: number): number {
  const k = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return k * k * (3 - 2 * k);
}

/** Zone weight and solidity at (u, v), written into `out` (the caller's own scratch). */
function zoneWeight(u: number, v: number, zones: GrowthZone[], e: number, out: { weight: number; solid: number }): { weight: number; solid: number } {
  let w = 0;
  let s = 0;
  for (const [u0, v0, u1, v1, k, solid = 0] of zones) {
    const a = smooth(u0 - e, u0 + e, u) * (1 - smooth(u1 - e, u1 + e, u)) * smooth(v0 - e, v0 + e, v) * (1 - smooth(v1 - e, v1 + e, v));
    w = Math.max(w, a * k);
    s = Math.max(s, a * solid);
  }
  out.weight = w;
  out.solid = s;
  return out;
}

export interface GrowthResult {
  mesh: Mesh;
  /** Vertices whose density the growth raised, and vertices hidden from the main camera. */
  added: number;
  hidden: number;
}

/**
 * Scatter-only mesh of `src` (world transform copied) with the extended density in
 * COLOR_0.R; G (length), B (AO) and A are kept. `view` is the main camera pose.
 */
export function growthMesh(src: Mesh, view: CameraPose, o: GrowthOptions): GrowthResult {
  return runSync(growthMeshSteps(src, view, o));
}

/** `growthMesh` as pausable steps (vegetation/slices.ts): identical result. */
export function* growthMeshSteps(src: Mesh, view: CameraPose, o: GrowthOptions): Steps<GrowthResult> {
  const tmpP = new Vector3();
  const zone = { weight: 0, solid: 0 };
  const g = src.geometry;
  const pos = g.getAttribute("position") as BufferAttribute | InterleavedBufferAttribute;
  const nrm = g.getAttribute("normal") as BufferAttribute | InterleavedBufferAttribute | undefined;
  const col = g.getAttribute("color") as BufferAttribute | InterleavedBufferAttribute | undefined;
  const n = pos.count;
  const out = new Float32Array(n * 4);
  src.updateWorldMatrix(true, false);
  const inv = new Quaternion().copy(view.quaternion).invert();
  const tanV = Math.tan((view.fov * Math.PI) / 360);
  const tanH = tanV * REFERENCE_ASPECT;
  const h = 0.5 * o.patch.softness;

  // every vertex in the main camera: screen u, v (0–1) and view depth (m; ≤ 0 behind)
  const W = new Float32Array(n * 3);
  const U = new Float32Array(n);
  const V = new Float32Array(n);
  const Z = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    if ((i & 2047) === 2047) yield;
    tmpP.fromBufferAttribute(pos, i).applyMatrix4(src.matrixWorld);
    W[i * 3] = tmpP.x;
    W[i * 3 + 1] = tmpP.y;
    W[i * 3 + 2] = tmpP.z;
    tmpP.sub(view.position).applyQuaternion(inv);
    const z = -tmpP.z;
    Z[i] = z;
    if (z > 0.05) {
      U[i] = 0.5 + 0.5 * (tmpP.x / (z * tanH));
      V[i] = 0.5 - 0.5 * (tmpP.y / (z * tanV));
    }
  }
  const depth = yield* selfDepthSteps(g.index, n, U, V, Z);

  let added = 0;
  let hidden = 0;
  for (let i = 0; i < n; i++) {
    if ((i & 511) === 511) yield;
    const r = col ? col.getX(i) : 0;
    const gch = col ? col.getY(i) : 0.5;
    const b = col ? col.getZ(i) : 1;
    const a = col && col.itemSize > 3 ? col.getW(i) : 1;
    const wx = W[i * 3];
    const wy = W[i * 3 + 1];
    const wz = W[i * 3 + 2];
    const z = Z[i];
    let extra = 0;
    let vis = 1;
    if (z > 0.05) {
      const u = U[i];
      const v = V[i];
      const px = Math.floor(u * ZW);
      const py = Math.floor(v * ZH);
      if (px >= 0 && px < ZW && py >= 0 && py < ZH) vis = 1 - smooth(HIDE_NEAR, HIDE_FAR, z - depth[py * ZW + px]);
      zoneWeight(u, v, o.zones, o.edge, zone);
      if (zone.weight > 0) {
        const ny = nrm ? nrm.getY(i) : 0;
        const facing = smooth(-0.55, -0.1, ny);
        const cav = smooth(o.cavity[0], o.cavity[1], b);
        const patch = smooth(o.patch.threshold - h, o.patch.threshold + h, fbm3(wx * o.patch.scale, wy * o.patch.scale, wz * o.patch.scale, o.seed));
        extra = o.strength * zone.weight * (o.flank + (1 - o.flank) * cav) * facing * (patch + (1 - patch) * zone.solid);
      }
    }
    if (extra > r) added++;
    if (vis < 0.5) hidden++;
    out[i * 4] = Math.max(r, extra) * (o.hiddenFloor + (1 - o.hiddenFloor) * vis);
    out[i * 4 + 1] = gch;
    out[i * 4 + 2] = b;
    out[i * 4 + 3] = a;
  }
  const geo = new BufferGeometry();
  geo.name = `${g.name || src.name}:growth`;
  geo.setAttribute("position", pos);
  if (nrm) geo.setAttribute("normal", nrm);
  geo.setAttribute("color", new BufferAttribute(out, 4));
  if (g.index) geo.setIndex(g.index);
  const mesh = new Mesh(geo);
  mesh.name = src.name;
  mesh.matrixAutoUpdate = false;
  mesh.matrix.copy(src.matrixWorld);
  mesh.matrixWorld.copy(src.matrixWorld);
  return { mesh, added, hidden };
}

/**
 * Front-most view depth per cell of the mesh's triangles (ZW × ZH cells over the frame,
 * +Infinity where nothing): 1/z interpolated across each triangle (perspective-correct).
 * Triangles with a vertex behind the near plane are skipped. Pausable steps: a checkpoint
 * whenever ≈ 64k cells (or 1024 triangles) have been visited since the last one.
 */
function* selfDepthSteps(index: BufferAttribute | null, n: number, U: Float32Array, V: Float32Array, Z: Float32Array): Steps<Float32Array> {
  const depth = new Float32Array(ZW * ZH).fill(Infinity);
  const tris = index ? index.count / 3 : n / 3;
  let work = 0;
  for (let t = 0; t < tris; t++) {
    if (work >= 65536) {
      work = 0;
      yield;
    }
    work += 64;
    const ia = index ? index.getX(t * 3) : t * 3;
    const ib = index ? index.getX(t * 3 + 1) : t * 3 + 1;
    const ic = index ? index.getX(t * 3 + 2) : t * 3 + 2;
    const za = Z[ia];
    const zb = Z[ib];
    const zc = Z[ic];
    if (za <= 0.05 || zb <= 0.05 || zc <= 0.05) continue;
    const ax = U[ia] * ZW;
    const ay = V[ia] * ZH;
    const bx = U[ib] * ZW;
    const by = V[ib] * ZH;
    const cx = U[ic] * ZW;
    const cy = V[ic] * ZH;
    const x0 = Math.max(0, Math.floor(Math.min(ax, bx, cx)));
    const x1 = Math.min(ZW - 1, Math.ceil(Math.max(ax, bx, cx)));
    const y0 = Math.max(0, Math.floor(Math.min(ay, by, cy)));
    const y1 = Math.min(ZH - 1, Math.ceil(Math.max(ay, by, cy)));
    if (x0 > x1 || y0 > y1) continue;
    work += (x1 - x0 + 1) * (y1 - y0 + 1);
    const area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    if (Math.abs(area) < 1e-9) continue;
    const ia2 = 1 / area;
    const wa = 1 / za;
    const wb = 1 / zb;
    const wc = 1 / zc;
    for (let y = y0; y <= y1; y++) {
      const py = y + 0.5;
      for (let x = x0; x <= x1; x++) {
        const px = x + 0.5;
        const l0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) * ia2;
        const l1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) * ia2;
        const l2 = 1 - l0 - l1;
        if (l0 < 0 || l1 < 0 || l2 < 0) continue;
        const z = 1 / (l0 * wa + l1 * wb + l2 * wc);
        const k = y * ZW + x;
        if (z < depth[k]) depth[k] = z;
      }
    }
  }
  return depth;
}
