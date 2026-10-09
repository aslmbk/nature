/**
 * The warm local pool of the stone set: the key is a SpotLight along `key_stone`,
 * centred on the emblem, shaped by a code-made light cookie (`SpotLight.map`, projected
 * through the light's shadow frustum). Frame 14 is not a round pool: a broad plateau on
 * the emblem face out to its left edge, a short fall-off to the foot of the slab, a
 * longer one to the right, and a dark band above the emblem (the rock under the moss
 * strip). The shape's numbers live with the scene (StoneScene POOL_SHAPE).
 *
 * Cookie space: metres at the light's target, u along the light's right vector, v along
 * its up vector (for `key_stone`: u ≈ screen right, v ≈ screen up-right).
 * The cookie is HalfFloat (no banding in the long dark tail), built once, deterministic.
 */
import { ClampToEdgeWrapping, DataTexture, DataUtils, HalfFloatType, LinearFilter, RGBAFormat } from "three";

export interface PoolShape {
  /** Half extent of the cookie square at the target (m); the spot angle covers it. */
  extent: number;
  /** Pool centre (m, cookie space). */
  center: [number, number];
  /** Gaussian radii (m) towards −u, +u, −v, +v: exp(−(du/r)² − (dv/r)²). */
  radius: [number, number, number, number];
  /** Darker side: × (1 − strength · smoothstep(from, to, dir · d)), dir normalised. */
  cut: { dir: [number, number]; from: number; to: number; strength: number };
}

export function poolCookieValue(shape: PoolShape, u: number, v: number): number {
  const du = u - shape.center[0];
  const dv = v - shape.center[1];
  const a = du / (du < 0 ? shape.radius[0] : shape.radius[1]);
  const b = dv / (dv < 0 ? shape.radius[2] : shape.radius[3]);
  const g = Math.exp(-(a * a) - b * b);
  const c = shape.cut;
  const len = Math.hypot(c.dir[0], c.dir[1]) || 1;
  const along = (du * c.dir[0] + dv * c.dir[1]) / len;
  const s = Math.min(1, Math.max(0, (along - c.from) / Math.max(1e-6, c.to - c.from)));
  const cut = 1 - c.strength * s * s * (3 - 2 * s);
  // fade out before the edge of the square (no hard frustum edge)
  const r = Math.max(Math.abs(u), Math.abs(v)) / shape.extent;
  const edge = 1 - Math.min(1, Math.max(0, (r - 0.85) / 0.15));
  return g * cut * edge;
}

export function createPoolCookie(shape: PoolShape, size = 128): DataTexture {
  const data = new Uint16Array(size * size * 4);
  for (let j = 0; j < size; j++) {
    const v = (((j + 0.5) / size) * 2 - 1) * shape.extent;
    for (let i = 0; i < size; i++) {
      const u = (((i + 0.5) / size) * 2 - 1) * shape.extent;
      const h = DataUtils.toHalfFloat(poolCookieValue(shape, u, v));
      const k = (j * size + i) * 4;
      data[k] = h;
      data[k + 1] = h;
      data[k + 2] = h;
      data[k + 3] = DataUtils.toHalfFloat(1);
    }
  }
  const tex = new DataTexture(data, size, size, RGBAFormat, HalfFloatType);
  tex.name = "stone_pool_cookie";
  tex.magFilter = LinearFilter;
  tex.minFilter = LinearFilter;
  tex.wrapS = ClampToEdgeWrapping;
  tex.wrapT = ClampToEdgeWrapping;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}
