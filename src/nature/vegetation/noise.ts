/**
 * Tiny deterministic noise for build-time decisions (lean fields, clusters, patches).
 * Pure functions of their integer seed: identical on every machine.
 */

/** 32-bit integer hash of three lattice coordinates and a seed → [0, 1). */
export function hash3i(x: number, y: number, z: number, seed: number): number {
  let h = Math.imul(x | 0, 0x8da6b343) ^ Math.imul(y | 0, 0xd8163841) ^ Math.imul(z | 0, 0xcb1ab31f) ^ Math.imul(seed | 0, 0x165667b1);
  h = Math.imul(h ^ (h >>> 15), 0x2c1b3c6d);
  h = Math.imul(h ^ (h >>> 12), 0x297a2d39);
  h ^= h >>> 15;
  return (h >>> 0) / 4294967296;
}

/** Smooth 3D value noise in [0, 1]. */
export function valueNoise3(x: number, y: number, z: number, seed: number): number {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const zi = Math.floor(z);
  const fx = x - xi;
  const fy = y - yi;
  const fz = z - zi;
  const u = fx * fx * (3 - 2 * fx);
  const v = fy * fy * (3 - 2 * fy);
  const w = fz * fz * (3 - 2 * fz);
  const l = (a: number, b: number, k: number) => a + (b - a) * k;
  const h = (dx: number, dy: number, dz: number) => hash3i(xi + dx, yi + dy, zi + dz, seed);
  return l(
    l(l(h(0, 0, 0), h(1, 0, 0), u), l(h(0, 1, 0), h(1, 1, 0), u), v),
    l(l(h(0, 0, 1), h(1, 0, 1), u), l(h(0, 1, 1), h(1, 1, 1), u), v),
    w,
  );
}

/** Two-octave value noise in [0, 1]. */
export function fbm3(x: number, y: number, z: number, seed: number): number {
  return valueNoise3(x, y, z, seed) * 0.65 + valueNoise3(x * 2.07 + 11.3, y * 2.07 + 5.1, z * 2.07 + 2.9, seed + 1) * 0.35;
}
