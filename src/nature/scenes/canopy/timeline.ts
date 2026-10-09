/**
 * Canopy story timeline (video seconds). Every function here is pure in `t` (story) —
 * the reveal, the camera / focus track and the exit are re-evaluated from scratch each
 * frame, so scrolling back closes the crown the same way it opened.
 *
 * Measured on the reference (motion/03, clip sheet 10, frames 15–17, motion/04, clip 11):
 *  - growth 43.4 → 44.3 s: the crown's lower edge stays near 0.97 of the frame height
 *    while its top rises 0.87 → 0.47 → 0.34 → 0.18 → 0.11 (43.4 / 43.6 / 43.8 / 44.0 /
 *    44.3 s) and its width grows 0.14 → 0.31 → 0.45 → 0.54 → 0.58 of the frame width.
 *    Without the stone → canopy slide (follow 0.4: the canopy picture still sits 0.15 /
 *    0.06 / 0.02 fh low at 43.4 / 43.6 / 43.8 s) that is a crown scaled by ≈ 0.27 / 0.59 /
 *    0.77 / 0.93 / 1 about its bottom centre: `crownScale`. The camera holds
 *    `cam_canopy_in`; on top of that the clusters scale in from their bases, inner and
 *    lower ones first (`reveal_order`, quick overlapping windows: the young crown bubbles
 *    up rather than appearing whole);
 *  - exit 52.4 → 53.9 s (mode `over`): the leaves darken from ≈ 52.6 s, the dark middle
 *    widens from ≈ 52.9 s, by 53.2 s only the left / right edges and the top corners hold
 *    leaves, at 53.5 s thin dark strips at the left / right edges, by 53.8 s nothing.
 */
import { saturate, smoothstep } from "../../core/math";

export const CANOPY_T = {
  /** reveal: the first cluster (reveal_order 0) starts here … */
  revealStart: 43.05,
  /** … the last one (reveal_order 1) this much later … */
  revealSpread: 0.35,
  /** … and each one takes this long to grow from its base. */
  revealDuration: 0.38,
  /** the whole crown grows about its bottom centre (outQuad) */
  scaleStart: 43.12,
  scaleDuration: 1.18,
  /** dim leaves on the far shells around the crown */
  farStart: 43.7,
  farEnd: 44.5,
  /** camera keys (cam_* video_time_s extras) */
  in: 44.3,
  main: 46.5,
  out: 48.5,
  closeMain: 50.5,
  closeOut: 52.8,
  /** exit (canopyClose → finale, mode `over`) */
  exitStart: 52.42,
  exitEnd: 53.8,
  /** leaves darken while the middle opens */
  dimStart: 52.5,
  dimEnd: 53.15,
} as const;

/** 0 → 1 progress of one cluster's scale-in window. */
export function clusterGrow(t: number, order: number): number {
  const start = CANOPY_T.revealStart + CANOPY_T.revealSpread * order;
  return saturate((t - start) / CANOPY_T.revealDuration);
}

/** Cluster scale for a grow progress: quick start from the base, soft settle, no overshoot. */
export function growScale(p: number): number {
  const k = saturate(p);
  const outCubic = 1 - Math.pow(1 - k, 3);
  return 0.35 * k * k * (3 - 2 * k) + 0.65 * outCubic;
}

/** Scale of the whole crown about its bottom centre (0 before 43.12 s, 1 from 44.3 s). */
export function crownScale(t: number): number {
  const k = saturate((t - CANOPY_T.scaleStart) / CANOPY_T.scaleDuration);
  return 1 - (1 - k) * (1 - k);
}

/** Overall reveal 0 → 1 (scaffold, far shells). */
export function globalReveal(t: number): number {
  return saturate((t - CANOPY_T.revealStart) / (CANOPY_T.revealSpread + CANOPY_T.revealDuration));
}

/** Dim far leaves: grow in by their screen radius (inner first). */
export function farGrow(t: number): number {
  return saturate((t - CANOPY_T.farStart) / (CANOPY_T.farEnd - CANOPY_T.farStart));
}

/** Exit progress 0 → 1. */
export function exitProgress(t: number): number {
  return saturate((t - CANOPY_T.exitStart) / (CANOPY_T.exitEnd - CANOPY_T.exitStart));
}

/**
 * Radius R of the opening (screen radius over the half diagonal: 0 = frame centre,
 * 0.58 = top / bottom edge, 0.82 = left / right edge, 1 = corner) — monotone C¹ through
 * the measured keys, 0 until 52.62 s. The leaves of the close-up's dark middle (the
 * crown's hollow) go first, so the opening reads larger than R early on.
 */
const HOLE_KEYS: [number, number][] = [
  [52.62, 0],
  [52.9, 0.2],
  [53.2, 0.6],
  [53.47, 0.8],
  [53.76, 1.08],
  [53.9, 1.2],
];
const HOLE_TIMES = HOLE_KEYS.map(([t]) => t);
const HOLE_VALUES = HOLE_KEYS.map(([, r]) => [r]);
// function declarations below are hoisted: safe at module init
const HOLE_TANGENTS = monotoneTangents(HOLE_TIMES, HOLE_VALUES, true, false);

export function holeRadius(t: number): number {
  if (t <= HOLE_TIMES[0]) return 0;
  return Math.max(0, hermite(HOLE_TIMES, HOLE_VALUES, HOLE_TANGENTS, t)[0]);
}

/** Exposure multiplier of the canopy picture while it leaves (the leaves darken first). */
export function exitDim(t: number): number {
  return 1 - 0.5 * smoothstep(CANOPY_T.dimStart, CANOPY_T.dimEnd, t);
}

// ---------------------------------------------------------------------------
// Monotone cubic Hermite (Fritsch–Butland tangents), per component: a C¹ camera path
// through unevenly spaced keys without overshoot (the camera never stops at a key).
// ---------------------------------------------------------------------------

export function monotoneTangents(times: number[], values: number[][], startAtRest: boolean, endAtRest: boolean): number[][] {
  const n = times.length;
  const dims = values[0].length;
  const out: number[][] = [];
  for (let i = 0; i < n; i++) {
    const m: number[] = [];
    for (let d = 0; d < dims; d++) {
      if (i === 0) m.push(startAtRest ? 0 : (values[1][d] - values[0][d]) / (times[1] - times[0]));
      else if (i === n - 1) m.push(endAtRest ? 0 : (values[i][d] - values[i - 1][d]) / (times[i] - times[i - 1]));
      else {
        const h0 = times[i] - times[i - 1];
        const h1 = times[i + 1] - times[i];
        const s0 = (values[i][d] - values[i - 1][d]) / h0;
        const s1 = (values[i + 1][d] - values[i][d]) / h1;
        m.push(s0 * s1 <= 0 ? 0 : (3 * (h0 + h1)) / ((2 * h1 + h0) / s0 + (h1 + 2 * h0) / s1));
      }
    }
    out.push(m);
  }
  return out;
}

export function hermite(times: number[], values: number[][], tangents: number[][], t: number): number[] {
  const n = times.length;
  if (t <= times[0]) return values[0].slice();
  if (t >= times[n - 1]) return values[n - 1].slice();
  let i = 0;
  while (i < n - 2 && t > times[i + 1]) i++;
  const h = times[i + 1] - times[i];
  const s = (t - times[i]) / h;
  const s2 = s * s;
  const s3 = s2 * s;
  const h00 = 2 * s3 - 3 * s2 + 1;
  const h10 = s3 - 2 * s2 + s;
  const h01 = -2 * s3 + 3 * s2;
  const h11 = s3 - s2;
  return values[i].map((v0, d) => h00 * v0 + h10 * h * tangents[i][d] + h01 * values[i + 1][d] + h11 * h * tangents[i + 1][d]);
}
