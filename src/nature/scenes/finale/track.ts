/**
 * Camera track of the finale (pure functions of video time).
 *
 *   ≤ 53.2 s   `finale_in` (incoming behind the parting canopy from 52.4 s)
 *   53.2–54.3  dolly in → `finale_main`
 *   54.3–57.5  drift main → `finale_out`
 *   ≥ 57.5 s   `finale_out`
 *
 * Measured on clip 11 (12 frames 52.3–55.5 s) and frames 18–20, 1440×1020:
 *  - the stone's width follows `dolly_from_in` (cam_finale_main extras: share of the
 *    width change in → main, 53.2 → 54.3 s, decelerating). The apparent width goes with
 *    1 / distance, so the share is converted into the camera's progress along the path
 *    (a linear position lerp would grow the stone too late);
 *  - the stone's top edge (screen v from the top) stays near the `in` height while the
 *    camera starts to move and only rises late — the reference's page scroll:
 *      53.17 0.430 · 53.46 0.422 · 53.75 0.397 · 54.05 0.325 · 54.34 0.268 (main)
 *      · 54.63 0.235 · 54.92 0.214 · 55.21 0.202 · 55.50 0.195 · 57.5 0.182 (out, frame 20)
 *    In → main the pitch is solved per sample so the stone top follows that curve on top
 *    of the dolly (the scroll shows up as a slow tilt); main → out is a pure drift whose
 *    share follows the rise: 1 − exp(−(t − 54.3) / 0.63), normalised to reach `out` at
 *    57.5 s (fits 0.38 / 0.63 / 0.77 / 0.85 at the clip times).
 * The track is sampled at 30 Hz into linear keys (no per-frame state).
 */
import { MathUtils, Quaternion, Vector3 } from "three";
import type { CameraPose, PoseKey } from "../../CameraRig";

export interface FinaleTrackInput {
  in: CameraPose;
  main: CameraPose;
  out: CameraPose;
  /** Video seconds of the three poses (camera extras `video_time_s`). */
  tIn: number;
  tMain: number;
  tOut: number;
  /** (t, share of the stone width change in → main), from `dolly_from_in`. */
  dolly: [number, number][];
  /** Highest point of the stone (world). */
  stoneTop: Vector3;
  /** Point the stone's width is measured around (world). */
  stoneCentre: Vector3;
}

/** Share of the stone-top rise in → main at the clip times (from the v values above). */
const TOP_RISE_IN_MAIN: [number, number][] = [
  [0, 0],
  [0.236, 0.055],
  [0.5, 0.209],
  [0.773, 0.65],
  [1, 1],
];
/** Time constant of the drift main → out (s). */
const DRIFT_TAU = 0.63;
const SAMPLE_HZ = 30;

export const TRACK_DEFAULTS = {
  tIn: 53.2,
  tMain: 54.3,
  tOut: 57.5,
  dolly: [
    [53.2, 0],
    [53.4, 0.27],
    [53.6, 0.55],
    [53.8, 0.72],
    [54.0, 0.92],
    [54.3, 1],
  ] as [number, number][],
};

/** Parse the `dolly_from_in` extra (JSON array of [t, share]). */
export function readDolly(value: unknown): [number, number][] | null {
  try {
    const arr = typeof value === "string" ? (JSON.parse(value) as unknown) : value;
    if (!Array.isArray(arr) || arr.length < 2) return null;
    const out: [number, number][] = [];
    for (const p of arr) {
      if (!Array.isArray(p) || p.length < 2) return null;
      const t = Number(p[0]);
      const s = Number(p[1]);
      if (!Number.isFinite(t) || !Number.isFinite(s)) return null;
      out.push([t, s]);
    }
    return out.sort((a, b) => a[0] - b[0]);
  } catch {
    return null;
  }
}

/** Monotone cubic (Fritsch–Carlson) through (x, y) samples; clamped outside. */
export function monotoneCurve(points: [number, number][]): (x: number) => number {
  const n = points.length;
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const d: number[] = [];
  for (let i = 0; i < n - 1; i++) d.push((ys[i + 1] - ys[i]) / Math.max(1e-9, xs[i + 1] - xs[i]));
  const m: number[] = new Array(n).fill(0);
  m[0] = d[0];
  m[n - 1] = d[n - 2];
  for (let i = 1; i < n - 1; i++) m[i] = d[i - 1] * d[i] <= 0 ? 0 : (d[i - 1] + d[i]) / 2;
  for (let i = 0; i < n - 1; i++) {
    if (Math.abs(d[i]) < 1e-12) {
      m[i] = 0;
      m[i + 1] = 0;
      continue;
    }
    const a = m[i] / d[i];
    const b = m[i + 1] / d[i];
    const h = a * a + b * b;
    if (h > 9) {
      const tau = 3 / Math.sqrt(h);
      m[i] = tau * a * d[i];
      m[i + 1] = tau * b * d[i];
    }
  }
  return (x: number) => {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    let i = 0;
    while (i < n - 2 && x > xs[i + 1]) i++;
    const h = xs[i + 1] - xs[i];
    const s = (x - xs[i]) / h;
    const s2 = s * s;
    const s3 = s2 * s;
    return (2 * s3 - 3 * s2 + 1) * ys[i] + (s3 - 2 * s2 + s) * h * m[i] + (-2 * s3 + 3 * s2) * ys[i + 1] + (s3 - s2) * h * m[i + 1];
  };
}

const tmpV = new Vector3();
const tmpF = new Vector3();
const tmpU = new Vector3();

/** Screen v (0 top … 1 bottom) of a world point for a pose at the reference aspect. */
export function screenV(pose: CameraPose, p: Vector3): number {
  const inv = tmpV.copy(p).sub(pose.position);
  const f = tmpF.set(0, 0, -1).applyQuaternion(pose.quaternion);
  const u = tmpU.set(0, 1, 0).applyQuaternion(pose.quaternion);
  const depth = Math.max(1e-4, inv.dot(f));
  const y = inv.dot(u) / depth / Math.tan(MathUtils.degToRad(pose.fov) / 2);
  return 0.5 - 0.5 * y;
}

/** Distance along the view axis from a pose to a point (DOF focus). */
export function viewDepth(pose: CameraPose, p: Vector3): number {
  const f = tmpF.set(0, 0, -1).applyQuaternion(pose.quaternion);
  return tmpV.copy(p).sub(pose.position).dot(f);
}

function lerpPose(a: CameraPose, b: CameraPose, s: number, out: CameraPose): CameraPose {
  out.position.copy(a.position).lerp(b.position, s);
  out.quaternion.copy(a.quaternion).slerp(b.quaternion, s);
  out.fov = a.fov + (b.fov - a.fov) * s;
  return out;
}

/** Bisection on a monotone function f(s) = target over [lo, hi]. */
function solve(f: (s: number) => number, target: number, lo: number, hi: number): number {
  let a = lo;
  let b = hi;
  const fa = f(a) - target;
  const increasing = f(b) - target > fa;
  for (let i = 0; i < 48; i++) {
    const mid = 0.5 * (a + b);
    const v = f(mid) - target;
    if (v === 0) return mid;
    if ((v < 0) === increasing) a = mid;
    else b = mid;
  }
  return 0.5 * (a + b);
}

export interface FinaleTrack {
  keys: PoseKey[];
  /** Share of the way in → main of the camera position at t (debug). */
  dollyAt(t: number): number;
}

export function buildFinaleTrack(input: FinaleTrackInput): FinaleTrack {
  const { tIn, tMain, tOut } = input;
  const widthShare = monotoneCurve(input.dolly);
  const dIn = input.in.position.distanceTo(input.stoneCentre);
  const dMain = input.main.position.distanceTo(input.stoneCentre);
  const posAt = new Vector3();
  // width share w → position share s along in → main: 1/d(s) = 1/dIn + w (1/dMain − 1/dIn)
  const positionShare = (w: number): number => {
    const target = 1 / (1 / dIn + w * (1 / dMain - 1 / dIn));
    return solve((s) => posAt.copy(input.in.position).lerp(input.main.position, s).distanceTo(input.stoneCentre), target, -0.2, 1.2);
  };
  const vIn = screenV(input.in, input.stoneTop);
  const vMain = screenV(input.main, input.stoneTop);
  const rise = monotoneCurve(TOP_RISE_IN_MAIN.map(([k, s]) => [tIn + k * (tMain - tIn), s] as [number, number]));
  const driftNorm = 1 - Math.exp(-(tOut - tMain) / DRIFT_TAU);
  const drift = (t: number) => Math.min(1, Math.max(0, (1 - Math.exp(-(t - tMain) / DRIFT_TAU)) / driftNorm));

  const keys: PoseKey[] = [{ t: tIn, pose: clonePose(input.in), ease: "linear" }];
  const pose: CameraPose = { position: new Vector3(), quaternion: new Quaternion(), fov: input.in.fov };
  const probe: CameraPose = { position: new Vector3(), quaternion: new Quaternion(), fov: input.in.fov };
  const step = 1 / SAMPLE_HZ;
  const n = Math.max(1, Math.round((tOut - tIn) / step));
  for (let i = 1; i <= n; i++) {
    const t = tIn + ((tOut - tIn) * i) / n;
    if (t <= tMain + 1e-6) {
      const s = positionShare(widthShare(t));
      const vTarget = vIn + (vMain - vIn) * rise(t);
      // orientation: the share of the in → main rotation that puts the stone top at vTarget
      const q = solve(
        (k) => {
          probe.position.copy(input.in.position).lerp(input.main.position, s);
          probe.quaternion.copy(input.in.quaternion).slerp(input.main.quaternion, k);
          probe.fov = input.in.fov + (input.main.fov - input.in.fov) * s;
          return screenV(probe, input.stoneTop);
        },
        vTarget,
        -0.5,
        1.5,
      );
      pose.position.copy(input.in.position).lerp(input.main.position, s);
      pose.quaternion.copy(input.in.quaternion).slerp(input.main.quaternion, q);
      pose.fov = input.in.fov + (input.main.fov - input.in.fov) * s;
    } else {
      lerpPose(input.main, input.out, drift(t), pose);
    }
    keys.push({ t, pose: clonePose(pose), ease: "linear" });
  }
  // exact end poses
  keys[keys.length - 1] = { t: tOut, pose: clonePose(input.out), ease: "linear" };
  return {
    keys,
    dollyAt: (t: number) => (t <= tIn ? 0 : t >= tMain ? 1 : positionShare(widthShare(t))),
  };
}

function clonePose(p: CameraPose): CameraPose {
  return { position: p.position.clone(), quaternion: p.quaternion.clone(), fov: p.fov };
}
