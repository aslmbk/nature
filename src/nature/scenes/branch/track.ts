/**
 * Camera track and wipe edge of the branch set, from the measurements stored in
 * branch.glb (camera extras) and in motion/02 / frame 10.
 *
 * All three cameras share one position; the story is a tilt only:
 *  - before `main` (31.5 s) the picture slides up into place: `slide_before_main_fh` on
 *    `cam_branch_main` gives the vertical offset of the picture (frame heights, relative
 *    to the settled `p1` framing) between 31.2 and 31.5 s. It decays like a smoothed
 *    page scroll; earlier (the streams → branch wipe) it continues the same exponential
 *    (motion/02: the broken end's tip is at 0.72 / 0.62 / 0.56 of the frame height at
 *    31.0 / 31.2 / 31.4 s, i.e. offsets ≈ 0.24 / 0.13 / 0.075 — the fit below gives
 *    0.24 / 0.133 / 0.075);
 *  - `main` → `p1` follows `drift_from_main` (progress 0–1, settled at `settle_video_s`);
 *  - `p1` holds until `hold_until_video_s` (37.0 s), then tilts on to `out` (38.5 s)
 *    while the branch → stone slide moves the picture away.
 * A picture offset of S frame heights is a tilt of atan(S · 2 tan(fov/2)) (main → p1:
 * 0.055 fh ↔ 1.98°, the two cameras' tilts are 6.0° / 7.983°).
 *
 * The wipe edge (streams → branch) is driven by the same smoothed scroll: measured mean
 * edge heights (from the bottom) 0.08–0.12 / ≈0.5 / 0.69 / 0.81 / 0.83 at 30.8 / 31.0 /
 * 31.2 / 31.4 / 31.5 s fit E(t) = 0.96 − 0.27·exp(−(t − 31.2) / 0.34): a ragged dark band
 * is still at the top of frame 10. It is pushed out of the frame after 31.5 s.
 */
import { MathUtils, Quaternion, Vector3 } from "three";
import type { CameraPose, PoseKey } from "../../CameraRig";

export type Samples = [number, number][];

export interface BranchTrackInput {
  main: CameraPose;
  p1: CameraPose;
  out: CameraPose;
  /** Video time of `main` (31.5). */
  mainT: number;
  /** `slide_before_main_fh`: (video s, picture offset in frame heights relative to p1). */
  slide: Samples;
  /** `drift_from_main`: (video s, progress main → p1). */
  drift: Samples;
  /** `settle_video_s` (33.6). */
  settleT: number;
  /** `hold_until_video_s` (37.0). */
  holdT: number;
  /** `video_time_s` of `out` (38.5). */
  outT: number;
}

/** Defaults = the values exported into branch.glb (used when an extra is missing). */
export const TRACK_DEFAULTS = {
  mainT: 31.5,
  slide: [
    [31.2, 0.133],
    [31.267, 0.11],
    [31.333, 0.09],
    [31.4, 0.075],
    [31.467, 0.061],
    [31.5, 0.055],
  ] as Samples,
  drift: [
    [31.5, 0],
    [31.6, 0.25],
    [31.7, 0.393],
    [31.8, 0.5],
    [31.9, 0.536],
    [32.0, 0.607],
    [32.2, 0.714],
    [32.4, 0.786],
    [32.6, 0.857],
    [32.8, 0.893],
    [33.0, 0.929],
    [33.2, 0.964],
    [33.6, 1],
  ] as Samples,
  settleT: 33.6,
  holdT: 37.0,
  outT: 38.5,
} as const;

/** The incoming picture is never offset by more than this (frame heights). */
const MAX_SLIDE_FH = 0.8;
/** First sampled time of the track (the wipe starts at 30.65 s). */
const TRACK_START = 30.4;
const SAMPLE_HZ = 60;

/** Parse an extras array of [t, value] pairs (null if malformed). */
export function readSamples(value: unknown): Samples | null {
  if (!Array.isArray(value)) return null;
  const out: Samples = [];
  for (const p of value) {
    if (!Array.isArray(p) || p.length < 2) return null;
    const t = Number(p[0]);
    const v = Number(p[1]);
    if (!Number.isFinite(t) || !Number.isFinite(v)) return null;
    out.push([t, v]);
  }
  out.sort((a, b) => a[0] - b[0]);
  return out.length >= 2 ? out : null;
}

/**
 * Monotone cubic interpolation (Fritsch–Carlson) through samples: no overshoot between
 * the measured points, C¹, held flat outside.
 */
export class MonotoneCurve {
  private readonly t: number[];
  private readonly v: number[];
  private readonly m: number[];

  constructor(samples: Samples) {
    this.t = samples.map((s) => s[0]);
    this.v = samples.map((s) => s[1]);
    const n = samples.length;
    const d: number[] = [];
    for (let i = 0; i < n - 1; i++) d.push((this.v[i + 1] - this.v[i]) / Math.max(1e-9, this.t[i + 1] - this.t[i]));
    const m: number[] = new Array(n).fill(0);
    m[0] = d[0];
    m[n - 1] = d[n - 2];
    for (let i = 1; i < n - 1; i++) m[i] = d[i - 1] * d[i] <= 0 ? 0 : (d[i - 1] + d[i]) / 2;
    for (let i = 0; i < n - 1; i++) {
      if (d[i] === 0) {
        m[i] = 0;
        m[i + 1] = 0;
        continue;
      }
      const a = m[i] / d[i];
      const b = m[i + 1] / d[i];
      const s = a * a + b * b;
      if (s > 9) {
        const k = 3 / Math.sqrt(s);
        m[i] = k * a * d[i];
        m[i + 1] = k * b * d[i];
      }
    }
    this.m = m;
  }

  get first(): [number, number] {
    return [this.t[0], this.v[0]];
  }

  get last(): [number, number] {
    const n = this.t.length - 1;
    return [this.t[n], this.v[n]];
  }

  at(x: number): number {
    const { t, v, m } = this;
    const n = t.length;
    if (x <= t[0]) return v[0];
    if (x >= t[n - 1]) return v[n - 1];
    let i = 0;
    while (i < n - 2 && x > t[i + 1]) i++;
    const h = t[i + 1] - t[i];
    const s = (x - t[i]) / h;
    const s2 = s * s;
    const s3 = s2 * s;
    return (2 * s3 - 3 * s2 + 1) * v[i] + (s3 - 2 * s2 + s) * h * m[i] + (-2 * s3 + 3 * s2) * v[i + 1] + (s3 - s2) * h * m[i + 1];
  }
}

const X_AXIS = new Vector3(1, 0, 0);
const tmpQ = new Quaternion();

/** Tilt (rad, + = up) that moves the picture by `fh` frame heights at the frame centre. */
export function tiltForSlide(fh: number, fovDeg: number): number {
  return Math.atan(fh * 2 * Math.tan(MathUtils.degToRad(fovDeg) / 2));
}

export class BranchTrack {
  private readonly slide: MonotoneCurve;
  private readonly drift: MonotoneCurve;
  /** Time constant of the exponential slide before the first slide sample. */
  readonly slideTau: number;

  constructor(readonly input: BranchTrackInput) {
    this.slide = new MonotoneCurve(input.slide);
    this.drift = new MonotoneCurve(input.drift);
    const [t0, s0] = this.slide.first;
    const [t1, s1] = this.slide.last;
    this.slideTau = s0 > s1 && s1 > 0 ? (t1 - t0) / Math.log(s0 / s1) : 0.34;
  }

  /** Picture offset (frame heights, relative to p1) before `main`. */
  slideAt(t: number): number {
    const [t0, s0] = this.slide.first;
    if (t >= t0) return this.slide.at(t);
    return Math.min(MAX_SLIDE_FH, s0 * Math.exp((t0 - t) / this.slideTau));
  }

  /** Camera pose at video time t (pure). */
  evaluate(t: number, out: CameraPose): CameraPose {
    const { main, p1, out: last, mainT, settleT, holdT, outT } = this.input;
    out.position.copy(main.position);
    out.fov = main.fov;
    if (t < mainT) {
      // tilt up from main by the extra picture offset
      const extra = tiltForSlide(this.slideAt(t), main.fov) - tiltForSlide(this.slideAt(mainT), main.fov);
      out.quaternion.copy(main.quaternion).multiply(tmpQ.setFromAxisAngle(X_AXIS, extra));
    } else if (t < settleT) {
      out.quaternion.copy(main.quaternion).slerp(p1.quaternion, Math.min(1, Math.max(0, this.drift.at(t))));
    } else if (t < holdT) {
      out.quaternion.copy(p1.quaternion);
    } else if (t < outT) {
      const k = (t - holdT) / Math.max(1e-6, outT - holdT);
      out.quaternion.copy(p1.quaternion).slerp(last.quaternion, 0.5 - 0.5 * Math.cos(Math.PI * k));
      out.position.lerp(last.position, k);
    } else {
      out.position.copy(last.position);
      out.quaternion.copy(last.quaternion);
      out.fov = last.fov;
    }
    return out;
  }

  /** Dense keys for CameraRig.setTrack (linear between samples). */
  keys(): PoseKey[] {
    const { outT } = this.input;
    const keys: PoseKey[] = [];
    const end = outT + 0.5;
    const n = Math.ceil((end - TRACK_START) * SAMPLE_HZ);
    for (let i = 0; i <= n; i++) {
      const t = TRACK_START + i / SAMPLE_HZ;
      const pose = this.evaluate(t, { position: new Vector3(), quaternion: new Quaternion(), fov: 35 });
      keys.push({ t, pose, ease: "linear" });
    }
    return keys;
  }
}

// ---------------------------------------------------------------------------
// streams → branch wipe edge
// ---------------------------------------------------------------------------

/**
 * Exponential approach of the wipe edge (height from the bottom, 0–1), fitted to the
 * dark band in motion/02 (30.8–31.4 s) and clip 09 (lower edge of the band, 75th
 * percentile over the columns: 0.26 / 0.21 / 0.16 / 0.12 / 0.09 of the height from the
 * top at 31.2 / 31.3 / 31.4 / 31.5 / 31.6 s, gone by ≈ 31.9 s); from `exitFrom` it is
 * blended out to the top by the window's end.
 */
export const WIPE_EDGE = { t0: 31.15, e0: 0.69, final: 0.96, tau: 0.34, exitFrom: 31.4 } as const;

/**
 * Edge height (0 = bottom, 1 = top; −margin … 1 + margin) of the streams → branch wipe
 * at video time t, for a window ending at `end`.
 */
export function wipeEdgeAt(t: number, end: number, margin: number): number {
  const w = WIPE_EDGE;
  const measured = w.final - (w.final - w.e0) * Math.exp(-(t - w.t0) / w.tau);
  const from = Math.min(w.exitFrom, end - 0.05);
  const k = Math.min(1, Math.max(0, (t - from) / Math.max(1e-3, end - from)));
  const exit = k * k * (3 - 2 * k);
  const e = measured + (1 + margin - measured) * exit;
  return Math.max(-margin, Math.min(1 + margin, e));
}
