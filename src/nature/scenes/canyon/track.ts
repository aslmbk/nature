/**
 * Camera track and light timeline of the canyon set (pure functions of video time t).
 *
 *   ≤ 16.4 s   `canyon_in` — incoming inside the arch → canyon dip (14.6–16.4 s); the
 *              set is unlit until 15.8 s, so the dip opens onto black
 *   16.4–19.8  slow pull back and rise → `canyon_main` (frame 06, 19.8 s)
 *   19.8–21.2  the same drift goes on → `canyon_out`, inside the dark crossfade to the
 *              oracle (20.3–21.5 s)
 *   ≥ 21.2 s   `canyon_out`
 *
 * One C¹ path through the three poses: cubic Hermite per component (position,
 * quaternion, fov, focus), starting at rest at `in`, Catmull-Rom tangent at `main`
 * (the drift does not stop at frame 06: in → main ≈ 0.12 m/s, main → out ≈ 0.13 m/s),
 * a soft landing at `out` (the frame is black by then). Sampled at 30 Hz into linear
 * rig keys: the pose stays a pure function of t. The DOF focus distance (camera extras
 * `focus_distance_m`) rides on the same curve.
 *
 * Light: the key (and with it fill, rim and the air of canyon/air.ts) comes up 15.8 →
 * 16.8 s — the canyon opens out of the dark at the end of the arch → canyon dip — and
 * goes down 20.6 → 21.2 s, so the canyon side of the canyon → oracle crossfade is black
 * at 21.2 s and the oracle's particles come out of darkness. In between the far end of
 * the cleft slowly brightens (`farGlow`).
 */
import { Quaternion, Vector3 } from "three";
import type { CameraPose, PoseKey } from "../../CameraRig";
import { smoothstep } from "../../core/math";

export interface CanyonKey {
  t: number;
  pose: CameraPose;
  /** DOF focus distance (m). */
  focus: number;
}

/** Defaults = the camera extras exported into canyon.glb (used when an extra is missing). */
export const TRACK_DEFAULTS = {
  tIn: 16.4,
  tMain: 19.8,
  tOut: 21.2,
  focusIn: 4.05,
  focusMain: 4.45,
  focusOut: 4.6,
} as const;

/** Key / fill / rim fade in (video s): the canyon opens out of the dip. */
export const LIGHT_IN: readonly [number, number] = [15.8, 16.8];
/** Fade out into the canyon → oracle crossfade: near-black at 21.2 s. */
export const LIGHT_OUT: readonly [number, number] = [20.6, 21.2];

/** 0–1 light level of the set at video time t. */
export function lightGain(t: number): number {
  return smoothstep(LIGHT_IN[0], LIGHT_IN[1], t) * (1 - smoothstep(LIGHT_OUT[0], LIGHT_OUT[1], t));
}

/**
 * The far end of the cleft (canyon/air.ts) brightens while the camera drifts back, 0.8 →
 * 1.25 × over 17.2–20.4 s: the light the orb comes out of after the crossfade. On top of
 * `lightGain`, so it still goes dark with everything else 20.6 → 21.2 s.
 */
export const FAR_GLOW = { from: 0.8, to: 1.25, t: [17.2, 20.4] as const };

export function farGlow(t: number): number {
  return FAR_GLOW.from + (FAR_GLOW.to - FAR_GLOW.from) * smoothstep(FAR_GLOW.t[0], FAR_GLOW.t[1], t);
}

const SAMPLE_HZ = 30;
/** Velocity kept at `out` (× the main → out chord): a soft landing, not a stop. */
const END_SCALE = 0.5;

export interface CanyonTrack {
  keys: PoseKey[];
  /** (t, focus) samples on the same curve. */
  focus: [number, number][];
}

/** Hermite path in → main → out, sampled into rig keys (+ focus samples). */
export function buildCanyonTrack(keysIn: [CanyonKey, CanyonKey, CanyonKey]): CanyonTrack {
  const keys = keysIn.map((k) => ({ ...k, pose: { ...k.pose, quaternion: k.pose.quaternion.clone() } }));
  // quaternions on one hemisphere
  for (let i = 1; i < keys.length; i++) {
    const q = keys[i].pose.quaternion;
    if (q.dot(keys[i - 1].pose.quaternion) < 0) q.set(-q.x, -q.y, -q.z, -q.w);
  }
  const comps = (k: CanyonKey) => [
    k.pose.position.x,
    k.pose.position.y,
    k.pose.position.z,
    k.pose.quaternion.x,
    k.pose.quaternion.y,
    k.pose.quaternion.z,
    k.pose.quaternion.w,
    k.pose.fov,
    k.focus,
  ];
  const v = keys.map(comps);
  const t = keys.map((k) => k.t);
  const dims = v[0].length;
  // tangents: at rest at `in`, Catmull-Rom across `main`, END_SCALE × chord at `out`
  const m0 = new Array<number>(dims).fill(0);
  const m1 = v[0].map((_, d) => (v[2][d] - v[0][d]) / (t[2] - t[0]));
  const m2 = v[0].map((_, d) => ((v[2][d] - v[1][d]) / (t[2] - t[1])) * END_SCALE);
  const tangents = [m0, m1, m2];

  const out: PoseKey[] = [];
  const focus: [number, number][] = [];
  const steps = Math.max(2, Math.ceil((t[2] - t[0]) * SAMPLE_HZ));
  for (let s = 0; s <= steps; s++) {
    const ts = t[0] + ((t[2] - t[0]) * s) / steps;
    const i = ts <= t[1] ? 0 : 1;
    const h = t[i + 1] - t[i];
    const u = Math.min(1, Math.max(0, (ts - t[i]) / h));
    const u2 = u * u;
    const u3 = u2 * u;
    const h00 = 2 * u3 - 3 * u2 + 1;
    const h10 = u3 - 2 * u2 + u;
    const h01 = -2 * u3 + 3 * u2;
    const h11 = u3 - u2;
    const c = v[i].map((a, d) => h00 * a + h10 * h * tangents[i][d] + h01 * v[i + 1][d] + h11 * h * tangents[i + 1][d]);
    out.push({
      t: ts,
      pose: { position: new Vector3(c[0], c[1], c[2]), quaternion: new Quaternion(c[3], c[4], c[5], c[6]).normalize(), fov: c[7] },
      ease: "linear",
    });
    focus.push([ts, c[8]]);
  }
  return { keys: out, focus };
}

/** Focus distance at t from the track's samples (held outside). */
export function focusAt(samples: readonly [number, number][], t: number, fallback: number): number {
  const n = samples.length;
  if (n === 0) return fallback;
  if (t <= samples[0][0]) return samples[0][1];
  if (t >= samples[n - 1][0]) return samples[n - 1][1];
  // uniform sampling: direct index
  const t0 = samples[0][0];
  const dt = (samples[n - 1][0] - t0) / (n - 1);
  const f = (t - t0) / dt;
  const i = Math.min(n - 2, Math.max(0, Math.floor(f)));
  const k = f - i;
  return samples[i][1] + (samples[i + 1][1] - samples[i][1]) * k;
}
