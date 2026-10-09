/**
 * The fuzzy ball's closed path (branch.glb: empties `anchor_ball_0…11`).
 *
 * Closed centripetal Catmull-Rom through the anchors in index order. Every anchor
 * carries the loop phase at which the reference ball passes it (`loop_phase`, measured
 * from clip 09); the phase → curve parameter mapping is piecewise linear through those
 * phases (`loop_note` on anchor 0), not uniform: segment i spans
 * [phase_i, phase_i+1) and maps to getPoint((i + w) / n).
 *
 * The loop is ambient (the reference ball keeps moving while the page is still):
 * phase = fract((timeSec − t0) / period) with t0 = `video_time_s` of anchor 0
 * (33.533 s) and period `period_s` (2.367 s), so `?t=34.5&time=34.5` puts the ball where
 * the reference shows it at 34.5 s. Pure function of the ambient clock.
 */
import { CatmullRomCurve3, Vector3, type Object3D } from "three";
import { fract } from "../../core/math";

export interface BallLoopSpec {
  /** Anchor positions in the space the ball is animated in. */
  points: Vector3[];
  /** Loop phase of each anchor, strictly increasing in [0, 1). */
  phases: number[];
  period: number;
  /** Ambient time at which the ball passes anchor 0. */
  t0: number;
  /** Ball radius (m) as modelled. */
  radius: number;
}

export const BALL_LOOP_DEFAULTS = { period: 2.367, t0: 33.533, radius: 0.06 } as const;

export class BallLoop {
  readonly curve: CatmullRomCurve3;

  constructor(readonly spec: BallLoopSpec) {
    this.curve = new CatmullRomCurve3(spec.points, true, "centripetal");
  }

  /**
   * Collect `anchor_ball_<i>` empties under `root`; positions in the space of `space`
   * (default: root). Null when there are fewer than 4 anchors.
   */
  static fromObject(root: Object3D, space: Object3D = root): BallLoop | null {
    const anchors: { index: number; object: Object3D }[] = [];
    root.traverse((o) => {
      const m = /^anchor_ball_(\d+)$/.exec(o.name);
      if (m) anchors.push({ index: Number(m[1]), object: o });
    });
    if (anchors.length < 4) return null;
    anchors.sort((a, b) => a.index - b.index);
    root.updateMatrixWorld(true);
    space.updateMatrixWorld(true);
    const n = anchors.length;
    const points = anchors.map((a) => space.worldToLocal(a.object.getWorldPosition(new Vector3())));
    // phases: extras when present and increasing, uniform otherwise
    let phases = anchors.map((a) => Number(a.object.userData.loop_phase));
    const valid = phases.every((p, i) => Number.isFinite(p) && p >= 0 && p < 1 && (i === 0 || p > phases[i - 1]));
    if (!valid) phases = anchors.map((_, i) => i / n);
    const first = anchors[0].object.userData;
    const period = Number(first.period_s);
    const t0 = Number(first.video_time_s);
    const radius = Number(first.ball_radius_m);
    return new BallLoop({
      points,
      phases,
      period: Number.isFinite(period) && period > 0 ? period : BALL_LOOP_DEFAULTS.period,
      t0: Number.isFinite(t0) ? t0 : BALL_LOOP_DEFAULTS.t0,
      radius: Number.isFinite(radius) && radius > 0 ? radius : BALL_LOOP_DEFAULTS.radius,
    });
  }

  /** Loop phase 0–1 at ambient time `timeSec`. */
  phase(timeSec: number): number {
    return fract((timeSec - this.spec.t0) / this.spec.period);
  }

  /** Curve parameter (0–1, getPoint) for a loop phase. */
  param(phase: number): number {
    const ph = this.spec.phases;
    const n = ph.length;
    const p = fract(phase);
    // before anchor 0's phase (only if it is > 0): the closing segment n−1 → 0
    let i = n - 1;
    for (let k = 0; k < n; k++) {
      const next = k + 1 < n ? ph[k + 1] : 1 + ph[0];
      if (p >= ph[k] && p < next) {
        i = k;
        break;
      }
    }
    const a = ph[i];
    const b = i + 1 < n ? ph[i + 1] : 1 + ph[0];
    const q = p < a ? p + 1 : p;
    const w = Math.min(1, Math.max(0, (q - a) / Math.max(1e-6, b - a)));
    return (i + w) / n;
  }

  /** Ball centre at ambient time `timeSec`. */
  position(timeSec: number, out: Vector3): Vector3 {
    return this.curve.getPoint(this.param(this.phase(timeSec)), out);
  }
}
