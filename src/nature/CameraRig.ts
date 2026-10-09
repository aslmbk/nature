/**
 * Camera poses and their interpolation.
 *
 * Poses (position, quaternion, vertical fov) are named `<episode>_<pose>`; they come
 * from GLB cameras named `cam_<episode>_<pose>` (`addPosesFromObject`) or from code
 * (`addPose`, `CameraRig.lookAtPose`). A track maps video seconds to poses; the pose
 * is a pure function of t (lerp position, slerp quaternion, lerp fov per segment).
 *
 * Aspect: the reference frame is 1440:1020. Wider screens keep the vertical fov.
 * Narrower screens keep at least `fit.minHorizontalFraction` of the reference width
 * (fov grows, capped by `fit.maxFov`) and shift the lens towards `fit.subjectX`, so the
 * main subject stays in frame; the taller window they get may move up or down towards
 * `fit.subjectY` (0 = centred as before).
 *
 * Pointer parallax (off by default, ≤ 1°) is applied on top of the pose and never
 * accumulated into it.
 */
import { MathUtils, Object3D, PerspectiveCamera, Quaternion, Vector3, Euler } from "three";
import { REFERENCE_ASPECT } from "./SceneConfig";
import { clamp, damp, ease, invLerp, saturate } from "./core/math";
import type { EaseName, Viewport } from "./types";

export interface CameraPose {
  position: Vector3;
  quaternion: Quaternion;
  /** Vertical fov in degrees at the reference aspect. */
  fov: number;
}

export interface PoseKey {
  /** Video seconds. */
  t: number;
  /** Pose name or an inline pose. */
  pose: string | CameraPose;
  /** Ease of the segment that ends at this key (default inOutSine). */
  ease?: EaseName;
}

export interface FitOptions {
  /** Fraction (0–1) of the reference frame width that must stay visible on narrow screens. */
  minHorizontalFraction: number;
  /** Subject position in the reference frame, NDC x (−1..1); narrow frames re-centre towards it. */
  subjectX: number;
  /** Upper bound for the adapted vertical fov (degrees). */
  maxFov: number;
  /**
   * Vertical subject position in the reference frame, NDC y (−1..1, default 0): the
   * window that narrow screens get (taller than the reference frame) is centred on it as
   * far as it can while it still holds the reference frame's full height (lens shift).
   */
  subjectY?: number;
}

const tmpV = new Vector3();
const tmpQ = new Quaternion();
const tmpE = new Euler();

export class CameraRig {
  readonly poses = new Map<string, CameraPose>();
  private track: { t: number; pose: CameraPose; ease: EaseName }[] = [];
  fit: FitOptions = { minHorizontalFraction: 0.72, subjectX: 0, maxFov: 70 };
  /** Optional pointer parallax (off by default). */
  parallax = { enabled: false, maxDeg: 1 };
  private pointer = { x: 0, y: 0 };
  private readonly current: CameraPose = { position: new Vector3(), quaternion: new Quaternion(), fov: 35 };
  /**
   * Without a track the pose is the camera's own pose, captured once on the first
   * `evaluate` (after build). Reading the camera every frame would feed `applyPose`'s
   * output back in (the aspect-fitted fov would grow towards `fit.maxFov`).
   */
  private still: CameraPose | null = null;

  constructor(readonly camera: PerspectiveCamera) {}

  static lookAtPose(position: [number, number, number] | Vector3, target: [number, number, number] | Vector3, fov = 35): CameraPose {
    const pos = Array.isArray(position) ? new Vector3(...position) : position.clone();
    const tgt = Array.isArray(target) ? new Vector3(...target) : target.clone();
    // Object3D.lookAt points +Z at the target for non-cameras; cameras look down −Z.
    const cam = new PerspectiveCamera();
    cam.position.copy(pos);
    cam.lookAt(tgt);
    return { position: pos, quaternion: cam.quaternion.clone(), fov };
  }

  addPose(name: string, pose: CameraPose): this {
    this.poses.set(name, { position: pose.position.clone(), quaternion: pose.quaternion.clone(), fov: pose.fov });
    return this;
  }

  pose(name: string): CameraPose | undefined {
    return this.poses.get(name);
  }

  /**
   * Collect `cam_<episode>_<pose>` cameras under `root` (e.g. gltf.scene) as poses
   * named `<episode>_<pose>`. Returns the names found.
   */
  addPosesFromObject(root: Object3D): string[] {
    const found: string[] = [];
    root.updateMatrixWorld(true);
    root.traverse((o) => {
      const cam = o as PerspectiveCamera;
      const match = /^cam_([A-Za-z]+)_([A-Za-z0-9]+)/.exec(o.name);
      if (!match || !cam.isPerspectiveCamera) return;
      const name = `${match[1]}_${match[2]}`;
      const position = new Vector3();
      const quaternion = new Quaternion();
      o.matrixWorld.decompose(position, quaternion, tmpV);
      this.poses.set(name, { position, quaternion, fov: cam.fov });
      found.push(name);
    });
    return found;
  }

  /** Keys in video seconds (sorted here). Before the first / after the last key the pose holds. */
  setTrack(keys: PoseKey[]): this {
    this.still = null;
    this.track = keys
      .map((k) => {
        const pose = typeof k.pose === "string" ? this.poses.get(k.pose) : k.pose;
        if (!pose) throw new Error(`CameraRig: unknown pose "${String(k.pose)}"`);
        return { t: k.t, pose, ease: k.ease ?? "inOutSine" };
      })
      .sort((a, b) => a.t - b.t);
    return this;
  }

  /** Pose at video time t (pure). */
  evaluate(t: number, out: CameraPose = this.current): CameraPose {
    const tr = this.track;
    if (tr.length === 0) {
      if (!this.still) {
        const cam = this.camera;
        this.still = { position: cam.position.clone(), quaternion: cam.quaternion.clone(), fov: cam.fov };
      }
      return copyPose(this.still, out);
    }
    if (t <= tr[0].t || tr.length === 1) return copyPose(tr[0].pose, out);
    const last = tr[tr.length - 1];
    if (t >= last.t) return copyPose(last.pose, out);
    let i = 1;
    while (i < tr.length && tr[i].t < t) i++;
    const a = tr[i - 1];
    const b = tr[i];
    const k = ease(b.ease, saturate(invLerp(a.t, b.t, t)));
    out.position.copy(a.pose.position).lerp(b.pose.position, k);
    out.quaternion.copy(a.pose.quaternion).slerp(b.pose.quaternion, k);
    out.fov = a.pose.fov + (b.pose.fov - a.pose.fov) * k;
    return out;
  }

  /**
   * Pose the camera for video time t and the current viewport.
   * `pointerNdc` (or null) drives the optional parallax; `dt` smooths it.
   */
  apply(t: number, viewport: Viewport, pointerNdc: { x: number; y: number } | null = null, dt = 0): void {
    const pose = this.evaluate(t);
    this.applyPose(pose, viewport, pointerNdc, dt);
  }

  applyPose(pose: CameraPose, viewport: Viewport, pointerNdc: { x: number; y: number } | null = null, dt = 0): void {
    const cam = this.camera;
    cam.position.copy(pose.position);
    cam.quaternion.copy(pose.quaternion);

    if (this.parallax.enabled && pointerNdc) {
      const lambda = 4;
      this.pointer.x = dt > 0 ? damp(this.pointer.x, pointerNdc.x, lambda, dt) : pointerNdc.x;
      this.pointer.y = dt > 0 ? damp(this.pointer.y, pointerNdc.y, lambda, dt) : pointerNdc.y;
      const max = MathUtils.degToRad(Math.min(1, Math.max(0, this.parallax.maxDeg)));
      tmpE.set(this.pointer.y * max * 0.6, -this.pointer.x * max, 0, "YXZ");
      cam.quaternion.multiply(tmpQ.setFromEuler(tmpE));
    }

    // aspect fit
    const aspect = viewport.aspect > 0 ? viewport.aspect : REFERENCE_ASPECT;
    const tanV = Math.tan(MathUtils.degToRad(pose.fov) / 2);
    let tanVOut = tanV;
    let shift = 0;
    let shiftY = 0;
    if (aspect < REFERENCE_ASPECT) {
      const refHalfW = tanV * REFERENCE_ASPECT;
      const needHalfW = refHalfW * clamp(this.fit.minHorizontalFraction, 0, 1);
      tanVOut = Math.max(tanV, needHalfW / aspect);
      const maxTan = Math.tan(MathUtils.degToRad(this.fit.maxFov) / 2);
      tanVOut = Math.min(tanVOut, Math.max(tanV, maxTan));
      // lens shift towards the subject, keeping the window inside the reference frame
      const halfW = tanVOut * aspect;
      const room = Math.max(0, refHalfW - halfW);
      shift = clamp(this.fit.subjectX * refHalfW, -room, room);
      const roomY = tanVOut - tanV;
      shiftY = clamp((this.fit.subjectY ?? 0) * tanV, -roomY, roomY);
    }
    cam.fov = MathUtils.radToDeg(2 * Math.atan(tanVOut));
    cam.aspect = aspect;
    cam.filmOffset = shift * cam.getFilmWidth();
    cam.updateProjectionMatrix();
    if (shiftY !== 0) {
      // vertical lens shift (three has no vertical film offset): the same frustum, moved
      const n = cam.near;
      const halfW = tanVOut * aspect;
      cam.projectionMatrix.makePerspective((shift - halfW) * n, (shift + halfW) * n, (shiftY + tanVOut) * n, (shiftY - tanVOut) * n, n, cam.far, cam.coordinateSystem, cam.reversedDepth);
      cam.projectionMatrixInverse.copy(cam.projectionMatrix).invert();
    }
    cam.updateMatrixWorld();
  }
}

function copyPose(src: CameraPose, out: CameraPose): CameraPose {
  out.position.copy(src.position);
  out.quaternion.copy(src.quaternion);
  out.fov = src.fov;
  return out;
}
