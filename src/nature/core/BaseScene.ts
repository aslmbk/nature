/**
 * Convenience base class for scene sets. Subclasses implement `build()` and usually
 * set a camera track; everything else has sensible defaults:
 *
 *   class FooScene extends BaseScene {
 *     readonly id = "foo";
 *     protected async build(ctx: SceneContext) {
 *       // a private copy of the GLB's graph (the cached GLTF itself is shared: never add gltf.scene)
 *       const world = await ctx.assets.instance("models/foo.glb");
 *       if (world) { this.scene.add(world); this.rig.addPosesFromObject(world); }
 *       else this.buildPlaceholder(ctx);
 *       this.rig.setTrack([{ t: 31, pose: "branch_main" }, { t: 37, pose: "branch_out" }]);
 *     }
 *     protected animate(frame, local) { ... pure function of frame / local ... }
 *   }
 *
 * Shader programs are compiled by the engine once `prepare()` has resolved (against
 * the render target the set is actually drawn into, a few materials per frame, linked
 * off the main thread), textures are uploaded one by one, followed by one off-screen
 * warm-up draw when the set is preloaded; scenes do not compile themselves. A set that
 * renders extra scenes of its own (a layer drawn from `animate`) returns them from
 * `warmTargets()` so they are compiled the same way.
 */
import { Fog, PerspectiveCamera, Scene } from "three";
import { CameraRig } from "../CameraRig";
import { REFERENCE_ASPECT } from "../SceneConfig";
import { LightRig } from "../rendering/LightRig";
import type { FrameState, LookParams, NatureScene, SceneContext, SceneLocal, SceneSetId, WarmTarget } from "../types";
import { disposeObject } from "./dispose";

export abstract class BaseScene implements NatureScene {
  abstract readonly id: SceneSetId;
  readonly scene = new Scene();
  readonly camera = new PerspectiveCamera(35, REFERENCE_ASPECT, 0.05, 120);
  readonly rig = new CameraRig(this.camera);
  /** Default lights driven by look.lights (remove from the scene to opt out). */
  readonly lights = new LightRig();
  private ctxRef: SceneContext | null = null;
  private readonly owned = new Set<{ dispose(): void }>();

  constructor() {
    // Fog is always present so programs never recompile when looks change;
    // the engine updates colour / near / far from LookParams each frame.
    this.scene.fog = new Fog(0x000000, 1e4, 2e4);
    this.scene.add(this.lights.group);
  }

  protected get ctx(): SceneContext {
    if (!this.ctxRef) throw new Error(`${this.constructor.name}: ctx is only available after prepare()`);
    return this.ctxRef;
  }

  async prepare(ctx: SceneContext): Promise<void> {
    this.ctxRef = ctx;
    await this.build(ctx);
  }

  /** Build geometry, materials, poses. Must be deterministic for ctx.seed. */
  protected abstract build(ctx: SceneContext): Promise<void> | void;

  /**
   * Extra scenes this set draws itself besides `this.scene` (render-to-texture layers),
   * compiled and warmed by the engine with it. None by default.
   */
  warmTargets(): readonly WarmTarget[] {
    return [];
  }

  getLook(_frame: FrameState, _local: SceneLocal, base: LookParams): LookParams {
    return base;
  }

  update(frame: FrameState, local: SceneLocal, look: LookParams): void {
    this.rig.parallax.enabled = this.ctx.debug.parallax;
    const parallax = this.rig.parallax.enabled && !frame.capture && !frame.reducedMotion;
    this.rig.apply(local.t, frame.viewport, parallax ? frame.pointerNdc : null, frame.deltaSec);
    this.lights.apply(look);
    this.animate(frame, local, look);
  }

  /** Per-frame animation hook. Pure function of (frame, local) — never accumulate. */
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  protected animate(frame: FrameState, local: SceneLocal, look: LookParams): void {}

  /** Dispose `resource` together with the scene. */
  protected own<T extends { dispose(): void }>(resource: T): T {
    this.owned.add(resource);
    return resource;
  }

  dispose(): void {
    const shared = this.ctxRef ? this.ctxRef.assets.isShared : () => false;
    disposeObject(this.scene, shared);
    for (const r of this.owned) if (!shared(r)) r.dispose();
    this.owned.clear();
    this.scene.clear();
  }
}
