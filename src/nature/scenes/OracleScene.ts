/**
 * Oracle scene set — episodes `oracle` (21–25 s, the particle orb) and `streams`
 * (25–31 s, twisted loop and particle streams), camera move 24.5–26 s.
 * Fully code-generated (particles, the orb, orbit and loop are allowed procedural
 * content), no assets. References: frames 07–09, details 04–05, motion/02,
 * clip sheet 08 (21.0–31.5 s). Timeline: `oracle/timeline.ts`.
 *
 * Two passes per view (see oracle/layer.ts — keeps identical frames bit-identical):
 *   blend layer (single-sample, drawn from animate): the opaque parts depth only, then
 *     atmosphere (top beam + wide field, at the far plane) · dust (world) · streams
 *     behind the orb → orb: back shell (dark body) → swarm → threads → nodes → front
 *     shell (film, rim) → streams in front of the orb
 *   engine pass (this.scene, the engine's MSAA target): only opaque draws — orbit + tag
 *     (world, stay behind when the camera pans down), loop (follows the orb), dark
 *     filament (inside the orb) — and the layer composite on top
 *
 * Everything is a pure function of (local.t, frame.timeSec): story-driven motion uses
 * the video time, ambient motion (spin, pulses, drift, twinkle) the ambient clock.
 *
 * The build (≈ 100 ms of particle / curve / mesh generation) runs time-sliced: one step
 * generator (`buildSteps`) with checkpoints inside the particle, shell, filament and loop
 * builders, run by `runSliced` in ≈ 8 ms slices with a macrotask yield in between
 * (vegetation/slices.ts), so the episode on screen keeps rendering while the set
 * prepares. Same rng draws, same objects in the same order as in one go.
 */
import { Group, Quaternion, Vector3 } from "three";
import { CameraRig, type CameraPose } from "../CameraRig";
import { BaseScene } from "../core/BaseScene";
import { smoothstep } from "../core/math";
import type { FrameState, LookParams, SceneContext, SceneLocal, WarmTarget } from "../types";
import { runSliced, yieldToBrowser, type Steps } from "../vegetation/slices";
import { buildAtmosphere, type Atmosphere } from "./oracle/atmosphere";
import { buildLoop, buildOrbit, buildTag, type Loop, type Orbit, type Tag } from "./oracle/curves";
import { BlendLayer } from "./oracle/layer";
import { buildFilament, buildNodes, buildShell, buildThreads, layoutNodes, type Filament, type Nodes, type Shell, type Threads } from "./oracle/orb";
import {
  ORACLE_PALETTE,
  buildDust,
  buildOrbSwarm,
  buildStreams,
  createSpriteUniforms,
  type Dust,
  type OrbSwarm,
  type Streams,
} from "./oracle/particles";
import {
  CAMERA_KEYS,
  CAM_FOV,
  CAM_X,
  ORBIT_RELEASE_T,
  ORB_RADIUS,
  STREAMS_T0,
  assembleProgress,
  exitProgress,
  fountain,
  frameHeightAt,
  loopLight,
  loopReveal,
  orbLit,
  orbScreenY,
  orbitDraw,
  shellOpacity,
  streamHeight,
  streamLoose,
  streamSpread,
  streamThickness,
  streamsOn,
  swirlAngle,
  swirlOpacity,
} from "./oracle/timeline";

const P = CameraRig.lookAtPose;
const REF_HEIGHT = 1020;
/** Orbit radius (orb units) and its tilt: seen ~17° from above, rolled 19° (frame 08). */
const ORBIT_RADIUS = 1.42;
const ORBIT_TILT = 0.3;
const ORBIT_ROLL = 0.33;
/** Share of the particle budget per layer (the orb swarm also forms the swirl). */
const BUDGET = { swarm: 0.52, swirl: 0.27, streams: 0.16, dust: 0.05 };
/** Main-thread slice of the time-sliced build (ms). */
const BUILD_SLICE_MS = 8;

export class OracleScene extends BaseScene {
  readonly id = "oracle" as const;

  private readonly shared = createSpriteUniforms();
  /** Follows the orb's world position (no rotation). */
  private readonly anchor = new Group();
  /** Orb parts, scaled to the orb radius. */
  private readonly orb = new Group();
  /** Loop and streams, also in orb units but outside the spinning interior. */
  private readonly attached = new Group();
  /** Orbit + tag: world space, around the orb's oracle position. */
  private readonly orbitGroup = new Group();
  /** The same orb anchor / scales inside the blend layer's own scene. */
  private readonly layerAnchor = new Group();
  private readonly layerOrb = new Group();
  private readonly layerAttached = new Group();

  private blend!: BlendLayer;
  private atmosphere!: Atmosphere;
  private dust!: Dust;
  private swarm!: OrbSwarm;
  private streams!: Streams;
  private shell!: Shell;
  private nodes!: Nodes;
  private threads!: Threads;
  private filament!: Filament;
  private orbit!: Orbit;
  private tag!: Tag;
  private loop!: Loop;

  private maxPointPx = 64;
  private particleCount = 0;
  private readonly pose: CameraPose = { position: new Vector3(), quaternion: new Quaternion(), fov: CAM_FOV };
  private readonly tmp = new Vector3();
  private readonly tmp2 = new Vector3();
  private debug = { orbY: 0, camY: 0, spin: 0 };

  protected async build(ctx: SceneContext): Promise<void> {
    // the first slice starts in a task of its own (not after the module load / factory),
    // and the engine's compile / warm-up that follows starts in a fresh one again
    await yieldToBrowser();
    await runSliced(this.buildSteps(ctx), BUILD_SLICE_MS, ctx);
    await yieldToBrowser();
  }

  /** The whole build as one step generator (`yield` = checkpoint between pieces of work). */
  private *buildSteps(ctx: SceneContext): Steps<void> {
    const budget = ctx.quality.particles;
    // fewer particles on lower presets: keep the overall light similar
    const gain = Math.min(1.7, Math.sqrt(16000 / Math.max(1000, budget)));
    const pal = ORACLE_PALETTE;

    const gl = ctx.renderer.getContext();
    const range = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE) as Float32Array | number[] | null;
    this.maxPointPx = range && range.length > 1 ? Math.max(8, Number(range[1])) : 64;

    // lights are not used by the custom shaders here
    this.scene.remove(this.lights.group);

    // --- blend layer: every blended draw of the set goes there (oracle/layer.ts) ---
    this.blend = this.own(new BlendLayer(ctx.renderer));
    this.scene.add(this.blend.composite);

    // --- air -------------------------------------------------------------------
    this.atmosphere = buildAtmosphere();
    this.blend.scene.add(this.atmosphere.mesh);

    this.dust = yield* buildDust(
      ctx.rng("dust"),
      this.shared,
      Math.round(budget * BUDGET.dust),
      { x: [-7, 7], y: [-6.8, 3.6], z: [-9, 3.0] },
      pal,
    );
    this.dust.points.renderOrder = 1;
    this.blend.scene.add(this.dust.points);
    ctx.layers.assign(this.dust.points, "particles");

    // --- orb -------------------------------------------------------------------
    this.anchor.name = "oracle_anchor";
    this.scene.add(this.anchor);
    this.orb.name = "oracle_orb";
    this.orb.scale.setScalar(ORB_RADIUS);
    this.anchor.add(this.orb);
    this.attached.name = "oracle_attached";
    this.attached.scale.setScalar(ORB_RADIUS);
    this.anchor.add(this.attached);
    this.layerAnchor.name = "oracle_layer_anchor";
    this.blend.scene.add(this.layerAnchor);
    this.layerOrb.name = "oracle_layer_orb";
    this.layerOrb.scale.setScalar(ORB_RADIUS);
    this.layerAttached.name = "oracle_layer_attached";
    this.layerAttached.scale.setScalar(ORB_RADIUS);
    this.layerAnchor.add(this.layerOrb, this.layerAttached);

    const nodeList = layoutNodes(ctx.rng("nodes"), pal);
    this.shell = yield* buildShell(ctx.rng("shell"));
    yield;
    this.swarm = yield* buildOrbSwarm(ctx.rng("swarm"), this.shared, {
      count: Math.round(budget * BUDGET.swarm),
      swirlCount: Math.round(budget * BUDGET.swirl),
      attractors: nodeList.filter((n) => n.type !== 1).map((n) => n.position),
      palette: pal,
      gain,
    });
    yield;
    this.nodes = buildNodes(nodeList, ctx.rng("node-phase"));
    this.threads = buildThreads(nodeList, ctx.rng("threads"));
    yield;
    this.filament = yield* buildFilament(ctx.rng("filament"));
    this.shell.back.renderOrder = 10;
    this.swarm.points.renderOrder = 11;
    this.threads.lines.renderOrder = 12;
    this.nodes.mesh.renderOrder = 13;
    this.shell.front.renderOrder = 14;
    this.layerOrb.add(this.shell.back, this.swarm.points, this.threads.lines, this.nodes.mesh, this.shell.front);
    this.orb.add(this.filament.mesh);
    ctx.layers.assign(this.orb, "orb");
    ctx.layers.assign(this.layerOrb, "orb");

    // --- streams (behind / in front of the orb: same buffers, two draws) --------
    yield;
    this.streams = yield* buildStreams(ctx.rng("streams"), this.shared, Math.round(budget * BUDGET.streams), pal, gain);
    this.streams.back.renderOrder = 2;
    this.streams.front.renderOrder = 15;
    this.layerAttached.add(this.streams.back, this.streams.front);
    ctx.layers.assign(this.streams.back, "particles");
    ctx.layers.assign(this.streams.front, "particles");

    // --- loop ------------------------------------------------------------------
    yield;
    this.loop = yield* buildLoop(ctx.rng("loop"), { halfWidth: 0.05, halfThickness: 0.033, twists: 4, segments: 900, radial: 12 });
    this.attached.add(this.loop.mesh);
    ctx.layers.assign(this.loop.mesh, "fx");

    // --- orbit + tag (world) ---------------------------------------------------
    yield;
    this.orbitGroup.name = "oracle_orbit_group";
    this.orbitGroup.scale.setScalar(ORB_RADIUS);
    this.orbitGroup.rotation.set(ORBIT_TILT, 0, ORBIT_ROLL, "ZYX");
    this.scene.add(this.orbitGroup);
    this.orbit = buildOrbit(ORBIT_RADIUS, 0.0048);
    this.orbit.uniforms.uStart.value = 1.85;
    this.tag = buildTag(0.17, 0.068);
    this.orbitGroup.add(this.orbit.mesh, this.tag.mesh);
    ctx.layers.assign(this.orbitGroup, "fx");

    // the opaque parts hide blended content behind them (depth only in the layer)
    this.blend.addOccluders(this.filament.mesh.material, this.orbit.mesh.material, this.tag.mesh.material, this.loop.mesh.material);

    this.particleCount =
      this.swarm.points.geometry.getAttribute("position").count +
      this.streams.back.geometry.getAttribute("position").count +
      this.dust.points.geometry.getAttribute("position").count;
    ctx.reportInstances("particles", this.particleCount);
    ctx.reportInstances("orbNodes", nodeList.length);

    // --- camera ----------------------------------------------------------------
    this.rig.fit = { minHorizontalFraction: 0.62, subjectX: 0, maxFov: 62 };
    this.rig.setTrack(CAMERA_KEYS.map((k) => ({ t: k.t, pose: P([CAM_X, k.y, k.z], [CAM_X, k.y, 0], CAM_FOV) })));
  }

  /** The blend layer's scene is drawn by the set itself: the engine compiles it with `this.scene`. */
  warmTargets(): readonly WarmTarget[] {
    return this.blend ? [this.blend.warmTarget()] : [];
  }

  getLook(_frame: FrameState, local: SceneLocal, base: LookParams): LookParams {
    // the orb carries its own light: a touch more bloom once the streams core is lit
    const core = smoothstep(25.2, 27.0, local.t);
    base.bloom.strength *= 1 + 0.15 * core;
    return base;
  }

  protected animate(frame: FrameState, local: SceneLocal): void {
    const t = local.t;
    const reduced = frame.reducedMotion;
    const time = reduced ? frame.timeSec * 0.2 : frame.timeSec;
    const vp = frame.viewport;
    const cam = this.camera;

    // --- orb placement: held at a measured screen height while the camera pans ----
    const orbY = this.orbWorldY(t);
    const camY = this.pose.position.y;
    this.anchor.position.set(0, orbY, 0);
    this.anchor.updateMatrixWorld(true);
    this.debug.orbY = orbY;
    this.debug.camY = camY;

    // --- shared sprite uniforms ----------------------------------------------------
    const dpr = vp.dpr;
    const bufferScale = (vp.height * dpr) / REF_HEIGHT;
    const sh = this.shared;
    sh.uTime.value = time;
    sh.uScale.value = (vp.height * dpr) / (2 * Math.tan((cam.fov * Math.PI) / 360));
    sh.uMinPx.value = 1.5 * Math.max(1, bufferScale);
    sh.uMaxPx.value = Math.min(this.maxPointPx, 80 * Math.max(1, bufferScale));
    this.tmp.set(0, orbY, 0).applyMatrix4(cam.matrixWorldInverse);
    const orbDepth = this.tmp.z;
    sh.uFocus.value = Math.max(0.5, -orbDepth);
    sh.uCocNear.value = 15 * bufferScale;
    sh.uCocFar.value = 2.2 * bufferScale;

    // --- assembly / swirl -------------------------------------------------------------
    const spin = (t - 23.5) * 0.08 + (time - 2) * 0.05;
    this.debug.spin = spin;
    const shellK = shellOpacity(t);
    const exit = exitProgress(t);
    const sw = this.swarm.uniforms;
    sw.uAssemble.value = assembleProgress(t);
    sw.uSwirl.value = swirlAngle(t) + time * 0.04;
    sw.uSwirlOpacity.value = swirlOpacity(t);
    sw.uSpin.value = spin;
    sw.uSparks.value = shellK;
    sw.uFountain.value = fountain(t);
    sw.uLit.value = orbLit(t);
    sw.uCore.value = 0.7 * smoothstep(25.0, 26.8, t);
    sw.uExit.value = exit;
    sw.uOpacity.value = 1 - 0.35 * exit;

    const su = this.shell.uniforms;
    su.uOpacity.value = shellK;
    su.uRot.value = spin * 0.5;
    su.uExit.value = exit;

    const nu = this.nodes.uniforms;
    nu.uSpin.value = spin;
    nu.uTime.value = time;
    nu.uGrow.value = smoothstep(21.9, 22.75, t);
    nu.uCore.value = 0.25 + 1.0 * smoothstep(25.0, 26.8, t);
    nu.uExit.value = exit;

    this.threads.uniforms.uSpin.value = spin;
    this.threads.uniforms.uOpacity.value = smoothstep(22.2, 22.9, t) * (1 - exit);
    this.threads.uniforms.uExit.value = exit;

    this.filament.uniforms.uSpin.value = spin;
    // the dark cable draws itself with the shell and unwinds while the loop grows out of the orb
    this.filament.uniforms.uGrow.value = smoothstep(22.0, 22.8, t) * (1 - smoothstep(24.8, 25.9, t));
    this.filament.mesh.visible = this.filament.uniforms.uGrow.value > 0.001;
    this.filament.uniforms.uExit.value = exit;

    // --- orbit + tag (stay in the world when the camera pans away) -----------------
    const draw = orbitDraw(t);
    this.orbitGroup.position.set(0, t < ORBIT_RELEASE_T ? orbY : this.orbWorldY(ORBIT_RELEASE_T), 0);
    this.orbit.uniforms.uDraw.value = draw;
    this.orbit.uniforms.uLit.value = 0.25 + time * 0.05;
    const phi = 3.45 - (t - 23.5) * 0.6 - (time - 2) * 0.04;
    const r = this.orbit.radius;
    this.tag.uniforms.uCenter.value.set(Math.cos(phi) * r, 0, Math.sin(phi) * r);
    this.tag.uniforms.uTangent.value.set(-Math.sin(phi), 0, Math.cos(phi));
    this.tag.uniforms.uVisible.value = t > 22.75 && draw > 0.999 ? 1 : 0;
    this.orbitGroup.visible = draw > 0.001;

    // --- loop -------------------------------------------------------------------------
    const reveal = loopReveal(t);
    const lu = this.loop.uniforms;
    lu.uReveal.value = reveal;
    lu.uLight.value = loopLight(t);
    this.loop.mesh.visible = reveal > 0.001;
    // a slow sway around the vertical axis (ambient clock)
    this.loop.mesh.rotation.y = 0.1 * Math.sin(time * 0.23 + 0.6);
    this.tmp2.set(0, orbY, 0).applyMatrix4(cam.matrixWorldInverse);
    lu.uOrbPos.value.copy(this.tmp2);
    lu.uOrbRange.value = ORB_RADIUS * 1.9;

    // --- streams ----------------------------------------------------------------------
    const on = streamsOn(t);
    const st = this.streams.uniforms;
    st.uOn.value = on;
    st.uStory.value = Math.max(0, t - STREAMS_T0);
    st.uHeight.value = streamHeight(t);
    st.uThick.value = streamThickness(t);
    st.uSpread.value = streamSpread(t);
    st.uLoose.value = streamLoose(t);
    st.uOpacity.value = 1 - 0.3 * exit;
    st.uOrbDepth.value = orbDepth;
    const streamsVisible = on > 0.001;
    this.streams.back.visible = streamsVisible;
    this.streams.front.visible = streamsVisible;

    // --- air ----------------------------------------------------------------------------
    this.atmosphere.uniforms.uAspect.value = vp.aspect;
    this.dust.uniforms.uOpacity.value = 1;

    // --- blend layer for this view (everything above is set) ----------------------------
    this.layerAnchor.position.copy(this.anchor.position);
    this.blend.render(this.scene, cam, this.ctx.layers.maskFor(this.ctx.debug.layers));
  }

  /** World height of the orb centre at video time t (pure: camera track + measured screen height). */
  private orbWorldY(t: number): number {
    this.rig.evaluate(t, this.pose);
    return this.pose.position.y - (orbScreenY(t) - 0.5) * frameHeightAt(this.pose.position.z);
  }

  debugInfo(): Record<string, string | number | boolean> {
    return {
      particles: this.particleCount,
      orbY: Number(this.debug.orbY.toFixed(3)),
      camY: Number(this.debug.camY.toFixed(3)),
      spin: Number(this.debug.spin.toFixed(3)),
      maxPointPx: this.maxPointPx,
    };
  }
}
