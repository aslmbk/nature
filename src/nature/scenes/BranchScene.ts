/**
 * Branch scene set — episode `branch` (31–37 s), the entry wipe from the streams
 * (30.65 s on) and the slide out to the stone (37 s on). Frames 10–13, details 06 / 07,
 * clip 09, motion/02.
 *
 * `models/branch.glb` (Blender: assets-src/blender/build_branch.py):
 *  - `wood_branch_j` / `wood_branch_frag`  bark (grain along V) → createBarkMaterial with
 *    the painted grain toned down and moss-stained next to the moss (branch/bark.ts); the
 *    J's bark darkens towards its broken end (the lower arm in shade, frame 10)
 *  - `wood_branch_stumps`  the two stumps and torn fibres: near-black matt bark, pale warm
 *    fresh wood where COLOR_0.G == 0 (breaks, splinter lips, fibre ends)
 *  - `moss_branch_j` / `moss_branch_frag`  moss cushions → moss layer material + the
 *    `branchMoss` vegetation (fine fuzz, tufts, white flower heads, moss on the bark and
 *    the stumps), built in time slices (`buildVegetationAsync`)
 *  - `moss_ball` (at the origin, r ≈ 6 cm) → the fuzzy ball's core, own group, `ballFur`
 *  - `cam_branch_{main,p1,out}` one position, tilts only; extras drive the track
 *    (branch/track.ts): slide into place before main, drift main → p1, hold, out
 *  - `key_branch` empty: the key light's direction (local −Z; KEY_DIR when missing),
 *    tilted a little towards the zenith
 *  - `anchor_ball_0…11` the ball's closed loop with measured phases (branch/ballLoop.ts)
 * The ball travels on its loop as an ambient motion (keeps going while the page is
 * still), tumbles slowly and its hairs sway with the shared wind. On the visible arc it
 * is in front of the wood, 1.1–1.6 m from the lens: sharp on entry (frame 10), soft
 * through the near-field DOF from 33.5 s on (frames 11 / 12); around loop phase 0.75 it
 * leaves the frame on the right, passes above the camera and comes back in from the top.
 * Branch → stone: the outgoing picture goes soft (≈ 4 px) over the first half of the slide.
 * Light: key / sky fill / back light of the LightRig. The pile's form shading (moss
 * cushions, the ball's coat) comes from the vegetation's own `surfaceShade` / `clumpAo`
 * options (branch/recipes.ts).
 * Without branch.glb a small PLACEHOLDER set keeps the episode working.
 */
import { Box3, Color, Frustum, Group, MathUtils, Matrix4, Vector3, type Mesh, type MeshStandardMaterial, type Object3D } from "three";
import { CameraRig, type CameraPose } from "../CameraRig";
import { BaseScene } from "../core/BaseScene";
import { clamp, smoothstep } from "../core/math";
import { REFERENCE_ASPECT, TRANSITIONS, TRANSITION_TUNING_DEFAULTS } from "../SceneConfig";
import { createBarkMaterial, createMossBaseMaterial } from "../rendering/Materials";
import type { ActiveTransition, FrameState, LookParams, SceneContext, SceneLocal, TransitionParams, Viewport } from "../types";
import { createMossLayerMaterial } from "../vegetation/MossLayer";
import type { ScatterView } from "../vegetation/SurfaceScatter";
import { buildVegetationAsync, loadKit, type VegetationBuild } from "../vegetation/Vegetation";
import { toneBark } from "./branch/bark";
import { BallLoop } from "./branch/ballLoop";
import { ballFur, branchMoss } from "./branch/recipes";
import { BranchTrack, TRACK_DEFAULTS, readSamples, wipeEdgeAt } from "./branch/track";
import { blobGeometry, curvePoint, placeholderMesh, tubeGeometry } from "./placeholders";

/** Fallback when the extra is missing (the value exported into branch.glb). */
const FOCUS_DISTANCE = 2.38;
/**
 * Key light, direction of travel, when `key_branch` is missing: from above and in front,
 * a little from the right (frames 10–12: the stem's right moss is the brighter side, the
 * bark between the moss is lit, the J's underside dark).
 */
const KEY_DIR = new Vector3(-0.12, -0.7, -0.7).normalize();
/** Share of the way from `key_branch` towards straight down (≈ 13°: less key on the faces turned to the lens). */
const KEY_TO_ZENITH = 0.3;
/** Weathered bark of the stumps: the J's bark × this (linear), near-black and matt. */
const STUMP_TINT = new Color(0.4, 0.36, 0.32);
/** Fresh wood on the stumps' breaks and torn fibres (COLOR_0.G == 0): pale and warm. */
const FRESH_WOOD = { color: new Color(0.46, 0.33, 0.24), grain: 0.15, roughness: 0.9 };
/** Damp, moss-stained bark where COLOR_0.R is high (next to the moss border, deep cracks). */
const MOSS_STAIN = { darken: 0.5, from: 0.25, to: 0.65 };
/**
 * The J's bark darkens towards the broken end (frame 10: the lower arm in shade, the bend
 * lit): × `darken` up to `full` m from the end, none from `none` m on, along the grain.
 */
const LOWER_ARM = { full: 0.5, none: 1.05, darken: 0.45 };
/** TEXCOORD_0.v units per metre along the grain (asset contract: 1 unit = 0.5 m). */
const V_PER_METRE = 2;
/** Hemisphere fill "up": from above and the camera side. */
const FILL_DIR = new Vector3(0, 0.85, 0.5).normalize();
/** Rim light, direction of travel: from behind the branch, above left, towards the lens. */
const RIM_DIR = new Vector3(0.3, -0.35, 0.89).normalize();
/** Near DOF range (m) while the ball is sharp (branch entry), and the rack back (video s). */
const BALL_SHARP_NEAR: [number, number] = [0.5, 1.25];
const BALL_FOCUS_RACK: [number, number] = [31.6, 33.5];
/** Blur of the outgoing picture (branch → stone): ≈ `px` at `at` m, ramp `span` m long, reached at `share` of the window. */
const OUT_BLUR = { share: 0.5, px: 4, at: 2.5, span: 30 };
/**
 * Screens narrower than the reference. The rig's `fit` keeps the window inside the reference
 * frame, but the broken end lies on its left edge (frame 11), so portrait frames cut it off.
 * Instead the frame is fitted to the subject (`fitNarrow`): the broken end, the stem's right
 * side and the ball's loop from its entry at the top round to the lower right (`loop`
 * phases; after that it leaves past the lens at the lower right, as in clip 09), with
 * `overhang` (plants) / `ballR` (fur) and `margin` of the frame on every side; never a
 * narrower vertical fov than the reference, at most `maxFov`. The top stays within
 * `coverTop` × the reference half-height (> 1) above the axis: plants are scattered for the
 * reference frames + 15 %, higher up the stem would be bare. The extra height goes below the
 * J. Margin and overhang ramp in over `ramp` of aspect below the reference, so 1440 × 1020
 * and anything wider keep the rig's own projection.
 */
const NARROW = { margin: 0.04, ramp: 0.1, maxFov: 78, coverTop: 1.13, overhang: 0.03, ballR: 0.085, loop: [0.8, 1.5] as [number, number], loopStep: 0.025, extremes: 4 };
/**
 * Phones (viewports narrower than `maxWidth` CSS px, the overlay's phone layout): the J is
 * the picture in the upper part of the frame and the copy and the two method cards sit below
 * it on the backdrop (BranchSection.module.css: copy from min(45 %, 100 % − 410 px) of the
 * height, cards at the bottom). The J's lowest point (`extremes` lowest vertices, + `overhang`)
 * therefore stays above min(`share` × height, height − `reserve` px): where it would not (less
 * tall screens), the window grows below the J (the J gets smaller; its top edge stays at
 * `coverTop`, so no bare stem comes into view), up to a vertical fov of `maxFov`. The gap to
 * the copy (≈ 4 % of the height) also absorbs a canvas a little taller than the overlay
 * (100lvh vs 100svh while a phone browser shows its bars). Tablets and wider: as before.
 */
const PHONE = { maxWidth: 700, share: 0.41, reserve: 445, maxFov: 86 } as const;
/** Slow tumble of the ball (ambient clock): axis and rate (rad/s). */
const BALL_SPIN_AXIS = new Vector3(0.35, 1, 0.2).normalize();
const BALL_SPIN_RATE = 0.55;

const tmpV = new Vector3();
const tmpTarget = new Vector3();

export class BranchScene extends BaseScene {
  readonly id = "branch" as const;

  private track: BranchTrack | null = null;
  private loop: BallLoop | null = null;
  private readonly ball = new Group();
  private focus = FOCUS_DISTANCE;
  private readonly keyDir = KEY_DIR.clone();
  private readonly vegetation: { label: string; build: VegetationBuild }[] = [];
  private readonly frustum = new Frustum();
  private readonly projView = new Matrix4();
  private visible: Record<string, number> = {};
  private placeholder = false;
  private placeholderBalls: Mesh[] = [];
  private debug = { phase: 0, ballDepth: 0, tiltDeg: 0, narrow: "rig" };
  /** Narrow screens: world points (with a radius, m) the frame keeps in view; tan(fov / 2) of the poses. */
  private narrow: { p: Vector3; r: number }[] = [];
  /** Phones: the J's lowest points (with a radius, m), kept above the copy (PHONE). */
  private narrowLow: { p: Vector3; r: number }[] = [];
  private narrowTanV = Math.tan(MathUtils.degToRad(35) / 2);

  protected async build(ctx: SceneContext): Promise<void> {
    const [gltf, kit] = await Promise.all([ctx.assets.tryGltf("models/branch.glb"), loadKit(ctx)]);
    if (!gltf) {
      await this.buildPlaceholder(ctx);
      return;
    }

    // own copy of the node tree (geometry stays registry-owned)
    const world = gltf.scene.clone(true);
    world.name = "branch";
    this.scene.add(world);
    world.updateMatrixWorld(true);

    const [bark, jBark, stumpBark, moss, ballCore] = await Promise.all([
      // weathered bark (frame 10, detail 06): warm mid-brown, the light and dark from the
      // modelled strands, creases and cracks and their baked AO; the texture's orange is
      // taken back by a greenish-grey tint, its painted grain kept at 55 % (branch/bark.ts)
      // and repeated twice per UV unit (fibres rather than planks), a little sheen on the ridges
      createBarkMaterial(ctx.assets, { scale: 2, roughness: 0.85, aoStrength: 1, color: "#AAB4A6" }),
      // the J (`wood_branch_j`): the same bark, darker towards the broken end
      createBarkMaterial(ctx.assets, { scale: 2, roughness: 0.85, aoStrength: 1, color: "#AAB4A6" }),
      // the stumps (`wood_branch_stumps`): near-black matt weathered bark, pale warm fresh
      // wood on the breaks and torn fibres (COLOR_0.G == 0; frame 10, detail 06)
      createBarkMaterial(ctx.assets, { scale: 2, roughness: 1, aoStrength: 1, color: "#AAB4A6" }),
      // the base under the fuzz: an olive yellow-green, so gaps between blades stay green;
      // the faces turned down (underside of the J, frame 10) much darker
      createMossLayerMaterial(ctx, {
        tint: "#E8E094",
        brightness: 4.0,
        saturation: 0.8,
        patchScale: 9,
        patchContrast: 0.5,
        sheen: 0.55,
        sheenColor: "#D2DA9C",
        aoContrast: 1.5,
        shade: { top: 1.1, bottom: 0.5, topColor: "#E0DC80", topAmount: 0.2 },
      }),
      // ball core: darker (the hair parts show it, detail 07)
      createMossLayerMaterial(ctx, { tint: "#C8D870", brightness: 1.4, saturation: 1.0, patchScale: 30, patchContrast: 0.35, sheen: 0.4, sheenColor: "#C8D870" }),
    ]);
    const endV = brokenEndV(world);
    const fade = endV ? { vFull: endV.v0 + endV.dir * LOWER_ARM.full * V_PER_METRE, vNone: endV.v0 + endV.dir * LOWER_ARM.none * V_PER_METRE, darken: LOWER_ARM.darken } : undefined;
    for (const [mat, tone] of [
      [bark, { contrast: 0.55, stain: MOSS_STAIN }],
      [jBark, { contrast: 0.55, stain: MOSS_STAIN, fade }],
      [stumpBark, { contrast: 0.7, stain: MOSS_STAIN, fresh: FRESH_WOOD }],
    ] as const) {
      mat.color.multiplyScalar(1.15);
      toneBark(mat, tone);
      mat.normalScale.multiplyScalar(1.4);
    }
    stumpBark.color.multiply(STUMP_TINT);
    this.own(bark);
    this.own(jBark);
    this.own(stumpBark);
    this.own(moss);
    this.own(ballCore);

    const meshes = new Map<string, Mesh>();
    world.traverse((o) => {
      const m = o as Mesh;
      if (m.isMesh) meshes.set(m.name, m);
    });
    const woodByName: Record<string, MeshStandardMaterial> = { wood_branch_j: jBark, wood_branch_stumps: stumpBark };
    for (const [name, mesh] of meshes) {
      if (name.startsWith("wood_")) {
        mesh.material = woodByName[name] ?? bark;
        ctx.layers.assign(mesh, "wood");
      } else if (name === "moss_ball") {
        mesh.material = ballCore;
        ctx.layers.assign(mesh, "moss");
      } else {
        mesh.material = moss;
        ctx.layers.assign(mesh, "moss");
      }
      mesh.castShadow = name !== "moss_ball";
      mesh.receiveShadow = true;
    }

    // --- camera track (extras) ------------------------------------------------
    const found = this.rig.addPosesFromObject(world);
    for (const n of ["branch_main", "branch_p1", "branch_out"]) if (!found.includes(n)) throw new Error(`branch.glb: missing camera cam_${n}`);
    const extras = (name: string): Record<string, unknown> => (world.getObjectByName(name)?.userData ?? {}) as Record<string, unknown>;
    const num = (v: unknown, fallback: number) => (Number.isFinite(Number(v)) && v !== null && v !== undefined ? Number(v) : fallback);
    const eMain = extras("cam_branch_main");
    const eP1 = extras("cam_branch_p1");
    const eOut = extras("cam_branch_out");
    this.focus = num(eMain.focus_distance_m, FOCUS_DISTANCE);
    this.track = new BranchTrack({
      main: this.rig.pose("branch_main") as CameraPose,
      p1: this.rig.pose("branch_p1") as CameraPose,
      out: this.rig.pose("branch_out") as CameraPose,
      mainT: num(eMain.video_time_s, TRACK_DEFAULTS.mainT),
      slide: readSamples(eMain.slide_before_main_fh) ?? TRACK_DEFAULTS.slide,
      drift: readSamples(eP1.drift_from_main) ?? TRACK_DEFAULTS.drift,
      settleT: num(eP1.settle_video_s, TRACK_DEFAULTS.settleT),
      holdT: num(eP1.hold_until_video_s, TRACK_DEFAULTS.holdT),
      outT: num(eOut.video_time_s, TRACK_DEFAULTS.outT),
    });
    this.rig.setTrack(this.track.keys());
    // narrow screens: replaced by fitNarrow once the set is built (kept for the placeholder)
    this.rig.fit = { minHorizontalFraction: 0.74, subjectX: 0.08, maxFov: 62 };

    // --- light ------------------------------------------------------------------------
    const keyEmpty = world.getObjectByName("key_branch");
    if (keyEmpty) this.keyDir.set(0, 0, -1).transformDirection(keyEmpty.matrixWorld);
    // a little more from above than key_branch: the bark and moss turned to the lens get
    // less of it (frame 10: stem bark p50 ≈ 0.36), the tops stay lit
    this.keyDir.lerp(tmpV.set(0, -1, 0), KEY_TO_ZENITH).normalize();
    const key = this.lights.key;
    key.castShadow = true;
    key.shadow.mapSize.set(ctx.quality.shadowMapSize, ctx.quality.shadowMapSize);
    key.shadow.bias = -0.0004;
    key.shadow.normalBias = 0.008;
    key.shadow.camera.near = 0.5;
    key.shadow.camera.far = 14;
    key.shadow.camera.layers.enableAll();
    // sky fill from above and the camera side (the faces towards the lens are lit, frame 10)
    this.lights.fill.position.copy(FILL_DIR);
    // high-key light on the moss (frame 10): a strong key (the pile's surface shade keeps
    // it off the faces turned away, so the cushions and the J's underside stay dark), a
    // slightly weaker sky fill, the back light makes the outlines (moss, ball) glow
    this.lights.scale = { key: 3.0, fill: 0.9, rim: 3.0 };
    // --- the ball ---------------------------------------------------------------
    this.ball.name = "branch_ball";
    world.add(this.ball);
    const ballMesh = meshes.get("moss_ball");
    if (ballMesh) {
      ballMesh.position.set(0, 0, 0);
      ballMesh.quaternion.identity();
      ballMesh.updateMatrix();
      this.ball.add(ballMesh);
    }
    this.ball.updateMatrixWorld(true);
    this.loop = BallLoop.fromObject(world, world);
    // narrow screens: the points the frame is fitted to (fitNarrow), picked as seen from p1
    const p1 = this.rig.pose("branch_p1") as CameraPose;
    this.narrowTanV = Math.tan(MathUtils.degToRad(p1.fov) / 2);
    const subject = ["wood_branch_j", "moss_branch_j", "wood_branch_stumps"].map((n) => meshes.get(n)).filter((m): m is Mesh => !!m);
    const narrow = narrowSubject(subject, this.loop, world, p1, this.narrowTanV);
    this.narrow = narrow.points;
    this.narrowLow = narrow.low;

    // --- vegetation -------------------------------------------------------------
    const views: ScatterView[] = ["branch_main", "branch_p1", "branch_out"].map((n) => this.scatterView(this.rig.pose(n) as CameraPose, 1));
    const pick = (...names: string[]) => names.map((n) => meshes.get(n)).filter((m): m is Mesh => !!m);
    // built in ≈ 8 ms slices (the current episode keeps rendering meanwhile); one after the
    // other, same plants and creation order as a synchronous build; the meshes and the ball
    // group do not move before the set is ready
    this.addVegetation(
      "branch",
      await buildVegetationAsync(ctx, branchMoss, {
        meshes: pick("moss_branch_j", "moss_branch_frag", "wood_branch_j", "wood_branch_stumps", "wood_branch_frag"),
        parent: world,
        views,
        kit,
        label: "branch",
      }),
    );
    if (ballMesh) {
      // the ball turns every side to the lens: one view, every face counts
      const ballView = this.scatterView(CameraRig.lookAtPose([0, 0, 0.6], [0, 0, 0]), 1, { margin: 0.5 });
      this.addVegetation(
        "ball",
        await buildVegetationAsync(ctx, ballFur(ctx.quality.level), {
          meshes: [ballMesh],
          parent: this.ball,
          views: [ballView],
          kit,
          label: "ball",
          importance: { facingMin: -2, facingFull: -1.5 },
        }),
      );
    }
  }

  private addVegetation(label: string, build: VegetationBuild): void {
    this.vegetation.push({ label, build });
    build.report(this.ctx, `${label}.`);
  }

  private scatterView(pose: CameraPose, weight: number, extra: Partial<ScatterView> = {}): ScatterView {
    return { position: pose.position.clone(), quaternion: pose.quaternion.clone(), fov: pose.fov, aspect: REFERENCE_ASPECT, weight, margin: 0.15, ...extra };
  }

  getLook(_frame: FrameState, local: SceneLocal, base: LookParams): LookParams {
    if (this.placeholder) return base;
    base.dof.focusDistance = this.focus;
    // frame 10 (31.5 s): the ball (≈ 1.4 m away) is sharp, combed streaks visible; frames
    // 11 / 12: soft. The near blur ends at 1.25 m on entry and racks back to the look's
    // range by 33.5 s (the stumps, ≥ 2.0 m, stay sharp either way)
    const rack = smoothstep(BALL_FOCUS_RACK[0], BALL_FOCUS_RACK[1], local.t);
    base.dof.nearStart += (BALL_SHARP_NEAR[0] - base.dof.nearStart) * (1 - rack);
    base.dof.nearEnd += (BALL_SHARP_NEAR[1] - base.dof.nearEnd) * (1 - rack);
    // branch → stone (frame 13): the whole outgoing picture goes soft while it slides up.
    // The near-blur ramp is stretched over OUT_BLUR.span metres and moved out so that
    // everything around OUT_BLUR.at (the set: 1.4–3 m) gets ≈ OUT_BLUR.px of blur, nearly
    // uniform; the ball's own soft focus stays as it is. It grows over the first half of
    // the window (37.0 s is the plain look).
    const out = TRANSITIONS.find((c) => c.id === "branch-stone");
    if (out && local.t > out.start && base.dof.maxBlurPx > 0) {
      const w = smoothstep(out.start, out.start + (out.end - out.start) * OUT_BLUR.share, local.t);
      const end = OUT_BLUR.at + (OUT_BLUR.span * OUT_BLUR.px) / base.dof.maxBlurPx;
      base.dof.nearStart += (end - OUT_BLUR.span - base.dof.nearStart) * w;
      base.dof.nearEnd += (end - base.dof.nearEnd) * w;
    }
    return base;
  }

  /**
   * streams → branch: the wipe edge follows the measured smoothed scroll (track.ts), and
   * `k` (the blend of the global grade: vignette, bloom) follows the share of the screen
   * the branch already covers, so the revealed part keeps the branch's own vignette
   * (motion/02: its bottom is dark from the first frames) instead of a linear blend over
   * the whole window. `k` also drives the slow morph of the ragged edge.
   * branch → stone: see below.
   */
  getTransitionOverride(tr: ActiveTransition, frame: FrameState): Partial<TransitionParams> | null {
    if (tr.reduced) return null;
    const cfg = TRANSITIONS.find((c) => c.id === tr.id);
    const tun = { ...TRANSITION_TUNING_DEFAULTS, ...(cfg?.tuning ?? {}) };
    if (tr.id === "streams-branch" && tr.mode === "wipe") {
      const margin = tun.raggedness * 1.45 + tun.softness * 2;
      const edge = wipeEdgeAt(frame.t, tr.end, margin);
      return { edge, k: Math.min(1, Math.max(0, edge * 1.6)) };
    }
    if (tr.id === "branch-stone" && tr.mode === "slide") {
      // branch → stone: the wide soft fade of the trailing edge (frame 13) grows in with
      // the slide, so the frame does not darken and blur its lower third in one step at
      // the start of the window
      const grow = smoothstep(0, 0.3, tr.k);
      return { seamSoftness: Math.max(1e-3, tun.seamSoftness * grow), seamBlurPx: tun.seamBlurPx * grow };
    }
    return null;
  }

  protected animate(frame: FrameState): void {
    if (this.placeholder) {
      this.animatePlaceholder(frame);
      return;
    }
    this.fitNarrow(frame.viewport);
    const cam = this.camera;

    // key light along key_branch, aimed at the focus point; rim from behind
    const target = cam.getWorldDirection(tmpTarget).multiplyScalar(this.focus).add(cam.position);
    this.lights.setKeyDirection(this.keyDir, target, 6);
    this.lights.setRimDirection(RIM_DIR, target, 6);
    const sc = this.lights.key.shadow.camera;
    sc.left = -1.35;
    sc.right = 1.35;
    sc.top = 1.1;
    sc.bottom = -1.1;
    sc.updateProjectionMatrix();

    // the ball: ambient loop + slow tumble (pure functions of the ambient clock)
    if (this.loop) {
      const time = frame.reducedMotion ? this.loop.spec.t0 + (frame.timeSec - this.loop.spec.t0) * 0.25 : frame.timeSec;
      this.loop.position(time, this.ball.position);
      this.ball.quaternion.setFromAxisAngle(BALL_SPIN_AXIS, time * BALL_SPIN_RATE);
      this.debug.phase = this.loop.phase(time);
      this.ball.updateMatrixWorld();
      cam.updateMatrixWorld();
      this.debug.ballDepth = -this.ball.getWorldPosition(tmpV).applyMatrix4(cam.matrixWorldInverse).z;
    }

    // visible instance statistics (debug panel)
    if (frame.frame % 15 === 0 || frame.capture) {
      cam.updateMatrixWorld();
      this.projView.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
      this.frustum.setFromProjectionMatrix(this.projView);
      this.visible = {};
      for (const v of this.vegetation) {
        const n = v.build.visibleCount(this.frustum);
        this.visible[v.label] = n;
        this.ctx.reportInstances(`view.${v.label}`, n);
      }
      const e = tmpV.set(0, 0, -1).applyQuaternion(cam.quaternion);
      this.debug.tiltDeg = (Math.asin(Math.max(-1, Math.min(1, e.y))) * 180) / Math.PI;
    }
  }

  /**
   * Screens narrower than the reference (NARROW): replaces the rig's projection, after the
   * rig has posed the camera, by a window fitted to the subject points, in tan units of the
   * current pose (so the track's tilts still move the picture). Pure function of the camera
   * pose and the viewport; the reference aspect and wider keep the rig's projection. On
   * phones (PHONE) the window also keeps the J's lowest point above the copy.
   */
  private fitNarrow(viewport: Viewport): void {
    const aspect = viewport.aspect;
    const k = clamp((REFERENCE_ASPECT - aspect) / NARROW.ramp, 0, 1);
    if (k <= 0 || this.narrow.length === 0) {
      this.debug.narrow = "rig";
      return;
    }
    const cam = this.camera;
    const tanV = this.narrowTanV;
    let uMin = Infinity;
    let uMax = -Infinity;
    for (const { p, r } of this.narrow) {
      tmpV.copy(p).applyMatrix4(cam.matrixWorldInverse);
      const z = -tmpV.z;
      if (z <= 0.05) continue;
      const pad = (r * k) / z;
      uMin = Math.min(uMin, tmpV.x / z - pad);
      uMax = Math.max(uMax, tmpV.x / z + pad);
    }
    if (!(uMax > uMin)) return;
    const m = NARROW.margin * k;
    const maxH = Math.max(2 * tanV, 2 * Math.tan(MathUtils.degToRad(NARROW.maxFov) / 2));
    let h = Math.min(Math.max(2 * tanV, (uMax - uMin) / (1 - 2 * m) / aspect), maxH);
    const top = Math.min(h / 2, NARROW.coverTop * tanV);
    if (viewport.width < PHONE.maxWidth && viewport.height > 0) {
      // the J's lowest point (tan units, below the axis) at most `share` of the height from the top
      let vMin = Infinity;
      for (const { p, r } of this.narrowLow) {
        tmpV.copy(p).applyMatrix4(cam.matrixWorldInverse);
        const z = -tmpV.z;
        if (z > 0.05) vMin = Math.min(vMin, (tmpV.y - r) / z);
      }
      const share = Math.min(PHONE.share, (viewport.height - PHONE.reserve) / viewport.height);
      const phoneMaxH = 2 * Math.tan(MathUtils.degToRad(PHONE.maxFov) / 2);
      if (Number.isFinite(vMin) && share > 0.1) h = Math.max(h, Math.min((top - vMin) / share, phoneMaxH));
    }
    const w = h * aspect;
    // as close to the camera axis as the subject and its margins allow
    const lo = uMax + m * w - w / 2;
    const hi = uMin - m * w + w / 2;
    const cx = lo <= hi ? clamp(0, lo, hi) : (uMin + uMax) / 2;
    const n = cam.near;
    cam.projectionMatrix.makePerspective((cx - w / 2) * n, (cx + w / 2) * n, top * n, (top - h) * n, n, cam.far, cam.coordinateSystem, cam.reversedDepth);
    cam.projectionMatrixInverse.copy(cam.projectionMatrix).invert();
    this.debug.narrow = `x ${(cx - w / 2).toFixed(3)}..${(cx + w / 2).toFixed(3)} y ${(top - h).toFixed(3)}..${top.toFixed(3)} (tan)`;
  }

  debugInfo(): Record<string, string | number | boolean> {
    const out: Record<string, string | number | boolean> = {};
    for (const v of this.vegetation) {
      out[`${v.label} built`] = Object.entries(v.build.counts)
        .map(([k, n]) => `${k} ${n}`)
        .join(", ");
      out[`${v.label} in view`] = this.visible[v.label] ?? 0;
    }
    out["ball phase"] = Number(this.debug.phase.toFixed(3));
    out["ball depth m"] = Number(this.debug.ballDepth.toFixed(3));
    out["camera tilt°"] = Number(this.debug.tiltDeg.toFixed(2));
    out["narrow frame"] = this.debug.narrow;
    return out;
  }

  // ---------------------------------------------------------------------------
  // PLACEHOLDER fallback (branch.glb missing)
  // ---------------------------------------------------------------------------

  private async buildPlaceholder(ctx: SceneContext): Promise<void> {
    this.placeholder = true;
    const [bark, moss] = await Promise.all([createBarkMaterial(ctx.assets), createMossBaseMaterial(ctx.assets)]);
    this.own(bark);
    this.own(moss);
    const root = new Group();
    root.name = "PLACEHOLDER_branch";
    this.scene.add(root);
    const path = [
      new Vector3(0.16, 1.05, -0.3),
      new Vector3(0.13, 0.45, -0.2),
      new Vector3(0.06, -0.12, -0.1),
      new Vector3(-0.14, -0.42, 0),
      new Vector3(-0.42, -0.46, 0.05),
      new Vector3(-0.62, -0.3, 0.06),
    ];
    placeholderMesh(ctx, root, tubeGeometry(path, (u) => 0.075 - 0.03 * u, { rng: ctx.rng("j"), bumps: 0.16, tubular: 160 }), bark, "wood", { name: "wood_j_branch" });
    [0.18, 0.43, 0.66, 0.84].forEach((u, i) => {
      const p = curvePoint(path, u);
      placeholderMesh(ctx, root, blobGeometry(ctx.rng(`cushion-${i}`), { radius: 0.07 + 0.02 * (i % 2), scale: [1.2, 0.8, 1.1] }), moss, "moss", {
        position: [p.x + 0.02, p.y + 0.03, p.z + 0.05],
        name: `moss_cushion_${i}`,
      });
    });
    placeholderMesh(
      ctx,
      root,
      tubeGeometry([new Vector3(0.48, -0.62, 0.1), new Vector3(0.68, -0.52, 0), new Vector3(0.92, -0.5, -0.12)], (u) => 0.05 - 0.015 * u, { rng: ctx.rng("fragment") }),
      bark,
      "wood",
      { name: "wood_fragment" },
    );
    this.placeholderBalls = [placeholderMesh(ctx, root, blobGeometry(ctx.rng("ball-0"), { radius: 0.085, roughness: 0.4, frequency: 3 }), moss, "moss", { name: "moss_ball_0" })];
    this.lights.setKeyDirection(new Vector3(-0.5, -0.7, -0.5));
    const P = CameraRig.lookAtPose;
    this.rig.setTrack([
      { t: 31, pose: P([0, 0.02, 2.4], [0, 0.02, 0]) },
      { t: 37, pose: P([0.06, 0.04, 2.2], [0.02, 0.02, 0]) },
    ]);
  }

  private animatePlaceholder(frame: FrameState): void {
    this.placeholderBalls.forEach((ball, i) => {
      const a = frame.timeSec * 0.9 + 4.7 + i * Math.PI;
      ball.position.set(0.62 * Math.cos(a), -0.05 + 0.12 * Math.sin(2 * a), 0.55 + 0.95 * Math.sin(a));
      ball.rotation.set(a * 0.7, a, 0);
    });
  }
}

/**
 * TEXCOORD_0.v at the broken end of `wood_branch_j` (the end next to the stumps; V runs
 * along the grain, 1 unit = 0.5 m) and the sign of V towards the stem; null without UVs.
 */
/**
 * World points the narrow-screen frame keeps in view (NARROW), as seen from `pose`: the
 * `extremes` left-most vertices of `meshes` (the broken end) and the right-most ones below
 * the cover top (the stem's right side), radius `overhang`; the ball's loop from `loop[0]` to
 * `loop[1]` (phases, in `space`), radius `ballR`. `low`: the `extremes` lowest vertices of
 * `meshes` (the underside of the J's bend), radius `overhang` (PHONE).
 */
function narrowSubject(
  meshes: Mesh[],
  loop: BallLoop | null,
  space: Object3D,
  pose: CameraPose,
  tanV: number,
): { points: { p: Vector3; r: number }[]; low: { p: Vector3; r: number }[] } {
  const toCam = new Matrix4().compose(pose.position, pose.quaternion, new Vector3(1, 1, 1)).invert();
  type Pick = { score: number; p: Vector3 };
  const left: Pick[] = [];
  const right: Pick[] = [];
  const bottom: Pick[] = [];
  // keeps the NARROW.extremes entries with the highest score, best first
  const keep = (list: Pick[], score: number, p: Vector3) => {
    if (list.length >= NARROW.extremes && score <= list[list.length - 1].score) return;
    const i = list.findIndex((e) => score > e.score);
    list.splice(i < 0 ? list.length : i, 0, { score, p: p.clone() });
    if (list.length > NARROW.extremes) list.pop();
  };
  const w = new Vector3();
  const c = new Vector3();
  for (const mesh of meshes) {
    mesh.updateMatrixWorld(true);
    const pos = mesh.geometry.getAttribute("position");
    for (let i = 0; i < pos.count; i++) {
      w.fromBufferAttribute(pos, i).applyMatrix4(mesh.matrixWorld);
      c.copy(w).applyMatrix4(toCam);
      const z = -c.z;
      if (z <= 0.05) continue;
      keep(left, -c.x / z, w);
      if (c.y / z <= NARROW.coverTop * tanV) keep(right, c.x / z, w);
      keep(bottom, -c.y / z, w);
    }
  }
  const out = [...left, ...right].map((e) => ({ p: e.p, r: NARROW.overhang }));
  if (loop) {
    space.updateMatrixWorld(true);
    for (let ph = NARROW.loop[0]; ph <= NARROW.loop[1] + 1e-6; ph += NARROW.loopStep) {
      const p = loop.curve.getPoint(loop.param(ph - Math.floor(ph)), new Vector3()).applyMatrix4(space.matrixWorld);
      out.push({ p, r: NARROW.ballR });
    }
  }
  return { points: out, low: bottom.map((e) => ({ p: e.p, r: NARROW.overhang })) };
}

function brokenEndV(world: Object3D): { v0: number; dir: number } | null {
  const j = world.getObjectByName("wood_branch_j") as Mesh | undefined;
  const uv = j?.geometry.getAttribute("uv");
  const pos = j?.geometry.getAttribute("position");
  if (!j || !uv || !pos) return null;
  let iMin = 0;
  let iMax = 0;
  for (let i = 1; i < uv.count; i++) {
    if (uv.getY(i) < uv.getY(iMin)) iMin = i;
    if (uv.getY(i) > uv.getY(iMax)) iMax = i;
  }
  const stumps = world.getObjectByName("wood_branch_stumps");
  const near = stumps ? new Box3().setFromObject(stumps).getCenter(new Vector3()) : null;
  const at = (i: number) => j.localToWorld(new Vector3().fromBufferAttribute(pos, i));
  // the broken end is the V end next to the stumps (without them: the lower end)
  const minIsEnd = near ? at(iMin).distanceTo(near) <= at(iMax).distanceTo(near) : at(iMin).y <= at(iMax).y;
  return minIsEnd ? { v0: uv.getY(iMin), dir: 1 } : { v0: uv.getY(iMax), dir: -1 };
}
