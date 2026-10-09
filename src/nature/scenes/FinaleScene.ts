/**
 * Finale scene set — episode `finale` (52.8–59 s): the standing stone with the seed cut
 * through it on a plateau of moss hills, dark far walls, slow dust in the cold air.
 * Appears behind the parting canopy leaves (52.4–53.9 s, mode `over`; the canopy owns the
 * leaves, this set only renders the picture behind them).
 *
 * `models/finale.glb` (Blender: assets-src/blender/build_finale.py):
 *  - `stone_finale`        rock material, tuned pale grey-beige (finale/materials.ts);
 *                          COLOR_0.R = moss in the crack and at the foot → stains + a
 *                          little fuzz (finaleStoneMoss)
 *  - `moss_finale_hills`   a pile of round moss mounds (rev 3): moss base + dense fuzz,
 *                          tufts on the crests, white star flowers, a few sprigs
 *                          (finaleMoss), under the key's pool of light around the stone
 *  - `moss_finale_near__fg` near mound in the lower band and the bank in front of the
 *                          island, out of focus (finaleNearMoss; finaleBankMoss and a
 *                          lighter, greener look on narrow screens)
 *  - `far_finale_left/right` dark leaning trunks and rock lumps with soft moss patches
 *  - cameras `cam_finale_{in,main,out}` (+ extras `focus_distance_m`, `video_time_s`,
 *    `dolly_from_in`), `key_finale` (local −Z = key direction)
 * Camera: holds `in` while incoming (52.4–53.2), dolly in → `main` (53.2–54.3) following
 * `dolly_from_in` with the reference's late tilt, then a slow drift to `out` (57.5), hold
 * (finale/track.ts). Dust: finale/dust.ts. Without finale.glb a small PLACEHOLDER set
 * keeps the episode working.
 */
import {
  Color,
  Frustum,
  Group,
  MathUtils,
  Matrix4,
  Quaternion,
  ShapeGeometry,
  Vector3,
  type BufferGeometry,
  type IUniform,
  type Mesh,
  type MeshStandardMaterial,
  type Object3D,
  type Points,
  type ShaderMaterial,
} from "three";
import { CameraRig, type CameraPose } from "../CameraRig";
import { BaseScene } from "../core/BaseScene";
import { lerp, smoothstep } from "../core/math";
import { REFERENCE_ASPECT, TRANSITIONS, TRANSITION_TUNING_DEFAULTS } from "../SceneConfig";
import { createFarMaterial, createMossBaseMaterial, createRockMaterial } from "../rendering/Materials";
import type { ActiveTransition, FrameState, LookParams, SceneContext, SceneLocal } from "../types";
import { createMossLayerMaterial } from "../vegetation/MossLayer";
import { plantUniforms } from "../vegetation/PlantMaterial";
import type { ScatterView } from "../vegetation/SurfaceScatter";
import { buildVegetationAsync, loadKit, type VegetationBuild } from "../vegetation/Vegetation";
import { buildDust, type Dust } from "./finale/dust";
import { applyLightPool, lightPoolUniforms, tuneFinaleFar, tuneFinaleStone, type LightPool, type LightPoolUniforms } from "./finale/materials";
import { finaleBankMoss, finaleMoss, finaleNearMoss, finaleStoneMoss, NEAR_FUZZ_COLORS } from "./finale/recipes";
import { buildFinaleTrack, readDolly, screenV, TRACK_DEFAULTS, viewDepth, type FinaleTrack } from "./finale/track";
import { blobGeometry, particleCloud, placeholderMesh, pointScale, rockGeometry, seedEmblemShapes } from "./placeholders";

/**
 * Stone reference points of build_finale.py (glTF, Y up): the top point and the middle of
 * the widest row its camera solve put on the measured screen box (STONE_TOP_REL 0.696,
 * STONE_WIDE_REL 0.265 above VIS0 = 0.35 m), and a point behind the cut-through seed
 * (the background glow sits there: frame 19 is lightest seen through the holes).
 */
const STONE_TOP = new Vector3(-0.085, 1.046, 0.02);
const STONE_WIDE_CENTRE = new Vector3(0, 0.615, 0.07);
const GLOW_POINT = new Vector3(0, 0.66, -0.4);
/** Moss crest height under the stone (Z_B, m): the stone's green bounce fades above it. */
const STONE_FOOT_Y = 0.3;

/** DOF near field as fractions of the focus distance (the near mound blurred, the plateau sharp). */
const DOF_NEAR_START = 0.45;
const DOF_NEAR_END = 0.72;

/**
 * Key light, direction of travel (world): ≈ 48° up, from the left and a little in front.
 * `key_finale` suggests (−0.46, −0.73, −0.50), a light from behind the camera on the right
 * that lights every visible face alike. Frame 19 / detail 10: the stone is lit from the
 * upper left (its left third ≈ 1.4× brighter than its right third, the top-left planes
 * brightest), the hills' mounds are lit on their left / upper faces with their right
 * sides falling off (frame 19's crest band is bimodal: lit faces and shadowed sides, not
 * an evenly lit felt); the falloff below the crest and towards the edges is the pool.
 */
const KEY_DIR = new Vector3(0.6, -0.72, -0.25).normalize();

/**
 * The key's pool of light on the plateau (finale/materials.ts applyLightPool): an ellipse
 * on the ground around the stone (world x / z half-axes 1.05 / 0.6 m) — the mounds around
 * the stone fully lit, ≈ 70 % at z 0.6 m, ≈ 40 % at z 0.8 m and 15 % on the plateau's front
 * edge (z ≈ 1 m, the lower moss at 54.3 fades into the dark lower band), ≈ 30 % at the
 * frame's left and right edges (|x| ≈ 1.3 m); the front lip (world y below 0.12 m) down
 * to 20 % more. Not a height falloff over the whole pile: on finale.glb rev 3 (a pile of
 * round mounds) that blackened every hollow between the mounds.
 */
const FINALE_POOL: LightPool = { centre: [0.05, 0.1], axes: [1.05, 0.6], radius: [0.5, 1.5], edge: 0.15, y: [-0.04, 0.12], floor: 0.2 };

/**
 * Rim / kicker light: from the upper left and a little behind (direction of travel) — the
 * bright top-left edge of the stone and the glowing crest outlines of frame 19.
 */
const RIM_DIR = new Vector3(0.55, -0.5, 0.67).normalize();

/**
 * Light on the near mound and the bank in front of the island (`moss_finale_near__fg`): a
 * second, wider pool around the stone, so the bank darkens towards the camera (≈ 75 % at
 * z 2 m, ≈ 40 % under a portrait frame's lower edge at z ≈ 2.9 m) instead of reading as one
 * flat haze; no height term.
 */
const NEAR_POOL: LightPool = { centre: [0.05, 0.1], axes: [2.2, 2.0], radius: [0.5, 1.6], edge: 0.35, y: [-10, -9], floor: 1 };

/**
 * Narrow screens (portrait phones and tablets): the engine widens the vertical fov (69.5°
 * at 375×812) and the frame's lower third looks down onto the bank in front of the island
 * (finale.glb rev 3) instead of 1440×1020's dark lower band. There the bank is lighter and
 * greener and the bottom vignette softer. Weight 1 up to aspect 0.95, 0 from 1.2 up: every
 * landscape screen, 1440×1020 included, renders exactly as without this.
 */
const NARROW_ASPECT: [number, number] = [0.95, 1.2];
const NARROW_VIGNETTE = { bottom: 0.55, bottomSize: 0.24 };
/**
 * CameraRig fit: the taller window of a narrow screen moves up by up to 0.63 × the
 * reference half height (tan units 0.2 at the finale's 35°): on a 375×812 phone the stone
 * sits at ≈ 55–60 % of the height under the dark sky and its trunks, the bank in front of
 * the island takes the lowest ≈ 17 % instead of ≈ 32 %; a 768×1024 tablet keeps the
 * reference frame's lower edge. No effect from aspect 1.41 (the reference) up.
 */
const NARROW_SUBJECT_Y = 0.63;
/** The portrait the bank's plants are scattered for (375×812; covers 768×1024 too). */
const PORTRAIT_ASPECT = 375 / 812;

function narrowWeight(aspect: number): number {
  return aspect >= NARROW_ASPECT[1] ? 0 : 1 - smoothstep(NARROW_ASPECT[0], NARROW_ASPECT[1], aspect);
}

/** Near mound / bank: the landscape look and the narrow-screen look it blends to. */
const NEAR_LOOK = {
  // a dark olive moss under the blur (it was a grey-brown haze)
  moss: { tint: "#A8B47C", saturation: 0.85, patchContrast: 0.3, sheen: 0.08, shade: { top: 1.25, bottom: 0.55, topColor: "#B4C460", topAmount: 0.2 } },
  narrow: { tint: "#C4D08A", brightness: 1.35, saturation: 1.0 },
  fuzzNarrow: { root: "#1E2A0A", mid: "#4A5C1C", tip: "#8A9C40", tip2: "#7A8C36" },
};

/**
 * Brightness of the finale image over time, as a linear (pre-tone-mapping) factor of the
 * full picture (frame 19). Measured on frames 18–20, motion/04 and clip 11 (display-linear
 * luminance relative to frame 19) and converted through the ACES curve — its toe crushes
 * dark values much harder than bright ones, so the dark sky needs a larger factor than the
 * lit island to reach the same share of its final brightness:
 *  - entry, the island (moss, stone): smoothstep(52.95, 54.05, t) — 53.2 ≈ 0.13 (frame 18:
 *    moss p90 at 4 %, stone p50 at 1.7 % of frame 19's display luminance), 53.6 ≈ 0.64
 *    (motion/04 ≈ 0.55 displayed), 53.9 ≈ 0.95, 54.05 = 1;
 *  - entry, the background (sky behind the stone): earlier, smoothstep(52.85, 53.95, t) —
 *    53.2 ≈ 0.24 (frame 18: 9 % of frame 19's display luminance; at the island's factor the
 *    toe would leave it pure black);
 *  - footer (57.5, frame 20): no dimming. Frame 20 against frame 19 on matched regions: the
 *    moss crest (luma p90 193 / 192), the stone's middle (display-linear 0.158 / 0.163) and
 *    the sky outside the glow are unchanged; only the stone's top is darker (0.091 /
 *    0.171), which the top vignette gives at the drifted framing (stone top at ≈ 19 % of
 *    the frame height).
 * The composite already scales the incoming set by its reveal (SceneConfig revealSpan of
 * canopyClose-finale); the exposure supplies target / reveal (capped), the background stops
 * the remaining background / island ratio.
 */
const ENTRY_ISLAND: [number, number] = [52.95, 54.05];
/**
 * The first glimpse: from ≈ 52.9 s the island is faintly there between the parting leaves
 * (motion/04 at 53.0 s: dim hills in the hole, ≈ 4 % of the final display luminance) — at
 * least `level` of the island curve, reached over `span`; the smoothstep above takes over
 * from ≈ 53.15 s (frame 18 at 53.2 s unchanged).
 */
const ENTRY_GLIMPSE = { span: [52.8, 53.1] as [number, number], level: 0.06 };
const ENTRY_BACKGROUND: [number, number] = [52.85, 53.95];
const ENTRY_MAX_GAIN = 4;

/** Reveal of the incoming set in the composite (SceneDirector: smoothstep(revealSpan, k)). */
function compositeReveal(tr: ActiveTransition): number {
  const cfg = TRANSITIONS.find((c) => c.id === tr.id);
  const span = cfg?.tuning?.revealSpan ?? TRANSITION_TUNING_DEFAULTS.revealSpan;
  return smoothstep(span[0], span[1], tr.k);
}

/** Entry exposure and background gain of the finale view (pure function of local). */
function entryLight(local: SceneLocal): { exposure: number; background: number } {
  const tr = local.transition;
  // reduced motion: a plain crossfade (and local.t may be the destination's time)
  if (tr && tr.reduced) return { exposure: 1, background: 1 };
  const island = Math.max(smoothstep(ENTRY_ISLAND[0], ENTRY_ISLAND[1], local.t), ENTRY_GLIMPSE.level * smoothstep(ENTRY_GLIMPSE.span[0], ENTRY_GLIMPSE.span[1], local.t));
  const sky = smoothstep(ENTRY_BACKGROUND[0], ENTRY_BACKGROUND[1], local.t);
  const reveal = tr && tr.mode === "over" && tr.toSet === "finale" ? compositeReveal(tr) : 1;
  return {
    exposure: reveal > 1e-4 ? Math.min(ENTRY_MAX_GAIN, island / reveal) : 0,
    background: island > 1e-4 ? Math.min(ENTRY_MAX_GAIN, sky / island) : 1,
  };
}

const tmpTarget = new Vector3();
const tmpColor = new Color();
const FUZZ_COLOR_UNIFORMS = [
  ["uRootColor", "root"],
  ["uMidColor", "mid"],
  ["uTipColor", "tip"],
  ["uTip2Color", "tip2"],
] as const;

export class FinaleScene extends BaseScene {
  readonly id = "finale" as const;

  private placeholder = false;
  private motes: Points<BufferGeometry, ShaderMaterial> | null = null;
  private dust: Dust | null = null;
  private readonly pool: LightPoolUniforms = lightPoolUniforms(FINALE_POOL);
  private readonly nearPool: LightPoolUniforms = lightPoolUniforms(NEAR_POOL);
  /** Near mound / bank material and fuzz with their landscape and narrow-screen values. */
  private near: { material: MeshStandardMaterial; moss: Record<string, IUniform>; base: Color; narrow: Color; fuzz: Record<string, IUniform>[] } | null = null;
  /** Plants on the bank, scattered for portrait views: drawn on narrow screens only. */
  private bank: VegetationBuild | null = null;
  /** Narrow-screen weight the near look is set to (0 = as built). */
  private narrowApplied = 0;
  private track: FinaleTrack | null = null;
  /** World point the DOF focus follows (on the stone's front face, from the camera extras). */
  private readonly focusPoint = new Vector3(0, 0.68, 0.125);
  private readonly keyDir = new Vector3(-0.4613, -0.732, -0.5014).normalize();
  private readonly vegetation: { label: string; build: VegetationBuild }[] = [];
  private readonly tmpPose: CameraPose = { position: new Vector3(), quaternion: new Quaternion(), fov: 35 };
  private readonly frustum = new Frustum();
  private readonly projView = new Matrix4();
  private visible: Record<string, number> = {};
  private info: Record<string, string | number> = {};

  protected async build(ctx: SceneContext): Promise<void> {
    const [gltf, kit] = await Promise.all([ctx.assets.tryGltf("models/finale.glb"), loadKit(ctx)]);
    if (!gltf) {
      await this.buildPlaceholder(ctx);
      return;
    }

    // own copy of the node tree (geometry stays registry-owned)
    const world = gltf.scene.clone(true);
    world.name = "finale";
    this.scene.add(world);
    world.updateMatrixWorld(true);

    const [stone, hills, near] = await Promise.all([
      // warm pale beige, matte, the rock set at 1.8 repeats per metre; its cavity AO at 0.55:
      // frame 19 / detail 10 read smooth with bright fine details, not dark pitting
      createRockMaterial(ctx.assets, { scale: 1.8, roughness: 1, color: "#C8C4B0", aoStrength: 0.55 }),
      // under the fuzz (≈ 80 % of the moss pixels at 54.3 are this base between the shoots):
      // darker olive than the shoots, lighter and yellower on the tops — the gaps in the pile
      // read as depth, the shoots carry the highlights
      createMossLayerMaterial(ctx, {
        tint: "#ECF6A0",
        brightness: 1.1,
        saturation: 1.25,
        patchContrast: 0.25,
        sheen: 0.15,
        shade: { top: 1.6, bottom: 0.6, topColor: "#C8D864", topAmount: 0.35 },
        aoStrength: 0.45,
        aoContrast: 0.4,
      }),
      // near mound and bank: dark olive moss, a shape under the blur (NEAR_LOOK)
      createMossLayerMaterial(ctx, NEAR_LOOK.moss),
    ]);
    tuneFinaleStone(stone, {
      saturation: 0.35,
      mottle: 0.08,
      mossColor: "#4A5220",
      // the crack and the foot: an olive stain, not a green stripe down the face (detail 10)
      moss: 0.45,
      bounce: 0.5,
      bounceColor: "#C8D890",
      footY: STONE_FOOT_Y,
      normal: 0.25,
    });
    // dark teal trunks and rocks, lit alike from both sides by the haze (emissive), the key
    // adds dark green moss on the faces it reaches (the right rock read as a black hole
    // without it); the moss in soft round patches (ribLength ≈ ribScale): long slanted
    // streaks read as light beams in the dark
    const far = createFarMaterial({ color: "#030405", roughness: 1, aoStrength: 0.7 });
    tuneFinaleFar(far, {
      ribs: 0.35,
      ribScale: 1.6,
      ribSharpness: 1.4,
      ribLength: 1.4,
      slant: 0.45,
      mossColor: "#1C2814",
      moss: 0.7,
      patches: 0.45,
      emissive: "#1A2024",
      mossGlow: "#14200F",
      specular: 0,
    });
    applyLightPool(hills, this.pool);
    applyLightPool(near, this.nearPool);
    this.near = {
      material: near,
      moss: near.userData.mossLayerUniforms as Record<string, IUniform>,
      base: near.color.clone(),
      narrow: new Color(NEAR_LOOK.narrow.tint).multiplyScalar(NEAR_LOOK.narrow.brightness),
      fuzz: [],
    };
    for (const m of [stone, hills, near, far]) this.own(m);

    const meshes = new Map<string, Mesh>();
    world.traverse((o) => {
      const m = o as Mesh;
      if (m.isMesh) meshes.set(m.name, m);
    });
    for (const [name, mesh] of meshes) {
      if (name.startsWith("stone_")) {
        mesh.material = stone;
        ctx.layers.assign(mesh, "stone");
        mesh.castShadow = true;
        mesh.receiveShadow = true;
      } else if (name.startsWith("moss_")) {
        const fg = name.endsWith("__fg");
        mesh.material = fg ? near : hills;
        ctx.layers.assign(mesh, "moss");
        mesh.castShadow = !fg;
        mesh.receiveShadow = !fg;
      } else if (name.startsWith("far_")) {
        mesh.material = far;
        ctx.layers.assign(mesh, "far");
        mesh.castShadow = false;
        mesh.receiveShadow = false;
      }
    }

    // --- camera track ------------------------------------------------------------
    const found = this.rig.addPosesFromObject(world);
    for (const n of ["finale_in", "finale_main", "finale_out"]) if (!found.includes(n)) throw new Error(`finale.glb: missing camera cam_${n}`);
    const extras = (name: string): Record<string, unknown> => {
      let out: Record<string, unknown> = {};
      world.traverse((o) => {
        if (o.name === name) out = o.userData ?? {};
      });
      return out;
    };
    const exIn = extras("cam_finale_in");
    const exMain = extras("cam_finale_main");
    const exOut = extras("cam_finale_out");
    const num = (v: unknown, fallback: number) => (Number.isFinite(Number(v)) && v !== undefined && v !== null ? Number(v) : fallback);
    const pIn = this.rig.pose("finale_in") as CameraPose;
    const pMain = this.rig.pose("finale_main") as CameraPose;
    const pOut = this.rig.pose("finale_out") as CameraPose;
    this.track = buildFinaleTrack({
      in: pIn,
      main: pMain,
      out: pOut,
      tIn: num(exIn.video_time_s, TRACK_DEFAULTS.tIn),
      tMain: num(exMain.video_time_s, TRACK_DEFAULTS.tMain),
      tOut: num(exOut.video_time_s, TRACK_DEFAULTS.tOut),
      dolly: readDolly(exMain.dolly_from_in) ?? TRACK_DEFAULTS.dolly,
      stoneTop: STONE_TOP,
      stoneCentre: STONE_WIDE_CENTRE,
    });
    this.rig.fit.subjectX = 0;
    this.rig.fit.subjectY = NARROW_SUBJECT_Y;
    this.rig.setTrack(this.track.keys);
    // DOF focus: the point `focus_distance_m` in front of the main camera (on the stone's
    // front face); every other pose focuses on the same point
    const fMain = num(exMain.focus_distance_m, 3.853);
    this.focusPoint.set(0, 0, -1).applyQuaternion(pMain.quaternion).multiplyScalar(fMain).add(pMain.position);
    this.info = {
      "focus in/out (extras)": `${num(exIn.focus_distance_m, 0).toFixed(2)} / ${num(exOut.focus_distance_m, 0).toFixed(2)}`,
      "focus in/out (point)": `${viewDepth(pIn, this.focusPoint).toFixed(2)} / ${viewDepth(pOut, this.focusPoint).toFixed(2)}`,
    };

    // --- light ---------------------------------------------------------------------
    const keyEmpty = world.getObjectByName("key_finale");
    if (keyEmpty) this.info.key_finale = new Vector3(0, 0, -1).transformDirection(keyEmpty.matrixWorld).toArray().map((v) => v.toFixed(2)).join(", ");
    this.keyDir.copy(KEY_DIR);
    this.lights.setKeyDirection(this.keyDir, STONE_WIDE_CENTRE, 8);
    this.lights.setRimDirection(RIM_DIR, STONE_WIDE_CENTRE, 8);
    this.lights.fill.position.set(0, 1, 0);
    const key = this.lights.key;
    key.castShadow = true;
    key.shadow.mapSize.set(ctx.quality.shadowMapSize, ctx.quality.shadowMapSize);
    key.shadow.bias = -0.0004;
    key.shadow.normalBias = 0.01;
    const sc = key.shadow.camera;
    // the whole plateau seen by the three poses (x −3.2…3.3 m, depth −1.6…1.5 m)
    sc.left = -3.6;
    sc.right = 3.6;
    sc.top = 2.6;
    sc.bottom = -2.6;
    sc.near = 1;
    sc.far = 16;
    sc.updateProjectionMatrix();
    sc.layers.enableAll();

    // --- vegetation ------------------------------------------------------------------
    const pose = (n: string) => this.rig.pose(n) as CameraPose;
    const along = (t: number) => this.rig.evaluate(t, { position: new Vector3(), quaternion: new Quaternion(), fov: 35 });
    const views: ScatterView[] = [
      this.scatterView(pose("finale_in"), 0.55),
      this.scatterView(along(53.7), 0.8),
      this.scatterView(pose("finale_main"), 1),
      this.scatterView(pose("finale_out"), 1),
    ];
    const hillsMesh = meshes.get("moss_finale_hills");
    const stoneMesh = meshes.get("stone_finale");
    const nearMesh = meshes.get("moss_finale_near__fg");
    if (hillsMesh) {
      const build = await buildVegetationAsync(ctx, finaleMoss(ctx.quality), { meshes: [hillsMesh], parent: world, views, kit, label: "finale" });
      for (const b of [...build.grass, ...build.kit]) applyLightPool(b.material, this.pool);
      this.addVegetation("hills", build);
    }
    if (stoneMesh) {
      this.addVegetation("stone", await buildVegetationAsync(ctx, finaleStoneMoss, { meshes: [stoneMesh], parent: world, views, kit, label: "finale-stone" }));
    }
    if (nearMesh) {
      const nearViews = [this.scatterView(pose("finale_main"), 1), this.scatterView(pose("finale_out"), 1)];
      const build = await buildVegetationAsync(ctx, finaleNearMoss, { meshes: [nearMesh], parent: world, views: nearViews, kit, label: "finale-fg" });
      for (const b of build.grass) {
        applyLightPool(b.material, this.nearPool);
        this.near?.fuzz.push(plantUniforms(b.material));
      }
      this.addVegetation("near", build);
      // narrow screens look down onto the bank (NARROW_SUBJECT_Y): its own plants, scattered
      // for the portrait frustum of every pose and hidden on wider screens
      const bankViews = [pose("finale_in"), along(53.7), pose("finale_main"), pose("finale_out")].map((p) => this.portraitView(p));
      const bank = await buildVegetationAsync(ctx, finaleBankMoss, { meshes: [nearMesh], parent: world, views: bankViews, kit, label: "finale-bank" });
      for (const b of bank.grass) applyLightPool(b.material, this.nearPool);
      bank.group.visible = false;
      this.bank = bank;
      this.addVegetation("bank", bank);
    }

    // --- dust ------------------------------------------------------------------------
    const dust = buildDust(ctx.rng("dust"), {
      count: Math.round(ctx.quality.particles * 0.08),
      box: { min: new Vector3(-3.4, -0.1, -2.6), max: new Vector3(3.4, 2.7, 5.2) },
      lightCentre: new Vector3(0, 0.75, 0.2),
      lightRadius: 1.6,
      color: "#EEF3F6",
      intensity: 9,
      drift: new Vector3(0.011, 0.004, 0.003),
    });
    this.dust = dust;
    world.add(dust.points);
    ctx.layers.assign(dust.points, "particles");
    ctx.reportInstances("particles", dust.count);
  }

  /** Near mound / bank look for narrow-screen weight w (0 = the built landscape values, untouched). */
  private applyNarrow(w: number): void {
    const n = this.near;
    if (!n || w === this.narrowApplied) return;
    this.narrowApplied = w;
    n.material.color.copy(n.base).lerp(n.narrow, w);
    n.moss.uMossSaturation.value = lerp(NEAR_LOOK.moss.saturation, NEAR_LOOK.narrow.saturation, w);
    for (const u of n.fuzz) {
      for (const [name, key] of FUZZ_COLOR_UNIFORMS) (u[name].value as Color).set(NEAR_FUZZ_COLORS[key]).lerp(tmpColor.set(NEAR_LOOK.fuzzNarrow[key]), w);
    }
  }

  private addVegetation(label: string, build: VegetationBuild): void {
    this.vegetation.push({ label, build });
    build.report(this.ctx, `${label}.`);
  }

  /** A pose as a portrait screen sees it (the rig's aspect fit at PORTRAIT_ASPECT, no lens shift). */
  private portraitView(pose: CameraPose): ScatterView {
    const fit = this.rig.fit;
    const tanV = Math.tan(MathUtils.degToRad(pose.fov) / 2);
    const need = (tanV * REFERENCE_ASPECT * fit.minHorizontalFraction) / PORTRAIT_ASPECT;
    const tan = Math.min(Math.max(tanV, need), Math.max(tanV, Math.tan(MathUtils.degToRad(fit.maxFov) / 2)));
    return { position: pose.position.clone(), quaternion: pose.quaternion.clone(), fov: MathUtils.radToDeg(2 * Math.atan(tan)), aspect: PORTRAIT_ASPECT, weight: 1, margin: 0.15 };
  }

  private scatterView(pose: CameraPose, weight: number): ScatterView {
    return { position: pose.position.clone(), quaternion: pose.quaternion.clone(), fov: pose.fov, aspect: REFERENCE_ASPECT, weight, margin: 0.15 };
  }

  /** DOF focus distance at video time t: depth of the focus point from the pose at t. */
  private focusAt(t: number): number {
    return Math.max(0.5, viewDepth(this.rig.evaluate(t, this.tmpPose), this.focusPoint));
  }

  getLook(frame: FrameState, local: SceneLocal, base: LookParams): LookParams {
    if (this.placeholder) return base;
    const narrow = narrowWeight(frame.viewport.aspect);
    if (narrow > 0) {
      base.vignette.bottom = lerp(base.vignette.bottom, NARROW_VIGNETTE.bottom, narrow);
      base.vignette.bottomSize = lerp(base.vignette.bottomSize, NARROW_VIGNETTE.bottomSize, narrow);
    }
    const t = local.t;
    const f = this.focusAt(t);
    base.dof.focusDistance = f;
    base.dof.nearStart = f * DOF_NEAR_START;
    base.dof.nearEnd = f * DOF_NEAR_END;
    // the cold glow stays behind the stone while the camera moves
    const pose = this.rig.evaluate(t, this.tmpPose);
    base.background.glow.x = 0.5;
    base.background.glow.y = 1 - screenV(pose, GLOW_POINT);
    const entry = entryLight(local);
    base.exposure *= entry.exposure;
    if (entry.background !== 1) {
      const bg = base.background;
      bg.top.multiplyScalar(entry.background);
      bg.mid.multiplyScalar(entry.background);
      bg.bottom.multiplyScalar(entry.background);
      bg.glow.intensity *= entry.background;
    }
    return base;
  }

  protected animate(frame: FrameState, local: SceneLocal): void {
    if (this.placeholder) {
      if (this.motes) {
        this.motes.material.uniforms.uTime.value = frame.timeSec;
        this.motes.material.uniforms.uScale.value = pointScale(frame.viewport.height, frame.viewport.dpr, this.camera.fov);
      }
      return;
    }
    const cam = this.camera;
    const narrow = narrowWeight(frame.viewport.aspect);
    this.applyNarrow(narrow);
    if (this.bank) this.bank.group.visible = narrow > 0;
    if (this.dust) {
      const u = this.dust.uniforms;
      const pxScale = pointScale(frame.viewport.height, frame.viewport.dpr, cam.fov);
      // reference px (frame height 1020) → drawing-buffer px
      const ref = (frame.viewport.height * frame.viewport.dpr) / 1020;
      u.uTime.value = frame.timeSec;
      u.uScale.value = pxScale;
      // frame 19: the specks are soft 2–3 px dots
      u.uMinPx.value = 2.4 * ref;
      u.uMaxPx.value = 56 * ref;
      u.uCocNear.value = 16 * ref;
      u.uCocFar.value = 1.2 * ref;
      u.uFocus.value = this.focusAt(local.t);
    }

    // visible instance statistics (debug panel)
    if (frame.frame % 15 === 0 || frame.capture) {
      this.projView.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
      this.frustum.setFromProjectionMatrix(this.projView);
      this.visible = {};
      for (const v of this.vegetation) {
        const n = v.build.visibleCount(this.frustum);
        this.visible[v.label] = n;
        this.ctx.reportInstances(`view.${v.label}`, n);
      }
      this.info.focus = this.focusAt(local.t).toFixed(2);
      this.info["dolly share"] = this.track ? this.track.dollyAt(local.t).toFixed(3) : "-";
      this.info["stone top v"] = screenV(this.rig.evaluate(local.t, this.tmpPose), STONE_TOP).toFixed(3);
      const target = cam.getWorldDirection(tmpTarget);
      this.info.pitch = ((Math.asin(Math.max(-1, Math.min(1, target.y))) * 180) / Math.PI).toFixed(2);
    }
  }

  debugInfo(): Record<string, string | number | boolean> {
    const out: Record<string, string | number | boolean> = { ...this.info };
    for (const v of this.vegetation) {
      out[`${v.label} built`] = Object.entries(v.build.counts)
        .map(([k, n]) => `${k} ${n}`)
        .join(", ");
      out[`${v.label} in view`] = this.visible[v.label] ?? 0;
    }
    if (this.dust) out.dust = this.dust.count;
    return out;
  }

  // ---------------------------------------------------------------------------
  // PLACEHOLDER fallback (finale.glb missing)
  // ---------------------------------------------------------------------------

  private async buildPlaceholder(ctx: SceneContext): Promise<void> {
    this.placeholder = true;
    const [moss, rock] = await Promise.all([createMossBaseMaterial(ctx.assets), createRockMaterial(ctx.assets)]);
    const far = createFarMaterial({ color: "#0E1214" });
    const hole = createFarMaterial({ color: "#020304", roughness: 1 });
    const root = new Group();
    root.name = "PLACEHOLDER_finale";
    this.scene.add(root);

    // moss hills
    placeholderMesh(ctx, root, blobGeometry(ctx.rng("hill-a"), { radius: 1.6, roughness: 0.14, scale: [1.5, 0.42, 1], flattenBottom: 0.6 }), moss, "moss", {
      position: [-0.6, -0.95, -0.2],
      name: "moss_hill_a",
    });
    placeholderMesh(ctx, root, blobGeometry(ctx.rng("hill-b"), { radius: 1.3, roughness: 0.14, scale: [1.4, 0.4, 1], flattenBottom: 0.6 }), moss, "moss", {
      position: [1.7, -0.85, -0.6],
      name: "moss_hill_b",
    });
    placeholderMesh(ctx, root, blobGeometry(ctx.rng("hill-c"), { radius: 1.2, roughness: 0.16, scale: [1.8, 0.3, 0.9], flattenBottom: 0.6 }), moss, "moss", {
      position: [0.3, -1.25, 1.2],
      name: "moss_hill_front",
    });

    // standing stone with the seed cut through it (dark inlay stands in for the hole)
    placeholderMesh(ctx, root, rockGeometry(ctx.rng("stone"), [0.95, 1.5, 0.38], 7), rock, "stone", {
      position: [0.05, 0.5, 0.05],
      rotation: [0, 0.08, -0.03],
      name: "stone_standing",
    });
    const cut: Object3D = new Group();
    cut.name = "PLACEHOLDER_emblem_cut";
    cut.position.set(0.05, 0.55, 0.32);
    root.add(cut);
    for (const [i, shape] of seedEmblemShapes(0.62).entries()) {
      placeholderMesh(ctx, cut, new ShapeGeometry(shape, 24), hole, "emblem", { name: `emblem_cut_${i}` });
    }

    // far dark forms
    [
      { p: [-5.2, -0.6, -13], r: 2.4 },
      { p: [5.6, -0.3, -14], r: 2.8 },
      { p: [0.9, -1.6, -17], r: 3.2 },
    ].forEach((f, i) =>
      placeholderMesh(ctx, root, blobGeometry(ctx.rng(`far-${i}`), { radius: f.r, roughness: 0.2 }), far, "far", {
        position: f.p as [number, number, number],
        name: `far_${i}`,
      }),
    );

    // small white motes
    const rng = ctx.rng("motes");
    this.motes = particleCloud(rng, {
      count: Math.min(600, ctx.quality.particles),
      place: (_i, out) => out.set(rng.range(-3, 3), rng.range(-0.6, 2.4), rng.range(-2.5, 2)),
      size: [0.004, 0.01],
      colors: ["#F2F6F8", "#DDE8EE"],
      brightness: [0.25, 0.8],
      hot: 3,
      drift: 0.04,
    });
    root.add(this.motes);
    ctx.layers.assign(this.motes, "particles");
    ctx.reportInstances("particles", this.motes.geometry.getAttribute("position").count);

    this.lights.setKeyDirection(new Vector3(-0.5, -0.6, -0.62), new Vector3(0, 0.4, 0), 8);
    this.lights.setRimDirection(new Vector3(0.35, -0.25, 0.9), new Vector3(0, 0.4, 0), 8);
    const P = CameraRig.lookAtPose;
    this.rig.setTrack([
      { t: 52.8, pose: P([0, 0.9, 5.2], [0, 0.5, 0]) },
      { t: 54.3, pose: P([0, 0.72, 4.3], [0, 0.45, 0]) },
      { t: 59, pose: P([0, 0.66, 4.0], [0, 0.45, 0]) },
    ]);
  }
}
