/**
 * Stone scene set — episode `stone` (37–43.3 s): the dark episode between the light
 * branch and the canopy. Frames 13 (38.5 s, rising from below while the branch slides
 * up) and 14 (41.7 s); the exit (43.0 s on) in motion/03 and clip sheet 10.
 *
 * `models/stone.glb` (Blender: assets-src/blender/build_stone.py):
 *  - `rock_stone_slab`  the diagonal slab with fissure, chips, cracks and the walls of
 *    the seed recess → triplanar rock (grain from the material: scale / roughness),
 *    recess walls tinted golden below the emblem plane (stone/surfaces.ts)
 *  - `emblem_stone`     the recess floor (≈ 3.6 cm deep, COLOR_0.B = contact shadow) →
 *    emblem material + fine sand grain (casts no shadow; no normal offset in its lookup)
 *  - `moss_stone_edge`  the moss strip on the lit top edge → moss layer + `stoneMoss`
 *    vegetation (stone/recipes.ts)
 *  - `cam_stone_{in,main,out}` one position, pedestal only; extras: focus distance,
 *    frame height, `exit_track` (measured upward travel of the picture 43.0–43.6 s)
 *  - `key_stone` empty (local −Z = key direction) with `light_target_gltf` (the emblem
 *    point) and `falloff_radius_m`
 *
 * Camera / transitions: the camera holds `stone_main` for the whole set. The picture
 * is moved only by the composite slides: branch → stone (37.0–40.0, `branch-stone`)
 * brings it up from below, stone → canopy (`stone-canopy`, SceneConfig) carries it out
 * on the measured `exit_track`. Frame 13 has the branch picture 0.5 frame heights up
 * but the stone picture only ≈ 0.3 low (measured: 305 px): the stone picture follows
 * the branch at ENTRY_FOLLOW of its travel (getTransitionOverride, incoming side). The
 * `in` / `out` poses are not used (they would move the picture a second time).
 *
 * Light: a warm local pool along `key_stone` — a shadowed SpotLight (recess walls,
 * moss) shaped by a code-made cookie (stone/pool.ts, tuned to frame 14): full on the
 * emblem face, falling off to the foot and the right part, none on the rock under the
 * moss strip; the rest of the slab falls to black. A faint unshadowed key from the same
 * side (rig key), a dark warm hemisphere fill, a saturated golden bounce in the recess
 * (walls and the floor in the lip's shadow). The moss strip has its own light: the top
 * light (from above and behind: it misses the faces turned to the lens; the rig rim is
 * kept faint so the rock under the right fringe stays black, the moss gets a moss-only
 * top light) and a moss-only key along `key_stone` that fades along the strip
 * (stone/surfaces.ts addMossLight — the cookie keeps the rock under it dark). Frame 13
 * (38.5 s) shows the rock already lit but no moss on the rim: the moss light and the top
 * light ramp up from dark at 38.5 s to full at 40.5 s (pure function of local.t). The
 * yellow-green haze along the top is a code-made glow (stone/haze.ts).
 * Without stone.glb a small PLACEHOLDER set keeps the episode working.
 */
import { Color, ExtrudeGeometry, Frustum, Group, Matrix4, SpotLight, Vector3, type BufferGeometry, type Mesh, type Object3D } from "three";
import { CameraRig, type CameraPose } from "../CameraRig";
import { BaseScene } from "../core/BaseScene";
import { ease, smoothstep, windowProgress } from "../core/math";
import { REFERENCE_ASPECT, TRANSITIONS } from "../SceneConfig";
import { createEmblemMaterial, createMossBaseMaterial, createRockMaterial } from "../rendering/Materials";
import type { ActiveTransition, FrameState, LookParams, SceneContext, SceneLocal, TransitionParams } from "../types";
import { createMossLayerMaterial } from "../vegetation/MossLayer";
import type { ScatterView } from "../vegetation/SurfaceScatter";
import { buildVegetationAsync, loadKit, type VegetationBuild } from "../vegetation/Vegetation";
import { blobGeometry, placeholderMesh, rockGeometry, seedEmblemShapes } from "./placeholders";
import { buildHaze, type Haze } from "./stone/haze";
import { createPoolCookie, type PoolShape } from "./stone/pool";
import { stoneMoss } from "./stone/recipes";
import { addDensityShade, addMossLight, addRecessTint, addSandGrain, createMossLightUniforms, dropShadowNormalOffset } from "./stone/surfaces";

/** Fallbacks for the stone.glb extras. */
const FOCUS_DISTANCE = 3.49;
const KEY_DIR = new Vector3(0.551, -0.732, -0.401).normalize();
const KEY_TARGET = new Vector3(0.814, -0.4996, 0);
const POOL_RADIUS = 1.1;
/** Measured upward travel of the picture during the exit (video s, frame heights). */
const EXIT_TRACK: [number, number][] = [
  [43.0, 0],
  [43.1, 0.0745],
  [43.2, 0.284],
  [43.3, 0.488],
  [43.4, 0.641],
  [43.5, 0.753],
  [43.6, 0.839],
];

/**
 * branch → stone: share of the branch picture's travel the stone picture follows
 * (frame 13: branch up 0.5 fh, stone 0.30 fh low ⇒ 0.3 / 0.5).
 */
const ENTRY_FOLLOW = 0.6;
/** Distance of the pool spot from the emblem point along the key direction (m). */
const POOL_DISTANCE = 3.2;
/** Pool irradiance at the emblem point relative to the look's key intensity. */
const POOL_GAIN = 1.1;
/**
 * Shape of the pool (light cookie, metres at the emblem point; u ≈ screen right, v ≈
 * screen up-right — see stone/pool.ts). The cookie square spans ± falloff_radius_m
 * (stone.glb). First fitted (least squares, linear light, 20 px cells) to frame 14
 * against captures with a flat pool and without it; for stone.glb rev 2 re-tuned on the
 * luma of the lit face (rock pixels round the emblem, frame 14 vs capture: p25 / p50 /
 * p75 / p90 of the lit side, below the emblem, right of it, under the moss): brighter
 * on the emblem face and its left edge, a short fall-off to the foot of the slab (no
 * bottom vignette in this look), a quicker one to the right part, and none on the rock
 * under the moss strip (the half above the emblem fades out within ≈ 0.3 m).
 */
const POOL_SHAPE: Omit<PoolShape, "extent"> = {
  center: [-0.08, 0.08],
  radius: [0.855, 0.285, 0.34, 1.14],
  cut: { dir: [-0.384, 0.923], from: 0, to: 0.3, strength: 1 },
};
/** Unshadowed key from the same side (rig key), relative to the look's key intensity. */
const KEY_BASE = 0.05;
/**
 * Top light (rig rim), direction of travel: from above, a little from behind and the
 * right — it lights the moss on the top edge and misses the faces turned to the lens.
 * On the rock it stays faint (RIM_ROCK of the look's rim: frame 14 is black under the
 * right fringe); the moss gets the rest through its own top light (MOSS_TOP).
 */
const TOP_DIR = new Vector3(-0.2, -0.75, 0.63).normalize();
const RIM_ROCK = 0.15;
/** Moss strip + top light: dark → full (video s). */
const MOSS_RAMP: [number, number] = [38.5, 40.5];
/** Light left on the moss before the ramp (0–1). */
const MOSS_GAIN_MIN = 0.03;
/**
 * Moss-only key (stone/surfaces.ts addMossLight) relative to the look's key intensity,
 * its wrap and fall-off radius (m) from the pool's centre (frame 14: the moss is
 * brightest on the left edge next to the emblem and dims along the strip to the right).
 */
const MOSS_KEY = 1.25;
const MOSS_KEY_WRAP = 0.0;
const MOSS_KEY_RADIUS = 0.75;
/**
 * Moss-only share of the top light, relative to the look's rim (the rig rim is faint on
 * everything: RIM_ROCK), and the share of it that shows through blades lit from behind
 * (the glow of the strands on the right fringe).
 */
const MOSS_TOP = 1.8;
const MOSS_TOP_TRANSLUCENCY = 1.3;
/**
 * Moss strip base darker where the mesh asks for few plants (COLOR_0.R from → to): the
 * thin right fringe (R ≈ 0.63–0.72) is a dark edge with single strands (frame 14).
 */
const MOSS_DENSITY_SHADE = { from: 0.74, to: 0.86, dark: 0.3 };

/**
 * Rock: triplanar repeats per metre (grain size), tint (linear, > 1: the texture's albedo
 * is darker than the lit slab of frame 14), roughness, baked AO, texture AO (halved: the
 * dim right part of the slab is lit by the fill alone and read blotchy at full AO).
 */
const ROCK = { scale: 2.0, color: new Color(1.15, 1.24, 1.45), roughness: 1, aoStrength: 1, textureAoStrength: 0.5 };
/**
 * Recess floor and the walls (tinted like the floor). stone.glb rev 3: walls 0–3.2 cm
 * above the floor, a 45° chamfer 3.2–3.6 cm, the face ring at 3.6 cm. Walls and the whole
 * chamfer are tinted (`wallDepth`: full / gone, m; the face is kept by the facing test),
 * the bounce falls off up to `wallTop`; `flatNormal` takes the rock's normal map off walls
 * and chamfer (the upper-left chamfer sits at the key's terminator, N·L ≈ 0.12: with the
 * rock relief its pixels flipped between lit and black — the specks along the lip).
 */
const EMBLEM = { color: "#9C9075", roughness: 0.85, aoStrength: 0.55, wall: "#9C8C70", wallDepth: [0.0366, 0.0386] as [number, number], wallTop: 0.036, flatNormal: 1 };
/**
 * Warm bounce light in the recess, × the look's key × the floor albedo² (light that has
 * bounced off the golden floor and walls: the shadowed walls and the floor in the lip's
 * shadow read deep saturated brown-gold in frame 14, not grey): walls (falling to
 * RECESS_BOUNCE_TOP at the lip), floor.
 */
const RECESS_BOUNCE = 1.08;
const RECESS_BOUNCE_TOP = 0.3;
const FLOOR_BOUNCE = 0.288;

const tmpV = new Vector3();

export class StoneScene extends BaseScene {
  readonly id = "stone" as const;

  private focus = FOCUS_DISTANCE;
  private readonly spot = new SpotLight(0xffffff, 0, 0, 0.3, 1, 2);
  private readonly mossLight = createMossLightUniforms();
  private readonly keyToLight = new Vector3(0, 1, 0);
  private readonly topToLight = TOP_DIR.clone().negate();
  private readonly poolCenter = new Vector3();
  private readonly recessBounce = { value: new Color(0, 0, 0) };
  private readonly floorBounce = { value: new Color(0, 0, 0) };
  /** Floor albedo² (linear): the colour of light bounced twice in the golden recess. */
  private readonly floorAlbedo2 = new Color(EMBLEM.color).multiply(new Color(EMBLEM.color));
  private haze: Haze | null = null;
  private readonly vegetation: { label: string; build: VegetationBuild }[] = [];
  private readonly trOverride: Partial<TransitionParams> = {};
  private readonly frustum = new Frustum();
  private readonly projView = new Matrix4();
  private visible: Record<string, number> = {};
  private exitTrack: [number, number][] = EXIT_TRACK;
  private placeholder = false;
  private debug = { moss: 0 };

  protected async build(ctx: SceneContext): Promise<void> {
    const [gltf, kit] = await Promise.all([ctx.assets.tryGltf("models/stone.glb"), loadKit(ctx)]);
    if (!gltf) {
      await this.buildPlaceholder(ctx);
      return;
    }

    // own copy of the node tree (geometry stays registry-owned)
    const world = gltf.scene.clone(true);
    world.name = "stone";
    this.scene.add(world);
    world.updateMatrixWorld(true);

    const meshes = new Map<string, Mesh>();
    world.traverse((o) => {
      const m = o as Mesh;
      if (m.isMesh) meshes.set(m.name, m);
    });
    const slab = meshes.get("rock_stone_slab");
    const floor = meshes.get("emblem_stone");
    const strip = meshes.get("moss_stone_edge");

    // --- materials ----------------------------------------------------------------
    const [rock, moss] = await Promise.all([
      // frame 14: light warm grey-beige under the key, a fine lumpy grain (the mesh relief
      // is milder): the 2048² rock set at 0.5 m per tile, fully rough (options: ROCK)
      createRockMaterial(ctx.assets, ROCK),
      // the strip's base under the fuzz: lit yellow-green, darker folds and undersides
      createMossLayerMaterial(ctx, {
        tint: "#DCEFA8",
        brightness: 3.6,
        saturation: 1.0,
        patchScale: 12,
        patchContrast: 0.35,
        sheen: 0.45,
        sheenColor: "#C6E06A",
        aoContrast: 1.0,
        shade: { top: 1.2, bottom: 0.5, topColor: "#D6EA6E", topAmount: 0.3 },
      }),
    ]);
    const emblem = createEmblemMaterial({ color: EMBLEM.color, roughness: EMBLEM.roughness, aoStrength: EMBLEM.aoStrength });
    addSandGrain(emblem, { frequency: 380, contrast: 0.32, bump: 0.0005, bounce: this.floorBounce });
    dropShadowNormalOffset(emblem);
    addMossLight(moss, this.mossLight);
    addDensityShade(moss, MOSS_DENSITY_SHADE);
    if (floor && slab) addRecessTint(rock, { ...emblemFrame(floor.geometry), depth: EMBLEM.wallDepth, wallTop: EMBLEM.wallTop, flatNormal: EMBLEM.flatNormal, color: EMBLEM.wall, bounce: this.recessBounce, bounceTop: RECESS_BOUNCE_TOP });
    this.own(rock);
    this.own(moss);
    this.own(emblem);

    for (const [name, mesh] of meshes) {
      if (name.startsWith("emblem_")) {
        mesh.material = emblem;
        ctx.layers.assign(mesh, "emblem");
        mesh.castShadow = false;
      } else if (name.startsWith("moss_")) {
        mesh.material = moss;
        ctx.layers.assign(mesh, "moss");
        mesh.castShadow = true;
      } else {
        mesh.material = rock;
        ctx.layers.assign(mesh, "rock");
        mesh.castShadow = true;
      }
      mesh.receiveShadow = true;
    }

    // --- camera: one pose for the whole set ------------------------------------------
    const found = this.rig.addPosesFromObject(world);
    if (!found.includes("stone_main")) throw new Error("stone.glb: missing camera cam_stone_main");
    const extras = (name: string): Record<string, unknown> => (world.getObjectByName(name)?.userData ?? {}) as Record<string, unknown>;
    const eMain = extras("cam_stone_main");
    this.focus = finite(eMain.focus_distance_m, FOCUS_DISTANCE);
    this.exitTrack = parseTrack(extras("cam_stone_out").exit_track) ?? EXIT_TRACK;
    this.rig.setTrack([{ t: 37, pose: "stone_main" }]);
    // narrow screens: keep most of the width, centred towards the slab on the right
    this.rig.fit = { minHorizontalFraction: 0.6, subjectX: 0.45, maxFov: 62 };

    // --- lights ------------------------------------------------------------------------
    const keyEmpty = world.getObjectByName("key_stone");
    const eKey = (keyEmpty?.userData ?? {}) as Record<string, unknown>;
    const keyDir = keyEmpty ? localMinusZ(keyEmpty) : KEY_DIR.clone();
    const target = vec3(eKey.light_target_gltf) ?? KEY_TARGET.clone();
    const radius = finite(eKey.falloff_radius_m, POOL_RADIUS);

    // the pool: a spot along the key with a code-made cookie (shape above), its cone just
    // covers the cookie square (the cookie fades out before the square's edge)
    const spot = this.spot;
    spot.name = "stone_pool";
    spot.angle = Math.atan(radius / POOL_DISTANCE);
    spot.penumbra = 0.2;
    spot.map = this.own(createPoolCookie({ ...POOL_SHAPE, extent: radius }));
    spot.decay = 2;
    spot.distance = 0;
    spot.position.copy(target).addScaledVector(keyDir, -POOL_DISTANCE);
    spot.target.position.copy(target);
    spot.castShadow = true;
    spot.shadow.mapSize.set(ctx.quality.shadowMapSize, ctx.quality.shadowMapSize);
    // back faces cast (three's default for FrontSide), so a lit face never meets its own
    // depth: no depth bias, which keeps the lip's shadow tight at the foot of the walls
    // (-0.0003 left a lit hairline there); normalBias guards the bumps of the rock
    spot.shadow.bias = 0;
    spot.shadow.normalBias = 0.006;
    spot.shadow.camera.near = 1.0;
    spot.shadow.camera.far = POOL_DISTANCE + 3;
    spot.shadow.camera.layers.enableAll();
    this.scene.add(spot, spot.target);

    this.keyToLight.copy(keyDir).negate();
    this.mossLight.uStoneMossWrap.value = MOSS_KEY_WRAP;
    this.mossLight.uStoneMossRadius.value = MOSS_KEY_RADIUS;
    this.poolCenter.copy(target);
    this.lights.setKeyDirection(keyDir, target, 6);
    this.lights.setRimDirection(TOP_DIR, target, 6);
    this.lights.fill.position.set(0, 1, 0);
    this.lights.scale = { key: KEY_BASE, fill: 1, rim: RIM_ROCK };

    // --- vegetation ----------------------------------------------------------------------
    const view = this.scatterView(this.rig.pose("stone_main") as CameraPose);
    if (strip) {
      // time-sliced (≈ 8 ms slices): the same plants, the set prepares while the branch renders
      const build = await buildVegetationAsync(ctx, stoneMoss, {
        meshes: [strip, ...(slab ? [slab] : [])],
        parent: world,
        views: [view],
        kit,
        label: "stone",
      });
      for (const g of build.grass) addMossLight(g.material, this.mossLight, MOSS_TOP_TRANSLUCENCY);
      for (const k of build.kit) addMossLight(k.material, this.mossLight, MOSS_TOP_TRANSLUCENCY);
      this.vegetation.push({ label: "stone", build });
      build.report(ctx);
    }

    // --- haze along the top -----------------------------------------------------------
    this.haze = buildHaze();
    this.scene.add(this.haze.mesh);
    ctx.layers.assign(this.haze.mesh, "fx");
  }

  private scatterView(pose: CameraPose): ScatterView {
    return { position: pose.position.clone(), quaternion: pose.quaternion.clone(), fov: pose.fov, aspect: REFERENCE_ASPECT, weight: 1, margin: 0.15 };
  }

  getLook(_frame: FrameState, local: SceneLocal, base: LookParams): LookParams {
    if (this.placeholder) return base;
    base.dof.focusDistance = this.focus;
    // the top light comes up with the moss (frame 13: right of the fissure still dark)
    base.lights.rimIntensity *= mossRamp(local.t);
    return base;
  }

  protected animate(frame: FrameState, local: SceneLocal, look: LookParams): void {
    if (this.placeholder) return;
    const moss = mossRamp(local.t);
    this.debug.moss = moss;
    const ml = this.mossLight;
    ml.uStoneLightGain.value = MOSS_GAIN_MIN + (1 - MOSS_GAIN_MIN) * moss;
    ml.uStoneMossKey.value.copy(look.lights.keyColor).multiplyScalar(look.lights.keyIntensity * MOSS_KEY);
    this.camera.updateMatrixWorld();
    ml.uStoneMossDir.value.copy(this.keyToLight).transformDirection(this.camera.matrixWorldInverse);
    ml.uStoneMossCenter.value.copy(this.poolCenter).applyMatrix4(this.camera.matrixWorldInverse);
    ml.uStoneMossTop.value.copy(look.lights.rimColor).multiplyScalar(look.lights.rimIntensity * MOSS_TOP);
    ml.uStoneMossTopDir.value.copy(this.topToLight).transformDirection(this.camera.matrixWorldInverse);

    // the pool: the look's key colour / intensity as irradiance at the emblem point
    this.spot.color.copy(look.lights.keyColor);
    this.spot.intensity = look.lights.keyIntensity * POOL_GAIN * POOL_DISTANCE * POOL_DISTANCE;
    this.recessBounce.value.copy(look.lights.keyColor).multiplyScalar(look.lights.keyIntensity * RECESS_BOUNCE).multiply(this.floorAlbedo2);
    this.floorBounce.value.copy(look.lights.keyColor).multiplyScalar(look.lights.keyIntensity * FLOOR_BOUNCE).multiply(this.floorAlbedo2);

    if (this.haze) {
      const u = this.haze.uniforms;
      u.uAspect.value = frame.viewport.aspect;
      u.uTime.value = frame.reducedMotion ? frame.timeSec * 0.25 : frame.timeSec;
    }

    // visible instance statistics (debug panel)
    if (frame.frame % 15 === 0 || frame.capture) {
      const cam = this.camera;
      cam.updateMatrixWorld();
      this.projView.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
      this.frustum.setFromProjectionMatrix(this.projView);
      this.visible = {};
      for (const v of this.vegetation) {
        const n = v.build.visibleCount(this.frustum);
        this.visible[v.label] = n;
        this.ctx.reportInstances(`view.${v.label}`, n);
      }
    }
  }

  /**
   * branch → stone (incoming): the stone picture rides up at ENTRY_FOLLOW of the branch
   * picture's travel (frame 13). The exit (stone → canopy) is configured in SceneConfig.
   */
  getTransitionOverride(tr: ActiveTransition): Partial<TransitionParams> | null {
    if (this.placeholder || tr.reduced || tr.id !== "branch-stone" || tr.mode !== "slide") return null;
    this.trOverride.follow = ENTRY_FOLLOW;
    return this.trOverride;
  }

  debugInfo(): Record<string, string | number | boolean> {
    const out: Record<string, string | number | boolean> = {};
    for (const v of this.vegetation) {
      out[`${v.label} built`] = Object.entries(v.build.counts)
        .map(([k, n]) => `${k} ${n}`)
        .join(", ");
      out[`${v.label} in view`] = this.visible[v.label] ?? 0;
    }
    out["moss light"] = Number(this.debug.moss.toFixed(3));
    // exit slide (SceneConfig stone-canopy) against the measured track
    const cfg = TRANSITIONS.find((c) => c.id === "stone-canopy");
    if (cfg) {
      for (const [t, measured] of this.exitTrack) {
        if (t < 43.15) continue;
        const k = ease(cfg.ease, windowProgress(t, cfg.start, cfg.end));
        out[`exit ${t.toFixed(1)} s`] = `${k.toFixed(3)} (measured ${measured.toFixed(3)})`;
      }
    }
    return out;
  }

  // ---------------------------------------------------------------------------
  // PLACEHOLDER fallback (stone.glb missing)
  // ---------------------------------------------------------------------------

  private async buildPlaceholder(ctx: SceneContext): Promise<void> {
    this.placeholder = true;
    const [rock, moss] = await Promise.all([createRockMaterial(ctx.assets), createMossBaseMaterial(ctx.assets)]);
    const emblem = createEmblemMaterial();
    this.own(rock);
    this.own(moss);
    this.own(emblem);
    const root = new Group();
    root.name = "PLACEHOLDER_stone";
    this.scene.add(root);

    placeholderMesh(ctx, root, rockGeometry(ctx.rng("slab"), [3.2, 1.35, 1.0]), rock, "stone", {
      position: [0.75, -0.2, -0.38],
      rotation: [0.05, -0.12, 0.52],
      name: "stone_slab",
    });
    placeholderMesh(ctx, root, blobGeometry(ctx.rng("strip"), { radius: 0.5, roughness: 0.3, scale: [3.0, 0.22, 0.75] }), moss, "moss", {
      position: [0.58, 0.42, -0.42],
      rotation: [0, -0.12, 0.52],
      name: "moss_strip",
    });
    const emblemGroup = new Group();
    emblemGroup.name = "PLACEHOLDER_emblem";
    emblemGroup.position.set(0.52, -0.08, 0.16);
    root.add(emblemGroup);
    for (const [i, shape] of seedEmblemShapes(0.52).entries()) {
      const geo = new ExtrudeGeometry(shape, { depth: 0.035, bevelEnabled: true, bevelThickness: 0.006, bevelSize: 0.004, bevelSegments: 2, curveSegments: 24 });
      placeholderMesh(ctx, emblemGroup, geo, emblem, "emblem", { name: `emblem_${i}` });
    }

    this.lights.setKeyDirection(new Vector3(-0.55, -0.55, -0.62), new Vector3(0.5, 0, 0), 8);
    this.lights.setRimDirection(new Vector3(0.6, -0.1, 0.8), new Vector3(0.5, 0, 0), 8);
    this.rig.fit.subjectX = 0.3;
    const P = CameraRig.lookAtPose;
    this.rig.setTrack([
      { t: 37, pose: P([0, 0, 3.1], [0, 0, 0]) },
      { t: 43.3, pose: P([0.05, 0.02, 2.8], [0.03, 0, 0]) },
    ]);
  }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

function mossRamp(t: number): number {
  return smoothstep(MOSS_RAMP[0], MOSS_RAMP[1], t);
}

function finite(v: unknown, fallback: number): number {
  const n = Number(v);
  return v !== null && v !== undefined && Number.isFinite(n) ? n : fallback;
}

function vec3(v: unknown): Vector3 | null {
  return Array.isArray(v) && v.length === 3 && v.every((x) => Number.isFinite(Number(x))) ? new Vector3(Number(v[0]), Number(v[1]), Number(v[2])) : null;
}

/** `exit_track` extra: "[[t, fh], …]" (JSON string or array). */
function parseTrack(v: unknown): [number, number][] | null {
  let data: unknown = v;
  if (typeof v === "string") {
    try {
      data = JSON.parse(v);
    } catch {
      return null;
    }
  }
  if (!Array.isArray(data)) return null;
  const out: [number, number][] = [];
  for (const row of data) {
    if (!Array.isArray(row) || row.length < 2) return null;
    const a = Number(row[0]);
    const b = Number(row[1]);
    if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
    out.push([a, b]);
  }
  return out.length ? out : null;
}

/** World direction of an object's local −Z (key light empties). */
function localMinusZ(o: Object3D): Vector3 {
  o.updateWorldMatrix(true, false);
  return new Vector3(0, 0, -1).transformDirection(o.matrixWorld).normalize();
}

/**
 * Plane and outline box of the recess floor (positions are world-space metres: the GLB
 * transforms are applied): area-weighted centre and normal, the long axis of the seed
 * from the in-plane covariance, half sizes from the extents (+ a few mm for the walls).
 */
function emblemFrame(geometry: BufferGeometry): { point: Vector3; normal: Vector3; center: Vector3; axisU: Vector3; axisV: Vector3; halfU: number; halfV: number } {
  const pos = geometry.getAttribute("position");
  const index = geometry.index;
  const triCount = index ? index.count / 3 : pos.count / 3;
  const a = new Vector3();
  const b = new Vector3();
  const c = new Vector3();
  const n = new Vector3();
  const normal = new Vector3();
  const center = new Vector3();
  let area = 0;
  for (let i = 0; i < triCount; i++) {
    const ia = index ? index.getX(i * 3) : i * 3;
    const ib = index ? index.getX(i * 3 + 1) : i * 3 + 1;
    const ic = index ? index.getX(i * 3 + 2) : i * 3 + 2;
    a.fromBufferAttribute(pos, ia);
    b.fromBufferAttribute(pos, ib);
    c.fromBufferAttribute(pos, ic);
    n.subVectors(b, a).cross(tmpV.subVectors(c, a));
    const ar = n.length() * 0.5;
    normal.add(n);
    center.addScaledVector(a.add(b).add(c), ar / 3);
    area += ar;
  }
  center.multiplyScalar(1 / Math.max(area, 1e-9));
  normal.normalize();
  // out of the recess (towards the lens), whatever the winding: follow the vertex normals
  const vn = geometry.getAttribute("normal");
  if (vn) {
    n.set(0, 0, 0);
    for (let i = 0; i < vn.count; i++) n.add(a.fromBufferAttribute(vn, i));
    if (n.dot(normal) < 0) normal.negate();
  }
  // in-plane basis, then the principal axis of the vertices
  const u0 = new Vector3(1, 0, 0).sub(tmpV.copy(normal).multiplyScalar(normal.x)).normalize();
  const v0 = new Vector3().crossVectors(normal, u0);
  let suu = 0;
  let svv = 0;
  let suv = 0;
  for (let i = 0; i < pos.count; i++) {
    a.fromBufferAttribute(pos, i).sub(center);
    const pu = a.dot(u0);
    const pv = a.dot(v0);
    suu += pu * pu;
    svv += pv * pv;
    suv += pu * pv;
  }
  const theta = 0.5 * Math.atan2(2 * suv, suu - svv);
  const major = u0.clone().multiplyScalar(Math.cos(theta)).addScaledVector(v0, Math.sin(theta)).normalize();
  const axisV = major;
  const axisU = new Vector3().crossVectors(axisV, normal).normalize();
  let halfU = 0;
  let halfV = 0;
  for (let i = 0; i < pos.count; i++) {
    a.fromBufferAttribute(pos, i).sub(center);
    halfU = Math.max(halfU, Math.abs(a.dot(axisU)));
    halfV = Math.max(halfV, Math.abs(a.dot(axisV)));
  }
  const margin = 0.008;
  return { point: center.clone(), normal, center, axisU, axisV, halfU: halfU + margin, halfV: halfV + margin };
}
