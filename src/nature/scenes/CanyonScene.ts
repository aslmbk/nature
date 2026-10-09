/**
 * Canyon scene set — episode `canyon` (15.5–21 s): the dark rock cleft between the arch
 * and the oracle. Frame 06 (19.8 s) is the mood guide: two rock walls framing a deep,
 * dark gap, a large rock face at the upper right with small leafy islands on its ledges,
 * a narrow lit edge on the left, a dark near form at the bottom left.
 *
 * `models/canyon.glb` (Blender: assets-src/blender/build_canyon.py):
 *  - `rock_canyon_right` / `rock_canyon_left` / `rock_canyon_low__fg` → triplanar rock;
 *    the near form's baked AO is scaled down in the GLB (it stays dark), it sits at
 *    2–3.5 m and is blurred by the near-field DOF
 *  - `moss_canyon_right` / `moss_canyon_left` → moss pads on the ledges (moss layer)
 *  - COLOR_0.R of all five → where the `canyonIslands` plants grow (canyon/recipes.ts);
 *    the scene adds the overgrowth of frame 06 inside the asset's plant zones and
 *    keeps it to the faces the camera sees (canyon/growth.ts, scatter-only copies)
 *  - `cam_canyon_{in,main,out}` (extras: focus distance, video time) → camera track
 *    (canyon/track.ts); `key_canyon` empty (local −Z) → key light direction
 *
 * Light: no flood. The key is a local spot pool along `key_canyon` (turned towards the
 * front, KEY), shadowed, inverse-square falloff with a distance cut-off, centred on the
 * upper part of the right face; a second narrow spot from the same side grazes the inner
 * edge of the left fragment (EDGE). The look's own (directional) key is off
 * (`lights.scale.key` 0, hidden). The cleft itself is shaped by cool light from its open
 * top, all unshadowed: a wide sky spot from high above the gap (SKY, wall tops lighter
 * than their feet), the hemisphere fill as that sky over a black floor (FILL) and a point
 * light deep in the far end of the gap (CLEFT) that catches the walls' inner edges, so
 * both walls stand out against the haze; the rim from behind shows in the plants'
 * silhouettes.
 *
 * Air (canyon/air.ts, two draw calls): the far end of the cleft — a dim teal glow
 * between ragged far walls, slow mist banks, a low floor mist (one far-plane triangle,
 * drawn only where no geometry is) — warm dusty light shafts entering at the top of the
 * frame and slanting down to the left onto the left wall's lit edge (lens-facing
 * ribbons, additive) and a few hundred dust motes drifting through the main shaft.
 * Placement from `ctx.rng`, motion a pure function of `frame.timeSec`. While the canyon
 * is lit the look's vignette bands are eased (VIGNETTE), so the haze, the shafts and the
 * walls' feet are not swallowed; at light 0 the bands are the look's own, so the mixes
 * with the arch and the oracle (graded with the blended look) are unchanged there.
 *
 * Everything comes up 15.8 → 16.8 s (the canyon opens out of the arch → canyon dip) and
 * goes down 20.6 → 21.2 s (into the canyon → oracle crossfade): canyon/track.ts
 * `lightGain`; in between the far glow brightens (`farGlow`).
 *
 * Without canyon.glb a small PLACEHOLDER set keeps the episode working.
 */
import { Color, Frustum, Group, Matrix4, PointLight, Quaternion, SpotLight, Vector3, type Mesh, type Object3D } from "three";
import { CameraRig, type CameraPose } from "../CameraRig";
import { BaseScene } from "../core/BaseScene";
import { hashString } from "../core/rng";
import { REFERENCE_ASPECT } from "../SceneConfig";
import { createRockMaterial } from "../rendering/Materials";
import type { FrameState, LookParams, SceneContext, SceneLocal } from "../types";
import { createMossLayerMaterial } from "../vegetation/MossLayer";
import { runSliced, yieldToBrowser, type Steps } from "../vegetation/slices";
import type { ScatterView } from "../vegetation/SurfaceScatter";
import { buildVegetationAsync, loadKit, type VegetationBuild } from "../vegetation/Vegetation";
import { buildAir, buildMotes, type Air, type BeamSpec, type HazeSpec, type Motes } from "./canyon/air";
import { growthMeshSteps, type GrowthZone } from "./canyon/growth";
import { canyonIslands } from "./canyon/recipes";
import { TRACK_DEFAULTS, buildCanyonTrack, farGlow, focusAt, lightGain, type CanyonKey } from "./canyon/track";
import { placeholderMesh, pointScale, rockGeometry } from "./placeholders";

/** Fallback for the canyon.glb key empty (`key_canyon`, local −Z). */
const KEY_DIR = new Vector3(-0.419, -0.789, -0.449).normalize();

/**
 * The key spot, tuned against frame 06. Direction: the empty's, turned `front` of the
 * way towards the view direction (−Z) — the empty's own direction lit the whole right
 * face from above, top to bottom; frame 06 lights the faces turned to the lens and the
 * slab flanks. Aimed at the upper part of the right face (frame 06 is lit down to
 * v ≈ 0.42 of the frame, the lower face falls off into the dark), from `distance` m
 * along that direction (the inverse-square falloff across the pool), a narrow soft
 * cone (half-angle `angle`°), cut off at `cutoff` × the distance. `gain`: irradiance
 * at the target relative to the look's key intensity.
 */
const KEY = { front: 0.45, target: new Vector3(1.5, 1.3, -0.2), distance: 3.4, angle: 16, penumbra: 1, gain: 3.5, cutoff: 2.6 };

/**
 * The narrow light grazing the inner edge of the left fragment (frame 06: u ≤ 0.1,
 * v 0.22–0.45 of the frame): from the right, above and a little in front, so the slab
 * tops and their right-facing sides catch it and the undersides stay dark.
 */
const EDGE = { target: new Vector3(-1.45, 0.5, 0.8), dir: new Vector3(-0.85, -0.4, -0.35).normalize(), distance: 3.8, angle: 10, penumbra: 1, gain: 3.0 };

/** Rim, direction of travel: from behind the rocks towards the lens, a little from above. */
const RIM_DIR = new Vector3(0.12, -0.35, 1).normalize();

/** Near-field DOF distances as fractions of the focus distance (the near form at 2–3.5 m blurs). */
const DOF_NEAR = { start: 0.55, end: 0.76 };

/** Rock: triplanar repeats per metre and a neutral grey tint (the warm key makes frame 06's warm grey), matte. */
const ROCK = { scale: 1.1, color: "#C8C6C0", roughness: 1 };

/**
 * Scatter importance of the canyon views: plants go to the faces turned to the lens
 * (dot(normal, to camera) from −0.1, fully from 0.25; the default counts grazing back
 * faces from −0.3). Together with the growth mask's visibility (canyon/growth.ts) the
 * budget lands where frame 06 shows plants instead of behind the slabs.
 */
const IMPORTANCE = { facingMin: -0.1, facingFull: 0.25 };

/** Plant zones of build_canyon.py (ZONES, cam_canyon_main screen: u0, v0, u1, v1, weight[, solid]). */
const GROWTH_RIGHT: GrowthZone[] = [
  [0.68, -0.1, 0.8, 0.27, 1.0],
  [0.77, 0.25, 0.875, 0.5, 0.9],
  [0.875, -0.1, 1.1, 0.34, 1.0],
  [0.89, 0.3, 1.1, 0.52, 0.65],
  // the dense clumps of frame 06: the dark leafy band along the upper-left outline and
  // the lit clump on the lower-left ledge
  [0.69, -0.1, 0.84, 0.15, 1.0, 0.6],
  [0.77, 0.27, 0.85, 0.43, 1.0, 0.85],
];
/** The grass clump at the right border and the darker grass below it (`clump` layer). */
const GRASS_RIGHT: GrowthZone[] = [
  [0.9, -0.1, 1.1, 0.28, 1.0, 0.8],
  [0.9, 0.28, 1.1, 0.55, 0.7, 0.5],
  [0.68, -0.1, 0.77, 0.22, 0.5, 0.3],
];
const GROWTH_LEFT: GrowthZone[] = [
  [-0.1, 0.27, 0.105, 0.48, 0.55],
  [-0.1, 0.62, 0.095, 0.9, 0.45],
  // frame 06: the green along the lit edge above (v 0.08–0.3)
  [-0.1, 0.08, 0.1, 0.32, 0.8, 0.5],
];
/**
 * Shape of the added growth (canyon/growth.ts): soft zone edges, half of it on the open
 * flanks next to the cracks, patches about 45 cm across covering roughly a third of
 * each zone (the clumps of frame 06, bare rock between), a quarter of the density kept
 * where the surface is hidden from the main camera.
 */
const GROWTH = {
  edge: 0.03,
  flank: 0.5,
  cavity: [0.75, 0.35] as [number, number],
  patch: { scale: 2.2, threshold: 0.54, softness: 0.08 },
  hiddenFloor: 0.25,
};
/** Growth strength (R units) per rock; the grass clump grows on the flanks as much as in the cracks. */
const GROW = { right: 0.85, left: 0.75, grass: 1, grassFlank: 0.8 };

const tmpV = new Vector3();

/**
 * Cool sky light falling into the cleft from its open top (no shadows): from high above
 * the gap, a little behind the walls, a wide soft cone over both walls. Its inverse-square
 * falloff down the walls keeps their tops lighter than their feet. `irradiance` at the
 * target (× the light gain; independent of the look's key).
 */
const SKY = { position: new Vector3(0.1, 6.0, -1.6), target: new Vector3(0.3, 0.2, -0.1), angle: 40, penumbra: 1, color: "#A3BACB", irradiance: 2.5 };

/**
 * The far end of the cleft: a cool point light deep in the gap behind the walls (no
 * shadows). It catches the inner edges of both walls (faces turned to the gap and away
 * from the lens) and the plants on them, so the two walls stand out against the haze.
 * `irradiance` at `reference` m, cut off at `cutoff` m.
 */
const CLEFT = { position: new Vector3(-0.1, 0.2, -3.2), color: "#A8BCBC", irradiance: 1.5, reference: 3.2, cutoff: 9 };

/** Hemisphere fill: the cool sky over the cleft, a black floor (× the light gain). */
const FILL = { sky: "#8FA3B3", ground: "#040504", intensity: 1.1 };

/** Vignette bands while lit (the look's 0.55 / 0.95 hid the haze, the beams and the walls' feet). */
const VIGNETTE = { top: 0.3, bottom: 0.6 };

/**
 * Light shafts falling through the dusty air of the cleft from high up on the far side,
 * slanting down to the left towards the lens (canyon/air.ts), all parallel: the main one
 * lands on the lit inner edge of the left wall (where EDGE lights it), a wide faint glow
 * around it, a thinner one through the middle and a faint one past the right wall's edge
 * (its top hidden behind the wall).
 */
const BEAM_DIR = new Vector3(-0.36, -0.88, 0.3).normalize();
const BEAMS: BeamSpec[] = [
  { top: new Vector3(-0.85, 1.85, 0.27), dir: BEAM_DIR, length: 1.9, width: [0.2, 0.34], strength: 1.3, phase: 0.3, streaks: 4, rise: 0.08, fall: 0.6 },
  { top: new Vector3(-0.8, 1.9, 0.22), dir: BEAM_DIR, length: 2.0, width: [0.45, 0.8], strength: 0.25, phase: 5.1, streaks: 1.5, rise: 0.08, fall: 0.8 },
  { top: new Vector3(0.1, 2.3, -0.4), dir: BEAM_DIR, length: 3.2, width: [0.1, 0.24], strength: 0.55, phase: 2.7, streaks: 3, rise: 0.12, fall: 1.4 },
  { top: new Vector3(0.95, 2.7, -0.7), dir: BEAM_DIR, length: 4.0, width: [0.14, 0.36], strength: 0.4, phase: 4.2, streaks: 3, rise: 0.25, fall: 1.6 },
];
/** Linear colour of the beams (warm-neutral, the key's) and of the far haze (cool). */
const BEAM_COLOR = new Color(0.045, 0.04, 0.03);
const HAZE: HazeSpec = {
  sky: { color: new Color(0.016, 0.025, 0.029), x: 0.44, halfWidth: 0.62, floor: -0.35, top: 1.15 },
  end: { color: new Color(0.018, 0.031, 0.031), x: 0.45, y: 0.45, rx: 0.15, ry: 0.62 },
  floor: { color: new Color(0.03, 0.044, 0.048), height: 0.4 },
  far: { left: 0.33, right: 0.59, ragged: 0.1, shade: 0.7, rim: 0.4 },
};
/** Dust motes in the main beam: share of `quality.particles`, volume radius (m), colour, drift (m/s), brightness outside the beam. */
const MOTES = { share: 0.02, radius: 0.45, color: new Color(1.1, 1.0, 0.85), fall: 0.015, dim: 0.06 };

export class CanyonScene extends BaseScene {
  readonly id = "canyon" as const;

  private focusSamples: [number, number][] = [];
  private readonly keySpot = new SpotLight(0xffffff, 0, 0, 0.4, 1, 2);
  private readonly edgeSpot = new SpotLight(0xffffff, 0, 0, 0.2, 1, 2);
  private readonly skySpot = new SpotLight(0xffffff, 0, 0, 0.6, 1, 2);
  private readonly cleftLight = new PointLight(0xffffff, 0, 0, 2);
  private air: Air | null = null;
  private motes: Motes | null = null;
  private readonly vegetation: { label: string; build: VegetationBuild }[] = [];
  private readonly frustum = new Frustum();
  private readonly projView = new Matrix4();
  private visible: Record<string, number> = {};
  private gain = 0;
  private readonly growthAdded: Record<string, string> = {};
  private placeholder = false;

  protected async build(ctx: SceneContext): Promise<void> {
    const [gltf, kit] = await Promise.all([ctx.assets.tryGltf("models/canyon.glb"), loadKit(ctx)]);
    if (!gltf) {
      await this.buildPlaceholder(ctx);
      return;
    }
    // the GLB parse (or the last texture upload) ran in this task: go on in a fresh one, so
    // the build never extends a long engine task (the same after the materials below)
    await yieldToBrowser();

    // own copy of the node tree (geometry stays registry-owned)
    const world = gltf.scene.clone(true);
    world.name = "canyon";
    this.scene.add(world);
    world.updateMatrixWorld(true);

    const meshes = new Map<string, Mesh>();
    world.traverse((o) => {
      const m = o as Mesh;
      if (m.isMesh) meshes.set(m.name, m);
    });

    // --- materials ----------------------------------------------------------------
    const [rock, moss] = await Promise.all([
      createRockMaterial(ctx.assets, ROCK),
      // the pads under the leaves: dark olive, matte, darker folds
      createMossLayerMaterial(ctx, {
        tint: "#D8DCA0",
        brightness: 1.5,
        saturation: 0.8,
        patchScale: 9,
        patchContrast: 0.3,
        sheen: 0.2,
        shade: { top: 1.15, bottom: 0.45 },
        aoContrast: 0.8,
      }),
    ]);
    this.own(rock);
    this.own(moss);
    await yieldToBrowser();

    for (const [name, mesh] of meshes) {
      if (name.startsWith("moss_")) {
        mesh.material = moss;
        ctx.layers.assign(mesh, "moss");
      } else {
        mesh.material = rock;
        ctx.layers.assign(mesh, "rock");
      }
      mesh.castShadow = true;
      mesh.receiveShadow = true;
    }

    // --- camera track ----------------------------------------------------------------
    const found = this.rig.addPosesFromObject(world);
    for (const n of ["canyon_in", "canyon_main", "canyon_out"]) if (!found.includes(n)) throw new Error(`canyon.glb: missing camera cam_${n}`);
    const extras = (name: string) => (world.getObjectByName(name)?.userData ?? {}) as Record<string, unknown>;
    const key = (pose: "in" | "main" | "out", tDef: number, fDef: number): CanyonKey => {
      const e = extras(`cam_canyon_${pose}`);
      return { t: finite(e.video_time_s, tDef), pose: this.rig.pose(`canyon_${pose}`) as CameraPose, focus: finite(e.focus_distance_m, fDef) };
    };
    const track = buildCanyonTrack([
      key("in", TRACK_DEFAULTS.tIn, TRACK_DEFAULTS.focusIn),
      key("main", TRACK_DEFAULTS.tMain, TRACK_DEFAULTS.focusMain),
      key("out", TRACK_DEFAULTS.tOut, TRACK_DEFAULTS.focusOut),
    ]);
    this.rig.setTrack(track.keys);
    this.focusSamples = track.focus;
    // narrow screens: keep most of the width, re-centred towards the lit right face
    this.rig.fit = { minHorizontalFraction: 0.62, subjectX: 0.45, maxFov: 62 };

    // --- lights ------------------------------------------------------------------------
    const keyEmpty = world.getObjectByName("key_canyon");
    const keyDir = keyEmpty ? localMinusZ(keyEmpty) : KEY_DIR.clone();
    keyDir.lerp(tmpV.set(0, 0, -1), KEY.front).normalize();
    const target = KEY.target;
    this.keySpot.angle = (KEY.angle * Math.PI) / 180;
    this.keySpot.penumbra = KEY.penumbra;
    this.edgeSpot.angle = (EDGE.angle * Math.PI) / 180;
    this.edgeSpot.penumbra = EDGE.penumbra;
    this.setupSpot(this.keySpot, "canyon_key", target, keyDir, KEY.distance, KEY.distance * KEY.cutoff, ctx);
    this.setupSpot(this.edgeSpot, "canyon_edge", EDGE.target, EDGE.dir, EDGE.distance, 0, null);
    // the cool light of the open sky over the cleft and of its far end (no shadows)
    const skyDir = tmpV.subVectors(SKY.target, SKY.position);
    const skyDistance = skyDir.length();
    this.skySpot.angle = (SKY.angle * Math.PI) / 180;
    this.skySpot.penumbra = SKY.penumbra;
    this.skySpot.color.set(SKY.color);
    this.setupSpot(this.skySpot, "canyon_sky", SKY.target, skyDir.normalize(), skyDistance, 0, null);
    this.cleftLight.name = "canyon_cleft";
    this.cleftLight.color.set(CLEFT.color);
    this.cleftLight.distance = CLEFT.cutoff;
    this.cleftLight.position.copy(CLEFT.position);
    this.scene.add(this.cleftLight);

    // the look's directional key would flood every face: the spots are the key (hidden,
    // so the programs do not shade a light of intensity 0)
    this.lights.scale = { key: 0, fill: 1, rim: 1 };
    this.lights.key.visible = false;
    this.lights.setKeyDirection(keyDir, target, 6);
    this.lights.setRimDirection(RIM_DIR, target, 6);
    this.lights.fill.position.set(0, 1, 0);

    // --- vegetation ------------------------------------------------------------------
    const views: ScatterView[] = [
      ...(["canyon_in", "canyon_main", "canyon_out"] as const).map((n) => scatterView(this.rig.pose(n) as CameraPose, 1)),
      ...[17.6, 20.5].map((t) => scatterView(this.rig.evaluate(t, { position: new Vector3(), quaternion: new Quaternion(), fov: 35 }), 0.8)),
    ];
    // the rock faces: scatter-only copies with the growth of frame 06 added inside the
    // asset's plant zones (canyon/growth.ts); the pads and the near form as exported.
    // Time-sliced like the vegetation below (≈ 8 ms slices, same results as in one go).
    const main = this.rig.pose("canyon_main") as CameraPose;
    const growthAdded = this.growthAdded;
    const grown = function* (name: string, zones: GrowthZone[], strength: number, label: string, rename?: string, flank = GROWTH.flank): Steps<Mesh | undefined> {
      const src = meshes.get(name);
      if (!src) return undefined;
      const g = yield* growthMeshSteps(src, main, { ...GROWTH, flank, zones, strength, seed: hashString(`${ctx.seed}|canyon|${label}`) });
      if (rename) g.mesh.name = rename;
      growthAdded[label] = `${g.added} grown, ${g.hidden} hidden`;
      return g.mesh;
    };
    await yieldToBrowser();
    const [right, left, grass] = await runSliced(
      (function* (): Steps<(Mesh | undefined)[]> {
        const r = yield* grown("rock_canyon_right", GROWTH_RIGHT, GROW.right, "right");
        yield;
        const l = yield* grown("rock_canyon_left", GROWTH_LEFT, GROW.left, "left");
        yield;
        const g = yield* grown("rock_canyon_right", GRASS_RIGHT, GROW.grass, "grass", "grass_canyon_right", GROW.grassFlank);
        return [r, l, g];
      })(),
      8,
      ctx,
    );
    const targets = [meshes.get("moss_canyon_right"), meshes.get("moss_canyon_left"), right, left, grass, meshes.get("rock_canyon_low__fg")].filter((m): m is Mesh => !!m);
    // time-sliced (same draws, same instances): the current episode keeps rendering
    // meanwhile; its first slice starts in a fresh task, not after the last growth slice
    await yieldToBrowser();
    const build = await buildVegetationAsync(ctx, canyonIslands, {
      meshes: targets,
      parent: world,
      views,
      kit,
      label: "canyon",
      importance: IMPORTANCE,
    });
    this.vegetation.push({ label: "canyon", build });
    build.report(ctx, "canyon.");

    // --- air: far haze, light shafts, dust motes (canyon/air.ts) ------------------------
    const facing = tmpV.set(0, 0, 1).applyQuaternion(main.quaternion).normalize().clone();
    const air = buildAir({ beams: BEAMS, beamColor: BEAM_COLOR, haze: HAZE, facing });
    this.air = air;
    world.add(air.mesh);
    ctx.layers.assign(air.mesh, "fx");
    const motes = buildMotes(ctx.rng("canyon-motes"), {
      count: Math.round(ctx.quality.particles * MOTES.share),
      beam: BEAMS[0],
      radius: MOTES.radius,
      facing,
      color: MOTES.color,
      fall: MOTES.fall,
      dim: MOTES.dim,
    });
    this.motes = motes;
    world.add(motes.points);
    ctx.layers.assign(motes.points, "particles");
    ctx.reportInstances("particles", motes.count);

    // the last vegetation slice ran in this task: the engine's compile / warm-up that
    // follows the build starts in a fresh one
    await yieldToBrowser();
  }

  private setupSpot(spot: SpotLight, name: string, target: Vector3, dir: Vector3, distance: number, cutoff: number, ctx: SceneContext | null): void {
    spot.name = name;
    spot.decay = 2;
    spot.distance = cutoff;
    spot.position.copy(target).addScaledVector(dir, -distance);
    spot.target.position.copy(target);
    spot.userData.distanceToTarget = distance;
    if (ctx) {
      spot.castShadow = true;
      spot.shadow.mapSize.set(ctx.quality.shadowMapSize, ctx.quality.shadowMapSize);
      spot.shadow.bias = -0.0003;
      spot.shadow.normalBias = 0.012;
      spot.shadow.camera.near = 0.5;
      spot.shadow.camera.far = distance + 4;
      spot.shadow.camera.layers.enableAll();
    }
    this.scene.add(spot, spot.target);
  }

  getLook(_frame: FrameState, local: SceneLocal, base: LookParams): LookParams {
    if (this.placeholder) {
      const g = lightGain(local.t);
      base.lights.keyIntensity *= g;
      base.lights.fillIntensity *= g;
      base.lights.rimIntensity *= g;
      return base;
    }
    const t = local.t;
    const f = focusAt(this.focusSamples, t, TRACK_DEFAULTS.focusMain);
    base.dof.focusDistance = f;
    base.dof.nearStart = f * DOF_NEAR.start;
    base.dof.nearEnd = f * DOF_NEAR.end;
    const g = lightGain(t);
    base.lights.keyIntensity *= g;
    base.lights.rimIntensity *= g;
    // the cool sky over the cleft as the fill (the look's is almost nothing)
    base.lights.fillSky.set(FILL.sky);
    base.lights.fillGround.set(FILL.ground);
    base.lights.fillIntensity = FILL.intensity * g;
    // eased bands only as far as the canyon is lit: the mixes with the arch and the oracle
    // grade with the blended look, so outside the light the neighbours keep their bands
    base.vignette.top += (VIGNETTE.top - base.vignette.top) * g;
    base.vignette.bottom += (VIGNETTE.bottom - base.vignette.bottom) * g;
    return base;
  }

  protected animate(frame: FrameState, local: SceneLocal, look: LookParams): void {
    if (this.placeholder) return;
    this.gain = lightGain(local.t);
    // spots: the look's key colour / intensity as irradiance at their targets
    const l = look.lights;
    for (const [spot, gain] of [
      [this.keySpot, KEY.gain],
      [this.edgeSpot, EDGE.gain],
    ] as const) {
      const d = spot.userData.distanceToTarget as number;
      spot.color.copy(l.keyColor);
      spot.intensity = l.keyIntensity * gain * d * d;
    }
    const g = this.gain;
    const skyD = this.skySpot.userData.distanceToTarget as number;
    this.skySpot.intensity = SKY.irradiance * g * skyD * skyD;
    this.cleftLight.intensity = CLEFT.irradiance * g * CLEFT.reference * CLEFT.reference;

    // the air: ambient clock (slowed under reduced motion), opacity with the light
    const time = frame.reducedMotion ? frame.timeSec * 0.25 : frame.timeSec;
    if (this.air) {
      const u = this.air.uniforms;
      u.uTime.value = time;
      u.uAspect.value = frame.viewport.aspect;
      u.uBeamOpacity.value = g;
      u.uHazeOpacity.value = g;
      u.uEndColor.value.copy(HAZE.end.color).multiplyScalar(farGlow(local.t));
    }
    if (this.motes) {
      const u = this.motes.uniforms;
      const vp = frame.viewport;
      // reference px (frame height 1020) → drawing-buffer px
      const ref = (vp.height * vp.dpr) / 1020;
      u.uTime.value = time;
      u.uScale.value = pointScale(vp.height, vp.dpr, this.camera.fov);
      u.uMinPx.value = 2.4 * ref;
      u.uMaxPx.value = 40 * ref;
      u.uCocNear.value = 12 * ref;
      u.uCocFar.value = 1.2 * ref;
      u.uFocus.value = look.dof.focusDistance;
      u.uOpacity.value = g;
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

  debugInfo(): Record<string, string | number | boolean> {
    const out: Record<string, string | number | boolean> = {};
    for (const v of this.vegetation) {
      out[`${v.label} built`] = Object.entries(v.build.counts)
        .map(([k, n]) => `${k} ${n}`)
        .join(", ");
      out[`${v.label} in view`] = this.visible[v.label] ?? 0;
    }
    out["light gain"] = Number(this.gain.toFixed(3));
    for (const [k, n] of Object.entries(this.growthAdded)) out[`growth ${k} (vertices)`] = n;
    out["key spot cd"] = Number(this.keySpot.intensity.toFixed(1));
    out["sky spot cd"] = Number(this.skySpot.intensity.toFixed(1));
    out["cleft light cd"] = Number(this.cleftLight.intensity.toFixed(1));
    out["air (haze + beams)"] = this.air ? `${BEAMS.length} beams` : "off";
    out["motes"] = this.motes?.count ?? 0;
    return out;
  }

  // ---------------------------------------------------------------------------
  // PLACEHOLDER fallback (canyon.glb missing)
  // ---------------------------------------------------------------------------

  private async buildPlaceholder(ctx: SceneContext): Promise<void> {
    this.placeholder = true;
    const rock = await createRockMaterial(ctx.assets);
    this.own(rock);
    const root = new Group();
    root.name = "PLACEHOLDER_canyon";
    this.scene.add(root);

    placeholderMesh(ctx, root, rockGeometry(ctx.rng("rock-left"), [0.85, 3.4, 0.95]), rock, "rock", {
      position: [-1.3, -0.15, -0.25],
      rotation: [0.05, 0.3, 0.07],
      name: "rock_left",
    });
    placeholderMesh(ctx, root, rockGeometry(ctx.rng("rock-right"), [1.9, 1.15, 1.2]), rock, "rock", {
      position: [1.1, 0.78, -0.55],
      rotation: [0.25, 0.4, -0.22],
      name: "rock_right",
    });
    placeholderMesh(ctx, root, rockGeometry(ctx.rng("rock-low"), [2.7, 0.9, 1.6]), rock, "rock", {
      position: [0.25, -1.42, 0.05],
      rotation: [0.1, -0.2, 0.04],
      name: "rock_low",
    });

    this.lights.setKeyDirection(new Vector3(-0.62, -0.5, -0.5), new Vector3(0, 0, 0), 8);
    this.lights.setRimDirection(new Vector3(0.5, -0.2, 0.8));
    const P = CameraRig.lookAtPose;
    this.rig.setTrack([
      { t: 15.5, pose: P([0, 0.05, 4.2], [0, 0, 0]) },
      { t: 21, pose: P([0.05, 0.08, 3.7], [0.02, 0.02, 0]) },
    ]);
  }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

function scatterView(pose: CameraPose, weight: number): ScatterView {
  return { position: pose.position.clone(), quaternion: pose.quaternion.clone(), fov: pose.fov, aspect: REFERENCE_ASPECT, weight, margin: 0.15 };
}

function finite(v: unknown, fallback: number): number {
  const n = Number(v);
  return v !== null && v !== undefined && Number.isFinite(n) ? n : fallback;
}

/** World direction of an object's local −Z (key light empties). */
function localMinusZ(o: Object3D): Vector3 {
  o.updateWorldMatrix(true, false);
  return new Vector3(0, 0, -1).transformDirection(o.matrixWorld).normalize();
}
