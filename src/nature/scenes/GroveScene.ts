/**
 * Grove scene set — episodes `hero` (0–8 s) and `arch` (10–15.5 s), the T01 camera
 * descent between them (8–10 s) and the dark exit into the canyon (14.6–16.4 s).
 *
 * One continuous world from `models/grove.glb` (Blender: assets-src/blender/build_grove.py):
 *  - `moss_*`   overgrown masses / cushions → dark moss base (vegetation layer A)
 *  - `wood_*`   arch trunks and low branch → bark texture set (grain along V)
 *  - `*__fg`    near-camera foreground, blurred by the near-field DOF;
 *               `moss_hero_near__fg` rides down with the camera and leaves the frame
 *               during the descent (it would sweep through the picture otherwise)
 *  - cameras `cam_hero_{main,out,p1,p2}`, `cam_arch_{in,main,out}` with
 *    `focus_distance_m` extras → camera track + DOF focus
 *  - empties `key_hero` / `key_arch` (local −Z = light direction): the key light follows
 *    directions set in the hero / arch camera spaces instead (HERO_KEY_CAM / ARCH_KEY_CAM);
 *    during the descent it swings to a strong light grazing the wall from the left
 *    (T01_KEY) and on to the arch
 *  - narrow (portrait) screens get a framing per beat (NARROW_FIT → CameraRig `fit`)
 * Vegetation (layers B / C) comes from the `heroMeadow`, `foregroundMoss` and
 * `archMoss` recipes, scattered once at build time where the story's cameras look.
 * Without grove.glb a small PLACEHOLDER set keeps the episode working.
 */
import { Frustum, Group, Matrix4, Quaternion, Vector3, type Mesh, type MeshStandardMaterial, type PerspectiveCamera, type WebGLProgramParametersWithUniforms, type WebGLRenderer } from "three";
import { CameraRig, type CameraPose, type PoseKey } from "../CameraRig";
import { BaseScene } from "../core/BaseScene";
import { smoothstep } from "../core/math";
import { REFERENCE_ASPECT } from "../SceneConfig";
import { createBarkMaterial, createFarMaterial, createMossBaseMaterial } from "../rendering/Materials";
import type { ActiveTransition, FrameState, LookParams, SceneContext, SceneLocal, TransitionParams } from "../types";
import { createMossLayerMaterial } from "../vegetation/MossLayer";
import { archMoss, foregroundMoss, heroMeadow } from "../vegetation/recipes";
import type { ScatterView } from "../vegetation/SurfaceScatter";
import { buildVegetationAsync, loadKit, type VegetationBuild } from "../vegetation/Vegetation";
import { blobGeometry, placeholderMesh, tubeGeometry } from "./placeholders";

/**
 * Pose keys of the camera track (video seconds). The descent past the wall follows the
 * reference clip (motion/07, 8–11 s): its camera height, fitted from the keyhole edge per
 * frame, is ≈ 3.78 / 3.62 / 3.56 / 3.53 / 3.48 / 3.45 / 3.39 m at 8.8 / 9.0 / 9.1 / 9.2 /
 * 9.4 / 9.6 / 9.7 s — fast past the crest, slow along the keyhole, then the plunge to the
 * arch. `cam_hero_p1` (3.66 m) is passed at 8.95 s, `cam_hero_p2` (3.38 m) at 9.72 s, with
 * a key at 9.15 s 42 % of the way between them (the track then stays within 0.02 m of
 * the fitted heights).
 */
const TRACK = {
  heroMainHold: 0.7,
  heroOut: 2.2,
  descentStart: 8.0,
  p1: 8.95,
  p12: 9.15,
  p2: 9.72,
  archIn: 10.0,
  archMain: 12.5,
  archOut: 15.2,
  exit: 16.4,
} as const;
/** Position of the 9.15 s key between p1 and p2. */
const P12_FRACTION = 0.42;

/**
 * Hero key, direction of travel in the space of `cam_hero_main` (x right, y up, −z
 * into the frame): from the upper right and a little behind the masses. Frame 01: the
 * top and right of the mound and the hanging lower fringe of the overhang catch it
 * (the fringe shows it through the blades), the underside of the overhang and the
 * insides of the masses stay in shade and live on the sky fill from above.
 */
const HERO_KEY_CAM = new Vector3(-0.42, -0.72, 0.55).normalize();

/**
 * Where the hero's sky fill comes from, same camera space: mostly the camera side, a
 * little above — the sage studio around the masses lights the face of the overhang that
 * hangs towards the lens (frame 01: mid olive) more than the mound's top, which the key
 * already lights. From the descent on the fill comes from straight above.
 */
const HERO_FILL_CAM = new Vector3(0.15, 0.12, 0.98).normalize();

/**
 * The hero foreground mass sits this far lower than modelled (m): its plants then end
 * where the modelled surface ended, at the band edge of frame 01 (y ≈ 860 px).
 */
const HERO_FG_DROP = 0.08;

/**
 * T01 (8–10 s): a strong key grazes the wall from the left through the keyhole, a little
 * from behind: the chin bulge on the keyhole's edge catches it and shades the lobe under
 * it, the face turned to the camera stays in shade (fill only, dimmed), and the rim light
 * (raised) makes the keyhole's edges glow through their blades (frame 03, motion/01).
 * Direction of travel of the light; with z ≤ 0.13 the lobe under the chin lights up too,
 * with z ≥ 0.16 the chin falls dark.
 */
const T01_KEY = new Vector3(0.85, -0.5, 0.15).normalize();
/** T01 light multipliers at the height of the descent. */
const T01_LIGHT = { key: 4.2, fill: 0.25, rim: 2.5 } as const;
/**
 * T01 near-field DOF (m from the camera): over the crest the mound's near part (1.1–1.25 m)
 * stays sharp; along the wall its face (1.77–1.94 m) is only just touched — frame 03 reads
 * slightly soft there, and any stronger near blur turns the wall's leaves to mush.
 */
const T01_DOF = { crestStart: 0.6, crestEnd: 1.0, wallEnd: 1.8 } as const;

/**
 * Arch key, direction of travel in the space of `cam_arch_main`: from high up on the
 * left, a little behind the trunks. Frame 04 / detail 03: the tops of the cushions and
 * the moss on the right trunk's inner (left) face catch it, the bark of the right trunk
 * (turned to the lens) stays in shade; the bark in the left hollow faces up into it.
 * Replaces the `key_arch` empty's suggestion, which lit the right trunk's bark.
 */
const ARCH_KEY_CAM = new Vector3(0.3, -0.93, -0.05).normalize();

/**
 * The arch vegetation is drawn from here on (video s): before the descent it lies behind
 * and below the hero masses (checked: hero frames 0–8 s identical with and without it).
 */
const ARCH_VEG_FROM = 8.0;

/**
 * Exit into the canyon (14.6–16.4 s, frame 05 at 15.2 s): the light dims a little, the
 * frame darkens, the lower forms sink into the bottom vignette (`vignetteBottom` over the
 * lower `vignetteSize` of the frame) while the upper left stays lit, then the grove fades
 * to black by `blackAt` (the canyon is unlit until 15.8 s).
 */
const EXIT = { dim: 0.25, exposure: 0.3, vignetteBottom: 0.8, vignetteSize: 0.35, blackFrom: 15.25, blackAt: 15.85 } as const;

/**
 * Exit: the near mass (`moss_arch_near__fg`) is the dark band of frame 04, but frame 05
 * shows it as a dim saddle all the way down once the camera tips onto it: its faces turned
 * from the sky (the hummocks' fronts, in the shade of their tops) get `front` × more
 * albedo (MossLayer shade bottom) over `from`–`to` (video s).
 */
const EXIT_FG = { front: 1.2, from: 14.4, to: 15.1 } as const;

/**
 * The wall leaves are scattered for the 9.2 / 9.6 s views moved down and to the right
 * (m) and turned to the right (yaw, rad): onto the face of the wall, off the keyhole's
 * edge and its lit chin (grass there) and mostly off the front of the mound the hero
 * frames see (a few still reach it; with a 0.3 m drop they covered it).
 */
const WALL_VIEW_SHIFT = { down: 0.45, right: 0.35, yaw: 0.29 } as const;

/**
 * Hero near field (m): frame 01 keeps the left column's foot (≈ 1.1–1.3 m) sharp, the
 * foreground band (0.84–0.92 m) stays well blurred.
 */
const HERO_DOF = { start: 0.7, end: 1.08 } as const;

/** DOF near-field distances as fractions of the focus distance. */
const DOF_NEAR_START = 0.43;
const DOF_NEAR_END = 0.74;

/**
 * Narrow screens (portrait phones / tablets; CameraRig `fit`, applied only below the
 * reference aspect): per beat, the share of the reference width that stays visible and
 * where the taller window sits (reference-frame NDC). Eased between the keys (video s);
 * a key's `tablet` values take over from its phone values between the aspects in
 * FIT_ASPECT (a tablet's window is only a little taller than the reference frame).
 */
interface NarrowFit {
  minHorizontalFraction: number;
  subjectX: number;
  subjectY: number;
  maxFov: number;
}
const NARROW_FIT: { t: number; phone: NarrowFit; tablet?: NarrowFit }[] = [
  // hero: the meadow is the picture. Phones: about half the reference width, shifted left
  // and down: the clover mound right of the middle, the arm's elbow in the upper left behind
  // the head, the blurred foreground band low in the frame behind the mock and the blurb.
  // The window ends at the band's underside (≈ 2 reference half-heights below the axis;
  // further down the masses are bare), inside the hero views' scatter. A tablet has the room
  // for the arm's body too.
  {
    t: 8.0,
    phone: { minHorizontalFraction: 0.52, subjectX: -0.2, subjectY: -0.41, maxFov: 72 },
    tablet: { minHorizontalFraction: 0.9, subjectX: 0, subjectY: 0.4, maxFov: 72 },
  },
  // T01: a crop around the keyhole's edge and its chin (a taller window would show the
  // wall's top and bottom ends)
  { t: 8.6, phone: { minHorizontalFraction: 0.33, subjectX: -0.25, subjectY: 0, maxFov: 72 } },
  { t: 9.75, phone: { minHorizontalFraction: 0.33, subjectX: -0.25, subjectY: 0, maxFov: 72 } },
  // arch: the opening between both trunks under the beam
  { t: 10.4, phone: { minHorizontalFraction: 0.6, subjectX: 0.05, subjectY: 0.1, maxFov: 72 } },
  { t: 14.6, phone: { minHorizontalFraction: 0.6, subjectX: 0.05, subjectY: 0.1, maxFov: 72 } },
  // exit: the near mass ends just below frame 05, so the window grows upwards only
  { t: 15.2, phone: { minHorizontalFraction: 0.5, subjectX: -0.2, subjectY: 1, maxFov: 72 } },
];
const FIT_KEYS = ["minHorizontalFraction", "subjectX", "subjectY", "maxFov"] as const;
/** Viewport aspects (width / height) of the phone and the tablet values of NARROW_FIT. */
const FIT_ASPECT = { phone: 0.5, tablet: 0.75 } as const;

/**
 * Narrow screens see the hero's arm above the reference frame (NARROW_FIT): one more hero
 * view, `cam_hero_main` pitched up by this much (rad) with this vertical fov (deg), keeps
 * its plants going up there.
 */
const HERO_TALL_VIEW = { pitch: 0.52, fov: 26, aspect: 2 } as const;

interface SplineKey {
  t: number;
  pose: CameraPose;
  focus: number;
}

const tmpV = new Vector3();
const tmpTarget = new Vector3();
const tmpFill = new Vector3();
const WORLD_UP = new Vector3(0, 1, 0);
/** Rim light: from behind the masses towards the lens, a little from above. */
const RIM_DIR = new Vector3(0.05, -0.15, 1).normalize();

export class GroveScene extends BaseScene {
  readonly id = "grove" as const;

  private focusKeys: { t: number; focus: number }[] = [];
  private keyHero = new Vector3(0.25, -0.86, -0.45).normalize();
  private keyArch = new Vector3(-0.55, -0.78, -0.3).normalize();
  private fillHero = new Vector3(0, 1, 0);
  private heroFg: Group | null = null;
  private fgStart = new Vector3();
  private readonly vegetation: { label: string; build: VegetationBuild }[] = [];
  /** Arch plants: hidden behind the hero masses until the descent (no draw calls in the hero). */
  private archVeg: VegetationBuild | null = null;
  private readonly tmpPose: CameraPose = { position: new Vector3(), quaternion: new Quaternion(), fov: 35 };
  /** Reused transition override (no per-frame allocation). */
  private readonly trOverride: Partial<TransitionParams> = {};
  private readonly frustum = new Frustum();
  private readonly projView = new Matrix4();
  private visible: Record<string, number> = {};
  private placeholder = false;
  /** Hero masses: above the arch, they stop casting once the camera is below them. */
  private heroCasters: Mesh[] = [];
  /** Shade uniform (x: albedo of the faces turned from the sky) of the arch's near mass (EXIT_FG). */
  private mossArchFgShade: Vector3 | null = null;

  protected async build(ctx: SceneContext): Promise<void> {
    const [gltf, kit] = await Promise.all([ctx.assets.tryGltf("models/grove.glb"), loadKit(ctx)]);
    if (!gltf) {
      await this.buildPlaceholder(ctx);
      return;
    }

    // Our own copy of the node tree (geometry stays shared / registry-owned), so the
    // cached GLTF is never mutated and a rebuild (seed / quality) starts clean.
    const world = gltf.scene.clone(true);
    world.name = "grove";
    this.scene.add(world);
    world.updateMatrixWorld(true);

    const [bark, barkLow, moss, mossArch, mossFg, mossArchFg] = await Promise.all([
      // the bark set is warm orange-brown with deep grooves; frame 04 / detail 03 read as a
      // smooth, warm light brown (lit ≈ #8A785D) with a fine, gently flowing grain of low
      // contrast: finer repeat, soft normals, shallow cavities, little warp (tuneBark)
      createBarkMaterial(ctx.assets, { scale: 1.25, roughness: 1, aoStrength: 0.45, color: "#DCD0CC" }),
      // the low branch: its top faces the key, yet frame 04 shows it dark brown under its
      // lit grass tuft — the same bark, darker
      createBarkMaterial(ctx.assets, { scale: 1.25, roughness: 1, aoStrength: 0.45, color: "#DCD0CC" }),
      // hero masses: the base under the long grass is what shows between the blades —
      // a soft, warm olive (frame 01, detail 01), lumpy: lighter on the sky-facing tops
      createMossLayerMaterial(ctx, {
        tint: "#FFE4A6",
        brightness: 4.2,
        saturation: 0.72,
        patchContrast: 0.3,
        patchColor: "#7A7442",
        sheen: 0.2,
        sheenColor: "#B4B88A",
        shade: { top: 1.25, bottom: 0.7, topColor: "#CFCB8A", topAmount: 0.3 },
      }),
      // arch cushions: a lit, light green carpet under the clover (frame 04, detail 03)
      createMossLayerMaterial(ctx, {
        tint: "#FFF4C8",
        brightness: 3.2,
        saturation: 0.75,
        patchContrast: 0.3,
        sheen: 0.1,
        // lighter yellow-green on the sky-facing tops, dark folds and undersides (detail 03)
        shade: { top: 1.5, bottom: 0.4, topColor: "#E4EC9A", topAmount: 0.45 },
        aoContrast: 1.2,
      }),
      // hero foreground band: dark grey-green, only faint shapes under the blur (detail 02)
      createMossLayerMaterial(ctx, { tint: "#C4C8B4", brightness: 1.7, saturation: 0.7, patchContrast: 0.2, sheen: 0.08 }),
      // arch foreground band: it faces the arch key, but frame 04 keeps it near black
      // (luma ≈ 18 under the blur): a dark base, the plants on it thinned out (archMoss)
      createMossLayerMaterial(ctx, { tint: "#A4A890", brightness: 3.0, saturation: 0.7, patchContrast: 0.2, sheen: 0.04 }),
    ]);
    bark.color.multiplyScalar(3.6);
    bark.normalScale.multiplyScalar(0.4);
    tuneBark(bark, { saturation: 0.55, cavity: 0.3, warp: 0.035, contrast: 0.45 });
    barkLow.color.multiplyScalar(3.6 * 0.35);
    barkLow.normalScale.multiplyScalar(0.4);
    tuneBark(barkLow, { saturation: 0.55, cavity: 0.3, warp: 0.035, contrast: 0.45 });
    this.own(bark);
    this.own(barkLow);
    this.own(moss);
    this.own(mossArch);
    this.own(mossFg);
    this.own(mossArchFg);
    this.mossArchFgShade = (mossArchFg.userData.mossLayerUniforms as { uMossShade?: { value: Vector3 } } | undefined)?.uMossShade?.value ?? null;

    const meshes = new Map<string, Mesh>();
    world.traverse((o) => {
      const m = o as Mesh;
      if (m.isMesh) meshes.set(m.name, m);
    });
    for (const [name, mesh] of meshes) {
      const fg = name.endsWith("__fg");
      if (name.startsWith("wood_")) {
        mesh.material = name === "wood_arch_low" ? barkLow : bark;
        ctx.layers.assign(mesh, "wood");
      } else {
        mesh.material = name.startsWith("moss_arch") ? (fg ? mossArchFg : mossArch) : fg ? mossFg : moss;
        ctx.layers.assign(mesh, "moss");
      }
      mesh.castShadow = !fg;
      mesh.receiveShadow = true;
    }
    this.heroCasters = ["moss_crown_mass", "moss_left_column", "moss_hero_upper"].map((n) => meshes.get(n)).filter((m): m is Mesh => !!m);

    // --- camera track ---------------------------------------------------------
    const found = this.rig.addPosesFromObject(world);
    const focus = new Map<string, number>();
    world.traverse((o) => {
      const match = /^cam_([A-Za-z]+)_([A-Za-z0-9]+)/.exec(o.name);
      const cam = o as PerspectiveCamera;
      if (match && cam.isPerspectiveCamera) focus.set(`${match[1]}_${match[2]}`, Number(o.userData.focus_distance_m) || 2);
    });
    const need = ["hero_main", "hero_out", "hero_p1", "hero_p2", "arch_in", "arch_main", "arch_out"];
    for (const n of need) if (!found.includes(n)) throw new Error(`grove.glb: missing camera cam_${n}`);
    this.buildTrack(focus);

    // --- key light / fill directions, set in the space of the main camera poses ---
    this.keyHero.copy(HERO_KEY_CAM).applyQuaternion((this.rig.pose("hero_main") as CameraPose).quaternion).normalize();
    this.fillHero.copy(HERO_FILL_CAM).applyQuaternion((this.rig.pose("hero_main") as CameraPose).quaternion).normalize();
    this.keyArch.copy(ARCH_KEY_CAM).applyQuaternion((this.rig.pose("arch_main") as CameraPose).quaternion).normalize();
    const key = this.lights.key;
    key.castShadow = true;
    key.shadow.mapSize.set(ctx.quality.shadowMapSize, ctx.quality.shadowMapSize);
    key.shadow.bias = -0.0004;
    key.shadow.normalBias = 0.01;
    key.shadow.camera.near = 0.2;
    key.shadow.camera.far = 16;
    key.shadow.camera.layers.enableAll();
    this.lights.fill.position.set(0, 1, 0);

    // --- foreground of the hero frame: its own group so it can leave the frame ----
    const heroFgMesh = meshes.get("moss_hero_near__fg");
    if (heroFgMesh) {
      const g = new Group();
      g.name = "hero_fg";
      world.add(g);
      g.add(heroFgMesh);
      this.heroFg = g;
    }

    // --- vegetation -----------------------------------------------------------
    const views = (names: string[], weight = 1, extra: Partial<ScatterView> = {}): ScatterView[] =>
      names.map((n) => this.scatterView(this.rig.pose(n) as CameraPose, weight, extra));
    const along = (times: number[], weight = 1, extra: Partial<ScatterView> = {}): ScatterView[] =>
      times.map((t) => this.scatterView(this.rig.evaluate(t, { position: new Vector3(), quaternion: new Quaternion(), fov: 35 }), weight, extra));

    const pick = (...names: string[]) => names.map((n) => meshes.get(n)).filter((m): m is Mesh => !!m);
    // tags: "hero" (frames 01 / 02, plus the arm above them that narrow screens see),
    // "t01" (the descent past the big mass), "wall" (its face below the strip the hero
    // frames see: the 9.2 / 9.6 s views moved by WALL_VIEW_SHIFT, so the wall's own leaves
    // stay off the hero's mound and the keyhole's chin), "arch"
    const wallViews = [9.2, 9.6].map((t) => {
      const pose = this.rig.evaluate(t, { position: new Vector3(), quaternion: new Quaternion(), fov: 35 });
      pose.position.y -= WALL_VIEW_SHIFT.down;
      pose.position.x += WALL_VIEW_SHIFT.right;
      pose.quaternion.multiply(new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), -WALL_VIEW_SHIFT.yaw));
      return this.scatterView(pose, 1, { tag: "wall" });
    });
    const heroMain = this.rig.pose("hero_main") as CameraPose;
    const heroTall: ScatterView = {
      position: heroMain.position.clone(),
      quaternion: heroMain.quaternion.clone().multiply(new Quaternion().setFromAxisAngle(new Vector3(1, 0, 0), HERO_TALL_VIEW.pitch)),
      fov: HERO_TALL_VIEW.fov,
      aspect: HERO_TALL_VIEW.aspect,
      weight: 1,
      margin: 0.1,
      tag: "hero",
    };
    const heroViews = [
      ...views(["hero_main", "hero_out"], 1, { tag: "hero" }),
      heroTall,
      ...along([8.4, 8.8, 9.2, 9.6], 1, { tag: "t01" }),
      ...wallViews,
      ...views(["arch_in", "arch_main"], 0.3, { maxDistance: 6, tag: "arch" }),
    ];
    // time-sliced builds (identical plants): the current episode keeps rendering meanwhile
    const heroVeg = await buildVegetationAsync(ctx, heroMeadow, {
      meshes: pick("moss_crown_mass", "moss_left_column", "moss_hero_upper"),
      parent: world,
      views: heroViews,
      kit,
      label: "hero",
    });
    this.addVegetation("hero", heroVeg);
    if (this.heroFg && heroFgMesh) {
      this.addVegetation(
        "heroFg",
        await buildVegetationAsync(ctx, foregroundMoss, {
          meshes: [heroFgMesh],
          parent: this.heroFg,
          views: views(["hero_main", "hero_out"]),
          kit,
          label: "hero-fg",
        }),
      );
    }
    const archViews = [...views(["arch_in", "arch_main", "arch_out"], 1), ...along([9.85, 13.8, TRACK.exit], 0.7)];
    this.archVeg = await buildVegetationAsync(ctx, archMoss, {
      meshes: pick("moss_arch_left", "moss_arch_right", "moss_arch_low", "moss_arch_ground", "moss_arch_near__fg", "wood_arch_left", "wood_arch_right", "wood_arch_low"),
      parent: world,
      views: archViews,
      kit,
      label: "arch",
    });
    this.addVegetation("arch", this.archVeg);
  }

  private addVegetation(label: string, build: VegetationBuild): void {
    this.vegetation.push({ label, build });
    build.report(this.ctx, `${label}.`);
  }

  private scatterView(pose: CameraPose, weight: number, extra: Partial<ScatterView> = {}): ScatterView {
    return { position: pose.position.clone(), quaternion: pose.quaternion.clone(), fov: pose.fov, aspect: REFERENCE_ASPECT, weight, margin: 0.15, ...extra };
  }

  /**
   * hero_main → hero_out (0.7 → 2.2 s), hold to 8.0, then one C¹ path through
   * p1, the 9.15 s key, p2, arch_in, arch_main, arch_out and on to the exit pose (16.4 s): monotone
   * cubic Hermite per component (no overshoot between unevenly spaced keys), sampled
   * at 30 Hz into the rig's track.
   */
  private buildTrack(focus: Map<string, number>): void {
    const pose = (n: string) => this.rig.pose(n) as CameraPose;
    const archMain = pose("arch_main");
    const archOut = pose("arch_out");
    const p1 = pose("hero_p1");
    const p2 = pose("hero_p2");
    const p1Focus = focus.get("hero_p1") ?? 1.5;
    const p2Focus = focus.get("hero_p2") ?? 1.75;
    // slow along the keyhole: on the straight line p1 → p2, still looking like p1
    const p12: CameraPose = { position: p1.position.clone().lerp(p2.position, P12_FRACTION), quaternion: p1.quaternion.clone(), fov: p1.fov };
    const exit: CameraPose = {
      position: archOut.position.clone().add(tmpV.copy(archOut.position).sub(archMain.position).multiplyScalar(0.45)),
      quaternion: archMain.quaternion.clone().slerp(archOut.quaternion, 1.35),
      fov: archOut.fov,
    };
    const keys: SplineKey[] = [
      { t: TRACK.descentStart, pose: pose("hero_out"), focus: focus.get("hero_out") ?? 1.65 },
      { t: TRACK.p1, pose: p1, focus: p1Focus },
      { t: TRACK.p12, pose: p12, focus: p1Focus + (p2Focus - p1Focus) * P12_FRACTION },
      { t: TRACK.p2, pose: p2, focus: p2Focus },
      { t: TRACK.archIn, pose: pose("arch_in"), focus: focus.get("arch_in") ?? 3.9 },
      { t: TRACK.archMain, pose: archMain, focus: focus.get("arch_main") ?? 4.75 },
      // the exit tips down onto the lower mossy forms: frame 05 keeps them sharp, so the
      // focus (near-field blur only) comes in to them instead of the extras' 4.2 m
      { t: TRACK.archOut, pose: archOut, focus: Math.min(focus.get("arch_out") ?? 4.2, 2.6) },
      { t: TRACK.exit, pose: exit, focus: 2.2 },
    ];
    // quaternions on one hemisphere
    for (let i = 1; i < keys.length; i++) if (keys[i].pose.quaternion.dot(keys[i - 1].pose.quaternion) < 0) {
      const q = keys[i].pose.quaternion;
      keys[i] = { ...keys[i], pose: { ...keys[i].pose, quaternion: new Quaternion(-q.x, -q.y, -q.z, -q.w) } };
    }
    const comps = (k: SplineKey) => [k.pose.position.x, k.pose.position.y, k.pose.position.z, k.pose.quaternion.x, k.pose.quaternion.y, k.pose.quaternion.z, k.pose.quaternion.w, k.pose.fov, k.focus];
    const values = keys.map(comps);
    const times = keys.map((k) => k.t);
    const tangents = monotoneTangents(times, values, { startAtRest: true, endScale: 0.6 });

    const track: PoseKey[] = [
      { t: 0, pose: "hero_main" },
      { t: TRACK.heroMainHold, pose: "hero_main" },
      { t: TRACK.heroOut, pose: "hero_out", ease: "inOutSine" },
    ];
    this.focusKeys = [
      { t: 0, focus: focus.get("hero_main") ?? 1.75 },
      { t: TRACK.heroMainHold, focus: focus.get("hero_main") ?? 1.75 },
      { t: TRACK.heroOut, focus: focus.get("hero_out") ?? 1.65 },
    ];
    const step = 1 / 30;
    for (let t = TRACK.descentStart; t <= TRACK.exit + 1e-6; t += step) {
      const v = hermite(times, values, tangents, Math.min(t, TRACK.exit));
      const q = new Quaternion(v[3], v[4], v[5], v[6]).normalize();
      track.push({ t, pose: { position: new Vector3(v[0], v[1], v[2]), quaternion: q, fov: v[7] }, ease: "linear" });
      this.focusKeys.push({ t, focus: v[8] });
    }
    this.rig.fit.subjectX = 0;
    this.rig.setTrack(track);
  }

  /** DOF focus distance at video time t (follows the camera extras). */
  private focusAt(t: number): number {
    const k = this.focusKeys;
    if (k.length === 0) return 2;
    if (t <= k[0].t) return k[0].focus;
    for (let i = 1; i < k.length; i++) {
      if (t <= k[i].t) {
        const a = k[i - 1];
        const b = k[i];
        const s = (t - a.t) / Math.max(1e-6, b.t - a.t);
        const e = i <= 2 ? 0.5 - 0.5 * Math.cos(Math.PI * s) : s;
        return a.focus + (b.focus - a.focus) * e;
      }
    }
    return k[k.length - 1].focus;
  }

  /** Narrow screens: the fit of the beat (NARROW_FIT), set before the rig poses the camera. */
  update(frame: FrameState, local: SceneLocal, look: LookParams): void {
    const fit = this.rig.fit;
    const keys = NARROW_FIT;
    const t = local.t;
    let i = 0;
    while (i < keys.length - 1 && keys[i + 1].t <= t) i++;
    const a = keys[i];
    const b = keys[Math.min(i + 1, keys.length - 1)];
    const k = b.t > a.t ? smoothstep(a.t, b.t, t) : 0;
    const wide = smoothstep(FIT_ASPECT.phone, FIT_ASPECT.tablet, frame.viewport.aspect);
    for (const name of FIT_KEYS) {
      const va = a.phone[name] + ((a.tablet ?? a.phone)[name] - a.phone[name]) * wide;
      const vb = b.phone[name] + ((b.tablet ?? b.phone)[name] - b.phone[name]) * wide;
      fit[name] = va + (vb - va) * k;
    }
    super.update(frame, local, look);
  }

  getLook(_frame: FrameState, local: SceneLocal, base: LookParams): LookParams {
    const t = local.t;
    if (this.placeholder) {
      if (t > 14.6) base.exposure *= 1 - 0.6 * smoothstep(14.6, 15.8, t);
      return base;
    }
    const f = this.focusAt(t);
    base.dof.focusDistance = f;
    base.dof.nearStart = f * DOF_NEAR_START;
    base.dof.nearEnd = f * DOF_NEAR_END;
    // hero near field (HERO_DOF), then T01 (motion/01): over the crest (8.0–8.8 s) the near
    // part of the mound (1.1–1.25 m) stays sharp; along the wall (8.8–9.75 s) only the
    // nearest blades of its face (1.77–1.94 m) are touched, the keyhole's edges (≥ 2 m)
    // and everything beyond stay sharp
    const heroPhase = 1 - smoothstep(8.0, 8.3, t);
    base.dof.nearStart += (HERO_DOF.start - base.dof.nearStart) * heroPhase;
    base.dof.nearEnd += (HERO_DOF.end - base.dof.nearEnd) * heroPhase;
    const crest = smoothstep(8.0, 8.3, t) * (1 - smoothstep(8.6, 8.85, t));
    const wall = smoothstep(8.6, 8.85, t) * (1 - smoothstep(9.7, 9.95, t));
    base.dof.nearEnd += (T01_DOF.crestEnd - base.dof.nearEnd) * crest + (T01_DOF.wallEnd - base.dof.nearEnd) * wall;
    base.dof.nearStart += (T01_DOF.crestStart - base.dof.nearStart) * crest;
    // T01: the wall's face in shade (fill down), the high key (T01_KEY) on the tops of its
    // lobes, the keyhole's edges glowing in the raised rim light
    const t01 = smoothstep(8.3, 8.85, t) * (1 - smoothstep(9.75, 10.15, t));
    base.lights.keyIntensity *= 1 + (T01_LIGHT.key - 1) * t01;
    base.lights.fillIntensity *= 1 + (T01_LIGHT.fill - 1) * t01;
    base.lights.rimIntensity *= 1 + (T01_LIGHT.rim - 1) * t01;
    // exit (EXIT): the light dims while the camera tips towards the dark lower forms; frame
    // 05 keeps the upper left lit and the lower forms near black
    if (t > 14.6) {
      const exit = smoothstep(14.7, 15.4, t);
      base.vignette.bottom += (EXIT.vignetteBottom - base.vignette.bottom) * exit;
      base.vignette.bottomSize += (EXIT.vignetteSize - base.vignette.bottomSize) * exit;
      const dim = 1 - EXIT.dim * smoothstep(14.8, 15.6, t);
      base.lights.keyIntensity *= dim;
      base.lights.fillIntensity *= dim;
      // the whole frame darkens (frame 05: the empty background at ≈ 1/3 of frame 04's)
      base.exposure *= 1 - EXIT.exposure * smoothstep(14.7, 15.2, t);
    }
    // the grove is gone (black) before the canyon's lights come up at 15.8 s
    if (t > EXIT.blackFrom) base.exposure *= 1 - smoothstep(EXIT.blackFrom, EXIT.blackAt, t);
    return base;
  }

  protected animate(frame: FrameState, local: SceneLocal): void {
    if (this.placeholder) return;
    const t = local.t;
    const cam = this.camera;

    // the arch's plants sit behind / below the hero masses: never seen before the descent
    if (this.archVeg) this.archVeg.group.visible = t >= ARCH_VEG_FROM;

    // key light: key_hero → T01 key → key_arch across the descent, aimed at the focus point;
    // the hero masses shade themselves until the camera has dropped below them
    const dir = tmpV.copy(this.keyHero).lerp(T01_KEY, smoothstep(8.3, 8.85, t)).lerp(this.keyArch, smoothstep(9.8, 10.2, t)).normalize();
    for (const m of this.heroCasters) m.castShadow = t < 9.95;
    const f = this.focusAt(t);
    const target = cam.getWorldDirection(tmpTarget).multiplyScalar(f).add(cam.position);
    this.lights.setKeyDirection(dir, target, 8);
    this.lights.setRimDirection(RIM_DIR, target, 8);
    // sky fill: above + camera side in the hero, straight above from the descent on
    this.lights.fill.position.copy(tmpFill.copy(this.fillHero).lerp(WORLD_UP, smoothstep(8.0, 8.8, t)).normalize());
    // shadow frustum covers the visible frame around the focus distance
    const frameH = 2 * f * Math.tan((cam.fov * Math.PI) / 360);
    // the exit looks down a long slope: cover it all, or the frustum edge cuts a lit band
    const extent = Math.max(1.0, frameH * 1.05);
    const sc = this.lights.key.shadow.camera;
    sc.left = -extent * 1.2;
    sc.right = extent * 1.2;
    sc.top = extent;
    sc.bottom = -extent;
    sc.updateProjectionMatrix();

    // exit: the near mass's shaded fronts come up into frame 05's dim saddle (EXIT_FG)
    if (this.mossArchFgShade) this.mossArchFgShade.x = 1 + EXIT_FG.front * smoothstep(EXIT_FG.from, EXIT_FG.to, t);

    // hero foreground: rides with the camera from 8.0 s and drops out of the frame
    if (this.heroFg) {
      if (t <= TRACK.descentStart) {
        this.heroFg.position.set(0, -HERO_FG_DROP, 0);
        this.heroFg.visible = true;
      } else if (t < 8.7) {
        this.rig.evaluate(TRACK.descentStart, this.tmpPose);
        this.fgStart.copy(this.tmpPose.position);
        this.rig.evaluate(t, this.tmpPose);
        this.heroFg.position.copy(this.tmpPose.position).sub(this.fgStart);
        this.heroFg.position.y -= HERO_FG_DROP + 0.5 * smoothstep(8.0, 8.65, t);
        this.heroFg.visible = true;
      } else {
        this.heroFg.visible = false;
      }
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
        // instances in chunks intersecting the view → stats / debug panel ("view.hero" …)
        this.ctx.reportInstances(`view.${v.label}`, n);
      }
    }
  }

  /**
   * arch → canyon: the configured dip (0.9) darkens 15.2 s far below frame 05, where the
   * upper left is still lit and only the lower forms are black: no dip in the first third
   * of the window (the grove darkens itself: vignette, dimmer light, black by 15.85 s),
   * the configured dip from the middle on.
   */
  getTransitionOverride(tr: ActiveTransition): Partial<TransitionParams> | null {
    if (tr.id !== "arch-canyon" || tr.reduced) return null;
    this.trOverride.dip = 0.9 * smoothstep(0.35, 0.65, tr.k);
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
    return out;
  }

  // ---------------------------------------------------------------------------
  // PLACEHOLDER fallback (grove.glb missing)
  // ---------------------------------------------------------------------------

  private async buildPlaceholder(ctx: SceneContext): Promise<void> {
    this.placeholder = true;
    const [moss, bark] = await Promise.all([createMossBaseMaterial(ctx.assets), createBarkMaterial(ctx.assets)]);
    const far = createFarMaterial({ color: "#4A5440" });
    const root = new Group();
    root.name = "PLACEHOLDER_grove";
    this.scene.add(root);
    const rng = (label: string) => ctx.rng(label);
    placeholderMesh(ctx, root, blobGeometry(rng("hero-overhang"), { radius: 0.34, scale: [1.5, 0.75, 1] }), moss, "moss", { position: [-0.62, 0.4, -0.1] });
    placeholderMesh(ctx, root, blobGeometry(rng("hero-left"), { radius: 0.36, flattenBottom: 0.3 }), moss, "moss", { position: [-0.74, -0.44, 0] });
    placeholderMesh(ctx, root, blobGeometry(rng("hero-mound"), { radius: 0.42, scale: [1.3, 0.8, 1], flattenBottom: 0.5 }), moss, "moss", { position: [0.48, -0.66, 0.05] });
    const trunk = (pts: Vector3[], r0: number, label: string) =>
      placeholderMesh(ctx, root, tubeGeometry(pts, (u) => r0 - 0.1 * u, { rng: rng(label), bumps: 0.14 }), bark, "wood");
    trunk([new Vector3(-1.35, -4.9, -1.35), new Vector3(-1.15, -3.6, -1.45), new Vector3(-0.85, -2.4, -1.6), new Vector3(-0.35, -1.75, -1.7)], 0.24, "arch-l");
    trunk([new Vector3(1.4, -4.9, -1.2), new Vector3(1.2, -3.5, -1.4), new Vector3(0.85, -2.3, -1.6), new Vector3(0.3, -1.72, -1.7)], 0.26, "arch-r");
    placeholderMesh(ctx, root, blobGeometry(rng("arch-far"), { radius: 1.8, roughness: 0.2 }), far, "far", { position: [0.4, -2.6, -9] });
    this.lights.setKeyDirection(new Vector3(-0.45, -0.75, -0.5), new Vector3(0, -1.5, 0), 8);
    const P = CameraRig.lookAtPose;
    this.rig.setTrack([
      { t: 0, pose: P([0, 0.02, 1.75], [0, 0, 0]) },
      { t: 8, pose: P([0.03, -0.05, 1.62], [0.02, -0.07, 0]) },
      { t: 10, pose: P([0, -3.0, 3.3], [0, -3.2, -1.5]) },
      { t: 15.5, pose: P([0, -3.05, 2.95], [0, -3.25, -1.5]) },
    ]);
  }
}

// ---------------------------------------------------------------------------
// Monotone cubic Hermite (Fritsch–Butland tangents), per component
// ---------------------------------------------------------------------------

function monotoneTangents(times: number[], values: number[][], opts: { startAtRest: boolean; endScale: number }): number[][] {
  const n = times.length;
  const dims = values[0].length;
  const out: number[][] = [];
  for (let i = 0; i < n; i++) {
    const m: number[] = [];
    for (let d = 0; d < dims; d++) {
      if (i === 0) {
        m.push(opts.startAtRest ? 0 : (values[1][d] - values[0][d]) / (times[1] - times[0]));
      } else if (i === n - 1) {
        m.push(((values[i][d] - values[i - 1][d]) / (times[i] - times[i - 1])) * opts.endScale);
      } else {
        const h0 = times[i] - times[i - 1];
        const h1 = times[i + 1] - times[i];
        const s0 = (values[i][d] - values[i - 1][d]) / h0;
        const s1 = (values[i + 1][d] - values[i][d]) / h1;
        m.push(s0 * s1 <= 0 ? 0 : (3 * (h0 + h1)) / ((2 * h1 + h0) / s0 + (h1 + 2 * h0) / s1));
      }
    }
    out.push(m);
  }
  return out;
}

function hermite(times: number[], values: number[][], tangents: number[][], t: number): number[] {
  const n = times.length;
  if (t <= times[0]) return values[0].slice();
  if (t >= times[n - 1]) return values[n - 1].slice();
  let i = 0;
  while (i < n - 2 && t > times[i + 1]) i++;
  const h = times[i + 1] - times[i];
  const s = (t - times[i]) / h;
  const s2 = s * s;
  const s3 = s2 * s;
  const h00 = 2 * s3 - 3 * s2 + 1;
  const h10 = s3 - 2 * s2 + s;
  const h01 = -2 * s3 + 3 * s2;
  const h11 = s3 - s2;
  return values[i].map((v0, d) => h00 * v0 + h10 * h * tangents[i][d] + h01 * values[i + 1][d] + h11 * h * tangents[i + 1][d]);
}

/**
 * Bark look on top of the shared bark factory (frame 04, detail 03): albedo towards grey
 * by `saturation`, its luminance contrast around the mean scaled by `contrast` (the set's
 * dark grooves otherwise read as stripes), darkened in the crevices by the texture AO
 * (ORM.r ^ cavity) — the texture AO otherwise only reaches indirect light — and the grain
 * set flowing instead of ruler-straight: U is waved slowly along V (`warp` in UV units;
 * the tangent frame comes from UV derivatives, so the normals follow).
 */
function tuneBark(mat: MeshStandardMaterial, o: { saturation: number; cavity: number; warp: number; contrast: number }): void {
  // mean luminance of colour × bark map (linear; the map averages ≈ 0.167)
  const mean = 0.167 * (0.2126 * mat.color.r + 0.7152 * mat.color.g + 0.0722 * mat.color.b);
  const uniforms = {
    uBarkSaturation: { value: o.saturation },
    uBarkCavity: { value: o.cavity },
    uBarkWarp: { value: o.warp },
    uBarkContrast: { value: o.contrast },
    uBarkMean: { value: mean },
  };
  const base = mat.onBeforeCompile;
  mat.onBeforeCompile = (shader: WebGLProgramParametersWithUniforms, renderer: WebGLRenderer) => {
    base.call(mat, shader, renderer);
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader.replace("#include <common>", `#include <common>
uniform float uBarkWarp;`).replace(
      "#include <uv_vertex>",
      `#include <uv_vertex>
#ifdef USE_MAP
  {
    vec2 bw = vec2(uBarkWarp * (sin(vMapUv.y * 3.2 + 2.5 * vMapUv.x) + 0.3 * sin(vMapUv.y * 7.0 - 1.3 * vMapUv.x)), 0.0);
    vMapUv += bw;
  #ifdef USE_NORMALMAP
    vNormalMapUv += bw;
  #endif
  #ifdef USE_ROUGHNESSMAP
    vRoughnessMapUv += bw;
  #endif
  #ifdef USE_AOMAP
    vAoMapUv += bw;
  #endif
  }
#endif`,
    );
    shader.fragmentShader = shader.fragmentShader
      .replace("#include <common>", `#include <common>
uniform float uBarkSaturation;
uniform float uBarkCavity;
uniform float uBarkContrast;
uniform float uBarkMean;`)
      .replace(
        "#include <color_fragment>",
        `#include <color_fragment>
  {
    float bl = dot(diffuseColor.rgb, vec3(0.2126, 0.7152, 0.0722));
    diffuseColor.rgb = mix(vec3(bl), diffuseColor.rgb, uBarkSaturation);
    diffuseColor.rgb *= (uBarkMean + (bl - uBarkMean) * uBarkContrast) / max(bl, 1e-4);
  }
#ifdef USE_AOMAP
  diffuseColor.rgb *= pow(max(texture2D(aoMap, vAoMapUv).r, 0.15), uBarkCavity);
#endif`,
      );
  };
  const key = mat.customProgramCacheKey();
  mat.customProgramCacheKey = () => `${key}|grove-bark`;
}
