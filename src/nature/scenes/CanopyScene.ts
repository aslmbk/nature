/**
 * Canopy scene set — episodes `canopy` (43.3–48.5 s) and `canopyClose` (48.5–52.8 s):
 * one tree crown on black, seen from below and in front, lit from above-front.
 *
 * `models/canopy.glb` (assets-src/blender/build_canopy.py) holds 46 twig clusters
 * (`wood_canopy_cNN`, origin = cluster base, extras `reveal_order` / `radius_m`), the
 * scaffold limbs, two dark far shells and the cameras. The leaves are instanced here from
 * the kit (`leaf_canopy_a/b/c`) in small sprays on the twigs (canopy/foliage.ts, built
 * in ≈ 8 ms slices while the previous episode keeps rendering) and drawn with one
 * patched material (canopy/leafMaterial.ts).
 *
 * Story (canopy/timeline.ts, all pure in `local.t`):
 *  - 43.05–44.3 reveal: the whole crown grows about its bottom centre (it starts as a
 *    small bush at the bottom edge and settles centred), and every cluster scales in
 *    from its base in `reveal_order` (inner and lower first, overlapping windows), its
 *    leaves unfurling from the base outwards; the camera holds the opening pose
 *    (`cam_canopy_in`, moved back 0.65 m), the stone → canopy slide (follow 0.4) carries
 *    the picture up. Scrolling back closes it the same way.
 *  - 44.3–52.8 camera: canopy_in → canopy_main → canopy_out → canopyClose_main →
 *    canopyClose_out on a monotone C¹ path (no stop at the keys), focus from the extras.
 *  - 52.42–53.9 exit (mode `over`, transparent clear): the picture darkens, the dark
 *    middle opens and the leaves part towards the frame edges in clumps (area
 *    preserving); twigs retract into their bases, scaffold and backdrop fade, so the
 *    finale shows through.
 * Reduced motion: no growth and no parting — the configured crossfades only.
 */
import {
  Color,
  IcosahedronGeometry,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  Quaternion,
  Vector3,
  type BufferGeometry,
  type Object3D,
  type PerspectiveCamera,
} from "three";
import { CameraRig, type CameraPose, type PoseKey } from "../CameraRig";
import { BaseScene } from "../core/BaseScene";
import { clamp, saturate, smoothstep } from "../core/math";
import { createBarkMaterial } from "../rendering/Materials";
import type { FrameState, LookParams, SceneContext, SceneLocal } from "../types";
import { yieldToBrowser } from "../vegetation/slices";
import { loadKit } from "../vegetation/Vegetation";
import { buildFoliage, LEAF_ITEMS, type FoliageBuild, type FoliageHollow } from "./canopy/foliage";
import { createLeafMaterial, createLeafMesh, createLeafUniforms, type LeafUniforms } from "./canopy/leafMaterial";
import { CANOPY_T, clusterGrow, crownScale, exitDim, exitProgress, farGrow, globalReveal, growScale, hermite, holeRadius, monotoneTangents } from "./canopy/timeline";

const FAR_INNER_ID = 46;
const FAR_OUTER_ID = 47;
/** Leaves per `quality.leaves` unit (high 16000 → 96k leaves, medium 45k, low 21k). */
const LEAVES_PER_BUDGET = 6;
/** `quality.leaves` of the high preset: the reference for leaf size and shading density. */
const HIGH_LEAF_BUDGET = 16000;
/** Camera keys (pose, video seconds) — the `video_time_s` extras of the cameras. */
const CAMERA_KEYS: [string, number][] = [
  ["canopy_in", CANOPY_T.in],
  ["canopy_main", CANOPY_T.main],
  ["canopy_out", CANOPY_T.out],
  ["canopyClose_main", CANOPY_T.closeMain],
  ["canopyClose_out", CANOPY_T.closeOut],
];
/** Camera path sampling rate (keys per second, linear in between). */
const TRACK_HZ = 30;
/** Reference frame aspect (the poses' design aspect). */
const DESIGN_ASPECT = 1440 / 1020;
/** Height in the `cam_canopy_in` frame (0 top … 1 bottom) of the point the crown grows from. */
const GROW_FROM_V = 0.97;
/**
 * `cam_canopy_in` is moved back along its view axis by this much (m). Measured on the
 * reference: the crown is 1.22–1.36 × wider in frame 16 (46.5 s) than in frame 15
 * (44.3 s), the GLB keys give 1.14 × (5.55 → 4.85 m); canopy_main already matches
 * frame 16, so the opening pose is the one that moves (6.2 m from the crown centre).
 */
const IN_DOLLY_M = 0.65;
/**
 * The key: `key_canopy` (above, in front, from the left) turned this far towards a
 * frontal light from slightly above. Frame 16's crown is lit almost evenly over its
 * front, its lower right included (the GLB direction leaves that side in the crown's
 * own shade: median 0.14 against the reference's 0.28 luminance there).
 */
const KEY_FRONTAL = 0.45;

interface Cluster {
  id: number;
  mesh: Mesh;
  /** reveal_order extra, 0 (first) … 1 (last). */
  order: number;
  base: Vector3;
  radius: number;
}

interface FadeItem {
  mesh: Mesh;
  material: MeshStandardMaterial;
  /** Rest transform (for growth about a pivot). */
  position: Vector3;
}

const tmpV = new Vector3();
const tmpQ = new Quaternion();
/** Backdrop inside the crown: deep green from afar (frame 16), black in the close-up (frame 17). */
const BACKDROP_FAR = new Color("#14200C");
const BACKDROP_CLOSE = new Color("#060A04");

export class CanopyScene extends BaseScene {
  readonly id = "canopy" as const;
  private clusters: Cluster[] = [];
  private scaffold: FadeItem | null = null;
  private farInner: Mesh | null = null;
  private farOuter: Mesh | null = null;
  private backdrop: { mesh: Mesh; material: MeshBasicMaterial } | null = null;
  private leafU: LeafUniforms | null = null;
  private foliage: FoliageBuild | null = null;
  private focus: { times: number[]; values: number[][]; tangents: number[][] } | null = null;
  /** Hidden centre of the crown: the scaffold grows out of it. */
  private readonly hidden = new Vector3(0, -0.35, -0.95);
  private placeholder: Mesh | null = null;
  /** The canopy.glb copy (scaled about `pivot` while the crown grows). */
  private world: Object3D | null = null;
  private readonly pivot = new Vector3(0, -1.5, 0);
  private lobePhase = 0;
  /** The opening pose actually used (cam_canopy_in moved back, see IN_DOLLY_M). */
  private inPose: CameraPose | null = null;
  private readonly state = { reveal: 0, scale: 1, exit: 0, hole: 0, twigs: 0 };

  protected async build(ctx: SceneContext): Promise<void> {
    const [gltf, kit] = await Promise.all([ctx.assets.tryGltf("models/canopy.glb"), loadKit(ctx)]);
    if (!gltf) {
      this.buildPlaceholder(ctx);
      return;
    }
    // the GLB parse (or the last texture upload) ran in this task: go on in a fresh one, so
    // the build never extends a long engine task (the same after the materials below)
    await yieldToBrowser();
    // our own copy of the node tree (geometry stays shared / registry-owned): the cached
    // GLTF is never modified, a rebuild (seed / quality) starts clean
    const world = gltf.scene.clone(true);
    world.name = "canopy";
    this.scene.add(world);
    world.updateMatrixWorld(true);
    this.world = world;
    this.lobePhase = ctx.rng("canopy-exit-lobes").range(0, Math.PI * 2);

    // --- materials ------------------------------------------------------------------
    const [twigBark, scaffoldBark] = await Promise.all([
      // twigs read as dark grey-brown lines between the leaves (frames 15 / 16)
      createBarkMaterial(ctx.assets, { color: "#7A736C", roughness: 0.95 }),
      // the limbs inside the crown: dark, they only show as silhouettes in the hollow
      createBarkMaterial(ctx.assets, { color: "#26221F", roughness: 0.95 }),
    ]);
    await yieldToBrowser();
    // the scaffold fades out in the exit (mode `over`: its coverage would hide the
    // finale); transparent from the start so the program never changes
    scaffoldBark.transparent = true;

    const meshes = new Map<string, Mesh>();
    world.traverse((o) => {
      const m = o as Mesh;
      if (m.isMesh) meshes.set(m.name, m);
    });
    const clusterMeshes = [...meshes.entries()].filter(([n]) => /^wood_canopy_c\d+$/.test(n)).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
    if (clusterMeshes.length > FAR_INNER_ID) throw new Error(`canopy.glb: ${clusterMeshes.length} clusters, the leaf uniforms hold ${FAR_INNER_ID}`);
    this.clusters = clusterMeshes.map(([, mesh], id) => {
      mesh.material = twigBark;
      ctx.layers.assign(mesh, "wood");
      const order = Number(mesh.userData.reveal_order);
      const radius = Number(mesh.userData.radius_m);
      return {
        id,
        mesh,
        order: Number.isFinite(order) ? saturate(order) : id / Math.max(1, clusterMeshes.length - 1),
        base: mesh.getWorldPosition(new Vector3()),
        radius: Number.isFinite(radius) && radius > 0 ? radius : 0.8,
      };
    });
    const scaffold = meshes.get("wood_canopy_scaffold");
    if (scaffold) {
      scaffold.material = scaffoldBark;
      ctx.layers.assign(scaffold, "wood");
      this.scaffold = { mesh: scaffold, material: scaffoldBark, position: scaffold.position.clone() };
    }
    // far shells: the surfaces the dim far leaves are scattered on. The outer one is not
    // drawn (black on black; lit, its specular turns it grey). The inner cap stays as
    // an unlit, almost black green backdrop: the gaps inside the crown read as deep
    // foliage instead of holes (frame 16); towards the close-up it goes black (frame 17).
    // It fades in with the crown and out first in the exit (mode `over`: its coverage
    // would hide the finale).
    this.farInner = meshes.get("far_canopy_inner") ?? null;
    this.farOuter = meshes.get("far_canopy_outer") ?? null;
    if (this.farOuter) this.farOuter.visible = false;
    if (this.farInner) {
      const material = new MeshBasicMaterial({ name: "SilvaCanopyBackdrop", color: BACKDROP_FAR.clone(), transparent: true, depthWrite: true });
      this.farInner.material = material;
      ctx.layers.assign(this.farInner, "far");
      this.backdrop = { mesh: this.farInner, material };
    }

    // --- camera path and focus --------------------------------------------------------
    this.rig.addPosesFromObject(world);
    const focusOf = new Map<string, number>();
    world.traverse((o) => {
      const match = /^cam_([A-Za-z]+)_([A-Za-z0-9]+)/.exec(o.name);
      if (match && (o as PerspectiveCamera).isPerspectiveCamera) focusOf.set(`${match[1]}_${match[2]}`, Number(o.userData.focus_distance_m));
    });
    this.buildTrack(focusOf);
    // the crown grows about the point of the crown-centre plane seen at the bottom middle
    // of the opening frame
    const camIn = this.inPose;
    if (camIn) {
      const fwd = tmpV.set(0, 0, -1).applyQuaternion(camIn.quaternion);
      const depth = -camIn.position.dot(fwd);
      const tanV = Math.tan((camIn.fov * Math.PI) / 360);
      const ray = new Vector3(0, -(GROW_FROM_V * 2 - 1) * tanV, -1).applyQuaternion(camIn.quaternion);
      this.pivot.copy(camIn.position).addScaledVector(ray, depth > 0.1 ? depth : 5);
    }

    // --- lights: key from key_canopy (above-front, slightly left) turned towards the lens
    // axis (KEY_FRONTAL), fill from the lens side
    const keyNode = world.getObjectByName("key_canopy");
    const keyDir = new Vector3(0.2608, -0.8124, -0.5215);
    if (keyNode) {
      keyNode.updateMatrixWorld(true);
      keyDir.set(0, 0, -1).transformDirection(keyNode.matrixWorld);
    }
    keyDir.lerp(new Vector3(0, -0.2, -1).normalize(), KEY_FRONTAL).normalize();
    this.lights.setKeyDirection(keyDir, new Vector3(0, 0, 0), 8);
    this.lights.fill.position.set(0, 0.45, 1);
    this.lights.scale.rim = 0;

    // --- leaves -----------------------------------------------------------------------
    const geoms = LEAF_ITEMS.map((n) => kit.meshes.get(n)?.geometry ?? null);
    if (geoms.some((g) => !g)) {
      console.warn("[canopy] kit.glb has no leaf_canopy_a/b/c — crown without leaves");
      return;
    }
    const kitGeoms = geoms as BufferGeometry[];
    const itemLength = kitGeoms.map((g) => {
      g.computeBoundingBox();
      return g.boundingBox ? g.boundingBox.max.y : 0.025;
    });
    const itemArea = kitGeoms.map(bladeArea);
    const farInnerBase = this.farInner ? this.farInner.getWorldPosition(new Vector3()) : new Vector3();
    const farOuterBase = this.farOuter ? this.farOuter.getWorldPosition(new Vector3()) : new Vector3();
    const budget = Math.max(500, ctx.quality.leaves);
    const sizeScale = clamp(Math.pow(HIGH_LEAF_BUDGET / budget, 0.35), 1, 1.6);
    // time-sliced (≈ 8 ms slices, canopy/foliage.ts): the current episode keeps rendering;
    // the set-up above gets a task of its own, the first slice starts a fresh one
    await yieldToBrowser();
    this.foliage = await buildFoliage({
      clusters: this.clusters.map((c) => ({ mesh: c.mesh, base: c.base, radius: c.radius })),
      farInner: this.farInner,
      farOuter: this.farOuter,
      farInnerId: FAR_INNER_ID,
      farOuterId: FAR_OUTER_ID,
      farInnerBase,
      farOuterBase,
      space: world,
      hidden: this.hidden,
      toLight: keyDir.clone().negate().normalize(),
      hollow: this.hollowFrom("canopyClose_main"),
      nearPath: ["canopyClose_main", "canopyClose_out"].flatMap((n) => {
        const pose = this.rig.pose(n);
        return pose ? [pose.position.clone()] : [];
      }),
      itemLength,
      itemArea,
      count: Math.round(budget * LEAVES_PER_BUDGET),
      // fewer leaves on lower qualities: a little larger, so the crown keeps its density
      sizeScale,
      densityScale: (HIGH_LEAF_BUDGET / budget) / (sizeScale * sizeScale),
      rng: (label) => ctx.rng(label),
      seed: ctx.seed,
    }, 8, ctx);
    const u = createLeafUniforms();
    this.leafU = u;
    for (const c of this.clusters) u.uClA.value[c.id].set(c.base.x, c.base.y, c.base.z, 1);
    u.uClA.value[FAR_INNER_ID].set(farInnerBase.x, farInnerBase.y, farInnerBase.z, 1);
    u.uClA.value[FAR_OUTER_ID].set(farOuterBase.x, farOuterBase.y, farOuterBase.z, 1);
    const leafMat = createLeafMaterial(kit.map, u, ctx.wind);
    LEAF_ITEMS.forEach((item, i) => {
      const n = this.foliage?.counts[i] ?? 0;
      if (!n || !this.foliage) return;
      const mesh = createLeafMesh(`leaves_${item}`, kitGeoms[i], this.foliage.data[i], n, leafMat);
      world.add(mesh);
      ctx.layers.assign(mesh, "leaves");
    });
    ctx.reportInstances("leaves", this.foliage.total);
    ctx.reportInstances("leaves.crown", this.foliage.crown);
    ctx.reportInstances("leaves.far", this.foliage.far);
    if (ctx.debug.debug) console.info("[canopy]", JSON.stringify(this.debugInfo()));
    // the last foliage slice ran in this task: the engine's compile / warm-up that follows
    // the build starts in a fresh one
    await yieldToBrowser();
  }

  /**
   * The dark middle of the close-up (frame 17: lit foliage left, right and below, a dark
   * hollow from the top edge to ≈ 0.85 of the height, ≈ 0.3–0.7 of the width), as an
   * ellipse in that camera's frame. Seen from canopy_main the same cone is a smaller,
   * dimmer gap (frame 16).
   */
  private hollowFrom(poseName: string): FoliageHollow | null {
    const pose = this.rig.pose(poseName);
    if (!pose) return null;
    return {
      position: pose.position.clone(),
      quaternion: pose.quaternion.clone(),
      fov: pose.fov,
      aspect: DESIGN_ASPECT,
      centre: [0.48, 0.42],
      radii: [0.2, 0.44],
      soft: 0.35,
      ragged: 0.9,
    };
  }

  /** Monotone cubic path through the five camera keys, sampled for the rig; focus alongside. */
  private buildTrack(focusOf: Map<string, number>): void {
    const poses = CAMERA_KEYS.map(([name]) => this.rig.pose(name));
    if (poses.some((p) => !p)) {
      const missing = CAMERA_KEYS.filter(([name]) => !this.rig.pose(name)).map(([n]) => `cam_${n}`);
      throw new Error(`canopy.glb: missing ${missing.join(", ")}`);
    }
    const times = CAMERA_KEYS.map(([, t]) => t);
    const values: number[][] = [];
    let prevQ: Quaternion | null = null;
    (poses as CameraPose[]).forEach((pose, i) => {
      let p = pose;
      const f0 = focusOf.get(CAMERA_KEYS[i][0]);
      let f = Number.isFinite(f0) && (f0 as number) > 0 ? (f0 as number) : pose.position.length();
      if (CAMERA_KEYS[i][0] === "canopy_in") {
        const back = new Vector3(0, 0, 1).applyQuaternion(pose.quaternion).multiplyScalar(IN_DOLLY_M);
        p = { position: pose.position.clone().add(back), quaternion: pose.quaternion.clone(), fov: pose.fov };
        f += IN_DOLLY_M;
        this.inPose = p;
      }
      const q = tmpQ.copy(p.quaternion);
      if (prevQ && prevQ.dot(q) < 0) q.set(-q.x, -q.y, -q.z, -q.w);
      prevQ = q.clone();
      values.push([p.position.x, p.position.y, p.position.z, q.x, q.y, q.z, q.w, p.fov, f]);
    });
    const tangents = monotoneTangents(times, values, true, true);
    this.focus = { times, values, tangents };
    const keys: PoseKey[] = [];
    const t0 = times[0];
    const t1 = times[times.length - 1];
    const steps = Math.max(1, Math.round((t1 - t0) * TRACK_HZ));
    for (let i = 0; i <= steps; i++) {
      const t = t0 + ((t1 - t0) * i) / steps;
      const v = hermite(times, values, tangents, t);
      const q = new Quaternion(v[3], v[4], v[5], v[6]).normalize();
      keys.push({ t, pose: { position: new Vector3(v[0], v[1], v[2]), quaternion: q, fov: v[7] }, ease: "linear" });
    }
    this.rig.setTrack(keys);
  }

  // ---------------------------------------------------------------------------
  // per frame (pure in local.t and frame.timeSec)
  // ---------------------------------------------------------------------------

  getLook(frame: FrameState, local: SceneLocal, base: LookParams): LookParams {
    const t = local.t;
    if (this.focus) {
      // the episode looks hold the DOF of the main poses; follow the focus along the path
      const f = hermite(this.focus.times, this.focus.values, this.focus.tangents, t)[8];
      const shift = f - base.dof.focusDistance;
      base.dof.focusDistance = f;
      base.dof.nearStart = Math.max(0.05, base.dof.nearStart + shift);
      base.dof.nearEnd = Math.max(base.dof.nearStart + 0.05, base.dof.nearEnd + shift);
    }
    // the leaves darken before they part (motion/04: 52.7 s already dimmer)
    if (!frame.reducedMotion) base.exposure *= exitDim(t);
    return base;
  }

  protected animate(frame: FrameState, local: SceneLocal): void {
    const t = local.t;
    const still = frame.reducedMotion;
    if (this.placeholder) {
      const s = still ? 1 : growScale(globalReveal(t));
      this.placeholder.scale.setScalar(Math.max(1e-3, s));
      return;
    }
    const u = this.leafU;
    const e = still ? 0 : exitProgress(t);
    const hole = still ? 0 : holeRadius(t);
    // twigs retract into their bases while the scene darkens; scaffold and shells fade
    const twigs = 1 - smoothstep(0.05, 0.45, e);
    const reveal = still ? 1 : globalReveal(t);
    // the whole crown grows about its bottom centre
    const S = still ? 1 : crownScale(t);
    const world = this.world;
    if (world) {
      world.visible = S > 1e-3;
      world.scale.setScalar(Math.max(1e-3, S));
      world.position.copy(this.pivot).multiplyScalar(1 - S);
    }
    this.state.reveal = reveal;
    this.state.scale = S;
    this.state.exit = e;
    this.state.hole = hole;
    this.state.twigs = twigs;

    for (const c of this.clusters) {
      const p = still ? 1 : clusterGrow(t, c.order);
      const s = growScale(p);
      const twigScale = s * twigs;
      c.mesh.visible = twigScale > 1e-3;
      c.mesh.scale.setScalar(Math.max(1e-3, twigScale));
      if (u) {
        u.uClA.value[c.id].w = s;
        u.uClB.value[c.id].w = p;
      }
    }

    // scaffold: grows out of the hidden centre a little behind the first clusters
    if (this.scaffold) {
      const s = growScale(saturate((reveal - 0.1) / 0.9));
      const sc = this.scaffold;
      sc.mesh.scale.setScalar(Math.max(1e-3, s));
      sc.mesh.position.copy(this.hidden).addScaledVector(tmpV.copy(sc.position).sub(this.hidden), s);
      sc.material.opacity = 1 - smoothstep(0.05, 0.3, e);
      sc.mesh.visible = s > 1e-3 && sc.material.opacity > 1e-3;
    }
    const far = still ? 1 : farGrow(t);
    // 0 at canopy_main … 1 at canopyClose_main: the crown's middle goes dark (frame 17)
    const closeK = smoothstep(CANOPY_T.out - 0.5, CANOPY_T.closeMain, t);
    if (this.backdrop) {
      const b = this.backdrop;
      b.material.opacity = smoothstep(0.35, 0.95, reveal) * (1 - smoothstep(0.02, 0.22, e));
      b.mesh.visible = b.material.opacity > 1e-3;
      b.material.color.copy(BACKDROP_FAR).lerp(BACKDROP_CLOSE, closeK);
    }

    if (u) {
      u.uClB.value[FAR_INNER_ID].w = smoothstep(0.15, 0.9, reveal);
      u.uClB.value[FAR_OUTER_ID].w = far;
      // a young crown has few leaves between a leaf and the key (the outer clusters come
      // last): its inside is lit almost like its surface, the full depth from ≈ 43.9 s
      u.uShadeK.value = 0.12 + 0.88 * smoothstep(0.25, 1, reveal);
      // the hollow reads as a shaded middle from afar and goes dark in the close-up
      u.uHollow.value = 0.3 + 0.7 * closeK;
      const cam = this.camera;
      u.uExit.value.set(hole, Math.tan((cam.fov * Math.PI) / 360) * Math.hypot(cam.aspect, 1), 0.35, 0);
      // camera in crown space (world = pivot·(1 − S) + S·local; no rotation)
      if (world) u.uExitCam.value.copy(cam.position).sub(world.position).divideScalar(Math.max(1e-3, S));
      else u.uExitCam.value.copy(cam.position);
      u.uExitF.value.set(0, 0, -1).applyQuaternion(cam.quaternion);
      u.uExitRight.value.set(1, 0, 0).applyQuaternion(cam.quaternion);
      u.uExitUp.value.set(0, 1, 0).applyQuaternion(cam.quaternion);
      u.uExitLobe.value.w = this.lobePhase;
    }
  }

  debugInfo(): Record<string, string | number | boolean> {
    if (this.placeholder) return { canopy: "PLACEHOLDER (canopy.glb missing)" };
    const f = this.foliage;
    const out: Record<string, string | number | boolean> = {
      reveal: `${this.state.reveal.toFixed(3)} scale ${this.state.scale.toFixed(3)}`,
      exit: `${this.state.exit.toFixed(3)} hole ${this.state.hole.toFixed(3)} twigs ${this.state.twigs.toFixed(2)}`,
    };
    if (f) {
      const q = (v: { p10: number; p50: number; p90: number; mean: number }) => `p10 ${v.p10.toFixed(3)} p50 ${v.p50.toFixed(3)} p90 ${v.p90.toFixed(3)} mean ${v.mean.toFixed(3)}`;
      out.leaves = `${f.total} (crown ${f.crown} of ${f.candidates} candidates in ${f.sprays} sprays, far ${f.far}; a/b/c ${f.counts.join("/")})`;
      out["key transmittance"] = q(f.keyT);
      out["view visibility"] = q(f.visibility);
      out["hollow membership"] = q(f.hollow);
      out["foliage build ms (wall, sliced)"] = `${Math.round(f.ms)} (${f.phases})`;
    }
    return out;
  }

  // ---------------------------------------------------------------------------
  // PLACEHOLDER fallback (canopy.glb missing)
  // ---------------------------------------------------------------------------

  private buildPlaceholder(ctx: SceneContext): void {
    const material = new MeshStandardMaterial({ name: "PLACEHOLDER_canopy", color: "#4E6E2C", roughness: 0.8, flatShading: true });
    const mesh = new Mesh(new IcosahedronGeometry(1.2, 2), material);
    mesh.name = "PLACEHOLDER_canopy";
    mesh.position.set(0, 0, 0);
    this.scene.add(mesh);
    ctx.layers.assign(mesh, "placeholder");
    this.placeholder = mesh;
    this.lights.setKeyDirection(new Vector3(0.26, -0.81, -0.52));
    this.rig.setTrack([
      { t: CANOPY_T.in, pose: CameraRig.lookAtPose([0, 0.1, 5.5], [0, 0.1, 0]) },
      { t: CANOPY_T.out, pose: CameraRig.lookAtPose([0, 0.1, 4.5], [0, 0.1, 0]) },
      { t: CANOPY_T.closeOut, pose: CameraRig.lookAtPose([0, 0.2, 2.8], [0, 0.1, 0]) },
    ]);
  }
}

/** One-sided area of a geometry (m²). */
function bladeArea(g: BufferGeometry): number {
  const pos = g.getAttribute("position");
  const index = g.index;
  const n = index ? index.count : pos.count;
  const a = new Vector3();
  const b = new Vector3();
  const c = new Vector3();
  let area = 0;
  for (let i = 0; i + 2 < n; i += 3) {
    const i0 = index ? index.getX(i) : i;
    const i1 = index ? index.getX(i + 1) : i + 1;
    const i2 = index ? index.getX(i + 2) : i + 2;
    a.fromBufferAttribute(pos, i0);
    b.fromBufferAttribute(pos, i1).sub(a);
    c.fromBufferAttribute(pos, i2).sub(a);
    area += b.cross(c).length() * 0.5;
  }
  return area > 0 ? area : 1.9e-4;
}
