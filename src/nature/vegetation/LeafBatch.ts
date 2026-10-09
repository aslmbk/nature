/**
 * Instanced kit plants (layer C): leaves on petioles, clover, small flowers, sprigs,
 * seed heads — real geometry from `models/kit.glb` (origin at the root, +Y growth
 * axis, COLOR_0: R flex 0 root → 1 tip, G shade, B 1 on blades / 0 on stems), textured
 * with `textures/kit_basecolor.webp` (no alpha: silhouettes are geometry).
 *
 * Instances come in small clusters (2–5 around a point, usually of one species) with
 * a random yaw around a growth axis between the surface normal and world up, a
 * per-instance scale, lift and tint (some leaves pale, almost whitish; seed heads
 * dry beige). One draw call per kit item and spatial chunk. Wind: the shared sway
 * bends the plant about its root with the flex weight (R²-like via uWindRootPower),
 * blades also flutter along their normal. Leaves receive shadows (unless
 * `receiveShadow: false`), do not cast them.
 *
 * Opt-in pile shading (PlantShade.ts): `surfaceShade`, `clumpAo`, `baseAo: { power, open }`,
 * `translucencyFloor`.
 *
 * The build also runs as pausable steps (`LeafBatch.steps`; slices.ts) with exactly the
 * results of the constructor.
 */
import { BufferGeometry, Color, DoubleSide, Group, Mesh, MeshStandardMaterial, Vector3, type Object3D, type Texture } from "three";
import type { GLTF } from "three/examples/jsm/loaders/GLTFLoader.js";
import type { Rng } from "../core/rng";
import type { SceneContext } from "../types";
import { chunkMeshesSteps } from "./GrassBatch";
import { patchPlantMaterial } from "./PlantMaterial";
import {
  baseAoStrength,
  plantShadeAttributes,
  plantShadeSetup,
  resolveClump,
  shapeBaseAo,
  writeShadeData,
  type BaseAoOption,
  type ClumpAoSpec,
  type PlantShadeSetup,
  type SurfaceShadeSpec,
} from "./PlantShade";
import { runSync, type Steps } from "./slices";
import type { SurfaceSamples, ViewImportanceOptions } from "./SurfaceScatter";
import { maxWindFraction } from "./WindField";
import { hash3i } from "./noise";

export interface KitItemSpec {
  /** Node name in kit.glb (e.g. "leaf_round_a"). */
  name: string;
  /** Relative frequency inside the layer. */
  weight: number;
  /** Uniform scale range (1 = modelled size). */
  scale: [number, number];
  /** Lift of the root along the surface normal (m), so leaves stand above the grass. */
  lift?: [number, number];
  /** Colour multiplier (sRGB hex), default white. */
  tint?: string;
  /**
   * Chance and strength (0–1) of a pale, whitish-sage leaf. `upFacing` 0–1: how much
   * the chance is limited to surfaces facing up (leaves catching the sky on top of a
   * mound read pale, the ones on steep or overhanging faces stay green).
   */
  pale?: { chance: number; amount: [number, number]; upFacing?: number };
  /** Preference 0–1 for silhouette faces (seed heads along the rims). */
  rimBias?: number;
  /** Override the layer's up bias. */
  upBias?: number;
  /** Override the layer's `upright`. */
  upright?: number;
  /** Override the layer's `tiltRange`. */
  tiltRange?: [number, number];
}

export interface KitLayerSpec {
  kind: "kit";
  /** Stats key (`ctx.reportInstances`). */
  name: string;
  /** Debug layer (default "leaves"). */
  layer?: string;
  /** Only these of the recipe's target meshes (name prefix / exact / RegExp); default all. */
  targets?: (string | RegExp)[];
  /** Density multiplier per target for this layer (first match wins; default: the recipe's). */
  targetWeights?: [string | RegExp, number][];
  /** Scatter only where the views with these tags look (ScatterView.tag); default all views. */
  viewTags?: string[];
  /** View importance of this layer's scatter (`facingMin` / `facingFull`); default the build's `importance`. */
  importance?: ViewImportanceOptions;
  /**
   * Size multiplier per target (first match wins, default 1): a smaller species of the
   * same plants on one mass, e.g. the small leaves of the low hero mound.
   */
  targetScale?: [string | RegExp, number][];
  /**
   * Patchy growth: density × smoothstep(threshold ± 0.5·softness) of a value-noise field
   * with `scale` cells per metre (e.g. leafy patches between grassy ones).
   */
  patches?: { scale: number; threshold: number; softness?: number };
  /** Share of `quality.leaves` for this layer (before the recipe's budget scale); required unless `count` is set. */
  budget?: number;
  /**
   * Fixed number of plants, independent of the quality preset (still × the recipe's
   * `densityScale` and the build's `budgetScale`); overrides `budget`.
   */
  count?: number;
  /** Albedo saturation of the plants (1 = the atlas, 0 = grey; default 1). */
  saturation?: number;
  /** Bias of the per-item scale distribution: > 1 favours the small end (default 1). */
  scaleSkew?: number;
  items: KitItemSpec[];
  /** 0 = grow along the surface normal, 1 = straight up (default 0.35). */
  upBias?: number;
  /** Random tilt of the growth axis (default 0.35). */
  tiltJitter?: number;
  /**
   * 0–1 pull of the (jittered) growth axis towards world up (+Y of the scatter space) after
   * `upBias` / `tiltJitter` (default 0): 1 stands every plant straight up (cups, stalks).
   */
  upright?: number;
  /**
   * Tilt (rad) of the growth axis by an angle in [min, max] towards a random side, after
   * `upright`; replaces `tiltJitter` (e.g. `upright: 1, tiltRange: [0, 0.25]`).
   */
  tiltRange?: [number, number];
  /** Plants per cluster and cluster radius (m). Default 1–4 within 2.5 cm. */
  cluster?: { size: [number, number]; radius: number; sameSpecies?: number };
  /** Density weight = R^densityPower (default 1.5). */
  densityPower?: number;
  /** Weight by surface orientation: × smoothstep(min, max, normal·up); default none. */
  facing?: { min: number; max: number };
  /**
   * Placement on silhouette faces of the recipe views: weight × (1 + rimBias · rim^rimPower)
   * (default 0: no preference). Seed heads and buds that dot the outlines.
   */
  rimBias?: number;
  /** Sharpness of the silhouette preference (default 1). */
  rimPower?: number;
  /** Width of the silhouette zone: |cos(normal, view ray)| below which a face counts as rim (default 0.55). */
  rimWidth?: number;
  stiffness?: [number, number];
  /** Leaf flutter amplitude multiplier (default 1). */
  flutter?: number;
  translucency?: number;
  roughness?: number;
  /** Per-instance brightness range (default [0.85, 1.12]). */
  brightness?: [number, number];
  /** Base surface AO: strength 0–1 (default 0.7), or `{ strength, power, open }` (min(1, AO / open) ^ power first). */
  baseAo?: BaseAoOption;
  /** Albedo factor at the root of the stem (default 0.45). */
  rootAo?: number;
  /** Max spatial chunks per item (default 3). */
  chunks?: number;
  /** Back-light glow on plants whose base faces the camera, relative to the outline (default 0.12). */
  translucencyFloor?: number;
  /** Form shading from the base surface (default off; see GrassLayerSpec / PlantShade.ts). */
  surfaceShade?: SurfaceShadeSpec;
  /** Occlusion inside clumps of `size` (default 0.04 m; depth default 0.8 × the plant height); `fine`: a second scale. */
  clumpAo?: ClumpAoSpec;
  /** Receive the key light's shadow map (default true). */
  receiveShadow?: boolean;
}

const STRIDE = 17;

const KIT_ATTRIBUTES: Record<string, [number, number]> = {
  aRoot: [0, 3],
  aUp: [3, 3],
  aFwd: [6, 3],
  aMisc: [9, 4],
  // rgb: tint × brightness × base AO, a: pale amount
  aTint: [13, 4],
};

/** Whitish sage of pale leaves (sRGB), reached at pale amount 1. */
const PALE_LEAF = "#B9C6AC";

const VERTEX_PARS = /* glsl */ `
#if !defined(USE_COLOR) && !defined(USE_COLOR_ALPHA)
attribute vec4 color;
#endif
attribute vec3 aRoot;
attribute vec3 aUp;
attribute vec3 aFwd;
attribute vec4 aMisc;
attribute vec4 aTint;
uniform float uLeafFlutter;
uniform float uLeafRootAo;
uniform vec3 uLeafPale;
centroid varying vec4 vSilvaPale;
`;

const FRAGMENT_PARS = /* glsl */ `
centroid varying vec4 vSilvaPale;
uniform float uLeafSaturation;
`;

/**
 * Atlas albedo × tint, then towards the pale colour (keeping some of the texture's
 * detail), then the layer's saturation.
 */
const FRAGMENT_COLOR = /* glsl */ `
  {
    vec3 leafAlbedo = diffuseColor.rgb;
    float leafL = dot(leafAlbedo, vec3(0.2126, 0.7152, 0.0722));
    diffuseColor.rgb = mix(leafAlbedo * vSilvaColor, vSilvaPale.rgb * (0.6 + 1.25 * leafL), vSilvaPale.a);
    diffuseColor.rgb = mix(vec3(dot(diffuseColor.rgb, vec3(0.2126, 0.7152, 0.0722))), diffuseColor.rgb, uLeafSaturation);
  }
`;

const VERTEX_MAIN = /* glsl */ `
  vec3 kU = aUp;
  vec3 kF = aFwd;
  vec3 kZ = cross(kF, kU);
  mat3 kR = mat3(kF, kU, kZ);
  vec3 rel = kR * (position * aMisc.x);
  vec3 kN = normalize(kR * normal);
  float flex = color.r;
  vec3 sway = silvaWindSway(aRoot, aMisc.w, aMisc.z);
  float fl = silvaFlutter(aMisc.w, 1.6) * uWindFlutter * uWindStrength * uLeafFlutter / max(aMisc.z, 0.2);
  rel = silvaBend(rel, flex, kU, sway, aMisc.y);
  rel += kN * (fl * color.b * flex * aMisc.y * 0.6);
  vec3 silvaPos = aRoot + rel;
  vec3 silvaNormal = kN;
  vec3 silvaRootPos = aRoot;
  vec3 silvaRootUp = kU;
  float leafShade = color.g * mix(uLeafRootAo, 1.0, smoothstep(0.0, 0.55, flex));
  vSilvaColor = aTint.rgb * leafShade;
  vSilvaPale = vec4(uLeafPale * dot(aTint.rgb, vec3(0.2126, 0.7152, 0.0722)) * leafShade, aTint.a * smoothstep(0.2, 0.6, color.b + 0.2 * flex));
  vSilvaTransl = color.b * (1.0 - 0.5 * aTint.a);
#ifdef SILVA_PLANT_SHADE
  float silvaShadeH = flex;
  float silvaShadeDepth = flex * aMisc.y;
#endif
`;

export function createKitMaterial(ctx: SceneContext, spec: KitLayerSpec, map: Texture | null, shade: PlantShadeSetup = plantShadeSetup(spec)): MeshStandardMaterial {
  const mat = new MeshStandardMaterial({
    name: `SilvaKit:${spec.name}`,
    color: map ? 0xffffff : new Color("#7FA266"),
    map,
    roughness: spec.roughness ?? 0.8,
    metalness: 0,
    side: DoubleSide,
  });
  patchPlantMaterial(
    mat,
    {
      key: `kit|${map ? "map" : "flat"}`,
      vertexPars: VERTEX_PARS,
      vertexMain: VERTEX_MAIN,
      translucency: { value: spec.translucency ?? 0.75 },
      uniforms: {
        uLeafFlutter: { value: spec.flutter ?? 1 },
        uLeafRootAo: { value: spec.rootAo ?? 0.45 },
        uLeafPale: { value: new Color(PALE_LEAF) },
        uLeafSaturation: { value: spec.saturation ?? 1 },
      },
      fragmentPars: FRAGMENT_PARS,
      fragmentColor: FRAGMENT_COLOR,
      shade,
    },
    ctx.wind,
  );
  return mat;
}

/** Kit meshes by node name. */
export function kitMeshes(kit: GLTF | null): Map<string, Mesh> {
  const out = new Map<string, Mesh>();
  kit?.scene.traverse((o: Object3D) => {
    const m = o as Mesh;
    if (m.isMesh) out.set(m.name, m);
  });
  return out;
}

const vUp = new Vector3();
const vN = new Vector3();
const vF = new Vector3();
const vT = new Vector3();
const vB = new Vector3();
const vR = new Vector3();
const vT2 = new Vector3();
const vB2 = new Vector3();
const WORLD_UP = new Vector3(0, 1, 0);

function perpendicular(n: Vector3, out: Vector3): Vector3 {
  if (Math.abs(n.y) < 0.9) out.set(0, 1, 0);
  else out.set(1, 0, 0);
  return out.sub(vR.copy(n).multiplyScalar(out.dot(n))).normalize();
}

export interface KitBuildInput {
  /** Cluster centres (already weighted by the recipe). */
  centres: SurfaceSamples;
  /** Per-centre rim factor 0–1 (silhouette), or null. */
  rim: Float32Array | null;
  /** Per-centre size multiplier (layer `targetScale`), or null. */
  scale?: Float32Array | null;
  /** Target number of plants (clusters × members ≈ this). */
  count: number;
}

/** The parts of a LeafBatch (what `LeafBatch.steps` builds before constructing it). */
export interface LeafBatchParts {
  group: Group;
  material: MeshStandardMaterial;
  count: number;
  perItem: Record<string, number>;
  meshes: Mesh[];
}

/** A built kit layer: per item, a few chunk meshes sharing one material. */
export class LeafBatch {
  readonly group: Group;
  readonly material: MeshStandardMaterial;
  readonly count: number;
  readonly perItem: Record<string, number>;
  readonly meshes: Mesh[];

  /** `parts`: the result of `leafBatchSteps` (from `LeafBatch.steps`); built now when omitted. */
  constructor(ctx: SceneContext, spec: KitLayerSpec, kit: Map<string, Mesh>, map: Texture | null, input: KitBuildInput, rng: Rng, seed: number, parts?: LeafBatchParts) {
    const p = parts ?? runSync(leafBatchSteps(ctx, spec, kit, map, input, rng, seed));
    this.group = p.group;
    this.material = p.material;
    this.count = p.count;
    this.perItem = p.perItem;
    this.meshes = p.meshes;
  }

  /** The constructor as pausable steps (slices.ts): `const batch = yield* LeafBatch.steps(...)`. */
  static *steps(ctx: SceneContext, spec: KitLayerSpec, kit: Map<string, Mesh>, map: Texture | null, input: KitBuildInput, rng: Rng, seed: number): Steps<LeafBatch> {
    const parts = yield* leafBatchSteps(ctx, spec, kit, map, input, rng, seed);
    return new LeafBatch(ctx, spec, kit, map, input, rng, seed, parts);
  }
}

/** Material, plants and chunk meshes of a kit layer, in the order the batch always made them. */
function* leafBatchSteps(ctx: SceneContext, spec: KitLayerSpec, kit: Map<string, Mesh>, map: Texture | null, input: KitBuildInput, rng: Rng, seed: number): Steps<LeafBatchParts> {
  const group = new Group();
  const perItemCount: Record<string, number> = {};
  const meshes: Mesh[] = [];
  group.name = `SilvaKit:${spec.name}`;
  const shade = plantShadeSetup(spec);
  const material = createKitMaterial(ctx, spec, map, shade);
  const stride = STRIDE + shade.floats;
  const items = spec.items.filter((it) => kit.has(it.name) && it.weight > 0);
  const totalWeight = items.reduce((s, it) => s + it.weight, 0);
  const perItem: number[][] = items.map(() => []);
  const heights = items.map((it) => {
    const g = (kit.get(it.name) as Mesh).geometry;
    g.computeBoundingBox();
    const bb = g.boundingBox;
    return bb ? Math.max(bb.max.y, 0.005) : 0.05;
  });
  const tints = items.map((it) => new Color(it.tint ?? "#ffffff"));
  const cl = spec.cluster ?? { size: [1, 4], radius: 0.025, sameSpecies: 0.75 };
  const sameSpecies = cl.sameSpecies ?? 0.75;
  const upBiasLayer = spec.upBias ?? 0.35;
  const tiltJitter = spec.tiltJitter ?? 0.35;
  const uprightLayer = spec.upright ?? 0;
  const tiltRangeLayer = spec.tiltRange;
  const [s0, s1] = spec.stiffness ?? [0.8, 1.3];
  const [b0, b1] = spec.brightness ?? [0.85, 1.12];
  const baseAo = baseAoStrength(spec.baseAo, 0.7);
  const scaleSkew = spec.scaleSkew ?? 1;
  // clump AO: default depth 0.8 × the mean modelled height of the layer's items
  const meanHeight = items.length ? items.reduce((s, it, k) => s + heights[k] * 0.5 * (it.scale[0] + it.scale[1]), 0) / items.length : 0.05;
  const clumpDefaults = { size: 0.04, depth: 0.8 * meanHeight };
  const clump = shade.clump && spec.clumpAo ? resolveClump(spec.clumpAo, clumpDefaults) : null;
  const fine = shade.clump2 && spec.clumpAo?.fine ? resolveClump(spec.clumpAo.fine, clumpDefaults) : null;
  const extra = new Float32Array(Math.max(1, shade.floats));
  const centreScale = input.scale ?? null;
  let maxCentreScale = 1;
  if (centreScale) for (let i = 0; i < centreScale.length; i++) maxCentreScale = Math.max(maxCentreScale, centreScale[i]);

  const pickItem = (r: number, rimness: number): number => {
    // rim-loving items get more weight on silhouettes, less inside
    let total = 0;
    const w = items.map((it) => {
      const rb = it.rimBias ?? 0;
      const v = it.weight * (1 - rb + rb * 2.2 * rimness);
      total += v;
      return v;
    });
    let x = r * total;
    for (let i = 0; i < w.length; i++) {
      x -= w[i];
      if (x <= 0) return i;
    }
    return w.length - 1;
  };

  const c = input.centres;
  let produced = 0;
  const target = input.count;
  // The samples come sorted by surface (stratified CDF order) and the loop stops at
  // the target count: visit them in a shuffled order so the cut is spread evenly
  // instead of dropping the last target meshes.
  const order = new Uint32Array(c.count);
  for (let i = 0; i < c.count; i++) order[i] = i;
  for (let i = c.count - 1; i > 0; i--) {
    const j = Math.floor(rng.next() * (i + 1));
    const tmp = order[i];
    order[i] = order[j];
    order[j] = tmp;
  }
  for (let oi = 0; oi < c.count && produced < target && totalWeight > 0; oi++) {
    const i = order[oi];
    const px = c.position[i * 3];
    const py = c.position[i * 3 + 1];
    const pz = c.position[i * 3 + 2];
    vN.set(c.normal[i * 3], c.normal[i * 3 + 1], c.normal[i * 3 + 2]);
    const ao = c.color[i * 4 + 2];
    const rimness = input.rim ? input.rim[i] : 0;
    const members = Math.max(1, Math.round(cl.size[0] + (cl.size[1] - cl.size[0]) * rng.next()));
    const species = pickItem(rng.next(), rimness);
    const phaseBase = hash3i(Math.floor(px / 0.08), Math.floor(py / 0.08), Math.floor(pz / 0.08), seed) * 0.35;
    // tangent frame of the surface for the cluster offsets
    perpendicular(vN, vT);
    vB.crossVectors(vN, vT);
    for (let m = 0; m < members && produced < target; m++) {
      const k = rng.chance(sameSpecies) ? species : pickItem(rng.next(), rimness);
      const it = items[k];
      const ang = rng.range(0, Math.PI * 2);
      const rad = m === 0 ? 0 : cl.radius * Math.sqrt(rng.next());
      const lift = it.lift ? rng.range(it.lift[0], it.lift[1]) : 0;
      const rx = px + (vT.x * Math.cos(ang) + vB.x * Math.sin(ang)) * rad + vN.x * lift;
      const ry = py + (vT.y * Math.cos(ang) + vB.y * Math.sin(ang)) * rad + vN.y * lift;
      const rz = pz + (vT.z * Math.cos(ang) + vB.z * Math.sin(ang)) * rad + vN.z * lift;
      // growth axis between the normal and world up, randomly tilted
      const ub = it.upBias ?? upBiasLayer;
      vUp.copy(vN).multiplyScalar(1 - ub).addScaledVector(WORLD_UP, ub);
      // the same three draws with or without `upright` / `tiltRange` (placement unchanged)
      const j1 = rng.range(-1, 1);
      const j2 = rng.range(-1, 1);
      const j3 = rng.range(-1, 1);
      const tiltRange = it.tiltRange ?? tiltRangeLayer;
      if (!tiltRange) {
        vUp.x += j1 * tiltJitter;
        vUp.y += j2 * tiltJitter * 0.5;
        vUp.z += j3 * tiltJitter;
      }
      if (vUp.lengthSq() < 1e-6) vUp.copy(vN);
      vUp.normalize();
      const upright = it.upright ?? uprightLayer;
      if (upright > 0) {
        vUp.multiplyScalar(1 - upright).addScaledVector(WORLD_UP, upright);
        if (vUp.lengthSq() < 1e-6) vUp.copy(WORLD_UP);
        vUp.normalize();
      }
      if (tiltRange) {
        // tilt by an angle in [min, max] towards a random side
        const th = tiltRange[0] + (tiltRange[1] - tiltRange[0]) * (0.5 + 0.5 * j1);
        const az = Math.PI * j3;
        perpendicular(vUp, vT2);
        vB2.crossVectors(vUp, vT2);
        const st = Math.sin(th);
        vUp.multiplyScalar(Math.cos(th)).addScaledVector(vT2, Math.cos(az) * st).addScaledVector(vB2, Math.sin(az) * st).normalize();
      }
      perpendicular(vUp, vF);
      const yaw = rng.range(0, Math.PI * 2);
      vR.crossVectors(vUp, vF);
      vF.multiplyScalar(Math.cos(yaw)).addScaledVector(vR, Math.sin(yaw)).normalize();
      const scale = (it.scale[0] + (it.scale[1] - it.scale[0]) * Math.pow(rng.next(), scaleSkew)) * (centreScale ? centreScale[i] : 1);
      const stiff = rng.range(s0, s1);
      const phase = phaseBase + rng.next() * 0.08;
      const bright = rng.range(b0, b1) * (1 - baseAo + baseAo * shapeBaseAo(ao, spec.baseAo));
      const t = tints[k];
      let paleChance = it.pale ? it.pale.chance : 0;
      if (it.pale?.upFacing) {
        const up = Math.min(1, Math.max(0, (vN.y + 0.1) / 0.8));
        paleChance *= 1 - it.pale.upFacing + it.pale.upFacing * up * up * (3 - 2 * up);
      }
      const paleAmount = it.pale && rng.chance(paleChance) ? rng.range(it.pale.amount[0], it.pale.amount[1]) : 0;
      perItem[k].push(rx, ry, rz, vUp.x, vUp.y, vUp.z, vF.x, vF.y, vF.z, scale, heights[k] * scale, stiff, phase, t.r * bright, t.g * bright, t.b * bright, paleAmount);
      if (shade.floats) {
        // clump cells are looked up on the surface (below the lifted root)
        writeShadeData(extra, 0, shade, clump, seed, rx - vN.x * lift, ry - vN.y * lift, rz - vN.z * lift, vN.x, vN.y, vN.z, lift, fine);
        for (let q = 0; q < shade.floats; q++) perItem[k].push(extra[q]);
      }
      produced++;
    }
    if ((oi & 63) === 63) yield;
  }

  const windFrac = maxWindFraction(ctx.wind, s0, spec.flutter ?? 1);
  const attributes = shade.floats ? { ...KIT_ATTRIBUTES, ...plantShadeAttributes(shade, STRIDE) } : KIT_ATTRIBUTES;
  for (let k = 0; k < items.length; k++) {
    const it = items[k];
    const list = perItem[k];
    if (!list.length) continue;
    const data = Float32Array.from(list);
    const src = (kit.get(it.name) as Mesh).geometry;
    const base = new BufferGeometry();
    for (const key of ["position", "normal", "uv", "color"]) {
      const a = src.getAttribute(key);
      if (a) base.setAttribute(key, a.clone());
    }
    if (src.index) base.setIndex(src.index.clone());
    const maxScale = it.scale[1] * maxCentreScale;
    const reach = heights[k] * maxScale * (1 + windFrac) + (it.lift ? it.lift[1] : 0);
    const built = yield* chunkMeshesSteps(base, data, stride, attributes, material, spec.chunks ?? 3, reach, `kit:${spec.name}:${it.name}`, spec.receiveShadow ?? true);
    for (const mesh of built.meshes) {
      meshes.push(mesh);
      group.add(mesh);
    }
    perItemCount[it.name] = built.count;
  }
  ctx.layers.assign(group, spec.layer ?? "leaves");
  return { group, material, count: produced, perItem: perItemCount, meshes };
}
