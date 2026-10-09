/**
 * Instanced grass blades (layer B of the vegetation).
 *
 * One shared ribbon (root at y = 0, tip at y = 1, x across the blade, `segments` rows
 * + a tip vertex) drawn with per-instance attributes. The vertex shader builds each
 * blade as a circular arc in the plant's frame (growth axis `aUp`, lean `aFwd`):
 * initial tilt θ0 plus a bend Δθ along the length, an optional twist of the width
 * axis, then the shared wind bends it about its fixed root (h² weight, length kept).
 * Normals are rounded across the blade; colour runs from a dark root to a light
 * (yellow-green) tip, with dry tips on a few blades; AO comes from the base surface
 * (COLOR_0.B at the root) and from the height (roots darker).
 *
 * Instances are split into a few spatial chunks (one draw call each) so frustum
 * culling works; chunk bounds are grown by the longest blade plus the wind reach.
 * Blades receive shadows (unless `receiveShadow: false`) and do not cast them: casting
 * would draw every blade a second time into the key's shadow map.
 *
 * Opt-in pile shading (PlantShade.ts): `surfaceShade`, `clumpAo`, `baseAo: { power, open }`,
 * `translucencyFloor`; `comb` lays hair-like blades towards a direction.
 *
 * The instance writing also runs as pausable steps (`GrassBatch.steps`, `grassInstancesSteps`,
 * `chunkMeshesSteps`; slices.ts) with exactly the results of the plain calls.
 */
import {
  BufferAttribute,
  BufferGeometry,
  Color,
  DoubleSide,
  Group,
  InstancedBufferGeometry,
  InstancedInterleavedBuffer,
  InterleavedBufferAttribute,
  Mesh,
  MeshStandardMaterial,
  Sphere,
  Vector3,
  Box3,
} from "three";
import type { Rng } from "../core/rng";
import type { SceneContext } from "../types";
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
import { spatialChunksSteps, type SurfaceSamples, type ViewImportanceOptions } from "./SurfaceScatter";
import { maxWindFraction } from "./WindField";
import { fbm3, hash3i } from "./noise";

export interface GrassColors {
  /** sRGB hex. Root → mid → tip gradient; tip mixes between tip and tip2 per blade. */
  root: string;
  mid: string;
  tip: string;
  tip2: string;
  /** Dry (straw) colour for a few tips. */
  dry: string;
}

export interface GrassLayerSpec {
  kind: "grass";
  /** Stats key (`ctx.reportInstances`). */
  name: string;
  /** Share of `quality.grassBlades` for this layer (before the recipe's budget scale); required unless `count` is set. */
  budget?: number;
  /**
   * Fixed number of blades, independent of the quality preset (still × the recipe's
   * `densityScale` and the build's `budgetScale`); overrides `budget`.
   */
  count?: number;
  /** Debug layer (default "grass"). */
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
   * Patchy growth: density × smoothstep(threshold ± 0.5·softness) of a value-noise field
   * with `scale` cells per metre (e.g. leafy patches between grassy ones).
   */
  patches?: { scale: number; threshold: number; softness?: number };
  /** Blade length range (m) before the COLOR_0.G length mask. */
  length: [number, number];
  /** Bias of the length distribution: > 1 favours short blades (default 1.3). */
  lengthSkew?: number;
  /** Influence 0–1 of COLOR_0.G (length mask) on the length (default 0.8). */
  lengthMask?: number;
  /** Blade width range at the root (m). */
  width: [number, number];
  /** Rows along the blade (default 4; the tip vertex is extra). */
  segments?: number;
  /** Initial tilt away from the growth axis (rad). */
  tilt: [number, number];
  /** Additional bend accumulated along the blade (rad, total). */
  bend: [number, number];
  /** Max twist of the blade around its length (rad, default 0.7). */
  twist?: number;
  /** 0–1: blades on steep faces / overhangs lean down the slope and droop more (default 0.5). */
  droop?: number;
  /** 0–1: random deviation of the growth axis from the surface normal (default 0.35). Ignored when `cone` is set. */
  normalJitter?: number;
  /**
   * Half-angle (rad) of the cone around the surface normal in which growth axes are
   * spread uniformly (e.g. 0.75 ≈ 43°): the blades of a mat point every which way.
   */
  cone?: number;
  /** Spatially coherent lean (neighbours lean alike): strength 0–1, frequency 1/m. */
  lean?: { strength: number; scale: number };
  /**
   * Tufts: the surface is split into Voronoi cells of about `size` metres around jittered
   * centres; a blade leans out of its tuft's centre (`strength` 0–1 of its azimuth),
   * turned by up to ±`swirl`·90° per tuft (whorls). Blades near a centre stay random.
   */
  tufts?: { size: number; strength: number; swirl?: number };
  /** Chance 0–1 that a blade bends back across its own lean (tangles instead of a comb). */
  bendFlip?: number;
  /**
   * How the bend grows with the blade length: dth × (L / mean length)^bendLength
   * (default 0: the bend range is per blade). Long blades curl over and lie on the mat.
   */
  bendLength?: number;
  /** Largest angle (rad) between the growth axis and the tip direction (default π). */
  maxAngle?: number;
  /** Width at the tip as a fraction of the root width (default 0.1). */
  tipWidth?: number;
  /**
   * Extra-long blades that break the silhouette. `rimBoost` multiplies their chance on
   * silhouette faces (1 + rimBoost · rim); `pale` is the chance that one is pale / dry.
   */
  outliers?: { fraction: number; scale: number; rimBoost?: number; pale?: number };
  /**
   * Height (m) above the surface over which the grass AO goes from `rootAo` to open
   * (the depth of the mat, default 0.7 × the mean length). Short blades stay inside
   * the dark mat, only the tops of the mat and the long blades catch the light.
   */
  canopy?: number;
  colors: GrassColors;
  /** Fraction of blades with dry tips (default 0.06). */
  dryFraction?: number;
  /** ± brightness variation per blade (default 0.15). */
  brightnessJitter?: number;
  /** Albedo factor at the root (height AO, default 0.3). */
  rootAo?: number;
  /**
   * Base surface AO (COLOR_0.B at the root): strength 0–1 (default 0.85), or
   * `{ strength, power, open }` — min(1, AO / open) ^ power first (crevices darker, open
   * faces unchanged).
   */
  baseAo?: BaseAoOption;
  stiffness: [number, number];
  roughness?: number;
  /** Back-light strength (default 0.8). */
  translucency?: number;
  /**
   * Back-light glow on blades whose base faces the camera, as a fraction of the glow on the
   * outline (default 0.12). Lower keeps a back light from brightening the whole pile.
   */
  translucencyFloor?: number;
  /**
   * Form shading from the base surface (default off, PlantShade.ts): direct lights scaled by
   * the terminator of the base normal, sky / ground fill bent towards it, outline spared.
   */
  surfaceShade?: SurfaceShadeSpec;
  /**
   * Occlusion inside clumps (default off): darker towards the clump centres (by default
   * the `tufts` cells) and the roots, darker gaps between clumps, per-clump variation;
   * `fine` adds a second clump scale.
   */
  clumpAo?: ClumpAoSpec;
  /**
   * Hair lay (default off): azimuths blended towards `dir` (parent space, projected onto
   * each blade's tangent plane) by `strength` 0–1, plus `lay` rad of extra tilt (× strength);
   * after `tufts` / `lean`, before `droop`. Where `dir` is along the growth axis, the hair parts.
   */
  comb?: { dir: [number, number, number]; strength: number; lay?: number };
  /** Receive the key light's shadow map (default true). */
  receiveShadow?: boolean;
  /** 0–1 rounding of the normal across the blade (default 0.55). */
  roundness?: number;
  /** Density weight = R^densityPower (COLOR_0.R, default 1.5). */
  densityPower?: number;
  /**
   * Weight by surface orientation: × smoothstep(min, max, normal·up) — e.g. `{ min: -0.4,
   * max: 0.1 }` keeps a layer off overhanging undersides. Default: no preference.
   */
  facing?: { min: number; max: number };
  /** Extra weight on silhouette faces of the recipe views: × (1 + rimBias · rim^rimPower), default 0. */
  rimBias?: number;
  /** Sharpness of the silhouette preference (default 1). */
  rimPower?: number;
  /** Width of the silhouette zone: |cos(normal, view ray)| below which a face counts as rim (default 0.55). */
  rimWidth?: number;
  /** Cell size (m) of the shared wind phase cluster (default 0.06). */
  clusterSize?: number;
  /** Max number of spatial chunks (draw calls), default 6. */
  chunks?: number;
}

const STRIDE = 20;

/** Canonical blade: x ∈ [−0.5, 0.5] across, y ∈ [0, 1] along. Winding faces the arc's outer side. */
export function createBladeGeometry(segments = 4): BufferGeometry {
  const rows = Math.max(1, Math.floor(segments));
  const pos: number[] = [];
  for (let i = 0; i < rows; i++) {
    const h = i / rows;
    pos.push(-0.5, h, 0, 0.5, h, 0);
  }
  pos.push(0, 1, 0);
  const tip = rows * 2;
  const idx: number[] = [];
  for (let i = 0; i < rows - 1; i++) {
    const l0 = i * 2;
    const r0 = l0 + 1;
    const l1 = l0 + 2;
    const r1 = l0 + 3;
    idx.push(l0, l1, r0, r0, l1, r1);
  }
  const lLast = (rows - 1) * 2;
  idx.push(lLast, tip, lLast + 1);
  const g = new BufferGeometry();
  g.setAttribute("position", new BufferAttribute(new Float32Array(pos), 3));
  g.setIndex(idx);
  g.name = `SilvaBlade${rows}`;
  return g;
}

const VERTEX_PARS = /* glsl */ `
attribute vec3 aRoot;
attribute vec3 aUp;
attribute vec3 aFwd;
attribute vec4 aShape;
attribute vec4 aVar;
attribute vec3 aVar2;
uniform vec3 uRootColor;
uniform vec3 uMidColor;
uniform vec3 uTipColor;
uniform vec3 uTip2Color;
uniform vec3 uDryColor;
uniform float uRootAo;
uniform float uBaseAo;
uniform float uRoundness;
uniform float uCanopy;
uniform float uTipWidth;
`;

const VERTEX_MAIN = /* glsl */ `
  float sH = position.y;
  float sX = position.x;
  vec3 gU = aUp;
  vec3 gF = aFwd;
  vec3 gS = cross(gU, gF);
  float gL = aShape.x;
  float th0 = aShape.z;
  float dth = aShape.w;
  float ang = th0 + dth * sH;
  float sa = sin(ang);
  float ca = cos(ang);
  vec3 gT = sa * gF + ca * gU;
  vec3 gC = abs(dth) > 1e-3
    ? (gL / dth) * (gF * (cos(th0) - ca) + gU * (sa - sin(th0)))
    : (gL * sH) * (sin(th0) * gF + cos(th0) * gU);
  vec3 gN = cross(gT, gS);
  float tw = aVar2.y * sH;
  vec3 gB = cos(tw) * gS + sin(tw) * gN;
  vec3 gNw = cos(tw) * gN - sin(tw) * gS;
  float wid = aShape.y * mix(1.0, uTipWidth, pow(sH, 2.0));
  vec3 rel = gC + gB * (sX * wid);
  vec3 sway = silvaWindSway(aRoot, aVar.y, aVar.x);
  float fl = silvaFlutter(aVar.y, 2.1) * uWindFlutter * uWindTipFraction * uWindStrength * 0.7 / max(aVar.x, 0.2);
  rel = silvaBend(rel, sH, gU, sway + gNw * fl, gL);
  vec3 silvaPos = aRoot + rel;
  vec3 silvaNormal = normalize(gNw + gB * (sX * 2.0 * uRoundness));
  vec3 silvaRootPos = aRoot;
  vec3 silvaRootUp = gU;

  vec3 tipC = mix(uTipColor, uTip2Color, aVar.z);
  vec3 gcol = mix(uRootColor, uMidColor, smoothstep(0.0, 0.45, sH));
  gcol = mix(gcol, tipC, smoothstep(0.3, 1.0, sH));
  gcol = mix(gcol, uDryColor, aVar2.x * smoothstep(0.2, 0.9, sH));
  gcol *= aVar2.z;
  // AO = depth inside the mat: height of the (unbent) point above the root along the
  // growth axis — or half the way travelled along the blade, for blades that curl over
  // the top of the mat — against the canopy depth, plus a little of the blade's own height
  float hAbs = max(dot(gC, gU), 0.5 * sH * gL);
  float hao = mix(uRootAo, 1.0, clamp(0.8 * smoothstep(0.0, uCanopy, hAbs) + 0.2 * sH, 0.0, 1.0));
  float bao = mix(1.0, aVar.w, uBaseAo);
  vSilvaColor = gcol * hao * bao;
  vSilvaTransl = smoothstep(0.15, 1.0, sH) * (1.0 - 0.5 * aVar2.x);
#ifdef SILVA_PLANT_SHADE
  float silvaShadeH = sH;
  float silvaShadeDepth = hAbs;
#endif
`;

/** Canopy depth of a layer (m): where the height AO opens. */
function canopyOf(spec: GrassLayerSpec): number {
  return Math.max(1e-3, spec.canopy ?? 0.35 * (spec.length[0] + spec.length[1]));
}

/** Create the patched blade material of a layer (`shade`: its plantShadeSetup, computed when omitted). */
export function createGrassMaterial(ctx: SceneContext, spec: GrassLayerSpec, shade: PlantShadeSetup = plantShadeSetup(spec)): MeshStandardMaterial {
  const c = spec.colors;
  const mat = new MeshStandardMaterial({
    name: `SilvaGrass:${spec.name}`,
    color: 0xffffff,
    roughness: spec.roughness ?? 0.62,
    metalness: 0,
    side: DoubleSide,
  });
  patchPlantMaterial(
    mat,
    {
      key: "grass",
      vertexPars: VERTEX_PARS,
      vertexMain: VERTEX_MAIN,
      translucency: { value: spec.translucency ?? 0.8 },
      uniforms: {
        uRootColor: { value: new Color(c.root) },
        uMidColor: { value: new Color(c.mid) },
        uTipColor: { value: new Color(c.tip) },
        uTip2Color: { value: new Color(c.tip2) },
        uDryColor: { value: new Color(c.dry) },
        uRootAo: { value: spec.rootAo ?? 0.3 },
        uBaseAo: { value: baseAoStrength(spec.baseAo, 0.85) },
        uRoundness: { value: spec.roundness ?? 0.55 },
        uCanopy: { value: canopyOf(spec) },
        uTipWidth: { value: spec.tipWidth ?? 0.1 },
      },
      shade,
    },
    ctx.wind,
  );
  return mat;
}

const vUp = new Vector3();
const vN = new Vector3();
const vF = new Vector3();
const vD = new Vector3();
const vR = new Vector3();
const vL = new Vector3();
const vT = new Vector3();
const vB = new Vector3();
const vC = new Vector3();
const WORLD_DOWN = new Vector3(0, -1, 0);

function randomUnit(rng: Rng, out: Vector3): Vector3 {
  const z = rng.range(-1, 1);
  const a = rng.range(0, Math.PI * 2);
  const r = Math.sqrt(1 - z * z);
  return out.set(r * Math.cos(a), r * Math.sin(a), z);
}

/** Any unit vector perpendicular to n. */
function perpendicular(n: Vector3, out: Vector3): Vector3 {
  if (Math.abs(n.y) < 0.9) out.set(0, 1, 0);
  else out.set(1, 0, 0);
  return out.sub(vR.copy(n).multiplyScalar(out.dot(n))).normalize();
}

function smoothstep01(x: number): number {
  const c = Math.min(1, Math.max(0, x));
  return c * c * (3 - 2 * c);
}

/**
 * Centre of the tuft (Voronoi cell of jittered lattice points `size` apart) that holds
 * p, written to `out`; returns the tuft's swirl value −1..1.
 */
function tuftCentre(px: number, py: number, pz: number, size: number, seed: number, out: Vector3): number {
  const gx = Math.floor(px / size);
  const gy = Math.floor(py / size);
  const gz = Math.floor(pz / size);
  let best = Infinity;
  let swirl = 0;
  for (let dz = -1; dz <= 1; dz++) {
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const cx = gx + dx;
        const cy = gy + dy;
        const cz = gz + dz;
        const x = (cx + 0.1 + 0.8 * hash3i(cx, cy, cz, seed + 101)) * size;
        const y = (cy + 0.1 + 0.8 * hash3i(cx, cy, cz, seed + 202)) * size;
        const z = (cz + 0.1 + 0.8 * hash3i(cx, cy, cz, seed + 303)) * size;
        const d = (x - px) * (x - px) + (y - py) * (y - py) + (z - pz) * (z - pz);
        if (d < best) {
          best = d;
          out.set(x, y, z);
          swirl = hash3i(cx, cy, cz, seed + 404) * 2 - 1;
        }
      }
    }
  }
  return swirl;
}

/**
 * Turn surface samples into packed blade instances (`stride` floats each: STRIDE, plus the
 * opt-in shade data of `plantShadeSetup(spec)` — base normal, clump values — behind it).
 * `rim` (optional, per sample 0–1): how much the sample's face is seen edge-on.
 * Shade options and `comb` draw no random numbers: the blades of a layer stay the same
 * whether it opts in or not (`comb` only turns them).
 */
export function buildGrassInstances(
  samples: SurfaceSamples,
  spec: GrassLayerSpec,
  rng: Rng,
  seed: number,
  rim: Float32Array | null = null,
  shade: PlantShadeSetup = plantShadeSetup(spec),
): GrassInstances {
  return runSync(grassInstancesSteps(samples, spec, rng, seed, rim, shade));
}

/** Packed blade instances (`stride` floats each) and the longest blade. */
export interface GrassInstances {
  data: Float32Array;
  maxLength: number;
  stride: number;
}

/** `buildGrassInstances` as pausable steps (slices.ts; checkpoint every 256 blades). */
export function* grassInstancesSteps(
  samples: SurfaceSamples,
  spec: GrassLayerSpec,
  rng: Rng,
  seed: number,
  rim: Float32Array | null = null,
  shade: PlantShadeSetup = plantShadeSetup(spec),
): Steps<GrassInstances> {
  const n = samples.count;
  const stride = STRIDE + shade.floats;
  const data = new Float32Array(n * stride);
  const clumpDefaults = { size: spec.tufts?.size ?? 0.04, depth: 1.3 * canopyOf(spec) };
  const clump = shade.clump && spec.clumpAo ? resolveClump(spec.clumpAo, clumpDefaults) : null;
  const fine = shade.clump2 && spec.clumpAo?.fine ? resolveClump(spec.clumpAo.fine, clumpDefaults) : null;
  const comb = spec.comb && spec.comb.strength > 0 ? spec.comb : null;
  const combDir = comb ? new Vector3(comb.dir[0], comb.dir[1], comb.dir[2]).normalize() : null;
  const [l0, l1] = spec.length;
  const [w0, w1] = spec.width;
  const skew = spec.lengthSkew ?? 1.3;
  const lengthMask = spec.lengthMask ?? 0.8;
  const droop = spec.droop ?? 0.5;
  const jitter = spec.normalJitter ?? 0.35;
  const cosCone = spec.cone !== undefined ? Math.cos(Math.min(Math.PI / 2, Math.max(0, spec.cone))) : null;
  const lean = spec.lean ?? { strength: 0.45, scale: 6 };
  const tufts = spec.tufts ?? null;
  const bendFlip = spec.bendFlip ?? 0;
  const bendLength = spec.bendLength ?? 0;
  const maxAngle = spec.maxAngle ?? Math.PI;
  const outliers = spec.outliers ?? { fraction: 0.04, scale: 1.45 };
  const rimBoost = outliers.rimBoost ?? 0;
  const outlierPale = outliers.pale ?? 0;
  const dryFraction = spec.dryFraction ?? 0.06;
  const bj = spec.brightnessJitter ?? 0.15;
  const twistMax = spec.twist ?? 0.7;
  const cell = spec.clusterSize ?? 0.06;
  const meanL = (l0 + l1) / 2;
  let maxLength = 0;

  for (let i = 0; i < n; i++) {
    const px = samples.position[i * 3];
    const py = samples.position[i * 3 + 1];
    const pz = samples.position[i * 3 + 2];
    vN.set(samples.normal[i * 3], samples.normal[i * 3 + 1], samples.normal[i * 3 + 2]);
    const gLen = samples.color[i * 4 + 1];
    const ao = samples.color[i * 4 + 2];
    const rimness = rim ? rim[i] : 0;

    // tangent frame of the surface
    perpendicular(vN, vT);
    vB.crossVectors(vN, vT);

    // growth axis: uniform in a cone around the normal (or the normal randomly nudged)
    if (cosCone !== null) {
      const cz = 1 - rng.next() * (1 - cosCone);
      const sz = Math.sqrt(Math.max(0, 1 - cz * cz));
      const phi = rng.range(0, Math.PI * 2);
      vUp.copy(vN).multiplyScalar(cz).addScaledVector(vT, sz * Math.cos(phi)).addScaledVector(vB, sz * Math.sin(phi)).normalize();
    } else {
      vUp.copy(vN).addScaledVector(randomUnit(rng, vR), jitter).normalize();
    }
    // how much the face hangs: 0 on top, 0.5 vertical, 1 upside down
    const hang = Math.min(1, Math.max(0, (1 - vUp.y) / 2));
    // down-slope direction in the plant's tangent plane
    vD.copy(WORLD_DOWN).addScaledVector(vUp, -WORLD_DOWN.dot(vUp));
    const dLen = vD.length();
    if (dLen > 1e-3) vD.multiplyScalar(1 / dLen);
    else perpendicular(vUp, vD);
    // random azimuth
    perpendicular(vUp, vF);
    const az = rng.range(0, Math.PI * 2);
    vL.crossVectors(vUp, vF);
    vF.multiplyScalar(Math.cos(az)).addScaledVector(vL, Math.sin(az));
    // tufts: lean out of the tuft centre, turned by the tuft's swirl
    if (tufts && tufts.strength > 0) {
      const sw = tuftCentre(px, py, pz, tufts.size, seed, vC) * (tufts.swirl ?? 0) * (Math.PI / 2);
      vC.set(px - vC.x, py - vC.y, pz - vC.z);
      vC.addScaledVector(vUp, -vC.dot(vUp));
      const dist = vC.length();
      if (dist > 1e-6) {
        vC.multiplyScalar(1 / dist);
        vL.crossVectors(vUp, vC);
        vC.multiplyScalar(Math.cos(sw)).addScaledVector(vL, Math.sin(sw));
        const w = tufts.strength * smoothstep01((dist / tufts.size - 0.08) / 0.35);
        vF.multiplyScalar(1 - w).addScaledVector(vC, w);
      }
    }
    // coherent lean field
    if (lean.strength > 0) {
      const la = fbm3(px * lean.scale, py * lean.scale, pz * lean.scale, seed + 17) * Math.PI * 4;
      vL.set(Math.cos(la), 0.35 * Math.sin(la * 0.7), Math.sin(la));
      vL.addScaledVector(vUp, -vL.dot(vUp));
      if (vL.lengthSq() > 1e-6) vL.normalize();
      else vL.copy(vF);
      vF.multiplyScalar(1 - lean.strength).addScaledVector(vL, lean.strength);
    }
    // comb: lay the blade towards a fixed direction (its projection on the tangent plane)
    let combLay = 0;
    if (comb && combDir) {
      vL.copy(combDir).addScaledVector(vUp, -combDir.dot(vUp));
      const cl = vL.length();
      if (cl > 1e-4) {
        vL.multiplyScalar(1 / cl);
        vF.multiplyScalar(1 - comb.strength).addScaledVector(vL, comb.strength);
        combLay = (comb.lay ?? 0) * comb.strength * Math.min(1, cl * 2);
      }
    }
    const dW = droop * Math.min(1, hang * 2);
    vF.addScaledVector(vD, dW * 1.6);
    if (vF.lengthSq() < 1e-6) perpendicular(vUp, vF);
    vF.addScaledVector(vUp, -vF.dot(vUp)).normalize();

    // size
    let L = l0 + (l1 - l0) * Math.pow(rng.next(), skew);
    L *= 1 + lengthMask * (0.55 + 0.9 * gLen - 1);
    const outlier = rng.chance(Math.min(1, outliers.fraction * (1 + rimBoost * rimness)));
    if (outlier) L *= outliers.scale * rng.range(0.85, 1.15);
    L = Math.max(0.003, L);
    maxLength = Math.max(maxLength, L);
    const W = w0 + (w1 - w0) * rng.next();
    let th0 = Math.min(maxAngle, spec.tilt[0] + (spec.tilt[1] - spec.tilt[0]) * rng.next() + dW * 0.35);
    if (combLay !== 0) th0 = Math.min(maxAngle, th0 + combLay);
    let dth = (spec.bend[0] + (spec.bend[1] - spec.bend[0]) * rng.next()) * (bendLength > 0 ? Math.pow(L / meanL, bendLength) : 1);
    if (rng.chance(bendFlip)) dth = -dth;
    dth += dW * 0.6;
    // keep the tip within maxAngle of the growth axis (on either side)
    dth = Math.min(maxAngle - th0, Math.max(-maxAngle - th0, dth));
    const stiff = (spec.stiffness[0] + (spec.stiffness[1] - spec.stiffness[0]) * rng.next()) * Math.sqrt(meanL / L);

    // wind cluster
    const cx = Math.floor(px / cell);
    const cy = Math.floor(py / cell);
    const cz = Math.floor(pz / cell);
    const phase = hash3i(cx, cy, cz, seed) * 0.35 + rng.next() * 0.05;
    const colorVar = rng.next();
    let dry = rng.chance(dryFraction) ? rng.range(0.45, 1) : 0;
    if (outlier && rng.chance(outlierPale)) dry = Math.max(dry, rng.range(0.55, 1));
    const twist = rng.range(-twistMax, twistMax);
    const bright = 1 + rng.range(-bj, bj);

    const o = i * stride;
    const sink = W * 0.3;
    data[o] = px - vN.x * sink;
    data[o + 1] = py - vN.y * sink;
    data[o + 2] = pz - vN.z * sink;
    data[o + 3] = vUp.x;
    data[o + 4] = vUp.y;
    data[o + 5] = vUp.z;
    data[o + 6] = vF.x;
    data[o + 7] = vF.y;
    data[o + 8] = vF.z;
    data[o + 9] = L;
    data[o + 10] = W;
    data[o + 11] = th0;
    data[o + 12] = dth;
    data[o + 13] = stiff;
    data[o + 14] = phase;
    data[o + 15] = colorVar;
    data[o + 16] = shapeBaseAo(ao, spec.baseAo);
    data[o + 17] = dry;
    data[o + 18] = twist;
    data[o + 19] = bright;
    if (shade.floats) writeShadeData(data, o + STRIDE, shade, clump, seed, px, py, pz, vN.x, vN.y, vN.z, 0, fine);
    // checkpoint between blades: every blade starts from fresh scratch vectors
    if ((i & 255) === 255) yield;
  }
  return { data, maxLength, stride };
}

export interface ChunkedInstances {
  meshes: Mesh[];
  count: number;
}

/**
 * Build chunk meshes sharing `base` (index + position etc.) with interleaved instance
 * data. `attributes` maps attribute name → [offset, itemSize] inside `stride`.
 */
export function buildChunkMeshes(
  base: BufferGeometry,
  data: Float32Array,
  stride: number,
  attributes: Record<string, [number, number]>,
  material: MeshStandardMaterial,
  maxChunks: number,
  reach: number,
  name: string,
  receiveShadow = true,
): ChunkedInstances {
  return runSync(chunkMeshesSteps(base, data, stride, attributes, material, maxChunks, reach, name, receiveShadow));
}

/** `buildChunkMeshes` as pausable steps (slices.ts; checkpoints inside the sort, the copies and between chunks). */
export function* chunkMeshesSteps(
  base: BufferGeometry,
  data: Float32Array,
  stride: number,
  attributes: Record<string, [number, number]>,
  material: MeshStandardMaterial,
  maxChunks: number,
  reach: number,
  name: string,
  receiveShadow = true,
): Steps<ChunkedInstances> {
  const count = data.length / stride;
  if (count === 0) return { meshes: [], count: 0 };
  const roots = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    roots[i * 3] = data[i * stride];
    roots[i * 3 + 1] = data[i * stride + 1];
    roots[i * 3 + 2] = data[i * stride + 2];
  }
  const groups = yield* spatialChunksSteps(roots, count, maxChunks);
  const meshes: Mesh[] = [];
  const box = new Box3();
  const p = new Vector3();
  for (let ci = 0; ci < groups.length; ci++) {
    const idx = groups[ci];
    const arr = new Float32Array(idx.length * stride);
    box.makeEmpty();
    for (let j = 0; j < idx.length; j++) {
      const so = idx[j] * stride;
      const to = j * stride;
      for (let q = 0; q < stride; q++) arr[to + q] = data[so + q];
      box.expandByPoint(p.set(data[so], data[so + 1], data[so + 2]));
      if ((j & 8191) === 8191) yield;
    }
    const geo = new InstancedBufferGeometry();
    geo.name = `${name}#${ci}`;
    geo.index = base.index;
    for (const [key, attr] of Object.entries(base.attributes)) geo.setAttribute(key, attr);
    const buffer = new InstancedInterleavedBuffer(arr, stride, 1);
    for (const [key, [offset, size]] of Object.entries(attributes)) geo.setAttribute(key, new InterleavedBufferAttribute(buffer, size, offset));
    geo.instanceCount = idx.length;
    box.expandByScalar(reach);
    geo.boundingBox = box.clone();
    geo.boundingSphere = box.getBoundingSphere(new Sphere());
    const mesh = new Mesh(geo, material);
    mesh.name = `${name}#${ci}`;
    mesh.castShadow = false;
    mesh.receiveShadow = receiveShadow;
    mesh.matrixAutoUpdate = false;
    mesh.updateMatrix();
    meshes.push(mesh);
    yield;
  }
  return { meshes, count };
}

export const GRASS_ATTRIBUTES: Record<string, [number, number]> = {
  aRoot: [0, 3],
  aUp: [3, 3],
  aFwd: [6, 3],
  aShape: [9, 4],
  aVar: [13, 4],
  aVar2: [17, 3],
};

/** The parts of a GrassBatch (what `GrassBatch.steps` builds before constructing it). */
export interface GrassBatchParts {
  group: Group;
  material: MeshStandardMaterial;
  count: number;
  meshes: Mesh[];
  maxLength: number;
}

/** A built grass layer: a group of chunk meshes sharing one material and one blade. */
export class GrassBatch {
  readonly group: Group;
  readonly material: MeshStandardMaterial;
  readonly count: number;
  readonly meshes: Mesh[];
  readonly maxLength: number;

  /** `parts`: the result of `grassBatchSteps` (from `GrassBatch.steps`); built now when omitted. */
  constructor(ctx: SceneContext, spec: GrassLayerSpec, samples: SurfaceSamples, rng: Rng, seed: number, rim: Float32Array | null = null, parts?: GrassBatchParts) {
    const p = parts ?? runSync(grassBatchSteps(ctx, spec, samples, rng, seed, rim));
    this.group = p.group;
    this.material = p.material;
    this.count = p.count;
    this.meshes = p.meshes;
    this.maxLength = p.maxLength;
  }

  /** The constructor as pausable steps (slices.ts): `const batch = yield* GrassBatch.steps(...)`. */
  static *steps(ctx: SceneContext, spec: GrassLayerSpec, samples: SurfaceSamples, rng: Rng, seed: number, rim: Float32Array | null = null): Steps<GrassBatch> {
    const parts = yield* grassBatchSteps(ctx, spec, samples, rng, seed, rim);
    return new GrassBatch(ctx, spec, samples, rng, seed, rim, parts);
  }
}

/** Material, blade instances and chunk meshes of a grass layer, in the order the batch always made them. */
function* grassBatchSteps(ctx: SceneContext, spec: GrassLayerSpec, samples: SurfaceSamples, rng: Rng, seed: number, rim: Float32Array | null = null): Steps<GrassBatchParts> {
  const group = new Group();
  group.name = `SilvaGrass:${spec.name}`;
  const shade = plantShadeSetup(spec);
  const material = createGrassMaterial(ctx, spec, shade);
  const { data, maxLength, stride } = yield* grassInstancesSteps(samples, spec, rng, seed, rim, shade);
  const base = createBladeGeometry(spec.segments ?? 4);
  const reach = maxLength * (1 + maxWindFraction(ctx.wind, spec.stiffness[0]));
  const attributes = shade.floats ? { ...GRASS_ATTRIBUTES, ...plantShadeAttributes(shade, STRIDE) } : GRASS_ATTRIBUTES;
  const built = yield* chunkMeshesSteps(base, data, stride, attributes, material, spec.chunks ?? 6, reach, `grass:${spec.name}`, spec.receiveShadow ?? true);
  for (const m of built.meshes) group.add(m);
  ctx.layers.assign(group, spec.layer ?? "grass");
  return { group, material, count: built.count, meshes: built.meshes, maxLength };
}
