/**
 * Optional pile shading for instanced plants — per-layer options of `GrassLayerSpec` /
 * `KitLayerSpec`, all off by default: a layer that opts into none of them compiles to
 * exactly the shader (and packs exactly the instance data) it had before.
 *
 *  surfaceShade       the pile follows the form of the surface it grows on. Every direct
 *                     light is scaled by a soft terminator of the base-surface normal
 *                     (light from below the base's horizon would have to cross the mass:
 *                     the shadow side of a log stays dark, a back light only reaches the
 *                     outline), and the sky / ground fill is looked up with a normal bent
 *                     towards the base normal (an overhang's underside sees the ground).
 *                     Instances seen edge-on (the outline) and tips are partly spared.
 *                     With `dir` the albedo is darkened by a fixed world direction instead
 *                     of the per-light terminator (the scenes/branch SurfaceShade look,
 *                     built in); `radial` takes the base normal from the batch origin
 *                     (fur on a ball).
 *  clumpAo            occlusion inside clumps: Voronoi cells of `size` (by default a grass
 *                     layer's own `tufts` cells, so the dark cores sit where the blades
 *                     lean out of); towards a clump's centre and towards the roots the pile
 *                     is darker, the crown and the outer blades stay open, the borders
 *                     between clumps (gaps) are darker, each clump a little lighter or
 *                     darker than its neighbours.
 *  baseAo             number (strength, as before) or { strength, power, open }: the base
 *                     surface AO (COLOR_0.B at the scatter point) as min(1, AO / open) ^
 *                     power — crevices darker, open faces unchanged.
 *  translucencyFloor  back-light glow on plants whose base faces the camera (default 0.12).
 *
 * Per instance everything is computed at scatter time and appended to the layer's
 * interleaved instance data (base normal: 3 floats, clump: 4 floats); per frame only
 * uniforms. No extra draw calls, nothing depends on time.
 */
import { Vector3, Vector4, type IUniform } from "three";
import { hash3i } from "./noise";

export interface SurfaceShadeSpec {
  /** 0–1 amount of the whole effect (default 1). */
  strength?: number;
  /** Light factor on a plant whose base faces fully away from the light (default 0.2). */
  darkest?: number;
  /** Half-width of the terminator in n·l (default 0.45). */
  soft?: number;
  /** 0–1: how far the sky / ground fill normal is bent towards the base normal (default 0.6). */
  fill?: number;
  /** 0–1: how much plants on faces seen edge-on (the outline) are spared (default 0.6). */
  edge?: number;
  /** 0–1: how much the tips escape the shading (they stick out of the pile; default 0.25). */
  tips?: number;
  /**
   * Fixed world direction towards a light: albedo × mix(darkest, 1, terminator(n·dir))
   * instead of the per-light terminator. The fill bend still follows `fill` (0 = the plain
   * albedo look). `plantUniforms(material).uSilvaSurfDir` may be updated per frame.
   */
  dir?: [number, number, number];
  /** Base normal = direction of the plant's root from the batch origin (a pile on a ball). */
  radial?: boolean;
}

export interface ClumpAoSpec {
  /** 0–1 darkening of a clump's core at the roots. */
  strength: number;
  /** Clump (Voronoi cell) size in m; default: a grass layer's `tufts.size` (same cells), else 0.04. */
  size?: number;
  /** Depth (m) of the dark core at a clump's centre (default: grass 1.3 × canopy, kit 0.8 × plant height). */
  depth?: number;
  /** 0–1 darkening of the borders between clumps, whole plant (default 0.3). */
  gaps?: number;
  /** ± brightness variation per clump (default 0.12). */
  variation?: number;
  /**
   * 0–1: with `surfaceShade`, each clump is shaded as a small dome — the base normal the
   * terminator and the fill use tilts outwards from the clump centre, up to ≈ 45° at 1 on
   * the clump's border (default 0): the side of a tuft turned to the light is lit, its far
   * side and the valley behind it fall into shade.
   */
  dome?: number;
  /**
   * A second clump scale on the same plants, e.g. 2 cm sponge clumps inside 20 cm dome
   * cushions: its own cells of `fine.size` (default as `size`) and the same fields and
   * defaults; the two occlusions multiply and the dome tilts add (+3 floats per instance).
   */
  fine?: Omit<ClumpAoSpec, "fine">;
}

export interface BaseAoSpec {
  /** 0–1 strength (the layer's former number option). */
  strength?: number;
  /** AO ^ power before it is applied (default 1): > 1 darkens the crevices more. */
  power?: number;
  /**
   * AO at or above this counts as fully open (default 1): min(1, AO / open) ^ power. Bakes
   * rarely reach 1 on open faces (grove: ≈ 0.93); with `open` ≈ that value the open faces
   * stay unchanged and only the crevices darken.
   */
  open?: number;
}

export type BaseAoOption = number | BaseAoSpec;

/** The options shared by grass and kit layers. */
export interface PlantShadeOptions {
  surfaceShade?: SurfaceShadeSpec;
  clumpAo?: ClumpAoSpec;
  translucencyFloor?: number;
}

export interface ResolvedClump {
  strength: number;
  size: number;
  depth: number;
  gaps: number;
  variation: number;
  dome: number;
}

export interface PlantShadeSetup {
  /** Material defines (empty when nothing is opted into). */
  defines: Record<string, string>;
  uniforms: Record<string, IUniform>;
  /** Program cache key suffix ("" when nothing is opted into). */
  key: string;
  /** The instances carry their base normal (3 floats). */
  baseNormal: boolean;
  /** The instances carry clump values (4 floats). */
  clump: boolean;
  /** The instances carry the second (`clumpAo.fine`) clump scale (3 floats). */
  clump2: boolean;
  /** Extra floats per instance (0, 3, 4, 7, or + 3 with `fine`). */
  floats: number;
}

/** Strength of a `baseAo` option (number or spec) with the layer's default. */
export function baseAoStrength(o: BaseAoOption | undefined, fallback: number): number {
  if (typeof o === "number") return o;
  return o?.strength ?? fallback;
}

/** Base AO value as stored per instance: min(1, ao / open) ^ power (exactly `ao` without either). */
export function shapeBaseAo(ao: number, o: BaseAoOption | undefined): number {
  if (typeof o !== "object") return ao;
  const p = o.power ?? 1;
  const open = o.open ?? 1;
  if (p === 1 && open === 1) return ao;
  const a = Math.min(1, Math.max(0, ao) / Math.max(1e-3, open));
  return p === 1 ? a : Math.pow(a, p);
}

/** Defines / uniforms / instance layout for a layer's options. */
export function plantShadeSetup(o: PlantShadeOptions): PlantShadeSetup {
  const defines: Record<string, string> = {};
  const uniforms: Record<string, IUniform> = {};
  const key: string[] = [];
  const s = o.surfaceShade;
  const surface = !!s && (s.strength ?? 1) > 0;
  const clump = !!o.clumpAo && clumpActive(o.clumpAo);
  const clump2 = clump && !!o.clumpAo?.fine && clumpActive(o.clumpAo.fine);
  if (surface || clump) defines.SILVA_PLANT_SHADE = "";
  if (surface && s) {
    defines.SILVA_SURFACE_SHADE = "";
    // x darkest, y soft, z fill bend, w edge sparing
    uniforms.uSilvaSurf = { value: new Vector4(s.darkest ?? 0.2, Math.max(1e-3, s.soft ?? 0.45), s.fill ?? 0.6, s.edge ?? 0.6) };
    // x strength, y tips
    uniforms.uSilvaSurf2 = { value: new Vector4(Math.min(1, Math.max(0, s.strength ?? 1)), s.tips ?? 0.25, 0, 0) };
    key.push("surf");
    if (s.dir) {
      defines.SILVA_SURFACE_DIR = "";
      uniforms.uSilvaSurfDir = { value: new Vector3(s.dir[0], s.dir[1], s.dir[2]).normalize() };
      key.push("dir");
    }
    if (s.radial) {
      defines.SILVA_SURFACE_RADIAL = "";
      key.push("radial");
    }
  }
  if (clump) {
    defines.SILVA_CLUMP_AO = "";
    key.push("clump");
  }
  if (clump2) {
    defines.SILVA_CLUMP_AO2 = "";
    key.push("clump2");
  }
  if (o.translucencyFloor !== undefined) {
    defines.SILVA_TRANSL_FLOOR = "";
    uniforms.uSilvaTranslFloor = { value: o.translucencyFloor };
    key.push("floor");
  }
  const baseNormal = surface && !s?.radial;
  return { defines, uniforms, key: key.join(","), baseNormal, clump, clump2, floats: (baseNormal ? 3 : 0) + (clump ? 4 : 0) + (clump2 ? 3 : 0) };
}

function clumpActive(c: Omit<ClumpAoSpec, "fine">): boolean {
  return c.strength + (c.gaps ?? 0.3) + (c.variation ?? 0.12) > 0;
}

/** Interleaved attribute map of the shade data that starts at `offset`. */
export function plantShadeAttributes(setup: PlantShadeSetup, offset: number): Record<string, [number, number]> {
  const out: Record<string, [number, number]> = {};
  let o = offset;
  if (setup.baseNormal) {
    out.aBaseN = [o, 3];
    o += 3;
  }
  if (setup.clump) {
    out.aClump = [o, 4];
    o += 4;
  }
  if (setup.clump2) out.aClump2 = [o, 3];
  return out;
}

export function resolveClump(c: Omit<ClumpAoSpec, "fine">, defaults: { size: number; depth: number }): ResolvedClump {
  return {
    strength: Math.min(1, Math.max(0, c.strength)),
    size: Math.max(1e-3, c.size ?? defaults.size),
    depth: Math.max(1e-4, c.depth ?? defaults.depth),
    gaps: Math.min(1, Math.max(0, c.gaps ?? 0.3)),
    variation: c.variation ?? 0.12,
    dome: Math.max(0, c.dome ?? 0),
  };
}

export interface VoronoiHit {
  /** Distance to the nearest / second-nearest centre. */
  d1: number;
  d2: number;
  /** 0–1 hash of the nearest cell. */
  cell: number;
  /** Nearest centre. */
  cx: number;
  cy: number;
  cz: number;
}

/**
 * Nearest and second-nearest of the jittered lattice points `size` apart (the lattice and
 * jitter of GrassBatch tufts: equal size and seed give the same centres).
 */
export function voronoiF12(px: number, py: number, pz: number, size: number, seed: number, out: VoronoiHit): VoronoiHit {
  const gx = Math.floor(px / size);
  const gy = Math.floor(py / size);
  const gz = Math.floor(pz / size);
  let b1 = Infinity;
  let b2 = Infinity;
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
        if (d < b1) {
          b2 = b1;
          b1 = d;
          out.cell = hash3i(cx, cy, cz, seed + 505);
          out.cx = x;
          out.cy = y;
          out.cz = z;
        } else if (d < b2) b2 = d;
      }
    }
  }
  out.d1 = Math.sqrt(b1);
  out.d2 = Math.sqrt(b2);
  return out;
}

function smooth(e0: number, e1: number, x: number): number {
  const k = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return k * k * (3 - 2 * k);
}

const voro: VoronoiHit = { d1: 0, d2: 0, cell: 0, cx: 0, cy: 0, cz: 0 };
const voro2: VoronoiHit = { d1: 0, d2: 0, cell: 0, cx: 0, cy: 0, cz: 0 };
const tilt = { x: 0, y: 0, z: 0 };

/**
 * Dome tilt of a clump: the outward direction from the clump centre in the tangent plane of
 * n0, scaled to `amount` × (0 at the centre → 1 on the border), into `tilt`; false at the centre.
 */
function domeOffset(h: VoronoiHit, px: number, py: number, pz: number, n0x: number, n0y: number, n0z: number, amount: number, t: number): boolean {
  let ox = px - h.cx;
  let oy = py - h.cy;
  let oz = pz - h.cz;
  const dn = ox * n0x + oy * n0y + oz * n0z;
  ox -= dn * n0x;
  oy -= dn * n0y;
  oz -= dn * n0z;
  const ol = Math.hypot(ox, oy, oz);
  if (ol <= 1e-9) return false;
  const k = (amount * Math.min(1, t)) / ol;
  tilt.x = ox * k;
  tilt.y = oy * k;
  tilt.z = oz * k;
  return true;
}

/**
 * Append the opt-in shade data of one instance at data[o…] (the layout of
 * `plantShadeAttributes`): the base normal (n, tilted out of its clump's centre by
 * `clump.dome`), then the clump values — x core occlusion 0–1, y..z the plant-local
 * height range (m) over which it opens, w a constant factor (gaps × per-clump variation) —
 * then, with `fine` (the second clump scale), its x, y, z (its gaps / variation are folded
 * into w, its dome into the base normal).
 * (px, py, pz): the point on the base surface; `lift`: the plant's root sits this far above
 * it (kit plants), so the plant's own heights start there. Draws no random numbers.
 */
export function writeShadeData(
  data: Float32Array,
  o: number,
  setup: PlantShadeSetup,
  clump: ResolvedClump | null,
  seed: number,
  px: number,
  py: number,
  pz: number,
  nx: number,
  ny: number,
  nz: number,
  lift = 0,
  fine: ResolvedClump | null = null,
): void {
  let t = 0;
  if (clump) {
    voronoiF12(px, py, pz, clump.size, seed, voro);
    // 0 at the cell centre → 1 on the border between two clumps
    t = (2 * voro.d1) / Math.max(1e-9, voro.d1 + voro.d2);
  }
  const second = clump !== null && fine !== null && setup.clump2;
  let t2 = 0;
  if (second && fine) {
    voronoiF12(px, py, pz, fine.size, seed, voro2);
    t2 = (2 * voro2.d1) / Math.max(1e-9, voro2.d1 + voro2.d2);
  }
  let e = o;
  if (setup.baseNormal) {
    const n0x = nx;
    const n0y = ny;
    const n0z = nz;
    let tilted = false;
    if (clump && clump.dome > 0 && domeOffset(voro, px, py, pz, n0x, n0y, n0z, clump.dome, t)) {
      nx += tilt.x;
      ny += tilt.y;
      nz += tilt.z;
      tilted = true;
    }
    if (second && fine && fine.dome > 0 && domeOffset(voro2, px, py, pz, n0x, n0y, n0z, fine.dome, t2)) {
      nx += tilt.x;
      ny += tilt.y;
      nz += tilt.z;
      tilted = true;
    }
    if (tilted) {
      const l = Math.hypot(nx, ny, nz);
      nx /= l;
      ny /= l;
      nz /= l;
    }
    data[e] = nx;
    data[e + 1] = ny;
    data[e + 2] = nz;
    e += 3;
  }
  if (clump) {
    const core = 1 - smooth(0, 1, t);
    const depth = clump.depth * (0.3 + 0.7 * core);
    data[e] = clump.strength * (0.3 + 0.7 * core);
    data[e + 1] = -lift;
    data[e + 2] = depth - lift;
    let w = (1 - clump.gaps * smooth(0.7, 1, t)) * (1 + clump.variation * (2 * voro.cell - 1));
    if (second && fine) w *= (1 - fine.gaps * smooth(0.7, 1, t2)) * (1 + fine.variation * (2 * voro2.cell - 1));
    data[e + 3] = w;
    e += 4;
  }
  if (second && fine) {
    const core2 = 1 - smooth(0, 1, t2);
    data[e] = fine.strength * (0.3 + 0.7 * core2);
    data[e + 1] = -lift;
    data[e + 2] = fine.depth * (0.3 + 0.7 * core2) - lift;
  }
}

// ---------------------------------------------------------------------------
// GLSL (every block is behind its define)
// ---------------------------------------------------------------------------

/** Vertex declarations. */
export const SHADE_VERTEX_PARS = /* glsl */ `
#ifdef SILVA_PLANT_SHADE
centroid varying float vSilvaOcc;
#endif
#ifdef SILVA_SURFACE_SHADE
uniform vec4 uSilvaSurf;
uniform vec4 uSilvaSurf2;
centroid varying vec3 vSilvaBaseN;
centroid varying float vSilvaSurfK;
#ifdef SILVA_SURFACE_DIR
uniform vec3 uSilvaSurfDir;
#endif
#ifndef SILVA_SURFACE_RADIAL
attribute vec3 aBaseN;
#endif
#endif
#ifdef SILVA_CLUMP_AO
attribute vec4 aClump;
#endif
#ifdef SILVA_CLUMP_AO2
attribute vec3 aClump2;
#endif
#ifdef SILVA_TRANSL_FLOOR
uniform float uSilvaTranslFloor;
#endif
`;

/**
 * Vertex code after the plant's own main (needs silvaPos, silvaRootPos and the plant's
 * `silvaShadeH` 0 root → 1 tip and `silvaShadeDepth` m above the root).
 */
export const SHADE_VERTEX_MAIN = /* glsl */ `
#ifdef SILVA_PLANT_SHADE
  vSilvaOcc = 1.0;
#endif
#ifdef SILVA_SURFACE_SHADE
  {
#ifdef SILVA_SURFACE_RADIAL
    vec3 sbN = normalize(silvaRootPos);
#else
    vec3 sbN = aBaseN;
#endif
    vec3 sbV = normalize(normalMatrix * sbN);
    vec3 sbE = normalize(-(modelViewMatrix * vec4(silvaRootPos, 1.0)).xyz);
    float sbEdge = smoothstep(0.3, 0.85, 1.0 - abs(dot(sbV, sbE)));
    float sbK = uSilvaSurf2.x * (1.0 - uSilvaSurf.w * sbEdge) * (1.0 - uSilvaSurf2.y * silvaShadeH);
#ifdef SILVA_SURFACE_DIR
    float sbLit = smoothstep(-uSilvaSurf.y, uSilvaSurf.y, dot(normalize(mat3(modelMatrix) * sbN), uSilvaSurfDir));
    vSilvaOcc *= mix(1.0, mix(uSilvaSurf.x, 1.0, sbLit), sbK);
#endif
    vSilvaBaseN = sbV;
    vSilvaSurfK = sbK;
  }
#endif
#ifdef SILVA_CLUMP_AO
  vSilvaOcc *= aClump.w * (1.0 - aClump.x * (1.0 - smoothstep(aClump.y, aClump.z, silvaShadeDepth)));
#endif
#ifdef SILVA_CLUMP_AO2
  vSilvaOcc *= 1.0 - aClump2.x * (1.0 - smoothstep(aClump2.y, aClump2.z, silvaShadeDepth));
#endif
`;

/** Fragment declarations. */
export const SHADE_FRAGMENT_PARS = /* glsl */ `
#ifdef SILVA_PLANT_SHADE
centroid varying float vSilvaOcc;
#endif
#ifdef SILVA_SURFACE_SHADE
uniform vec4 uSilvaSurf;
centroid varying vec3 vSilvaBaseN;
centroid varying float vSilvaSurfK;
#endif
`;

/** After the plant's albedo code. */
export const SHADE_FRAGMENT_COLOR = /* glsl */ `
#ifdef SILVA_PLANT_SHADE
  diffuseColor.rgb *= vSilvaOcc;
#endif
`;

/**
 * After lights_physical_pars_fragment: the hemisphere fill looked up with the normal bent
 * towards the base normal (the call in lights_fragment_begin is redirected by a macro).
 */
export const SHADE_HEMI = /* glsl */ `
#if defined( SILVA_SURFACE_SHADE ) && ( NUM_HEMI_LIGHTS > 0 )
vec3 silvaHemiIrradiance( const in HemisphereLight hemiLight, const in vec3 normal ) {
  vec3 n = mix( normal, normalize( vSilvaBaseN ), uSilvaSurf.z * vSilvaSurfK );
  n *= inversesqrt( max( dot( n, n ), 1e-8 ) );
  return getHemisphereLightIrradiance( hemiLight, n );
}
#define getHemisphereLightIrradiance( hemiLight, normal ) silvaHemiIrradiance( hemiLight, normal )
#endif
`;
