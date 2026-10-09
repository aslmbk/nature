/**
 * Finale-only material tuning on top of the shared factories (rendering/Materials.ts is
 * not touched: these wrap a factory material's `onBeforeCompile` and re-use the tokens it
 * leaves in place, like GroveScene's `tuneBark`).
 *
 * - `tuneFinaleStone`: the standing stone (frame 19, detail 10) is a warm pale, matte beige
 *   with soft large mottling and a fine grain (the rock set at a fine triplanar scale),
 *   olive moss stains where `COLOR_0.R` marks the crack and the foot, and a green bounce
 *   from the lit moss on its lower part.
 * - `tuneFinaleFar`: the far walls stay near-black (frame 19: p50 ≈ #0B0F12) with faint
 *   moss on them (highlights ≈ #191E18): ribs following the walls' inner edges, broken into
 *   soft patches (`ribLength`) so that nothing reads as a beam of light.
 * - `applyLightPool`: the key's pool of light on the moss (an ellipse on the ground around
 *   the stone, darker on the lip below the moss).
 */
import {
  Color,
  Vector4,
  type ColorRepresentation,
  type Material,
  type MeshStandardMaterial,
  type WebGLProgramParametersWithUniforms,
  type WebGLRenderer,
} from "three";

// ---------------------------------------------------------------------------
// Stone
// ---------------------------------------------------------------------------

export interface StoneTuning {
  /** 0–1 saturation of the rock texture (1 = as painted). */
  saturation: number;
  /** ± brightness of the large mottling. */
  mottle: number;
  /** Moss stain colour (sRGB) and strength where COLOR_0.R marks it. */
  mossColor: ColorRepresentation;
  moss: number;
  /** Green bounce from the moss on the stone's lower part (albedo tint strength 0–1). */
  bounce: number;
  bounceColor: ColorRepresentation;
  /** World height (m) of the moss crest under the stone: the bounce fades out above it. */
  footY: number;
  /** 0–1 strength of the rock normal map (the grain), 1 = as the texture set. */
  normal: number;
}

export function tuneFinaleStone(mat: MeshStandardMaterial, o: StoneTuning): void {
  const uniforms = {
    uStoneSat: { value: o.saturation },
    uStoneMottle: { value: o.mottle },
    uStoneMossColor: { value: new Color(o.mossColor) },
    uStoneMoss: { value: o.moss },
    uStoneBounce: { value: o.bounce },
    uStoneBounceColor: { value: new Color(o.bounceColor) },
    uStoneFootY: { value: o.footY },
    uStoneNormal: { value: o.normal },
  };
  const base = mat.onBeforeCompile;
  mat.onBeforeCompile = (shader: WebGLProgramParametersWithUniforms, renderer: WebGLRenderer) => {
    base.call(mat, shader, renderer);
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", "#include <common>\nvarying float vStoneMoss;\nvarying vec3 vStoneWorld;")
      .replace("#include <color_vertex>", "#include <color_vertex>\n  vStoneMoss = color.r;")
      .replace("#include <worldpos_vertex>", "#include <worldpos_vertex>\n  vStoneWorld = (modelMatrix * vec4(transformed, 1.0)).xyz;");
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>
varying float vStoneMoss;
varying vec3 vStoneWorld;
uniform float uStoneSat;
uniform float uStoneMottle;
uniform vec3 uStoneMossColor;
uniform float uStoneMoss;
uniform float uStoneBounce;
uniform vec3 uStoneBounceColor;
uniform float uStoneFootY;
uniform float uStoneNormal;`,
      )
      .replace(
        "#include <color_fragment>",
        `#include <color_fragment>
  {
    // pale grey-beige: the painted rock towards grey, large soft mottling (5–20 cm)
    float sl = dot(diffuseColor.rgb, vec3(0.2126, 0.7152, 0.0722));
    diffuseColor.rgb = mix(vec3(sl), diffuseColor.rgb, uStoneSat);
    float sm = tpNoise3(vTpPos * 5.0 + 1.7) * 0.6 + tpNoise3(vTpPos * 14.0 + 5.3) * 0.4;
    diffuseColor.rgb *= 1.0 + uStoneMottle * (2.0 * sm - 1.0);
    // moss stains (COLOR_0.R: crack, foot, a few spots): olive, broken up at 1–3 cm
    float sn = tpNoise3(vTpPos * 38.0) * 0.6 + tpNoise3(vTpPos * 90.0 + 2.0) * 0.4;
    float st = smoothstep(0.08, 0.55, vStoneMoss * (0.55 + 0.9 * sn)) * uStoneMoss;
    diffuseColor.rgb = mix(diffuseColor.rgb, uStoneMossColor * (0.55 + 0.9 * sn), st);
    // green bounce of the lit moss on the lower part of the stone
    float sb = (1.0 - smoothstep(uStoneFootY, uStoneFootY + 0.32, vStoneWorld.y)) * uStoneBounce;
    diffuseColor.rgb *= mix(vec3(1.0), uStoneBounceColor, sb);
  }`,
      )
      // softer grain: the triplanar normal mixed back towards the geometric one
      .replace(
        "#include <clearcoat_normal_fragment_begin>",
        "normal = normalize(mix(nonPerturbedNormal, normal, uStoneNormal));\n#include <clearcoat_normal_fragment_begin>",
      );
  };
  const key = mat.customProgramCacheKey();
  mat.customProgramCacheKey = () => `${key}|finale-stone`;
  mat.userData.finaleStoneUniforms = uniforms;
}

// ---------------------------------------------------------------------------
// Far walls
// ---------------------------------------------------------------------------

export interface FarTuning {
  /** Strength (albedo ×) of the long ribs. */
  ribs: number;
  /** Ribs per metre across the slant. */
  ribScale: number;
  /** Exponent of the rib profile (default 4): higher = thinner ribs. */
  ribSharpness?: number;
  /**
   * Rib frequency along the slant, per metre (default 0.12: ribs several metres long).
   * Higher breaks them into blotches (no long streaks that read as light beams).
   */
  ribLength?: number;
  /** How far the ribs lean in (Δx per metre up, mirrored at x = 0). */
  slant: number;
  /** Greenish moss colour (sRGB, linear albedo) on the ribs. */
  mossColor: ColorRepresentation;
  /** Amount of green on the ribs. */
  moss: number;
  /** Large-scale brightness variation ±. */
  patches: number;
  /**
   * Self-lit share (linear sRGB colour): the walls stand in the cold haze, lit alike from
   * both sides (frame 19: left and right walls both ≈ #0B0F12); the key adds a little on
   * top. Modulated like the albedo; the green moss on the ribs adds `mossGlow`.
   */
  emissive: ColorRepresentation;
  mossGlow: ColorRepresentation;
  /**
   * Specular scale (default 1). Even at roughness 1 the walls' grazing faces pick up a
   * grey Fresnel sheen from the key that their near-black albedo cannot hide; 0 removes it.
   */
  specular?: number;
}

export function tuneFinaleFar(mat: MeshStandardMaterial, o: FarTuning): void {
  const uniforms = {
    uFarRibs: { value: o.ribs },
    uFarRibScale: { value: o.ribScale },
    uFarSlant: { value: o.slant },
    uFarMossColor: { value: new Color(o.mossColor) },
    uFarMoss: { value: o.moss },
    uFarPatches: { value: o.patches },
    uFarMossGlow: { value: new Color(o.mossGlow) },
    uFarSpec: { value: o.specular ?? 1 },
    uFarRibSharp: { value: o.ribSharpness ?? 4 },
    uFarRibLen: { value: o.ribLength ?? 0.12 },
  };
  mat.emissive.set(o.emissive);
  const base = mat.onBeforeCompile;
  mat.onBeforeCompile = (shader: WebGLProgramParametersWithUniforms, renderer: WebGLRenderer) => {
    base.call(mat, shader, renderer);
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", "#include <common>\nvarying vec3 vFarPos;\nvarying float vFarR;")
      .replace("#include <color_vertex>", "#include <color_vertex>\n  vFarR = color.r;")
      .replace("#include <worldpos_vertex>", "#include <worldpos_vertex>\n  vFarPos = (modelMatrix * vec4(transformed, 1.0)).xyz;");
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>
varying vec3 vFarPos;
varying float vFarR;
uniform float uFarRibs;
uniform float uFarRibScale;
uniform float uFarSlant;
uniform vec3 uFarMossColor;
uniform float uFarMoss;
uniform float uFarPatches;
uniform vec3 uFarMossGlow;
uniform float uFarSpec;
uniform float uFarRibSharp;
uniform float uFarRibLen;
float farHash(vec3 p) {
  p = fract(p * 0.1031);
  p += dot(p, p.zyx + 31.32);
  return fract((p.x + p.y) * p.z);
}
float farNoise(vec3 p) {
  vec3 i = floor(p);
  vec3 f = fract(p);
  vec3 u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(mix(farHash(i), farHash(i + vec3(1, 0, 0)), u.x), mix(farHash(i + vec3(0, 1, 0)), farHash(i + vec3(1, 1, 0)), u.x), u.y),
    mix(mix(farHash(i + vec3(0, 0, 1)), farHash(i + vec3(1, 0, 1)), u.x), mix(farHash(i + vec3(0, 1, 1)), farHash(i + vec3(1, 1, 1)), u.x), u.y),
    u.z);
}`,
      )
      .replace(
        "#include <color_fragment>",
        `#include <color_fragment>
  {
    // ribs follow the walls' inner edges: they lean in towards the middle going up
    vec3 fp = vFarPos;
    float side = fp.x < 0.0 ? -1.0 : 1.0;
    float s = fp.x - side * uFarSlant * fp.y;
    vec3 rq = vec3(s * uFarRibScale, fp.y * uFarRibLen, fp.z * uFarRibScale * 0.5);
    float rn = farNoise(rq) * 0.65 + farNoise(rq * 2.3 + 7.1) * 0.35;
    float rib = pow(1.0 - abs(2.0 * rn - 1.0), uFarRibSharp);
    // breaks along the ribs (patchy moss, not ruled lines)
    rib *= smoothstep(0.35, 0.75, farNoise(vec3(s * 0.6, fp.y * 0.55, fp.z * 0.4) + 3.3));
    float big = farNoise(fp * 0.32 + 11.0);
    float farMod = (1.0 + uFarPatches * (2.0 * big - 1.0)) * (1.0 + uFarRibs * rib);
    diffuseColor.rgb *= farMod;
    totalEmissiveRadiance *= farMod;
    float green = clamp(rib * uFarMoss * (0.5 + 2.0 * vFarR), 0.0, 1.0);
    diffuseColor.rgb = mix(diffuseColor.rgb, uFarMossColor, green);
    totalEmissiveRadiance += uFarMossGlow * green;
  }`,
      )
      .replace(
        "#include <aomap_fragment>",
        "reflectedLight.directSpecular *= uFarSpec;\n  reflectedLight.indirectSpecular *= uFarSpec;\n#include <aomap_fragment>",
      );
  };
  const key = mat.customProgramCacheKey();
  mat.customProgramCacheKey = () => `${key}|finale-far`;
  mat.userData.finaleFarUniforms = uniforms;
}

// ---------------------------------------------------------------------------
// Light pool
// ---------------------------------------------------------------------------

export interface LightPool {
  /** Centre of the pool on the ground (world x, z in m): under the stone. */
  centre: [number, number];
  /** Half-axes of the pool's ellipse on the ground (world x, z in m). */
  axes: [number, number];
  /** Elliptic distance (1 = on the ellipse): fully lit up to `radius[0]`, `edge` from `radius[1]` on. */
  radius: [number, number];
  /** Light left outside the pool (0–1). */
  edge: number;
  /** World height (m): `floor` at or below `y[0]`, no change from `y[1]` up (the plateau's lip). */
  y: [number, number];
  floor: number;
}

/**
 * The finale's key reads as a pool of light on the plateau (frames 19 / 20): the mounds
 * around the stone lit, the plateau falling off softly towards the front and the frame
 * edges, and the lip below the moss darker still. A directional key cannot fall off, so
 * this scales what a material sends out by a world-space factor — an ellipse on the ground
 * around the stone, not a height: the hollows between the mounds keep their light (a
 * height falloff turned them black) — computed per vertex, the same factor for the moss
 * base and every plant on it. Shared uniforms: one object for all materials of the pool.
 */
export interface LightPoolUniforms {
  [name: string]: { value: Vector4 };
  uFinalePoolA: { value: Vector4 };
  uFinalePoolB: { value: Vector4 };
  uFinalePoolC: { value: Vector4 };
}

export function lightPoolUniforms(o: LightPool): LightPoolUniforms {
  return {
    // centre x, centre z, 1 / axis x, 1 / axis z
    uFinalePoolA: { value: new Vector4(o.centre[0], o.centre[1], 1 / o.axes[0], 1 / o.axes[1]) },
    // radius 0, radius 1, edge, -
    uFinalePoolB: { value: new Vector4(o.radius[0], o.radius[1], o.edge, 0) },
    // y0, y1, floor, -
    uFinalePoolC: { value: new Vector4(o.y[0], o.y[1], o.floor, 0) },
  };
}

// centroid: a sliver blade's fragment must not extrapolate the factor (PlantMaterial)
const POOL_PARS = `uniform vec4 uFinalePoolA;
uniform vec4 uFinalePoolB;
uniform vec4 uFinalePoolC;
centroid varying float vFinalePool;`;

/** Scale `mat`'s outgoing light by the pool (uniform objects shared, never copied). */
export function applyLightPool(mat: Material, uniforms: LightPoolUniforms): void {
  const base = mat.onBeforeCompile;
  mat.onBeforeCompile = (shader: WebGLProgramParametersWithUniforms, renderer: WebGLRenderer) => {
    base.call(mat, shader, renderer);
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", `#include <common>
${POOL_PARS}`)
      .replace(
        "#include <project_vertex>",
        `#include <project_vertex>
  {
    vec4 fpw = vec4(transformed, 1.0);
#ifdef USE_INSTANCING
    fpw = instanceMatrix * fpw;
#endif
    fpw = modelMatrix * fpw;
    float fpd = length((fpw.xz - uFinalePoolA.xy) * uFinalePoolA.zw);
    float fpr = smoothstep(uFinalePoolB.x, uFinalePoolB.y, fpd);
    float fpy = smoothstep(uFinalePoolC.x, uFinalePoolC.y, fpw.y);
    vFinalePool = mix(1.0, uFinalePoolB.z, fpr) * mix(uFinalePoolC.z, 1.0, fpy);
  }`,
      );
    shader.fragmentShader = shader.fragmentShader
      .replace("#include <common>", `#include <common>
${POOL_PARS}`)
      .replace("#include <opaque_fragment>", "outgoingLight *= clamp(vFinalePool, 0.0, 1.0);\n#include <opaque_fragment>");
  };
  const key = mat.customProgramCacheKey();
  mat.customProgramCacheKey = () => `${key}|finale-pool`;
}
