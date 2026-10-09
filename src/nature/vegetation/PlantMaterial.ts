/**
 * Shared shader plumbing for instanced plants (grass blades, kit leaves / flowers).
 *
 * Plants are `MeshStandardMaterial`s patched in `onBeforeCompile`, so three.js keeps
 * doing lights, shadows (received), fog and tone mapping. Each plant kind provides:
 *
 *   vertexPars   attributes / uniforms / functions (WIND_GLSL is included for it)
 *   vertexMain   code that must define `vec3 silvaPos` (object space of the batch mesh)
 *                and `vec3 silvaNormal`, `vec3 silvaRootPos` / `vec3 silvaRootUp` (root
 *                and growth axis, object space), and write `vSilvaColor` (linear albedo
 *                multiplier incl. AO) and `vSilvaTransl` (0–1 back-light strength)
 *
 * The fragment side multiplies the albedo by `vSilvaColor` and adds a thin-sheet
 * translucency term to every direct light: light that reaches the back of a blade
 * shows through it, strongest when looking into the light. It is an imitation, not
 * subsurface scattering, and it uses the shadowed light colour (no glow in shadow).
 * Light from behind the plant (relative to the camera) only shows through where the
 * plant's surface is seen edge-on (`vSilvaRim`): on a face turned to the camera the
 * mass itself would block a back light, so the glow stays on the silhouettes (rims of
 * the masses, the keyhole edge of T01) — also for lights that cast no shadows.
 *
 * Optional pile shading (`shade`, PlantShade.ts: surface shade, clump AO, translucency
 * floor) is compiled in only through its defines; without them the program is the same
 * as before. The vertex code keeps the line `vec3 objectNormal = silvaNormal;` and the
 * names silvaPos / silvaRootPos / silvaRootUp / vSilvaColor (scenes/branch hooks them).
 *
 * All Silva varyings are `centroid`. Under MSAA (the post stack's scene target) a pixel
 * that only some samples of a sliver triangle cover — a wide blade seen edge-on — still
 * runs the fragment shader once, at the pixel centre, outside the triangle; plain
 * varyings are then extrapolated far beyond their vertex values, and the multipliers
 * (albedo × AO, clump AO, terminator, translucency) multiplied up to ~16 000× white:
 * single firefly pixels. Centroid sampling keeps them inside the covered part.
 */
import type { IUniform, MeshStandardMaterial, WebGLProgramParametersWithUniforms } from "three";
import { SHADE_FRAGMENT_COLOR, SHADE_FRAGMENT_PARS, SHADE_HEMI, SHADE_VERTEX_MAIN, SHADE_VERTEX_PARS, type PlantShadeSetup } from "./PlantShade";
import { WIND_GLSL, bindWindUniforms, type WindUniforms } from "./WindField";

export interface PlantShaderParts {
  /** Unique program key of this configuration. */
  key: string;
  vertexPars: string;
  vertexMain: string;
  uniforms: Record<string, IUniform>;
  /** Strength of the translucency term (multiplies vSilvaTransl). */
  translucency: IUniform<number>;
  defines?: Record<string, string>;
  /** Extra fragment declarations (varyings / uniforms of `fragmentColor`). */
  fragmentPars?: string;
  /** Albedo code replacing `diffuseColor.rgb *= vSilvaColor;` (map already applied). */
  fragmentColor?: string;
  /**
   * Optional pile shading (PlantShade.ts `plantShadeSetup`). `vertexMain` must then also
   * define `float silvaShadeH` (0 root → 1 tip) and `float silvaShadeDepth` (m above the
   * root) inside `#ifdef SILVA_PLANT_SHADE`.
   */
  shade?: PlantShadeSetup;
}

const COMMON_VERTEX_PARS = /* glsl */ `
centroid varying vec3 vSilvaColor;
centroid varying float vSilvaTransl;
centroid varying float vSilvaRim;
${WIND_GLSL}
${SHADE_VERTEX_PARS}
`;

/** Silhouette factor of the plant's base surface (1 = seen edge-on). */
const RIM_MAIN = /* glsl */ `
  {
    vec3 rimN = normalize(normalMatrix * silvaRootUp);
    vec3 rimV = normalize(-(modelViewMatrix * vec4(silvaRootPos, 1.0)).xyz);
#ifdef SILVA_TRANSL_FLOOR
    vSilvaRim = mix(uSilvaTranslFloor, 1.0, smoothstep(0.3, 0.85, 1.0 - abs(dot(rimN, rimV))));
#else
    vSilvaRim = mix(0.12, 1.0, smoothstep(0.3, 0.85, 1.0 - abs(dot(rimN, rimV))));
#endif
  }
`;

const FRAGMENT_PARS = /* glsl */ `
centroid varying vec3 vSilvaColor;
centroid varying float vSilvaTransl;
centroid varying float vSilvaRim;
uniform float uSilvaTranslucency;
${SHADE_FRAGMENT_PARS}
`;

/**
 * Replaces RE_Direct: physical BRDF + thin-sheet translucency. With surface shade (per-light
 * mode) the light is first scaled by the terminator of the base normal: light from below the
 * base surface's horizon would have to cross the mass the plant grows on.
 */
const TRANSLUCENCY = /* glsl */ `
#if defined( SILVA_SURFACE_SHADE ) && !defined( SILVA_SURFACE_DIR )
void RE_Direct_SilvaPlant( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
  IncidentLight silvaLight = directLight;
  float silvaLit = smoothstep( -uSilvaSurf.y, uSilvaSurf.y, dot( normalize( vSilvaBaseN ), directLight.direction ) );
  silvaLight.color *= mix( 1.0, mix( uSilvaSurf.x, 1.0, silvaLit ), vSilvaSurfK );
  RE_Direct_Physical( silvaLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
  float back = saturate( dot( -geometryNormal, silvaLight.direction ) );
  float into = pow( saturate( dot( -geometryViewDir, silvaLight.direction ) ), 3.0 );
  float behind = smoothstep( 0.0, 0.6, dot( -geometryViewDir, silvaLight.direction ) );
  float t = vSilvaTransl * uSilvaTranslucency * mix( 1.0, vSilvaRim, behind );
  reflectedLight.directDiffuse += silvaLight.color * BRDF_Lambert( material.diffuseContribution ) * t * ( 0.7 * back + 1.6 * into * ( 0.35 + 0.65 * back ) );
}
#else
void RE_Direct_SilvaPlant( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
  RE_Direct_Physical( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
  float back = saturate( dot( -geometryNormal, directLight.direction ) );
  float into = pow( saturate( dot( -geometryViewDir, directLight.direction ) ), 3.0 );
  float behind = smoothstep( 0.0, 0.6, dot( -geometryViewDir, directLight.direction ) );
  float t = vSilvaTransl * uSilvaTranslucency * mix( 1.0, vSilvaRim, behind );
  reflectedLight.directDiffuse += directLight.color * BRDF_Lambert( material.diffuseContribution ) * t * ( 0.7 * back + 1.6 * into * ( 0.35 + 0.65 * back ) );
}
#endif
#undef RE_Direct
#define RE_Direct RE_Direct_SilvaPlant
${SHADE_HEMI}
`;

/**
 * Patch `material` into a plant material. `wind` is referenced (shared), never copied.
 * Call once per material; sets `customProgramCacheKey`.
 */
export function patchPlantMaterial(material: MeshStandardMaterial, parts: PlantShaderParts, wind: WindUniforms): void {
  const shade = parts.shade;
  const uniforms: Record<string, IUniform> = { ...parts.uniforms, ...(shade?.uniforms ?? {}), uSilvaTranslucency: parts.translucency };
  material.userData.silvaPlantUniforms = uniforms;
  const defines = { ...(parts.defines ?? {}), ...(shade?.defines ?? {}) };
  if (Object.keys(defines).length) material.defines = { ...(material.defines ?? {}), ...defines };
  material.onBeforeCompile = (shader: WebGLProgramParametersWithUniforms) => {
    bindWindUniforms(shader.uniforms, wind);
    for (const [k, v] of Object.entries(uniforms)) shader.uniforms[k] = v;

    let vs = shader.vertexShader;
    vs = vs.replace("#include <common>", `#include <common>\n${COMMON_VERTEX_PARS}\n${parts.vertexPars}`);
    vs = vs.replace("#include <beginnormal_vertex>", `${parts.vertexMain}\n${RIM_MAIN}\n${SHADE_VERTEX_MAIN}\n  vec3 objectNormal = silvaNormal;`);
    vs = vs.replace("#include <begin_vertex>", "vec3 transformed = silvaPos;");
    shader.vertexShader = vs;

    let fs = shader.fragmentShader;
    fs = fs.replace("#include <common>", `#include <common>\n${FRAGMENT_PARS}\n${parts.fragmentPars ?? ""}`);
    fs = fs.replace("#include <color_fragment>", `${parts.fragmentColor ?? "diffuseColor.rgb *= vSilvaColor;"}\n${SHADE_FRAGMENT_COLOR}`);
    fs = fs.replace("#include <lights_physical_pars_fragment>", `#include <lights_physical_pars_fragment>\n${TRANSLUCENCY}`);
    shader.fragmentShader = fs;
  };
  const key = shade?.key ? `silva-plant|${parts.key}|${shade.key}` : `silva-plant|${parts.key}`;
  material.customProgramCacheKey = () => key;
}

/** The uniforms a plant material was patched with (to tweak them later). */
export function plantUniforms(material: MeshStandardMaterial): Record<string, IUniform> {
  return (material.userData.silvaPlantUniforms ?? {}) as Record<string, IUniform>;
}

/** Linear RGB triple of a THREE.Color-like for attribute packing. */
export interface Rgb {
  r: number;
  g: number;
  b: number;
}
