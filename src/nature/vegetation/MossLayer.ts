/**
 * Layer A of the vegetation: the dense dark moss base under the grass and leaves.
 *
 * It is the `moss_*` meshes themselves (their hummocks and cushions are modelled in
 * Blender) shaded with the triplanar moss texture set from `rendering/Materials.ts`,
 * plus what makes it read as a base layer rather than a green rubber skin:
 *  - darker albedo and a mostly matte response (almost no highlights),
 *  - large patches of colour / brightness variation, much bigger than single plants,
 *  - a soft sheen at grazing angles (the fuzz of the moss catching the sky light),
 *  - baked AO from COLOR_0.B (factory), texture AO on indirect light.
 * The short "fuzz" blades on top of it are an ordinary grass layer in the recipe.
 */
import { Color, Vector3, type ColorRepresentation, type MeshStandardMaterial, type WebGLProgramParametersWithUniforms, type WebGLRenderer } from "three";
import { createMossBaseMaterial } from "../rendering/Materials";
import type { SceneContext } from "../types";

export interface MossLayerOptions {
  /** Albedo multiplier on the moss texture (sRGB hex; the texture itself is already dark). */
  tint?: ColorRepresentation;
  /** Extra linear albedo factor on top of the tint (may exceed 1, default 1). */
  brightness?: number;
  /** Albedo saturation (1 = the texture's, 0 = grey; default 1). */
  saturation?: number;
  /** Texture repeats per metre (moss tiles are 0.3 m: 3.33). */
  scale?: number;
  roughness?: number;
  /** Strength of the baked AO from COLOR_0.B (default 1). */
  aoStrength?: number;
  /** Frequency of the large colour patches (1/m, default 1.8). */
  patchScale?: number;
  /** 0–1 brightness contrast of the patches (default 0.35). */
  patchContrast?: number;
  /** Second patch colour mixed in by the patches (sRGB hex, default olive). */
  patchColor?: ColorRepresentation;
  /** Grazing-angle sheen strength (default 0.25). */
  sheen?: number;
  /** Sheen colour (sRGB hex). */
  sheenColor?: ColorRepresentation;
  /**
   * Light from above on the cushions: albedo × mix(bottom, top, smoothstep(−0.6, 0.8, n.y))
   * (object-space normal; the GLB transforms are applied, so +Y is up) and a shift towards
   * `topColor` (sRGB) by `topAmount` on the faces turned to the sky. Default: none.
   */
  shade?: { top: number; bottom: number; topColor?: ColorRepresentation; topAmount?: number };
  /** Extra contrast of the baked AO (COLOR_0.B): albedo × ao^aoContrast on top of the base AO (default 0). */
  aoContrast?: number;
}

/** Moss base material (triplanar, metric) with the layer-A look on top. */
export async function createMossLayerMaterial(ctx: SceneContext, opts: MossLayerOptions = {}): Promise<MeshStandardMaterial> {
  const mat = await createMossBaseMaterial(ctx.assets, {
    color: opts.tint ?? "#FFFFFF",
    scale: opts.scale ?? 3.33,
    roughness: opts.roughness ?? 1,
    aoStrength: opts.aoStrength ?? 1,
  });
  mat.name = "SilvaMossLayer";
  if (opts.brightness !== undefined) mat.color.multiplyScalar(opts.brightness);
  const uniforms = {
    uMossPatchScale: { value: opts.patchScale ?? 1.8 },
    uMossPatchContrast: { value: opts.patchContrast ?? 0.35 },
    uMossPatchColor: { value: new Color(opts.patchColor ?? "#6B6A3A") },
    uMossSheen: { value: opts.sheen ?? 0.25 },
    uMossSheenColor: { value: new Color(opts.sheenColor ?? "#9FB07A") },
    uMossSaturation: { value: opts.saturation ?? 1 },
    uMossShade: { value: new Vector3(opts.shade?.bottom ?? 1, opts.shade?.top ?? 1, opts.shade?.topAmount ?? 0) },
    uMossTopColor: { value: new Color(opts.shade?.topColor ?? "#FFFFFF") },
    uMossAoContrast: { value: opts.aoContrast ?? 0 },
  };
  mat.userData.mossLayerUniforms = uniforms;
  const base = mat.onBeforeCompile;
  mat.onBeforeCompile = (shader: WebGLProgramParametersWithUniforms, renderer: WebGLRenderer) => {
    base.call(mat, shader, renderer);
    Object.assign(shader.uniforms, uniforms);
    let fs = shader.fragmentShader;
    fs = fs.replace(
      "#include <common>",
      `#include <common>
uniform float uMossPatchScale;
uniform float uMossPatchContrast;
uniform vec3 uMossPatchColor;
uniform float uMossSheen;
uniform vec3 uMossSheenColor;
uniform float uMossSaturation;
uniform vec3 uMossShade;
uniform vec3 uMossTopColor;
uniform float uMossAoContrast;`,
    );
    // large patches (object space, metric): brightness and a shift towards olive
    fs = fs.replace(
      "#include <color_fragment>",
      `#include <color_fragment>
  {
    float mp = tpNoise3(vTpPos * uMossPatchScale) * 0.7 + tpNoise3(vTpPos * uMossPatchScale * 2.7 + 3.1) * 0.3;
    diffuseColor.rgb *= 1.0 - uMossPatchContrast + 2.0 * uMossPatchContrast * mp;
    diffuseColor.rgb = mix(diffuseColor.rgb, uMossPatchColor * dot(diffuseColor.rgb, vec3(0.33)) * 2.2, 0.35 * smoothstep(0.45, 0.8, mp));
    diffuseColor.rgb = mix(vec3(dot(diffuseColor.rgb, vec3(0.2126, 0.7152, 0.0722))), diffuseColor.rgb, uMossSaturation);
    // sky-facing tops lighter and yellower, undersides and folds darker
    float mUp = smoothstep(-0.6, 0.8, vTpNormal.y * inversesqrt(max(dot(vTpNormal, vTpNormal), 1e-8)));
    diffuseColor.rgb *= mix(uMossShade.x, uMossShade.y, mUp);
    diffuseColor.rgb = mix(diffuseColor.rgb, uMossTopColor * dot(diffuseColor.rgb, vec3(0.2126, 0.7152, 0.0722)) * 1.4, uMossShade.z * mUp);
    // pow(0, 0) is undefined in GLSL (NaN on some GPUs): only with a positive exponent
    if (uMossAoContrast > 0.0) diffuseColor.rgb *= pow(clamp(vSilvaAo, 1e-4, 1.0), uMossAoContrast);
  }`,
    );
    // fuzz sheen at grazing angles, lit by the hemisphere fill only (never glows in shadow)
    fs = fs.replace(
      "#include <lights_fragment_end>",
      `#include <lights_fragment_end>
  {
    float nv = saturate(dot(normal, normalize(vViewPosition)));
    float graze = pow(1.0 - nv, 3.0);
    reflectedLight.indirectDiffuse += uMossSheenColor * uMossSheen * graze * irradiance * RECIPROCAL_PI * diffuseColor.rgb * 4.0;
  }`,
    );
    shader.fragmentShader = fs;
  };
  const baseKey = mat.customProgramCacheKey();
  mat.customProgramCacheKey = () => `${baseKey}|moss-layer`;
  return mat;
}
