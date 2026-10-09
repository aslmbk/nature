/**
 * Bark tone of the branch set: the shared bark texture is a fine wood grain with dark
 * lines painted into its albedo; at the branch's scale they read as driftwood. Here the
 * albedo keeps only a share of that grain (`contrast`, around the texture's mean
 * colour), so the light and dark of the bark come from the geometry (strands, creases,
 * cracks), its baked AO (COLOR_0.B) and the normal map instead (detail 06, frame 10).
 *
 * Optional moss stain (`stain`): where COLOR_0.R (vegetation density) is high, i.e. on the
 * bark within a few cm of the moss border and in the deep cracks, the albedo darkens to
 * damp, moss-stained bark (frame 10: the bark is mottled and dark next to the moss).
 *
 * Optional fresh wood (`wood_branch_stumps`: COLOR_0.G == 0 on the breaks, the splinter
 * lips and the torn fibre ends, ≥ 0.3 everywhere else): there the albedo is a pale warm
 * colour of its own with only a trace of the painted grain (no grain darkening) and its
 * own roughness; the edge follows the interpolated G (0.03–0.07), so a break line stays a
 * crisp line (frame 10: pale broken tops on dark weathered stumps).
 *
 * Patches one material created by `createBarkMaterial` (wraps its onBeforeCompile and
 * gives it its own program key). Needs three.js's `#include <map_fragment>` chunk; if it
 * is missing the material is left as it is (one warning).
 */
import { Color, Vector3, type MeshStandardMaterial, type WebGLProgramParametersWithUniforms, type WebGLRenderer } from "three";

/** Mean linear colour of bark_basecolor.webp (measured over all texels). */
const BARK_TEXTURE_MEAN = new Vector3(0.261, 0.163, 0.092);
const ANCHOR = "#include <map_fragment>";
let warned = false;

/** Fresh wood where COLOR_0.g < 0.05. */
export interface FreshWood {
  /** linear albedo */
  color: Color;
  /** share of the painted grain's variation kept on it (0 = flat colour) */
  grain: number;
  roughness: number;
}

/** Moss stain where COLOR_0.R is high: albedo × `darken` at R ≥ `to`, none below `from`. */
export interface MossStain {
  darken: number;
  from: number;
  to: number;
}

/** Darkening along the grain (TEXCOORD_0.v): albedo × `darken` at v = `vFull`, none at v = `vNone`. */
export interface GrainFade {
  vFull: number;
  vNone: number;
  darken: number;
}

export interface BarkTone {
  /** share of the painted grain kept (0–1), around the texture's mean colour */
  contrast: number;
  stain?: MossStain;
  fresh?: FreshWood;
  fade?: GrainFade;
}

export function toneBark(material: MeshStandardMaterial, tone: BarkTone): void {
  const { contrast, stain, fresh, fade } = tone;
  const uniforms = {
    uFade: { value: new Vector3(fade?.vNone ?? 0, (fade?.vFull ?? 1) - (fade?.vNone ?? 0) || 1e-3, fade?.darken ?? 1) },
    uBarkMean: { value: BARK_TEXTURE_MEAN.clone() },
    uBarkContrast: { value: contrast },
    uStain: { value: new Vector3(stain?.darken ?? 1, stain?.from ?? 0, Math.max((stain?.to ?? 1) - (stain?.from ?? 0), 1e-3)) },
    uFreshColor: { value: (fresh?.color ?? new Color(1, 1, 1)).clone() },
    uFreshGrain: { value: fresh?.grain ?? 0 },
    uFreshRoughness: { value: fresh?.roughness ?? 1 },
  };
  const usesColor = !!(fresh || stain);
  const base = material.onBeforeCompile;
  material.onBeforeCompile = (shader: WebGLProgramParametersWithUniforms, renderer: WebGLRenderer) => {
    base.call(material, shader, renderer);
    if (!shader.fragmentShader.includes(ANCHOR)) {
      if (!warned) console.warn("[nature] branch bark: map_fragment chunk not found, grain contrast unchanged");
      warned = true;
      return;
    }
    Object.assign(shader.uniforms, uniforms);
    if (usesColor) {
      // COLOR_0 is declared by the factory's AO patch (Materials.ts) when vertexColors is off
      shader.vertexShader = shader.vertexShader
        .replace("#include <common>", ["#include <common>", "varying vec2 vBarkRG;"].join("\n"))
        .replace("#include <color_vertex>", ["#include <color_vertex>", "  vBarkRG = color.rg;"].join("\n"));
    }
    if (fade) {
      // the raw TEXCOORD_0 (three declares `uv` for every program)
      shader.vertexShader = shader.vertexShader
        .replace("#include <common>", ["#include <common>", "varying float vBarkV;"].join("\n"))
        .replace("#include <begin_vertex>", ["#include <begin_vertex>", "  vBarkV = uv.y;"].join("\n"));
    }
    const pars = [
      "#include <common>",
      usesColor ? "varying vec2 vBarkRG;" : "",
      fade ? "#define BARK_FADE\nvarying float vBarkV;\nuniform vec3 uFade;" : "",
      stain ? "#define BARK_STAIN" : "",
      fresh ? "#define BARK_FRESH" : "",
      "uniform vec3 uBarkMean;",
      "uniform float uBarkContrast;",
      "uniform vec3 uStain;",
      "uniform vec3 uFreshColor;",
      "uniform float uFreshGrain;",
      "uniform float uFreshRoughness;",
    ];
    const albedo = [
      ANCHOR,
      "  diffuseColor.rgb = mix(diffuse * uBarkMean, diffuseColor.rgb, uBarkContrast);",
      "  #ifdef BARK_STAIN",
      "    diffuseColor.rgb *= mix(1.0, uStain.x, clamp((vBarkRG.x - uStain.y) / uStain.z, 0.0, 1.0));",
      "  #endif",
      "  #ifdef BARK_FADE",
      "    diffuseColor.rgb *= mix(1.0, uFade.z, smoothstep(0.0, 1.0, clamp((vBarkV - uFade.x) / uFade.y, 0.0, 1.0)));",
      "  #endif",
      "  float barkFresh = 0.0;",
      "  #ifdef BARK_FRESH",
      "    barkFresh = 1.0 - smoothstep(0.03, 0.07, vBarkRG.y);",
      "    #ifdef USE_MAP",
      "      vec3 freshGrain = mix(vec3(1.0), clamp(sampledDiffuseColor.rgb / uBarkMean, 0.6, 1.6), uFreshGrain);",
      "    #else",
      "      vec3 freshGrain = vec3(1.0);",
      "    #endif",
      "    diffuseColor.rgb = mix(diffuseColor.rgb, uFreshColor * freshGrain, barkFresh);",
      "  #endif",
    ];
    shader.fragmentShader = shader.fragmentShader
      .replace("#include <common>", pars.join("\n"))
      .replace(ANCHOR, albedo.join("\n"))
      .replace("#include <roughnessmap_fragment>", ["#include <roughnessmap_fragment>", "  roughnessFactor = mix(roughnessFactor, uFreshRoughness, barkFresh);"].join("\n"));
  };
  const key = material.customProgramCacheKey.bind(material);
  material.customProgramCacheKey = () => `${key()}|branch-bark-tone${fresh ? "|fresh" : ""}${stain ? "|stain" : ""}${fade ? "|fade" : ""}`;
  material.needsUpdate = true;
}
