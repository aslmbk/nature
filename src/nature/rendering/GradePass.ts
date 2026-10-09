/**
 * Final pass to the canvas: + bloom, top/bottom darkening bands, soft corner
 * vignette, then tone mapping and the sRGB transfer — exactly once — and output
 * dithering against banding.
 */
import {
  ACESFilmicToneMapping,
  AgXToneMapping,
  ColorManagement,
  Mesh,
  NeutralToneMapping,
  OrthographicCamera,
  SRGBTransfer,
  Vector2,
  Vector4,
  type ColorSpace,
  type RawShaderMaterial,
  type Texture,
  type ToneMapping,
  type WebGLRenderer,
} from "three";
import type { LookParams } from "../types";
import { FullscreenPass, GLSL_COMMON, createPassMaterial } from "./FullscreenPass";

const FRAGMENT = /* glsl */ `
precision highp float;
precision highp sampler2D;
uniform sampler2D tColor;
uniform sampler2D tBloom;
uniform float uBloomStrength;
uniform vec4 uVignette;      // top, bottom, corners, -
uniform vec2 uVignetteSize;  // topSize, bottomSize
uniform float uAspect;
in vec2 vUv;
out highp vec4 outColor;
${GLSL_COMMON}
#include <tonemapping_pars_fragment>
#include <colorspace_pars_fragment>
void main() {
  vec3 c = texture(tColor, vUv).rgb;
#ifdef USE_BLOOM
  c += texture(tBloom, vUv).rgb * uBloomStrength;
#endif
#ifdef USE_VIGNETTE
  float topBand = 1.0 - smoothstep(0.0, max(uVignetteSize.x, 1e-4), 1.0 - vUv.y);
  float bottomBand = 1.0 - smoothstep(0.0, max(uVignetteSize.y, 1e-4), vUv.y);
  vec2 d = (vUv - 0.5) * vec2(uAspect / 1.41, 1.0) * 2.0;
  float corner = smoothstep(0.75, 1.65, length(d));
  float shade = (1.0 - uVignette.x * topBand) * (1.0 - uVignette.y * bottomBand) * (1.0 - uVignette.z * corner);
  c *= max(shade, 0.0);
#endif
  vec4 color = vec4(max(c, vec3(0.0)), 1.0);
#if defined(ACES_FILMIC_TONE_MAPPING)
  color.rgb = ACESFilmicToneMapping(color.rgb);
#elif defined(AGX_TONE_MAPPING)
  color.rgb = AgXToneMapping(color.rgb);
#elif defined(NEUTRAL_TONE_MAPPING)
  color.rgb = NeutralToneMapping(color.rgb);
#else
  color.rgb = clamp(color.rgb, 0.0, 1.0);
#endif
#ifdef SRGB_TRANSFER
  color = sRGBTransferOETF(color);
#endif
  color.rgb += (ign(gl_FragCoord.xy) - 0.5) / 255.0;
  outColor = color;
}
`;

export interface GradeOptions {
  bloom: Texture | null;
  bloomStrength: number;
  vignette: boolean;
  /**
   * The composite already carries each view's own vignette (wipe / slide): the bands and
   * corners stay out here (same program, zero strength).
   */
  vignetteInSource?: boolean;
  toneMapping: ToneMapping;
  outputColorSpace: ColorSpace;
}

/** Camera for `compile` (fullscreen passes ignore it). */
const COMPILE_CAMERA = new OrthographicCamera(-1, 1, 1, -1, 0, 1);

/**
 * One material per variant (tone mapping × sRGB × bloom × vignette), all sharing one
 * uniforms object, so a variant that a later frame needs (the first look with bloom)
 * can be compiled ahead of time and stays alive; switching variants is a material
 * swap, never a recompile.
 */
export class GradePass {
  private readonly uniforms = {
    tColor: { value: null as Texture | null },
    tBloom: { value: null as Texture | null },
    uBloomStrength: { value: 0 },
    uVignette: { value: new Vector4() },
    uVignetteSize: { value: new Vector2(0.4, 0.3) },
    uAspect: { value: 1 },
    toneMappingExposure: { value: 1 },
  };
  private readonly variants = new Map<string, RawShaderMaterial>();
  private readonly pass: FullscreenPass;

  constructor() {
    this.pass = new FullscreenPass(this.variant(ACESFilmicToneMapping, true, false, true));
  }

  private variant(toneMapping: ToneMapping, srgb: boolean, bloom: boolean, vignette: boolean): RawShaderMaterial {
    const key = `${toneMapping}|${srgb}|${bloom ? 1 : 0}|${vignette ? 1 : 0}`;
    let m = this.variants.get(key);
    if (!m) {
      const defines: Record<string, string> = {};
      if (srgb) defines.SRGB_TRANSFER = "";
      if (toneMapping === ACESFilmicToneMapping) defines.ACES_FILMIC_TONE_MAPPING = "";
      else if (toneMapping === AgXToneMapping) defines.AGX_TONE_MAPPING = "";
      else if (toneMapping === NeutralToneMapping) defines.NEUTRAL_TONE_MAPPING = "";
      if (bloom) defines.USE_BLOOM = "";
      if (vignette) defines.USE_VIGNETTE = "";
      m = createPassMaterial({ name: "SilvaGrade", fragmentShader: FRAGMENT, uniforms: this.uniforms, defines });
      this.variants.set(key, m);
    }
    return m;
  }

  render(renderer: WebGLRenderer, color: Texture, look: LookParams, opts: GradeOptions, width: number, height: number): void {
    const u = this.uniforms;
    u.tColor.value = color;
    u.tBloom.value = opts.bloom;
    u.uBloomStrength.value = opts.bloomStrength;
    const v = look.vignette;
    if (opts.vignetteInSource) u.uVignette.value.set(0, 0, 0, 0);
    else u.uVignette.value.set(v.top, v.bottom, v.corners, 0);
    u.uVignetteSize.value.set(v.topSize, v.bottomSize);
    u.uAspect.value = width / Math.max(1, height);
    u.toneMappingExposure.value = 1;

    const srgb = ColorManagement.getTransfer(opts.outputColorSpace) === SRGBTransfer;
    this.pass.mesh.material = this.variant(opts.toneMapping, srgb, opts.bloom !== null, opts.vignette);
    this.pass.render(renderer, null);
  }

  /**
   * Compile, in parallel where the browser can, the variants a session will need for
   * these output settings: without and with bloom (the first look with bloom would
   * otherwise stall on a synchronous compile in the middle of the story).
   */
  compile(renderer: WebGLRenderer, toneMapping: ToneMapping, outputColorSpace: ColorSpace, vignette: boolean): Promise<unknown> {
    const srgb = ColorManagement.getTransfer(outputColorSpace) === SRGBTransfer;
    const jobs = [false, true].map((bloom) => {
      const mesh = new Mesh(this.pass.mesh.geometry, this.variant(toneMapping, srgb, bloom, vignette));
      mesh.frustumCulled = false;
      return renderer.compileAsync(mesh, COMPILE_CAMERA);
    });
    return Promise.all(jobs);
  }

  /** The variant materials (warm-up bookkeeping: their programs are touched once linked). */
  materials(): RawShaderMaterial[] {
    return [...this.variants.values()];
  }

  dispose(): void {
    for (const m of this.variants.values()) m.dispose();
    this.variants.clear();
    this.pass.dispose();
  }
}
