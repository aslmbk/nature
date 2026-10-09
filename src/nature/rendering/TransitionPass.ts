/**
 * Composites one or two linear scene images (before any tone mapping):
 *  - single: A × exposureA
 *  - mix:    crossfade, optional dip through a colour (dark crossfades)
 *  - wipe:   B revealed below a ragged organic edge that travels up (motion/02)
 *  - slide:  A moves up like a page scroll, B follows from below (`follow` 1) or stays in
 *            place under it (`follow` 0), soft dark blurred seam (frame 13)
 *  - over:   premultiplied A (rendered with a transparent clear) over B (motion/04)
 *
 * Per-view vignette (wipe, slide): each picture gets its own look's bands and corners at
 * its own position before the two are combined (a sliding picture takes its lower band
 * along to the seam), and the grade then leaves the vignette out.
 */
import { Color, OrthographicCamera, Vector2, Vector4, type Texture, type WebGLRenderer, type WebGLRenderTarget } from "three";
import type { LookParams, TransitionParams } from "../types";
import { FullscreenPass, GLSL_COMMON, createPassMaterial } from "./FullscreenPass";
import { REFERENCE_HEIGHT } from "../SceneConfig";

const MODE_INDEX = { single: 0, mix: 1, wipe: 2, slide: 3, over: 4 } as const;

const FRAGMENT = /* glsl */ `
precision highp float;
precision highp sampler2D;
uniform sampler2D tA;
uniform sampler2D tB;
uniform int uMode;
uniform float uK;
uniform float uExpA;
uniform float uExpB;
uniform float uAspect;
uniform vec2 uTexel;
// mix
uniform float uDip;
uniform vec3 uDipColor;
// wipe
uniform float uEdge;
uniform float uRagged;
uniform float uSoft;
uniform float uNoiseScale;
uniform float uDirection;
uniform float uEdgeDarken;
// slide
uniform float uOffset;
uniform float uSeamSoft;
uniform float uSeamDark;
uniform float uSeamBlur;
uniform float uSeamWave;
uniform float uFollow;
// over
uniform float uReveal;
uniform float uOpacityA;
uniform vec3 uBackA;
// per-view vignette (wipe, slide): top, bottom, corners, on (0 / 1) and top / bottom band sizes
uniform vec4 uVigA;
uniform vec2 uVigSizeA;
uniform vec4 uVigB;
uniform vec2 uVigSizeB;
in vec2 vUv;
out highp vec4 outColor;
${GLSL_COMMON}

// GradePass's vignette, at a picture's own uv
float vignetteShade(vec2 uv, vec4 v, vec2 size) {
  if (v.w < 0.5) return 1.0;
  float topBand = 1.0 - smoothstep(0.0, max(size.x, 1e-4), 1.0 - uv.y);
  float bottomBand = 1.0 - smoothstep(0.0, max(size.y, 1e-4), uv.y);
  vec2 d = (uv - 0.5) * vec2(uAspect / 1.41, 1.0) * 2.0;
  float corner = smoothstep(0.75, 1.65, length(d));
  return max((1.0 - v.x * topBand) * (1.0 - v.y * bottomBand) * (1.0 - v.z * corner), 0.0);
}

// exposure scales colour only; alpha (coverage of a transparent-cleared A) is kept.
// The targets have no mipmaps: explicit LOD 0 (no derivatives needed inside loops / branches).
vec4 sampleA(vec2 uv) { vec4 t = textureLod(tA, uv, 0.0); return vec4(t.rgb * uExpA, t.a); }
vec4 sampleB(vec2 uv) { vec4 t = textureLod(tB, uv, 0.0); return vec4(t.rgb * uExpB, t.a); }

// 1 + 12 tap disk blur of A (radius in uv-height units); radius 0 = plain sample
vec4 blurA(vec2 uv, float radius) {
  vec4 acc = sampleA(uv);
  if (radius < uTexel.y * 0.75) return acc;
  float n = 1.0;
  float r0 = radius;
  for (int i = 0; i < 12; i++) {
    float fi = float(i) + 0.5;
    float a = fi * 2.39996323;
    float r = sqrt(fi / 12.0) * r0;
    vec2 o = vec2(cos(a) / uAspect, sin(a)) * r;
    acc += sampleA(uv + o);
    n += 1.0;
  }
  return acc / n;
}

void main() {
  vec2 uv = vUv;
  if (uMode == 0) {
    outColor = vec4(sampleA(uv).rgb, 1.0);
    return;
  }
  if (uMode == 1) {
    vec4 c = mix(sampleA(uv), sampleB(uv), uK);
    float d = uDip * sin(3.14159265 * clamp(uK, 0.0, 1.0));
    c.rgb = mix(c.rgb, uDipColor, d);
    outColor = vec4(c.rgb, 1.0);
    return;
  }
  if (uMode == 2) {
    float y = uDirection > 0.0 ? uv.y : 1.0 - uv.y;
    // the ragged shape is attached to the edge and morphs slowly with progress
    vec2 p = vec2(uv.x * uAspect, (y - uEdge) * 1.15) * uNoiseScale + vec2(0.0, uK * 1.7);
    float n = fbm(p + 3.7) * 2.0 - 1.0;
    n += 0.35 * (fbm(p * 3.1 + 11.0) * 2.0 - 1.0);
    float boundary = uEdge + uRagged * n;
    float maskB = 1.0 - smoothstep(boundary - uSoft, boundary + uSoft, y);
    vec4 a = sampleA(uv);
    float lip = 1.0 - smoothstep(0.0, uRagged * 1.6 + 1e-4, y - boundary);
    a.rgb *= 1.0 - uEdgeDarken * lip;
    a.rgb *= vignetteShade(uv, uVigA, uVigSizeA);
    vec4 b = sampleB(uv);
    b.rgb *= vignetteShade(uv, uVigB, uVigSizeB);
    outColor = vec4(mix(a.rgb, b.rgb, maskB), 1.0);
    return;
  }
  if (uMode == 3) {
    float y = uDirection > 0.0 ? uv.y : 1.0 - uv.y;
    float wave = uSeamWave * (fbm(vec2(uv.x * uAspect * 2.2, 5.31)) * 2.0 - 1.0);
    float seam = uOffset + wave;
    vec4 c;
    if (y >= seam) {
      float ya = y - uOffset;
      vec2 uvA = vec2(uv.x, uDirection > 0.0 ? ya : 1.0 - ya);
      float dA = y - seam;
      float edge = smoothstep(0.0, max(uSeamSoft, 1e-4), dA);
      c = blurA(uvA, uSeamBlur * (1.0 - edge));
      c.rgb *= mix(1.0 - uSeamDark, 1.0, edge);
      c.rgb *= vignetteShade(uvA, uVigA, uVigSizeA);
    } else {
      float yb = y + (1.0 - uOffset) * uFollow;
      vec2 uvB = vec2(uv.x, uDirection > 0.0 ? yb : 1.0 - yb);
      float dB = seam - y;
      c = sampleB(uvB);
      c.rgb *= mix(1.0 - uSeamDark, 1.0, smoothstep(0.0, max(uSeamSoft * 0.35, 1e-4), dB));
      c.rgb *= vignetteShade(uvB, uVigB, uVigSizeB);
    }
    outColor = vec4(c.rgb, 1.0);
    return;
  }
  // over
  vec4 a = sampleA(uv) * uOpacityA;
  vec3 behind = mix(uBackA * uExpA, sampleB(uv).rgb, uReveal);
  outColor = vec4(a.rgb + behind * (1.0 - a.a), 1.0);
}
`;

/** Camera for `compile` (fullscreen passes ignore it). */
const COMPILE_CAMERA = new OrthographicCamera(-1, 1, 1, -1, 0, 1);

/** |n| bound of the wipe edge noise (fbm × 2 − 1 plus 0.35 × the same), see FRAGMENT. */
const WIPE_NOISE_MAX = 1.35;

/**
 * Which images a composite actually reads with a non-zero weight for `params`, from the
 * bounds of the shader math over the whole frame (y in 0–1). An image that does not
 * contribute need not be rendered: `render` then binds the other one in its place, whose
 * weight is exactly 0 (mix at k = 0 / 1, a wipe edge off screen, an empty slide region,
 * `over` with opacityA or reveal 0). Never returns both false.
 */
export function transitionInputs(params: TransitionParams, out: { a: boolean; b: boolean }): { a: boolean; b: boolean } {
  let a = true;
  let b = true;
  switch (params.mode) {
    case "mix":
      a = params.k !== 1;
      b = params.k !== 0;
      break;
    case "wipe": {
      const reach = Math.abs(params.raggedness) * WIPE_NOISE_MAX + params.softness;
      if (params.softness > 0) {
        b = params.edge + reach > 0;
        a = params.edge - reach < 1;
      }
      break;
    }
    case "slide": {
      const wave = Math.abs(params.seamWave);
      a = params.offset - wave <= 1;
      b = params.offset + wave > 0;
      break;
    }
    case "over":
      a = params.opacityA !== 0;
      b = params.reveal !== 0;
      break;
  }
  out.a = a || !b;
  out.b = b;
  return out;
}

export class TransitionPass {
  private readonly pass = new FullscreenPass(
    createPassMaterial({
      name: "SilvaTransition",
      fragmentShader: FRAGMENT,
      uniforms: {
        tA: { value: null },
        tB: { value: null },
        uMode: { value: 0 },
        uK: { value: 0 },
        uExpA: { value: 1 },
        uExpB: { value: 1 },
        uAspect: { value: 1 },
        uTexel: { value: new Vector2() },
        uDip: { value: 0 },
        uDipColor: { value: new Color() },
        uEdge: { value: 0 },
        uRagged: { value: 0.08 },
        uSoft: { value: 0.005 },
        uNoiseScale: { value: 5 },
        uDirection: { value: 1 },
        uEdgeDarken: { value: 0.4 },
        uOffset: { value: 0 },
        uSeamSoft: { value: 0.15 },
        uSeamDark: { value: 0.85 },
        uSeamBlur: { value: 0 },
        uSeamWave: { value: 0.02 },
        uFollow: { value: 1 },
        uReveal: { value: 0 },
        uOpacityA: { value: 1 },
        uBackA: { value: new Color() },
        uVigA: { value: new Vector4() },
        uVigSizeA: { value: new Vector2(0.4, 0.3) },
        uVigB: { value: new Vector4() },
        uVigSizeB: { value: new Vector2(0.4, 0.3) },
      },
    }),
  );

  /** The modes whose pictures take their own vignette (`render`'s `vignettes`). */
  static perViewVignette(mode: TransitionParams["mode"]): boolean {
    return mode === "wipe" || mode === "slide";
  }

  render(
    renderer: WebGLRenderer,
    a: Texture,
    exposureA: number,
    b: Texture | null,
    exposureB: number,
    params: TransitionParams | null,
    target: WebGLRenderTarget,
    vignettes: readonly [LookParams["vignette"], LookParams["vignette"]] | null = null,
  ): void {
    const u = this.pass.material.uniforms;
    const vig = vignettes && params && b && TransitionPass.perViewVignette(params.mode) ? vignettes : null;
    for (const [i, key, size] of [
      [0, "uVigA", "uVigSizeA"],
      [1, "uVigB", "uVigSizeB"],
    ] as const) {
      const v = vig ? vig[i] : null;
      (u[key].value as Vector4).set(v ? v.top : 0, v ? v.bottom : 0, v ? v.corners : 0, v ? 1 : 0);
      if (v) (u[size].value as Vector2).set(v.topSize, v.bottomSize);
    }
    u.tA.value = a;
    u.tB.value = b ?? a;
    u.uExpA.value = exposureA;
    u.uExpB.value = exposureB;
    u.uAspect.value = target.width / Math.max(1, target.height);
    (u.uTexel.value as Vector2).set(1 / Math.max(1, target.width), 1 / Math.max(1, target.height));
    if (!b || !params) {
      u.uMode.value = MODE_INDEX.single;
    } else {
      u.uMode.value = MODE_INDEX[params.mode];
      u.uK.value = params.k;
      u.uDip.value = params.dip;
      (u.uDipColor.value as Color).copy(params.dipColor);
      u.uEdge.value = params.edge;
      u.uRagged.value = params.raggedness;
      u.uSoft.value = params.softness;
      u.uNoiseScale.value = params.noiseScale;
      u.uDirection.value = params.direction;
      u.uEdgeDarken.value = params.edgeDarken;
      u.uOffset.value = params.offset;
      u.uSeamSoft.value = params.seamSoftness;
      u.uSeamDark.value = params.seamDarkness;
      u.uSeamBlur.value = params.seamBlurPx / REFERENCE_HEIGHT;
      u.uSeamWave.value = params.seamWave;
      u.uFollow.value = params.follow;
      u.uReveal.value = params.reveal;
      u.uOpacityA.value = params.opacityA;
      (u.uBackA.value as Color).copy(params.backA);
    }
    this.pass.render(renderer, target);
  }

  /**
   * Compile the program ahead of its first use, in parallel where the browser can. A
   * single view at exposure 1 never runs this pass, so without it the first transition
   * frame would stall on the synchronous compile (≈ 0.2 s on D3D11).
   */
  compile(renderer: WebGLRenderer): Promise<unknown> {
    return renderer.compileAsync(this.pass.mesh, COMPILE_CAMERA);
  }

  dispose(): void {
    this.pass.dispose();
  }
}
