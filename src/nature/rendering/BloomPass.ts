/**
 * Bloom on the composited linear HDR image: soft-threshold prefilter, 13-tap
 * downsample chain (Karis average on the first level against fireflies), tent
 * upsample with per-level weights driven by `radius`. Skipped entirely when the
 * strength is 0.
 */
import {
  LinearFilter,
  OrthographicCamera,
  RGBAFormat,
  Vector2,
  WebGLRenderTarget,
  type Texture,
  type TextureDataType,
  type WebGLRenderer,
} from "three";
import { FullscreenPass, createPassMaterial } from "./FullscreenPass";

/** Camera for `compile` (fullscreen passes ignore it). */
const COMPILE_CAMERA = new OrthographicCamera(-1, 1, 1, -1, 0, 1);

const DOWNSAMPLE = /* glsl */ `
precision highp float;
precision highp sampler2D;
uniform sampler2D tSrc;
uniform vec2 uSrcTexel;
uniform float uPrefilter;
uniform float uThreshold;
uniform float uKnee;
in vec2 vUv;
out highp vec4 outColor;
float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
vec3 threshold(vec3 c) {
  float br = max(c.r, max(c.g, c.b));
  float soft = clamp(br - uThreshold + uKnee, 0.0, 2.0 * uKnee);
  soft = soft * soft / (4.0 * uKnee + 1e-4);
  float contrib = max(soft, br - uThreshold) / max(br, 1e-4);
  return c * contrib;
}
void main() {
  vec2 t = uSrcTexel;
  vec3 a = texture(tSrc, vUv + t * vec2(-2.0, 2.0)).rgb;
  vec3 b = texture(tSrc, vUv + t * vec2(0.0, 2.0)).rgb;
  vec3 c = texture(tSrc, vUv + t * vec2(2.0, 2.0)).rgb;
  vec3 d = texture(tSrc, vUv + t * vec2(-2.0, 0.0)).rgb;
  vec3 e = texture(tSrc, vUv).rgb;
  vec3 f = texture(tSrc, vUv + t * vec2(2.0, 0.0)).rgb;
  vec3 g = texture(tSrc, vUv + t * vec2(-2.0, -2.0)).rgb;
  vec3 h = texture(tSrc, vUv + t * vec2(0.0, -2.0)).rgb;
  vec3 i = texture(tSrc, vUv + t * vec2(2.0, -2.0)).rgb;
  vec3 j = texture(tSrc, vUv + t * vec2(-1.0, 1.0)).rgb;
  vec3 k = texture(tSrc, vUv + t * vec2(1.0, 1.0)).rgb;
  vec3 l = texture(tSrc, vUv + t * vec2(-1.0, -1.0)).rgb;
  vec3 m = texture(tSrc, vUv + t * vec2(1.0, -1.0)).rgb;
  vec3 res;
  if (uPrefilter > 0.5) {
    // Karis average of the five 2×2 groups: luminance-weighted, kills single bright pixels
    vec3 g0 = (a + b + d + e) * 0.25;
    vec3 g1 = (b + c + e + f) * 0.25;
    vec3 g2 = (d + e + g + h) * 0.25;
    vec3 g3 = (e + f + h + i) * 0.25;
    vec3 g4 = (j + k + l + m) * 0.25;
    float w0 = 0.125 / (1.0 + luma(g0));
    float w1 = 0.125 / (1.0 + luma(g1));
    float w2 = 0.125 / (1.0 + luma(g2));
    float w3 = 0.125 / (1.0 + luma(g3));
    float w4 = 0.5 / (1.0 + luma(g4));
    res = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
    res = threshold(max(res, vec3(0.0)));
  } else {
    res = e * 0.125;
    res += (a + c + g + i) * 0.03125;
    res += (b + d + f + h) * 0.0625;
    res += (j + k + l + m) * 0.125;
  }
  outColor = vec4(max(res, vec3(0.0)), 1.0);
}
`;

const UPSAMPLE = /* glsl */ `
precision highp float;
precision highp sampler2D;
uniform sampler2D tLow;
uniform sampler2D tHigh;
uniform vec2 uLowTexel;
uniform float uLowWeight;
uniform float uHighWeight;
in vec2 vUv;
out highp vec4 outColor;
void main() {
  vec2 t = uLowTexel;
  vec3 s = texture(tLow, vUv).rgb * 4.0;
  s += texture(tLow, vUv + vec2(t.x, 0.0)).rgb * 2.0;
  s += texture(tLow, vUv - vec2(t.x, 0.0)).rgb * 2.0;
  s += texture(tLow, vUv + vec2(0.0, t.y)).rgb * 2.0;
  s += texture(tLow, vUv - vec2(0.0, t.y)).rgb * 2.0;
  s += texture(tLow, vUv + t).rgb;
  s += texture(tLow, vUv - t).rgb;
  s += texture(tLow, vUv + vec2(t.x, -t.y)).rgb;
  s += texture(tLow, vUv + vec2(-t.x, t.y)).rgb;
  vec3 up = s / 16.0;
  outColor = vec4(up * uLowWeight + texture(tHigh, vUv).rgb * uHighWeight, 1.0);
}
`;

const BASE_FACTORS = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1];

export class BloomPass {
  private mips: WebGLRenderTarget[] = [];
  private ups: WebGLRenderTarget[] = [];
  private width = 1;
  private height = 1;
  private levels: number;

  private readonly down = new FullscreenPass(
    createPassMaterial({
      name: "BloomDown",
      fragmentShader: DOWNSAMPLE,
      uniforms: {
        tSrc: { value: null },
        uSrcTexel: { value: new Vector2() },
        uPrefilter: { value: 0 },
        uThreshold: { value: 1 },
        uKnee: { value: 0.5 },
      },
    }),
  );
  private readonly up = new FullscreenPass(
    createPassMaterial({
      name: "BloomUp",
      fragmentShader: UPSAMPLE,
      uniforms: {
        tLow: { value: null },
        tHigh: { value: null },
        uLowTexel: { value: new Vector2() },
        uLowWeight: { value: 1 },
        uHighWeight: { value: 1 },
      },
    }),
  );

  constructor(
    private readonly type: TextureDataType,
    levels: number,
  ) {
    this.levels = Math.max(2, Math.min(6, levels));
    this.allocate();
  }

  private allocate(): void {
    for (const rt of [...this.mips, ...this.ups]) rt.dispose();
    this.mips = [];
    this.ups = [];
    let w = this.width;
    let h = this.height;
    for (let i = 0; i < this.levels; i++) {
      w = Math.max(1, Math.ceil(w / 2));
      h = Math.max(1, Math.ceil(h / 2));
      const opts = { type: this.type, format: RGBAFormat, depthBuffer: false, minFilter: LinearFilter, magFilter: LinearFilter, generateMipmaps: false };
      this.mips.push(new WebGLRenderTarget(w, h, opts));
      if (i < this.levels - 1) this.ups.push(new WebGLRenderTarget(w, h, opts));
    }
  }

  setLevels(levels: number): void {
    const l = Math.max(2, Math.min(6, levels));
    if (l !== this.levels) {
      this.levels = l;
      this.allocate();
    }
  }

  setSize(width: number, height: number): void {
    if (width === this.width && height === this.height) return;
    this.width = width;
    this.height = height;
    this.allocate();
  }

  /** Returns the bloom texture (half resolution) for `input`. */
  render(renderer: WebGLRenderer, input: Texture, radius: number, threshold: number): Texture {
    const du = this.down.material.uniforms;
    let src: Texture = input;
    let srcW = this.width;
    let srcH = this.height;
    for (let i = 0; i < this.levels; i++) {
      du.tSrc.value = src;
      (du.uSrcTexel.value as Vector2).set(1 / srcW, 1 / srcH);
      du.uPrefilter.value = i === 0 ? 1 : 0;
      du.uThreshold.value = threshold;
      du.uKnee.value = Math.max(0.05, threshold * 0.5);
      this.down.render(renderer, this.mips[i]);
      src = this.mips[i].texture;
      srcW = this.mips[i].width;
      srcH = this.mips[i].height;
    }

    const r = Math.min(1, Math.max(0, radius));
    const weight = (i: number) => {
      const f = BASE_FACTORS[Math.min(i, BASE_FACTORS.length - 1)];
      return f + (1.2 - f - f) * r;
    };
    const uu = this.up.material.uniforms;
    let low: WebGLRenderTarget = this.mips[this.levels - 1];
    let lowWeight = weight(this.levels - 1);
    for (let i = this.levels - 2; i >= 0; i--) {
      uu.tLow.value = low.texture;
      uu.tHigh.value = this.mips[i].texture;
      (uu.uLowTexel.value as Vector2).set(1 / low.width, 1 / low.height);
      uu.uLowWeight.value = lowWeight;
      uu.uHighWeight.value = weight(i);
      this.up.render(renderer, this.ups[i]);
      low = this.ups[i];
      lowWeight = 1;
    }
    return this.ups[0].texture;
  }

  /**
   * Compile the downsample and upsample programs ahead of their first use, in parallel
   * where the browser can: looks without bloom skip the pass, so the first bloomed frame
   * would otherwise stall on two synchronous compiles.
   */
  compile(renderer: WebGLRenderer): Promise<unknown> {
    return Promise.all([this.down, this.up].map((p) => renderer.compileAsync(p.mesh, COMPILE_CAMERA)));
  }

  dispose(): void {
    for (const rt of [...this.mips, ...this.ups]) rt.dispose();
    this.down.dispose();
    this.up.dispose();
  }
}
