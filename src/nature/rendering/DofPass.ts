/**
 * Near-field depth of field (foreground only, mid-ground stays sharp).
 *
 * Pipeline (half resolution unless noted):
 *  1. downsample colour (premultiplied RGBA) and compute the near circle of confusion
 *     from the scene depth (max of each 2×2 block, so thin near leaves survive);
 *  2. tile max of the CoC at 1/8 resolution, then a separable max dilation, so pixels
 *     next to a blurred foreground know how far that blur reaches;
 *  3. scatter-as-gather: each pixel collects samples whose own CoC covers the distance
 *     to it. Background pixels near a blurred foreground therefore receive the
 *     foreground's blur → no hard halo of sharp pixels around it. The result is a
 *     premultiplied "near layer" whose alpha is the coverage;
 *  4. small tent filter on the near layer;
 *  5. full resolution: result = near + sharp × (1 − near.a).
 * The depth comes from the same render as the colour, so wind-deformed geometry is
 * handled for free.
 */
import {
  HalfFloatType,
  LinearFilter,
  NearestFilter,
  OrthographicCamera,
  RGBAFormat,
  RedFormat,
  Vector2,
  WebGLRenderTarget,
  type DepthTexture,
  type PerspectiveCamera,
  type Texture,
  type TextureDataType,
  type WebGLRenderer,
} from "three";
import type { LookParams } from "../types";
import { FullscreenPass, GLSL_COMMON, createPassMaterial } from "./FullscreenPass";
import { REFERENCE_HEIGHT } from "../SceneConfig";

const TILE = 4; // half-res texels per tile → 1/8 of full resolution
const MAX_DILATE_TAPS = 16;
// ivec2 uniform values of the two dilation passes (constant: no per-frame arrays)
const AXIS_X: [number, number] = [1, 0];
const AXIS_Y: [number, number] = [0, 1];
/** Camera for `compile` (fullscreen passes ignore it). */
const COMPILE_CAMERA = new OrthographicCamera(-1, 1, 1, -1, 0, 1);

const LINEARIZE = /* glsl */ `
uniform float uNear;
uniform float uFar;
float viewDistance(float depth) {
  // perspective depth [0,1] → positive view-space distance
  return (uNear * uFar) / ((uFar - uNear) * depth - uFar) * -1.0;
}
`;

const DOWNSAMPLE_COLOR = /* glsl */ `
precision highp float;
precision highp sampler2D;
uniform sampler2D tColor;
in vec2 vUv;
out highp vec4 outColor;
void main() {
  // bilinear tap at the shared corner of a 2×2 block = box average
  outColor = texture(tColor, vUv);
}
`;

const COC = /* glsl */ `
precision highp float;
precision highp sampler2D;
uniform sampler2D tDepth;
uniform vec2 uFullSize;
uniform float uNearStart;
uniform float uNearEnd;
uniform float uMaxCoc;
${LINEARIZE}
in vec2 vUv;
out highp vec4 outColor;
float cocAt(ivec2 p) {
  p = clamp(p, ivec2(0), ivec2(uFullSize) - 1);
  float d = viewDistance(texelFetch(tDepth, p, 0).r);
  return uMaxCoc * clamp((uNearEnd - d) / max(uNearEnd - uNearStart, 1e-4), 0.0, 1.0);
}
void main() {
  ivec2 base = ivec2(gl_FragCoord.xy) * 2;
  float c = max(max(cocAt(base), cocAt(base + ivec2(1, 0))), max(cocAt(base + ivec2(0, 1)), cocAt(base + ivec2(1, 1))));
  outColor = vec4(c, 0.0, 0.0, 1.0);
}
`;

const TILE_MAX = /* glsl */ `
precision highp float;
precision highp sampler2D;
uniform sampler2D tCoc;
uniform vec2 uSrcSize;
in vec2 vUv;
out highp vec4 outColor;
void main() {
  ivec2 base = ivec2(gl_FragCoord.xy) * ${TILE};
  ivec2 lim = ivec2(uSrcSize) - 1;
  float m = 0.0;
  for (int y = 0; y < ${TILE}; y++) {
    for (int x = 0; x < ${TILE}; x++) {
      m = max(m, texelFetch(tCoc, min(base + ivec2(x, y), lim), 0).r);
    }
  }
  outColor = vec4(m, 0.0, 0.0, 1.0);
}
`;

const DILATE = /* glsl */ `
precision highp float;
precision highp sampler2D;
uniform sampler2D tTile;
uniform vec2 uTileSize;
uniform ivec2 uAxis;
uniform int uRadius;
in vec2 vUv;
out highp vec4 outColor;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  ivec2 lim = ivec2(uTileSize) - 1;
  float m = texelFetch(tTile, p, 0).r;
  for (int i = 1; i <= ${MAX_DILATE_TAPS}; i++) {
    if (i > uRadius) break;
    m = max(m, texelFetch(tTile, clamp(p + uAxis * i, ivec2(0), lim), 0).r);
    m = max(m, texelFetch(tTile, clamp(p - uAxis * i, ivec2(0), lim), 0).r);
  }
  outColor = vec4(m, 0.0, 0.0, 1.0);
}
`;

const GATHER = /* glsl */ `
precision highp float;
precision highp sampler2D;
uniform sampler2D tColorHalf;
uniform sampler2D tCoc;
uniform sampler2D tDilated;
uniform vec2 uFullSize;
uniform int uSamples;
in vec2 vUv;
out highp vec4 outColor;
${GLSL_COMMON}
const float GOLDEN = 2.39996323;
void main() {
  float own = texture(tCoc, vUv).r;
  float reach = max(own, texture(tDilated, vUv).r);
  if (reach < 0.5) { outColor = vec4(0.0); return; }
  vec2 pxToUv = 1.0 / uFullSize;
  float rot = ign(gl_FragCoord.xy) * 6.2831853;
  vec4 acc = vec4(0.0);
  float wsum = 0.0;
  float covered = 0.0;
  // centre sample
  float wc = step(0.5, own);
  acc += texture(tColorHalf, vUv) * wc;
  wsum += wc;
  covered += wc;
  float n = float(uSamples);
  for (int i = 0; i < 64; i++) {
    if (i >= uSamples) break;
    float fi = float(i) + 0.5;
    float r = sqrt(fi / n) * reach;
    float a = fi * GOLDEN + rot;
    vec2 off = vec2(cos(a), sin(a)) * r;
    vec2 uv = vUv + off * pxToUv;
    float coc = texture(tCoc, uv).r;
    // the sample's own blur disk must reach this pixel
    float w = step(0.5, coc) * clamp(coc - r + 1.0, 0.0, 1.0);
    acc += texture(tColorHalf, uv) * w;
    wsum += w;
    covered += w;
  }
  vec4 nearColor = acc / max(wsum, 1e-4);
  float coverage = covered / (n + 1.0);
  float alpha = max(clamp(coverage * 2.0, 0.0, 1.0), smoothstep(1.0, 3.0, own));
  outColor = nearColor * alpha;
}
`;

const TENT = /* glsl */ `
precision highp float;
precision highp sampler2D;
uniform sampler2D tNear;
uniform vec2 uTexel;
in vec2 vUv;
out highp vec4 outColor;
void main() {
  vec4 s = texture(tNear, vUv) * 4.0;
  s += texture(tNear, vUv + vec2(uTexel.x, 0.0)) * 2.0;
  s += texture(tNear, vUv - vec2(uTexel.x, 0.0)) * 2.0;
  s += texture(tNear, vUv + vec2(0.0, uTexel.y)) * 2.0;
  s += texture(tNear, vUv - vec2(0.0, uTexel.y)) * 2.0;
  s += texture(tNear, vUv + uTexel);
  s += texture(tNear, vUv - uTexel);
  s += texture(tNear, vUv + vec2(uTexel.x, -uTexel.y));
  s += texture(tNear, vUv + vec2(-uTexel.x, uTexel.y));
  outColor = s / 16.0;
}
`;

const COMPOSITE = /* glsl */ `
precision highp float;
precision highp sampler2D;
uniform sampler2D tSharp;
uniform sampler2D tNear;
in vec2 vUv;
out highp vec4 outColor;
void main() {
  vec4 sharp = texture(tSharp, vUv);
  vec4 near = texture(tNear, vUv);
  outColor = near + sharp * (1.0 - near.a);
}
`;

function makeTarget(w: number, h: number, type: TextureDataType, red = false, linear = true): WebGLRenderTarget {
  const rt = new WebGLRenderTarget(Math.max(1, w), Math.max(1, h), {
    type,
    format: red ? RedFormat : RGBAFormat,
    depthBuffer: false,
    stencilBuffer: false,
    generateMipmaps: false,
    minFilter: linear ? LinearFilter : NearestFilter,
    magFilter: linear ? LinearFilter : NearestFilter,
  });
  rt.texture.name = "SilvaDof";
  return rt;
}

export interface DofSettings {
  samples: number;
}

export class DofPass {
  private fullW = 1;
  private fullH = 1;
  private readonly colorHalf: WebGLRenderTarget;
  private readonly coc: WebGLRenderTarget;
  private readonly tileA: WebGLRenderTarget;
  private readonly tileB: WebGLRenderTarget;
  private readonly nearA: WebGLRenderTarget;
  private readonly nearB: WebGLRenderTarget;

  private readonly downsample = new FullscreenPass(
    createPassMaterial({ name: "DofDownsample", fragmentShader: DOWNSAMPLE_COLOR, uniforms: { tColor: { value: null } } }),
  );
  private readonly cocPass = new FullscreenPass(
    createPassMaterial({
      name: "DofCoc",
      fragmentShader: COC,
      uniforms: {
        tDepth: { value: null },
        uFullSize: { value: new Vector2() },
        uNear: { value: 0.1 },
        uFar: { value: 100 },
        uNearStart: { value: 0 },
        uNearEnd: { value: 1 },
        uMaxCoc: { value: 0 },
      },
    }),
  );
  private readonly tilePass = new FullscreenPass(
    createPassMaterial({ name: "DofTileMax", fragmentShader: TILE_MAX, uniforms: { tCoc: { value: null }, uSrcSize: { value: new Vector2() } } }),
  );
  private readonly dilatePass = new FullscreenPass(
    createPassMaterial({
      name: "DofDilate",
      fragmentShader: DILATE,
      uniforms: { tTile: { value: null }, uTileSize: { value: new Vector2() }, uAxis: { value: AXIS_X }, uRadius: { value: 1 } },
    }),
  );
  private readonly gatherPass = new FullscreenPass(
    createPassMaterial({
      name: "DofGather",
      fragmentShader: GATHER,
      uniforms: {
        tColorHalf: { value: null },
        tCoc: { value: null },
        tDilated: { value: null },
        uFullSize: { value: new Vector2() },
        uSamples: { value: 32 },
      },
    }),
  );
  private readonly tentPass = new FullscreenPass(
    createPassMaterial({ name: "DofTent", fragmentShader: TENT, uniforms: { tNear: { value: null }, uTexel: { value: new Vector2() } } }),
  );
  private readonly compositePass = new FullscreenPass(
    createPassMaterial({ name: "DofComposite", fragmentShader: COMPOSITE, uniforms: { tSharp: { value: null }, tNear: { value: null } } }),
  );

  constructor(private readonly type: TextureDataType) {
    this.colorHalf = makeTarget(1, 1, type);
    this.coc = makeTarget(1, 1, HalfFloatType, true, false);
    this.tileA = makeTarget(1, 1, HalfFloatType, true, true);
    this.tileB = makeTarget(1, 1, HalfFloatType, true, true);
    this.nearA = makeTarget(1, 1, type);
    this.nearB = makeTarget(1, 1, type);
  }

  setSize(width: number, height: number): void {
    this.fullW = width;
    this.fullH = height;
    const hw = Math.max(1, Math.ceil(width / 2));
    const hh = Math.max(1, Math.ceil(height / 2));
    const tw = Math.max(1, Math.ceil(hw / TILE));
    const th = Math.max(1, Math.ceil(hh / TILE));
    this.colorHalf.setSize(hw, hh);
    this.coc.setSize(hw, hh);
    this.nearA.setSize(hw, hh);
    this.nearB.setSize(hw, hh);
    this.tileA.setSize(tw, th);
    this.tileB.setSize(tw, th);
  }

  /** Blur radius in drawing-buffer pixels for a look (0 ⇒ the pass is skipped). */
  maxCocPx(look: LookParams): number {
    return Math.max(0, look.dof.maxBlurPx) * (this.fullH / REFERENCE_HEIGHT);
  }

  /**
   * Render the DOF result of `color`/`depth` into `output` (full resolution).
   * Returns false when nothing needed blurring (caller keeps the sharp image).
   */
  render(
    renderer: WebGLRenderer,
    color: Texture,
    depth: DepthTexture,
    camera: PerspectiveCamera,
    look: LookParams,
    settings: DofSettings,
    output: WebGLRenderTarget,
  ): boolean {
    const maxCoc = this.maxCocPx(look);
    if (maxCoc < 0.75 || settings.samples <= 0) return false;
    const hw = this.colorHalf.width;
    const hh = this.colorHalf.height;

    // 1. colour + CoC at half resolution
    this.downsample.material.uniforms.tColor.value = color;
    this.downsample.render(renderer, this.colorHalf);

    const cu = this.cocPass.material.uniforms;
    cu.tDepth.value = depth;
    (cu.uFullSize.value as Vector2).set(this.fullW, this.fullH);
    cu.uNear.value = camera.near;
    cu.uFar.value = camera.far;
    cu.uNearStart.value = Math.min(look.dof.nearStart, look.dof.nearEnd - 1e-3);
    cu.uNearEnd.value = look.dof.nearEnd;
    cu.uMaxCoc.value = maxCoc;
    this.cocPass.render(renderer, this.coc);

    // 2. tile max + separable dilation (radius in tiles, CoC is in full-res px)
    const tu = this.tilePass.material.uniforms;
    tu.tCoc.value = this.coc.texture;
    (tu.uSrcSize.value as Vector2).set(hw, hh);
    this.tilePass.render(renderer, this.tileA);

    const radiusTiles = Math.min(MAX_DILATE_TAPS, Math.ceil(maxCoc / (2 * TILE)) + 1);
    const du = this.dilatePass.material.uniforms;
    (du.uTileSize.value as Vector2).set(this.tileA.width, this.tileA.height);
    du.uRadius.value = radiusTiles;
    du.tTile.value = this.tileA.texture;
    du.uAxis.value = AXIS_X;
    this.dilatePass.render(renderer, this.tileB);
    du.tTile.value = this.tileB.texture;
    du.uAxis.value = AXIS_Y;
    this.dilatePass.render(renderer, this.tileA);

    // 3. gather
    const gu = this.gatherPass.material.uniforms;
    gu.tColorHalf.value = this.colorHalf.texture;
    gu.tCoc.value = this.coc.texture;
    gu.tDilated.value = this.tileA.texture;
    (gu.uFullSize.value as Vector2).set(this.fullW, this.fullH);
    gu.uSamples.value = Math.min(64, settings.samples);
    this.gatherPass.render(renderer, this.nearA);

    // 4. tent
    const tn = this.tentPass.material.uniforms;
    tn.tNear.value = this.nearA.texture;
    (tn.uTexel.value as Vector2).set(1 / hw, 1 / hh);
    this.tentPass.render(renderer, this.nearB);

    // 5. composite at full resolution
    const cp = this.compositePass.material.uniforms;
    cp.tSharp.value = color;
    cp.tNear.value = this.nearB.texture;
    this.compositePass.render(renderer, output);
    return true;
  }

  /**
   * Compile the pass programs ahead of their first use, in parallel where the browser
   * can: a look without near blur skips the pass, so the first look with one would
   * otherwise stall on seven synchronous compiles.
   */
  compile(renderer: WebGLRenderer): Promise<unknown> {
    const passes = [this.downsample, this.cocPass, this.tilePass, this.dilatePass, this.gatherPass, this.tentPass, this.compositePass];
    return Promise.all(passes.map((p) => renderer.compileAsync(p.mesh, COMPILE_CAMERA)));
  }

  /** CoC texture of the last render (debug view). */
  get cocTexture(): Texture {
    return this.coc.texture;
  }

  get textureType(): TextureDataType {
    return this.type;
  }

  dispose(): void {
    for (const rt of [this.colorHalf, this.coc, this.tileA, this.tileB, this.nearA, this.nearB]) rt.dispose();
    for (const p of [this.downsample, this.cocPass, this.tilePass, this.dilatePass, this.gatherPass, this.tentPass, this.compositePass]) p.dispose();
  }
}
