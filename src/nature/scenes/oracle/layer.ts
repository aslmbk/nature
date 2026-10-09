/**
 * Blend layer of the oracle set: every blended draw of the set (premultiplied films,
 * additive sprites, additive lines) goes into one single-sample HDR layer, which is
 * then composited into the engine's scene target by a single fullscreen triangle; the
 * engine pass of the set keeps only opaque draws (with its MSAA) and that composite.
 *
 * Why: drawn straight into the engine's multisampled half-float target, the blended
 * draws were not bit-stable between identical frames (headless Edge, ANGLE D3D11,
 * RTX 2060). Read-backs of the resolved target showed thousands of half-float values
 * changing by 1 ULP from frame to frame, all at pixels shared by several primitives of
 * one blended draw (internal edges of the shell mesh, sprite borders), and — rarely —
 * where a depth-tested fullscreen blend (the atmosphere) met the MSAA edges of opaque
 * parts: the order in which the GPU merges those samples is not fixed. Opaque draws,
 * an untested fullscreen blend and the whole set at quality=low (no MSAA) were
 * bit-stable. A single-sample layer has no per-pixel sample merging, and the composite
 * (no depth test) covers every sample of every pixel with one primitive.
 *
 * Same picture: each blended draw is an affine map  dst → u + v·dst  (additive: v = 1,
 * premultiplied over: v = 1 − a). Drawn in the same order over the layer's transparent
 * clear they accumulate (U, alpha = 1 − V) — additive draws leave the alpha untouched —
 * and the composite applies U + V·dst once (ONE, ONE_MINUS_SRC_ALPHA). The opaque parts
 * of the set are drawn into the layer first, depth only, so they still hide what is
 * behind them; their colour stays in the engine pass (with its MSAA).
 *
 * The layer is drawn from the set's `animate()`, i.e. right before the engine renders
 * that view (the engine updates and renders views one after the other), with the same
 * camera and the same debug layer mask.
 */
import {
  AddEquation,
  BufferGeometry,
  Color,
  CustomBlending,
  Float32BufferAttribute,
  HalfFloatType,
  LinearSRGBColorSpace,
  Mesh,
  NearestFilter,
  OneFactor,
  OneMinusSrcAlphaFactor,
  RGBAFormat,
  Scene,
  ShaderMaterial,
  UnsignedByteType,
  Vector2,
  WebGLRenderTarget,
  ZeroFactor,
  type Material,
  type PerspectiveCamera,
  type Texture,
  type WebGLRenderer,
} from "three";
import { disposeObject } from "../../core/dispose";
import type { WarmTarget } from "../../types";
import { OUTPUT_TAIL } from "./glsl";

/** Additive colour that leaves the layer alpha (coverage) untouched. */
export function additiveKeepAlpha<T extends Material>(m: T): T {
  m.blending = CustomBlending;
  m.blendEquation = AddEquation;
  m.blendSrc = OneFactor;
  m.blendDst = OneFactor;
  m.blendEquationAlpha = AddEquation;
  m.blendSrcAlpha = ZeroFactor;
  m.blendDstAlpha = OneFactor;
  return m;
}

const COMPOSITE_VERTEX = /* glsl */ `
void main() {
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

const COMPOSITE_FRAGMENT = /* glsl */ `
uniform sampler2D uLayer;
uniform vec2 uInvSize;
void main() {
  // same size as the target: pixel centres hit texel centres exactly (nearest)
  gl_FragColor = texture2D(uLayer, gl_FragCoord.xy * uInvSize);
  ${OUTPUT_TAIL}
}
`;

export class BlendLayer {
  /** Blended content of the set. Never rendered by the engine, only into the layer. */
  readonly scene = new Scene();
  /** Fullscreen composite of the layer; belongs to the set's main scene, drawn last. */
  readonly composite: Mesh<BufferGeometry, ShaderMaterial>;
  private target: WebGLRenderTarget | null = null;
  private readonly size = new Vector2();
  private readonly clearColor = new Color();
  /** Opaque materials of the main scene: depth only inside the layer. */
  private readonly occluders: Material[] = [];

  constructor(private readonly renderer: WebGLRenderer) {
    const geometry = new BufferGeometry();
    geometry.setAttribute("position", new Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));
    const material = new ShaderMaterial({
      name: "OracleLayerComposite",
      uniforms: {
        uLayer: { value: null as Texture | null },
        uInvSize: { value: new Vector2(1, 1) },
      },
      vertexShader: COMPOSITE_VERTEX,
      fragmentShader: COMPOSITE_FRAGMENT,
      transparent: true,
      depthTest: false,
      depthWrite: false,
      blending: CustomBlending,
      blendEquation: AddEquation,
      blendSrc: OneFactor,
      blendDst: OneMinusSrcAlphaFactor,
      blendEquationAlpha: AddEquation,
      blendSrcAlpha: OneFactor,
      blendDstAlpha: OneMinusSrcAlphaFactor,
    });
    this.composite = new Mesh(geometry, material);
    this.composite.name = "oracle_layer_composite";
    this.composite.frustumCulled = false;
    this.composite.renderOrder = 1e8;
  }

  /** Opaque materials of the main scene: they hide layer content behind them. */
  addOccluders(...materials: Material[]): void {
    this.occluders.push(...materials);
  }

  /**
   * The layer's own scene for the engine's warm-up (`NatureScene.warmTargets`): the engine
   * never renders it, so without this its programs would be compiled and linked
   * synchronously on the set's first draw. Drawn into a render target (linear, no tone
   * mapping), no fog, no lights — the same programs `render` uses.
   */
  warmTarget(): WarmTarget {
    return { scene: this.scene, offscreen: true };
  }

  /** Draw the layer for this view. Call after every uniform and transform of the frame is set. */
  render(main: Scene, camera: PerspectiveCamera, layerMask: number): void {
    const r = this.renderer;
    r.getDrawingBufferSize(this.size);
    const w = Math.max(1, Math.round(this.size.x));
    const h = Math.max(1, Math.round(this.size.y));
    const target = this.ensureTarget(w, h);
    const cm = this.composite.material;
    cm.uniforms.uInvSize.value.set(1 / w, 1 / h);
    cm.wireframe = false; // the debug wireframe toggle would leave only the triangle's edges

    const prevTarget = r.getRenderTarget();
    const prevAutoClear = r.autoClear;
    r.getClearColor(this.clearColor);
    const prevClearAlpha = r.getClearAlpha();
    const prevMask = camera.layers.mask;
    const compositeVisible = this.composite.visible;

    camera.layers.mask = layerMask;
    r.setRenderTarget(target);
    r.setClearColor(0x000000, 0);
    r.clear(true, true, false);
    r.autoClear = false;

    // 1. the opaque parts of the set, depth only (never the composite: it samples this target)
    this.composite.visible = false;
    for (const m of this.occluders) m.colorWrite = false;
    r.render(main, camera);
    for (const m of this.occluders) m.colorWrite = true;
    this.composite.visible = compositeVisible;

    // 2. everything blended, in its usual order, depth tested against them
    r.render(this.scene, camera);

    r.autoClear = prevAutoClear;
    r.setClearColor(this.clearColor, prevClearAlpha);
    r.setRenderTarget(prevTarget);
    camera.layers.mask = prevMask;
  }

  private ensureTarget(w: number, h: number): WebGLRenderTarget {
    if (this.target) {
      if (this.target.width !== w || this.target.height !== h) this.target.setSize(w, h);
      return this.target;
    }
    const ext = this.renderer.extensions;
    const float = ext.has("EXT_color_buffer_float") || ext.has("EXT_color_buffer_half_float");
    const target = new WebGLRenderTarget(w, h, {
      type: float ? HalfFloatType : UnsignedByteType,
      format: RGBAFormat,
      colorSpace: LinearSRGBColorSpace,
      samples: 0,
      depthBuffer: true,
      stencilBuffer: false,
      generateMipmaps: false,
      minFilter: NearestFilter,
      magFilter: NearestFilter,
    });
    target.texture.name = "OracleBlendLayer";
    this.composite.material.uniforms.uLayer.value = target.texture;
    this.target = target;
    return target;
  }

  dispose(): void {
    disposeObject(this.scene);
    this.scene.clear();
    this.target?.dispose();
    this.target = null;
    this.composite.geometry.dispose();
    this.composite.material.dispose();
  }
}
