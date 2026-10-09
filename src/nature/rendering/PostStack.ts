/**
 * The frame's render graph:
 *
 *   view i (≤ 2): background + scene → HalfFloat MSAA target with depth texture
 *                 → near-field DOF (optional) → view image i (linear)
 *   TransitionPass: view images (× exposure) → composite (linear HDR); skipped for a
 *                 single view at exposure 1 (bloom and grade read the view image)
 *   BloomPass (strength > 0) → GradePass: bands + vignette, tone map + sRGB once → canvas
 *
 * Inside a transition a view whose image has no weight (`transitionInputs`) is not
 * rendered; its slot is filled with the other image (weight exactly 0).
 *
 * `post=0` bypasses all of it: `renderStraight` draws one scene to the canvas with
 * the renderer's own tone mapping / output colour space.
 */
import {
  DepthTexture,
  FloatType,
  HalfFloatType,
  LinearFilter,
  LinearSRGBColorSpace,
  RGBAFormat,
  UnsignedByteType,
  WebGLRenderTarget,
  type ColorSpace,
  type PerspectiveCamera,
  type Scene,
  type Texture,
  type TextureDataType,
  type ToneMapping,
  type WebGLRenderer,
} from "three";
import type { LookParams, QualityPreset, TransitionParams } from "../types";
import { Background } from "./Background";
import { BloomPass } from "./BloomPass";
import { DofPass } from "./DofPass";
import { GradePass } from "./GradePass";
import { TransitionPass } from "./TransitionPass";

export interface PostView {
  scene: Scene;
  camera: PerspectiveCamera;
  look: LookParams;
  /** Clear to (0,0,0,0) and skip the background (outgoing view of an `over` transition). */
  transparentBackground: boolean;
  /** Camera layer mask for debug layers. */
  layerMask: number;
}

export interface PostFlags {
  dof: boolean;
  bloom: boolean;
  vignette: boolean;
}

export interface PostFinish {
  transition: TransitionParams | null;
  /** Look used for bloom and vignette (blend of the two views inside a transition). */
  globalLook: LookParams;
  /**
   * The views' own looks inside a transition ([0] outgoing, [1] incoming): wipe and slide
   * give each picture its own vignette before compositing (a sliding picture keeps its bands).
   */
  viewLooks?: readonly LookParams[] | null;
  flags: PostFlags;
  toneMapping: ToneMapping;
}

// constant pass names (no per-frame strings)
const PASS_SCENE = ["scene0", "scene1"] as const;
const PASS_DOF = ["dof0", "dof1"] as const;
const PASS_TRANSITION: Record<TransitionParams["mode"], string> = {
  mix: "transition:mix",
  wipe: "transition:wipe",
  slide: "transition:slide",
  over: "transition:over",
};

export class PostStack {
  private width = 1;
  private height = 1;
  private quality: QualityPreset;
  private readonly floatType: TextureDataType;
  readonly floatTargets: boolean;
  private readonly sceneTargets: (WebGLRenderTarget | null)[] = [null, null];
  private readonly dofTargets: (WebGLRenderTarget | null)[] = [null, null];
  private readonly viewImages: (Texture | null)[] = [null, null];
  private readonly viewExposure: number[] = [1, 1];
  private readonly composite: WebGLRenderTarget;
  private readonly background = new Background();
  private readonly dof: DofPass;
  private readonly bloom: BloomPass;
  private readonly transition = new TransitionPass();
  private readonly grade = new GradePass();
  private readonly noDof: PostFlags = { dof: false, bloom: false, vignette: false };
  /** The two views' vignettes for a per-view composite (reused every frame). */
  private readonly viewVignettes: [LookParams["vignette"], LookParams["vignette"]] = [
    { top: 0, bottom: 0, corners: 0, topSize: 0.4, bottomSize: 0.3 },
    { top: 0, bottom: 0, corners: 0, topSize: 0.4, bottomSize: 0.3 },
  ];
  /** Which passes ran last frame (stats / debug). Reused every frame: copy it to keep it. */
  readonly lastPasses: string[] = [];

  constructor(
    private readonly renderer: WebGLRenderer,
    quality: QualityPreset,
  ) {
    this.quality = quality;
    const ext = renderer.extensions;
    this.floatTargets = ext.has("EXT_color_buffer_float") || ext.has("EXT_color_buffer_half_float");
    this.floatType = this.floatTargets ? HalfFloatType : UnsignedByteType;
    this.composite = this.makeTarget(0, false);
    this.dof = new DofPass(this.floatType);
    this.bloom = new BloomPass(this.floatType, quality.bloomLevels);
  }

  private makeTarget(samples: number, withDepth: boolean): WebGLRenderTarget {
    const rt = new WebGLRenderTarget(this.width, this.height, {
      type: this.floatType,
      format: RGBAFormat,
      colorSpace: LinearSRGBColorSpace,
      samples,
      depthBuffer: withDepth,
      stencilBuffer: false,
      generateMipmaps: false,
      minFilter: LinearFilter,
      magFilter: LinearFilter,
    });
    if (withDepth) rt.depthTexture = new DepthTexture(this.width, this.height, FloatType);
    return rt;
  }

  private sceneTarget(i: number): WebGLRenderTarget {
    let rt = this.sceneTargets[i];
    if (!rt) {
      rt = this.makeTarget(this.quality.msaaSamples, true);
      rt.texture.name = `SilvaScene${i}`;
      this.sceneTargets[i] = rt;
    }
    return rt;
  }

  private dofTarget(i: number): WebGLRenderTarget {
    let rt = this.dofTargets[i];
    if (!rt) {
      rt = this.makeTarget(0, false);
      rt.texture.name = `SilvaDofOut${i}`;
      this.dofTargets[i] = rt;
    }
    return rt;
  }

  /** Drawing-buffer size in pixels. */
  setSize(width: number, height: number): void {
    const w = Math.max(1, Math.floor(width));
    const h = Math.max(1, Math.floor(height));
    if (w === this.width && h === this.height) return;
    this.width = w;
    this.height = h;
    for (const rt of [...this.sceneTargets, ...this.dofTargets]) rt?.setSize(w, h);
    this.composite.setSize(w, h);
    this.dof.setSize(w, h);
    this.bloom.setSize(w, h);
  }

  setQuality(quality: QualityPreset): void {
    const samplesChanged = quality.msaaSamples !== this.quality.msaaSamples;
    this.quality = quality;
    this.bloom.setLevels(quality.bloomLevels);
    if (samplesChanged) {
      for (let i = 0; i < 2; i++) {
        this.sceneTargets[i]?.dispose();
        this.sceneTargets[i] = null;
      }
    }
  }

  /**
   * Compile, in the background, the post programs that not every frame uses: the bloom
   * (looks with bloom; first, it may be needed by the very first frame), the composite
   * (every transition mode — mix / wipe / slide / over, with or without the seam blur —
   * and exposure ≠ 1 are one program: the mode is a uniform), the DOF passes (looks with
   * near blur; the same programs for both views) and the grade with and without bloom
   * (the first look with bloom, canyon → oracle, would otherwise compile it on the spot).
   * Their first frame then does not stall on a synchronous compile.
   */
  precompile(toneMapping: ToneMapping, vignette: boolean): Promise<void> {
    const r = this.renderer;
    const jobs: Promise<unknown>[] = [
      this.grade.compile(r, toneMapping, r.outputColorSpace as ColorSpace, vignette),
      this.bloom.compile(r),
      this.transition.compile(r),
    ];
    if (this.floatTargets && this.quality.dof !== "off") jobs.push(this.dof.compile(r));
    return Promise.all(jobs).then(() => undefined);
  }

  /**
   * The target scene materials are drawn into in post mode. Programs depend on it
   * (linear output, no tone mapping): compile against it, not against the canvas.
   */
  compileTarget(): WebGLRenderTarget {
    return this.sceneTarget(0);
  }

  beginFrame(): void {
    this.viewImages[0] = this.viewImages[1] = null;
    this.viewExposure[0] = this.viewExposure[1] = 1;
    this.lastPasses.length = 0;
  }

  private drawScene(view: PostView, target: WebGLRenderTarget | null, toCanvas: boolean, toneMapping: ToneMapping): void {
    const r = this.renderer;
    r.setRenderTarget(target);
    r.setClearColor(0x000000, view.transparentBackground ? 0 : 1);
    r.clear(true, true, true);
    const cam = view.camera;
    const savedMask = cam.layers.mask;
    const bg = this.background.mesh;
    try {
      cam.layers.mask = view.layerMask;
      if (!view.transparentBackground) {
        this.background.setLook(view.look, (target ? target.width / target.height : this.width / this.height) || 1, toCanvas, toneMapping);
        view.scene.add(bg);
      }
      r.render(view.scene, cam);
    } finally {
      // a throwing render must not leave the shared background in a scene or the mask changed
      if (bg.parent === view.scene) view.scene.remove(bg);
      cam.layers.mask = savedMask;
    }
  }

  /** Render view `index` (0 or 1) of this frame into its own linear image. */
  renderView(index: number, view: PostView, flags: PostFlags, toneMapping: ToneMapping): void {
    const target = this.sceneTarget(index);
    this.drawScene(view, target, false, toneMapping);
    this.lastPasses.push(PASS_SCENE[index]);
    let image: Texture = target.texture;
    const dofOn = flags.dof && this.floatTargets && this.quality.dof !== "off" && target.depthTexture !== null;
    if (dofOn) {
      const out = this.dofTarget(index);
      const did = this.dof.render(this.renderer, target.texture, target.depthTexture as DepthTexture, view.camera, view.look, { samples: this.quality.dofSamples }, out);
      if (did) {
        image = out.texture;
        this.lastPasses.push(PASS_DOF[index]);
      }
    }
    this.viewImages[index] = image;
    this.viewExposure[index] = view.look.exposure;
  }

  /** Exposure of a view that takes part in the composite without being rendered this frame. */
  setViewExposure(index: number, exposure: number): void {
    this.viewExposure[index] = exposure;
  }

  /**
   * Off-screen warm-up draw of a scene set that just became ready (not presented):
   * uploads its buffers, builds VAOs / shadow programs, so the first visible frame
   * does not pay for them. Drawn into slot 1 — where the incoming view of the coming
   * transition goes — so that target (and its DOF output) is allocated now as well.
   * Runs between frames; leaves no image behind for the next one.
   */
  renderWarm(view: PostView, toneMapping: ToneMapping): void {
    this.drawScene(view, this.sceneTarget(1), false, toneMapping);
    if (this.floatTargets && this.quality.dof !== "off") this.renderer.initRenderTarget(this.dofTarget(1));
  }

  /** Composite the views, bloom, grade, output to the canvas. */
  finish(opts: PostFinish): void {
    const imgA = this.viewImages[0];
    const imgB = this.viewImages[1];
    if (!imgA && !imgB) return;
    const tr = opts.transition;
    let source: Texture;
    let vignetteInSource = false;
    if (tr) {
      // a view without weight was not rendered: its slot gets the other image (× 0)
      const a = (imgA ?? imgB) as Texture;
      const b = imgB ?? imgA;
      const looks = opts.viewLooks;
      vignetteInSource = opts.flags.vignette && !!looks && looks.length >= 2 && TransitionPass.perViewVignette(tr.mode);
      if (vignetteInSource && looks) {
        this.viewVignettes[0] = looks[0].vignette;
        this.viewVignettes[1] = looks[1].vignette;
      }
      this.transition.render(this.renderer, a, this.viewExposure[0], b, this.viewExposure[1], tr, this.composite, vignetteInSource ? this.viewVignettes : null);
      this.lastPasses.push(PASS_TRANSITION[tr.mode]);
      source = this.composite.texture;
    } else {
      const index = imgA ? 0 : 1;
      const image = (imgA ?? imgB) as Texture;
      if (this.viewExposure[index] === 1) {
        // A × 1 is the image itself: bloom and grade read it, no full-frame copy
        source = image;
      } else {
        this.transition.render(this.renderer, image, this.viewExposure[index], null, 1, null, this.composite);
        this.lastPasses.push("composite");
        source = this.composite.texture;
      }
    }

    let bloomTex: Texture | null = null;
    const bl = opts.globalLook.bloom;
    if (opts.flags.bloom && bl.strength > 0.001) {
      bloomTex = this.bloom.render(this.renderer, source, bl.radius, bl.threshold);
      this.lastPasses.push("bloom");
    }
    this.grade.render(
      this.renderer,
      source,
      opts.globalLook,
      {
        bloom: bloomTex,
        bloomStrength: bl.strength,
        vignette: opts.flags.vignette,
        vignetteInSource,
        toneMapping: opts.toneMapping,
        outputColorSpace: this.renderer.outputColorSpace as ColorSpace,
      },
      this.width,
      this.height,
    );
    this.lastPasses.push("grade");
  }

  /** `post=0`: one scene straight to the canvas, renderer tone mapping + sRGB output. */
  renderStraight(view: PostView, toneMapping: ToneMapping): void {
    const r = this.renderer;
    const savedTone = r.toneMapping;
    const savedExposure = r.toneMappingExposure;
    r.toneMapping = toneMapping;
    r.toneMappingExposure = view.look.exposure;
    try {
      this.drawScene({ ...view, transparentBackground: false }, null, true, toneMapping);
    } finally {
      r.toneMapping = savedTone;
      r.toneMappingExposure = savedExposure;
    }
    this.lastPasses.length = 0;
    this.lastPasses.push("straight");
  }

  /**
   * Nothing ready yet: the look's background alone. In post mode it goes through the
   * same composite → bloom → grade (bands, vignette) as a scene frame, so the first set
   * does not pop in against a differently graded background.
   */
  renderEmpty(look: LookParams, toneMapping: ToneMapping, scene: Scene, camera: PerspectiveCamera, post: PostFlags | null): void {
    const view: PostView = { scene, camera, look, transparentBackground: false, layerMask: 1 };
    if (!post) {
      this.drawScene(view, null, true, toneMapping);
      this.lastPasses.length = 0;
      this.lastPasses.push("background-only");
      return;
    }
    this.beginFrame();
    this.lastPasses.push("background-only");
    this.renderView(0, view, this.noDof, toneMapping);
    this.finish({ transition: null, globalLook: look, flags: post, toneMapping });
  }

  get size(): { width: number; height: number } {
    return { width: this.width, height: this.height };
  }

  dispose(): void {
    for (const rt of [...this.sceneTargets, ...this.dofTargets]) rt?.dispose();
    this.composite.dispose();
    this.background.dispose();
    this.dof.dispose();
    this.bloom.dispose();
    this.transition.dispose();
    this.grade.dispose();
  }
}
