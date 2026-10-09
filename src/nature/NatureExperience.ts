/**
 * The engine: one WebGLRenderer (WebGL2, no canvas MSAA — MSAA lives on the scene
 * render targets), one requestAnimationFrame loop, one ambient clock.
 *
 *  - story time t (video seconds) comes from the scroll position (smoothed with
 *    delta-time damping; a jump of two or more scene sets snaps instead of flying
 *    through every set in between), from `?t=` / `scene=&progress=`, from autoplay,
 *    or is fixed in capture mode; the picture is a pure function of (t, timeSec,
 *    seed, quality, viewport, flags);
 *  - the ambient clock pauses while the tab is hidden and never jumps on return;
 *  - SceneDirector decides which scene set(s) to render (and preloads the next one in
 *    the direction of travel), PostStack renders them; a set that throws is disabled
 *    on its own, the others keep rendering;
 *  - the drawing buffer is capped by a per-quality pixel budget.
 *
 * Exposes `window.__NATURE__` (state, setters, stats) and `window.__NATURE_READY__`.
 */
import {
  ACESFilmicToneMapping,
  AgXToneMapping,
  BackSide,
  Color,
  DoubleSide,
  FrontSide,
  MeshDepthMaterial,
  MeshDistanceMaterial,
  NeutralToneMapping,
  PCFShadowMap,
  PerspectiveCamera,
  SRGBColorSpace,
  Scene,
  Vector2,
  WebGLRenderer,
  type BufferGeometry,
  type Camera,
  type Fog,
  type Light,
  type Line,
  type Material,
  type Mesh,
  type Object3D,
  type PointLight,
  type Points,
  type ShaderMaterial,
  type Side,
  type Sprite,
  type Texture,
  type ToneMapping,
  type WebGLProgram as ThreeProgram,
} from "three";
import { AssetRegistry, type UploadStep } from "./AssetRegistry";
import { SceneDirector, transitionNear, type PrepareOutcome, type Resolution } from "./SceneDirector";
import {
  CAPTURE_TIME_SEC,
  EPISODE_LOOKS,
  QUALITY_PRESETS,
  SCROLL_RESPONSE_SEC,
  VIDEO_DURATION,
  episodeAt,
  episodeConfig,
  isEpisodeId,
  setIndex,
  setSpan,
  timeForEpisode,
} from "./SceneConfig";
import { LayerRegistry } from "./core/layers";
import { clamp, damp, lambdaForResponse } from "./core/math";
import { rngFor } from "./core/rng";
import { parseUrlState, requestedTime, type UrlState } from "./core/urlState";
import { CaptureState } from "./debug/CaptureState";
import type { DebugControls, DebugInfo, DebugTarget, DebugToggle } from "./debug/DebugControls";
import { FrameStats, type FrameStatsSnapshot } from "./debug/FrameStats";
import { copyLook, episodeLook, lerpLook, resolveLook } from "./rendering/look";
import { PostStack, type PostView } from "./rendering/PostStack";
import { transitionInputs } from "./rendering/TransitionPass";
import { SCENE_FACTORIES } from "./scenes";
import { attachFramePacing, noteSliceOwner, sliceStats, type FramePacing } from "./vegetation/slices";
import type {
  DebugFlags,
  EpisodeId,
  FrameState,
  LookParams,
  NatureScene,
  QualityLevel,
  QualityPreset,
  SceneContext,
  SceneLocal,
  SceneSetId,
  ToneMappingName,
  TransitionParams,
  Viewport,
} from "./types";
import { WindField } from "./vegetation/WindField";

export type EngineStatus = "starting" | "running" | "context-lost" | "fallback" | "error";

export interface NatureStateSnapshot {
  t: number;
  targetT: number;
  episode: EpisodeId;
  sceneProgress: number;
  globalProgress: number;
  timeSec: number;
  frozen: boolean;
  seed: number;
  quality: QualityLevel;
  /**
   * Why this level: "url", "capture", "api", "auto: …" (renderer heuristic), "adapt: … → <level>
   * (step n)" (step-down) or "adapt: … → low dpr 1.0 (step n)" (the last step, inside low).
   */
  qualityReason: string;
  capture: boolean;
  reducedMotion: boolean;
  ready: boolean;
  views: { set: SceneSetId; episode: EpisodeId; progress: number; role: string; t: number }[];
  transition: { id: string; mode: string; k: number } | null;
  sets: Partial<Record<SceneSetId, string>>;
  /** Sets fading in (they turned ready while on screen, see FADE_IN_MS); `ready` waits for them. */
  fading: SceneSetId[];
  flags: Record<DebugToggle, boolean> & { layers: string[] | null; toneMapping: ToneMappingName };
  status: EngineStatus;
}

export interface NatureStatsSnapshot extends FrameStatsSnapshot {
  instances: Record<string, number>;
  dpr: number;
  drawingBuffer: { width: number; height: number };
  gpu: string;
  passes: string[];
  programs: number;
  geometries: number;
  textures: number;
  liveRenderers: number;
  rafLoops: number;
  createdRenderers: number;
  /** KHR_parallel_shader_compile available (programs link off the main thread, non-blocking status query). */
  parallelShaderCompile: boolean;
  /** Quality level, why, and the adaptive step-down (auto quality only). */
  quality: {
    level: QualityLevel;
    reason: string;
    adapt: {
      active: boolean;
      thresholdMs: number;
      gpuTimer: boolean;
      steps: number;
      /** DPR cap set by the last step inside `low` (ADAPT.lastDprCap), null before it. */
      dprCap: number | null;
      /** The last evaluated window: frame-interval p95, CPU p95, GPU median (ms). */
      last: { at: number; p95: number; cpuP95: number; gpuMs: number | null; samples: number } | null;
      log: AdaptLogEntry[];
    };
  };
  /** Warm-up queue (sets still compiling / uploading) and the last finished warm-ups. */
  warm: {
    pending: { set: SceneSetId; phase: string; units: number; unit: number; programs: number; textures: number }[];
    /** Longest warm work done inside one frame since start (ms). */
    maxFrameMs: number;
    /** Programs whose first use (link status, uniform locations) was done ahead of their first draw, and the longest one (ms). */
    touched: number;
    maxTouchMs: number;
    /** Page time (performance.now(), ms) of that longest first use. */
    maxTouchAt: number;
    log: WarmLogEntry[];
  };
  /**
   * How the prepares ended (counts since start, the last ones), and the slice scheduler:
   * queued build jobs / waiting yields, jobs cancelled so far.
   */
  builds: {
    counts: Record<PrepareOutcome, number>;
    log: BuildLogEntry[];
    slices: { jobs: number; waiting: number; cancelledJobs: number };
  };
}

export interface BuildLogEntry {
  set: SceneSetId;
  outcome: PrepareOutcome;
  /** Page time (performance.now(), ms) the prepare ended, and its length (ms). */
  at: number;
  ms: number;
  /** Released while preparing: from the release to the unwound, disposed build (ms). */
  afterReleaseMs: number | null;
}

export interface NatureApi {
  readonly ready: boolean;
  state(): NatureStateSnapshot;
  stats(): NatureStatsSnapshot;
  /** Jump story time (video seconds); pinned until the user scrolls. Non-finite values are ignored. */
  setT(t: number): void;
  setScene(episode: EpisodeId, progress?: number): void;
  /** Freeze the ambient clock at `seconds`, or null to let it run. Non-finite values are ignored. */
  setTime(seconds: number | null): void;
  setQuality(level: QualityLevel): void;
  setSeed(seed: number): void;
  setFlags(flags: Partial<Record<DebugToggle, boolean>> & { layers?: string[] | null; toneMapping?: ToneMappingName }): void;
}

declare global {
  interface Window {
    __NATURE__?: NatureApi;
  }
}

export interface NatureExperienceOptions {
  /** Element the canvas is appended to (fixed, full viewport). */
  host: HTMLElement;
  /** URL search string (defaults to location.search). */
  search?: string;
  onStatus?(status: EngineStatus, detail?: string): void;
}

const TONE_MAPPERS: Record<ToneMappingName, ToneMapping> = {
  aces: ACESFilmicToneMapping,
  agx: AgXToneMapping,
  neutral: NeutralToneMapping,
};

/**
 * Drawing-buffer pixel budget per quality: with two views a frame holds ≈ 155 bytes
 * per drawing-buffer pixel in render targets, so the device pixel ratio is lowered
 * on very large screens instead of allocating 1 GB+ at 4K. 1440×1020 at dpr 1
 * (captures) is far below every budget.
 */
const MAX_DRAWING_BUFFER_PX: Record<QualityLevel, number> = {
  high: 3.2e6,
  medium: 2.2e6,
  low: 1.4e6,
};

/**
 * Build work per frame (ms): a scene set preparing — vegetation scatter, instance writing,
 * the scenes' other sliced steps (vegetation/slices.ts) — runs in a task of its own right
 * after the frame was drawn, never several stretches back to back. A preload while a picture
 * is on screen gets a small share (frames stay at 60 Hz); a set a view on screen is waiting
 * for, or that the next transition window (within `soonSec` story seconds in the direction
 * of travel) shows, gets more; with only the background on screen (first load, a jump to a
 * set not prepared) the build goes first, at ≈ 30 fps.
 */
const BUILD_BUDGET_MS = { preload: 8, needed: 12, empty: 24, soonSec: 4 } as const;

/**
 * A scene set that turns ready while it is already on screen — the first load, a jump to a
 * set that was not prepared, a transition whose other set was missing — fades in over this
 * long (smoothstep) instead of popping in: from the look's background it replaces, or, in a
 * transition, from the other picture alone towards the real progress. Two sets never ramp at
 * once (the second starts when the first ends). Composite only: not in capture mode (same
 * URL ⇒ same picture) nor with `post=0`. `ready` (`__NATURE_READY__`) waits for it.
 */
const FADE_IN_MS = 450;

/** Post flags of the background image under a fade-in (as `PostStack.renderEmpty` draws it). */
const FADE_BACKGROUND_FLAGS = { dof: false, bloom: false, vignette: false } as const;

/**
 * Story moving through sets that are not ready (a fling, a jump, the browser's smooth scroll
 * to Home / End): the last picture stays up for at most this long (ms after it was drawn;
 * nothing is drawn, the canvas keeps it) instead of a flash of the background. A set that
 * turns ready meanwhile cuts in (no fade: the background was never shown). Not in capture mode.
 */
const HOLD_MS = 300;

/** Scene-set distance between two story times (SCENE_SET_ORDER steps). */
function setsApart(a: number, b: number): number {
  return Math.abs(setIndex(episodeAt(a).set) - setIndex(episodeAt(b).set));
}

class EngineFlags implements DebugFlags {
  debug: boolean;
  capture: boolean;
  post: boolean;
  dof: boolean;
  bloom: boolean;
  vignette: boolean;
  wind: boolean;
  wireframe = false;
  parallax: boolean;
  layers: Set<string> | null;

  constructor(url: UrlState) {
    this.debug = url.debug;
    this.capture = url.capture;
    this.post = url.post;
    this.dof = url.dof;
    this.bloom = url.bloom;
    this.vignette = url.vignette;
    this.wind = url.wind;
    this.parallax = url.parallax;
    this.layers = url.layers ? new Set(url.layers) : null;
  }

  isLayerVisible(name: string): boolean {
    return this.layers === null || this.layers.has(name);
  }
}

function autoQuality(): QualityLevel {
  try {
    const coarse = window.matchMedia("(pointer: coarse)").matches;
    const small = Math.min(window.screen.width, window.screen.height) < 820;
    return coarse && small ? "medium" : "high";
  } catch {
    return "high";
  }
}

function isTouch(): boolean {
  try {
    return window.matchMedia("(pointer: coarse)").matches;
  } catch {
    return false;
  }
}

/** Software rasterisers: always `low`. */
const SOFTWARE_GPU = /swiftshader|llvmpipe|softpipe|lavapipe|basic render driver|software rasterizer|software adapter/i;
/** Integrated / mobile GPUs: one level below the device default. Apple counts on touch devices only. */
const WEAK_GPU = /intel.*\b(hd|uhd|iris)\b|\bmali\b|adreno|powervr/i;
const APPLE_GPU = /\bapple\b/i;
const LOWER_QUALITY: Record<QualityLevel, QualityLevel | null> = { high: "medium", medium: "low", low: null };

/**
 * Auto quality from the renderer string (no `quality=` in the URL): software → low;
 * Intel HD / UHD / Iris, Mali, Adreno, PowerVR, or an Apple GPU on a touch device →
 * one level below the device default (desktop Intel → medium, phones → low).
 */
export function qualityForRenderer(base: QualityLevel, gpu: string, touch: boolean): { level: QualityLevel; reason: string } {
  if (SOFTWARE_GPU.test(gpu)) return { level: "low", reason: `auto: software renderer (${gpu})` };
  if (WEAK_GPU.test(gpu) || (touch && APPLE_GPU.test(gpu))) {
    const level = LOWER_QUALITY[base] ?? base;
    return { level, reason: `auto: ${touch ? "touch" : "desktop"} default ${base}, integrated / mobile GPU → ${level} (${gpu})` };
  }
  return { level: base, reason: `auto: ${touch ? "touch" : "desktop"} default ${base}` };
}

/**
 * Adaptive step-down (auto quality only, never in capture mode): the p95 of the frame
 * interval over `windowMs` of steady frames (no transition near, nothing building or
 * warming, a set on screen); above `thresholdMs` — and the frame's own GPU time (timer
 * query) or CPU time above `ownShare` × threshold, so a throttled browser does not count —
 * one level down, at most `maxSteps` times, never up. Below `low` there is one more step
 * (weak phones): the DPR cap of the current preset drops to `lastDprCap` (1.5× → 1×, about
 * half the pixels), taken only when it makes the drawing buffer smaller.
 */
const ADAPT = {
  thresholdMs: 20,
  windowMs: 2000,
  minSamples: 60,
  maxSteps: 3,
  cooldownMs: 1500,
  ownShare: 0.6,
  transitionLeadSec: 0.5,
  lastDprCap: 1.0,
} as const;

export interface AdaptLogEntry {
  /** Page time (performance.now(), ms) and story time of the step. */
  at: number;
  t: number;
  from: QualityLevel;
  to: QualityLevel;
  p95: number;
  cpuP95: number;
  gpuMs: number | null;
  /** Set on the last step inside `low` (from = to = "low"): the DPR cap it applied. */
  dprCap?: number;
}

function round1(v: number): number {
  return Math.round(v * 10) / 10;
}

function percentile(values: Float64Array, n: number, q: number): number {
  if (n === 0) return 0;
  const sorted = Array.from(values.subarray(0, n)).sort((a, b) => a - b);
  return sorted[Math.min(n - 1, Math.floor(q * (n - 1) + 0.5))];
}

function newFrameState(viewport: Viewport, quality: QualityPreset): FrameState {
  return {
    timeSec: 0,
    deltaSec: 0,
    t: 0,
    globalProgress: 0,
    episode: "hero",
    sceneProgress: 0,
    pointerNdc: { x: 0, y: 0 },
    reducedMotion: false,
    viewport,
    quality,
    capture: false,
    frame: 0,
  };
}

function newPostView(): PostView {
  return { scene: null as unknown as Scene, camera: null as unknown as PerspectiveCamera, look: null as unknown as LookParams, transparentBackground: false, layerMask: 1 };
}

export class NatureExperience {
  static liveRenderers = 0;
  static liveLoops = 0;
  static createdRenderers = 0;

  /**
   * WebGL 2 is really there: a browser can expose WebGL2RenderingContext with WebGL 2
   * disabled or blocklisted, so a context is created (and released at once). Called
   * before any renderer exists: a missing WebGL 2 logs nothing to the console.
   */
  static hasWebGL2(): boolean {
    if (typeof window === "undefined" || typeof window.WebGL2RenderingContext === "undefined") return false;
    try {
      const gl = document.createElement("canvas").getContext("webgl2");
      if (!gl) return false;
      gl.getExtension("WEBGL_lose_context")?.loseContext();
      return true;
    } catch {
      return false;
    }
  }

  private readonly host: HTMLElement;
  private readonly url: UrlState;
  private readonly onStatus?: (status: EngineStatus, detail?: string) => void;
  private status: EngineStatus = "starting";

  private renderer: WebGLRenderer | null = null;
  private canvas: HTMLCanvasElement | null = null;
  private post: PostStack | null = null;
  private assets: AssetRegistry | null = null;
  private director: SceneDirector | null = null;
  /** The frame loop paces the scene sets' sliced build work (vegetation/slices.ts). */
  private slicePacing: FramePacing | null = null;
  private readonly wind = new WindField();
  private readonly layers = new LayerRegistry();
  private readonly flags: EngineFlags;
  private readonly stats = new FrameStats();
  private capture: CaptureState | null = null;
  private debugPanel: DebugControls | null = null;

  private quality: QualityPreset;
  private seed: number;
  private toneMappingName: ToneMappingName;
  private reducedMotion = false;

  // time
  private rafId = 0;
  private lastNow = -1;
  private timeSec = 0;
  private frozenTime: number | null;
  private frame = 0;

  // story
  private t = 0;
  private targetT = 0;
  private pinnedT: number | null = null;
  private captureT = 0;
  private autoplay = false;
  private autoT = 0;
  private snapNext = true;
  private scrollMax = 0;
  private expectedScrollY: number | null = null;
  private lastScrollT = 0;
  /** Direction of travel along the story (preload side). */
  private travel: 1 | -1 = 1;
  /** Every view of the last frame was ready (or failed): autoplay only advances then. */
  private viewsSettled = false;
  /** Layout the scroll mapping was last anchored to (innerWidth, scrollHeight). */
  private layoutWidth = -1;
  private layoutHeight = -1;

  // view state
  private viewport: Viewport = { width: 1, height: 1, dpr: 1, aspect: 1 };
  private resizePending = false;
  private readonly pointer = { x: 0, y: 0 };
  // per view slot: 0 = outgoing / solo, 1 = incoming, 2 = warm-up draw of a preloaded set
  private readonly baseLooks: LookParams[] = [resolveLook(EPISODE_LOOKS.hero), resolveLook(EPISODE_LOOKS.hero), resolveLook(EPISODE_LOOKS.hero)];
  private readonly viewLooks: LookParams[] = [resolveLook(EPISODE_LOOKS.hero), resolveLook(EPISODE_LOOKS.hero), resolveLook(EPISODE_LOOKS.hero)];
  private readonly postViews: PostView[] = [newPostView(), newPostView(), newPostView()];
  private readonly globalLook: LookParams = resolveLook(EPISODE_LOOKS.hero);
  private readonly frameState: FrameState;
  private readonly warmFrame: FrameState;
  /** The frame as a transition's composite sees it while one of its pictures fades in. */
  private readonly fadeFrame: FrameState;
  private readonly frameScenes: (NatureScene | null)[] = [null, null];
  private frameViews: Resolution["views"] = [];
  private readonly onOverrideError = (i: number, err: unknown) => this.failView(this.frameViews, i, err);
  private readonly inputs = { a: true, b: true };
  private readonly emptyScene = new Scene();
  private readonly emptyCamera = new PerspectiveCamera();
  /** Loop time (rAF timestamp, ms) of the frame being rendered. */
  private frameNow = 0;
  /**
   * Ramp start (loop time, ms) of each set fading in, see FADE_IN_MS. NaN: the set turned
   * ready in this frame and is drawn once at weight 0 (its first draw, which may take long,
   * is not part of the ramp); its ramp starts with the next frame.
   */
  private readonly fadeStart = new Map<SceneSetId, number>();
  /** Loop time of the last frame that drew a scene; the canvas holds that picture (HOLD_MS). */
  private lastSceneAt = -1;
  private holding = false;
  /** Sets of the last post frame's views that were not ready (and not failed). */
  private fadeMissing = new Set<SceneSetId>();
  private fadeMissingNext = new Set<SceneSetId>();
  /** Presence 0–1 of view 0 / 1 in this frame (1 = no ramp). */
  private readonly fadePresence = [1, 1];
  /** View 0 / 1 is drawn for the first time in this frame, at weight 0. */
  private readonly fadeFirst = [false, false];
  /** The background a solo fade-in starts from (base look of view 0, as `renderEmpty`). */
  private readonly fadeLook: LookParams = resolveLook(EPISODE_LOOKS.hero);
  /** Composite of a solo fade-in: background (A) → view (B), a plain crossfade. */
  private readonly fadeParams: TransitionParams = {
    mode: "mix",
    k: 0,
    dip: 0,
    dipColor: new Color(0, 0, 0),
    edge: 0,
    raggedness: 0,
    softness: 0,
    noiseScale: 1,
    direction: 1,
    edgeDarken: 0,
    offset: 0,
    seamSoftness: 0,
    seamDarkness: 0,
    seamBlurPx: 0,
    seamWave: 0,
    follow: 1,
    reveal: 0,
    opacityA: 1,
    backA: new Color(0, 0, 0),
  };
  private lastRes: Resolution | null = null;
  private readonly instances = new Map<string, number>();
  private gpu = "unknown";
  private contextLost = false;
  private disposed = false;
  private started = false;
  private errorCount = 0;
  private wireframeApplied = false;
  private readonly cleanups: (() => void)[] = [];

  // warm-up (shader compile / texture upload / off-screen draw of a prepared set)
  private parallelCompile = false;
  private readonly warmJobs: WarmJob[] = [];
  /**
   * Aborted warm-ups (the set was released meanwhile) whose programs the driver is still
   * linking: the set is disposed only once they have linked (or after `WARM.drainMaxMs`).
   * Deleting a program that is still linking (KHR_parallel_shader_compile) makes Chrome's
   * GPU process log "GL_INVALID_VALUE: glGetProgramiv: Program object expected" for it,
   * many times over.
   */
  private readonly warmDrains: { job: WarmJob; error: unknown; since: number }[] = [];
  private readonly warmLog: WarmLogEntry[] = [];
  /** How the last prepares ended (released while preparing → "cancelled"). */
  private readonly buildLog: BuildLogEntry[] = [];
  private readonly buildCounts: Record<PrepareOutcome, number> = { ready: 0, cancelled: 0, dropped: 0, failed: 0 };
  private readonly warmMaterials = new Map<SceneSetId, Material[]>();
  private warmFrameMaxMs = 0;
  /** Programs whose first use is done (by a warm-up or by a draw). */
  private readonly touched = new WeakSet<object>();
  private touchedCount = 0;
  private touchMaxMs = 0;
  private touchMaxAt = 0;
  /** A scene set has been drawn (the first load is over); the last frame drew one. */
  private sceneDrawn = false;
  private lastDrawn = false;

  // quality: why this level, adaptive step-down (auto quality only)
  private qualityReason = "";
  private adaptActive = false;
  private adaptThreshold: number = ADAPT.thresholdMs;
  private adaptSteps = 0;
  /**
   * DPR cap below the preset's: the adaptive watcher's last step inside `low`. Never raised
   * by the watcher; an explicit `setQuality` (API / debug panel) clears it.
   */
  private dprLimit = Number.POSITIVE_INFINITY;
  private adaptWindowStart = -1;
  private adaptFirstFrame = 0;
  private adaptPrevNow = 0;
  private adaptN = 0;
  private adaptCooldownUntil = 0;
  private readonly adaptIntervals = new Float64Array(600);
  private readonly adaptCpu = new Float64Array(600);
  private adaptLast: { at: number; p95: number; cpuP95: number; gpuMs: number | null; samples: number } | null = null;
  private readonly adaptLog: AdaptLogEntry[] = [];
  private gpuTimer: GpuFrameTimer | null = null;
  private readonly gpuFrames = new Float64Array(600).fill(-1);
  private readonly gpuMs = new Float64Array(600);
  private readonly gpuScratch = new Float64Array(600);
  private gpuIdx = 0;
  /** performance.now() of the last frame in which the story moved (or was still heading somewhere). */
  private lastMoveAt = 0;

  constructor(opts: NatureExperienceOptions) {
    this.host = opts.host;
    this.onStatus = opts.onStatus;
    this.url = parseUrlState(opts.search ?? window.location.search);
    this.flags = new EngineFlags(this.url);
    this.seed = this.url.seed;
    this.quality = QUALITY_PRESETS[this.url.quality ?? (this.url.capture ? "high" : autoQuality())];
    this.qualityReason = this.url.quality ? "url" : this.url.capture ? "capture" : "auto";
    // ?adapt=<ms>: threshold of the adaptive step-down (frame p95), adapt=0 turns it off
    const params = new URLSearchParams(opts.search ?? window.location.search);
    const adapt = params.has("adapt") ? Number(params.get("adapt")) : Number.NaN;
    this.adaptThreshold = Number.isFinite(adapt) && adapt >= 0 ? adapt : ADAPT.thresholdMs;
    this.toneMappingName = this.url.toneMapping ?? "aces";
    this.frozenTime = this.url.time ?? (this.url.capture ? CAPTURE_TIME_SEC : null);
    if (this.frozenTime !== null) this.timeSec = this.frozenTime;
    this.frameState = newFrameState(this.viewport, this.quality);
    this.warmFrame = newFrameState(this.viewport, this.quality);
    this.fadeFrame = newFrameState(this.viewport, this.quality);
  }

  private setStatus(status: EngineStatus, detail?: string): void {
    this.status = status;
    this.onStatus?.(status, detail);
  }

  /** Create the renderer and start the loop. Throws if WebGL2 is unavailable. */
  start(): void {
    if (this.started || this.disposed) return;
    this.started = true;

    const canvas = document.createElement("canvas");
    canvas.className = "nature-canvas";
    canvas.style.display = "block";
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    canvas.setAttribute("aria-hidden", "true");

    const renderer = new WebGLRenderer({
      canvas,
      antialias: false,
      alpha: false,
      depth: true,
      stencil: false,
      premultipliedAlpha: true,
      preserveDrawingBuffer: false,
      powerPreference: "high-performance",
    });
    NatureExperience.liveRenderers++;
    NatureExperience.createdRenderers++;
    this.renderer = renderer;
    this.canvas = canvas;
    this.host.appendChild(canvas);

    renderer.autoClear = false;
    renderer.info.autoReset = false;
    renderer.outputColorSpace = SRGBColorSpace;
    renderer.toneMapping = TONE_MAPPERS[this.toneMappingName];
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = PCFShadowMap;
    renderer.debug.checkShaderErrors = process.env.NODE_ENV !== "production";

    const gl = renderer.getContext();
    try {
      const ext = gl.getExtension("WEBGL_debug_renderer_info");
      this.gpu = String(ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER));
    } catch {
      this.gpu = String(gl.getParameter(gl.RENDERER));
    }

    // auto quality (no quality= in the URL, not a capture): one level less on integrated /
    // mobile GPUs, low on software rasterisers; then the adaptive step-down watches
    if (!this.url.quality && !this.url.capture) {
      const pick = qualityForRenderer(this.quality.level, this.gpu, isTouch());
      this.quality = QUALITY_PRESETS[pick.level];
      this.qualityReason = pick.reason;
      this.adaptActive = this.adaptThreshold > 0;
      if (this.adaptActive) this.gpuTimer = GpuFrameTimer.create(gl as WebGL2RenderingContext);
    }

    // programs link off the main thread and their status can be polled without blocking
    this.parallelCompile = renderer.extensions.has("KHR_parallel_shader_compile");
    if (this.url.debug) {
      console.info(
        `[nature] KHR_parallel_shader_compile: ${this.parallelCompile ? "available — programs link in the background, up to " + WARM.programs.needed + " new per frame" : "not available — one new program and one first use per frame"} (${this.gpu})`,
      );
    }

    this.assets = new AssetRegistry(renderer, this.quality.anisotropy);
    void this.assets.init();
    this.post = new PostStack(renderer, this.quality);
    // from now on build slices run between frames, a budget per frame (before any set prepares)
    this.slicePacing = attachFramePacing();
    // compile errors surface (with logs) on first use
    this.post.precompile(TONE_MAPPERS[this.toneMappingName], this.flags.vignette).catch(() => undefined);
    this.director = new SceneDirector({
      factories: SCENE_FACTORIES,
      contextFor: (set) => this.contextFor(set),
      warm: (set, scene, stale) => this.warmSet(set, scene, stale),
      onReady: (_set, scene) => {
        if (this.flags.wireframe) setWireframe(scene, true);
      },
      onRelease: (set) => {
        this.forgetInstances(set);
        this.releaseWarm(set);
      },
      onError: (set) => {
        this.forgetInstances(set);
        this.releaseWarm(set);
      },
      onPrepareEnd: (set, outcome, ms, afterReleaseMs) => this.notePrepareEnd(set, outcome, ms, afterReleaseMs),
    });
    this.capture = new CaptureState(2);

    if (this.url.capture) document.documentElement.dataset.natureCapture = "1";
    this.setupReducedMotion();
    this.setupListeners(canvas);
    // forced: a host that measures 0×0 at start must still size the canvas (not 300×150)
    this.resize(true);
    this.measureScroll();
    this.setupStory();
    this.installApi();

    if (this.url.debug && !this.url.capture) {
      void import("./debug/DebugControls").then((m) => {
        if (this.disposed) return;
        this.debugPanel = new m.DebugControls(this.debugTarget());
      });
    }

    NatureExperience.liveLoops++;
    this.rafId = requestAnimationFrame(this.loop);
    this.setStatus("running");
  }

  // -------------------------------------------------------------------------
  // Setup
  // -------------------------------------------------------------------------

  private listen<K extends keyof WindowEventMap>(target: Window, type: K, fn: (e: WindowEventMap[K]) => void, opts?: AddEventListenerOptions): void;
  private listen(target: EventTarget, type: string, fn: (e: Event) => void, opts?: AddEventListenerOptions): void;
  private listen(target: EventTarget, type: string, fn: (e: Event) => void, opts?: AddEventListenerOptions): void {
    target.addEventListener(type, fn, opts);
    this.cleanups.push(() => target.removeEventListener(type, fn, opts));
  }

  private setupReducedMotion(): void {
    if (this.url.reducedMotion !== null) {
      this.reducedMotion = this.url.reducedMotion;
      return;
    }
    try {
      const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
      this.reducedMotion = mq.matches;
      this.listen(mq, "change", () => {
        this.reducedMotion = mq.matches;
        this.capture?.invalidate();
      });
    } catch {
      this.reducedMotion = false;
    }
  }

  private setupListeners(canvas: HTMLCanvasElement): void {
    // applied at the start of the next frame, right before drawing: resizing the canvas
    // clears it, and a resize between frames would show a black frame
    const ro = new ResizeObserver(() => {
      this.resizePending = true;
    });
    ro.observe(this.host);
    this.cleanups.push(() => ro.disconnect());

    const docRo = new ResizeObserver(() => this.measureScroll());
    docRo.observe(document.body);
    this.cleanups.push(() => docRo.disconnect());
    this.listen(window, "resize", () => this.measureScroll());

    this.listen(document, "visibilitychange", () => {
      // pause: the next visible frame starts with dt = 0, the ambient clock does not jump
      this.lastNow = -1;
      this.stats.resetInterval();
    });

    this.listen(window, "pointermove", (e) => {
      this.pointer.x = (e.clientX / Math.max(1, window.innerWidth)) * 2 - 1;
      this.pointer.y = -((e.clientY / Math.max(1, window.innerHeight)) * 2 - 1);
    }, { passive: true });

    const userInput = () => {
      if (this.url.capture) return;
      this.autoplay = false;
    };
    this.listen(window, "wheel", userInput, { passive: true });
    this.listen(window, "touchstart", userInput, { passive: true });
    this.listen(window, "keydown", userInput);
    this.listen(window, "hashchange", () => {
      // an anchor link: the scroll position it lands on is the story position
      this.pinnedT = null;
      this.expectedScrollY = null;
      if (!this.url.capture) this.autoplay = false;
    });

    this.listen(canvas, "webglcontextlost", (e) => {
      e.preventDefault();
      this.contextLost = true;
      // the queued programs are gone with the context: those sets compile on first use
      this.flushWarm();
      this.gpuTimer?.forget();
      this.capture?.invalidate();
      this.setStatus("context-lost");
    });
    this.listen(canvas, "webglcontextrestored", () => {
      this.contextLost = false;
      this.lastNow = -1;
      this.resize(true);
      this.setStatus("running");
    });
  }

  private setupStory(): void {
    const requested = requestedTime(this.url);
    if (this.url.capture) {
      this.captureT = requested ?? 0;
      this.t = this.targetT = this.captureT;
      return;
    }
    if (this.url.autoplay) {
      this.autoplay = true;
      this.autoT = requested ?? 0;
      this.t = this.targetT = this.autoT;
      return;
    }
    if (requested !== null) {
      try {
        window.history.scrollRestoration = "manual";
      } catch {
        // ignore
      }
      this.pin(requested);
    } else {
      this.t = this.targetT = this.scrollT();
    }
    this.snapNext = true;
  }

  /**
   * A fresh context per prepare. It is also the owner of the set's sliced build work
   * (vegetation/slices.ts): `rng()`, `reportInstances()` and `seed` tie a sliced step that
   * touches them to the set; once the director released the set while it was preparing
   * (`cancelSlices(ctx)`), every touch throws `SliceCancelled`, so the build unwinds at its
   * next touch (and a cancelled build records no instance counts).
   */
  private contextFor(set: SceneSetId): SceneContext {
    const renderer = this.renderer as WebGLRenderer;
    const assets = this.assets as AssetRegistry;
    const seed = this.seed;
    const reduced = () => this.reducedMotion;
    const ctx: SceneContext = {
      set,
      renderer,
      assets,
      get seed() {
        noteSliceOwner(ctx);
        return seed;
      },
      rng: (label: string) => {
        noteSliceOwner(ctx);
        return rngFor(seed, `${set}:${label}`);
      },
      quality: this.quality,
      wind: this.wind.uniforms,
      debug: this.flags,
      layers: this.layers,
      reportInstances: (key: string, count: number) => {
        noteSliceOwner(ctx);
        this.instances.set(`${set}.${key}`, count);
      },
      // live: follows prefers-reduced-motion changes after the set was built
      get reducedMotion() {
        return reduced();
      },
      capture: this.url.capture,
    };
    return ctx;
  }

  private notePrepareEnd(set: SceneSetId, outcome: PrepareOutcome, ms: number, afterReleaseMs: number | null): void {
    this.buildCounts[outcome]++;
    this.buildLog.push({ set, outcome, at: Math.round(performance.now()), ms: Math.round(ms), afterReleaseMs: afterReleaseMs === null ? null : Math.round(afterReleaseMs) });
    if (this.buildLog.length > 40) this.buildLog.shift();
    if (this.url.debug && outcome === "cancelled") {
      console.info(`[nature] build of "${set}" cancelled after ${Math.round(ms)} ms${afterReleaseMs === null ? "" : ` (stopped and freed ${Math.round(afterReleaseMs)} ms after its release)`}`);
    }
  }

  private forgetInstances(set: SceneSetId): void {
    for (const key of [...this.instances.keys()]) if (key.startsWith(`${set}.`)) this.instances.delete(key);
  }

  // -------------------------------------------------------------------------
  // Size, scroll, story time
  // -------------------------------------------------------------------------

  /** Device pixel ratio for a CSS size: device, quality cap (and the adaptive DPR step), pixel budget. */
  private dprFor(width: number, height: number): number {
    const device = window.devicePixelRatio || 1;
    const budget = Math.sqrt(MAX_DRAWING_BUFFER_PX[this.quality.level] / Math.max(1, width * height));
    return Math.min(device, this.quality.dprCap, this.dprLimit, budget);
  }

  private resize(force = false): void {
    const renderer = this.renderer;
    if (!renderer || !this.post) return;
    const w = Math.max(1, this.host.clientWidth || window.innerWidth);
    const h = Math.max(1, this.host.clientHeight || window.innerHeight);
    const dpr = this.dprFor(w, h);
    if (!force && w === this.viewport.width && h === this.viewport.height && dpr === this.viewport.dpr) return;
    renderer.setPixelRatio(dpr);
    renderer.setSize(w, h, false);
    const size = renderer.getDrawingBufferSize(this.tmpSize);
    this.post.setSize(size.x, size.y);
    this.viewport = { width: w, height: h, dpr, aspect: w / h };
    this.capture?.invalidate();
  }

  private readonly tmpSize = new Vector2();

  private measureScroll(): void {
    if (this.url.capture) return;
    const width = window.innerWidth;
    const height = document.documentElement.scrollHeight;
    const prevMax = this.scrollMax;
    this.scrollMax = Math.max(0, height - window.innerHeight);
    const first = this.layoutWidth < 0;
    const layoutChanged = width !== this.layoutWidth || height !== this.layoutHeight;
    this.layoutWidth = width;
    this.layoutHeight = height;
    // Keep the story position across real layout changes (resize in the middle of a
    // transition). A mobile browser bar showing / hiding changes only innerHeight (the
    // vh-based track keeps its height): scrolling then would stop a fling, so the
    // mapping (t = scrollY / scrollMax × 59, shared with the overlay) just follows.
    if (first || !layoutChanged || prevMax === this.scrollMax || this.autoplay) return;
    if (this.pinnedT !== null) this.scrollToT(this.pinnedT);
    else if (prevMax > 0) this.scrollToT(this.lastScrollT, false);
  }

  private scrollT(): number {
    if (!(this.scrollMax > 0)) return 0;
    const t = clamp(window.scrollY / this.scrollMax, 0, 1) * VIDEO_DURATION;
    return Number.isFinite(t) ? t : 0;
  }

  private scrollToT(t: number, track = true): void {
    if (this.url.capture) return;
    const y = (clamp(t, 0, VIDEO_DURATION) / VIDEO_DURATION) * this.scrollMax;
    if (!Number.isFinite(y)) return;
    window.scrollTo({ top: y, left: 0, behavior: "instant" });
    if (track) this.expectedScrollY = window.scrollY;
  }

  private pin(t: number): void {
    this.pinnedT = clamp(t, 0, VIDEO_DURATION);
    this.scrollToT(this.pinnedT);
    this.t = this.targetT = this.pinnedT;
    this.snapNext = true;
  }

  /** A scroll we did not cause (user, scrollbar drag, anchor) since our last scrollTo. */
  private foreignScroll(): boolean {
    return this.expectedScrollY !== null && Math.abs(window.scrollY - this.expectedScrollY) > 2;
  }

  private computeTarget(dt: number): number {
    if (this.url.capture) return this.captureT;
    if (this.autoplay) {
      if (this.foreignScroll()) {
        // scrollbar drag (no wheel / touch / key event): the user takes over
        this.autoplay = false;
        this.expectedScrollY = null;
      } else {
        // the clock waits while the scenes it shows are still being prepared
        if (this.viewsSettled) this.autoT = Math.min(VIDEO_DURATION, this.autoT + dt);
        this.scrollToT(this.autoT);
        if (this.autoT >= VIDEO_DURATION) this.autoplay = false;
        this.snapNext = true;
        return this.autoT;
      }
    }
    if (this.pinnedT !== null) {
      // a scroll we did not cause releases the pin
      if (this.foreignScroll()) {
        this.pinnedT = null;
        this.expectedScrollY = null;
      } else {
        return this.pinnedT;
      }
    }
    const t = this.scrollT();
    this.lastScrollT = t;
    return t;
  }

  // -------------------------------------------------------------------------
  // Frame loop
  // -------------------------------------------------------------------------

  private readonly loop = (now: number): void => {
    if (this.disposed) return;
    this.rafId = requestAnimationFrame(this.loop);
    if (this.contextLost || document.hidden) {
      this.lastNow = -1;
      return;
    }
    const dt = this.lastNow < 0 ? 0 : Math.min(0.1, Math.max(0, (now - this.lastNow) / 1000));
    const gap = this.lastNow < 0;
    this.lastNow = now;
    const loopStart = performance.now();
    this.gpuTimer?.begin(this.frame + 1);
    // a slice of shader / texture warm-up, before the frame: its synchronous queries
    // (program first use) then find the GPU process idle instead of busy with this frame
    let warmBusy = false;
    try {
      warmBusy = this.pumpWarm();
    } catch (err) {
      console.error("[nature] warm-up failed", err);
      this.flushWarm();
    }
    this.stats.frameStart(now);
    this.frameNow = now;
    const cpuStart = performance.now();
    try {
      this.renderFrame(dt);
      this.errorCount = 0;
    } catch (err) {
      // a failing scene set is handled per view inside renderFrame; this is the frame
      // itself failing (post stack, renderer): nothing can render
      this.lastDrawn = false;
      this.errorCount++;
      if (this.errorCount <= 3) console.error("[nature] frame failed", err);
      if (this.errorCount > 60) {
        this.gpuTimer?.end();
        cancelAnimationFrame(this.rafId);
        NatureExperience.liveLoops--;
        this.rafId = 0;
        this.setStatus("error", String(err));
        return;
      }
    }
    this.gpuTimer?.end();
    if (this.renderer) this.stats.frameEnd(performance.now() - cpuStart, this.renderer);
    this.watchQuality(now, performance.now() - loopStart, warmBusy || gap);
    // this frame's share of build work: a task that starts once the frame has been rendered
    this.slicePacing?.frame(this.buildBudget());
    this.debugPanel?.update(now);
  };

  /** Build work allowed after this frame (ms), see BUILD_BUDGET_MS. */
  private buildBudget(): number {
    if (!this.lastDrawn) return BUILD_BUDGET_MS.empty;
    const res = this.lastRes;
    const director = this.director;
    if (!res || !director) return BUILD_BUDGET_MS.preload;
    if (!director.settled(res)) return BUILD_BUDGET_MS.needed;
    const near = transitionNear(this.t, this.travel, BUILD_BUDGET_MS.soonSec);
    if (near && (director.state(episodeConfig(near.from).set) === "preparing" || director.state(episodeConfig(near.to).set) === "preparing")) {
      return BUILD_BUDGET_MS.needed;
    }
    return BUILD_BUDGET_MS.preload;
  }

  // -------------------------------------------------------------------------
  // Adaptive quality
  // -------------------------------------------------------------------------

  /**
   * Feed one frame to the adaptive watcher (auto quality only). A window collects the
   * intervals of consecutive steady frames; any other frame starts it over.
   */
  private watchQuality(now: number, cpuMs: number, busy: boolean): void {
    if (!this.adaptActive) return;
    this.gpuTimer?.poll((frame, ms) => {
      this.gpuFrames[this.gpuIdx] = frame;
      this.gpuMs[this.gpuIdx] = ms;
      this.gpuIdx = (this.gpuIdx + 1) % this.gpuMs.length;
    });
    if (busy || !this.steadyFrame(now)) {
      this.adaptN = 0;
      this.adaptWindowStart = -1;
      return;
    }
    if (this.adaptWindowStart < 0) {
      // the first steady frame: its interval still spans the frame before
      this.adaptWindowStart = now;
      this.adaptFirstFrame = this.frame;
      this.adaptPrevNow = now;
      return;
    }
    const n = this.adaptN;
    if (n < this.adaptIntervals.length) {
      this.adaptIntervals[n] = now - this.adaptPrevNow;
      this.adaptCpu[n] = cpuMs;
      this.adaptN = n + 1;
    }
    this.adaptPrevNow = now;
    if (now - this.adaptWindowStart >= ADAPT.windowMs && this.adaptN >= ADAPT.minSamples) this.evaluateQuality(now);
  }

  /** A frame that says something about the steady load (nothing building or warming, no transition near). */
  private steadyFrame(now: number): boolean {
    if (!this.lastDrawn || now < this.adaptCooldownUntil || this.warmJobs.length > 0 || this.fadeStart.size > 0) return false;
    const res = this.lastRes;
    if (!res || res.transition) return false;
    if (transitionNear(this.t, this.travel, ADAPT.transitionLeadSec)) return false;
    const director = this.director;
    if (!director) return false;
    for (const v of res.views) if (director.state(v.set) !== "ready") return false;
    return !director.busy();
  }

  private evaluateQuality(now: number): void {
    const n = this.adaptN;
    const p95 = percentile(this.adaptIntervals, n, 0.95);
    const cpuP95 = percentile(this.adaptCpu, n, 0.95);
    // GPU time of the window's frames (median), when the timer query exists
    let gpuMs: number | null = null;
    let m = 0;
    for (let i = 0; i < this.gpuMs.length; i++) {
      if (this.gpuFrames[i] >= this.adaptFirstFrame && this.gpuFrames[i] <= this.frame) this.gpuScratch[m++] = this.gpuMs[i];
    }
    if (m >= n / 3) gpuMs = percentile(this.gpuScratch, m, 0.5);
    this.adaptLast = { at: Math.round(now), p95: round1(p95), cpuP95: round1(cpuP95), gpuMs: gpuMs === null ? null : round1(gpuMs), samples: n };
    this.adaptN = 0;
    this.adaptWindowStart = -1;
    const limit = this.adaptThreshold;
    const own = gpuMs === null || gpuMs > ADAPT.ownShare * limit || cpuP95 > ADAPT.ownShare * limit;
    if (!(p95 > limit && own)) return;
    const from = this.quality.level;
    const to = LOWER_QUALITY[from];
    if (this.adaptSteps >= ADAPT.maxSteps || (!to && !this.canLowerDpr())) {
      this.stopAdapt();
      return;
    }
    const step = ++this.adaptSteps;
    // a lower preset, or (inside low) the last fallback: the same preset at a lower DPR cap
    const target = to ?? `${from} dpr ${ADAPT.lastDprCap.toFixed(1)}`;
    const entry: AdaptLogEntry = { at: Math.round(now), t: Math.round(this.t * 100) / 100, from, to: to ?? from, p95: round1(p95), cpuP95: round1(cpuP95), gpuMs: gpuMs === null ? null : round1(gpuMs) };
    if (!to) entry.dprCap = ADAPT.lastDprCap;
    this.adaptLog.push(entry);
    const gpuText = gpuMs === null ? "" : `, GPU ${gpuMs.toFixed(1)} ms`;
    const reason = `adapt: frame p95 ${p95.toFixed(1)} ms > ${limit} ms${gpuText} → ${target} (step ${step})`;
    if (to) this.applyQuality(to, reason, false);
    else this.limitDpr(ADAPT.lastDprCap, reason);
    if (this.url.debug) console.info(`[nature] quality ${from} → ${target}: frame p95 ${p95.toFixed(1)} ms over ${n} frames${gpuText}`);
    this.adaptCooldownUntil = now + ADAPT.cooldownMs;
    if (this.adaptSteps >= ADAPT.maxSteps || (!LOWER_QUALITY[this.quality.level] && !this.canLowerDpr())) this.stopAdapt();
  }

  /** The last step inside `low` would make the drawing buffer smaller (the DPR is above its cap). */
  private canLowerDpr(): boolean {
    return this.dprLimit > ADAPT.lastDprCap && this.dprFor(this.viewport.width, this.viewport.height) > ADAPT.lastDprCap;
  }

  /**
   * The last adaptive step (inside `low`): cap the DPR at `cap` for the current preset. Its
   * budgets, MSAA, DOF and the prepared sets stay; like a preset step the canvas is resized by
   * the next `renderFrame`, at its start, right before it draws (`resizePending`).
   */
  private limitDpr(cap: number, reason: string): void {
    this.qualityReason = reason;
    if (cap >= this.dprLimit) return;
    this.dprLimit = cap;
    this.resizePending = true;
    this.changed();
  }

  private stopAdapt(): void {
    this.adaptActive = false;
    this.gpuTimer?.dispose();
    this.gpuTimer = null;
  }

  /**
   * Switch the quality preset. `rebuildAll` (API, debug panel) drops every prepared set;
   * otherwise (adaptive step-down) the render side changes at once (MSAA, bloom levels,
   * DOF, anisotropy) and only the sets not on screen are dropped, to be prepared again
   * with the new budgets: the picture never goes blank.
   *
   * The DPR cap / pixel budget of the new level is applied by the next `renderFrame`, at
   * its start, right before it draws: resizing the canvas clears its drawing buffer, and
   * the step-down runs after this frame was drawn (`watchQuality` follows `renderFrame`),
   * so a resize here would present this frame empty — the CSS backdrop with no scene.
   */
  private applyQuality(level: QualityLevel, reason: string, rebuildAll: boolean): void {
    const preset = QUALITY_PRESETS[level];
    this.qualityReason = reason;
    if (!preset || preset === this.quality) return;
    this.quality = preset;
    this.post?.setQuality(preset);
    this.assets?.setAnisotropy(preset.anisotropy);
    this.resizePending = true;
    if (rebuildAll) this.director?.invalidateAll();
    else this.director?.invalidateExcept(new Set((this.lastRes?.views ?? []).map((v) => v.set)));
    this.changed();
  }

  private renderFrame(dt: number): void {
    const renderer = this.renderer;
    const post = this.post;
    const director = this.director;
    if (!renderer || !post || !director) return;
    this.frame++;

    // size: observed host resizes, DPR changes (zoom / monitor) — right before drawing
    let resized = false;
    if (this.resizePending) {
      this.resizePending = false;
      this.resize();
      resized = true;
    } else if (this.dprFor(this.viewport.width, this.viewport.height) !== this.viewport.dpr) {
      this.resize(true);
      resized = true;
    }

    // ambient clock
    if (this.frozenTime === null) this.timeSec += dt;
    const timeSec = this.frozenTime ?? this.timeSec;
    const deltaSec = this.frozenTime === null ? dt : 0;

    // story time
    const prevT = this.t;
    const target = this.computeTarget(dt);
    this.targetT = target;
    if (target > prevT + 1e-4) this.travel = 1;
    else if (target < prevT - 1e-4) this.travel = -1;
    // a jump over a whole scene set (anchor link, setT, fling) snaps: damping through it
    // would prepare and render every set in between
    if (this.url.capture || this.snapNext || setsApart(prevT, target) >= 2) {
      this.t = target;
      this.snapNext = false;
    } else {
      this.t = damp(this.t, target, lambdaForResponse(SCROLL_RESPONSE_SEC), dt);
      if (Math.abs(this.t - target) < 1e-4) this.t = target;
    }

    const res = director.resolve(this.t, this.reducedMotion);
    this.lastRes = res;
    const moving = this.t !== prevT || this.t !== target;
    if (moving) this.lastMoveAt = performance.now();
    // a preload (build + warm-up of the next set) does not start while a transition that
    // does not show it plays or is about to start, nor while a set on screen fades in
    director.sync(res, !this.url.capture, this.travel, (set) => !this.fadingIn() && this.mayWarm(set, WARM.preloadLeadSec));

    const frame = this.fillFrame(this.frameState, res.t, timeSec, deltaSec);
    frame.globalProgress = res.globalProgress;
    frame.episode = res.episode.id;
    frame.sceneProgress = res.sceneProgress;
    frame.pointerNdc.x = this.url.capture ? 0 : this.pointer.x;
    frame.pointerNdc.y = this.url.capture ? 0 : this.pointer.y;
    this.wind.update(timeSec, this.flags.wind, this.reducedMotion);
    this.applyWireframe();

    renderer.info.reset();
    const toneMapping = TONE_MAPPERS[this.toneMappingName];
    renderer.toneMapping = toneMapping;
    const layerMask = this.layers.maskFor(this.flags.layers);

    const wasDrawn = this.lastDrawn;
    const drawn = this.flags.post ? this.drawPost(res, frame, toneMapping, layerMask) : this.drawStraight(res, frame, toneMapping, layerMask);
    if (drawn) {
      this.sceneDrawn = true;
      this.lastSceneAt = this.frameNow;
    }
    this.lastDrawn = drawn;
    // moving through sets that are not ready: keep the last picture up for a moment (HOLD_MS;
    // not after a resize, which cleared the canvas)
    this.holding =
      !drawn && !this.url.capture && !resized && moving && (wasDrawn || this.holding) && this.frameNow - this.lastSceneAt < HOLD_MS;
    if (!drawn && !this.holding) {
      const look = this.baseLook(res.views[0].local, this.baseLooks[0]);
      post.renderEmpty(look, toneMapping, this.emptyScene, this.emptyCamera, this.flags.post ? this.flags : null);
    }

    // after drawing: a view may have failed in this frame
    let ready = 0;
    let live = 0;
    for (const v of res.views) {
      const s = director.state(v.set);
      if (s !== "failed") live++;
      if (s === "ready") ready++;
    }
    this.viewsSettled = director.settled(res);
    const settled =
      this.viewsSettled &&
      ready === live &&
      (this.assets?.pending ?? 0) === 0 &&
      this.fadeStart.size === 0 &&
      this.t === target; // the damping snaps onto the target once within 1e-4
    this.capture?.frameRendered(settled);
  }

  private fillFrame(frame: FrameState, t: number, timeSec: number, deltaSec: number): FrameState {
    frame.timeSec = timeSec;
    frame.deltaSec = deltaSec;
    frame.t = t;
    frame.reducedMotion = this.reducedMotion;
    frame.viewport = this.viewport;
    frame.quality = this.quality;
    frame.capture = this.url.capture;
    frame.frame = this.frame;
    return frame;
  }

  /**
   * Post mode: overrides → which images the composite needs → per view look + update
   * (+ render when needed) → composite. A view that throws disables its set only.
   * Returns false when nothing could be drawn.
   */
  private drawPost(res: Resolution, frame: FrameState, toneMapping: ToneMapping, layerMask: number): boolean {
    const director = this.director as SceneDirector;
    const post = this.post as PostStack;
    const views = res.views;
    const n = views.length;
    const scenes = this.frameScenes;
    scenes[0] = director.scene(views[0].set);
    scenes[1] = n > 1 ? director.scene(views[1].set) : null;

    // fade-in ramps (FADE_IN_MS): a view whose ramp waits for another one is left out, a
    // view whose ramp starts next frame is drawn at weight 0; one view alone that is fading
    // in is drawn over the background it replaces
    const presence = this.fadePresence;
    const first = this.fadeFirst;
    this.trackFades(views);
    for (let i = 0; i < 2; i++) {
      const p = i < n && scenes[i] ? this.presenceOf(views, i) : 1;
      first[i] = p < 0;
      presence[i] = Math.max(0, p);
      if (p === 0) scenes[i] = null;
    }
    const solo = scenes[0] && !scenes[1] ? 0 : !scenes[0] && scenes[1] ? 1 : -1;
    if (solo >= 0 && presence[solo] < 1) return this.drawFadeIn(res, frame, solo, presence[solo], toneMapping, layerMask);

    let params: TransitionParams | null = null;
    if (n === 2 && scenes[0] && scenes[1] && res.transition) {
      this.frameViews = views;
      params =
        presence[0] < 1 || presence[1] < 1
          ? this.fadedTransitionParams(res, frame, presence[0], presence[1])
          : director.transitionParams(res, frame, scenes, this.onOverrideError);
      if (!scenes[0] || !scenes[1]) params = null;
    }
    let needA = true;
    let needB = true;
    if (params) {
      transitionInputs(params, this.inputs);
      // a picture's first draw happens at weight 0, before its ramp
      needA = this.inputs.a || first[0];
      needB = this.inputs.b || first[1];
    }

    post.beginFrame();
    for (let i = 0; i < n; i++) {
      const scene = scenes[i];
      if (!scene) continue;
      const plan = views[i];
      try {
        const look = this.prepareView(scene, plan.local, frame, i);
        if (i === 0 ? needA : needB) {
          post.renderView(i, this.postView(i, scene, look, params !== null && plan.local.transparentBackground, layerMask), this.flags, toneMapping);
        } else {
          post.setViewExposure(i, look.exposure);
        }
      } catch (err) {
        this.failView(views, i, err);
      }
    }

    const okA = scenes[0] !== null;
    const okB = n > 1 && scenes[1] !== null;
    if (!okA && !okB) return false;
    let transition = params;
    if (transition && !(okA && okB)) {
      // a view failed in this frame: show the other one alone (render it if it was skipped)
      transition = null;
      const i = okA ? 0 : 1;
      if (!(i === 0 ? needA : needB)) {
        const scene = scenes[i] as NatureScene;
        try {
          post.renderView(i, this.postView(i, scene, this.viewLooks[i], false, layerMask), this.flags, toneMapping);
        } catch (err) {
          this.failView(views, i, err);
          return false;
        }
      }
    }
    if (transition) lerpLook(this.viewLooks[0], this.viewLooks[1], transition.k, this.globalLook);
    else copyLook(this.viewLooks[okA ? 0 : 1], this.globalLook);
    post.finish({ transition, globalLook: this.globalLook, viewLooks: transition ? this.viewLooks : null, flags: this.flags, toneMapping });
    return true;
  }

  /**
   * Fade-in bookkeeping, once per post frame (FADE_IN_MS). A set that is ready now and was
   * missing from the last frame's views starts its ramp — after the one still running, if
   * any. Finished ramps and sets that left the views are forgotten. Off in capture mode.
   */
  private trackFades(views: Resolution["views"]): void {
    if (this.url.capture) {
      this.clearFades();
      return;
    }
    const director = this.director as SceneDirector;
    const now = this.frameNow;
    const n = views.length;
    const twin = n === 2 && views[0].set === views[1].set;
    for (const [set, start] of this.fadeStart) {
      const shown = views[0].set === set || (n > 1 && views[1].set === set);
      if (!shown) {
        this.fadeStart.delete(set);
      } else if (Number.isNaN(start)) {
        this.fadeStart.set(set, now); // drawn once at weight 0 last frame: the ramp starts
      } else if (now - start >= (twin ? 2 : 1) * FADE_IN_MS) {
        // (the two views of one set — reduced-motion camera crossfade — ramp one after the other)
        this.fadeStart.delete(set);
      }
    }
    const missing = this.fadeMissingNext;
    missing.clear();
    for (const v of views) {
      const state = director.state(v.set);
      if (state === "ready") {
        // after a hold the background was never shown: the set cuts in
        if (this.fadeMissing.has(v.set) && !this.fadeStart.has(v.set) && !this.holding) {
          let after = -Infinity;
          for (const other of this.fadeStart.values()) after = Math.max(after, (Number.isNaN(other) ? now : other) + FADE_IN_MS);
          this.fadeStart.set(v.set, after > now ? after : Number.NaN);
        }
      } else if (state !== "failed") {
        missing.add(v.set);
      }
    }
    this.fadeMissingNext = this.fadeMissing;
    this.fadeMissing = missing;
  }

  /** A set on screen is fading in, or was missing in the last frame (it may turn ready now). */
  private fadingIn(): boolean {
    return this.fadeStart.size > 0 || this.fadeMissing.size > 0;
  }

  private clearFades(): void {
    this.fadeStart.clear();
    this.fadeMissing.clear();
  }

  /**
   * Presence 0–1 of view `i` in this frame: 1 not fading, 0 waiting for another ramp (not
   * drawn), −1 drawn at weight 0 (its first draw, or the frame its ramp starts).
   */
  private presenceOf(views: Resolution["views"], i: number): number {
    const start = this.fadeStart.get(views[i].set);
    if (start === undefined) return 1;
    const delay = i === 1 && views[0].set === views[1].set ? FADE_IN_MS : 0;
    if (Number.isNaN(start)) return delay > 0 ? 0 : -1;
    const x = (this.frameNow - start - delay) / FADE_IN_MS;
    if (x <= 0) return x < 0 ? 0 : -1;
    return x >= 1 ? 1 : x * x * (3 - 2 * x);
  }

  /**
   * Composite of a transition whose one picture is fading in: with the presence, its time
   * runs from the side where that picture has no weight (the window's start for the
   * incoming, its end for the outgoing picture) to the real time — a crossfade weight, a
   * wipe edge, a slide offset, a reveal catching up. The scenes' overrides are asked at
   * that time too (they may follow `frame.t`, not only the transition's progress). The
   * views themselves render at the real time.
   */
  private fadedTransitionParams(res: Resolution, frame: FrameState, presenceA: number, presenceB: number): TransitionParams | null {
    const director = this.director as SceneDirector;
    const tr = res.transition;
    if (!tr) return null;
    const incoming = presenceB < 1;
    const from = incoming ? tr.start : tr.end;
    const t = from + (res.t - from) * (incoming ? presenceB : presenceA);
    const at = Object.assign(this.fadeFrame, frame);
    at.t = t;
    return director.transitionParams({ ...res, t, transition: director.transitionAt(res, t) }, at, this.frameScenes, this.onOverrideError);
  }

  /**
   * View `i` alone, fading in over the background it replaces — the look's background,
   * drawn as `renderEmpty` draws it (base look of view 0, no DOF): background in slot 0,
   * the view in slot 1, crossfaded by `presence`; bloom and grade follow the blended look.
   * Returns false when the view failed (the caller then draws the background alone).
   */
  private drawFadeIn(res: Resolution, frame: FrameState, i: number, presence: number, toneMapping: ToneMapping, layerMask: number): boolean {
    const post = this.post as PostStack;
    const views = res.views;
    const scene = this.frameScenes[i] as NatureScene;
    post.beginFrame();
    try {
      const look = this.prepareView(scene, views[i].local, frame, i);
      const bgLook = this.baseLook(views[0].local, this.fadeLook);
      const bg = this.postViews[0];
      bg.scene = this.emptyScene;
      bg.camera = this.emptyCamera;
      bg.look = bgLook;
      bg.transparentBackground = false;
      bg.layerMask = 1;
      post.renderView(0, bg, FADE_BACKGROUND_FLAGS, toneMapping);
      post.renderView(1, this.postView(1, scene, look, false, layerMask), this.flags, toneMapping);
      this.fadeParams.k = presence;
      lerpLook(bgLook, look, presence, this.globalLook);
    } catch (err) {
      this.failView(views, i, err);
      return false;
    }
    post.finish({ transition: this.fadeParams, globalLook: this.globalLook, viewLooks: null, flags: this.flags, toneMapping });
    return true;
  }

  /** `post=0`: the dominant ready view straight to the canvas. */
  private drawStraight(res: Resolution, frame: FrameState, toneMapping: ToneMapping, layerMask: number): boolean {
    const director = this.director as SceneDirector;
    const post = this.post as PostStack;
    this.clearFades();
    const views = res.views;
    const scenes = this.frameScenes;
    scenes[0] = director.scene(views[0].set);
    scenes[1] = views.length > 1 ? director.scene(views[1].set) : null;
    let i = scenes[0] && scenes[1] ? ((res.transition?.k ?? 0) >= 0.5 ? 1 : 0) : scenes[0] ? 0 : 1;
    for (let attempt = 0; attempt < 2; attempt++) {
      const scene = scenes[i];
      if (!scene) return false;
      try {
        const look = this.prepareView(scene, views[i].local, frame, i);
        post.renderStraight(this.postView(i, scene, look, false, layerMask), toneMapping);
        return true;
      } catch (err) {
        this.failView(views, i, err);
        i = 1 - i;
      }
    }
    return false;
  }

  /** View `i` threw: disable its set for good (logged once), drop it from this frame. */
  private failView(views: Resolution["views"], i: number, err: unknown): void {
    const set = views[i].set;
    this.director?.fail(set, err);
    for (let j = 0; j < views.length; j++) if (views[j].set === set) this.frameScenes[j] = null;
  }

  private postView(slot: number, scene: NatureScene, look: LookParams, transparent: boolean, layerMask: number): PostView {
    const v = this.postViews[slot];
    v.scene = scene.scene;
    v.camera = scene.camera;
    v.look = look;
    v.transparentBackground = transparent;
    v.layerMask = layerMask;
    return v;
  }

  /** SceneConfig look for a view (blended inside same-set camera windows). */
  private baseLook(local: SceneLocal, out: LookParams): LookParams {
    if (local.episodeBlend) return lerpLook(episodeLook(local.episodeBlend.from), episodeLook(local.episodeBlend.to), local.episodeBlend.k, out);
    return copyLook(episodeLook(local.episode), out);
  }

  private prepareView(scene: NatureScene, local: SceneLocal, frame: FrameState, index: number): LookParams {
    const base = this.baseLook(local, this.baseLooks[index]);
    const look = copyLook(scene.getLook(frame, local, base), this.viewLooks[index]);
    scene.update(frame, local, look);
    const fog = scene.scene.fog as Fog | null;
    if (fog && (fog as Fog).isFog) {
      fog.color.copy(look.fog.color);
      fog.near = look.fog.near;
      fog.far = look.fog.far;
    }
    return look;
  }

  // -------------------------------------------------------------------------
  // Warm-up: shader compile, program first use, texture upload, off-screen draw
  // -------------------------------------------------------------------------

  /**
   * SceneDirector `warm` hook, after `prepare()`. Queues the set's warm-up, which the
   * frame loop advances a little every frame (`pumpWarm`) instead of in one task:
   *
   *  1. compile: every material of `scene.scene` and of the set's `warmTargets()`
   *     (visible or not) against the target it is drawn into (post: a linear render
   *     target; post=0: the canvas with tone mapping), plus the depth programs three's
   *     shadow map will use for its shadow casters — a few objects per frame, one new
   *     program per frame without KHR_parallel_shader_compile;
   *  2. settle: once the driver reports a program linked (non-blocking with the
   *     extension), its first use (uniform locations, link log) — one per step — and the
   *     upload of every texture its materials use, a piece per step (a large image in
   *     strips of ≈ 1 MB, `AssetRegistry.uploadStep`);
   *  3. for a preloaded set that is not on screen, one off-screen draw into the scene
   *     target (never presented), which uploads geometry and builds VAOs ahead of its
   *     first frame.
   *
   * A preloaded set is only advanced while no transition plays or is about to start
   * (`mayWarm`); a set the current views need is advanced every frame with a larger
   * budget. The returned promise resolves when the set may be drawn.
   */
  private warmSet(set: SceneSetId, scene: NatureScene, stale: () => boolean): Promise<void> {
    if (!this.renderer || !this.post || this.disposed || this.contextLost) return Promise.resolve();
    return new Promise<void>((resolve, reject) => {
      const job = new WarmJob(set, scene, stale, resolve, reject);
      this.collectWarm(job);
      this.warmJobs.push(job);
    });
  }

  /** Compile units of a set: one per distinct (materials, object kind, geometry layout). */
  private collectWarm(job: WarmJob): void {
    const scene = job.scene;
    const targets: { scene: Scene; camera: Camera; offscreen: boolean }[] = [{ scene: scene.scene, camera: scene.camera, offscreen: this.flags.post }];
    try {
      for (const w of scene.warmTargets?.() ?? []) targets.push({ scene: w.scene, camera: w.camera ?? scene.camera, offscreen: w.offscreen !== false });
    } catch (err) {
      // the set's draw surfaces the error; compile what is known
      if (this.url.debug) console.warn(`[nature] warmTargets() of "${job.set}" threw`, err);
    }
    const seen = new Set<string>();
    for (const tg of targets) {
      // which depth programs the shadow map will ask for (spot / directional vs point lights)
      let depth = false;
      let distance = false;
      tg.scene.traverseVisible((o) => {
        const light = o as Light;
        if (!light.isLight || !light.castShadow) return;
        if ((o as PointLight).isPointLight) distance = true;
        else depth = true;
      });
      tg.scene.traverse((o) => {
        const mesh = o as Mesh;
        if (!(mesh.isMesh || (o as Points).isPoints || (o as Line).isLine || (o as Sprite).isSprite) || !mesh.material) return;
        const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        const sig = objectSignature(o);
        const key = `${tg.scene.id}|${mats.map((m) => m.uuid).join(",")}|${sig}`;
        if (!seen.has(key)) {
          seen.add(key);
          job.units.push({ object: o, material: null, target: tg.scene, camera: tg.camera, offscreen: tg.offscreen, noFog: false });
          for (const m of mats) job.materials.add(m);
        }
        if (!mesh.castShadow || (o as Sprite).isSprite || !(depth || distance)) return;
        for (const m of mats) {
          for (const kind of SHADOW_KINDS) {
            if (kind === "depth" ? !depth : !distance) continue;
            const dkey = `${tg.scene.id}|${kind}|${depthVariantKey(m)}|${sig}`;
            if (seen.has(dkey)) continue;
            seen.add(dkey);
            const stand = depthStandIn(kind, m);
            job.depthMaterials.push(stand);
            // the shadow pass draws into the shadow map (a render target) without a scene: no fog
            job.units.push({ object: o, material: stand, target: tg.scene, camera: tg.camera, offscreen: true, noFog: true });
          }
        }
      });
    }
  }

  /** Compile one unit (the object alone, not its children); returns the programs it created. */
  private compileUnit(job: WarmJob, u: WarmUnit): number {
    const r = this.renderer as WebGLRenderer;
    const post = this.post as PostStack;
    const programs = r.info.programs;
    const before = programs ? programs.length : 0;
    const proxy = Object.create(u.object) as Object3D;
    proxy.children = [];
    if (u.material) (proxy as Mesh).material = u.material;
    const cam = u.camera;
    const mask = cam.layers.mask;
    const prevTarget = r.getRenderTarget();
    const prevTone = r.toneMapping;
    const fog = u.target.fog;
    try {
      cam.layers.enableAll();
      r.toneMapping = TONE_MAPPERS[this.toneMappingName];
      r.setRenderTarget(u.offscreen ? post.compileTarget() : null);
      if (u.noFog) u.target.fog = null;
      r.compile(proxy, cam, u.target);
    } catch {
      // compile errors surface on the first draw, with full logs
    } finally {
      u.target.fog = fog;
      r.setRenderTarget(prevTarget);
      r.toneMapping = prevTone;
      cam.layers.mask = mask;
    }
    let created = 0;
    if (programs) {
      for (let i = before; i < programs.length; i++) {
        job.programs.push(programs[i]);
        created++;
      }
    }
    return created;
  }

  /** Every texture the job's materials use (after compile: onBeforeCompile uniforms exist). */
  private collectTextures(job: WarmJob): void {
    const r = this.renderer as WebGLRenderer;
    const seen = new Set<Texture>();
    const add = (v: unknown): void => {
      if (Array.isArray(v)) {
        for (const x of v) add(x);
        return;
      }
      const t = v as Texture | null;
      if (t && t.isTexture && !seen.has(t)) {
        seen.add(t);
        job.textures.push(t);
      }
    };
    const fromUniforms = (u: Record<string, { value: unknown } | undefined> | undefined): void => {
      if (u) for (const k in u) add(u[k]?.value);
    };
    for (const m of [...job.materials, ...job.depthMaterials]) {
      const rec = m as unknown as Record<string, unknown>;
      for (const k in rec) add(rec[k]); // map, normalMap, … of built-in materials
      fromUniforms((m as ShaderMaterial).uniforms);
      fromUniforms((r.properties.get(m) as { uniforms?: Record<string, { value: unknown }> }).uniforms);
    }
  }

  /** One step of a texture's upload: the whole texture, or a ≈ 1 MB strip of a large one (`AssetRegistry.uploadStep`). */
  private uploadTexture(t: Texture): UploadStep {
    return this.assets ? this.assets.uploadStep(t) : "none";
  }

  /** Whether a set's warm-up may advance now (and whether a preload may start). */
  private mayWarm(set: SceneSetId, lead: number): boolean {
    if (this.url.capture || this.onScreen(set)) return true;
    // the story rests (no scroll, no autoplay movement): nothing is about to play
    if (performance.now() - this.lastMoveAt > WARM.restMs) return true;
    const tr = this.lastRes?.transition;
    if (tr) return tr.fromSet === set || tr.toSet === set;
    // about to enter a window: only the sets that window shows (they must be ready by then)
    for (const t of [this.t, this.targetT]) {
      const near = transitionNear(t, this.travel, lead);
      if (near && episodeConfig(near.from).set !== set && episodeConfig(near.to).set !== set) return false;
    }
    return true;
  }

  private onScreen(set: SceneSetId): boolean {
    return this.lastRes?.views.some((v) => v.set === set) ?? false;
  }

  /**
   * Advance the queued warm-ups within this frame's budget (called by the frame loop
   * right before the frame is drawn). Sets on screen first; then the first use of any
   * other linked program (post passes compiled at start: DOF, bloom, the grade variant
   * with bloom, the transition composite).
   */
  private pumpWarm(): boolean {
    const jobs = this.warmJobs;
    if (!this.renderer || this.contextLost || this.disposed) return false;
    if (this.warmDrains.length > 0) this.pumpDrains();
    const frameStart = performance.now();
    let busy = jobs.length > 0;
    const counters = { programs: 0, uses: 0, textures: 0 };
    let draws = 0;
    const ordered = [...jobs].sort((a, b) => Number(this.onScreen(b.set)) - Number(this.onScreen(a.set)));
    for (const job of ordered) {
      if (job.stale()) {
        this.finishWarm(job, true);
        continue;
      }
      const needed = this.onScreen(job.set);
      if (!this.mayWarm(job.set, WARM.leadSec)) continue;
      const budget = needed ? WARM.budgetMs.needed : WARM.budgetMs.preload;
      const limits = {
        programs: this.parallelCompile ? (needed ? WARM.programs.needed : WARM.programs.preload) : WARM.programs.serial,
        uses: this.parallelCompile ? (needed ? WARM.uses.needed : WARM.uses.preload) : WARM.uses.serial,
        textures: needed ? WARM.textures.needed : WARM.textures.preload,
      };
      const jobStart = performance.now();
      let worked = false;
      while (performance.now() - frameStart < budget) {
        if (job.phase === "draw" && (draws > 0 || performance.now() - frameStart > 1)) break; // the draw gets a frame of its own
        const s0 = performance.now();
        let step: WarmStepKind;
        try {
          step = this.warmStep(job, counters, limits);
        } catch (err) {
          // the set's update / draw threw during the off-screen draw: it fails like a prepare
          this.finishWarm(job, true, err);
          break;
        }
        const ms = performance.now() - s0;
        if (step === "wait") break;
        worked = true;
        if (step === "yield") break;
        if (step !== "phase") {
          job.steps++;
          if (ms > job.maxStepMs) job.maxStepMs = ms;
          if (ms > job.maxByKind[step]) job.maxByKind[step] = ms;
        }
        if (step === "draw") {
          draws++;
          job.drawMs = ms;
        }
        if (job.phase === "done") {
          this.finishWarm(job, false);
          break;
        }
      }
      if (worked) {
        const ms = performance.now() - jobStart;
        job.frames++;
        job.workMs += ms;
        if (ms > job.maxFrameMs) job.maxFrameMs = ms;
      }
    }
    // leftovers (a job adopts the untouched programs when it gets to its first uses): only
    // after the first load, with no job pending, not in a transition
    const quiet = jobs.length === 0 && this.sceneDrawn;
    const resting = this.url.capture || !this.lastRes?.transition || performance.now() - this.lastMoveAt > WARM.restMs;
    if (quiet && resting && performance.now() - frameStart < WARM.budgetMs.preload && this.touchPrograms(frameStart, WARM.budgetMs.preload) > 0) busy = true;
    const spent = performance.now() - frameStart;
    if (spent > this.warmFrameMaxMs) this.warmFrameMaxMs = spent;
    return busy;
  }

  /**
   * The renderer's programs that nobody has used yet (the post passes compiled at start:
   * DOF, bloom, the grade with bloom, the composite; programs of a set still in its
   * warm-up) join this job's first uses, so the frame that first shows the set does not
   * do them.
   */
  private adoptPrograms(job: WarmJob): void {
    const programs = this.renderer?.info.programs as WarmProgram[] | null | undefined;
    if (!programs) return;
    for (const p of programs) {
      if (this.touched.has(p) || p.program === undefined || job.programs.includes(p)) continue;
      job.programs.push(p);
    }
  }

  /**
   * First use (link status, logs, uniform locations — synchronous, otherwise done by the
   * first draw) of programs that finished linking and were not used yet, within `budget`.
   */
  private touchPrograms(frameStart: number, budget: number): number {
    const programs = this.renderer?.info.programs as WarmProgram[] | null | undefined;
    if (!programs) return 0;
    // nothing while any program still links (the queries would queue behind it)
    for (const p of programs) {
      if (this.touched.has(p) || p.program === undefined) continue;
      if (p.isReady && !p.isReady()) return 0;
    }
    let n = 0;
    const limit = WARM.uses.preload;
    for (const p of programs) {
      if (this.touched.has(p)) continue;
      if (p.program === undefined) continue; // destroyed (released meanwhile)
      if (n >= limit || performance.now() - frameStart >= budget) break;
      const s0 = performance.now();
      p.getUniforms();
      const ms = performance.now() - s0;
      this.touched.add(p);
      if (ms < 0.05) continue; // a draw had used it already (cached)
      this.touchedCount++;
      if (ms > this.touchMaxMs) {
        this.touchMaxMs = ms;
        this.touchMaxAt = s0;
      }
      n++;
    }
    return n;
  }

  /**
   * One unit of warm-up work: a compile (one object), a program's first use, a texture
   * upload, the off-screen draw, a phase change — or "wait" (nothing possible this frame).
   */
  private warmStep(job: WarmJob, counters: { programs: number; uses: number; textures: number }, limits: { programs: number; uses: number; textures: number }): WarmStepKind {
    if (job.phase === "compile") {
      if (job.unit >= job.units.length) {
        this.collectTextures(job);
        this.adoptPrograms(job);
        job.phase = "settle";
        // first uses from the next frame on: their synchronous queries would wait for
        // the GPU process to get through this frame's compiles first
        return counters.programs > 0 ? "yield" : "phase";
      }
      if (counters.programs >= limits.programs) return "wait";
      const created = this.compileUnit(job, job.units[job.unit++]);
      counters.programs += created;
      job.programCount += created;
      return "compile";
    }
    if (job.phase === "settle") {
      // first use of a linked program: link status, logs, uniform locations (otherwise
      // done — synchronously — by the set's first draw); only once all of the set's
      // programs have linked, and not after compiles or uploads in the same frame (the
      // synchronous queries would wait for them)
      let linking = false;
      for (let i = 0; i < job.programs.length; i++) {
        const p = job.programs[i] as WarmProgram | null;
        if (!p) continue;
        if (p.program === undefined) {
          job.programs[i] = null; // released meanwhile
          continue;
        }
        if (p.isReady && !p.isReady()) linking = true;
      }
      const canUse = !linking && counters.programs === 0 && counters.textures === 0 && counters.uses < limits.uses;
      for (let i = 0; canUse && i < job.programs.length; i++) {
        const p = job.programs[i] as WarmProgram | null;
        if (!p) continue;
        job.programs[i] = null;
        if (this.touched.has(p)) continue;
        const s0 = performance.now();
        p.getUniforms();
        this.touched.add(p);
        if (performance.now() - s0 < 0.05) continue; // a draw had used it already (cached)
        this.touchedCount++;
        counters.uses++;
        return "use";
      }
      const used = job.programs.every((p) => p === null);
      while (job.texture < job.textures.length) {
        if (counters.textures >= limits.textures) return "wait";
        // a large texture takes several steps (strips), on the next one only when complete
        const step = this.uploadTexture(job.textures[job.texture]);
        if (step !== "partial") job.texture++;
        if (step === "none") continue;
        counters.textures++;
        if (step === "done") job.textureCount++;
        return "texture";
      }
      if (!used) return "wait";
      job.phase = this.flags.post && !this.onScreen(job.set) ? "draw" : "done";
      return "phase";
    }
    if (job.phase === "draw") {
      job.phase = "done";
      // on screen meanwhile: the frame draws it anyway
      if (!this.flags.post || this.onScreen(job.set)) return "phase";
      this.warmDraw(job.set, job.scene);
      return "draw";
    }
    return "phase";
  }

  /**
   * Remove a job from the queue, keep its depth stand-ins with the set and let the set go
   * ready — or, with `error`, fail it (the director disposes and reports it). An aborted
   * job whose programs are still linking ends once they have linked (`warmDrains`; not
   * with `drain` false: context lost, engine disposed).
   */
  private finishWarm(job: WarmJob, aborted: boolean, error?: unknown, drain = true): void {
    const i = this.warmJobs.indexOf(job);
    if (i < 0) return;
    this.warmJobs.splice(i, 1);
    if (aborted && drain && this.linking(job)) {
      this.warmDrains.push({ job, error, since: performance.now() });
      return;
    }
    this.endWarm(job, aborted, error, null);
  }

  /** Whether the driver still links a program `job` compiled (only with KHR_parallel_shader_compile). */
  private linking(job: WarmJob): boolean {
    if (!this.parallelCompile || this.contextLost || this.disposed) return false;
    for (const p of job.programs as (WarmProgram | null)[]) {
      if (p && p.program !== undefined && p.isReady && !p.isReady()) return true;
    }
    return false;
  }

  /** End the drained warm-ups whose programs have linked, or that waited `WARM.drainMaxMs`. */
  private pumpDrains(): void {
    const now = performance.now();
    for (let i = this.warmDrains.length - 1; i >= 0; i--) {
      const d = this.warmDrains[i];
      if (now - d.since < WARM.drainMaxMs && this.linking(d.job)) continue;
      this.warmDrains.splice(i, 1);
      this.endWarm(d.job, true, d.error, now - d.since);
    }
  }

  /** Dispose an aborted job's stand-ins (or keep them with the set), log it, settle its promise. */
  private endWarm(job: WarmJob, aborted: boolean, error: unknown, drainMs: number | null): void {
    if (aborted) {
      for (const m of job.depthMaterials) m.dispose();
    } else {
      // the stand-ins hold the shadow programs until the shadow map has used them
      const old = this.warmMaterials.get(job.set);
      this.warmMaterials.set(job.set, job.depthMaterials);
      if (old && old !== job.depthMaterials) for (const m of old) m.dispose();
    }
    const now = performance.now();
    this.warmLog.push({
      set: job.set,
      aborted,
      at: Math.round(job.started),
      end: Math.round(now),
      ms: Math.round(now - job.started),
      frames: job.frames,
      steps: job.steps,
      units: job.units.length,
      programs: job.programCount,
      textures: job.textureCount,
      workMs: Math.round(job.workMs * 10) / 10,
      maxFrameMs: Math.round(job.maxFrameMs * 10) / 10,
      maxStepMs: Math.round(job.maxStepMs * 10) / 10,
      drawMs: Math.round(job.drawMs * 10) / 10,
      maxMs: {
        compile: Math.round(job.maxByKind.compile * 10) / 10,
        use: Math.round(job.maxByKind.use * 10) / 10,
        texture: Math.round(job.maxByKind.texture * 10) / 10,
        draw: Math.round(job.maxByKind.draw * 10) / 10,
      },
      ...(drainMs !== null ? { drainMs: Math.round(drainMs) } : {}),
    });
    if (this.warmLog.length > 24) this.warmLog.splice(0, this.warmLog.length - 24);
    if (error !== undefined) job.fail(error);
    else job.done();
  }

  /** Context lost / engine disposed: let every queued set go (its programs compile on first use). */
  private flushWarm(): void {
    for (const job of [...this.warmJobs]) this.finishWarm(job, true, undefined, false);
    const now = performance.now();
    for (const d of this.warmDrains.splice(0)) this.endWarm(d.job, true, d.error, now - d.since);
  }

  /** The set was released (or failed): free the shadow stand-ins kept for it. */
  private releaseWarm(set: SceneSetId): void {
    const mats = this.warmMaterials.get(set);
    if (!mats) return;
    this.warmMaterials.delete(set);
    for (const m of mats) m.dispose();
  }

  /**
   * One off-screen draw of a preloaded set into the scene target (never presented):
   * uploads its geometry, builds VAOs, renders its shadow maps once.
   */
  private warmDraw(set: SceneSetId, scene: NatureScene): void {
    const director = this.director;
    if (this.disposed || this.contextLost || !director || !this.renderer || !this.post) return;
    const span = setSpan(set);
    const t = clamp(this.t, span.start, span.end);
    const local = director.localFor(set, t, "solo", null, false);
    const ep = episodeAt(t);
    const frame = this.fillFrame(this.warmFrame, t, this.frozenTime ?? this.timeSec, 0);
    frame.globalProgress = t / VIDEO_DURATION;
    frame.episode = ep.id;
    frame.sceneProgress = clamp((t - ep.start) / (ep.end - ep.start), 0, 1);
    frame.pointerNdc.x = frame.pointerNdc.y = 0;
    const look = this.prepareView(scene, local, frame, 2);
    this.post.renderWarm(this.postView(2, scene, look, false, this.layers.maskFor(this.flags.layers)), TONE_MAPPERS[this.toneMappingName]);
  }

  private applyWireframe(): void {
    if (!this.director || this.wireframeApplied === this.flags.wireframe) return;
    this.wireframeApplied = this.flags.wireframe;
    for (const s of this.director.readyScenes()) setWireframe(s, this.flags.wireframe);
  }

  // -------------------------------------------------------------------------
  // Public API (window.__NATURE__) and debug target
  // -------------------------------------------------------------------------

  private changed(): void {
    this.capture?.invalidate();
  }

  setT(t: number): void {
    if (!Number.isFinite(t)) {
      console.warn(`[nature] setT(${String(t)}) ignored: not a finite number`);
      return;
    }
    if (this.url.capture) {
      this.captureT = clamp(t, 0, VIDEO_DURATION);
      this.t = this.targetT = this.captureT;
    } else {
      this.autoplay = false;
      this.pin(t);
    }
    this.changed();
  }

  setScene(episode: EpisodeId, progress = 0): void {
    if (typeof episode !== "string" || !isEpisodeId(episode) || !Number.isFinite(progress)) {
      console.warn(`[nature] setScene(${String(episode)}, ${String(progress)}) ignored: unknown episode or non-finite progress`);
      return;
    }
    this.setT(timeForEpisode(episode, progress));
  }

  setTime(seconds: number | null): void {
    if (seconds !== null && !Number.isFinite(seconds)) {
      console.warn(`[nature] setTime(${String(seconds)}) ignored: not a finite number`);
      return;
    }
    this.frozenTime = seconds === null ? null : Math.max(0, seconds);
    if (seconds !== null) this.timeSec = Math.max(0, seconds);
    this.changed();
  }

  setQuality(level: QualityLevel): void {
    if (!QUALITY_PRESETS[level]) return;
    // a chosen level is pinned: the adaptive watcher stops, its DPR step inside low goes
    this.stopAdapt();
    if (this.dprLimit !== Number.POSITIVE_INFINITY) {
      this.dprLimit = Number.POSITIVE_INFINITY;
      this.resizePending = true;
    }
    this.applyQuality(level, "api", true);
  }

  setSeed(seed: number): void {
    if (!Number.isFinite(seed) || seed === this.seed) return;
    this.seed = Math.trunc(seed);
    this.director?.invalidateAll();
    this.changed();
  }

  private setToggle(name: DebugToggle, value: boolean): void {
    this.flags[name] = value;
    this.changed();
  }

  private installApi(): void {
    const api: NatureApi = {
      get ready() {
        return window.__NATURE_READY__ === true;
      },
      state: () => this.snapshotState(),
      stats: () => this.snapshotStats(),
      setT: (t) => this.setT(t),
      setScene: (episode, progress = 0) => this.setScene(episode, progress),
      setTime: (s) => this.setTime(s),
      setQuality: (q) => this.setQuality(q),
      setSeed: (s) => this.setSeed(s),
      setFlags: (f) => {
        for (const key of ["post", "dof", "bloom", "vignette", "wind", "wireframe", "parallax"] as DebugToggle[]) {
          const v = f[key];
          if (typeof v === "boolean") this.flags[key] = v;
        }
        if (f.layers !== undefined) this.flags.layers = f.layers === null ? null : new Set(f.layers);
        if (f.toneMapping) this.toneMappingName = f.toneMapping;
        this.changed();
      },
    };
    window.__NATURE__ = api;
    this.cleanups.push(() => {
      if (window.__NATURE__ === api) delete window.__NATURE__;
    });
  }

  private snapshotState(): NatureStateSnapshot {
    const res = this.lastRes;
    return {
      t: this.t,
      targetT: this.targetT,
      episode: res?.episode.id ?? "hero",
      sceneProgress: res?.sceneProgress ?? 0,
      globalProgress: res?.globalProgress ?? 0,
      timeSec: this.frozenTime ?? this.timeSec,
      frozen: this.frozenTime !== null,
      seed: this.seed,
      quality: this.quality.level,
      qualityReason: this.qualityReason,
      capture: this.url.capture,
      reducedMotion: this.reducedMotion,
      ready: this.capture?.ready ?? false,
      views: (res?.views ?? []).map((v) => ({ set: v.set, episode: v.local.episode, progress: v.local.progress, role: v.local.role, t: v.local.t })),
      transition: res?.transition ? { id: res.transition.id, mode: res.transition.mode, k: res.transition.k } : null,
      sets: this.director?.states() ?? {},
      fading: [...this.fadeStart.keys()],
      flags: {
        post: this.flags.post,
        dof: this.flags.dof,
        bloom: this.flags.bloom,
        vignette: this.flags.vignette,
        wind: this.flags.wind,
        wireframe: this.flags.wireframe,
        parallax: this.flags.parallax,
        layers: this.flags.layers ? [...this.flags.layers] : null,
        toneMapping: this.toneMappingName,
      },
      status: this.status,
    };
  }

  private snapshotStats(): NatureStatsSnapshot {
    const r = this.renderer;
    const size = r ? r.getDrawingBufferSize(this.tmpSize) : { x: 0, y: 0 };
    return {
      ...this.stats.snapshot(),
      instances: Object.fromEntries(this.instances),
      dpr: this.viewport.dpr,
      drawingBuffer: { width: size.x, height: size.y },
      gpu: this.gpu,
      passes: this.post ? [...this.post.lastPasses] : [],
      programs: r?.info.programs?.length ?? 0,
      geometries: r?.info.memory.geometries ?? 0,
      textures: r?.info.memory.textures ?? 0,
      liveRenderers: NatureExperience.liveRenderers,
      rafLoops: NatureExperience.liveLoops,
      createdRenderers: NatureExperience.createdRenderers,
      parallelShaderCompile: this.parallelCompile,
      quality: {
        level: this.quality.level,
        reason: this.qualityReason,
        adapt: {
          active: this.adaptActive,
          thresholdMs: this.adaptThreshold,
          gpuTimer: this.gpuTimer !== null,
          steps: this.adaptSteps,
          dprCap: Number.isFinite(this.dprLimit) ? this.dprLimit : null,
          last: this.adaptLast ? { ...this.adaptLast } : null,
          log: this.adaptLog.map((e) => ({ ...e })),
        },
      },
      warm: {
        pending: this.warmJobs.map((j) => ({ set: j.set, phase: j.phase, units: j.units.length, unit: j.unit, programs: j.programCount, textures: j.textureCount })),
        maxFrameMs: Math.round(this.warmFrameMaxMs * 10) / 10,
        touched: this.touchedCount,
        maxTouchMs: Math.round(this.touchMaxMs * 10) / 10,
        maxTouchAt: Math.round(this.touchMaxAt),
        log: this.warmLog.map((e) => ({ ...e })),
      },
      builds: {
        counts: { ...this.buildCounts },
        log: this.buildLog.map((e) => ({ ...e })),
        slices: sliceStats(),
      },
    };
  }

  private debugTarget(): DebugTarget {
    return {
      info: (): DebugInfo => {
        const s = this.snapshotStats();
        const res = this.lastRes;
        const sets = this.director?.states() ?? {};
        return {
          episode: res ? `${res.episode.id}${res.transition ? ` (${res.transition.id} ${res.transition.mode} k=${res.transition.k.toFixed(2)})` : ""}` : "—",
          t: this.t,
          localProgress: res?.sceneProgress ?? 0,
          timeSec: this.frozenTime ?? this.timeSec,
          frozen: this.frozenTime !== null,
          seed: this.seed,
          quality: this.quality.level,
          toneMapping: this.toneMappingName,
          dpr: s.dpr,
          gpu: s.gpu,
          frameMsAvg: s.frameMsAvg,
          frameMsP95: s.frameMsP95,
          cpuMsAvg: s.cpuMsAvg,
          drawCalls: s.drawCalls,
          triangles: s.triangles,
          instances: s.instances,
          passes: s.passes,
          sets: Object.entries(sets)
            .map(([k, v]) => `${k}:${v}`)
            .join(" "),
          toggles: {
            post: this.flags.post,
            dof: this.flags.dof,
            bloom: this.flags.bloom,
            vignette: this.flags.vignette,
            wind: this.flags.wind,
            wireframe: this.flags.wireframe,
            parallax: this.flags.parallax,
          },
        };
      },
      setT: (t) => this.setT(t),
      followScroll: () => {
        this.pinnedT = null;
        this.expectedScrollY = null;
      },
      setFrozen: (frozen, time) => this.setTime(frozen ? time : null),
      setSeed: (seed) => this.setSeed(seed),
      setQuality: (q) => this.setQuality(q),
      setToneMapping: (name) => {
        this.toneMappingName = name;
        this.changed();
      },
      setToggle: (name, value) => this.setToggle(name, value),
      layerNames: () => this.layers.names(),
      isLayerVisible: (name) => this.flags.isLayerVisible(name),
      setLayerVisible: (name, visible) => {
        const all = this.layers.names();
        const current = this.flags.layers ?? new Set(all);
        if (visible) current.add(name);
        else current.delete(name);
        this.flags.layers = current.size === all.length ? null : current;
        this.changed();
      },
    };
  }

  // -------------------------------------------------------------------------
  // Teardown
  // -------------------------------------------------------------------------

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
      this.rafId = 0;
      NatureExperience.liveLoops--;
    }
    for (const fn of this.cleanups.splice(0)) fn();
    this.debugPanel?.dispose();
    this.debugPanel = null;
    // builds still queued finish on their own (macrotask slices); the director drops them
    this.slicePacing?.detach();
    this.slicePacing = null;
    this.flushWarm();
    this.stopAdapt();
    this.director?.dispose();
    for (const set of [...this.warmMaterials.keys()]) this.releaseWarm(set);
    this.post?.dispose();
    this.assets?.dispose();
    this.capture?.dispose();
    if (this.url.capture) delete document.documentElement.dataset.natureCapture;
    if (this.renderer) {
      this.renderer.dispose();
      this.renderer.forceContextLoss();
      NatureExperience.liveRenderers--;
    }
    this.canvas?.remove();
    this.renderer = null;
    this.canvas = null;
    this.post = null;
    this.assets = null;
    this.director = null;
  }
}

// ---------------------------------------------------------------------------
// Warm-up bookkeeping
// ---------------------------------------------------------------------------

/**
 * Warm-up pacing. Main-thread budget per frame (ms) and new programs per frame: with
 * KHR_parallel_shader_compile the driver links in the background, without it every
 * link runs on the GPU process's main thread (one per frame). A set the current views
 * need gets the larger budget; a preloaded one the small one, and none while a
 * transition plays or is about to start (`leadSec` story seconds ahead). First uses
 * (synchronous queries) wait until every program of the set has linked, so they never
 * queue behind a compile still running in the GPU process.
 */
const WARM = {
  budgetMs: { preload: 4, needed: 12 },
  programs: { preload: 1, needed: 8, serial: 1 },
  /** First uses per frame (without the extension a first use waits for the whole link: one per frame). */
  uses: { preload: 1, needed: 8, serial: 1 },
  /** Texture upload steps per frame (a step: a small texture, or a ≈ 1 MB strip of a large one). */
  textures: { preload: 1, needed: 8 },
  leadSec: 0.5,
  /** No preload starts this close to a transition that does not show the set (its build runs on its own once started). */
  preloadLeadSec: 1.0,
  /** Wall time without story movement after which the story counts as resting (warm-up anywhere). */
  restMs: 600,
  /** Longest wait of an aborted warm-up for its programs to finish linking before the set is disposed. */
  drainMaxMs: 5000,
} as const;

type WarmPhase = "compile" | "settle" | "draw" | "done";
/** What one warm step did (per-kind maxima go to the warm log). */
type WarmStepKind = "compile" | "use" | "texture" | "draw" | "phase" | "yield" | "wait";
type WarmTimedKind = "compile" | "use" | "texture" | "draw";

interface WarmUnit {
  /** The object to compile (alone, without its children). */
  object: Object3D;
  /** Material that replaces the object's own (shadow depth stand-in), null = its own. */
  material: Material | null;
  /** Scene whose lights / fog the programs see. */
  target: Scene;
  camera: Camera;
  /** Compile against a render target (linear, no tone mapping) instead of the canvas. */
  offscreen: boolean;
  /** Compile without the target's fog (the shadow pass draws without a scene). */
  noFog: boolean;
}

/** WebGLProgram members the typings leave out (three r186 runtime). */
type WarmProgram = ThreeProgram & { program?: unknown; isReady?: () => boolean };

class WarmJob {
  readonly units: WarmUnit[] = [];
  readonly materials = new Set<Material>();
  /** Programs this job created, null once used (or released). */
  readonly programs: (ThreeProgram | null)[] = [];
  readonly textures: Texture[] = [];
  /** Shadow depth stand-ins: they hold their programs until the set is released. */
  readonly depthMaterials: Material[] = [];
  phase: WarmPhase = "compile";
  unit = 0;
  texture = 0;
  programCount = 0;
  textureCount = 0;
  frames = 0;
  steps = 0;
  workMs = 0;
  maxFrameMs = 0;
  maxStepMs = 0;
  drawMs = 0;
  readonly maxByKind: Record<WarmTimedKind, number> = { compile: 0, use: 0, texture: 0, draw: 0 };
  readonly started = performance.now();

  constructor(
    readonly set: SceneSetId,
    readonly scene: NatureScene,
    readonly stale: () => boolean,
    readonly done: () => void,
    readonly fail: (error: unknown) => void,
  ) {}
}

export interface WarmLogEntry {
  set: SceneSetId;
  aborted: boolean;
  /** Page time (performance.now(), ms) the job was queued / finished. */
  at: number;
  end: number;
  /** Wall time from queueing to ready. */
  ms: number;
  /** Frames that did warm work for it, and the steps (one unit each). */
  frames: number;
  steps: number;
  units: number;
  programs: number;
  textures: number;
  workMs: number;
  /** Longest warm work of one frame / of one step (a compile, a texture, the draw). */
  maxFrameMs: number;
  maxStepMs: number;
  drawMs: number;
  /** Longest step per kind: compile (one object), first use of a program, texture upload, draw. */
  maxMs: Record<WarmTimedKind, number>;
  /** Aborted while its programs were still linking: how long the set waited for them before it was disposed. */
  drainMs?: number;
}

const SHADOW_KINDS = ["depth", "distance"] as const;
/** WebGLShadowMap's side flip for PCF shadows. */
const SHADOW_SIDE: Record<Side, Side> = { [FrontSide]: BackSide, [BackSide]: FrontSide, [DoubleSide]: DoubleSide } as Record<Side, Side>;
const GEOMETRY_SIGNATURES = new WeakMap<BufferGeometry, string>();

/** Attribute layout of a geometry as far as program selection cares. */
function geometrySignature(g: BufferGeometry): string {
  let s = GEOMETRY_SIGNATURES.get(g);
  if (s === undefined) {
    const attrs = Object.keys(g.attributes)
      .sort()
      .map((k) => (k === "color" ? `color${g.attributes.color.itemSize}` : k));
    s = `${attrs.join(",")}|${Object.keys(g.morphAttributes).sort().join(",")}`;
    GEOMETRY_SIGNATURES.set(g, s);
  }
  return s;
}

/** Object kind + geometry layout: the part of a program key that comes from the object. */
function objectSignature(o: Object3D): string {
  const x = o as Object3D & {
    isInstancedMesh?: boolean;
    instanceColor?: unknown;
    morphTexture?: unknown;
    isBatchedMesh?: boolean;
    isSkinnedMesh?: boolean;
    isPoints?: boolean;
    isLine?: boolean;
    isSprite?: boolean;
    geometry?: BufferGeometry;
  };
  const kind = `${x.isInstancedMesh ? `i${x.instanceColor ? "c" : ""}${x.morphTexture ? "m" : ""}` : ""}${x.isBatchedMesh ? "b" : ""}${x.isSkinnedMesh ? "s" : ""}${x.isPoints ? "p" : x.isLine ? "l" : x.isSprite ? "S" : "m"}`;
  return `${kind}|${x.geometry ? geometrySignature(x.geometry) : ""}`;
}

type DepthSource = Material & {
  map?: Texture | null;
  alphaMap?: Texture | null;
  displacementMap?: Texture | null;
  displacementScale?: number;
  displacementBias?: number;
  wireframe?: boolean;
};

/** The fields WebGLShadowMap.getDepthMaterial copies from a caster's material. */
function depthVariantKey(m: Material): string {
  const s = m as DepthSource;
  const side = m.shadowSide !== null ? m.shadowSide : SHADOW_SIDE[m.side];
  const alphaTest = m.alphaToCoverage ? 0.5 : m.alphaTest;
  return `${side}|${s.map ? s.map.channel : "-"}|${s.alphaMap ? s.alphaMap.channel : "-"}|${alphaTest > 0 ? 1 : 0}|${s.displacementMap ? 1 : 0}|${s.wireframe ? 1 : 0}|${m.clipShadows ? 1 : 0}`;
}

/** A depth (spot / directional) or distance (point) material as the shadow map builds it for `m`. */
function depthStandIn(kind: (typeof SHADOW_KINDS)[number], m: Material): Material {
  const s = m as DepthSource;
  const d = (kind === "depth" ? new MeshDepthMaterial() : new MeshDistanceMaterial()) as MeshDepthMaterial;
  d.name = `SilvaWarm${kind === "depth" ? "Depth" : "Distance"}`;
  d.side = m.shadowSide !== null ? m.shadowSide : SHADOW_SIDE[m.side];
  d.alphaMap = s.alphaMap ?? null;
  d.alphaTest = m.alphaToCoverage ? 0.5 : m.alphaTest;
  d.map = s.map ?? null;
  d.clipShadows = m.clipShadows;
  d.clippingPlanes = m.clippingPlanes;
  d.clipIntersection = m.clipIntersection;
  d.displacementMap = s.displacementMap ?? null;
  d.displacementScale = s.displacementScale ?? 1;
  d.displacementBias = s.displacementBias ?? 0;
  d.wireframe = s.wireframe === true;
  return d;
}

function setWireframe(scene: NatureScene, on: boolean): void {
  scene.scene.traverse((o) => {
    const mesh = o as Mesh;
    if (!mesh.isMesh || !mesh.material) return;
    for (const m of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
      const mat = m as Material & { wireframe?: boolean };
      if (typeof mat.wireframe === "boolean") mat.wireframe = on;
    }
  });
}

// ---------------------------------------------------------------------------
// GPU time per frame (adaptive quality)
// ---------------------------------------------------------------------------

interface TimerQueryExt {
  TIME_ELAPSED_EXT: number;
  GPU_DISJOINT_EXT: number;
}

/**
 * EXT_disjoint_timer_query_webgl2 around the frame loop's GPU work: one TIME_ELAPSED
 * query per frame, results read back frames later without blocking (availability
 * polling), dropped when the GPU reports a disjoint period. Only runs while the
 * adaptive quality watcher does.
 */
class GpuFrameTimer {
  private readonly free: WebGLQuery[] = [];
  private readonly pending: { query: WebGLQuery; frame: number }[] = [];
  private active: { query: WebGLQuery; frame: number } | null = null;

  private constructor(
    private readonly gl: WebGL2RenderingContext,
    private readonly ext: TimerQueryExt,
  ) {}

  static create(gl: WebGL2RenderingContext): GpuFrameTimer | null {
    const ext = gl.getExtension("EXT_disjoint_timer_query_webgl2") as TimerQueryExt | null;
    return ext ? new GpuFrameTimer(gl, ext) : null;
  }

  begin(frame: number): void {
    if (this.active || this.pending.length >= 6) return;
    // someone else (a profiler, a test harness) is timing: never start a second TIME_ELAPSED query
    if (this.gl.getQuery(this.ext.TIME_ELAPSED_EXT, this.gl.CURRENT_QUERY)) return;
    const query = this.free.pop() ?? this.gl.createQuery();
    if (!query) return;
    this.gl.beginQuery(this.ext.TIME_ELAPSED_EXT, query);
    this.active = { query, frame };
  }

  end(): void {
    const active = this.active;
    if (!active) return;
    this.active = null;
    if (this.gl.getQuery(this.ext.TIME_ELAPSED_EXT, this.gl.CURRENT_QUERY) !== active.query) {
      // ended by someone else: its result is not this frame's
      this.gl.deleteQuery(active.query);
      return;
    }
    this.gl.endQuery(this.ext.TIME_ELAPSED_EXT);
    this.pending.push(active);
  }

  /** Finished queries, oldest first: (frame, GPU ms). */
  poll(out: (frame: number, ms: number) => void): void {
    const gl = this.gl;
    const disjoint = gl.getParameter(this.ext.GPU_DISJOINT_EXT) as boolean;
    while (this.pending.length > 0) {
      const p = this.pending[0];
      if (!gl.getQueryParameter(p.query, gl.QUERY_RESULT_AVAILABLE)) break;
      const ns = gl.getQueryParameter(p.query, gl.QUERY_RESULT) as number;
      this.pending.shift();
      this.free.push(p.query);
      if (!disjoint) out(p.frame, ns / 1e6);
    }
  }

  /** Context lost: the queries are gone with it. */
  forget(): void {
    this.free.length = 0;
    this.pending.length = 0;
    this.active = null;
  }

  dispose(): void {
    if (this.active && this.gl.getQuery(this.ext.TIME_ELAPSED_EXT, this.gl.CURRENT_QUERY) === this.active.query) this.gl.endQuery(this.ext.TIME_ELAPSED_EXT);
    for (const q of this.free) this.gl.deleteQuery(q);
    for (const p of this.pending) this.gl.deleteQuery(p.query);
    if (this.active) this.gl.deleteQuery(this.active.query);
    this.forget();
  }
}
