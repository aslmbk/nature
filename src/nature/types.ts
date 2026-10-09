/**
 * Shared engine types. This file is the contract between the engine core and the
 * scene / vegetation authors: scenes implement `NatureScene` and receive a
 * `SceneContext`, a `FrameState` and a `SceneLocal` every frame.
 *
 * Runtime import is type-only (`three` is not pulled in here), so this module and
 * `SceneConfig.ts` can be imported from server components too.
 */
import type { Camera, Color, PerspectiveCamera, Scene, WebGLRenderer } from "three";
import type { AssetRegistry } from "./AssetRegistry";
import type { LayerRegistry } from "./core/layers";
import type { Rng } from "./core/rng";
import type { WindUniforms } from "./vegetation/WindField";

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

export const EPISODE_IDS = [
  "hero",
  "arch",
  "canyon",
  "oracle",
  "streams",
  "branch",
  "stone",
  "canopy",
  "canopyClose",
  "finale",
] as const;
/** Story episode, also the `scene=` URL value. */
export type EpisodeId = (typeof EPISODE_IDS)[number];

export const SCENE_SET_IDS = ["grove", "canyon", "oracle", "branch", "stone", "canopy", "finale"] as const;
/** Runtime scene class (one file in `src/nature/scenes/`), may serve several episodes. */
export type SceneSetId = (typeof SCENE_SET_IDS)[number];

export const QUALITY_LEVELS = ["high", "medium", "low"] as const;
export type QualityLevel = (typeof QUALITY_LEVELS)[number];

export const TONE_MAPPINGS = ["aces", "agx", "neutral"] as const;
export type ToneMappingName = (typeof TONE_MAPPINGS)[number];

export type TransitionMode = "mix" | "wipe" | "slide" | "over";

export type EaseName = "linear" | "smooth" | "smoother" | "inOutSine" | "inOutCubic" | "inCubic" | "outCubic";

export type DeepPartial<T> = { [K in keyof T]?: T[K] extends object ? DeepPartial<T[K]> : T[K] };

// ---------------------------------------------------------------------------
// Quality
// ---------------------------------------------------------------------------

export interface QualityPreset {
  level: QualityLevel;
  /** Max device pixel ratio used for the drawing buffer. */
  dprCap: number;
  /** MSAA samples on the HalfFloat scene render targets (the canvas itself has no MSAA). */
  msaaSamples: number;
  /** Vegetation budgets for one visible scene (see 02_IMPLEMENTATION §13). */
  grassBlades: number;
  leaves: number;
  particles: number;
  shadowMapSize: number;
  /** Near-field depth of field: full / light (fewer taps) / off. */
  dof: "full" | "light" | "off";
  dofSamples: number;
  bloomLevels: number;
  anisotropy: number;
}

// ---------------------------------------------------------------------------
// Look (per-episode grading); `LookSpec` is authored with sRGB hex strings,
// `LookParams` is the runtime form with linear `THREE.Color`s.
// ---------------------------------------------------------------------------

export interface LookShape<C> {
  background: {
    top: C;
    mid: C;
    bottom: C;
    /** Height of the mid stop: 0 = bottom edge, 1 = top edge. */
    midPosition: number;
    /** Radial glow added to the gradient (particle scenes). intensity 0 = off. x/y in 0–1 screen uv. */
    glow: { color: C; intensity: number; radius: number; x: number; y: number };
  };
  /** Linear multiplier applied to the scene image before compositing / tone mapping. */
  exposure: number;
  /** Applied by the engine to `scene.fog` when it is a `THREE.Fog`. */
  fog: { color: C; near: number; far: number };
  /**
   * Near-field depth of field. Distances in metres from the camera; blur radius in
   * reference pixels (frame height 1020) and scaled with the drawing buffer.
   * Fully blurred at `nearStart` and closer, sharp from `nearEnd` on. `maxBlurPx` 0 = off.
   */
  dof: { focusDistance: number; nearStart: number; nearEnd: number; maxBlurPx: number };
  /** strength 0 ⇒ bloom passes are skipped. */
  bloom: { strength: number; radius: number; threshold: number };
  /** Darkening bands (0–1 strength) and their size as a fraction of the frame height. */
  vignette: { top: number; bottom: number; corners: number; topSize: number; bottomSize: number };
  /** Hints for scene lights (`LightRig.apply`). */
  lights: {
    keyColor: C;
    keyIntensity: number;
    fillSky: C;
    fillGround: C;
    fillIntensity: number;
    rimColor: C;
    rimIntensity: number;
  };
}

export type LookParams = LookShape<Color>;
export type LookSpec = LookShape<string>;
export type LookPatch = DeepPartial<LookSpec>;

// ---------------------------------------------------------------------------
// Transitions
// ---------------------------------------------------------------------------

/** Runtime parameters of the composite between two scene images (TransitionPass). */
export interface TransitionParams {
  mode: TransitionMode;
  /** Eased progress 0–1. */
  k: number;
  // mix
  dip: number;
  dipColor: Color;
  // wipe: B is revealed below a ragged edge that travels from bottom (0) to top (1)
  edge: number;
  raggedness: number;
  softness: number;
  noiseScale: number;
  /** +1: edge moves up (B from below). −1: mirrored. Also used by slide. */
  direction: 1 | -1;
  edgeDarken: number;
  // slide: A moves up by `offset` screen heights, B follows from below
  offset: number;
  seamSoftness: number;
  seamDarkness: number;
  /** Blur radius of A's trailing edge in reference px. */
  seamBlurPx: number;
  seamWave: number;
  /** How far B travels with A: 1 = attached below A (page scroll), 0 = B stays in place under A. */
  follow: number;
  // over: A (transparent clear, premultiplied) over B
  reveal: number;
  opacityA: number;
  /** Flat colour that stands in for A's own background while B is not yet revealed. */
  backA: Color;
}

/** Static tuning of a transition in SceneConfig (hex colours, k-spans). */
export interface TransitionTuning {
  dip?: number;
  dipColor?: string;
  raggedness?: number;
  softness?: number;
  noiseScale?: number;
  direction?: 1 | -1;
  edgeDarken?: number;
  seamSoftness?: number;
  seamDarkness?: number;
  seamBlurPx?: number;
  seamWave?: number;
  follow?: number;
  /** over: k-range over which B is revealed behind A. */
  revealSpan?: [number, number];
  /** over: k-range over which A fades out. */
  fadeSpan?: [number, number];
  backA?: string;
}

/** Transition currently being composited, as seen by scenes. */
export interface ActiveTransition {
  id: string;
  from: EpisodeId;
  to: EpisodeId;
  fromSet: SceneSetId;
  toSet: SceneSetId;
  start: number;
  end: number;
  mode: TransitionMode;
  /** Linear 0–1 position inside the window. */
  linear: number;
  /** Eased 0–1. */
  k: number;
  /** True when reduced motion replaced the configured mode with a plain crossfade. */
  reduced: boolean;
}

// ---------------------------------------------------------------------------
// Frame & per-view state
// ---------------------------------------------------------------------------

export interface Viewport {
  /** CSS pixels. */
  width: number;
  height: number;
  /** Effective pixel ratio of the drawing buffer (capped by quality). */
  dpr: number;
  aspect: number;
}

export interface FrameState {
  /** Ambient clock (wind, particles). Frozen in capture / `time=`. */
  timeSec: number;
  /** Real delta of this frame, 0 when the clock is frozen. Never accumulate state with it. */
  deltaSec: number;
  /** Story position in video seconds (0–59). */
  t: number;
  /** t / 59. */
  globalProgress: number;
  /** Episode under `t`. */
  episode: EpisodeId;
  /** 0–1 inside `episode`. */
  sceneProgress: number;
  pointerNdc: { x: number; y: number };
  reducedMotion: boolean;
  viewport: Viewport;
  quality: QualityPreset;
  capture: boolean;
  /** Frame counter, diagnostics only. */
  frame: number;
}

export type ViewRole = "solo" | "outgoing" | "incoming";

/** What one scene set should show for the current frame. */
export interface SceneLocal {
  set: SceneSetId;
  /** Video time this view is rendered at (equals frame.t except for reduced-motion crossfades). */
  t: number;
  /** This set's episode at `t` (clamped into the set's own episodes). */
  episode: EpisodeId;
  /** 0–1 inside `episode`, clamped. */
  progress: number;
  /** Unclamped: < 0 before the episode starts (incoming), > 1 after it ended (outgoing). */
  rawProgress: number;
  /** 0–1 across the whole span of the set. */
  setProgress: number;
  role: ViewRole;
  /** The composite transition this view takes part in. */
  transition: ActiveTransition | null;
  /** Inside a same-set ("camera") transition window, e.g. hero → arch. */
  episodeBlend: { from: EpisodeId; to: EpisodeId; k: number } | null;
  /** The engine clears this view to transparent and skips the background (mode `over`, outgoing). */
  transparentBackground: boolean;
}

// ---------------------------------------------------------------------------
// Scene plumbing
// ---------------------------------------------------------------------------

/** Read-only view of the debug / URL flags. */
export interface DebugFlags {
  readonly debug: boolean;
  readonly capture: boolean;
  readonly post: boolean;
  readonly dof: boolean;
  readonly bloom: boolean;
  readonly vignette: boolean;
  readonly wind: boolean;
  readonly wireframe: boolean;
  /** Optional pointer parallax (`parallax=1`), never in capture / reduced motion. */
  readonly parallax: boolean;
  /** null = every layer visible. */
  readonly layers: ReadonlySet<string> | null;
  isLayerVisible(name: string): boolean;
}

export interface SceneContext {
  readonly set: SceneSetId;
  readonly renderer: WebGLRenderer;
  readonly assets: AssetRegistry;
  readonly seed: number;
  /** Independent deterministic stream, namespaced by the set id: rngFor(seed, `${set}:${label}`). */
  rng(label: string): Rng;
  readonly quality: QualityPreset;
  /** Shared wind uniforms (same object for every material in every scene). */
  readonly wind: WindUniforms;
  readonly debug: DebugFlags;
  /** Assign objects to named debug layers (`layers=` URL param / panel toggles). */
  readonly layers: LayerRegistry;
  /** Show instance counts in stats / debug panel, e.g. reportInstances('grass', 74000). */
  reportInstances(key: string, count: number): void;
  readonly reducedMotion: boolean;
  readonly capture: boolean;
}

/**
 * An extra scene that a set draws itself (e.g. a render-to-texture layer drawn from
 * `update`), compiled and warmed by the engine together with `NatureScene.scene`.
 */
export interface WarmTarget {
  /** The scene the set renders with `renderer.render(scene, camera)`. */
  scene: Scene;
  /** Defaults to the set's camera. */
  camera?: Camera;
  /**
   * Drawn into a render target (default: linear output, no tone mapping — every render
   * target compiles the same programs) or, with false, straight to the canvas.
   */
  offscreen?: boolean;
}

export interface NatureScene {
  readonly id: SceneSetId;
  readonly scene: Scene;
  readonly camera: PerspectiveCamera;
  /** Load / build everything. Awaited before the set is rendered. Must be deterministic for a seed. */
  prepare(ctx: SceneContext): Promise<void>;
  /**
   * Optional: scenes the set renders itself besides `scene` (layers drawn from `update`).
   * Called once after `prepare()`; the engine compiles every material in them (visible or
   * not, plus their shadow programs) and links the programs off the main thread before
   * the set is marked ready, exactly as it does for `scene`, so their first draw does not
   * stall on a synchronous compile.
   */
  warmTargets?(): readonly WarmTarget[];
  /** Grade for this frame. `base` is the SceneConfig look of `local.episode` (blended inside same-set windows). */
  getLook(frame: FrameState, local: SceneLocal, base: LookParams): LookParams;
  /** Pose and animate. Pure function of (frame, local): no accumulation between frames. */
  update(frame: FrameState, local: SceneLocal, look: LookParams): void;
  /** Optional per-frame override of the composite parameters while this scene takes part in one. */
  getTransitionOverride?(transition: ActiveTransition, frame: FrameState): Partial<TransitionParams> | null;
  /** Optional extra lines for the debug panel. */
  debugInfo?(): Record<string, string | number | boolean>;
  /** Free resources this scene created (shared assets are freed by AssetRegistry). */
  dispose(): void;
}

export type SceneFactory = () => Promise<NatureScene>;
