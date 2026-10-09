/**
 * Timeline, per-episode looks and quality presets. Pure data (no `three` import),
 * safe to import from server components and tools.
 *
 * Times are video seconds of the reference (0–59). Global scroll progress = t / 59.
 * Episode ranges and transition windows follow the tables in CLAUDE.md.
 */
import type {
  EaseName,
  EpisodeId,
  LookPatch,
  LookSpec,
  QualityLevel,
  QualityPreset,
  SceneSetId,
  TransitionMode,
  TransitionTuning,
} from "./types";

export const VIDEO_DURATION = 59;
/** Scroll length per video second (vh). 59 s ⇒ 1180vh of scrollable distance. */
export const VH_PER_SECOND = 20;
export const REFERENCE_WIDTH = 1440;
export const REFERENCE_HEIGHT = 1020;
export const REFERENCE_ASPECT = REFERENCE_WIDTH / REFERENCE_HEIGHT;
export const DEFAULT_SEED = 134;
/** Ambient clock used by `capture=1` when `time=` is not given. */
export const CAPTURE_TIME_SEC = 2;
/** Scroll smoothing response (time constant, seconds). */
export const SCROLL_RESPONSE_SEC = 0.16;

// ---------------------------------------------------------------------------
// Episodes
// ---------------------------------------------------------------------------

export interface EpisodeConfig {
  id: EpisodeId;
  set: SceneSetId;
  start: number;
  end: number;
  label: string;
}

export const EPISODES: readonly EpisodeConfig[] = [
  { id: "hero", set: "grove", start: 0, end: 8, label: "Overgrown frame" },
  { id: "arch", set: "grove", start: 10, end: 15.5, label: "Mossy trunk arch" },
  { id: "canyon", set: "canyon", start: 15.5, end: 21, label: "Dark rock canyon" },
  { id: "oracle", set: "oracle", start: 21, end: 25, label: "Particle orb" },
  { id: "streams", set: "oracle", start: 25, end: 31, label: "Loop and particle streams" },
  { id: "branch", set: "branch", start: 31, end: 37, label: "J-branch and fuzzy balls" },
  { id: "stone", set: "stone", start: 37, end: 43.3, label: "Stone with recessed seed" },
  { id: "canopy", set: "canopy", start: 43.3, end: 48.5, label: "Canopy on black" },
  { id: "canopyClose", set: "canopy", start: 48.5, end: 52.8, label: "Canopy close-up" },
  { id: "finale", set: "finale", start: 52.8, end: 59, label: "Seed stone on moss hills" },
];

/** Scene sets in story order. */
export const SCENE_SET_ORDER: readonly SceneSetId[] = ["grove", "canyon", "oracle", "branch", "stone", "canopy", "finale"];

// ---------------------------------------------------------------------------
// Transitions
// ---------------------------------------------------------------------------

export interface TransitionConfig {
  id: string;
  from: EpisodeId;
  to: EpisodeId;
  start: number;
  end: number;
  /** camera: same scene set, pose change only. composite: two scene sets rendered and composited. */
  kind: "camera" | "composite";
  mode: TransitionMode;
  ease: EaseName;
  tuning: TransitionTuning;
  note: string;
}

export const TRANSITIONS: readonly TransitionConfig[] = [
  {
    id: "hero-arch",
    from: "hero",
    to: "arch",
    start: 8.0,
    end: 10.0,
    kind: "camera",
    mode: "mix",
    ease: "inOutSine",
    tuning: {},
    note: "One continuous world: the camera descends past the big overgrown mass (motion/01). No fade.",
  },
  {
    id: "arch-canyon",
    from: "arch",
    to: "canyon",
    // a black dip (GroveScene deepens it over the window); the canyon is unlit until
    // 15.8 s and its lights come up 15.8 → 16.8 s (CanyonScene, canyon/track.ts), so
    // the frame is near black 15.6–16.0 s (mean luminance ≈ 1–2 / 255) and the canyon
    // opens out of the dark: no flash, no seam
    start: 14.6,
    end: 16.4,
    kind: "composite",
    mode: "mix",
    ease: "smooth",
    tuning: { dip: 0.9, dipColor: "#000000" },
    note: "Light dims, a dark lower form covers the frame, the canyon opens out of darkness.",
  },
  {
    id: "canyon-oracle",
    from: "canyon",
    to: "oracle",
    // the canyon's lights go down 20.6 → 21.2 s (CanyonScene): with the black dip the
    // frame is near black at 21.0–21.2 s and the oracle's particles come up out of it
    // (21.4 s ≈ frame 07's mean luminance)
    start: 20.3,
    end: 21.5,
    kind: "composite",
    mode: "mix",
    ease: "smooth",
    tuning: { dip: 0.65, dipColor: "#000000" },
    note: "Dark crossfade.",
  },
  {
    id: "oracle-streams",
    from: "oracle",
    to: "streams",
    start: 24.5,
    end: 26.0,
    kind: "camera",
    mode: "mix",
    ease: "inOutSine",
    tuning: {},
    note: "Same scene, pose change.",
  },
  {
    id: "streams-branch",
    from: "streams",
    to: "branch",
    // motion/02 + frame 10: the edge follows a smoothed scroll (≈0.1 / 0.5 / 0.69 / 0.81 of the
    // height at 30.8 / 31.0 / 31.2 / 31.4 s), a ragged dark band is still at the top at 31.5 s;
    // BranchScene.getTransitionOverride drives the edge on that curve and clears it by `end`
    start: 30.65,
    end: 31.9,
    kind: "composite",
    mode: "wipe",
    ease: "linear",
    tuning: { raggedness: 0.12, softness: 0.004, noiseScale: 4.2, direction: 1, edgeDarken: 0.5 },
    note: "The dark scene leaves upward with a ragged organic edge, the light branch is revealed from below (motion/02).",
  },
  {
    id: "branch-stone",
    from: "branch",
    to: "stone",
    // frame 13 (38.5 s): the branch picture has moved up ≈ 0.5 frame heights (ball, fragment
    // and the seam at ≈ 0.48 from the top) → the window is centred on 38.5 s; its lower
    // edge fades out over ≈ 0.25–0.3 of the height (0.2 → 0.47 from the top) and is blurred;
    // the stone picture is then only ≈ 0.30 frame heights low (0.6 × 0.5): `follow` 0.6
    start: 37.0,
    end: 40.0,
    kind: "composite",
    mode: "slide",
    ease: "inOutSine",
    tuning: { direction: 1, seamSoftness: 0.3, seamDarkness: 1, seamBlurPx: 28, seamWave: 0.03, follow: 0.6 },
    note: "Branch slides up with a soft blurred lower edge, the dark stone scene rises from below (frame 13).",
  },
  {
    id: "stone-canopy",
    from: "stone",
    to: "canopy",
    // the stone picture leaves fast and early (stone.glb `exit_track`, measured upward
    // travel 0.284 / 0.641 / 0.839 fh at 43.2 / 43.4 / 43.6 s): outCubic over 43.08–44.22
    // gives 0.284 / 0.628 / 0.839 (StoneScene debug panel: "exit …")
    start: 43.08,
    end: 44.22,
    kind: "composite",
    mode: "slide",
    ease: "outCubic",
    tuning: { direction: 1, seamSoftness: 0.12, seamDarkness: 0.95, seamBlurPx: 12, seamWave: 0.02, follow: 0.4 },
    note: "Stone slides up, the canopy grows from bottom-centre on black (motion/03): B only partly follows A.",
  },
  {
    id: "canopy-canopyClose",
    from: "canopy",
    to: "canopyClose",
    start: 48.0,
    end: 49.5,
    kind: "camera",
    mode: "mix",
    ease: "inOutSine",
    tuning: {},
    note: "Same scene, closer.",
  },
  {
    id: "canopyClose-finale",
    from: "canopyClose",
    to: "finale",
    start: 52.4,
    end: 53.9,
    kind: "composite",
    mode: "over",
    ease: "linear",
    // Finale side, measured on motion/04 (52.4 … 53.9 s), clip 11 and frame 18 (display-linear
    // luminance of the island / the background above it, relative to the full picture):
    // 53.0 ≈ 0.04, 53.2 ≈ 0.04 island / 0.09 sky (frame 18), 53.3 ≈ 0.15, 53.6 ≈ 0.55,
    // 53.9 ≈ 0.95, 54.05 = 1; the hills are already faintly there in the hole at 53.0 (no
    // black centre between the parting leaves). revealSpan [0.3, 1] = smoothstep(52.85, 53.9);
    // FinaleScene's entry light divides its own target curves by this reveal (island and sky
    // separately, through the ACES toe), so the picture keeps rising after the window ends.
    // fadeSpan / backA are the canopy side (not tuned here).
    tuning: { revealSpan: [0.3, 1.0], fadeSpan: [0.45, 1.0], backA: "#000000" },
    note: "Leaves part towards the frame edges, the finale appears in the middle (motion/04).",
  },
];

/** Defaults for every tuning field (merged under TransitionConfig.tuning). */
export const TRANSITION_TUNING_DEFAULTS: Required<TransitionTuning> = {
  dip: 0,
  dipColor: "#000000",
  raggedness: 0.08,
  softness: 0.005,
  noiseScale: 5,
  direction: 1,
  edgeDarken: 0.4,
  seamSoftness: 0.15,
  seamDarkness: 0.85,
  seamBlurPx: 14,
  seamWave: 0.02,
  follow: 1,
  revealSpan: [0, 0.7],
  fadeSpan: [0.4, 1],
  backA: "#000000",
};

// ---------------------------------------------------------------------------
// Quality presets (presets.example.json + render settings)
// ---------------------------------------------------------------------------

export const QUALITY_PRESETS: Record<QualityLevel, QualityPreset> = {
  high: {
    level: "high",
    dprCap: 1.5,
    msaaSamples: 4,
    grassBlades: 75000,
    leaves: 16000,
    particles: 16000,
    shadowMapSize: 2048,
    dof: "full",
    dofSamples: 48,
    bloomLevels: 5,
    anisotropy: 8,
  },
  medium: {
    level: "medium",
    // same cap as low (and high): a step down never raises the pixel count; on a phone all
    // three draw 1.5×, on a desktop / laptop the pixel budget (2.2 MP) is the tighter limit.
    dprCap: 1.5,
    msaaSamples: 2,
    grassBlades: 32000,
    leaves: 7500,
    particles: 7000,
    shadowMapSize: 1024,
    dof: "light",
    dofSamples: 24,
    bloomLevels: 5,
    anisotropy: 4,
  },
  low: {
    level: "low",
    // phones (≈ 0.3 MP of CSS pixels at DPR 2–3) get 1.5× instead of a soft 1×; the drawing
    // buffer stays inside low's 1.4 MP budget (NatureExperience MAX_DRAWING_BUFFER_PX), so a
    // 1440×1020 desktop keeps its 1405×995. No MSAA, no DOF (unmeasured on phone GPUs).
    dprCap: 1.5,
    msaaSamples: 0,
    grassBlades: 15000,
    leaves: 3500,
    particles: 3000,
    shadowMapSize: 512,
    dof: "off",
    dofSamples: 0,
    bloomLevels: 4,
    anisotropy: 2,
  },
};

/** presets.example.json → wind starting points. */
export const WIND_DEFAULTS = {
  tipDisplacementFraction: 0.07,
  mainCycleSeconds: 4.5,
  flutterRelativeAmplitude: 0.15,
  rootWeightPower: 2,
  /** Wind strength multiplier under prefers-reduced-motion. */
  reducedMotionScale: 0.15,
} as const;

/** Measured screen colours of the reference (02_IMPLEMENTATION §7). Not albedos. */
export const REFERENCE_COLORS = {
  heroBackground: "#5B654F",
  heroBackgroundLower: "#757F69",
  mossMid: "#5D6B22",
  mossLight: "#B1C076",
  barkMid: "#796449",
  darkBackground: "#060B05",
  coolBackground: "#191F22",
  leafMid: "#6F8034",
  stoneMid: "#8A816F",
  particleLime: "#BCEB6A",
  particleGreen: "#6CCD65",
  particleTurquoise: "#70C9BD",
} as const;

// ---------------------------------------------------------------------------
// Looks
// ---------------------------------------------------------------------------

// Background stops are pre-compensated for ACES so that, with the top band, the
// empty-background column of a frame lands on the measured screen colours
// (hero: 0.10 → #11130E, 0.20 → #2A3023, 0.35 → #545E48, 0.50 → #5F6953, 0.65 → #6A745E).
const BASE_LOOK: LookSpec = {
  background: {
    top: "#4F5647",
    mid: "#5E6655",
    bottom: "#7D8573",
    midPosition: 0.5,
    glow: { color: "#000000", intensity: 0, radius: 0.5, x: 0.5, y: 1.05 },
  },
  exposure: 1,
  fog: { color: "#5E6655", near: 8, far: 60 },
  dof: { focusDistance: 2, nearStart: 0.8, nearEnd: 1.4, maxBlurPx: 0 },
  bloom: { strength: 0, radius: 0.6, threshold: 1 },
  vignette: { top: 0.94, bottom: 0.9, corners: 0.22, topSize: 0.44, bottomSize: 0.3 },
  lights: {
    keyColor: "#FFF3DF",
    keyIntensity: 2.4,
    fillSky: "#C3CDB2",
    fillGround: "#3B4130",
    fillIntensity: 1.0,
    rimColor: "#EEF4DA",
    rimIntensity: 0.9,
  },
};

function patch(base: LookSpec, p: LookPatch): LookSpec {
  return {
    background: { ...base.background, ...p.background, glow: { ...base.background.glow, ...p.background?.glow } },
    exposure: p.exposure ?? base.exposure,
    fog: { ...base.fog, ...p.fog },
    dof: { ...base.dof, ...p.dof },
    bloom: { ...base.bloom, ...p.bloom },
    vignette: { ...base.vignette, ...p.vignette },
    lights: { ...base.lights, ...p.lights },
  };
}

// oracle / streams (frames 07–09). Stops pre-compensated for ACES like hero: the
// gradient alone lands on the measured edge colour (#030502 → #030602), the wide top
// glow lifts the top band to ~#0B1407; OracleScene adds the narrow top beam and the
// faint centre field (#0A1208 in the middle) in its own atmosphere layer.
const DARK_PARTICLES: LookPatch = {
  background: {
    top: "#10170E",
    mid: "#10170E",
    bottom: "#0F160D",
    midPosition: 0.5,
    glow: { color: "#11180A", intensity: 1, radius: 0.42, x: 0.47, y: 1.0 },
  },
  fog: { color: "#050A05", near: 6, far: 30 },
  dof: { focusDistance: 5, nearStart: 1.5, nearEnd: 2.5, maxBlurPx: 0 },
  bloom: { strength: 0.38, radius: 0.55, threshold: 1.1 },
  vignette: { top: 0, bottom: 0.35, corners: 0.35, topSize: 0.3, bottomSize: 0.22 },
  lights: { keyIntensity: 0.8, fillIntensity: 0.25, rimIntensity: 0.5, fillSky: "#46624A", fillGround: "#070A06" },
};

const BLACK_CANOPY: LookPatch = {
  background: {
    top: "#000000",
    mid: "#010201",
    bottom: "#000000",
    midPosition: 0.5,
    glow: { intensity: 0 },
  },
  fog: { color: "#000000", near: 10, far: 40 },
  bloom: { strength: 0.12, radius: 0.5, threshold: 0.9 },
  vignette: { top: 0.7, bottom: 0.7, corners: 0.3, topSize: 0.3, bottomSize: 0.25 },
  lights: { keyIntensity: 2.6, fillIntensity: 0.45, fillSky: "#7E9A6A", fillGround: "#0B0F08", rimIntensity: 0.8 },
};

export const EPISODE_LOOKS: Record<EpisodeId, LookSpec> = {
  hero: patch(BASE_LOOK, {
    dof: { focusDistance: 1.75, nearStart: 0.75, nearEnd: 1.3, maxBlurPx: 26 },
    fog: { color: "#5E6655", near: 2.5, far: 14 },
    // key from the upper right, a little behind (GroveScene); a bright soft sky fill from
    // above and the lens side lights the faces turned to the camera (the overhang's face
    // reads mid olive in frame 01); the insides of the mats stay dark through their AO
    lights: { keyIntensity: 1.95, fillSky: "#D2DBC2", fillGround: "#ADB088", fillIntensity: 3.1, rimIntensity: 2.0 },
  }),
  arch: patch(BASE_LOOK, {
    dof: { focusDistance: 4.8, nearStart: 1.9, nearEnd: 3.3, maxBlurPx: 22 },
    fog: { color: "#5E6655", near: 5, far: 22 },
    // key from high up on the left (GroveScene ARCH_KEY_CAM); a fuller sky fill keeps the
    // cushions turned away from it olive instead of black (frame 04)
    lights: { keyColor: "#FFEFD6", keyIntensity: 3.4, fillSky: "#C8D2B6", fillGround: "#4A5240", fillIntensity: 1.4 },
  }),
  // frame 06: the empty centre measures pure black; a breath of dark green in the air and
  // the fog (no light fog); the key is CanyonScene's spot pool (warm-neutral), the fill
  // almost nothing, the rim only shows in the plants' silhouettes; the near form at the
  // bottom left is blurred (CanyonScene sets the DOF along the camera track), the lower
  // band sinks to black
  canyon: patch(BASE_LOOK, {
    background: { top: "#000000", mid: "#010201", bottom: "#020302", midPosition: 0.5, glow: { intensity: 0 } },
    fog: { color: "#010201", near: 5.5, far: 26 },
    dof: { focusDistance: 4.45, nearStart: 2.45, nearEnd: 3.4, maxBlurPx: 16 },
    bloom: { strength: 0 },
    vignette: { top: 0.55, bottom: 0.95, corners: 0.35, topSize: 0.3, bottomSize: 0.45 },
    lights: {
      keyColor: "#F6EAD4",
      keyIntensity: 3.4,
      fillSky: "#2E3A2A",
      fillGround: "#030403",
      fillIntensity: 0.12,
      rimColor: "#DCE8BE",
      rimIntensity: 0.5,
    },
  }),
  // particles do their own defocus (OracleScene sprites), so the depth DOF stays off
  oracle: patch(BASE_LOOK, {
    ...DARK_PARTICLES,
  }),
  streams: patch(BASE_LOOK, {
    ...DARK_PARTICLES,
    bloom: { strength: 0.4, radius: 0.6, threshold: 1.1 },
  }),
  branch: patch(BASE_LOOK, {
    // fitted like hero: 0.20 → #181A17, 0.35 → #3B423D, 0.50 → #5D645F, 0.65 → #6B726D
    background: { top: "#2E312F", mid: "#5D625F", bottom: "#838984", midPosition: 0.5 },
    fog: { color: "#5D625F", near: 4, far: 30 },
    // the ball on its visible arc (1.4–1.6 m from the lens) is soft as a whole, ≈ 3–5 px
    // (frames 10, 12, detail 07: the reference ball is never crisp); closer swings blur
    // more; the branch and the fragment (≥ 1.9 m) stay sharp
    dof: { focusDistance: 2.38, nearStart: 0.6, nearEnd: 1.95, maxBlurPx: 12 },
    // frames 10–12: the lower fifth sinks to ≈ #242620 (the J's underside and the
    // fragment's foot included)
    vignette: { top: 0.86, bottom: 0.95, corners: 0.22, topSize: 0.58, bottomSize: 0.4 },
    lights: { keyColor: "#FFF6E6", keyIntensity: 2.8, fillSky: "#C6CEC6", fillGround: "#3A3F38" },
  }),
  // frame 14. Stops pre-compensated for ACES (the toe crushes the dark end): the empty
  // background is a warm near-black, lighter behind the slab's top (#13110C–#16150E),
  // #0C0A06 left of centre, black at the bottom left; the warm glow sits behind the slab.
  // The yellow-green haze along the top is StoneScene's own layer (stone/haze.ts), so the
  // top band of the vignette stays off. The pool / top light / moss ramp: StoneScene.
  stone: patch(BASE_LOOK, {
    background: {
      top: "#181611",
      mid: "#171510",
      bottom: "#050403",
      midPosition: 0.6,
      glow: { color: "#1D1A14", intensity: 1, radius: 0.5, x: 0.85, y: 0.68 },
    },
    fog: { color: "#050403", near: 6, far: 26 },
    // nothing near the lens: the slab is 3.1–4.3 m away, all in focus
    dof: { focusDistance: 3.49, nearStart: 1.6, nearEnd: 2.6, maxBlurPx: 0 },
    // no bottom band either: the slab rises from the bottom edge during branch → stone
    // (frame 13: its lit top is bright right at the edge); the pool's cookie darkens the
    // slab's foot instead
    vignette: { top: 0, bottom: 0, corners: 0.3, topSize: 0.3, bottomSize: 0.3 },
    lights: {
      keyColor: "#FFF2E0",
      keyIntensity: 4.5,
      fillSky: "#5A5547",
      fillGround: "#080706",
      fillIntensity: 0.3,
      rimColor: "#D8E6B8",
      // the top light on the moss strip (StoneScene: from above, a little behind)
      rimIntensity: 1.2,
    },
  }),
  // frames 15–17: one crown on pure black, lit from above-front (CanopyScene: key from
  // key_canopy, baked key transmittance per leaf, no rim). DOF values are for the main
  // poses; CanopyScene shifts them along its focus track. The far shells sink into the
  // black fog with distance.
  canopy: patch(BASE_LOOK, {
    ...BLACK_CANOPY,
    fog: { color: "#000000", near: 7, far: 16 },
    dof: { focusDistance: 4.47, nearStart: 2.8, nearEnd: 3.95, maxBlurPx: 8 },
    bloom: { strength: 0.1, radius: 0.5, threshold: 1.0 },
    vignette: { top: 0.55, bottom: 0.6, corners: 0.3, topSize: 0.3, bottomSize: 0.25 },
    lights: {
      keyColor: "#FFF6EA",
      keyIntensity: 5.4,
      fillSky: "#A0B890",
      fillGround: "#0A0D08",
      fillIntensity: 0.6,
      rimColor: "#EEF4DA",
      rimIntensity: 0,
    },
  }),
  canopyClose: patch(BASE_LOOK, {
    ...BLACK_CANOPY,
    fog: { color: "#000000", near: 5, far: 13 },
    dof: { focusDistance: 2.68, nearStart: 1.5, nearEnd: 2.45, maxBlurPx: 12 },
    bloom: { strength: 0.1, radius: 0.5, threshold: 1.0 },
    vignette: { top: 0.55, bottom: 0.6, corners: 0.35, topSize: 0.3, bottomSize: 0.25 },
    lights: {
      keyColor: "#FFF6EA",
      keyIntensity: 4.9,
      fillSky: "#A0B890",
      fillGround: "#0A0D08",
      fillIntensity: 0.5,
      rimColor: "#EEF4DA",
      rimIntensity: 0,
    },
  }),
  // finale (frames 18–20). Background pre-compensated for ACES + the top band: frame 19
  // measures #1A1F25 behind the stone, #141A1D / #0C1012 / #020404 at v 0.25 / 0.18 / 0.10
  // above it, which is one flat cold value (≈ 0.025, 0.031, 0.037 linear before ACES)
  // under the top vignette; the glow (moved with the stone by FinaleScene) adds the light
  // seen through the cut-through seed. DOF focus / near field are set per frame by the scene.
  finale: patch(BASE_LOOK, {
    background: {
      top: "#262A2E",
      mid: "#292E32",
      bottom: "#262B2F",
      midPosition: 0.55,
      glow: { color: "#181C21", intensity: 0.85, radius: 0.32, x: 0.5, y: 0.58 },
    },
    fog: { color: "#2A2F35", near: 4, far: 26 },
    dof: { focusDistance: 3.85, nearStart: 1.73, nearEnd: 2.77, maxBlurPx: 16 },
    bloom: { strength: 0.08, radius: 0.6, threshold: 0.9 },
    vignette: { top: 0.9, bottom: 0.95, corners: 0.3, topSize: 0.4, bottomSize: 0.3 },
    lights: {
      // a strong key and a low fill: the lit faces of the moss reach frame 19's crest
      // highlights (luma p90 ≈ 190) while the mounds' far sides stay dark; the falloff below
      // the crest and towards the edges is FinaleScene's light pool
      keyColor: "#FFF2DE",
      keyIntensity: 9.5,
      fillSky: "#9AAAB2",
      // the moss bounces green light into the pile and onto the stone's foot
      fillGround: "#46521C",
      fillIntensity: 0.55,
      rimColor: "#E8E6DC",
      rimIntensity: 2.5,
    },
  }),
};

// ---------------------------------------------------------------------------
// Lookups
// ---------------------------------------------------------------------------

const EPISODE_BY_ID = new Map<EpisodeId, EpisodeConfig>(EPISODES.map((e) => [e.id, e]));

export function episodeConfig(id: EpisodeId): EpisodeConfig {
  const ep = EPISODE_BY_ID.get(id);
  if (!ep) throw new Error(`Unknown episode ${id}`);
  return ep;
}

// computed once (EPISODES is static): the director asks for these every frame
const EPISODES_OF_SET = new Map<SceneSetId, readonly EpisodeConfig[]>();
const SET_SPANS = new Map<SceneSetId, { readonly start: number; readonly end: number }>();
for (const e of EPISODES) EPISODES_OF_SET.set(e.set, [...(EPISODES_OF_SET.get(e.set) ?? []), e]);
for (const [set, eps] of EPISODES_OF_SET) {
  Object.freeze(eps);
  SET_SPANS.set(set, Object.freeze({ start: eps[0].start, end: eps[eps.length - 1].end }));
}

/** Episodes of a scene set in story order (shared, read-only). */
export function episodesOfSet(set: SceneSetId): readonly EpisodeConfig[] {
  const eps = EPISODES_OF_SET.get(set);
  if (!eps) throw new Error(`Scene set ${set} has no episode`);
  return eps;
}

/** First start / last end of a scene set (shared, read-only). */
export function setSpan(set: SceneSetId): { readonly start: number; readonly end: number } {
  const span = SET_SPANS.get(set);
  if (!span) throw new Error(`Scene set ${set} has no episode`);
  return span;
}

/** Index of a scene set in SCENE_SET_ORDER (−1 if unknown). */
export function setIndex(set: SceneSetId): number {
  return SCENE_SET_ORDER.indexOf(set);
}

export function isEpisodeId(value: string): value is EpisodeId {
  return EPISODE_BY_ID.has(value as EpisodeId);
}

/** t for `scene=<episode>&progress=<0..1>`. */
export function timeForEpisode(id: EpisodeId, progress: number): number {
  const ep = episodeConfig(id);
  const p = Math.min(1, Math.max(0, progress));
  return ep.start + (ep.end - ep.start) * p;
}

/**
 * Episode under t. Inside a gap between episodes (8–10 s) the nearer side wins
 * (split at the midpoint of the gap). Before the first episode, and for NaN, the first
 * episode.
 */
export function episodeAt(t: number, list: readonly EpisodeConfig[] = EPISODES): EpisodeConfig {
  const first = list[0];
  if (!(t >= first.start)) return first;
  for (let i = 0; i < list.length; i++) {
    const ep = list[i];
    if (t < ep.end || i === list.length - 1) {
      if (t >= ep.start) return ep;
      // in the gap before ep
      const prev = list[i - 1];
      return t < (prev.end + ep.start) / 2 ? prev : ep;
    }
  }
  return list[list.length - 1];
}

// ---------------------------------------------------------------------------
// Scroll sections (server-rendered heights, VH_PER_SECOND per video second)
// ---------------------------------------------------------------------------

export interface ScrollSection {
  /** Episode id, or the transition id for gaps (e.g. "hero-arch"). */
  id: string;
  kind: "episode" | "gap";
  start: number;
  end: number;
}

export const SCROLL_SECTIONS: readonly ScrollSection[] = (() => {
  const out: ScrollSection[] = [];
  let cursor = 0;
  for (const ep of EPISODES) {
    if (ep.start > cursor + 1e-6) {
      const gap = TRANSITIONS.find((tr) => tr.start <= cursor + 1e-6 && tr.end >= ep.start - 1e-6);
      out.push({ id: gap ? gap.id : `gap-${cursor}`, kind: "gap", start: cursor, end: ep.start });
    }
    out.push({ id: ep.id, kind: "episode", start: ep.start, end: ep.end });
    cursor = ep.end;
  }
  return out;
})();
