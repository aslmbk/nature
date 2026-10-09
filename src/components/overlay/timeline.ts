/**
 * Overlay timing, in video seconds (the engine's `t`, 0–59). Scroll position maps to t
 * exactly like the engine does it (t = scrollY / maxScroll × 59), so CSS scroll-driven
 * animations use `animation-range: <t0/59 %> <t1/59 %>` of the root scroller and the
 * fallback sets `--t` from the same formula.
 *
 * Every number here was chosen against the reference frames / contact sheets:
 * text blurs in shortly after a scene has settled and slides up + fades with the
 * scene transition that follows (see the report for the per-frame comparison).
 */
import { EPISODES, VH_PER_SECOND, VIDEO_DURATION } from "@/nature/SceneConfig";
import type { EpisodeId } from "@/nature/types";

export { VH_PER_SECOND, VIDEO_DURATION };

export type Range = readonly [number, number];

/** CSS easing curves used by the overlay. */
export const EASE = {
  out: "cubic-bezier(0.22, 0.61, 0.36, 1)",
  in: "cubic-bezier(0.5, 0, 0.75, 0)",
  accel: "cubic-bezier(0.45, 0, 0.85, 0.45)",
  inOutSine: "cubic-bezier(0.37, 0, 0.63, 1)",
  /** soft start, nearly constant speed after (content carried away by the scroll) */
  glide: "cubic-bezier(0.3, 0, 0.65, 0.7)",
  linear: "linear",
} as const;

export interface StageTiming {
  /** The stage is visible (and interactive) only inside this window. */
  show: Range;
  /** Slide window of the exit (the whole stage moves by `shift` vh). null = no exit. */
  exit: Range | null;
  /**
   * Fade window of the exit. Pieces that would slide into the top bar's row before it ends
   * fade earlier, each by its own height (exitFades.ts).
   */
  fade: Range | null;
  /** Exit translation in vh (negative = up). */
  shift: number;
  ease: string;
  /**
   * Fade window of the exit under reduced motion (nothing slides there): inside the first
   * half of the scene transition, so the next stage only fades in once this one is gone.
   * Default: `fade`.
   */
  reduce?: Range;
}

function stage(
  show: Range,
  exit: Range | null,
  fade: Range | null,
  shift = 0,
  ease: string = EASE.accel,
  reduce?: Range,
): StageTiming {
  return { show, exit, fade, shift, ease, reduce };
}

/**
 * Per-episode stage windows. A stage is pinned (sticky) from its section start until
 * `max(section end, show end)`; the extra "hold" beyond the section end keeps it pinned
 * while an exit synchronised with the next scene transition runs.
 */
export const STAGES: Record<EpisodeId, StageTiming> = {
  // camera descends past the big mass (8–10 s): motion/01 shows the text ~27 vh up at
  // 8.4 s, ~72 vh at 8.8 s, gone by 9.2 s → linear (each piece is gone before it reaches
  // the top bar: exitFades.ts)
  hero: stage([0, 9.2], [8.0, 9.2], [8.1, 9.15], -100, EASE.linear, [8.05, 8.95]),
  // light dims at 14.6 s; frame 05 (15.2 s) has no text
  arch: stage([10.1, 15.05], [14.2, 15.05], [14.25, 14.95], -16, EASE.inOutSine),
  // cards still at rest at 19.8 s (frame 06), then carried up and away; reduced motion:
  // gone before the dark crossfade's midpoint (20.9 s)
  canyon: stage([16.2, 21.3], [19.9, 21.3], [21.0, 21.3], -100, EASE.glide, [20.3, 20.85]),
  // 21.5 s: eyebrow still blurred (frame 07); ~42 vh up at 25.2 s with the paragraph
  // still opaque, gone by 25.9 s (sheet 08); reduced motion: out before the streams cards
  oracle: stage([21.3, 25.55], [24.5, 25.5], [25.15, 25.55], -50, EASE.inOutSine, [24.5, 25.2]),
  // the ragged wipe (30.65–31.4 s) carries the text away (~40 vh up at 30.8 s)
  streams: stage([25.2, 31.0], [30.2, 31.0], [30.5, 31.0], -90, EASE.accel, [30.3, 30.85]),
  // moves with the branch scene's slide (37–39.5 s, inOutSine), but is gone by 37.95 s,
  // ~26 vh up, before the stone line rising behind it reaches the middle of the frame
  // (the reference keeps the cards to 38.5 s, where they ran into the top bar and the
  // marquee); reduced motion: out in the first half of the window
  branch: stage([31.45, 39.5], [37.0, 39.5], [37.1, 37.95], -80, EASE.inOutSine, [37.0, 38.1]),
  // rises with the stone scene (37–39.5 s), slides up 43.0–43.75 s (sheet 10); the marquee
  // line fades on its own just before (StoneSection)
  stone: stage([37.0, 43.75], [42.95, 43.75], [43.5, 43.75], -100, EASE.inOutSine, [43.0, 43.6]),
  // words resolve from 44.15 s (frame 15, sheet 10); same scene, closer at 48–49.5 s
  canopy: stage([44.1, 48.4], [47.6, 48.4], [47.7, 48.35], -18, EASE.inOutSine),
  // cards recede (scale about their common centre + fade) while the leaves part,
  // 52.4–53.0 s (sheet 11, motion/04)
  canopyClose: stage([48.9, 53.1], null, [52.35, 53.05], 0),
  // the footer panel grows in from 54 s and stays to the end
  finale: stage([53.8, VIDEO_DURATION], null, null, 0),
};

/** Section bounds from the engine timeline. */
export function episodeBounds(id: EpisodeId): Range {
  const ep = EPISODES.find((e) => e.id === id);
  if (!ep) throw new Error(`unknown episode ${id}`);
  return [ep.start, ep.end];
}

/** Seconds the stage stays pinned after its section ended. */
export function holdSeconds(id: EpisodeId): number {
  const [, end] = episodeBounds(id);
  return Math.max(0, STAGES[id].show[1] - end);
}

/**
 * Entrance helper: `count` items starting at `start`, `step` s apart, each `len` s long.
 * Returns [a, b] per item.
 */
export function stagger(start: number, count: number, step: number, len: number): Range[] {
  return Array.from({ length: count }, (_, i) => [start + i * step, start + i * step + len] as const);
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------

export interface NavItem {
  label: string;
  href: string;
  /** Active while the current section starts inside [from, to) video seconds. */
  from: number;
  to: number;
}

/**
 * Extra in-section anchors: a link lands where the section's text is fully in, not in
 * the middle of the incoming transition (`/#branch` still lands on t = 31 as before).
 */
export const ANCHORS = {
  about: { section: "arch" as EpisodeId, t: 12.0 },
  grove: { section: "oracle" as EpisodeId, t: 23.2 },
  method: { section: "branch" as EpisodeId, t: 33.4 },
  plans: { section: "canopyClose" as EpisodeId, t: 50.4 },
  contact: { section: "finale" as EpisodeId, t: 57.5 },
} as const;
export type AnchorId = keyof typeof ANCHORS;

export const NAV_ITEMS: readonly NavItem[] = [
  { label: "Home", href: "#hero", from: 0, to: 10 },
  { label: "About", href: "#about", from: 10, to: 21 },
  { label: "Method", href: "#method", from: 21, to: 43.3 },
  { label: "Plans", href: "#plans", from: 43.3, to: Number.POSITIVE_INFINITY },
];

export function navIndexForTime(t: number): number {
  const i = NAV_ITEMS.findIndex((n) => t >= n.from - 1e-6 && t < n.to);
  return i < 0 ? 0 : i;
}
