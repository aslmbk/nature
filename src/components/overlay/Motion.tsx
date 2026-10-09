/**
 * Server-side building blocks of the scroll-driven overlay (no hooks, no JS at runtime):
 * the pinned per-episode Stage, in-section anchors, staggered words and the custom
 * property helpers that feed motion.module.css. All times are video seconds.
 */
import { Fragment, type CSSProperties, type ReactNode } from "react";
import type { EpisodeId } from "@/nature/types";
import { cssVars, cx } from "./cssVars";
import m from "./motion.module.css";
import { ANCHORS, STAGES, VH_PER_SECOND, episodeBounds, holdSeconds, type AnchorId, type Range } from "./timeline";

export { m as motion };

interface WinOptions {
  /** Entrance rise in px (default 24). */
  rise?: number;
  /** Entrance blur in px (default 10). */
  blur?: number;
  ease?: string;
  /**
   * Entrance window under reduced motion, where nothing slides and stages follow each
   * other: for pieces that would otherwise appear while the previous stage is still there.
   */
  reduce?: Range;
}

/**
 * Custom properties for an entrance window [a, b] (classes m.in / m.inSoft / m.inout /
 * m.inoutSoft / m.inoutRecede; the last three also fade out with the stage's exit).
 */
export function win(a: number, b: number, opts: WinOptions = {}, extra?: CSSProperties): CSSProperties {
  return cssVars(
    {
      "--a": a,
      "--b": b,
      "--ra": opts.reduce?.[0],
      "--rb": opts.reduce?.[1],
      "--rise": opts.rise === undefined ? undefined : `${opts.rise}px`,
      "--blur": opts.blur === undefined ? undefined : `${opts.blur}px`,
      "--ease": opts.ease,
    },
    extra,
  );
}

export interface ScrubValues {
  x?: string;
  y?: string;
  s?: number;
  r?: string;
  op?: number;
}

/** Custom properties for a .scrub element: from → to over [a, b]. */
export function scrub(a: number, b: number, from: ScrubValues, to: ScrubValues, ease?: string, extra?: CSSProperties): CSSProperties {
  return cssVars(
    {
      "--a": a,
      "--b": b,
      "--ease": ease,
      "--x0": from.x,
      "--y0": from.y,
      "--s0": from.s,
      "--r0": from.r,
      "--op0": from.op,
      "--x1": to.x,
      "--y1": to.y,
      "--s1": to.s,
      "--r1": to.r,
      "--op1": to.op,
    },
    extra,
  );
}

/** Custom properties for a .move element: transform only, so it is safe on ancestors of glass. */
export function move(
  a: number,
  b: number,
  from: Omit<ScrubValues, "op">,
  to: Omit<ScrubValues, "op">,
  ease?: string,
  extra?: CSSProperties,
): CSSProperties {
  return scrub(a, b, from, to, ease, extra);
}

interface IntroOptions {
  /** Duration in seconds (default 0.8). */
  dur?: number;
  rise?: number;
  blur?: number;
  /** Start scale (default 1). */
  scale?: number;
  ease?: string;
}

/** Custom properties for the one-off load intro (m.intro / m.introSoft / m.introOut / m.introOutSoft), delay in s. */
export function intro(delay: number, opts: IntroOptions = {}, extra?: CSSProperties): CSSProperties {
  return cssVars(
    {
      "--d": `${delay}s`,
      "--dur": opts.dur === undefined ? undefined : `${opts.dur}s`,
      "--rise": opts.rise === undefined ? undefined : `${opts.rise}px`,
      "--blur": opts.blur === undefined ? undefined : `${opts.blur}px`,
      "--iscale": opts.scale,
      "--iease": opts.ease,
    },
    extra,
  );
}

interface StageProps {
  id: EpisodeId;
  className?: string;
  children: ReactNode;
}

/**
 * The pinned block of one episode: sticky, one viewport tall, visible only inside its
 * `show` window, slides up + fades in its exit window (timeline.ts → STAGES). The fade is
 * not on the stage itself (glass inside): its pieces fade themselves (m.out, m.inout,
 * m.inoutSoft, …) with the stage's window, or earlier by their own height (exitFades.ts).
 */
export function Stage({ id, className, children }: StageProps) {
  const st = STAGES[id];
  const hold = holdSeconds(id);
  const style = cssVars({
    "--p0": st.show[0],
    "--p1": st.show[1],
    // the finale never leaves; the hero is there from t = 0
    "--pfill": id === "finale" ? "forwards" : id === "hero" ? "backwards" : undefined,
    "--hold": hold > 0 ? `${hold * VH_PER_SECOND}vh` : undefined,
    "--o0": st.exit?.[0],
    "--o1": st.exit?.[1],
    "--oshift": st.exit ? st.shift : undefined,
    "--oease": st.exit ? st.ease : undefined,
    "--f0": st.fade?.[0],
    "--f1": st.fade?.[1],
    "--rf0": st.reduce?.[0],
    "--rf1": st.reduce?.[1],
  });
  return (
    <div className={m.stage} data-overlay-stage={id} style={style}>
      <div className={cx(m.layer, st.exit && m.shift, className)}>{children}</div>
    </div>
  );
}

/** Invisible in-section anchor: `#about` lands on t = ANCHORS.about.t. */
export function Anchor({ id }: { id: AnchorId }) {
  const a = ANCHORS[id];
  const [start] = episodeBounds(a.section);
  return <span id={id} className={m.anchor} style={{ top: `${(a.t - start) * VH_PER_SECOND}vh` }} aria-hidden="true" />;
}

interface WordsProps {
  text: string;
  /** Time of the first word of the whole headline. */
  from: number;
  /** Index of this run's first word inside the headline (continues the stagger). */
  index0?: number;
  /** Seconds between words. */
  step?: number;
  /** Duration of one word. */
  len?: number;
  className?: string;
  opts?: WinOptions;
  /**
   * The words also fade out with the stage's exit (default), each by its own line's
   * position (exitFades.ts), so a headline dissolves line by line from the top. false for
   * a stage without an exit fade.
   */
  exit?: boolean;
}

/** Word-by-word blur-in (frames 07, 15, sheets 08/09: headlines resolve left to right). */
export function Words({ text, from, index0 = 0, step = 0.08, len = 0.55, className, opts, exit = true }: WordsProps) {
  const words = text.split(/\s+/).filter(Boolean);
  return (
    <>
      {words.map((w, i) => {
        const a = from + (index0 + i) * step;
        return (
          <Fragment key={i}>
            <span className={cx(exit ? m.inout : m.in, m.word, className)} style={win(a, a + len, opts)}>
              {w}
            </span>
            {i < words.length - 1 ? " " : null}
          </Fragment>
        );
      })}
    </>
  );
}

/** [a, b] of word `i` (for elements that should follow a headline). */
export function wordWindow(from: number, i: number, step = 0.08, len = 0.55): Range {
  return [from + i * step, from + i * step + len];
}
