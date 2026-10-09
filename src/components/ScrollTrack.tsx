/**
 * The scroll distance that drives story time: one block per episode / gap,
 * VH_PER_SECOND vh per video second (59 s ⇒ 1180vh of scrollable distance), plus a
 * trailing 100vh so the last frame is reachable.
 *
 * Anchors: `/#<section id>` targets a zero-height marker inside the section placed at
 * the end of the transition window the section starts in (plus 1px against scroll
 * rounding), so a link lands on the settled scene, not in the middle of the incoming
 * transition: `/#canyon` → t ≈ 16.4 (not 15.5, nearly black), `/#branch` → 31.9,
 * `/#stone` → 40 (not 37, still the branch). Sections that do not start inside a
 * window (`/#hero`, `/#arch`) land on their start as before. The `<section>` keeps
 * `data-section` (= the id), `data-kind`, `data-start`, `data-end` and its height.
 *
 * `slots` (optional) renders content inside a section, keyed by section id — the HTML
 * overlay puts each episode's pinned text there. Ids, data-* attributes and heights
 * do not depend on the slots.
 */
import type { CSSProperties, ReactNode } from "react";
import { SCROLL_SECTIONS, TRANSITIONS, VH_PER_SECOND, type ScrollSection } from "@/nature/SceneConfig";
import styles from "./ScrollTrack.module.css";

export interface ScrollTrackProps {
  slots?: Readonly<Partial<Record<string, ReactNode>>>;
}

/** Story time an anchor to section `s` lands on: past the transition window it starts in. */
function anchorTime(s: ScrollSection): number {
  return TRANSITIONS.find((tr) => tr.start <= s.start && tr.end > s.start)?.end ?? s.start;
}

function anchorStyle(s: ScrollSection): CSSProperties {
  const offset = anchorTime(s) - s.start;
  return {
    position: "absolute",
    left: 0,
    top: offset > 0 ? `calc(${offset * VH_PER_SECOND}vh + 1px)` : 0,
    width: 1,
    height: 0,
    pointerEvents: "none",
  };
}

export default function ScrollTrack({ slots }: ScrollTrackProps = {}) {
  return (
    <div className={styles.track}>
      {SCROLL_SECTIONS.map((s) => (
        <section
          key={s.id}
          className={styles.section}
          data-section={s.id}
          data-kind={s.kind}
          data-start={s.start}
          data-end={s.end}
          aria-label={s.kind === "episode" ? s.id : undefined}
          style={{ height: `${(s.end - s.start) * VH_PER_SECOND}vh` }}
        >
          <span id={s.id} data-anchor-t={anchorTime(s)} style={anchorStyle(s)} aria-hidden="true" />
          {slots?.[s.id]}
        </section>
      ))}
      <div className={styles.tail} aria-hidden="true" />
    </div>
  );
}
