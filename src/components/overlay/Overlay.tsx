"use client";

/**
 * The HTML overlay on top of the WebGL canvas: the fixed top bar plus, as children, the
 * ScrollTrack whose sections carry each episode's pinned text (server components).
 *
 * Motion is CSS-driven (motion.module.css). This component only:
 *  - puts the engine's `?motion=reduce|full` override (fade-only text animation) on <html>
 *    before the first paint: a tiny script that Next inserts into the server HTML's
 *    <head> (useServerInsertedHTML, the App Router's hook for server-only head content),
 *    so the scroll-driven animations are created with the right --motion — set later,
 *    the compositor would keep the full-motion transforms. The script is never a React
 *    element on the client. Should the attribute still be missing after hydration, it is
 *    set then and the scroll-driven animations restart once (the same restart follows a
 *    change of the OS reduced-motion setting while the page is open, unless the URL flag
 *    pins it);
 *  - measures the per-piece exit windows that keep the text out of the top bar's row
 *    (exitFades.ts): on mount, once the web fonts have loaded and after resizes;
 *  - switches on the fallback path when the browser has no scroll-driven animations
 *    (or `?overlay=fallback` forces it): a passive scroll listener that requests at
 *    most one animation frame, which reads scrollY once and writes `--t` on the stages
 *    whose window is near (so only their subtrees restyle). Sizes are re-measured
 *    inside that same frame after a resize. No loop, no per-event layout reads,
 *    nothing runs while the page is still.
 *
 * `data-nature-overlay` on the root: `?capture=1` hides it (globals.css).
 */
import { useServerInsertedHTML } from "next/navigation";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { VIDEO_DURATION } from "@/nature/SceneConfig";
import { cssVars } from "./cssVars";
import { layoutExitFades } from "./exitFades";
import styles from "./Overlay.module.css";
import TopBar from "./TopBar";

function supportsScrollTimeline(): boolean {
  try {
    return typeof CSS !== "undefined" && typeof CSS.supports === "function" && CSS.supports("animation-timeline: scroll()");
  } catch {
    return false;
  }
}

/** Sets html[data-motion] from `?motion=reduce|full` (runs in the document head, pre-paint). */
const MOTION_FLAG_SCRIPT = `(function(){try{var m=new URLSearchParams(location.search).get("motion");if(m==="reduce"||m==="full")document.documentElement.setAttribute("data-motion",m)}catch(e){}})();`;

/** Recreates the scroll-driven animations (one style flush), e.g. after --motion changed. */
function restartMotion(root: HTMLElement) {
  root.dataset.overlayRemotion = "";
  void root.offsetWidth;
  delete root.dataset.overlayRemotion;
}

export default function Overlay({ children }: { children?: ReactNode }) {
  const rootRef = useRef<HTMLDivElement>(null);

  // Server render only (a no-op in the browser): the flag script, once per response.
  const [takeFlagScript] = useState(() => {
    let taken = false;
    return () => !taken && (taken = true);
  });
  useServerInsertedHTML(() => (takeFlagScript() ? <script dangerouslySetInnerHTML={{ __html: MOTION_FLAG_SCRIPT }} /> : null));

  // ---- reduced motion flag, exit windows ----
  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    const html = document.documentElement;
    const flag = new URLSearchParams(window.location.search).get("motion");
    const pinned = flag === "reduce" || flag === "full" ? flag : null;
    if (pinned && html.dataset.motion !== pinned) {
      html.dataset.motion = pinned;
      restartMotion(root);
    }
    layoutExitFades(root);

    let alive = true;
    let timer = 0;
    const relayout = () => {
      window.clearTimeout(timer);
      timer = window.setTimeout(() => layoutExitFades(root), 150);
    };
    void document.fonts?.ready.then(() => {
      if (alive) layoutExitFades(root);
    });
    window.addEventListener("resize", relayout, { passive: true });
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onReduceChange = () => {
      if (pinned) return;
      restartMotion(root);
      layoutExitFades(root);
    };
    reduce.addEventListener("change", onReduceChange);

    return () => {
      alive = false;
      window.clearTimeout(timer);
      window.removeEventListener("resize", relayout);
      reduce.removeEventListener("change", onReduceChange);
    };
  }, []);

  // ---- fallback path ----
  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    const params = new URLSearchParams(window.location.search);
    const native = params.get("overlay") !== "fallback" && supportsScrollTimeline();
    root.dataset.overlayPath = native ? "native" : "fallback";
    if (native) {
      return () => {
        delete root.dataset.overlayPath;
      };
    }

    // ---- fallback: --t from the scroll position, one rAF per burst of events ----
    // Stage windows come from the stages' own inline --p0/--p1 (no layout read). A stage
    // gets --t while t is near its window, plus once more when t leaves it (final state).
    const stages = Array.from(root.querySelectorAll<HTMLElement>("[data-overlay-stage]")).map((el) => ({
      el,
      p0: parseFloat(el.style.getPropertyValue("--p0")) || 0,
      p1: parseFloat(el.style.getPropertyValue("--p1")) || VIDEO_DURATION,
      inside: true,
    }));
    let frame = 0;
    let measureNext = true;
    let maxScroll = 1;
    let lastT = "";
    const update = () => {
      frame = 0;
      // reads first (layout is clean at the start of a frame), then writes only
      if (measureNext) {
        measureNext = false;
        maxScroll = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
      }
      const p = Math.min(1, Math.max(0, window.scrollY / maxScroll));
      const tNum = p * VIDEO_DURATION;
      const t = tNum.toFixed(3);
      if (t === lastT) return;
      lastT = t;
      for (const s of stages) {
        const inside = tNum >= s.p0 - 0.75 && tNum <= s.p1 + 0.75;
        if (inside || s.inside) s.el.style.setProperty("--t", t);
        s.inside = inside;
      }
    };
    const schedule = () => {
      if (frame === 0) frame = requestAnimationFrame(update);
    };
    const remeasure = () => {
      measureNext = true;
      schedule();
    };
    const ro = new ResizeObserver(remeasure);
    ro.observe(document.body);
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", remeasure, { passive: true });
    if (supportsScrollTimeline()) {
      // Fallback forced by ?overlay=fallback: the animations already run on scroll(root)
      // and would keep a stale current time when their timeline switches, so restart
      // them once (new ones start paused at 0). Browsers without scroll timelines have
      // no such animations yet and skip this (their load intro is left alone).
      root.dataset.overlayRestart = "";
      root.dataset.overlayFallback = "";
      void root.offsetWidth; // one style flush at start-up, not per scroll event
      delete root.dataset.overlayRestart;
    } else {
      root.dataset.overlayFallback = "";
    }
    update();

    return () => {
      if (frame) cancelAnimationFrame(frame);
      ro.disconnect();
      window.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", remeasure);
      delete root.dataset.overlayFallback;
      delete root.dataset.overlayPath;
      for (const s of stages) s.el.style.removeProperty("--t");
    };
  }, []);

  return (
    <div ref={rootRef} className={styles.root} data-nature-overlay="" style={cssVars({ "--T": VIDEO_DURATION })}>
      <TopBar />
      {children}
    </div>
  );
}
