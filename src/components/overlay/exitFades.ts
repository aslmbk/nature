/**
 * Exit fades that keep the overlay out of the top bar's row.
 *
 * When a stage leaves, its layer slides up (Stage: --o0 --o1 --oshift --oease) and its
 * pieces fade over the stage's window (--f0 --f1). A piece high in the frame would reach
 * the transparent top bar before that window ends and slide on under the logo and the
 * menu. Each fading piece therefore gets its own window here (--xf0 --xf1, used under full
 * motion only, motion.module.css): it fades while its top travels the last `zone` px
 * before the bar's row and is gone when it gets there; pieces lower down keep the stage's
 * window. Headline words fade one by one, so a headline dissolves line by line from the top.
 *
 * Positions come from layout (offsetTop: transforms are ignored, so the result does not
 * depend on the scroll position or on running entrances) plus the end pose of `.move`
 * wrappers that finished before the exit. Runs on mount, after the web fonts loaded and
 * after resizes (Overlay.tsx); writes only where a window changes.
 *
 * Elements marked `data-exit="span"` (soft halos behind a text column) take the span of
 * the windows of the other fading pieces beside them instead: they leave with the text.
 */
import m from "./motion.module.css";

/** Gap kept between a fading piece and the bottom of the top bar (px). */
const BAR_GAP = 12;
/** Shortest fade (video seconds). */
const MIN_FADE = 0.06;

const FADE_CLASSES = [m.out, m.inout, m.inoutSoft, m.inoutRecede, m.introOut, m.introOutSoft];

type Ease = (e: number) => number;

/** Inverse of a CSS easing (output progress → input progress); null when unsupported. */
function inverseEase(spec: string): Ease | null {
  const s = spec.trim();
  if (s === "" || s === "linear") return (e) => e;
  const match = /^cubic-bezier\(([^)]+)\)$/.exec(s);
  if (!match) return null;
  const [x1, y1, x2, y2] = match[1].split(",").map((v) => parseFloat(v));
  if (![x1, y1, x2, y2].every(Number.isFinite)) return null;
  const bez = (a: number, b: number, u: number) => 3 * (1 - u) * (1 - u) * u * a + 3 * (1 - u) * u * u * b + u * u * u;
  return (e) => {
    // the overlay's curves have y monotonic in u: bisection on y(u) = e, then x(u)
    let lo = 0;
    let hi = 1;
    for (let i = 0; i < 40; i++) {
      const mid = (lo + hi) / 2;
      if (bez(y1, y2, mid) < e) lo = mid;
      else hi = mid;
    }
    return bez(x1, x2, (lo + hi) / 2);
  };
}

/** A length from a --y1 value ("12vh", "40px", "-5%" of the element's height, "0"). */
function toPx(value: string, vh: number, el: HTMLElement): number {
  const v = value.trim();
  const n = parseFloat(v);
  if (!Number.isFinite(n)) return 0;
  if (v.endsWith("vh")) return n * vh;
  if (v.endsWith("%")) return (n / 100) * el.offsetHeight;
  return n;
}

/** Layout top of `el` inside `stage` (px), or null when it is not laid out. */
function topWithin(el: HTMLElement, stage: HTMLElement): number | null {
  let y = 0;
  let n: HTMLElement = el;
  while (n !== stage) {
    const parent = n.offsetParent as HTMLElement | null;
    if (!parent) return null;
    y += n.offsetTop;
    if (parent !== stage) y += parent.clientTop;
    n = parent;
  }
  return y;
}

/** Vertical offset of the finished `.move` / `.scrub` animations on `el` and its ancestors. */
function settledShift(el: HTMLElement, stage: HTMLElement, exitStart: number, vh: number, motion: number): number {
  let dy = 0;
  for (let n: HTMLElement | null = el; n && n !== stage; n = n.parentElement) {
    if (!n.classList.contains(m.move) && !n.classList.contains(m.scrub)) continue;
    const end = parseFloat(n.style.getPropertyValue("--b"));
    if (!(end <= exitStart + 1e-3)) continue;
    // a breakpoint may switch the animation off (streams cards on narrow screens)
    if (getComputedStyle(n).animationName === "none") continue;
    dy += toPx(n.style.getPropertyValue("--y1"), vh, n) * motion;
  }
  return dy;
}

interface Planned {
  el: HTMLElement;
  f0: number | null;
  f1: number | null;
}

/** Measures every stage and writes the per-piece exit windows (see the file comment). */
export function layoutExitFades(root: HTMLElement): void {
  const rawMotion = parseFloat(getComputedStyle(root).getPropertyValue("--motion"));
  const motion = Number.isFinite(rawMotion) ? rawMotion : 1;
  const full = motion >= 0.5;
  const selector = FADE_CLASSES.filter(Boolean)
    .map((c) => `.${CSS.escape(c)}`)
    .join(",");
  const plan: Planned[] = [];

  if (full) {
    const header = root.querySelector("header");
    const barBottom = header ? header.getBoundingClientRect().bottom : 0;
    const zoneTop = barBottom + BAR_GAP;
    const zone = Math.min(110, Math.max(48, 0.09 * window.innerHeight));
    // 1vh as the stylesheet resolves it (the large viewport on mobile)
    const probe = document.createElement("div");
    probe.style.cssText = "position:fixed;top:0;left:0;width:0;height:100vh;visibility:hidden;pointer-events:none";
    document.body.appendChild(probe);
    const vh = probe.getBoundingClientRect().height / 100 || window.innerHeight / 100;
    probe.remove();

    for (const stage of root.querySelectorAll<HTMLElement>("[data-overlay-stage]")) {
      const o0 = parseFloat(stage.style.getPropertyValue("--o0"));
      const o1 = parseFloat(stage.style.getPropertyValue("--o1"));
      const shift = parseFloat(stage.style.getPropertyValue("--oshift"));
      const inv = inverseEase(stage.style.getPropertyValue("--oease") || "linear");
      const pieces = Array.from(stage.querySelectorAll<HTMLElement>(selector));
      if (inv === null || !(Number.isFinite(o0) && Number.isFinite(o1) && o1 > o0 && shift < 0)) {
        for (const el of pieces) plan.push({ el, f0: null, f1: null });
        continue;
      }
      const travel = -shift * vh * motion; // px the layer moves up over the exit
      // time at which the slide has carried a piece `dist` px up (∞: not within the exit)
      const timeAt = (dist: number): number => {
        const e = dist / travel;
        if (e <= 0) return o0;
        if (e >= 1) return Number.POSITIVE_INFINITY;
        return o0 + (o1 - o0) * inv(e);
      };
      const spans: HTMLElement[] = [];
      const own = new Map<HTMLElement, [number, number]>();
      for (const el of pieces) {
        if (el.dataset.exit === "span") {
          spans.push(el);
          continue;
        }
        const cs = getComputedStyle(el);
        const F0 = parseFloat(cs.getPropertyValue("--f0"));
        const F1 = parseFloat(cs.getPropertyValue("--f1"));
        const top = topWithin(el, stage);
        if (!Number.isFinite(F0) || !Number.isFinite(F1) || top === null) {
          plan.push({ el, f0: null, f1: null });
          continue;
        }
        const y = top + settledShift(el, stage, o0, vh, motion);
        let f1 = Math.min(F1, timeAt(y - zoneTop));
        let f0 = Math.min(F0, timeAt(y - zoneTop - zone));
        if (f1 - f0 < MIN_FADE) f0 = f1 - MIN_FADE;
        f0 = Math.round(f0 * 1000) / 1000;
        f1 = Math.round(f1 * 1000) / 1000;
        const changed = Math.abs(f0 - F0) > 5e-4 || Math.abs(f1 - F1) > 5e-4;
        own.set(el, [f0, f1]);
        plan.push({ el, f0: changed ? f0 : null, f1: changed ? f1 : null });
      }
      for (const span of spans) {
        // the windows of the fading pieces that share the halo's parent
        let a = Number.POSITIVE_INFINITY;
        let b = Number.NEGATIVE_INFINITY;
        for (const [el, [f0, f1]] of own) {
          if (span.parentElement && span.parentElement.contains(el)) {
            a = Math.min(a, f0);
            b = Math.max(b, f1);
          }
        }
        const cs = getComputedStyle(span);
        const same =
          Math.abs(a - parseFloat(cs.getPropertyValue("--f0"))) <= 5e-4 && Math.abs(b - parseFloat(cs.getPropertyValue("--f1"))) <= 5e-4;
        plan.push(Number.isFinite(a) && Number.isFinite(b) && !same ? { el: span, f0: a, f1: b } : { el: span, f0: null, f1: null });
      }
    }
  } else {
    for (const el of root.querySelectorAll<HTMLElement>(selector)) plan.push({ el, f0: null, f1: null });
  }

  // writes after all reads (one style invalidation, no layout thrash)
  for (const { el, f0, f1 } of plan) {
    if (f0 === null || f1 === null) {
      el.style.removeProperty("--xf0");
      el.style.removeProperty("--xf1");
    } else {
      if (el.style.getPropertyValue("--xf0") !== String(f0)) el.style.setProperty("--xf0", String(f0));
      if (el.style.getPropertyValue("--xf1") !== String(f1)) el.style.setProperty("--xf1", String(f1));
    }
  }
}
