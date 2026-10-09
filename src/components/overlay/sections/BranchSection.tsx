/**
 * Branch, 31–37 s (frames 11, 13, sheet 09): light episode — white text with a soft
 * shadow, centred headline and paragraph, two frosted cards below. Own graphics: a focus
 * arc that draws itself as the cards arrive, and a four-step review cycle that turns
 * with the scroll (labels stay upright). Everything rides up with the branch scene's
 * slide (37–39.5 s).
 */
import { cssVars, cx } from "../cssVars";
import { Anchor, Stage, Words, motion as m, move, win } from "../Motion";
import { EASE } from "../timeline";
import { ui as u } from "../ui";
import sh from "./shared.module.css";
import s from "./BranchSection.module.css";

const W0 = 31.55;

// ---- focus arc (left card): circle around (227, 180), r 150, from 180° to 15.8° ----
const ARC_C = { x: 227, y: 180 };
const ARC_R = 150;
const pt = (deg: number, r = ARC_R) => {
  const a = (deg * Math.PI) / 180;
  return { x: ARC_C.x + r * Math.cos(a), y: ARC_C.y - r * Math.sin(a) };
};
const f1 = (v: number) => v.toFixed(1);
const ARC_START = pt(180);
const ARC_END = pt(15.8);
const ARC_D = `M${f1(ARC_START.x)} ${f1(ARC_START.y)}A${ARC_R} ${ARC_R} 0 0 1 ${f1(ARC_END.x)} ${f1(ARC_END.y)}`;
const SPOKES = [150, 122, 94, 66, 38].map((deg) => {
  const a = pt(deg, 46);
  const b = pt(deg, 142);
  return `M${f1(a.x)} ${f1(a.y)}L${f1(b.x)} ${f1(b.y)}`;
});

function FocusArc() {
  return (
    <svg className={s.arcSvg} viewBox="0 0 454 214" aria-hidden="true">
      <defs>
        <linearGradient id="branchArc" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stopColor="#e9fbc4" />
          <stop offset="0.5" stopColor="#c8f07a" />
          <stop offset="1" stopColor="#a6e23e" />
        </linearGradient>
      </defs>
      <g className={s.spokes}>
        {SPOKES.map((d) => (
          <path key={d} d={d} />
        ))}
        <path d={`M${f1(ARC_START.x)} ${ARC_C.y}H${f1(pt(0).x)}`} />
      </g>
      <path
        d={ARC_D}
        pathLength={100}
        strokeDasharray={100}
        className={cx(s.arc, m.draw)}
        stroke="url(#branchArc)"
        style={win(31.95, 32.85)}
      />
      <circle cx={f1(ARC_START.x)} cy={f1(ARC_START.y)} r="6.5" className={cx(s.dot, m.inSoft)} style={win(31.85, 32.05, { rise: 0 })} />
      <circle cx={f1(ARC_END.x)} cy={f1(ARC_END.y)} r="6.5" className={cx(s.dot, m.inSoft)} style={win(32.7, 32.9, { rise: 0 })} />
    </svg>
  );
}

// ---- review cycle (right card) -------------------------------------------------------
const CYCLE_R = 75;
const STEPS = ["Read", "Recall", "Rest", "Revisit"];
// screen angles (clockwise from +x): top, right, bottom, left
const STEP_ANGLES = [-90, 0, 90, 180];
const cp = (deg: number, r = CYCLE_R) => {
  const a = (deg * Math.PI) / 180;
  return { x: r * Math.cos(a), y: r * Math.sin(a) };
};
const CYCLE_ARCS = STEP_ANGLES.map((a0) => {
  const s0 = cp(a0 + 25);
  const e = a0 + 66;
  const e0 = cp(e);
  const dir = { x: -Math.sin((e * Math.PI) / 180), y: Math.cos((e * Math.PI) / 180) };
  const rad = { x: Math.cos((e * Math.PI) / 180), y: Math.sin((e * Math.PI) / 180) };
  const back = { x: e0.x - dir.x * 7, y: e0.y - dir.y * 7 };
  const w1 = { x: back.x + rad.x * 4.5, y: back.y + rad.y * 4.5 };
  const w2 = { x: back.x - rad.x * 4.5, y: back.y - rad.y * 4.5 };
  return {
    arc: `M${f1(s0.x)} ${f1(s0.y)}A${CYCLE_R} ${CYCLE_R} 0 0 1 ${f1(e0.x)} ${f1(e0.y)}`,
    head: `M${f1(w1.x)} ${f1(w1.y)}L${f1(e0.x)} ${f1(e0.y)}L${f1(w2.x)} ${f1(w2.y)}`,
  };
});
// turns with the scroll: 0° (Read on top) at 34.5 s, ≈ 37.5° per second (frames 11 → 13)
const TURN_A = 32.0;
const TURN_B = 39.5;
const TURN_0 = -94;
const TURN_1 = 187;

function ReviewCycle() {
  return (
    <div className={s.cycleWrap} aria-hidden="true">
      <div className={cx(s.cycle, m.move)} style={move(TURN_A, TURN_B, { r: `${TURN_0}deg` }, { r: `${TURN_1}deg` }, EASE.linear)}>
        <svg className={s.cycleSvg} viewBox="-100 -100 200 200">
          <defs>
            <linearGradient id="branchCycle" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stopColor="#f1fbd9" />
              <stop offset="1" stopColor="#b8e85c" />
            </linearGradient>
          </defs>
          {CYCLE_ARCS.map((c, i) => (
            <g key={i} className={s.cycleArc} stroke="url(#branchCycle)">
              <path d={c.arc} />
              <path d={c.head} />
            </g>
          ))}
        </svg>
        {STEPS.map((label, i) => {
          const p = cp(STEP_ANGLES[i]);
          return (
            <span
              key={label}
              className={cx(s.spot, m.move)}
              style={move(TURN_A, TURN_B, { r: `${-TURN_0}deg` }, { r: `${-TURN_1}deg` }, EASE.linear, cssVars({ "--px": p.x / 100, "--py": p.y / 100 }))}
            >
              <span className={s.pill}>{label}</span>
            </span>
          );
        })}
      </div>
    </div>
  );
}

const CARDS = [
  {
    key: "focus",
    title: "Flexible Focus Blocks",
    text: "Blocks grow or shrink a little with your energy and always stay inside the limits you choose.",
    art: <FocusArc />,
    size: "wide" as const,
  },
  {
    key: "cycle",
    title: "Review Cycle",
    text: "Read, recall, rest and revisit: a simple loop that keeps what you learn within reach.",
    art: <ReviewCycle />,
    size: "narrow" as const,
  },
];

export default function BranchSection() {
  return (
    <>
      <Anchor id="method" />
      <Stage id="branch">
        <div className={cx(sh.column, sh.onLight, s.column)}>
          <span className={cx(sh.halo, s.halo, m.inoutSoft)} data-exit="span" style={win(31.5, 32.4, { rise: 0 })} aria-hidden="true" />
          <p className={cx(u.eyebrow, s.eyebrow, m.inout)} style={win(31.5, 32.0, { rise: 0, blur: 10 })}>
            The Silva method
          </p>
          <h2 className={cx(u.title, sh.titleGap, s.title)}>
            <span className={u.line}>
              <Words text="Sessions that bend" from={W0} opts={{ rise: 0, blur: 14 }} />
            </span>
            <span className={cx(u.line, u.serif, u.italic)}>
              <Words text="around your days" from={W0} index0={3} opts={{ rise: 0, blur: 14 }} />
            </span>
          </h2>
          <p className={cx(u.lead, sh.leadGap, s.lead, m.inout)} style={win(32.05, 32.7, { rise: 10, blur: 8 })}>
            Each block is shaped around your energy, your pace and the goals you set, and it adjusts gently when a day goes
            differently.
          </p>
        </div>

        <ul className={s.cards}>
          {CARDS.map((c, i) => (
            <li
              key={c.key}
              className={cx(u.glass, s.card, c.size === "wide" ? s.wide : s.narrow, m.inoutSoft)}
              style={win(31.75 + i * 0.12, 32.5 + i * 0.12, { rise: 34 })}
            >
              <div className={s.art}>{c.art}</div>
              <div className={s.text}>
                <h3 className={cx(u.cardTitle, s.cardTitle)}>{c.title}</h3>
                <p className={cx(u.cardText, s.cardText)}>{c.text}</p>
              </div>
            </li>
          ))}
        </ul>
      </Stage>
    </>
  );
}
