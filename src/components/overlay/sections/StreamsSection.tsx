/**
 * Streams, 25–31 s (frame 09, sheet 08): centred eyebrow, two-line headline and
 * paragraph over the particle loop, two small glass cards at the sides. The cards are
 * carried by the scroll in opposite directions (the left one comes down from above,
 * the right one up from below), rest level with each other around 28–28.7 s and drift
 * apart again before the ragged wipe takes the scene away.
 *
 * The reference's left card holds a nature photo; here the blurred canvas shows
 * through the glass and the cards carry small own icons.
 */
import { cx } from "../cssVars";
import { Stage, Words, motion as m, move, win } from "../Motion";
import { EASE } from "../timeline";
import { ui as u } from "../ui";
import sh from "./shared.module.css";
import s from "./StreamsSection.module.css";

const W0 = 25.75;

function DottedRing() {
  return (
    <svg className={s.icon} viewBox="0 0 32 32" aria-hidden="true">
      <circle cx="16" cy="16" r="11" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" strokeDasharray="0.1 5.65" />
      <circle cx="16" cy="16" r="2.6" fill="currentColor" />
    </svg>
  );
}

function BalanceIcon() {
  return (
    <svg className={s.icon} viewBox="0 0 32 32" aria-hidden="true">
      <g fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round">
        <circle cx="8" cy="8" r="3.6" />
        <circle cx="24" cy="8" r="3.6" />
        <circle cx="8" cy="24" r="3.6" />
        <circle cx="24" cy="24" r="3.6" />
        <path d="M11.6 8h8.8M11.6 24h8.8M8 11.6v8.8M24 11.6v8.8" />
      </g>
    </svg>
  );
}

const CARDS = [
  {
    key: "outline",
    side: "left" as const,
    title: "Week Ahead Outline",
    text: "Plan the coming days around the time and energy you really have.",
    icon: <DottedRing />,
    // comes down from above, rests, then keeps drifting down; fades in only once it has
    // passed the top bar's row (on short screens its text would cross under the bar)
    in: { y: "-51vh" },
    out: { y: "29vh" },
    enter: [25.55, 26.05] as const,
  },
  {
    key: "balance",
    side: "right" as const,
    title: "Attention Balance",
    text: "Shows how evenly your focus is shared between subjects.",
    icon: <BalanceIcon />,
    // comes up from below, rests, then keeps drifting up
    in: { y: "47vh" },
    out: { y: "-19vh" },
    enter: [25.2, 25.75] as const,
  },
];

export default function StreamsSection() {
  return (
    <Stage id="streams">
      <div className={cx(sh.column, s.column)}>
        <span className={cx(sh.halo, s.halo, m.inoutSoft)} data-exit="span" style={win(25.7, 26.7, { rise: 0 })} aria-hidden="true" />
        <p className={cx(u.eyebrow, m.inout)} style={win(25.7, 26.3, { rise: 30 })}>
          Season dashboard
        </p>
        <h2 className={cx(u.title, sh.titleGap, s.title)}>
          <span className={u.line}>
            <Words text="A calmer way to" from={W0} opts={{ rise: 40, blur: 12 }} />
          </span>
          <span className={cx(u.line, u.serif, u.italic)}>
            <Words text="See Your Growth" from={W0} index0={4} opts={{ rise: 40, blur: 12 }} />
          </span>
        </h2>
        <p className={cx(u.lead, sh.leadGap, s.lead, m.inout)} style={win(26.4, 27.1, { rise: 14, blur: 6 })}>
          Every session leaves a trace. The season dashboard turns those traces into a few plain signals you can read at a
          glance, with no scores and no alarms.
        </p>
      </div>

      {CARDS.map((c) => (
        <div
          key={c.key}
          className={cx(s.slot, c.side === "left" ? s.left : s.right, m.move)}
          style={move(25.2, 27.6, c.in, { y: "0vh" }, EASE.out)}
        >
          <div className={cx(s.drift, m.move)} style={move(28.7, 30.2, { y: "0vh" }, c.out, EASE.out)}>
            {/* reduced motion: in only after the oracle text has gone (nothing slides there) */}
            <article
              className={cx(u.glassDark, s.card, c.side === "left" ? s.cardLeft : s.cardRight, m.inoutSoft)}
              style={win(c.enter[0], c.enter[1], { rise: 0, reduce: [25.3, 25.85] })}
            >
              <span className={c.side === "left" ? s.iconLime : s.iconGrey}>{c.icon}</span>
              <h3 className={cx(u.cardTitle, s.cardTitle)}>{c.title}</h3>
              <p className={cx(u.cardText, s.cardText)}>{c.text}</p>
            </article>
          </div>
        </div>
      ))}
    </Stage>
  );
}
