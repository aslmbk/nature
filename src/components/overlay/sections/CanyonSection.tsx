/**
 * Canyon, 15.5–21 s (frame 06): centred eyebrow + headline + paragraph at the top, three
 * glass cards below. The reference cards hold nature photos and grade widgets; here the
 * blurred canvas shows through the glass and each card carries a small own graphic
 * (bar week, sparkline window, streak list).
 */
import { cssVars, cx } from "../cssVars";
import { Stage, Words, motion as m, win } from "../Motion";
import { ui as u } from "../ui";
import sh from "./shared.module.css";
import s from "./CanyonSection.module.css";

const W0 = 16.4;
const WEEK = [0.46, 0.72, 0.58, 0.88, 0.64, 0.3, 0.18];
const DAYS = ["M", "T", "W", "T", "F", "S", "S"];

function WeekGraphic() {
  return (
    <div className={cx(s.panel, s.panelWeek)} aria-hidden="true">
      <p className={s.panelLabel}>Focus hours</p>
      <p className={s.big}>11.5</p>
      <p className={s.caption}>Across the last seven days</p>
      <div className={s.dash} />
      <div className={s.bars}>
        {WEEK.map((h, i) => (
          <span key={i} className={s.barCol}>
            <i style={cssVars({ "--h": h })} />
            <em>{DAYS[i]}</em>
          </span>
        ))}
      </div>
    </div>
  );
}

function WindowGraphic() {
  return (
    <div className={s.windowWrap} aria-hidden="true">
      <div className={s.chip}>
        <span className={s.chipIcon}>
          <svg viewBox="0 0 16 16">
            <path d="M8.6 2.2 4.4 9h3.1l-.9 4.8L11.6 6.6H8.4l.2-4.4Z" fill="currentColor" />
          </svg>
        </span>
        <span className={s.chipValue}>
          64%
          <svg viewBox="0 0 14 10" className={s.chipArrow}>
            <path d="M1 2.5 5 6.5 7.5 4 12.5 8.5M12.5 8.5V5.2M12.5 8.5H9.2" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </span>
        <span className={s.chipNote}>steadier than last week</span>
      </div>
      <div className={s.paper}>
        <p className={s.paperLabel}>Best window</p>
        <p className={s.paperBig}>
          9–11<small>am</small>
        </p>
        <svg className={s.spark} viewBox="0 0 200 70" preserveAspectRatio="none">
          <defs>
            <linearGradient id="canyonSpark" x1="0" x2="1" y1="0" y2="0">
              <stop offset="0" stopColor="#bfe86a" stopOpacity="0.2" />
              <stop offset="0.35" stopColor="#8fd43b" />
              <stop offset="1" stopColor="#bfe86a" stopOpacity="0.35" />
            </linearGradient>
          </defs>
          <path d="M6 52 C 34 52, 42 30, 70 30 S 104 50, 124 48 S 150 22, 172 26 S 190 40, 196 40" fill="none" stroke="url(#canyonSpark)" strokeWidth="5" strokeLinecap="round" />
          <circle cx="98" cy="42" r="6" fill="#ffffff" stroke="#8fd43b" strokeWidth="3" />
        </svg>
      </div>
    </div>
  );
}

function StreakGraphic() {
  const rows = [
    { label: "Reading", value: "4 h", tone: "#c9b458" },
    { label: "Practice", value: "3 h", tone: "#9ed04a" },
    { label: "Review", value: "2 h", tone: "#e8ece2" },
  ];
  return (
    <div className={cx(s.panel, s.panelStreak)} aria-hidden="true">
      <div className={s.panelHead}>
        <p className={s.panelLabel}>Consistency</p>
        <span className={s.pill}>Steady</span>
      </div>
      <p className={s.big}>
        5<span className={s.slash}>/</span>7
      </p>
      <p className={cx(s.caption, s.captionSerif)}>Days with a session this week</p>
      <ul className={s.list}>
        {rows.map((r) => (
          <li key={r.label}>
            <i style={{ background: r.tone }} />
            <span>{r.label}</span>
            <b>{r.value}</b>
          </li>
        ))}
      </ul>
      <div className={s.dash} />
    </div>
  );
}

const CARDS = [
  {
    key: "week",
    title: "Weekly Rhythm",
    text: "A calm view of how your study time spreads across the week.",
    art: <WeekGraphic />,
    tint: s.tintWarm,
  },
  {
    key: "window",
    title: "Focus Window",
    text: "See the hours of the day when your attention tends to settle.",
    art: <WindowGraphic />,
    tint: s.tintGreen,
  },
  {
    key: "streak",
    title: "Steady Streak",
    text: "Small daily sessions, kept in view without any pressure.",
    art: <StreakGraphic />,
    tint: s.tintCool,
  },
];

export default function CanyonSection() {
  return (
    <Stage id="canyon">
      <div className={cx(sh.column, s.column)}>
        <span className={cx(sh.halo, s.halo, m.inoutSoft)} data-exit="span" style={win(16.3, 17.2, { rise: 0 })} aria-hidden="true" />
        <p className={cx(u.eyebrow, m.inout)} style={win(16.3, 16.9, { rise: 10 })}>
          See the season ahead
        </p>
        <h2 className={cx(u.title, sh.titleGap, s.title)}>
          <span className={u.line}>
            <Words text="Notice the pattern" from={W0} />
          </span>
          <span className={cx(u.line, u.serif, u.italic)}>
            <Words text="before the week ends." from={W0} index0={3} />
          </span>
        </h2>
        <p className={cx(u.lead, sh.leadGap, s.lead, m.inout)} style={win(16.85, 17.5, { rise: 14, blur: 6 })}>
          Silva keeps a gentle record of when you focus best, what you return to and where time slips away, so each new week
          starts with a clearer view.
        </p>
      </div>

      <ul className={s.cards}>
        {CARDS.map((c, i) => (
          <li
            key={c.key}
            className={cx(u.glass, s.card, c.tint, m.inoutSoft)}
            style={win(16.95 + i * 0.15, 17.85 + i * 0.15, { rise: 60 })}
          >
            <div className={s.art}>{c.art}</div>
            <div className={s.text}>
              <h3 className={u.cardTitle}>{c.title}</h3>
              <p className={cx(u.cardText, s.cardText)}>{c.text}</p>
            </div>
          </li>
        ))}
      </ul>
    </Stage>
  );
}
