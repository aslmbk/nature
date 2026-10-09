/**
 * Hero, 0–8 s (frames 01, 02): eyebrow, two-line headline (grotesque + italic serif),
 * CTA, a small product mock in the middle (own graphics: growth rings drawn from the
 * seed emblem, two progress bars, a note bubble, a stat card), a short blurb bottom
 * left and "Scroll to explore" bottom right. Enters with a one-off load intro, leaves
 * upwards while the camera descends (8–9.3 s); each piece fades itself on the way (the
 * headline line by line), before it reaches the top bar's row (exitFades.ts).
 */
import { cssVars, cx } from "../cssVars";
import { Stage, intro, motion as m } from "../Motion";
import { SEED_PATHS } from "../SeedMark";
import { ArrowIcon, Button, Sparkle, ui as u } from "../ui";
import s from "./HeroSection.module.css";

const LINE_1 = ["Grow", "your", "focus"];
const LINE_2 = ["one", "season", "at", "a", "time."];
const WORD_START = 1.0;
const WORD_STEP = 0.09;

function Word({ text, index }: { text: string; index: number }) {
  return (
    <span className={cx(m.intro, m.word)} style={intro(WORD_START + index * WORD_STEP, { dur: 0.75, rise: 14, blur: 12 })}>
      {text}
    </span>
  );
}

function words(list: string[], index0: number) {
  return list.flatMap((w, i) => [i > 0 ? " " : null, <Word key={w + i} text={w} index={index0 + i} />]);
}

/** Lens (almond) of radius r around the emblem centres (±37.5, 0): the seed's growth rings. */
function lens(r: number, c = 37.5): string {
  const h = Math.sqrt(r * r - c * c);
  const f = (v: number) => v.toFixed(2);
  return `M0 ${f(-h)}A${r} ${r} 0 0 1 0 ${f(h)}A${r} ${r} 0 0 1 0 ${f(-h)}Z`;
}

const RINGS = [71, 81, 93, 107, 123, 141, 161, 183].map((r) => lens(r));

function GrowthRings() {
  return (
    <svg className={s.rings} viewBox="-170 -150 340 300" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
      <defs>
        <linearGradient id="heroSeedFill" x1="0" y1="-1" x2="0" y2="1">
          <stop offset="0" stopColor="#d6f08f" />
          <stop offset="0.55" stopColor="#8fbf3c" />
          <stop offset="1" stopColor="#4c6d22" />
        </linearGradient>
      </defs>
      <g transform="rotate(-8)">
        {RINGS.map((d, i) => (
          <path key={i} d={d} className={s.ring} style={cssVars({ "--i": i })} />
        ))}
        <g transform="scale(0.98)" fill="url(#heroSeedFill)" stroke="url(#heroSeedFill)" strokeWidth={2} strokeLinejoin="round">
          <path d={SEED_PATHS[0]} />
          <path d={SEED_PATHS[1]} />
        </g>
      </g>
      <circle cx="-104" cy="-62" r="3" className={s.spore} />
      <circle cx="118" cy="38" r="2.4" className={s.spore} />
      <circle cx="74" cy="-104" r="1.8" className={s.spore} />
    </svg>
  );
}

function InfoIcon() {
  return (
    <svg className={s.info} viewBox="0 0 20 20" aria-hidden="true">
      <circle cx="10" cy="10" r="8.6" fill="none" stroke="currentColor" strokeWidth="1.2" />
      <path d="M10 8.6v5.2" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
      <circle cx="10" cy="6.1" r="0.95" fill="currentColor" />
    </svg>
  );
}

export default function HeroSection() {
  return (
    <Stage id="hero" className={s.stage}>
      <div className={s.head}>
        <p className={cx(u.eyebrow, m.introOut)} style={intro(0.85, { rise: 8, blur: 8 })}>
          A calm companion for focused study
        </p>
        <h1 className={cx(u.title, s.title)}>
          <span className={cx(u.line, m.out)}>{words(LINE_1, 0)}</span>
          <span className={cx(u.line, u.serif, u.italic, s.serifLine, m.out)}>{words(LINE_2, LINE_1.length)}</span>
        </h1>
        <Button href="#plans" className={cx(m.introOutSoft, s.cta)} style={intro(1.65, { rise: 10 })}>
          Get started
        </Button>
      </div>

      {/* product mock: illustrative only */}
      <div className={s.cluster} aria-hidden="true">
        <div className={cx(s.card, m.introOutSoft)} style={intro(1.2, { dur: 1.1, rise: 60, scale: 0.94 })}>
          <GrowthRings />
          <div className={s.cardBody}>
            <p className={s.cardTitle}>Focus Seasons</p>
            <div className={s.rows}>
              <div className={s.row}>
                <span>This week</span>
                <span className={s.track}>
                  <i className={s.fill} style={cssVars({ "--w": 0.46 })} />
                </span>
              </div>
              <div className={s.row}>
                <span>Review</span>
                <span className={cx(s.track, s.amber)}>
                  <i className={s.fill} style={cssVars({ "--w": 0.78 })} />
                </span>
              </div>
            </div>
          </div>
        </div>

        <div className={cx(s.bubble, m.introOutSoft)} style={intro(1.55, { rise: 24, scale: 0.96 })}>
          <p className={s.bubbleSerif}>A slower week is still growth.</p>
          <p className={s.bubbleNote}>Three short sessions are planned for today.</p>
        </div>

        <div className={cx(s.stat, m.introOutSoft)} style={intro(1.75, { rise: 24, scale: 0.96 })}>
          <div className={s.statHead}>
            <div>
              <span className={s.count} />
              <span className={s.statLabel}>Weekly rhythm</span>
            </div>
            <InfoIcon />
          </div>
          <span className={s.statButton}>
            Open the planner
            <span className={s.statChip}>
              <ArrowIcon />
            </span>
          </span>
        </div>
      </div>

      <div className={s.bottom}>
        <p className={cx(s.blurb, m.introOut)} style={intro(1.95, { rise: 10, blur: 8 })}>
          Silva turns study time into seasons: short focused sessions, real rest and a quiet record of what you have grown.
        </p>
        <a href="#about" className={cx(u.textLink, s.more, m.introOutSoft)} style={intro(2.1, { rise: 8 })}>
          Learn more
        </a>
      </div>

      <div className={cx(s.hint, m.out)}>
        <span className={cx(s.hintInner, m.introSoft)} style={intro(2.35, { rise: 8 })}>
          <Sparkle className={s.sparkle} />
          <span>Scroll to explore</span>
        </span>
      </div>
    </Stage>
  );
}
