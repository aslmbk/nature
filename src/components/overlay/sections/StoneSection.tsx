/**
 * Stone, 37–43.3 s (frames 13, 14, sheet 10): a giant one-line statement that rises with
 * the dark stone scene (37–39.5 s) and keeps sliding left with the scroll (≈ 13 vw per
 * video second: x 630 px at 38.5 s, 40 px at 41.7 s), and a left text column with a
 * three-item list that resolves once the scene has settled. Leaves upwards with the
 * stone → canopy slide (43.0–43.75 s).
 *
 * The giant line fades itself (a .move track carries it sideways): out just before the
 * slide; under reduced motion, where nothing rises or slides, it fades in only once the
 * branch text has gone.
 */
import { cssVars, cx } from "../cssVars";
import { Stage, Words, motion as m, move, win } from "../Motion";
import { EASE } from "../timeline";
import { ui as u } from "../ui";
import s from "./StoneSection.module.css";

const W0 = 39.7;
const POINTS = [
  "Notices your focus patterns through the week",
  "Adjusts plans to the progress you really make",
  "Changes gently as your seasons change",
];

export default function StoneSection() {
  return (
    <Stage id="stone">
      <div className={cx(s.rise, m.move)} style={move(37.0, 39.5, { y: "86vh" }, { y: "0vh" }, EASE.inOutSine)}>
        <div className={cx(s.track, m.move)} style={move(37.0, 43.75, { x: "60.1vw" }, { x: "-26.2vw" }, EASE.linear)}>
          <p
            className={cx(s.giant, m.inoutSoft)}
            style={win(36.95, 37.0, { rise: 0, reduce: [38.35, 39.0] }, cssVars({ "--f0": 42.5, "--f1": 42.98 }))}
            aria-hidden="true"
          >
            <span className={s.lime}>Silva</span> <span className={s.white}>grows</span> <span className={s.dash} />{" "}
            <span className={s.grey}>season after season</span>
          </p>
        </div>

        <div className={s.column}>
          <p className={cx(u.eyebrow, m.inout)} style={win(39.6, 40.2, { rise: 12 })}>
            Personal by design
          </p>
          <h2 className={cx(u.title, s.title)}>
            <span className={u.srOnly}>Silva grows season after season. </span>
            <span className={u.line}>
              <Words text="Quietly shaped" from={W0} />
            </span>
            <span className={cx(u.line, u.serif)}>
              <Words text="by your habits" from={W0} index0={2} />
            </span>
          </h2>
          <p className={cx(u.lead, s.lead, m.inout)} style={win(40.35, 41.0, { rise: 14, blur: 6 })}>
            Silva learns from the way you actually study: your routines, your pauses and the subjects you return to. Over
            time the planner, the prompts and the pace settle into a shape that fits you.
          </p>
          <ul className={s.points}>
            {POINTS.map((p, i) => (
              <li key={p} className={m.inout} style={win(40.7 + i * 0.28, 41.3 + i * 0.28, { rise: 10, blur: 6 })}>
                {p}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </Stage>
  );
}
