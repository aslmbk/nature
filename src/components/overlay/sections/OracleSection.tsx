/**
 * Oracle, 21–25 s (frames 07, 08): eyebrow + two-line headline above the particle orb,
 * paragraph + CTA below it. The headline group rises ~110 px while it resolves (the
 * eyebrow is still blurred and low at 21.5 s), everything leaves upwards with the
 * oracle → streams camera move (24.5–25.5 s).
 */
import { cx } from "../cssVars";
import { Anchor, Stage, Words, motion as m, move, win } from "../Motion";
import { EASE } from "../timeline";
import { Button, ui as u } from "../ui";
import sh from "./shared.module.css";
import s from "./OracleSection.module.css";

const W0 = 21.55;

export default function OracleSection() {
  return (
    <>
      <Anchor id="grove" />
      <Stage id="oracle">
        <div className={cx(s.top, m.move)} style={move(21.25, 21.85, { y: "110px" }, { y: "0px" }, EASE.linear)}>
          <div className={cx(sh.column, s.column)}>
            <p className={cx(u.eyebrow, m.inout)} style={win(21.3, 21.9, { rise: 0, blur: 12 })}>
              The heart of Silva
            </p>
            <h2 className={cx(u.title, sh.titleGap, s.title)}>
              <span className={u.line}>
                <Words text="The Grove remembers" from={W0} opts={{ rise: 0, blur: 14 }} />
              </span>
              <span className={cx(u.line, u.serif, u.italic)}>
                <Words text="every small step" from={W0} index0={3} step={0.1} opts={{ rise: 0, blur: 14 }} />
              </span>
            </h2>
          </div>
        </div>

        <div className={cx(sh.column, s.bottom)}>
          <p className={cx(u.lead, s.lead, m.inout)} style={win(22.05, 22.75, { rise: 10, blur: 8 })}>
            The Grove gathers your sessions, notes and pauses into one living picture. It notices what is taking root and
            points to where your attention could go next.
          </p>
          <Button href="#method" className={cx(s.cta, m.inoutSoft)} style={win(22.3, 22.9, { rise: 12 })}>
            Explore the Grove
          </Button>
        </div>
      </Stage>
    </>
  );
}
