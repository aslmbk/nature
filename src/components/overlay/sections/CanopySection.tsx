/**
 * Canopy, 43.3–48.5 s (frames 15, 16, sheet 10): the canopy grows on black, then a
 * centred three-line statement resolves word by word in place (blurred words at
 * 44.3 s), a two-line paragraph and two buttons (solid + glass). Drifts up and fades
 * as the camera moves closer (48–49.5 s).
 *
 * Legibility over the lit leaves: a soft dark halo behind the column (wide screens; on
 * phones the copy sits on black above and below the canopy), a stronger shadow on the
 * small texts and a darker tint in the glass button.
 */
import { cx } from "../cssVars";
import { Stage, Words, motion as m, win } from "../Motion";
import { Button, ui as u } from "../ui";
import sh from "./shared.module.css";
import s from "./CanopySection.module.css";

const W0 = 44.15;
const STEP = 0.1;
const LEN = 0.55;
const OPTS = { rise: 6, blur: 14 };

export default function CanopySection() {
  return (
    <Stage id="canopy">
      <div className={cx(sh.column, s.column)}>
        <span className={cx(sh.halo, s.halo, m.inoutSoft)} data-exit="span" style={win(44.15, 45.4, { rise: 0 })} aria-hidden="true" />
        <p className={cx(u.eyebrow, s.eyebrow, m.inout)} style={win(44.15, 44.65, { rise: 0, blur: 10 })}>
          Silva. Quiet by default.
        </p>
        <h2 className={cx(u.title, sh.titleGap, s.title)}>
          <span className={u.line}>
            <Words text="Every mind keeps" from={W0} step={STEP} len={LEN} opts={OPTS} />
          </span>
          <span className={u.line}>
            <Words text="its own seasons." from={W0} index0={3} step={STEP} len={LEN} opts={OPTS} />{" "}
            <span className={cx(u.serif, u.italic)}>
              <Words text="Let" from={W0} index0={6} step={STEP} len={LEN} opts={OPTS} />
            </span>
          </span>
          <span className={cx(u.line, u.serif, u.italic, s.big)}>
            <Words text="your study follow yours." from={W0} index0={7} step={STEP} len={LEN} opts={OPTS} />
          </span>
        </h2>
        <p className={cx(u.lead, sh.leadGap, sh.scrim, s.lead, m.inout)} style={win(45.35, 45.95, { rise: 10, blur: 6 })}>
          Start small, keep a steady pace and let the record of your work grow on its own. Silva stays out of the way until
          you need it.
        </p>
        <div className={cx(sh.row, s.buttons)}>
          <Button href="#plans" className={m.inoutSoft} style={win(45.55, 46.15, { rise: 12 })}>
            Start with Silva
          </Button>
          <Button href="#method" variant="glass" className={cx(s.glass, m.inoutSoft)} style={win(45.65, 46.25, { rise: 12 })}>
            See how it works
          </Button>
        </div>
      </div>
    </Stage>
  );
}
