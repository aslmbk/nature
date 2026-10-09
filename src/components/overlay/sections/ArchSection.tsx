/**
 * Arch, 10–15.5 s (frame 04): centred eyebrow, headline (grotesque + italic serif word,
 * italic serif second line), grey paragraph, CTA — framed by the trunk arch.
 */
import { cx } from "../cssVars";
import { Anchor, Stage, Words, motion as m, win } from "../Motion";
import { Button, ui as u } from "../ui";
import sh from "./shared.module.css";
import s from "./ArchSection.module.css";

const W0 = 10.35;

export default function ArchSection() {
  return (
    <>
      <Anchor id="about" />
      <Stage id="arch">
        <div className={cx(sh.column, s.column)}>
          <p className={cx(u.eyebrow, m.inout)} style={win(10.25, 10.85, { rise: 10 })}>
            About Silva
          </p>
          <h2 className={cx(u.title, sh.titleGap)}>
            <span className={u.line}>
              <Words text="Steady hours" from={W0} />{" "}
              <span className={cx(u.serif, u.italic)}>
                <Words text="grow" from={W0} index0={2} />
              </span>
            </span>
            <span className={cx(u.line, u.serif, u.italic)}>
              <Words text="deep roots." from={W0} index0={3} />
            </span>
          </h2>
          <p className={cx(u.lead, sh.lead, sh.leadGap, s.lead, m.inout)} style={win(10.75, 11.4, { rise: 14, blur: 6 })}>
            Silva is built around a simple rhythm: a short block of focus, a real pause and a look back at what moved. Week by
            week, those small cycles become a study habit that feels like your own.
          </p>
          <Button href="#method" className={cx(sh.ctaGap, m.inoutSoft)} style={win(10.95, 11.55, { rise: 12 })}>
            Find your rhythm
          </Button>
        </div>
      </Stage>
    </>
  );
}
