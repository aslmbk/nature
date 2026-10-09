/**
 * Finale, 52.8–59 s (frames 19, 20, sheet 11, motion/04): the footer panel grows from the
 * bottom centre (≈ 0.55 → 1 between 54 and 55.2 s) over the moss hills, its contents
 * resolve in a short stagger, and it stays to the end.
 *
 * Glass rule: the scale runs on a transform-only wrapper, the panel fades itself.
 */
import { cx } from "../cssVars";
import { Anchor, Stage, Words, motion as m, move, win } from "../Motion";
import { Logo } from "../SeedMark";
import { EASE } from "../timeline";
import { Button, ui as u } from "../ui";
import NewsletterField from "./NewsletterField";
import s from "./FinaleSection.module.css";

const W0 = 53.95;
const LINKS = [
  { label: "Home", href: "#hero" },
  { label: "About", href: "#about" },
  { label: "The Grove", href: "#grove" },
  { label: "Method", href: "#method" },
  { label: "Plans", href: "#plans" },
];

export default function FinaleSection() {
  return (
    <>
      <Anchor id="contact" />
      <Stage id="finale">
        <div className={cx(s.grow, m.move)} style={move(53.95, 55.2, { s: 0.55 }, { s: 1 }, EASE.linear)}>
          <footer className={cx(u.glassDark, s.panel, m.inSoft)} style={win(53.9, 54.25, { rise: 0 })}>
            <div className={s.main}>
              <div className={cx(s.logo, m.in)} style={win(53.9, 54.3, { rise: 8, blur: 8 })}>
                <Logo />
              </div>
              <h2 className={cx(u.title, s.title)}>
                <span className={u.line}>
                  <Words text="Study gently," from={W0} len={0.45} opts={{ rise: 8, blur: 12 }} exit={false} />
                </span>
                <span className={cx(u.line, u.serif, u.italic)}>
                  <Words text="grow steadily." from={W0} index0={2} len={0.45} opts={{ rise: 8, blur: 12 }} exit={false} />
                </span>
              </h2>
              <p className={cx(s.blurb, m.in)} style={win(54.4, 54.9, { rise: 8, blur: 6 })}>
                Focus sessions, honest rest and a season-by-season view of your progress, in one calm place.
              </p>
              <Button href="#contact" className={cx(s.cta, m.inSoft)} style={win(54.85, 55.4, { rise: 10 })}>
                Get in touch
              </Button>
            </div>

            <nav className={s.col} aria-label="Footer">
              <p className={cx(s.colLabel, m.in)} style={win(54.55, 55.0, { rise: 8, blur: 6 })}>
                Paths
              </p>
              <ul className={s.links}>
                {LINKS.map((l, i) => (
                  <li key={l.href} className={m.in} style={win(54.6 + i * 0.06, 55.1 + i * 0.06, { rise: 8, blur: 6 })}>
                    <a href={l.href} className={u.textLink}>
                      {l.label}
                    </a>
                  </li>
                ))}
              </ul>
            </nav>

            <div className={cx(s.col, s.colWide)}>
              <p className={cx(s.colLabel, m.in)} style={win(54.65, 55.1, { rise: 8, blur: 6 })}>
                Keep in touch
              </p>
              <div className={m.in} style={win(54.75, 55.25, { rise: 8, blur: 6 })}>
                <NewsletterField />
              </div>
            </div>
          </footer>
        </div>
      </Stage>
    </>
  );
}
