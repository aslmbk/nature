/**
 * Canopy close-up, 48.5–52.8 s (frame 17, sheet 11): two dark glass plan cards over the
 * close canopy. While the leaves part for the finale (52.4–53.0 s) the pair recedes about
 * its common centre (each card scales towards the gap between them) and fades.
 * Plans and prices belong to the fictional Silva product.
 */
import { cssVars, cx } from "../cssVars";
import { Anchor, Stage, motion as m, win } from "../Motion";
import { Button, SeedBullet, ui as u } from "../ui";
import s from "./CanopyCloseSection.module.css";

const PLANS = [
  {
    key: "seedling",
    name: "Seedling",
    price: "$6.50",
    features: ["Focus sessions and breaks", "Weekly rhythm view", "Gentle reminders"],
    note: "Pause or cancel at any time.",
    cta: "Start with Seedling",
  },
  {
    key: "canopy",
    name: "Canopy",
    price: "$11.00",
    features: ["Everything in Seedling", "The Grove and season dashboard", "Review cycle from your notes"],
    note: "For a full year of steady study.",
    cta: "Start with Canopy",
  },
];

export default function CanopyCloseSection() {
  return (
    <>
      <Anchor id="plans" />
      <Stage id="canopyClose">
        <h2 className={u.srOnly}>Plans</h2>
        <ul className={s.plans}>
          {PLANS.map((p, i) => (
            <li
              key={p.key}
              className={cx(u.glassDark, s.plan, i === 0 ? s.first : s.second, m.inoutRecede)}
              style={win(49.0 + i * 0.15, 49.85 + i * 0.15, { rise: 50 }, cssVars({ "--rs": 0.5 }))}
            >
              <h3 className={s.name}>{p.name}</h3>
              <p className={s.priceRow}>
                <span className={s.price}>{p.price}</span>
                <span className={s.per}>/ month</span>
              </p>
              <ul className={s.features}>
                {p.features.map((f) => (
                  <li key={f}>
                    <SeedBullet className={s.bullet} />
                    {f}
                  </li>
                ))}
              </ul>
              <p className={s.note}>{p.note}</p>
              <Button href="#contact" className={s.cta}>
                {p.cta}
              </Button>
            </li>
          ))}
        </ul>
      </Stage>
    </>
  );
}
