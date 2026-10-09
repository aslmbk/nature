"use client";

/**
 * Fixed top bar (frames 02–20): logo left, centred pill with four section links, "Log in"
 * and the white "Get started" pill right; under 900 px: logo + menu button.
 *
 * The active link follows the scroll position without a scroll handler: an
 * IntersectionObserver watches the ScrollTrack sections through a thin strip at the top
 * of the viewport (the section crossing the viewport top is the current episode, the
 * same rule the engine uses) and maps the section's start time to a nav item.
 * The intro (logo, pill opening, links) is a one-off CSS animation on load.
 */
import { useEffect, useState } from "react";
import { cssVars, cx } from "./cssVars";
import { Logo } from "./SeedMark";
import { NAV_ITEMS, navIndexForTime } from "./timeline";
import styles from "./TopBar.module.css";
import u from "./ui.module.css";

export default function TopBar() {
  const [active, setActive] = useState(0);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (typeof IntersectionObserver === "undefined") return;
    const sections = Array.from(document.querySelectorAll<HTMLElement>("section[data-kind][data-start]"));
    if (sections.length === 0) return;
    const inside = new Set<HTMLElement>();
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          const el = e.target as HTMLElement;
          if (e.isIntersecting) inside.add(el);
          else inside.delete(el);
        }
        let start = -1;
        for (const el of inside) start = Math.max(start, Number(el.dataset.start));
        if (start >= 0) setActive(navIndexForTime(start));
      },
      // a 1 %-tall strip at the top edge of the viewport
      { rootMargin: "0px 0px -99% 0px", threshold: 0 },
    );
    for (const s of sections) io.observe(s);
    return () => io.disconnect();
  }, []);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    const mq = window.matchMedia("(min-width: 900px)");
    const onWide = () => {
      if (mq.matches) setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    mq.addEventListener("change", onWide);
    return () => {
      window.removeEventListener("keydown", onKey);
      mq.removeEventListener("change", onWide);
    };
  }, [open]);

  const close = () => setOpen(false);

  return (
    <header className={styles.bar}>
      <a href="#hero" className={styles.brand} aria-label="Silva — back to the start">
        <Logo markClassName={styles.brandMark} letterClassName={styles.letter} />
      </a>

      <nav className={styles.nav} aria-label="Sections">
        <span className={styles.pillBg} aria-hidden="true" />
        <ul className={styles.links}>
          {NAV_ITEMS.map((item, i) => (
            <li key={item.href}>
              <a
                href={item.href}
                className={styles.link}
                aria-current={active === i ? "true" : undefined}
                data-active={active === i ? "" : undefined}
                style={cssVars({ "--k": i })}
              >
                {item.label}
              </a>
            </li>
          ))}
        </ul>
      </nav>

      <div className={styles.actions}>
        <a href="#contact" className={styles.login}>
          Log in
        </a>
        <a href="#plans" className={cx(u.button, styles.cta)}>
          Get started
        </a>
      </div>

      <button
        type="button"
        className={styles.menuButton}
        aria-expanded={open}
        aria-controls="silva-menu"
        onClick={() => setOpen((v) => !v)}
      >
        <span>{open ? "Close" : "Menu"}</span>
        <span className={styles.menuIcon} data-open={open ? "" : undefined} aria-hidden="true">
          <i />
          <i />
        </span>
      </button>

      <div id="silva-menu" className={styles.menu} hidden={!open}>
        <ul className={styles.menuLinks}>
          {NAV_ITEMS.map((item, i) => (
            <li key={item.href}>
              <a href={item.href} onClick={close} aria-current={active === i ? "true" : undefined} data-active={active === i ? "" : undefined}>
                {item.label}
              </a>
            </li>
          ))}
        </ul>
        <div className={styles.menuActions}>
          <a href="#contact" className={styles.login} onClick={close}>
            Log in
          </a>
          <a href="#plans" className={u.button} onClick={close}>
            Get started
          </a>
        </div>
      </div>
    </header>
  );
}
