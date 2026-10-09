/**
 * The Silva emblem, "the seed": the two pieces of an upright almond split by a slanted
 * channel. Paths and viewBox are copied verbatim from `assets-src/emblem/seed.svg`
 * (the single source); round stroke joins give the 1-unit corner rounding.
 * Server-safe (no hooks), colour follows `currentColor`.
 */
import { cssVars, cx } from "./cssVars";
import styles from "./SeedMark.module.css";

/** Both pieces, in the coordinates of `assets-src/emblem/seed.svg`. */
export const SEED_PATHS = [
  "M12.87 -37 A62.5 62.5 0 0 0 0 -50 A62.5 62.5 0 0 0 -11.68 38.56 Z",
  "M17.06 -30.49 A62.5 62.5 0 0 1 0 50 A62.5 62.5 0 0 1 -7.08 43.81 Z",
] as const;
export const SEED_VIEWBOX = "-30 -55 60 110";

interface SeedMarkProps {
  className?: string;
  /** Accessible name; omitted = decorative (aria-hidden). */
  title?: string;
}

export default function SeedMark({ className, title }: SeedMarkProps) {
  return (
    <svg
      className={cx(styles.mark, className)}
      viewBox={SEED_VIEWBOX}
      xmlns="http://www.w3.org/2000/svg"
      role={title ? "img" : undefined}
      aria-label={title}
      aria-hidden={title ? undefined : true}
      focusable="false"
    >
      <g fill="currentColor" stroke="currentColor" strokeWidth={2} strokeLinejoin="round">
        <path d={SEED_PATHS[0]} />
        <path d={SEED_PATHS[1]} />
      </g>
    </svg>
  );
}

interface LogoProps {
  className?: string;
  /** Extra class for the seed (e.g. the intro animation). */
  markClassName?: string;
  /** Extra class for each letter of the wordmark (per-letter intro, `--k` = index). */
  letterClassName?: string;
}

/** Seed + wordmark "Silva". */
export function Logo({ className, markClassName, letterClassName }: LogoProps) {
  return (
    <span className={cx(styles.logo, className)}>
      <SeedMark className={cx(styles.logoMark, markClassName)} />
      {letterClassName ? (
        <span className={styles.wordmark}>
          <span className={styles.srOnly}>Silva</span>
          <span aria-hidden="true">
            {"Silva".split("").map((ch, i) => (
              <span key={i} className={letterClassName} style={cssVars({ "--k": i })}>
                {ch}
              </span>
            ))}
          </span>
        </span>
      ) : (
        <span className={styles.wordmark}>Silva</span>
      )}
    </span>
  );
}
