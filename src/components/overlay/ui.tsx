/** Small shared overlay pieces: pill buttons with the arrow chip, icons. Server-safe. */
import type { CSSProperties, ReactNode } from "react";
import { cx } from "./cssVars";
import { SEED_PATHS, SEED_VIEWBOX } from "./SeedMark";
import u from "./ui.module.css";

export { u as ui };

export function ArrowIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 12 12" fill="none" aria-hidden="true" focusable="false">
      <path d="M2 6h7.4M6.4 2.9 9.5 6l-3.1 3.1" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

interface ButtonProps {
  href: string;
  children: ReactNode;
  variant?: "light" | "dark" | "glass";
  className?: string;
  style?: CSSProperties;
}

/** Anchor styled as the white pill with a dark arrow chip (frames 02, 04, 08, 16, 17, 20). */
export function Button({ href, children, variant = "light", className, style }: ButtonProps) {
  return (
    <a
      href={href}
      className={cx(u.button, variant === "dark" && u.buttonDark, variant === "glass" && u.buttonGlass, className)}
      style={style}
    >
      <span>{children}</span>
      <span className={u.chip}>
        <ArrowIcon />
      </span>
    </a>
  );
}

/** Four-point sparkle (hero "scroll to explore"). */
export function Sparkle({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      <path d="M12 1.5c.5 5.6 2.9 8.6 10.5 10.5-7.6 1.9-10 4.9-10.5 10.5-.5-5.6-2.9-8.6-10.5-10.5C9.1 10.1 11.5 7.1 12 1.5Z" fill="currentColor" />
    </svg>
  );
}

/** Tiny seed used as list bullet (a scaled-down Silva emblem silhouette). */
export function SeedBullet({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox={SEED_VIEWBOX} aria-hidden="true" focusable="false">
      <g fill="currentColor" stroke="currentColor" strokeWidth={6} strokeLinejoin="round">
        <path d={SEED_PATHS[0]} />
        <path d={SEED_PATHS[1]} />
      </g>
    </svg>
  );
}
