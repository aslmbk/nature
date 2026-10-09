import type { CSSProperties } from "react";

/** Inline CSS custom properties (`{ "--a": 10.2 }`); numbers are written without units. */
export function cssVars(vars: Record<`--${string}`, string | number | undefined>, extra?: CSSProperties): CSSProperties {
  const out: Record<string, string | number> = {};
  for (const [k, v] of Object.entries(vars)) {
    if (v === undefined) continue;
    out[k] = typeof v === "number" ? round(v) : v;
  }
  return { ...(extra ?? {}), ...out } as CSSProperties;
}

/** Keeps inline styles short and stable between server and client renders. */
export function round(v: number, digits = 4): number {
  const f = 10 ** digits;
  return Math.round(v * f) / f;
}

export function cx(...names: (string | false | null | undefined)[]): string {
  return names.filter(Boolean).join(" ");
}
