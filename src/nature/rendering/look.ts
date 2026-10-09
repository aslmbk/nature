/**
 * Runtime look helpers: LookSpec (sRGB hex, authored in SceneConfig) → LookParams
 * (linear THREE.Color), copies, patches and interpolation. Colours are mixed in
 * linear space so crossings between episodes do not darken in the middle.
 */
import { Color } from "three";
import { EPISODE_LOOKS } from "../SceneConfig";
import type { EpisodeId, LookParams, LookPatch, LookSpec } from "../types";

type Node = Record<string, unknown>;

function isColor(v: unknown): v is Color {
  return typeof v === "object" && v !== null && (v as { isColor?: boolean }).isColor === true;
}

/** Structural clone of a spec where every string leaf becomes a Color. */
function build(spec: Node): Node {
  const out: Node = {};
  for (const [key, value] of Object.entries(spec)) {
    if (typeof value === "string") out[key] = new Color(value);
    else if (typeof value === "object" && value !== null) out[key] = build(value as Node);
    else out[key] = value;
  }
  return out;
}

function assignSpec(target: Node, spec: Node): void {
  for (const [key, value] of Object.entries(spec)) {
    if (value === undefined) continue;
    const cur = target[key];
    if (typeof value === "string" && isColor(cur)) cur.set(value);
    else if (typeof value === "object" && value !== null && typeof cur === "object" && cur !== null) assignSpec(cur as Node, value as Node);
    else if (typeof value === "number") target[key] = value;
  }
}

function copyInto(target: Node, src: Node): void {
  for (const [key, value] of Object.entries(src)) {
    const cur = target[key];
    if (isColor(value) && isColor(cur)) cur.copy(value);
    else if (typeof value === "object" && value !== null && typeof cur === "object" && cur !== null) copyInto(cur as Node, value as Node);
    else target[key] = value;
  }
}

function lerpInto(target: Node, a: Node, b: Node, k: number): void {
  for (const key of Object.keys(a)) {
    const va = a[key];
    const vb = b[key];
    const cur = target[key];
    if (isColor(va) && isColor(vb) && isColor(cur)) cur.copy(va).lerp(vb, k);
    else if (typeof va === "number" && typeof vb === "number") target[key] = va + (vb - va) * k;
    else if (typeof va === "object" && va !== null && typeof cur === "object" && cur !== null) lerpInto(cur as Node, va as Node, vb as Node, k);
  }
}

export function resolveLook(spec: LookSpec): LookParams {
  return build(spec as unknown as Node) as unknown as LookParams;
}

export function cloneLook(look: LookParams): LookParams {
  const out = resolveLook(EPISODE_LOOKS.hero);
  copyInto(out as unknown as Node, look as unknown as Node);
  return out;
}

export function copyLook(src: LookParams, out: LookParams): LookParams {
  copyInto(out as unknown as Node, src as unknown as Node);
  return out;
}

/** out = mix(a, b, k). `out` may alias `a` or `b`. */
export function lerpLook(a: LookParams, b: LookParams, k: number, out: LookParams): LookParams {
  if (out === b) {
    lerpInto(out as unknown as Node, b as unknown as Node, a as unknown as Node, 1 - k);
  } else {
    lerpInto(out as unknown as Node, a as unknown as Node, b as unknown as Node, k);
  }
  return out;
}

/** Apply a partial spec (hex colours) onto a runtime look in place. */
export function patchLook(look: LookParams, p: LookPatch): LookParams {
  assignSpec(look as unknown as Node, p as unknown as Node);
  return look;
}

const EPISODE_LOOK_CACHE = new Map<EpisodeId, LookParams>();

/** Runtime copy of the SceneConfig look of an episode (shared, read-only — clone before mutating). */
export function episodeLook(id: EpisodeId): LookParams {
  let look = EPISODE_LOOK_CACHE.get(id);
  if (!look) {
    look = resolveLook(EPISODE_LOOKS[id]);
    EPISODE_LOOK_CACHE.set(id, look);
  }
  return look;
}
