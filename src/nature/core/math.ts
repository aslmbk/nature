import type { EaseName } from "../types";

export const clamp = (x: number, min: number, max: number): number => (x < min ? min : x > max ? max : x);
export const saturate = (x: number): number => (x < 0 ? 0 : x > 1 ? 1 : x);
export const lerp = (a: number, b: number, k: number): number => a + (b - a) * k;
/** Position of x between a and b (unclamped). */
export const invLerp = (a: number, b: number, x: number): number => (b === a ? 0 : (x - a) / (b - a));
export const fract = (x: number): number => x - Math.floor(x);
export const degToRad = (deg: number): number => (deg * Math.PI) / 180;
export const radToDeg = (rad: number): number => (rad * 180) / Math.PI;

/** Map x from [inMin, inMax] to [outMin, outMax]; clamped by default. */
export function remap(x: number, inMin: number, inMax: number, outMin: number, outMax: number, clamped = true): number {
  let k = invLerp(inMin, inMax, x);
  if (clamped) k = saturate(k);
  return outMin + (outMax - outMin) * k;
}

export function smoothstep(edge0: number, edge1: number, x: number): number {
  const k = saturate(invLerp(edge0, edge1, x));
  return k * k * (3 - 2 * k);
}

export function smootherstep(edge0: number, edge1: number, x: number): number {
  const k = saturate(invLerp(edge0, edge1, x));
  return k * k * k * (k * (k * 6 - 15) + 10);
}

/**
 * Frame-rate independent smoothing factor: 1 - exp(-lambda * dt).
 * Use with lerp(current, target, dampFactor(lambda, dt)).
 */
export function dampFactor(lambda: number, dt: number): number {
  return 1 - Math.exp(-lambda * dt);
}

/** Delta-time damping of a scalar towards target. */
export function damp(current: number, target: number, lambda: number, dt: number): number {
  return lerp(current, target, dampFactor(lambda, dt));
}

/** lambda for a given response time (time constant, seconds). */
export function lambdaForResponse(responseSec: number): number {
  return 1 / Math.max(1e-4, responseSec);
}

const EASES: Record<EaseName, (x: number) => number> = {
  linear: (x) => x,
  smooth: (x) => x * x * (3 - 2 * x),
  smoother: (x) => x * x * x * (x * (x * 6 - 15) + 10),
  inOutSine: (x) => 0.5 - 0.5 * Math.cos(Math.PI * x),
  inOutCubic: (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2),
  inCubic: (x) => x * x * x,
  outCubic: (x) => 1 - Math.pow(1 - x, 3),
};

/** Apply a named ease to x (clamped to 0–1 first). */
export function ease(name: EaseName, x: number): number {
  return EASES[name](saturate(x));
}

/** Linear 0–1 position of t inside [start, end] (clamped). */
export function windowProgress(t: number, start: number, end: number): number {
  return saturate(invLerp(start, end, t));
}

/** Small deterministic hash of an integer → [0,1). */
export function hash01(n: number): number {
  let x = Math.imul(n ^ 0x9e3779b9, 0x85ebca6b);
  x ^= x >>> 13;
  x = Math.imul(x, 0xc2b2ae35);
  x ^= x >>> 16;
  return (x >>> 0) / 4294967296;
}
