/**
 * Shared wind. One uniform object for every vegetation material in every scene:
 * reference it with `bindWindUniforms(shader.uniforms, ctx.wind)` (never copy the
 * values), and `#include` the GLSL in `WIND_GLSL` into the vertex shader.
 *
 * The motion is a pure function of the ambient clock (`uWindTime` = FrameState.timeSec),
 * so wind never accumulates and `wind=0` (uWindStrength = 0) freezes every plant.
 *
 * Model (02_IMPLEMENTATION §6, starting points from presets.example.json):
 *   h          normalised height along the plant, 0 at the root, 1 at the tip
 *   weight     h ^ uWindRootPower (2): roots never move
 *   main sway  travelling wave along the wind (crests ≈ 1.5 m apart, period uWindCycle
 *              ≈ 4.5 s, ±8 % per cluster) × slow gust field drifting with the wind;
 *              mostly a lean with the wind plus a smaller cross-wind sway
 *   flutter    faster (≈ 2 Hz) small oscillation with its own phase, mainly for leaves
 *   bend       applied across the plant's own growth axis about its root, with the
 *              distance to the root preserved (bending, not stretching)
 * Tip displacement ≈ uWindTipFraction (0.07) of the plant height for stiffness 1.
 * Neighbours share the wave and the gust (spatially smooth) and differ by a cluster
 * phase, so they agree without marching in step.
 */
import { Vector3, type IUniform } from "three";
import { WIND_DEFAULTS } from "../SceneConfig";

export interface WindUniforms {
  [name: string]: IUniform;
  /** Ambient time in seconds (FrameState.timeSec). */
  uWindTime: IUniform<number>;
  /** Normalised world direction of the main flow. */
  uWindDirection: IUniform<Vector3>;
  /** Global strength multiplier: 0 = no wind (wind=0), reduced under prefers-reduced-motion. */
  uWindStrength: IUniform<number>;
  /** Spatial frequency of the gust field (1/m). */
  uWindGustScale: IUniform<number>;
  /** Drift speed of the gust field (m/s). */
  uWindGustSpeed: IUniform<number>;
  /** Tip displacement as a fraction of plant height (presets: 0.07). */
  uWindTipFraction: IUniform<number>;
  /** Main sway period in seconds (presets: 4.5). */
  uWindCycle: IUniform<number>;
  /** Flutter amplitude relative to the main sway (presets: 0.15). */
  uWindFlutter: IUniform<number>;
  /** Root weight exponent h^p (presets: 2). */
  uWindRootPower: IUniform<number>;
}

export class WindField {
  readonly uniforms: WindUniforms = {
    uWindTime: { value: 0 },
    uWindDirection: { value: new Vector3(1, 0, 0.35).normalize() },
    uWindStrength: { value: 1 },
    uWindGustScale: { value: 0.35 },
    uWindGustSpeed: { value: 0.6 },
    uWindTipFraction: { value: WIND_DEFAULTS.tipDisplacementFraction },
    uWindCycle: { value: WIND_DEFAULTS.mainCycleSeconds },
    uWindFlutter: { value: WIND_DEFAULTS.flutterRelativeAmplitude },
    uWindRootPower: { value: WIND_DEFAULTS.rootWeightPower },
  };

  /** Base strength before the enable / reduced-motion gates. */
  strength = 1;

  update(timeSec: number, enabled: boolean, reducedMotion: boolean): void {
    this.uniforms.uWindTime.value = timeSec;
    this.uniforms.uWindStrength.value = enabled ? this.strength * (reducedMotion ? WIND_DEFAULTS.reducedMotionScale : 1) : 0;
  }

  setDirection(x: number, y: number, z: number): void {
    this.uniforms.uWindDirection.value.set(x, y, z).normalize();
  }
}

/** Put references to the shared wind uniforms into a shader's uniform table (never copies). */
export function bindWindUniforms(target: Record<string, IUniform>, wind: WindUniforms): void {
  for (const key of Object.keys(wind)) target[key] = wind[key];
}

/**
 * Largest tip displacement the wind can produce, as a fraction of plant height, for a
 * plant of stiffness `minStiffness` (bounding volumes must grow by this × max height).
 */
export function maxWindFraction(wind: WindUniforms, minStiffness = 0.5, flutter = 0.3): number {
  const tip = wind.uWindTipFraction.value * Math.max(1, wind.uWindStrength.value);
  return (tip * 1.4) / Math.max(0.2, minStiffness) + tip * flutter;
}

/**
 * Vertex-shader GLSL: wind uniforms + helpers. Include once per program.
 *
 *   vec3  silvaWindSway(anchor, phase, stiffness)   world tip offset per metre of plant height
 *   float silvaFlutter(phase, hz)                   −1..1, fast and small
 *   vec3  silvaBend(rel, h, up, tipOffsetPerMetre, height)
 *         bends a point (`rel` = point − root) of a plant about its root
 */
export const WIND_GLSL = /* glsl */ `
uniform float uWindTime;
uniform vec3 uWindDirection;
uniform float uWindStrength;
uniform float uWindGustScale;
uniform float uWindGustSpeed;
uniform float uWindTipFraction;
uniform float uWindCycle;
uniform float uWindFlutter;
uniform float uWindRootPower;

float silvaHash13(vec3 p) {
  p = fract(p * 0.1031);
  p += dot(p, p.zyx + 31.32);
  return fract((p.x + p.y) * p.z);
}

float silvaNoise3(vec3 p) {
  vec3 i = floor(p);
  vec3 f = fract(p);
  vec3 u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(mix(silvaHash13(i), silvaHash13(i + vec3(1.0, 0.0, 0.0)), u.x),
        mix(silvaHash13(i + vec3(0.0, 1.0, 0.0)), silvaHash13(i + vec3(1.0, 1.0, 0.0)), u.x), u.y),
    mix(mix(silvaHash13(i + vec3(0.0, 0.0, 1.0)), silvaHash13(i + vec3(1.0, 0.0, 1.0)), u.x),
        mix(silvaHash13(i + vec3(0.0, 1.0, 1.0)), silvaHash13(i + vec3(1.0, 1.0, 1.0)), u.x), u.y),
    u.z);
}

/** Slow gust field 0..1 drifting along the wind. */
float silvaGust(vec3 anchor) {
  vec3 q = anchor * uWindGustScale - uWindDirection * (uWindTime * uWindGustSpeed * uWindGustScale);
  float n = silvaNoise3(q) * 0.65 + silvaNoise3(q * 2.13 + vec3(7.1, 3.7, 1.3)) * 0.35;
  return smoothstep(0.12, 0.88, n);
}

vec3 silvaWindSway(vec3 anchor, float phase, float stiffness) {
  float w = 6.2831853 / max(uWindCycle, 0.1);
  float travel = dot(anchor, uWindDirection) * 4.2;
  float a = sin(uWindTime * w * (0.92 + 0.16 * phase) - travel + phase * 6.2831853);
  float b = sin(uWindTime * w * 0.61 + phase * 11.3 - travel * 0.7 + 1.7);
  float gust = 0.4 + 0.6 * silvaGust(anchor);
  vec3 across = normalize(cross(uWindDirection, vec3(0.0, 1.0, 0.0)) + vec3(1e-4, 0.0, 0.0));
  vec3 v = uWindDirection * ((0.5 + 0.5 * a) * gust) + across * (0.38 * b * gust);
  return v * (uWindTipFraction * uWindStrength / max(stiffness, 0.2));
}

float silvaFlutter(float phase, float hz) {
  float t = uWindTime * 6.2831853;
  return 0.62 * sin(t * hz * (0.85 + 0.3 * phase) + phase * 40.0)
       + 0.38 * sin(t * hz * 1.73 + phase * 17.0 + 0.9);
}

vec3 silvaBend(vec3 rel, float h, vec3 up, vec3 tipOffsetPerMetre, float height) {
  vec3 d = tipOffsetPerMetre * height;
  d -= up * dot(d, up);
  vec3 moved = rel + d * pow(max(h, 0.0), uWindRootPower);
  float len = length(rel);
  return len > 1e-6 ? normalize(moved) * len : rel;
}
`;
