/**
 * Dust motes of the finale: fine specks drifting slowly in the cold air around the stone
 * (frames 19 / 20: a few dozen tiny white points, brightest near the stone where the key
 * light crosses the air; "barely visible dust is fine" — 01_VISUAL_STORYBOARD S10).
 *
 * One `THREE.Points` set, everything computed in the vertex shader from the ambient
 * clock: a mote starts at its seeded position, drifts with a slow common current (its
 * own speed) plus a small meander, and is wrapped into a box around the plateau; it
 * fades out near the box faces (so the wrap never pops) and very close to the lens.
 * Pure function of `uTime` (= FrameState.timeSec), nothing accumulates.
 *
 * Sprite model (like the oracle particles, own copy): a mote has a world size and a peak
 * colour at its in-focus size; tiny motes are clamped to a minimum size in pixels; motes
 * off the DOF focus distance grow a defocus disc — near the lens much more than far away —
 * keeping the energy of their in-focus sprite, so the few close to the camera become dim,
 * soft bokeh discs. Additive colour, alpha untouched (the DOF composite reads the scene
 * alpha as coverage), depth-tested against the moss and the stone, no depth writes.
 */
import {
  BufferAttribute,
  BufferGeometry,
  Color,
  CustomBlending,
  OneFactor,
  Points,
  ShaderMaterial,
  Vector2,
  Vector3,
  ZeroFactor,
  type IUniform,
} from "three";
import type { Rng } from "../../core/rng";

export interface DustUniforms {
  [name: string]: IUniform;
  uTime: IUniform<number>;
  /** Drawing-buffer px per metre at distance 1. */
  uScale: IUniform<number>;
  uMinPx: IUniform<number>;
  uMaxPx: IUniform<number>;
  /** DOF focus distance (m). */
  uFocus: IUniform<number>;
  /** Defocus disc (px) of a mote at half the focus distance / of far motes. */
  uCocNear: IUniform<number>;
  uCocFar: IUniform<number>;
  uBoxMin: IUniform<Vector3>;
  uBoxSize: IUniform<Vector3>;
  /** Common current (m/s). */
  uDrift: IUniform<Vector3>;
  uColor: IUniform<Color>;
  uOpacity: IUniform<number>;
  /** Centre / radius (m) of the lit air (motes there are brighter). */
  uLightCentre: IUniform<Vector3>;
  uLightRadius: IUniform<number>;
  /** Fade-in distance from the lens (m): [hidden, full]. */
  uNearFade: IUniform<Vector2>;
}

export interface Dust {
  points: Points<BufferGeometry, ShaderMaterial>;
  uniforms: DustUniforms;
  count: number;
}

export interface DustOptions {
  count: number;
  box: { min: Vector3; max: Vector3 };
  lightCentre: Vector3;
  lightRadius: number;
  /** Colour of a speck in focus (sRGB hex) … */
  color: string;
  /** … times this linear intensity. */
  intensity: number;
  /** Common current (m/s); every mote drifts with 0.45–1.55 × this. */
  drift: Vector3;
}

const VERTEX = /* glsl */ `
attribute vec4 aSeed;   // x: size (m), y: phase 0–1, z: speed 0–1, w: brightness
uniform float uTime;
uniform float uScale;
uniform float uMinPx;
uniform float uMaxPx;
uniform float uFocus;
uniform float uCocNear;
uniform float uCocFar;
uniform vec3 uBoxMin;
uniform vec3 uBoxSize;
uniform vec3 uDrift;
uniform vec3 uColor;
uniform float uOpacity;
uniform vec3 uLightCentre;
uniform float uLightRadius;
uniform vec2 uNearFade;
varying vec3 vColor;
varying float vSharp;

void main() {
  float ph = aSeed.y * 6.2831853;
  float sp = 0.45 + 1.1 * aSeed.z;
  vec3 p = position + uDrift * (sp * uTime);
  // slow meander (a few cm), every mote on its own phase
  p += vec3(
    0.05 * sin(uTime * (0.09 + 0.08 * aSeed.z) + ph),
    0.035 * sin(uTime * (0.07 + 0.06 * aSeed.y) + ph * 1.7),
    0.05 * cos(uTime * (0.08 + 0.07 * aSeed.w) + ph * 0.6)
  );
  // wrap into the box, fade out near its faces
  vec3 q = fract((p - uBoxMin) / uBoxSize);
  p = uBoxMin + q * uBoxSize;
  vec3 e = min(q, 1.0 - q);
  float edge = smoothstep(0.0, 0.07, min(min(e.x, e.y), e.z));

  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  gl_Position = projectionMatrix * mv;
  float z = max(0.05, -mv.z);
  float nominal = aSeed.x * uScale / z;
  float s0 = max(nominal, uMinPx);
  float dz = z - uFocus;
  float coc = (dz < 0.0 ? uCocNear : uCocFar) * abs(dz) / z;
  float S = clamp(sqrt(s0 * s0 + coc * coc), 1.0, uMaxPx);
  float sharp = clamp((s0 * s0) / (S * S), 0.0, 1.0);
  // energy of the in-focus sprite spread over the defocus disc
  float area = mix(0.6, 0.112, sharp) * S * S;
  float energy = 0.112 * s0 * s0;
  float r = length(p - uLightCentre) / max(uLightRadius, 1e-3);
  float lit = 0.25 + 0.75 * exp(-r * r);
  float tw = 0.75 + 0.25 * sin(uTime * (0.5 + 0.8 * aSeed.z) + ph * 3.0);
  float near = smoothstep(uNearFade.x, uNearFade.y, z);
  gl_PointSize = S;
  vColor = uColor * (aSeed.w * lit * tw * edge * near * uOpacity * energy / max(area, 1e-4));
  vSharp = sharp;
}
`;

const FRAGMENT = /* glsl */ `
varying vec3 vColor;
varying float vSharp;
void main() {
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(c, c);
  if (r2 > 1.0) discard;
  float core = exp(-r2 * 6.0);
  float disc = (1.0 - smoothstep(0.6, 1.0, r2)) * (0.75 + 0.35 * r2);
  gl_FragColor = vec4(vColor * mix(disc, core, vSharp), 1.0);
  // no-ops in the linear HDR pass; applied only when rendering straight to the canvas (post=0)
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

/**
 * Log-normal-ish brightness: most motes faint, a few brighter specks. The tail is clamped
 * (at +1.5σ, bright flakes ≤ 3×): an unclamped draw at +3σ made one in-focus mote a small
 * lamp with a bloom halo (≈ 40 linear at its core).
 */
function moteBrightness(rng: Rng): number {
  const b = 0.07 * Math.exp(0.8 * Math.max(-2.5, Math.min(1.5, rng.normal())));
  return rng.chance(0.07) ? b * rng.range(1.8, 3) : b;
}

export function buildDust(rng: Rng, opts: DustOptions): Dust {
  const n = Math.max(1, Math.round(opts.count));
  const pos = new Float32Array(n * 3);
  const seed = new Float32Array(n * 4);
  const { min, max } = opts.box;
  for (let i = 0; i < n; i++) {
    pos[i * 3] = rng.range(min.x, max.x);
    pos[i * 3 + 1] = rng.range(min.y, max.y);
    pos[i * 3 + 2] = rng.range(min.z, max.z);
    // 0.5–1.5 mm specks, a few larger flakes
    const size = rng.chance(0.08) ? rng.range(0.0018, 0.003) : rng.range(0.0005, 0.0014);
    seed[i * 4] = size;
    seed[i * 4 + 1] = rng.next();
    seed[i * 4 + 2] = rng.next();
    seed[i * 4 + 3] = moteBrightness(rng);
  }
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new BufferAttribute(pos, 3));
  geometry.setAttribute("aSeed", new BufferAttribute(seed, 4));
  geometry.name = "finale_dust";
  const size = new Vector3().subVectors(max, min);
  const uniforms: DustUniforms = {
    uTime: { value: 0 },
    uScale: { value: 1600 },
    uMinPx: { value: 1.4 },
    uMaxPx: { value: 64 },
    uFocus: { value: 4 },
    uCocNear: { value: 14 },
    uCocFar: { value: 1.5 },
    uBoxMin: { value: min.clone() },
    uBoxSize: { value: size },
    uDrift: { value: opts.drift.clone() },
    uColor: { value: new Color(opts.color).multiplyScalar(opts.intensity) },
    uOpacity: { value: 1 },
    uLightCentre: { value: opts.lightCentre.clone() },
    uLightRadius: { value: opts.lightRadius },
    uNearFade: { value: new Vector2(0.25, 0.6) },
  };
  const material = new ShaderMaterial({
    name: "FinaleDust",
    uniforms,
    vertexShader: VERTEX,
    fragmentShader: FRAGMENT,
    transparent: true,
    depthWrite: false,
    depthTest: true,
    // additive colour, alpha left as it is (coverage for the DOF composite)
    blending: CustomBlending,
    blendSrc: OneFactor,
    blendDst: OneFactor,
    blendSrcAlpha: ZeroFactor,
    blendDstAlpha: OneFactor,
  });
  const points = new Points(geometry, material);
  points.name = "finale_dust";
  points.frustumCulled = false;
  // after the plants (transparent list), before nothing else
  points.renderOrder = 10;
  return { points, uniforms, count: n };
}
