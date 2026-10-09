/**
 * Particle layers of the oracle scene set, all `THREE.Points` with trajectories
 * computed in the vertex shader from uniforms (story time / ambient clock):
 *
 *  - orb swarm (orb-local, units of the orb radius): interior clumps with dark gaps,
 *    a patchy inner membrane and a thin fountain of sparks out of the top. Each
 *    particle also has a start position on one of two crumpled sheets of a loose swirl
 *    (frame 07) and converges from there with its own delay (assembly).
 *  - streams (orb-anchor-local, orb radius units): a wide, flat, slowly turning disc of
 *    particles around the orb that rises and loosens over the episode: seen from above
 *    at first (a particle floor sweeping in from the lower left / right), edge-on at the
 *    height of the orb, then a sparse drifting field over both sides (frame 09, clip 08).
 *  - dust (world): sparse neutral specks over the whole volume, near ones as bokeh.
 *
 * Additive, depthTest on, depthWrite off, energy-conserving soft sprites (glsl.ts).
 *
 * The builders are step generators (vegetation/slices.ts): they `yield` between particles
 * every few hundred iterations, so OracleScene runs them in ≈ 8 ms slices. A yield never
 * changes what is drawn from the rng or in which order.
 */
import {
  BufferAttribute,
  BufferGeometry,
  Color,
  Points,
  ShaderMaterial,
  Vector3,
  type IUniform,
} from "three";
import type { Rng } from "../../core/rng";
import type { Steps } from "../../vegetation/slices";
import { NOISE_GLSL, SPRITE_FRAGMENT, SPRITE_VERTEX_GLSL } from "./glsl";
import { additiveKeepAlpha } from "./layer";

// ---------------------------------------------------------------------------
// Palette (pre-tone-mapping linear values; measured against frames 07–09 after ACES)
// ---------------------------------------------------------------------------

/** Accents at "unit" brightness: lime, green, teal and their pale versions, swirl greys. */
export const ORACLE_PALETTE = {
  lime: new Color(0.2, 1.0, 0.04),
  /** The most common dot colour inside the orb (frame 08: pale green, R ≈ B ≈ 0.3 G before ACES). */
  mint: new Color(0.3, 1.0, 0.36),
  green: new Color(0.09, 0.66, 0.11),
  teal: new Color(0.09, 0.6, 0.5),
  paleLime: new Color(0.5, 1.0, 0.34),
  paleTeal: new Color(0.36, 0.8, 0.7),
  /** The loose swirl is much paler than the settled orb (frame 07: R ≈ B ≈ 0.66 G). */
  swirlTeal: new Color(0.6, 1.0, 0.72),
  swirlLime: new Color(0.7, 1.0, 0.52),
  /** Dust specks are almost neutral. */
  dust: new Color(0.95, 1.0, 0.93),
} as const;

export type Palette = typeof ORACLE_PALETTE;

/** Uniforms shared by every sprite material (updated once per frame). */
export interface SpriteUniforms {
  uScale: IUniform<number>;
  uMinPx: IUniform<number>;
  uMaxPx: IUniform<number>;
  uFocus: IUniform<number>;
  uCocNear: IUniform<number>;
  uCocFar: IUniform<number>;
  uTime: IUniform<number>;
}

export function createSpriteUniforms(): SpriteUniforms {
  return {
    uScale: { value: 1600 },
    uMinPx: { value: 1.6 },
    uMaxPx: { value: 64 },
    uFocus: { value: 5 },
    uCocNear: { value: 16 },
    uCocFar: { value: 3 },
    uTime: { value: 0 },
  };
}

function spriteMaterial(name: string, vertexShader: string, shared: SpriteUniforms, own: Record<string, IUniform>): ShaderMaterial {
  // additive colour, layer alpha untouched (drawn into the set's blend layer, see layer.ts)
  return additiveKeepAlpha(
    new ShaderMaterial({
      name,
      uniforms: { ...shared, ...own },
      vertexShader,
      fragmentShader: SPRITE_FRAGMENT,
      transparent: true,
      depthWrite: false,
      depthTest: true,
    }),
  );
}

// ---------------------------------------------------------------------------
// Small seeded value noise (CPU side, for densities)
// ---------------------------------------------------------------------------

export type Noise3 = (x: number, y: number, z: number) => number;

/** Value noise in [0, 1], fully determined by the rng stream. */
export function makeNoise3(rng: Rng): Noise3 {
  const perm = new Uint8Array(512);
  const p = Array.from({ length: 256 }, (_, i) => i);
  for (let i = 255; i > 0; i--) {
    const j = Math.floor(rng.next() * (i + 1));
    [p[i], p[j]] = [p[j], p[i]];
  }
  for (let i = 0; i < 512; i++) perm[i] = p[i & 255];
  const vals = new Float32Array(256);
  for (let i = 0; i < 256; i++) vals[i] = rng.next();
  const h = (x: number, y: number, z: number) => vals[perm[perm[perm[x & 255] + (y & 255)] + (z & 255)]];
  const fade = (t: number) => t * t * (3 - 2 * t);
  const l = (a: number, b: number, k: number) => a + (b - a) * k;
  return (x, y, z) => {
    const xi = Math.floor(x);
    const yi = Math.floor(y);
    const zi = Math.floor(z);
    const u = fade(x - xi);
    const v = fade(y - yi);
    const w = fade(z - zi);
    return l(
      l(l(h(xi, yi, zi), h(xi + 1, yi, zi), u), l(h(xi, yi + 1, zi), h(xi + 1, yi + 1, zi), u), v),
      l(l(h(xi, yi, zi + 1), h(xi + 1, yi, zi + 1), u), l(h(xi, yi + 1, zi + 1), h(xi + 1, yi + 1, zi + 1), u), v),
      w,
    );
  };
}

function randomInBall(rng: Rng, radius: number, out: Vector3): Vector3 {
  const r = radius * Math.cbrt(rng.next());
  const z = rng.range(-1, 1);
  const a = rng.range(0, Math.PI * 2);
  const s = Math.sqrt(1 - z * z);
  return out.set(r * s * Math.cos(a), r * z, r * s * Math.sin(a));
}

function randomDirection(rng: Rng, out: Vector3): Vector3 {
  const z = rng.range(-1, 1);
  const a = rng.range(0, Math.PI * 2);
  const s = Math.sqrt(1 - z * z);
  return out.set(s * Math.cos(a), z, s * Math.sin(a));
}

/** Log-normal-ish brightness: most particles dim, a few hot ones. */
function brightness(rng: Rng, median: number, spread: number, hotChance: number, hot: [number, number]): number {
  let b = median * Math.exp(spread * rng.normal());
  if (rng.chance(hotChance)) b *= rng.range(hot[0], hot[1]);
  return b;
}

// ---------------------------------------------------------------------------
// Orb swarm
// ---------------------------------------------------------------------------

const ORB_VERTEX = /* glsl */ `
attribute vec3 aStart;
attribute vec4 aSeed;   // x: sprite extent (m), y: phase, z: delay, w: twinkle rate
attribute vec3 aColor;
attribute vec3 aColor0;
attribute float aGroup; // integer part: 0 interior, 1 membrane, 2 sparks, 3 swirl only; fraction: swirl rate
uniform float uAssemble;
uniform float uSwirl;
uniform float uSwirlOpacity;
uniform float uSpin;
uniform float uSparks;
uniform float uFountain;
uniform float uCore;
uniform float uLit;
uniform float uExit;
uniform float uOpacity;
${NOISE_GLSL}
${SPRITE_VERTEX_GLSL}
void main() {
  float ph = aSeed.y * 6.2831853;
  float grp = floor(aGroup + 0.001);
  float rate = fract(aGroup);
  vec3 p;
  vec3 col;
  float halo = 0.0;
  float sizeK = 1.0;
  float hide = 0.0;
  if (abs(grp - 2.0) < 0.5) {
    // sparks out of the top of the orb (looping on the ambient clock): a sparse dome of
    // slow sparks with individual reach in the oracle episode (frame 08), a tall column
    // in the streams episode
    float cyc = fract(aSeed.z + uTime * (0.045 + 0.05 * aSeed.w));
    p = position;
    p.y += cyc * mix(0.35 + 0.75 * fract(aSeed.y * 7.31), 2.3, uFountain);
    p.xz *= 1.0 + 0.9 * cyc;
    p += (0.03 + 0.08 * cyc) * oNoise3(position * 3.0 + vec3(0.0, uTime * 0.15, 0.0));
    p = oRotY(uSpin * 0.6) * p;
    p.y -= uExit * 0.6;
    float life = smoothstep(0.0, 0.1, cyc) * (1.0 - smoothstep(mix(0.25, 0.45, uFountain), 1.0, cyc));
    float show = step(aSeed.w, mix(0.26, 1.0, uFountain));
    col = aColor * life * show * uSparks;
  } else {
    float k = clamp((uAssemble - aSeed.z * 0.5) / 0.5, 0.0, 1.0);
    float e = k * k * (3.0 - 2.0 * k);
    // vortex: inner parts of the sheets turn faster than the outer ones (coherent shear)
    vec3 s = oRotY(uSwirl * rate) * aStart;
    s += 0.06 * oNoise3(aStart * 1.6 + vec3(uSwirl * 0.18, 0.0, 0.0));
    vec3 f = oRotY(uSpin) * position;
    float drift = grp > 0.5 ? 0.004 : 0.011;
    // swirl-only grains (group 3) settle into the membrane and fade out: skip them after that
    hide = step(2.5, grp) * step(0.999, e);
    f += drift * vec3(sin(uTime * 0.7 + ph), sin(uTime * 0.53 + ph * 1.7), cos(uTime * 0.61 + ph * 0.6));
    p = mix(s, f, e);
    // a curl while in flight (zero at both ends of the path)
    p = oRotY((1.0 - e) * e * 3.2 * (aSeed.y - 0.4)) * p;
    // exit: the orb loosens
    vec3 outward = normalize(f + vec3(1e-4, 2e-4, 0.0));
    p += (outward * 0.22 + vec3(0.0, -0.25, 0.0)) * uExit * (0.25 + aSeed.z);
    // the heart lights the distinct dots around it (not the faint dust: no haze)
    float coreBoost = 1.0 + uCore * (1.0 - smoothstep(0.0, 0.4, length(position))) * step(grp, 0.5) * step(0.25, aColor.g);
    // settled dots light up gradually once the orb has formed (uLit)
    col = mix(aColor0 * uSwirlOpacity, aColor * coreBoost * mix(0.45, 1.0, uLit), e);
    halo = clamp((max(col.r, col.g) - 1.4) * 0.1, 0.0, 0.3);
    // swirl particles are finer than the settled ones
    sizeK = mix(0.5, 1.0, e);
  }
  float tw = 0.72 + 0.28 * sin(uTime * (1.1 + 2.6 * aSeed.w) + ph);
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  gl_Position = hide > 0.5 ? vec4(2.0, 2.0, 2.0, 1.0) : projectionMatrix * mv;
  oSprite(mv, aSeed.x * sizeK, col * tw * uOpacity, halo);
}
`;

export interface OrbSwarmOptions {
  count: number;
  /** Extra fine grains that only exist in the swirl (they settle into the membrane and fade). */
  swirlCount: number;
  /** Node centres (orb units): extra clumps gather around them. */
  attractors: Vector3[];
  palette: Palette;
  /** Brightness compensation for low particle budgets. */
  gain: number;
}

export interface OrbSwarm {
  points: Points<BufferGeometry, ShaderMaterial>;
  uniforms: {
    uAssemble: IUniform<number>;
    uSwirl: IUniform<number>;
    uSwirlOpacity: IUniform<number>;
    uSpin: IUniform<number>;
    uSparks: IUniform<number>;
    uFountain: IUniform<number>;
    uCore: IUniform<number>;
    uLit: IUniform<number>;
    uExit: IUniform<number>;
    uOpacity: IUniform<number>;
  };
}

/** The swirl turns around a tilted axis through a knot left of the orb centre (frame 07). */
const SWIRL_KNOT = new Vector3(-0.22, 0.0, 0.1);
const SWIRL_AXIS = new Vector3(0.3, 1, 0.15).normalize();
const SWIRL_E1 = new Vector3().crossVectors(SWIRL_AXIS, new Vector3(0, 0, 1)).normalize();
const SWIRL_E2 = new Vector3().crossVectors(SWIRL_AXIS, SWIRL_E1);

/** A curtain-like ribbon of the swirl, wound around the knot axis. */
interface SwirlRibbon {
  phi0: number;
  rho0: number;
  rho1: number;
  h0: number;
  tilt: number;
  width: number;
  twist: number;
  weight: number;
}

function makeSwirlRibbons(rng: Rng, count: number): SwirlRibbon[] {
  return Array.from({ length: count }, (_, i) => {
    const rho0 = rng.range(0.06, 0.35);
    const rho1 = rng.range(0.75, 1.4);
    const width = rng.range(0.1, 0.36);
    return {
      phi0: (i % 3) * ((Math.PI * 2) / 3) + rng.range(-0.5, 0.5),
      rho0,
      rho1,
      h0: rng.range(-0.45, 0.45),
      tilt: rng.range(-0.5, 0.5),
      width,
      twist: rng.range(-2.2, 2.2),
      weight: (rho1 - rho0) * width,
    };
  });
}

/**
 * Start position in the loose swirl (orb units, frame 07): mostly narrow curtains wound
 * as logarithmic spirals around a knot left of the centre, so they converge on the knot
 * and fan out to the sides; denser along their edges, which read as bright seams; a
 * few loose wisps on crumpled noise iso-surfaces further out; the knot itself. Returns
 * the normalised distance from the knot (0 – 1), which sets the vortex rate.
 */
function swirlPoint(rng: Rng, noise: Noise3, ribbons: SwirlRibbon[], totalWeight: number, out: Vector3): number {
  const kind = rng.next();
  if (kind < 0.06) {
    out.set(rng.normal() * 0.1, rng.normal() * 0.07, rng.normal() * 0.08).add(SWIRL_KNOT);
    return 0.1;
  }
  if (kind < 0.9) {
    let pickW = rng.next() * totalWeight;
    let rb = ribbons[0];
    for (const r of ribbons) {
      pickW -= r.weight;
      rb = r;
      if (pickW <= 0) break;
    }
    const u = rng.next();
    const rho = rb.rho0 + (rb.rho1 - rb.rho0) * u;
    const phi = rb.phi0 + 1.25 * Math.log(rho / 0.06);
    // across the curtain: denser towards both edges (folds seen edge-on)
    let v = rng.range(-1, 1);
    if (rng.chance(0.45)) v = (rng.chance(0.5) ? 1 : -1) * (1 - Math.abs(rng.normal()) * 0.12);
    const tw = rb.twist * u;
    const w = v * rb.width * (0.6 + 0.6 * u);
    const c = Math.cos(phi);
    const s = Math.sin(phi);
    out
      .copy(SWIRL_KNOT)
      .addScaledVector(SWIRL_E1, c * (rho + Math.sin(tw) * w))
      .addScaledVector(SWIRL_E2, s * (rho + Math.sin(tw) * w) * 0.75)
      .addScaledVector(SWIRL_AXIS, rb.h0 * u + rb.tilt * u * u + Math.cos(tw) * w);
    const n = noise(out.x * 2.2 + 3.3, out.y * 2.2, out.z * 2.2) - 0.5;
    const m = noise(out.x * 4.4, out.y * 4.4 + 7.7, out.z * 4.4) - 0.5;
    out.x += n * 0.12;
    out.y += m * 0.08;
    out.z += (n - m) * 0.08;
    // the knot sits left of the centre: the curtains reach further to the right
    out.x = SWIRL_KNOT.x + (out.x - SWIRL_KNOT.x) * (out.x < SWIRL_KNOT.x ? 0.8 : 1.15);
    return Math.min(1, rho / 1.4);
  }
  let env = 1;
  for (let tries = 0; tries < 48; tries++) {
    randomInBall(rng, 1, out);
    env = out.length();
    out.x *= 1.4;
    out.z *= 0.8;
    const f = noise(out.x * 0.95 + 2.1, out.y * 0.95 - 0.7, out.z * 0.95 + 4.4);
    const g = noise(out.x * 2.6 + 9.2, out.y * 2.6, out.z * 2.6 - 1.3);
    const d = Math.abs(f - 0.5 + (g - 0.5) * 0.3);
    if (d < 0.018 + 0.03 * env && env > 0.45) break;
  }
  return env;
}

/** Dotted rings strung through the interior (frames 08 / 09). */
interface DotRing {
  centre: Vector3;
  t1: Vector3;
  t2: Vector3;
  radius: number;
  arc0: number;
  arcLen: number;
  count: number;
}

function makeRings(rng: Rng, n: number): DotRing[] {
  return Array.from({ length: n }, () => {
    const axis = randomDirection(rng, new Vector3());
    const ref = Math.abs(axis.y) < 0.9 ? new Vector3(0, 1, 0) : new Vector3(1, 0, 0);
    const t1 = new Vector3().crossVectors(axis, ref).normalize();
    const t2 = new Vector3().crossVectors(axis, t1);
    return {
      centre: randomInBall(rng, 0.15, new Vector3()),
      t1,
      t2,
      radius: rng.range(0.35, 0.82),
      arc0: rng.range(0, Math.PI * 2),
      arcLen: rng.range(Math.PI * 0.6, Math.PI * 1.6),
      count: 0,
    };
  });
}

export function* buildOrbSwarm(rng: Rng, shared: SpriteUniforms, opts: OrbSwarmOptions): Steps<OrbSwarm> {
  const nOrb = Math.max(200, opts.count);
  const n = nOrb + Math.max(0, opts.swirlCount);
  const noise = makeNoise3(rng.fork("noise"));
  const pos = new Float32Array(n * 3);
  const start = new Float32Array(n * 3);
  const seed = new Float32Array(n * 4);
  const color = new Float32Array(n * 3);
  const color0 = new Float32Array(n * 3);
  const group = new Float32Array(n);
  const v = new Vector3();
  const s = new Vector3();
  const pal = opts.palette;
  const gain = opts.gain;
  const pick = (r: number) =>
    r < 0.35 ? pal.mint : r < 0.55 ? pal.paleLime : r < 0.7 ? pal.lime : r < 0.85 ? pal.teal : pal.paleTeal;

  // clump centres: around the nodes plus free ones in the ball
  const clumps: { c: Vector3; sigma: number }[] = [];
  for (const a of opts.attractors) clumps.push({ c: a.clone(), sigma: rng.range(0.06, 0.1) });
  for (let i = 0; i < 22; i++) clumps.push({ c: randomInBall(rng, 0.8, new Vector3()), sigma: rng.range(0.05, 0.12) });
  const rings = makeRings(rng, 7);
  const ribbons = makeSwirlRibbons(rng.fork("swirl"), 14);
  const ribbonWeight = ribbons.reduce((sum, r) => sum + r.weight, 0);

  const nSparks = Math.round(nOrb * 0.035);
  const nMembrane = Math.round(nOrb * 0.12);
  for (let i = 0; i < n; i++) {
    if ((i & 127) === 127) yield;
    let g = 0;
    if (i >= nOrb) g = 3;
    else if (i < nSparks) g = 2;
    else if (i < nSparks + nMembrane) g = 1;
    let size: number;
    let b: number;
    let c: Color;
    if (g === 3) {
      // swirl-only grain: ends just inside the shell, dark
      randomDirection(rng, v).multiplyScalar(rng.range(0.88, 0.97));
      size = rng.range(0.012, 0.02);
      b = 0;
      c = pal.paleLime;
    } else if (g === 2) {
      // sparks start under the top of the shell and rise out of it (shader)
      const r = 0.85 * Math.sqrt(rng.next());
      const a = rng.range(0, Math.PI * 2);
      v.set(r * Math.cos(a), Math.sqrt(Math.max(0, 1 - r * r)) * rng.range(0.82, 0.98), r * Math.sin(a) * 0.8);
      size = rng.range(0.016, 0.03);
      b = brightness(rng, 0.65, 0.4, 0.1, [1.8, 3]);
      c = rng.chance(0.45) ? pal.lime : rng.chance(0.5) ? pal.paleLime : rng.chance(0.5) ? pal.green : pal.teal;
    } else if (g === 1) {
      // patchy membrane just inside the shell: very fine, dim
      let tries = 0;
      do {
        randomDirection(rng, v).multiplyScalar(rng.range(0.86, 0.975));
        tries++;
      } while (noise(v.x * 3.4 + 11, v.y * 3.4, v.z * 3.4) < 0.45 && tries < 12);
      size = rng.range(0.008, 0.013);
      b = brightness(rng, 0.13, 0.4, 0.015, [2, 3]);
      c = rng.chance(0.55) ? pal.paleLime : rng.chance(0.5) ? pal.paleTeal : pal.green;
    } else if (rng.chance(0.07)) {
      // dotted rings: evenly strung fine dots
      const ring = rng.pick(rings);
      const th = ring.arc0 + ring.arcLen * ((ring.count++ * 0.6180339887) % 1);
      v.copy(ring.centre).addScaledVector(ring.t1, Math.cos(th) * ring.radius).addScaledVector(ring.t2, Math.sin(th) * ring.radius);
      if (v.length() > 0.93) v.setLength(0.93);
      size = rng.range(0.01, 0.014);
      b = brightness(rng, 0.32, 0.3, 0.03, [2, 3]);
      c = rng.chance(0.6) ? pal.paleLime : pal.lime;
    } else {
      const r = rng.next();
      if (r < 0.15) {
        const cl = rng.pick(clumps);
        v.set(rng.normal(), rng.normal(), rng.normal()).multiplyScalar(cl.sigma).add(cl.c);
      } else {
        let tries = 0;
        do {
          randomInBall(rng, 0.9, v);
          tries++;
        } while (noise(v.x * 2.3, v.y * 2.3 + 5, v.z * 2.3) < 0.5 && tries < 12);
      }
      if (v.length() > 0.93) v.setLength(0.93 * rng.range(0.85, 1));
      // distinct crisp dots, the rest very fine faint dust (gaps between the dots stay dark)
      const dot = rng.chance(0.13);
      size = dot ? rng.range(0.02, 0.034) : rng.range(0.008, 0.012);
      b = dot ? brightness(rng, 0.9, 0.4, 0.08, [2, 3]) : brightness(rng, 0.045, 0.5, 0.01, [3, 5]);
      c = pick(rng.next());
    }
    pos.set([v.x, v.y, v.z], i * 3);

    // swirl start (sparks start in place); the left part is pale teal, the right pale lime
    let along = 0.5;
    if (g === 2) s.copy(v);
    else along = swirlPoint(rng, noise, ribbons, ribbonWeight, s);
    start.set([s.x, s.y, s.z], i * 3);
    const sheet = s.x + (noise(s.x * 2.2 + 1.1, s.y * 2.2, s.z * 2.2) - 0.5) * 0.6 < -0.05 ? 0 : 1;
    // vortex rate (fraction of aGroup): 0.95 in the middle of the swirl → 0.45 at its edge
    group[i] = g + 0.05 + 0.9 * (0.95 - 0.5 * along) * 0.98;

    color.set([c.r * b * gain, c.g * b * gain, c.b * b * gain], i * 3);
    // swirl colour: pale greys; a few saturated lime specks on the right sheet
    const hot = sheet === 1 && rng.chance(0.03);
    const c0 = hot ? pal.lime : sheet === 0 ? (rng.chance(0.7) ? pal.swirlTeal : pal.paleTeal) : pal.swirlLime;
    const b0 = (hot ? rng.range(0.5, 0.9) : brightness(rng, 0.26, 0.5, 0.015, [2, 3])) * gain;
    color0.set([c0.r * b0, c0.g * b0, c0.b * b0], i * 3);

    seed.set([size, rng.next(), rng.next(), rng.next()], i * 4);
  }
  yield;

  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new BufferAttribute(pos, 3));
  geometry.setAttribute("aStart", new BufferAttribute(start, 3));
  geometry.setAttribute("aSeed", new BufferAttribute(seed, 4));
  geometry.setAttribute("aColor", new BufferAttribute(color, 3));
  geometry.setAttribute("aColor0", new BufferAttribute(color0, 3));
  geometry.setAttribute("aGroup", new BufferAttribute(group, 1));
  geometry.computeBoundingSphere();

  const uniforms = {
    uAssemble: { value: 1 },
    uSwirl: { value: 0 },
    uSwirlOpacity: { value: 1 },
    uSpin: { value: 0 },
    uSparks: { value: 1 },
    uFountain: { value: 0 },
    uCore: { value: 0 },
    uLit: { value: 1 },
    uExit: { value: 0 },
    uOpacity: { value: 1 },
  };
  const material = spriteMaterial("OracleOrbSwarm", ORB_VERTEX, shared, uniforms);
  const points = new Points(geometry, material);
  points.name = "oracle_orb_swarm";
  points.frustumCulled = false;
  return { points, uniforms };
}

// ---------------------------------------------------------------------------
// Streams (a turning, rising particle disc around the orb)
// ---------------------------------------------------------------------------

const STREAM_VERTEX = /* glsl */ `
attribute vec4 aDisc;   // x: radius (orb units), y: start angle, z: height (x half thickness), w: own rise
attribute vec4 aSeed;   // x: sprite extent (m), y: phase, z: appear delay, w: speed jitter
attribute vec3 aColor;
attribute float aNear;  // extra offset towards the camera (orb units): soft foreground bokeh
uniform float uStory;    // story seconds since the streams began
uniform float uOn;       // 0 -> 1 presence
uniform float uHeight;   // disc height above the orb centre (orb units)
uniform float uThick;    // disc half thickness (orb units)
uniform float uSpread;   // radial scale of the disc
uniform float uLoose;    // 0 -> 1: the disc breaks up into a drifting field
uniform float uOpacity;
uniform float uOrbSide;  // +1: draw only particles in front of the orb centre, -1: only behind
uniform float uOrbDepth; // view-space z of the orb centre
${NOISE_GLSL}
${SPRITE_VERTEX_GLSL}
void main() {
  float r0 = aDisc.x;
  // differential rotation: inner parts turn faster; story-driven plus a slow ambient drift
  float omega = 0.45 * pow(r0, -0.75) * (0.85 + 0.3 * aSeed.w);
  float ang = aDisc.y + omega * (uStory + 0.3 * uTime);
  float r = r0 * uSpread * (1.0 + 0.12 * uLoose * aSeed.w);
  float sa = sin(ang);
  // the half towards the camera is flattened so the disc never sweeps right past the lens
  vec3 p = vec3(cos(ang) * r, uHeight + aDisc.z * uThick + aDisc.w * uLoose, sa * r * (sa > 0.0 ? 0.5 : 0.85));
  // organic flow: low-frequency displacement, stronger once the disc loosens
  p += (0.3 + 0.5 * uLoose) * oNoise3(p * 0.32 + vec3(1.7, -uTime * 0.03, uTime * 0.02));
  p.z += aNear;
  float appear = smoothstep(aSeed.z * 0.75, aSeed.z * 0.75 + 0.25, uOn);
  float ph = aSeed.y * 6.2831853;
  float tw = 0.72 + 0.28 * sin(uTime * (0.9 + 1.8 * aSeed.w) + ph);
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  float keep = step(0.0, (mv.z - uOrbDepth) * uOrbSide);
  gl_Position = keep > 0.5 ? projectionMatrix * mv : vec4(2.0, 2.0, 2.0, 1.0);
  // near-camera specks: partly make up for the bokeh dimming
  float near = smoothstep(3.6, 1.8, -mv.z);
  vec3 col = aColor * (appear * tw * uOpacity * keep * (1.0 + 0.8 * near));
  float halo = clamp((max(col.r, col.g) - 1.4) * 0.1, 0.0, 0.25);
  oSprite(mv, aSeed.x, col, halo);
}
`;

export interface Streams {
  /** Particles behind the orb centre (drawn before the orb). */
  back: Points<BufferGeometry, ShaderMaterial>;
  /** The same buffers, particles in front of the orb centre (drawn after the orb). */
  front: Points<BufferGeometry, ShaderMaterial>;
  uniforms: {
    uOrbDepth: IUniform<number>;
    uStory: IUniform<number>;
    uOn: IUniform<number>;
    uHeight: IUniform<number>;
    uThick: IUniform<number>;
    uSpread: IUniform<number>;
    uLoose: IUniform<number>;
    uOpacity: IUniform<number>;
  };
}

/** Disc radii (orb units): inner edge clear of the orb, ring of the densest stream. */
const DISC_INNER = 1.45;
const DISC_RING = 3.9;

export function* buildStreams(rng: Rng, shared: SpriteUniforms, count: number, palette: Palette, gain: number): Steps<Streams> {
  const n = Math.max(100, count);
  const disc = new Float32Array(n * 4);
  const seed = new Float32Array(n * 4);
  const color = new Float32Array(n * 3);
  const near = new Float32Array(n);
  // two broad lobes (left / right of the orb) carry most particles; clumps shear into streaks
  const lobeAngle = (left: boolean) => (left ? Math.PI : 0) - 0.45 + rng.normal() * 0.55;
  const clumps = Array.from({ length: Math.max(24, Math.round(n / 70)) }, () => ({
    r: Math.max(DISC_INNER, DISC_RING + rng.normal() * 1.3),
    a: lobeAngle(rng.chance(0.5)),
    z: rng.normal() * 0.5,
    spread: rng.range(0.6, 1.4),
  }));
  for (let i = 0; i < n; i++) {
    if ((i & 255) === 255) yield;
    let r: number;
    let a: number;
    let z: number;
    const kind = rng.next();
    if (kind < 0.35) {
      const cl = rng.pick(clumps);
      r = cl.r + rng.normal() * 0.22 * cl.spread;
      a = cl.a + (rng.normal() * 0.09 * cl.spread) / Math.max(1, cl.r * 0.3);
      z = cl.z + rng.normal() * 0.22 * cl.spread;
    } else if (kind < 0.85) {
      r = DISC_RING + rng.normal() * 0.8;
      a = lobeAngle(rng.chance(0.5));
      z = rng.normal() * 0.55;
    } else {
      r = DISC_INNER + 5.2 * Math.sqrt(rng.next());
      a = rng.range(0, Math.PI * 2);
      z = rng.normal() * 0.7;
    }
    r = Math.max(DISC_INNER, r);
    z = Math.max(-1.8, Math.min(1.8, z));
    const rise = rng.range(-0.4, 0.9);
    disc.set([r, a, z, rise], i * 4);
    // the left lobe arrives first (clip 08, 25.9 s), the right one a little later
    const sector = 0.5 + 0.5 * Math.cos(a - 2.6);
    const delay = 0.6 * (1 - sector) + 0.4 * rng.next();
    // a tenth of the particles drift closer to the lens: soft, dim bokeh discs (detail 05)
    const fg = rng.chance(0.1);
    near[i] = fg ? rng.range(1.5, 4.5) : 0;
    const s = rng.next();
    const big = s < 0.03;
    const mid = s < 0.3;
    const size = big ? rng.range(0.035, 0.05) : mid ? rng.range(0.026, 0.04) : rng.range(0.015, 0.023);
    let b = big
      ? brightness(rng, 0.35, 0.3, 0, [1, 1])
      : mid
        ? brightness(rng, 0.62, 0.4, 0.06, [2, 3])
        : brightness(rng, 0.2, 0.5, 0.02, [2, 4]);
    if (fg) b *= 0.8;
    b *= gain;
    const cr = rng.next();
    const c =
      cr < 0.4 ? palette.mint : cr < 0.55 ? palette.paleLime : cr < 0.7 ? palette.lime : cr < 0.85 ? palette.green : palette.teal;
    color.set([c.r * b, c.g * b, c.b * b], i * 3);
    seed.set([size, rng.next(), delay, rng.next()], i * 4);
  }
  const geometry = new BufferGeometry();
  // positions are computed in the shader; keep a dummy attribute for three.js
  geometry.setAttribute("position", new BufferAttribute(new Float32Array(n * 3), 3));
  geometry.setAttribute("aDisc", new BufferAttribute(disc, 4));
  geometry.setAttribute("aSeed", new BufferAttribute(seed, 4));
  geometry.setAttribute("aColor", new BufferAttribute(color, 3));
  geometry.setAttribute("aNear", new BufferAttribute(near, 1));
  const uniforms = {
    uOrbDepth: { value: -5 },
    uStory: { value: 0 },
    uOn: { value: 0 },
    uHeight: { value: 0 },
    uThick: { value: 0.3 },
    uSpread: { value: 1 },
    uLoose: { value: 0 },
    uOpacity: { value: 1 },
  };
  const make = (side: number, name: string) => {
    const material = spriteMaterial(name, STREAM_VERTEX, shared, { ...uniforms, uOrbSide: { value: side } });
    const points = new Points(geometry, material);
    points.name = name;
    points.frustumCulled = false;
    return points;
  };
  return { back: make(-1, "oracle_streams_back"), front: make(1, "oracle_streams_front"), uniforms };
}

// ---------------------------------------------------------------------------
// Dust
// ---------------------------------------------------------------------------

const DUST_VERTEX = /* glsl */ `
attribute vec4 aSeed;   // x: sprite extent (m), y: phase, z/w: drift rates
attribute vec3 aColor;
uniform float uOpacity;
${SPRITE_VERTEX_GLSL}
void main() {
  float ph = aSeed.y * 6.2831853;
  vec3 p = position + vec3(
    0.12 * sin(uTime * (0.03 + 0.04 * aSeed.z) + ph),
    0.1 * sin(uTime * (0.025 + 0.03 * aSeed.w) + ph * 1.3),
    0.12 * cos(uTime * (0.028 + 0.03 * aSeed.z) + ph * 0.7)
  );
  float tw = 0.75 + 0.25 * sin(uTime * (0.35 + 0.8 * aSeed.w) + ph);
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  gl_Position = projectionMatrix * mv;
  oSprite(mv, aSeed.x, aColor * tw * uOpacity, 0.0);
}
`;

export interface Dust {
  points: Points<BufferGeometry, ShaderMaterial>;
  uniforms: { uOpacity: IUniform<number> };
}

/** Dust in a world box covering the whole camera path (x, y, z ranges in m). */
export function* buildDust(
  rng: Rng,
  shared: SpriteUniforms,
  count: number,
  box: { x: [number, number]; y: [number, number]; z: [number, number] },
  palette: Palette,
): Steps<Dust> {
  const n = Math.max(50, count);
  const pos = new Float32Array(n * 3);
  const seed = new Float32Array(n * 4);
  const color = new Float32Array(n * 3);
  const c = new Color();
  for (let i = 0; i < n; i++) {
    if ((i & 255) === 255) yield;
    pos.set([rng.range(box.x[0], box.x[1]), rng.range(box.y[0], box.y[1]), rng.range(box.z[0], box.z[1])], i * 3);
    const size = rng.chance(0.12) ? rng.range(0.026, 0.04) : rng.range(0.012, 0.02);
    // neutral specks (frame 07: peaks around #555854), a few faintly green
    const b = brightness(rng, 0.1, 0.45, 0.05, [1.6, 2.4]);
    c.copy(palette.dust).lerp(palette.green, rng.chance(0.25) ? rng.range(0.1, 0.3) : 0).multiplyScalar(b);
    color.set([c.r, c.g, c.b], i * 3);
    seed.set([size, rng.next(), rng.next(), rng.next()], i * 4);
  }
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new BufferAttribute(pos, 3));
  geometry.setAttribute("aSeed", new BufferAttribute(seed, 4));
  geometry.setAttribute("aColor", new BufferAttribute(color, 3));
  const uniforms = { uOpacity: { value: 1 } };
  const material = spriteMaterial("OracleDust", DUST_VERTEX, shared, uniforms);
  const points = new Points(geometry, material);
  points.name = "oracle_dust";
  points.frustumCulled = false;
  return { points, uniforms };
}
