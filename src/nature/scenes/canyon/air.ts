/**
 * The air of the canyon: what turns the black gap between the two rock walls into depth.
 *
 *  - haze: a faint cool glow far down the cleft — screen space at the far plane, so it
 *    only shows where no rock or plant was drawn (the walls and their plants stand out
 *    against it as silhouettes): the sky high in the gap, a tall glow at the far end
 *    between ragged, darker far walls with a soft lit rim, slow horizontal mist banks
 *    and a low floor mist that the near form stands out against.
 *  - beams: soft shafts of warm light falling through the dusty air from high on the far
 *    side, slanting down to the left towards the lens (the main one lands on the left
 *    wall's lit edge) — world-space ribbons facing the lens, depth-tested (rock in front
 *    covers them), additive, with faint streaks along their length that slide slowly.
 *  - motes: a few hundred dust specks drifting slowly down the main beam (`THREE.Points`),
 *    bright inside it and almost invisible outside, faded out at both ends of the beam
 *    (the wrap back to the top never pops); specks off the focus distance grow a soft
 *    defocus disc keeping the energy of their in-focus sprite.
 *
 * One mesh (haze + beams) and one Points set: two draw calls. Additive colour, alpha left
 * as it is (the DOF composite reads the scene alpha as coverage), no depth writes, no
 * fog. Everything is a pure function of the ambient clock `uTime` (FrameState.timeSec)
 * and of the seed; nothing accumulates. Code-generated background and particles
 * (CLAUDE.md), no textures. Colours are linear HDR (before exposure and grade).
 */
import {
  BufferAttribute,
  BufferGeometry,
  Color,
  CustomBlending,
  Mesh,
  OneFactor,
  Points,
  ShaderMaterial,
  Vector2,
  Vector3,
  Vector4,
  ZeroFactor,
  type IUniform,
} from "three";
import type { Rng } from "../../core/rng";

/** A light shaft: straight axis from `top` along `dir` for `length` m, soft edges. */
export interface BeamSpec {
  top: Vector3;
  /** Direction of travel of the light (normalised by the builder). */
  dir: Vector3;
  length: number;
  /** Half widths (m) at the top and at the far end (light spreads a little). */
  width: [number, number];
  /** Peak strength (× `color`). */
  strength: number;
  /** Streak pattern: phase and number of streaks across the beam. */
  phase: number;
  streaks: number;
  /** Along the beam (0 top … 1 end): fade-in length and fall-off power towards the end. */
  rise: number;
  fall: number;
}

export interface AirOptions {
  beams: BeamSpec[];
  /** Linear colour of the beams at strength 1. */
  beamColor: Color;
  haze: HazeSpec;
  /** Faces of the lens-facing ribbons point along this view direction (towards the lens). */
  facing: Vector3;
}

/** The far haze in screen space (x 0–1 left → right, y 0–1 bottom → top; sizes in frame heights). */
export interface HazeSpec {
  /** Sky over the cleft: a glow from the top, centred in the gap; faded out at `floor`, full at `top`. */
  sky: { color: Color; x: number; halfWidth: number; floor: number; top: number };
  /** The far end of the cleft: a soft upright glow (centre, radii). */
  end: { color: Color; x: number; y: number; rx: number; ry: number };
  /** Mist lying on the floor of the cleft, up to `height`. */
  floor: { color: Color; height: number };
  /**
   * The cleft going on into the distance: two far walls in the mist, their inner edges
   * at `left` / `right` (0–1 of the width, ragged by `ragged`), `shade` darker than the
   * haze in front of them, a faint `rim` of the far glow along those edges.
   */
  far: { left: number; right: number; ragged: number; shade: number; rim: number };
}

export interface AirUniforms {
  [name: string]: IUniform;
  uTime: IUniform<number>;
  uAspect: IUniform<number>;
  uBeamOpacity: IUniform<number>;
  uHazeOpacity: IUniform<number>;
  uBeamColor: IUniform<Color>;
  uSkyColor: IUniform<Color>;
  /** x centre, y half width, z floor, w top. */
  uSky: IUniform<Vector4>;
  uEndColor: IUniform<Color>;
  /** x, y centre, z, w radii. */
  uEnd: IUniform<Vector4>;
  uFloorColor: IUniform<Color>;
  uFloorHeight: IUniform<number>;
  /** x left edge, y right edge, z raggedness, w shade; rim strength. */
  uFar: IUniform<Vector4>;
  uFarRim: IUniform<number>;
}

export interface Air {
  mesh: Mesh<BufferGeometry, ShaderMaterial>;
  uniforms: AirUniforms;
}

const NOISE_GLSL = /* glsl */ `
float airHash(vec2 p) {
  p = fract(p * vec2(127.13, 311.71));
  p += dot(p, p + 34.17);
  return fract(p.x * p.y);
}
float airNoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  float a = airHash(i);
  float b = airHash(i + vec2(1.0, 0.0));
  float c = airHash(i + vec2(0.0, 1.0));
  float d = airHash(i + vec2(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
float airFbm(vec2 p) {
  float s = 0.5 * airNoise(p);
  s += 0.3 * airNoise(p * 2.03 + 17.1);
  s += 0.2 * airNoise(p * 4.07 + 41.7);
  return s;
}
`;

const AIR_VERTEX = /* glsl */ `
attribute vec4 aAir;   // x: kind (0 haze, 1 beam), y: across (−1…1), z: along (0 top … 1 end), w: strength
attribute vec4 aBeam;  // x: streak phase, y: streaks across, z: fade-in length, w: fall-off power
varying vec4 vAir;
varying vec4 vBeam;
varying vec2 vScreen;
void main() {
  vAir = aAir;
  vBeam = aBeam;
  if (aAir.x < 0.5) {
    // fullscreen triangle at the far plane (depth 1: passes only where nothing was drawn)
    vScreen = position.xy * 0.5 + 0.5;
    gl_Position = vec4(position.xy, 1.0, 1.0);
  } else {
    vScreen = vec2(0.0);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
}
`;

const AIR_FRAGMENT = /* glsl */ `
uniform float uTime;
uniform float uAspect;
uniform float uBeamOpacity;
uniform float uHazeOpacity;
uniform vec3 uBeamColor;
uniform vec3 uSkyColor;
uniform vec4 uSky;
uniform vec3 uEndColor;
uniform vec4 uEnd;
uniform vec3 uFloorColor;
uniform float uFloorHeight;
uniform vec4 uFar;
uniform float uFarRim;
varying vec4 vAir;
varying vec4 vBeam;
varying vec2 vScreen;
${NOISE_GLSL}
void main() {
  vec3 c;
  if (vAir.x < 0.5) {
    vec2 uv = vScreen;
    // sky over the cleft: from the top, centred in the gap
    float dx = (uv.x - uSky.x) * uAspect / uSky.y;
    float rise = smoothstep(uSky.z, uSky.w, uv.y);
    float sky = exp(-dx * dx) * pow(rise, 1.5);
    // the far end: a soft upright glow low in the gap
    vec2 e = (uv - uEnd.xy) * vec2(uAspect, 1.0) / uEnd.zw;
    float far = exp(-dot(e, e));
    // slow mist: long soft banks drifting sideways, a finer layer sinking
    float mist = airFbm(vec2(uv.x * uAspect * 1.5 + uTime * 0.011, uv.y * 4.6 - uTime * 0.004));
    float fine = airNoise(vec2(uv.x * uAspect * 6.0 - uTime * 0.017, uv.y * 11.0 + uTime * 0.006));
    float m = 0.42 + 0.85 * mist + 0.14 * fine;
    // mist on the floor: banks of it, thicker low down
    float low = 1.0 - smoothstep(0.0, uFloorHeight, uv.y);
    float bank = airFbm(vec2(uv.x * uAspect * 1.6 - uTime * 0.008, uv.y * 5.0 + 3.7));
    float fl = low * low * smoothstep(0.2, 0.8, bank + 0.3 * low);
    // far walls: ragged upright edges, lost in the sky glow at the top and in the mist below
    float ry = uv.y * 3.2;
    float eL = uFar.x + uFar.z * (airFbm(vec2(ry, 3.1)) - 0.5) + 0.35 * uFar.z * (airNoise(vec2(uv.y * 14.0, 7.7)) - 0.5);
    float eR = uFar.y + uFar.z * (airFbm(vec2(ry + 11.3, 8.4)) - 0.5) + 0.35 * uFar.z * (airNoise(vec2(uv.y * 13.0, 1.9)) - 0.5);
    float soft = 0.02;
    float inL = 1.0 - smoothstep(eL - soft, eL + soft, uv.x);
    float inR = smoothstep(eR - soft, eR + soft, uv.x);
    float wall = max(inL, inR) * smoothstep(0.95, 0.55, uv.y) * smoothstep(-0.1, 0.3, uv.y);
    // the faces next to the edges turn towards the far glow: lighter there, darker outwards
    float dL = max(eL - uv.x, 0.0) * uAspect / 0.11;
    float dR = max(uv.x - eR, 0.0) * uAspect / 0.11;
    float rim = (inL * exp(-dL * dL) + inR * exp(-dR * dR)) * wall;
    vec3 haze = (uSkyColor * sky + uEndColor * far) * m;
    c = haze * (1.0 - uFar.w * wall) + uEndColor * (uFarRim * rim * (0.6 + 0.6 * far)) + uFloorColor * fl;
    c *= uHazeOpacity;
  } else {
    float u = vAir.y;
    float v = vAir.z;
    float prof = exp(-u * u * 2.2) * (1.0 - smoothstep(0.6, 1.0, abs(u)));
    float along = smoothstep(0.0, vBeam.z, v) * pow(max(1.0 - v, 0.0), vBeam.w);
    // streaks run along the beam (across-noise stretched along it) and slide slowly
    float streak = airNoise(vec2(u * vBeam.y + vBeam.x, v * 1.2 - uTime * 0.03));
    streak = 0.55 + 0.6 * streak * streak;
    // dust density drifting down the beam
    float dust = 0.78 + 0.32 * airFbm(vec2(u * 1.7 + vBeam.x * 3.0, v * 5.0 - uTime * 0.06));
    c = uBeamColor * (vAir.w * prof * along * streak * dust * uBeamOpacity);
  }
  gl_FragColor = vec4(c, 1.0);
  // no-ops in the linear HDR pass; applied only when rendering straight to the canvas (post=0)
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

/** Rows along each beam ribbon (keeps the across coordinate of the trapezoid ≈ exact). */
const BEAM_ROWS = 16;

/** Additive colour, alpha untouched (coverage for the DOF composite). */
function additiveKeepAlpha(m: ShaderMaterial): ShaderMaterial {
  m.transparent = true;
  m.depthWrite = false;
  m.depthTest = true;
  m.blending = CustomBlending;
  m.blendSrc = OneFactor;
  m.blendDst = OneFactor;
  m.blendSrcAlpha = ZeroFactor;
  m.blendDstAlpha = OneFactor;
  return m;
}

export function buildAir(opts: AirOptions): Air {
  const pos: number[] = [];
  const air: number[] = [];
  const beam: number[] = [];
  const index: number[] = [];

  // haze: one fullscreen triangle (clip space)
  for (const [x, y] of [
    [-1, -1],
    [3, -1],
    [-1, 3],
  ]) {
    pos.push(x, y, 0);
    air.push(0, 0, 0, 0);
    beam.push(0, 0, 0, 0);
  }
  index.push(0, 1, 2);

  // beams: lens-facing trapezoid ribbons, BEAM_ROWS rows each
  const axis = new Vector3();
  const side = new Vector3();
  const c = new Vector3();
  for (const b of opts.beams) {
    axis.copy(b.dir).normalize();
    side.crossVectors(axis, opts.facing).normalize();
    const base = pos.length / 3;
    for (let r = 0; r <= BEAM_ROWS; r++) {
      const v = r / BEAM_ROWS;
      const w = b.width[0] + (b.width[1] - b.width[0]) * v;
      c.copy(b.top).addScaledVector(axis, b.length * v);
      for (const s of [-1, 1]) {
        pos.push(c.x + side.x * w * s, c.y + side.y * w * s, c.z + side.z * w * s);
        air.push(1, s, v, b.strength);
        beam.push(b.phase, b.streaks, b.rise, b.fall);
      }
      if (r > 0) {
        const i0 = base + (r - 1) * 2;
        index.push(i0, i0 + 1, i0 + 2, i0 + 1, i0 + 3, i0 + 2);
      }
    }
  }

  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new BufferAttribute(new Float32Array(pos), 3));
  geometry.setAttribute("aAir", new BufferAttribute(new Float32Array(air), 4));
  geometry.setAttribute("aBeam", new BufferAttribute(new Float32Array(beam), 4));
  geometry.setIndex(index);
  geometry.name = "canyon_air";

  const uniforms: AirUniforms = {
    uTime: { value: 0 },
    uAspect: { value: 1440 / 1020 },
    uBeamOpacity: { value: 1 },
    uHazeOpacity: { value: 1 },
    uBeamColor: { value: opts.beamColor.clone() },
    uSkyColor: { value: opts.haze.sky.color.clone() },
    uSky: { value: new Vector4(opts.haze.sky.x, opts.haze.sky.halfWidth, opts.haze.sky.floor, opts.haze.sky.top) },
    uEndColor: { value: opts.haze.end.color.clone() },
    uEnd: { value: new Vector4(opts.haze.end.x, opts.haze.end.y, opts.haze.end.rx, opts.haze.end.ry) },
    uFloorColor: { value: opts.haze.floor.color.clone() },
    uFloorHeight: { value: opts.haze.floor.height },
    uFar: { value: new Vector4(opts.haze.far.left, opts.haze.far.right, opts.haze.far.ragged, opts.haze.far.shade) },
    uFarRim: { value: opts.haze.far.rim },
  };
  const material = additiveKeepAlpha(
    new ShaderMaterial({
      name: "CanyonAir",
      uniforms,
      vertexShader: AIR_VERTEX,
      fragmentShader: AIR_FRAGMENT,
      fog: false,
    }),
  );
  const mesh = new Mesh(geometry, material);
  mesh.name = "canyon_air";
  mesh.frustumCulled = false;
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  // the far haze right after the engine background; the beams blend additively (order-free)
  mesh.renderOrder = -1e8;
  return { mesh, uniforms };
}

// ---------------------------------------------------------------------------
// Motes
// ---------------------------------------------------------------------------

export interface MoteUniforms {
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
  uOpacity: IUniform<number>;
  uColor: IUniform<Color>;
  /** Beam frame: top, axis (unit), two unit sides; length, radius of the mote volume. */
  uTop: IUniform<Vector3>;
  uAxis: IUniform<Vector3>;
  uSideA: IUniform<Vector3>;
  uSideB: IUniform<Vector3>;
  uLength: IUniform<number>;
  uRadius: IUniform<number>;
  /** Beam half widths at the top / far end (m): where the motes catch the light. */
  uWidth: IUniform<Vector2>;
  /** Brightness of a mote outside the beam (× its lit brightness). */
  uDim: IUniform<number>;
  /** Drift down the beam (m/s, every mote 0.5–1.5 × this). */
  uFall: IUniform<number>;
  /** Fade-in distance from the lens (m): [hidden, full]. */
  uNearFade: IUniform<Vector2>;
}

export interface Motes {
  points: Points<BufferGeometry, ShaderMaterial>;
  uniforms: MoteUniforms;
  count: number;
}

export interface MoteOptions {
  count: number;
  beam: BeamSpec;
  /** Radius (m) of the mote volume around the beam axis (beyond the lit width they are dim). */
  radius: number;
  facing: Vector3;
  /** Linear colour of a speck in focus at brightness 1. */
  color: Color;
  fall: number;
  dim: number;
}

const MOTE_VERTEX = /* glsl */ `
attribute vec4 aSeed;   // x: size (m), y: phase 0–1, z: speed 0–1, w: brightness
uniform float uTime;
uniform float uScale;
uniform float uMinPx;
uniform float uMaxPx;
uniform float uFocus;
uniform float uCocNear;
uniform float uCocFar;
uniform float uOpacity;
uniform vec3 uColor;
uniform vec3 uTop;
uniform vec3 uAxis;
uniform vec3 uSideA;
uniform vec3 uSideB;
uniform float uLength;
uniform float uRadius;
uniform vec2 uWidth;
uniform float uDim;
uniform float uFall;
uniform vec2 uNearFade;
varying vec3 vColor;
varying float vSharp;

void main() {
  // position = (along, a, b) in the beam's frame, along in 0–1, a / b in −1…1
  float ph = aSeed.y * 6.2831853;
  float sp = 0.5 + aSeed.z;
  float along = fract(position.x + uTime * uFall * sp / uLength);
  float a = position.y + 0.08 * sin(uTime * (0.07 + 0.06 * aSeed.z) + ph);
  float b = position.z + 0.08 * cos(uTime * (0.06 + 0.05 * aSeed.w) + ph * 1.3);
  vec3 p = uTop + uAxis * (along * uLength) + uSideA * (a * uRadius) + uSideB * (b * uRadius);
  // a small sway of its own (a few cm)
  p += vec3(0.03 * sin(uTime * (0.11 + 0.07 * aSeed.w) + ph * 2.1), 0.02 * sin(uTime * (0.09 + 0.05 * aSeed.y) + ph), 0.0);

  // lit where the beam is: radius against the beam's width at that point, fading at its ends
  float w = mix(uWidth.x, uWidth.y, along);
  float r = length(vec2(a, b)) * uRadius / w;
  float lit = exp(-r * r * 1.6) * smoothstep(0.0, 0.12, along) * (1.0 - smoothstep(0.6, 1.0, along));
  // the wrap from the far end back to the top never pops: every mote fades out at both ends
  float ends = smoothstep(0.0, 0.06, along) * (1.0 - smoothstep(0.92, 1.0, along));
  float bright = (uDim + (1.0 - uDim) * lit) * ends;

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
  float tw = 0.75 + 0.25 * sin(uTime * (0.4 + 0.7 * aSeed.z) + ph * 3.0);
  float near = smoothstep(uNearFade.x, uNearFade.y, z);
  gl_PointSize = S;
  vColor = uColor * (aSeed.w * bright * tw * near * uOpacity * energy / max(area, 1e-4));
  vSharp = sharp;
}
`;

const MOTE_FRAGMENT = /* glsl */ `
varying vec3 vColor;
varying float vSharp;
void main() {
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(c, c);
  if (r2 > 1.0) discard;
  float core = exp(-r2 * 6.0);
  float disc = (1.0 - smoothstep(0.6, 1.0, r2)) * (0.75 + 0.35 * r2);
  gl_FragColor = vec4(vColor * mix(disc, core, vSharp), 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

/** Most specks faint, a few brighter ones (clamped tail: no single lamp-like speck). */
function moteBrightness(rng: Rng): number {
  const b = Math.exp(0.7 * Math.max(-2.5, Math.min(1.4, rng.normal())));
  return rng.chance(0.06) ? b * rng.range(1.6, 2.4) : b;
}

export function buildMotes(rng: Rng, opts: MoteOptions): Motes {
  const n = Math.max(1, Math.round(opts.count));
  const pos = new Float32Array(n * 3);
  const seed = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    // uniform in a disc of radius 1 around the axis, uniform along it
    const ang = rng.range(0, Math.PI * 2);
    const rad = Math.sqrt(rng.next());
    pos[i * 3] = rng.next();
    pos[i * 3 + 1] = Math.cos(ang) * rad;
    pos[i * 3 + 2] = Math.sin(ang) * rad;
    // 0.8–2 mm specks, a few larger flakes
    seed[i * 4] = rng.chance(0.12) ? rng.range(0.0025, 0.004) : rng.range(0.0008, 0.002);
    seed[i * 4 + 1] = rng.next();
    seed[i * 4 + 2] = rng.next();
    seed[i * 4 + 3] = moteBrightness(rng);
  }
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new BufferAttribute(pos, 3));
  geometry.setAttribute("aSeed", new BufferAttribute(seed, 4));
  geometry.name = "canyon_motes";

  const axis = opts.beam.dir.clone().normalize();
  const sideA = new Vector3().crossVectors(axis, opts.facing).normalize();
  const sideB = new Vector3().crossVectors(sideA, axis).normalize();
  const uniforms: MoteUniforms = {
    uTime: { value: 0 },
    uScale: { value: 1600 },
    uMinPx: { value: 1.6 },
    uMaxPx: { value: 48 },
    uFocus: { value: 4.4 },
    uCocNear: { value: 12 },
    uCocFar: { value: 1.5 },
    uOpacity: { value: 1 },
    uColor: { value: opts.color.clone() },
    uTop: { value: opts.beam.top.clone() },
    uAxis: { value: axis },
    uSideA: { value: sideA },
    uSideB: { value: sideB },
    uLength: { value: opts.beam.length },
    uRadius: { value: opts.radius },
    uWidth: { value: new Vector2(opts.beam.width[0], opts.beam.width[1]) },
    uDim: { value: opts.dim },
    uFall: { value: opts.fall },
    uNearFade: { value: new Vector2(0.6, 1.4) },
  };
  const material = additiveKeepAlpha(
    new ShaderMaterial({
      name: "CanyonMotes",
      uniforms,
      vertexShader: MOTE_VERTEX,
      fragmentShader: MOTE_FRAGMENT,
      fog: false,
    }),
  );
  const points = new Points(geometry, material);
  points.name = "canyon_motes";
  points.frustumCulled = false;
  points.castShadow = false;
  points.receiveShadow = false;
  // after the plants (transparent list)
  points.renderOrder = 10;
  return { points, uniforms, count: n };
}
