/**
 * Oracle scene set: dimensions and the story timeline as pure functions of video
 * time t (seconds). Measured from frames 07–09 and the 21.0–31.5 s clip sheet:
 *
 *   20.3–21.5  dark crossfade in from the canyon; a loose swirl of particle sheets
 *   21.3–22.7  the swirl converges into the orb, the orbit draws itself, shell fades in
 *   22.7–24.5  steady orb, tag rides the orbit
 *   24.5–25.7  the camera pans down (page-scroll feel): the orbit stays where it is and
 *              leaves through the top, the orb is held near the lower centre
 *   24.9–25.9  the twisted loop appears above the orb (revealed from the top down)
 *   25.3–31    particle streams sweep in from the lower left / right and rise
 *   25.7–27.4  the orb settles low (screen y 0.63 → 0.75)
 *   30.4–31.4  ragged wipe to the branch scene (director); the orb sinks further
 */
import { clamp, smoothstep } from "../../core/math";

/** Orb shell radius (m). Everything orb-related is authored in units of this radius. */
export const ORB_RADIUS = 0.5;
/** Camera distance to the orb plane (m) in the steady poses. */
export const CAM_DISTANCE = 5.0;
export const CAM_FOV = 35;
const TAN_HALF_FOV = Math.tan((CAM_FOV * Math.PI) / 360);
/** Visible frame height (m) at distance d. */
export const frameHeightAt = (d: number): number => 2 * d * TAN_HALF_FOV;

/** Camera x offset that puts the orb at x ≈ 0.508 of the frame (frames 08 / 09). */
export const CAM_X = -0.036;
/** Camera height in the oracle pose: orb centre (world origin) at y = 0.55 of the frame. */
export const CAM_Y_ORACLE = 0.05 * frameHeightAt(CAM_DISTANCE);

type Key = readonly [number, number];

/**
 * Monotone cubic (PCHIP) interpolation through keys sorted by x; flat at both ends,
 * clamped outside. C1 continuous, never overshoots — no stop-and-go between keys.
 */
export function pchip(keys: readonly Key[], x: number): number {
  const n = keys.length;
  if (x <= keys[0][0]) return keys[0][1];
  if (x >= keys[n - 1][0]) return keys[n - 1][1];
  let i = 0;
  while (i < n - 2 && x > keys[i + 1][0]) i++;
  const slope = (j: number) => (keys[j + 1][1] - keys[j][1]) / (keys[j + 1][0] - keys[j][0]);
  const tangent = (j: number) => {
    if (j <= 0 || j >= n - 1) return 0;
    const d0 = slope(j - 1);
    const d1 = slope(j);
    if (d0 * d1 <= 0) return 0;
    const h0 = keys[j][0] - keys[j - 1][0];
    const h1 = keys[j + 1][0] - keys[j][0];
    const w1 = 2 * h1 + h0;
    const w2 = h1 + 2 * h0;
    return (w1 + w2) / (w1 / d0 + w2 / d1);
  };
  const x0 = keys[i][0];
  const h = keys[i + 1][0] - x0;
  const s = (x - x0) / h;
  const s2 = s * s;
  const s3 = s2 * s;
  const m0 = tangent(i) * h;
  const m1 = tangent(i + 1) * h;
  return (2 * s3 - 3 * s2 + 1) * keys[i][1] + (s3 - 2 * s2 + s) * m0 + (-2 * s3 + 3 * s2) * keys[i + 1][1] + (s3 - s2) * m1;
}

/** Screen height of the orb centre (0 = top, 1 = bottom) over video time (clip sheet 08). */
const ORB_SCREEN_Y: readonly Key[] = [
  [20.3, 0.6],
  [21.6, 0.6],
  [22.5, 0.553],
  [23.1, 0.55],
  [24.5, 0.55],
  [25.2, 0.585],
  [25.9, 0.63],
  [26.6, 0.67],
  [27.4, 0.745],
  [28.5, 0.765],
  [30.1, 0.78],
  [30.4, 0.785],
  [30.6, 0.81],
  [30.8, 0.87],
  [31.0, 0.93],
  [31.4, 1.0],
];

export function orbScreenY(t: number): number {
  return pchip(ORB_SCREEN_Y, t);
}

/** Camera keys (video s): height of the camera (it looks horizontally) and distance. */
export const CAMERA_KEYS: readonly { t: number; y: number; z: number }[] = [
  { t: 20.3, y: CAM_Y_ORACLE, z: 5.55 },
  { t: 22.8, y: CAM_Y_ORACLE, z: CAM_DISTANCE },
  { t: 24.5, y: CAM_Y_ORACLE, z: CAM_DISTANCE },
  { t: 25.7, y: CAM_Y_ORACLE - 2.1, z: CAM_DISTANCE },
  { t: 27.6, y: CAM_Y_ORACLE - 2.5, z: CAM_DISTANCE },
  { t: 30.4, y: CAM_Y_ORACLE - 2.7, z: CAM_DISTANCE },
  { t: 31.4, y: CAM_Y_ORACLE - 2.8, z: CAM_DISTANCE },
];

/** Assembly of the orb from the swirl: 0 → 1 (per-particle delays inside). */
export const assembleProgress = (t: number): number => clamp((t - 21.45) / 1.3, 0, 1);
/** Visibility of the swirl itself while the canyon → oracle dip lifts. */
export const swirlOpacity = (t: number): number => smoothstep(20.75, 21.35, t);
/** Rotation (rad) of the swirl sheets around the vertical axis (story driven; authored pose at 21.5 s). */
export const swirlAngle = (t: number): number => (t - 21.5) * 0.8;
/** Shell / nodes / filament fade-in at the end of the assembly. */
export const shellOpacity = (t: number): number => smoothstep(22.0, 22.85, t);
/** Orbit draw-on 0 → 1. */
export const orbitDraw = (t: number): number => smoothstep(21.05, 21.95, t);
/** The settled dots light up once the orb has formed. */
export const orbLit = (t: number): number => smoothstep(22.1, 23.0, t);
/** The orbit is carried with the orb until the camera pans away, then stays in the world. */
export const ORBIT_RELEASE_T = 24.5;
/** The loop grows up out of the orb (0 → 1) while the dark cable inside unwinds... */
export const loopReveal = (t: number): number => smoothstep(24.6, 25.4, t);
/**
 * ...first as a dark shape, then its glassy highlights come up (clip 08: dark at 25.2 s,
 * lit at 25.9 s); they fade again before the wipe (motion/02: dim strands at 30.8 s).
 */
export const loopLight = (t: number): number => smoothstep(25.1, 26.0, t) * (1 - 0.75 * smoothstep(29.6, 30.8, t));
/** Streams presence 0 → 1. */
export const streamsOn = (t: number): number => smoothstep(25.3, 26.5, t);
/** Story time origin of the streams disc rotation. */
export const STREAMS_T0 = 25.3;

/**
 * The streams disc relative to the orb centre (orb units). The camera sits 0.5–1.8 orb
 * radii above the orb during the episode, so a disc below that height reads as a floor
 * of particles (25.9 s), at the orb height as a flat band through it (26.6 s), and
 * thick and high as a sparse field over both sides (28–30 s, frame 09).
 */
const STREAM_HEIGHT: readonly Key[] = [
  [25.3, -1.2],
  [25.9, -0.7],
  [26.6, 0.0],
  [27.4, 0.7],
  [28.5, 1.4],
  [30.1, 1.8],
  [30.4, 1.85],
  // the orb sinks out of the frame at the end: the field stays where it was on screen
  [31.4, 3.4],
];
const STREAM_THICKNESS: readonly Key[] = [
  [25.3, 0.25],
  [26.6, 0.5],
  [27.4, 0.7],
  [28.5, 1.25],
  [30.1, 1.6],
];
const STREAM_SPREAD: readonly Key[] = [
  [25.3, 1.3],
  [26.3, 1.0],
  [28.5, 1.05],
  [30.1, 1.18],
];
export const streamHeight = (t: number): number => pchip(STREAM_HEIGHT, t);
export const streamThickness = (t: number): number => pchip(STREAM_THICKNESS, t);
export const streamSpread = (t: number): number => pchip(STREAM_SPREAD, t);
/** The disc loosens into a drifting field. */
export const streamLoose = (t: number): number => smoothstep(26.8, 29.8, t);
/** Fountain of sparks out of the top of the orb: low in the oracle episode, tall in streams. */
export const fountain = (t: number): number => smoothstep(25.0, 26.6, t);
/** Break-up of the orb during the exit wipe (motion/02: intact at 30.6 s, crumbling at 30.8 s). */
export const exitProgress = (t: number): number => smoothstep(30.55, 31.05, t);
