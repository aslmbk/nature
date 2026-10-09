/**
 * Seeded PRNG. Every random decision in the project comes from here:
 * `rngFor(seed, "grove:moss-scatter")` gives an independent, deterministic stream
 * per label, so adding a new consumer never shifts the numbers of another one.
 * Never use Math.random() for anything that affects the picture.
 */

export interface Rng {
  readonly label: string;
  /** Uniform in [0, 1). */
  next(): number;
  range(min: number, max: number): number;
  /** Integer in [min, max] (inclusive). */
  int(min: number, max: number): number;
  pick<T>(items: readonly T[]): T;
  chance(probability: number): boolean;
  /** Gaussian (Box–Muller). */
  normal(mean?: number, deviation?: number): number;
  sign(): 1 | -1;
  /** Child stream, independent of how many numbers this stream has produced. */
  fork(label: string): Rng;
}

/** 128-bit string hash (cyrb128) → four 32-bit seeds. */
export function hash128(input: string): [number, number, number, number] {
  let h1 = 1779033703;
  let h2 = 3144134277;
  let h3 = 1013904242;
  let h4 = 2773480762;
  for (let i = 0; i < input.length; i++) {
    const k = input.charCodeAt(i);
    h1 = h2 ^ Math.imul(h1 ^ k, 597399067);
    h2 = h3 ^ Math.imul(h2 ^ k, 2869860233);
    h3 = h4 ^ Math.imul(h3 ^ k, 951274213);
    h4 = h1 ^ Math.imul(h4 ^ k, 2716044179);
  }
  h1 = Math.imul(h3 ^ (h1 >>> 18), 597399067);
  h2 = Math.imul(h4 ^ (h2 >>> 22), 2869860233);
  h3 = Math.imul(h1 ^ (h3 >>> 17), 951274213);
  h4 = Math.imul(h2 ^ (h4 >>> 19), 2716044179);
  h1 ^= h2 ^ h3 ^ h4;
  h2 ^= h1;
  h3 ^= h1;
  h4 ^= h1;
  return [h1 >>> 0, h2 >>> 0, h3 >>> 0, h4 >>> 0];
}

/** 32-bit string hash. */
export function hashString(input: string): number {
  return hash128(input)[0];
}

/**
 * sfc32 generator: small, fast, good statistical quality, identical in every JS engine.
 *
 * The state is four int32 in an Int32Array; only the output is made unsigned. Every step
 * is arithmetic mod 2^32 (`| 0`, shifts, xor), so the numbers are exactly those of the
 * uint32 formulation (`a >>>= 0` … before each draw). The state lives in a typed array:
 * closure variables beyond the small-integer range (31 bits in Chromium) stay boxed heap
 * numbers, which made the generator ≈ 5× slower per draw in Edge.
 */
function sfc32(a: number, b: number, c: number, d: number): () => number {
  const s = new Int32Array(4);
  s[0] = a | 0;
  s[1] = b | 0;
  s[2] = c | 0;
  s[3] = d | 0;
  return () => {
    const a0 = s[0];
    const b0 = s[1];
    const c0 = s[2];
    const d1 = (s[3] + 1) | 0;
    const t = (((a0 + b0) | 0) + d1) | 0;
    s[0] = b0 ^ (b0 >>> 9);
    s[1] = (c0 + (c0 << 3)) | 0;
    s[2] = (((c0 << 21) | (c0 >>> 11)) + t) | 0;
    s[3] = d1;
    return (t >>> 0) / 4294967296;
  };
}

class SeededRng implements Rng {
  private readonly gen: () => number;
  private spare: number | null = null;

  constructor(
    private readonly seed: number,
    readonly label: string,
  ) {
    const [a, b, c, d] = hash128(`${seed}|${label}`);
    this.gen = sfc32(a, b, c, d);
    // Warm-up: discard the first outputs so similar labels decorrelate fully.
    for (let i = 0; i < 12; i++) this.gen();
  }

  next(): number {
    return this.gen();
  }

  range(min: number, max: number): number {
    return min + (max - min) * this.gen();
  }

  int(min: number, max: number): number {
    return min + Math.floor(this.gen() * (max - min + 1));
  }

  pick<T>(items: readonly T[]): T {
    if (items.length === 0) throw new Error(`rng(${this.label}).pick on an empty list`);
    return items[Math.min(items.length - 1, Math.floor(this.gen() * items.length))];
  }

  chance(probability: number): boolean {
    return this.gen() < probability;
  }

  normal(mean = 0, deviation = 1): number {
    if (this.spare !== null) {
      const v = this.spare;
      this.spare = null;
      return mean + deviation * v;
    }
    let u = 0;
    let v = 0;
    let s = 0;
    do {
      u = this.gen() * 2 - 1;
      v = this.gen() * 2 - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const m = Math.sqrt((-2 * Math.log(s)) / s);
    this.spare = v * m;
    return mean + deviation * u * m;
  }

  sign(): 1 | -1 {
    return this.gen() < 0.5 ? -1 : 1;
  }

  fork(label: string): Rng {
    return new SeededRng(this.seed, `${this.label}/${label}`);
  }
}

/** Independent deterministic stream for (seed, label). */
export function rngFor(seed: number, label: string): Rng {
  return new SeededRng(seed, label);
}
