/**
 * Frame timing over the last 120 frames (interval between frames and CPU time spent
 * in the frame function) plus renderer counters summed over every pass of a frame
 * (`renderer.info.autoReset = false`, reset once per frame by the engine).
 * GPU time is not measured.
 */
import type { WebGLRenderer } from "three";

const WINDOW = 120;

export interface FrameStatsSnapshot {
  samples: number;
  frameMsAvg: number;
  frameMsP95: number;
  cpuMsAvg: number;
  cpuMsP95: number;
  drawCalls: number;
  triangles: number;
  points: number;
  lines: number;
}

function avg(values: Float64Array, n: number): number {
  if (n === 0) return 0;
  let s = 0;
  for (let i = 0; i < n; i++) s += values[i];
  return s / n;
}

function p95(values: Float64Array, n: number): number {
  if (n === 0) return 0;
  const sorted = Array.from(values.subarray(0, n)).sort((a, b) => a - b);
  return sorted[Math.min(n - 1, Math.floor(0.95 * (n - 1) + 0.5))];
}

export class FrameStats {
  private readonly intervals = new Float64Array(WINDOW);
  private readonly cpu = new Float64Array(WINDOW);
  private intervalCount = 0;
  private intervalIdx = 0;
  private cpuCount = 0;
  private cpuIdx = 0;
  private lastStart = -1;
  drawCalls = 0;
  triangles = 0;
  points = 0;
  lines = 0;

  frameStart(now: number): void {
    if (this.lastStart >= 0) {
      this.intervals[this.intervalIdx] = now - this.lastStart;
      this.intervalIdx = (this.intervalIdx + 1) % WINDOW;
      this.intervalCount = Math.min(WINDOW, this.intervalCount + 1);
    }
    this.lastStart = now;
  }

  /** Forget the interval across a pause (hidden tab, context loss). */
  resetInterval(): void {
    this.lastStart = -1;
  }

  frameEnd(cpuMs: number, renderer: WebGLRenderer): void {
    this.cpu[this.cpuIdx] = cpuMs;
    this.cpuIdx = (this.cpuIdx + 1) % WINDOW;
    this.cpuCount = Math.min(WINDOW, this.cpuCount + 1);
    const r = renderer.info.render;
    this.drawCalls = r.calls;
    this.triangles = r.triangles;
    this.points = r.points;
    this.lines = r.lines;
  }

  snapshot(): FrameStatsSnapshot {
    return {
      samples: this.intervalCount,
      frameMsAvg: avg(this.intervals, this.intervalCount),
      frameMsP95: p95(this.intervals, this.intervalCount),
      cpuMsAvg: avg(this.cpu, this.cpuCount),
      cpuMsP95: p95(this.cpu, this.cpuCount),
      drawCalls: this.drawCalls,
      triangles: this.triangles,
      points: this.points,
      lines: this.lines,
    };
  }
}
