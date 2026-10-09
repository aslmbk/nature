/**
 * `window.__NATURE_READY__`: becomes true once everything the current frame needs is
 * loaded and prepared and at least two frames of that final state have been
 * rendered. Any change that alters the picture (setT, setTime, quality, flags,
 * scroll target not reached) resets it.
 */

declare global {
  interface Window {
    __NATURE_READY__?: boolean;
  }
}

export class CaptureState {
  private settledFrames = 0;
  private isReady = false;

  constructor(private readonly requiredFrames = 2) {
    window.__NATURE_READY__ = false;
  }

  get ready(): boolean {
    return this.isReady;
  }

  invalidate(): void {
    this.settledFrames = 0;
    this.set(false);
  }

  /** Call after a frame has been submitted. */
  frameRendered(settled: boolean): void {
    if (!settled) {
      this.settledFrames = 0;
      if (this.isReady) this.set(false);
      return;
    }
    this.settledFrames++;
    if (this.settledFrames >= this.requiredFrames && !this.isReady) this.set(true);
  }

  private set(value: boolean): void {
    this.isReady = value;
    window.__NATURE_READY__ = value;
  }

  dispose(): void {
    delete window.__NATURE_READY__;
  }
}
