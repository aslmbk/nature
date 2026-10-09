/**
 * Cooperative time slicing for long builds (vegetation scatter and instance writing).
 *
 * A build step is a generator that `yield`s at cheap checkpoints and returns its result.
 * `runSync` runs it to the end in one go (the checkpoints cost next to nothing): the
 * synchronous API. `runSliced` runs the very same generator a stretch at a time, so the
 * current episode keeps rendering while another set prepares. Both execute identical code
 * in identical order — same random draws, same data; only where the browser gets a turn
 * differs.
 *
 * Frame pacing: while the engine's frame loop drives the slices (`attachFramePacing`), build
 * work runs only in one task per frame, posted by the frame loop after it has drawn, for at
 * most that frame's budget (the engine's choice: a few ms while a picture is on screen, more
 * while only the background shows). A frame is rendered between any two stretches of build
 * work. `yieldToBrowser()` then resumes after the next frame. With several builds queued the
 * newest goes first (after a jump that is the set the story landed on; sets merely passed
 * on the way wait), one build per frame.
 *
 * Cancelling: sliced work belongs to an owner (any object; the engine uses the set's
 * `SceneContext`), given to `runSliced(steps, ms, owner)` or learnt from inside a running
 * stretch through `noteSliceOwner(owner)` (the engine's context calls it from `rng()`,
 * `reportInstances()` and its `seed` getter, so a scene's own `runSliced(steps)` is tied to
 * its set as soon as its steps touch the context). `cancelSlices(owner)` stops the owner's
 * sliced work at its next slice boundary: its queued jobs reject with `SliceCancelled` (a
 * dropped generator is closed: its `finally` blocks run), a later `runSliced` for that owner
 * rejects at once, and `noteSliceOwner(owner)` throws from then on — so the build's other
 * code unwinds at its next touch of the context too (in a stretch of a job not yet tied to
 * it, or between awaits); loads it had already asked for finish in the asset registry and
 * stay cached. Whoever started the build (the director) disposes what the partial build had
 * created.
 *
 * Without a driver (no engine running) `runSliced` hands the main thread back with a plain
 * macrotask (MessageChannel) whenever a slice has run for `sliceMs`. That alone does not
 * guarantee a frame in between: Chromium runs such tasks back to back without rendering —
 * measured: about a dozen 8 ms slices in a row, frames of 100–117 ms while a set built.
 * (`scheduler.yield()` is worse: its continuations run ahead of rendering.)
 *
 * Rule for step code: no module-level scratch state may be live across a checkpoint
 * (another build may run while this one waits). Checkpoints sit between loop iterations
 * that start from fresh values; state that spans iterations lives in the generator.
 */

/** A pausable build step returning `T` (`yield` = checkpoint). */
export type Steps<T> = Generator<void, T, void>;

/** The build was cancelled (its owner, e.g. a scene set, is no longer wanted). */
export class SliceCancelled extends Error {
  constructor() {
    super("build cancelled: the set is no longer wanted");
    this.name = "SliceCancelled";
  }
}

/** Run `steps` to the end synchronously. */
export function runSync<T>(steps: Steps<T>): T {
  for (;;) {
    const r = steps.next();
    if (r.done) return r.value;
  }
}

/**
 * Run `steps` a stretch at a time: paced by the engine's frame loop when it drives the
 * slices (its per-frame budget), else yielding a macrotask whenever a slice has taken
 * `sliceMs` (default 8 ms). `owner` (optional): `cancelSlices(owner)` stops it.
 */
export function runSliced<T>(steps: Steps<T>, sliceMs = 8, owner: object | null = null): Promise<T> {
  if (owner && cancelled.has(owner)) {
    close(steps);
    return Promise.reject(new SliceCancelled());
  }
  if (drivers === 0) return runFree(steps, sliceMs, owner);
  return new Promise<T>((resolve, reject) => {
    jobs.push({ steps, owner, resolve: resolve as (value: unknown) => void, reject });
  });
}

/**
 * Let the browser run (input, rendering, other tasks). Paced: resumes after the next frame
 * was drawn; otherwise a macrotask (MessageChannel, else setTimeout).
 */
export function yieldToBrowser(): Promise<void> {
  if (drivers === 0) return macrotask();
  return new Promise((resolve) => waiting.push(resolve));
}

/**
 * Stop the sliced work of `owner` at its next slice boundary (see the header). Idempotent;
 * an owner stays cancelled (use a fresh owner object per build).
 */
export function cancelSlices(owner: object): void {
  if (cancelled.has(owner)) return;
  cancelled.add(owner);
  for (let i = jobs.length - 1; i >= 0; i--) {
    const job = jobs[i];
    if (job.owner !== owner) continue;
    jobs.splice(i, 1);
    dropJob(job);
  }
}

/** `owner` was cancelled. */
export function isSliceCancelled(owner: object): boolean {
  return cancelled.has(owner);
}

/**
 * Build code touching `owner` (the engine's SceneContext does it from `rng()`,
 * `reportInstances()` and `seed`): throws `SliceCancelled` once `owner` is cancelled; inside a
 * running stretch of a job started without an owner, ties that job to `owner`.
 */
export function noteSliceOwner(owner: object): void {
  if (cancelled.has(owner)) throw new SliceCancelled();
  if (running && running.owner === null) running.owner = owner;
}

/** Queued work and the jobs cancelled so far (debug / stats). */
export function sliceStats(): { jobs: number; waiting: number; cancelledJobs: number } {
  return { jobs: jobs.length, waiting: waiting.length, cancelledJobs: counts.cancelledJobs };
}

/** The engine's hold on the slice scheduler (one per running engine). */
export interface FramePacing {
  /**
   * Once per frame, after the frame was drawn: let up to `budgetMs` of build work run, in a
   * task of its own that starts after this frame has been rendered. A build step is never
   * cut, so the last one may overrun the budget by its own length.
   */
  frame(budgetMs: number): void;
  /** Build work is queued (a stretch or a resume is waiting for a frame). */
  readonly busy: boolean;
  /** Stop driving: queued work continues on macrotask yields. */
  detach(): void;
}

/** Drive the slices from a frame loop (the engine, at start); `detach()` at dispose. */
export function attachFramePacing(): FramePacing {
  drivers++;
  let attached = true;
  return {
    frame(budgetMs: number): void {
      if (!attached || (jobs.length === 0 && waiting.length === 0)) return;
      pumpBudget = Math.max(0, budgetMs);
      if (pumpPosted) return;
      pumpPosted = true;
      postPump();
    },
    get busy(): boolean {
      return jobs.length > 0 || waiting.length > 0;
    },
    detach(): void {
      if (!attached) return;
      attached = false;
      drivers--;
      if (drivers === 0) releaseQueued();
    },
  };
}

interface SliceJob {
  steps: Steps<unknown>;
  /** Set by `runSliced(…, owner)` or by the first `noteSliceOwner` from inside a stretch. */
  owner: object | null;
  resolve(value: unknown): void;
  reject(error: unknown): void;
}

/** Frame loops driving the slices (normally 0 or 1). */
let drivers = 0;
/** Builds waiting for their next stretch, oldest first (the newest runs first). */
const jobs: SliceJob[] = [];
/** `yieldToBrowser()` calls waiting for a frame. */
const waiting: (() => void)[] = [];
/** Cancelled owners (weak: an owner's entry goes with it). */
const cancelled = new WeakSet<object>();
const counts = { cancelledJobs: 0 };
/** The job whose steps run right now (for `noteSliceOwner`). */
let running: { owner: object | null } | null = null;
let pumpPosted = false;
let pumpBudget = 0;
/** The last frame's task resumed a yield: the next one goes to the slices first. */
let lastResumed = false;

/**
 * The frame's build task: either resume one `yieldToBrowser()` (its code runs right after
 * this task) or run the newest build's steps until the budget is used, the job finishes (its
 * caller goes on right after this task) or throws. One job per frame. A step is never cut:
 * the loop stops before a step that would probably overrun the budget (the last step's
 * length as the guess), the first step of a frame always runs.
 */
function runFrameSlices(): void {
  pumpPosted = false;
  if (waiting.length > 0 && (jobs.length === 0 || !lastResumed)) {
    lastResumed = true;
    (waiting.shift() as () => void)();
    return;
  }
  lastResumed = false;
  const job = jobs[jobs.length - 1];
  if (!job) return;
  const start = now();
  running = job;
  try {
    for (;;) {
      const s0 = now();
      let r: IteratorResult<void, unknown>;
      try {
        r = job.steps.next();
      } catch (err) {
        jobs.splice(jobs.indexOf(job), 1);
        if (err instanceof SliceCancelled) counts.cancelledJobs++;
        job.reject(err);
        return;
      }
      if (r.done) {
        jobs.splice(jobs.indexOf(job), 1);
        job.resolve(r.value);
        return;
      }
      const t = now();
      if (t - start + (t - s0) >= pumpBudget) return;
    }
  } finally {
    running = null;
  }
}

/** A cancelled job: close its generator (`finally` blocks run) and reject its promise. */
function dropJob(job: SliceJob): void {
  close(job.steps);
  counts.cancelledJobs++;
  job.reject(new SliceCancelled());
}

function close(steps: Steps<unknown>): void {
  try {
    steps.return(undefined as never);
  } catch {
    // a `finally` block threw: the build is dropped either way
  }
}

/** No frame loop drives the slices any more: let queued builds finish on macrotask yields. */
function releaseQueued(): void {
  for (const job of jobs.splice(0)) {
    runFree(job.steps, 8, job.owner).then(job.resolve, job.reject);
  }
  for (const resume of waiting.splice(0)) void macrotask().then(resume);
}

/** `runSliced` without a frame loop: slices of `sliceMs` with a macrotask in between. */
async function runFree<T>(steps: Steps<T>, sliceMs: number, owner: object | null): Promise<T> {
  const job = { owner };
  let start = now();
  for (;;) {
    let r: IteratorResult<void, T>;
    running = job;
    try {
      r = steps.next();
    } finally {
      running = null;
    }
    if (r.done) return r.value;
    if (now() - start >= sliceMs) {
      await macrotask();
      if (job.owner && cancelled.has(job.owner)) {
        close(steps);
        counts.cancelledJobs++;
        throw new SliceCancelled();
      }
      start = now();
    }
  }
}

function macrotask(): Promise<void> {
  return new Promise((resolve) => postTask(resolve));
}

/** One channel for the frame's build task (its handler shows up by name in profiles / LoAF). */
let pumpChannel: MessageChannel | null = null;

function postPump(): void {
  if (typeof MessageChannel === "undefined") {
    setTimeout(runFrameSlices, 0);
    return;
  }
  if (!pumpChannel) {
    pumpChannel = new MessageChannel();
    pumpChannel.port1.onmessage = runFrameSlices;
  }
  pumpChannel.port2.postMessage(null);
}

/** Run `fn` in a task of its own (MessageChannel: no 4 ms clamp; else setTimeout). */
function postTask(fn: () => void): void {
  if (typeof MessageChannel !== "undefined") {
    const channel = new MessageChannel();
    channel.port1.onmessage = () => {
      channel.port1.close();
      fn();
    };
    channel.port2.postMessage(null);
    return;
  }
  setTimeout(fn, 0);
}

function now(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}
