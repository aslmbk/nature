/**
 * Maps story time t to what has to be rendered: one scene set, or two inside a
 * composite transition window (at most two per frame). Prepares scene sets lazily:
 * the sets of the current views first, then — once those are ready — the neighbour
 * in the direction of travel (preload), then, with nothing else preparing, the resident
 * set (`grove`: hero, About and the logo are the most linked targets). Keeps the current
 * set, its two neighbours and the resident set, disposes the rest; only a quality or seed
 * change (`invalidateAll`) drops the resident set too. Computes transition parameters
 * (with optional per-scene overrides) for the PostStack.
 *
 * A set goes idle → preparing (factory, `prepare()`, COLOR_0 check, `warm` hook: the
 * engine's spread-out shader compile against the real target, texture uploads and
 * off-screen warm-up draw) → ready. A set
 * whose prepare or render throws is marked failed (disposed, logged once) and the
 * others keep rendering; released sets are tried again when they are needed again.
 *
 * A set still preparing is kept while it is on screen, resident, or wanted beyond the moment
 * (started or taken over as the preload, or on screen for `PASSING_MS`) and still a
 * neighbour. Anything else — a set the story only passed during a Home / End smooth scroll
 * or a fling, a preload the story left behind — is released at once: its sliced build work
 * stops at the next slice boundary (`cancelSlices` on the set's context, vegetation/
 * slices.ts), the rest of its prepare unwinds at its next touch of the context, and whatever
 * the partial build created is disposed, so the frame budget goes to the set that is wanted.
 *
 * Reduced motion: composite transitions become plain crossfades; same-set camera
 * moves (hero → arch, oracle → streams, canopy → canopyClose) become a crossfade of
 * the end pose and the start pose instead of a fly-through.
 */
import { Color } from "three";
import {
  SCENE_SET_ORDER,
  TRANSITIONS,
  TRANSITION_TUNING_DEFAULTS,
  VIDEO_DURATION,
  episodeAt,
  episodeConfig,
  episodesOfSet,
  setSpan,
  type EpisodeConfig,
  type TransitionConfig,
} from "./SceneConfig";
import { clamp, ease, lerp, saturate, smoothstep, windowProgress } from "./core/math";
import { prepareSilvaGeometry } from "./rendering/Materials";
import { cancelSlices } from "./vegetation/slices";
import type {
  ActiveTransition,
  FrameState,
  NatureScene,
  SceneContext,
  SceneFactory,
  SceneLocal,
  SceneSetId,
  TransitionParams,
  TransitionTuning,
  ViewRole,
} from "./types";

export interface ViewPlan {
  set: SceneSetId;
  local: SceneLocal;
}

export interface Resolution {
  t: number;
  episode: EpisodeConfig;
  sceneProgress: number;
  globalProgress: number;
  views: ViewPlan[];
  transition: ActiveTransition | null;
  config: TransitionConfig | null;
}

export type SetState = "idle" | "preparing" | "ready" | "failed";

interface SetEntry {
  state: SetState;
  scene: NatureScene | null;
  generation: number;
  error?: unknown;
  /** The context of its prepare: the owner `cancelSlices` stops the build by. */
  ctx: SceneContext | null;
  /** Wanted beyond the moment while preparing: the preload, or on screen ≥ PASSING_MS. */
  kept: boolean;
  /** performance.now(): prepare started; on screen since (NaN: not on screen); released (NaN: not). */
  startedAt: number;
  viewSince: number;
  releasedAt: number;
}

/**
 * How a prepare ended: "ready"; "cancelled" (released while preparing: stopped, partial
 * build disposed); "dropped" (built, but no longer wanted: disposed before its warm-up);
 * "failed".
 */
export type PrepareOutcome = "ready" | "cancelled" | "dropped" | "failed";

export interface DirectorOptions {
  factories: Record<SceneSetId, SceneFactory>;
  contextFor(set: SceneSetId): SceneContext;
  /**
   * Runs after `prepare()` resolved, before the set is marked ready (shader compile
   * against the real render target, off-screen warm-up draw). A throw fails the set.
   * `stale()` turns true once the set was released or invalidated meanwhile: stop then
   * (the director disposes the scene after the promise settles).
   */
  warm?(set: SceneSetId, scene: NatureScene, stale: () => boolean): Promise<void>;
  onReady?(set: SceneSetId, scene: NatureScene): void;
  onRelease?(set: SceneSetId): void;
  onError?(set: SceneSetId, error: unknown): void;
  /**
   * A prepare ended (`ms` since it started; `afterReleaseMs`: from its release to the moment
   * the cancelled build had unwound and was disposed, null unless it was released).
   */
  onPrepareEnd?(set: SceneSetId, outcome: PrepareOutcome, ms: number, afterReleaseMs: number | null): void;
}

/** SceneConfig tuning merged with the defaults, colours parsed once per transition. */
interface ResolvedTuning {
  tun: Required<TransitionTuning>;
  dipColor: Color;
  backA: Color;
}

const setOfEpisode = (id: TransitionConfig["from"]): SceneSetId => episodeConfig(id).set;

function compositeAt(t: number): TransitionConfig | null {
  for (const tr of TRANSITIONS) if (tr.kind === "composite" && t >= tr.start && t < tr.end) return tr;
  return null;
}

function cameraInsideAt(t: number): TransitionConfig | null {
  for (const tr of TRANSITIONS) if (tr.kind === "camera" && t > tr.start && t < tr.end) return tr;
  return null;
}

function cameraOfSetAt(set: SceneSetId, t: number): TransitionConfig | null {
  for (const tr of TRANSITIONS) if (tr.kind === "camera" && t >= tr.start && t <= tr.end && setOfEpisode(tr.from) === set) return tr;
  return null;
}

/**
 * Sets kept once prepared wherever the story is (and prepared at idle): jumps to them (nav
 * links, the logo, Home) show them at once. Released only by `invalidateAll`.
 */
const RESIDENT_SETS: readonly SceneSetId[] = ["grove"];

/**
 * A set still preparing that was on screen for less than this (ms) — the story only passed
 * it: a Home / End smooth scroll, a fling, the way of a jump — is released (its build
 * cancelled) as soon as it is off screen; on screen longer, it is kept like the preload
 * while it stays a neighbour (so scrolling back and forth at a boundary does not restart it).
 */
const PASSING_MS = 400;

/** Dip of the crossfades that replace wipe / slide / over / camera moves under reduced motion. */
const REDUCED_MOTION_DIP = 0.4;

/**
 * The transition window (composite or camera move) that story time t is inside, or that
 * it reaches within `lead` story seconds in the direction of travel; null when t is
 * clear of every window. The engine keeps preloads and shader warm-up out of these.
 */
export function transitionNear(t: number, travel: 1 | -1, lead: number): TransitionConfig | null {
  for (const tr of TRANSITIONS) {
    const from = travel > 0 ? tr.start - lead : tr.start;
    const to = travel > 0 ? tr.end : tr.end + lead;
    if (t >= from && t <= to) return tr;
  }
  return null;
}

/**
 * Copy a scene's override into the reused params object; colours are copied into the
 * params' own Color instances (never adopt the scene's objects: they would be
 * overwritten by the next frame's defaults).
 */
function applyOverride(params: TransitionParams, o: Partial<TransitionParams>): void {
  const target = params as unknown as Record<string, unknown>;
  for (const key in o) {
    if (!Object.prototype.hasOwnProperty.call(o, key)) continue;
    const value = (o as Record<string, unknown>)[key];
    if (value === undefined) continue;
    if (key === "dipColor" || key === "backA") params[key].copy(value as Color);
    else target[key] = value;
  }
}

export class SceneDirector {
  private readonly entries = new Map<SceneSetId, SetEntry>();
  private generation = 0;
  private disposed = false;
  /** Sets the last `sync` wanted (current views + neighbours + resident). */
  private readonly wanted = new Set<SceneSetId>();
  /** Sets on screen in the last `sync`. */
  private readonly viewSets = new Set<SceneSetId>();
  /** Sets whose failure was already logged (log once per set and session). */
  private readonly reported = new Set<SceneSetId>();
  private readonly tunings = new Map<TransitionConfig, ResolvedTuning>();
  private readonly params: TransitionParams = {
    mode: "mix",
    k: 0,
    dip: 0,
    dipColor: new Color(),
    edge: 0,
    raggedness: 0,
    softness: 0,
    noiseScale: 0,
    direction: 1,
    edgeDarken: 0,
    offset: 0,
    seamSoftness: 0,
    seamDarkness: 0,
    seamBlurPx: 0,
    seamWave: 0,
    follow: 1,
    reveal: 0,
    opacityA: 1,
    backA: new Color(),
  };

  constructor(private readonly opts: DirectorOptions) {}

  // -------------------------------------------------------------------------
  // Pure timeline resolution
  // -------------------------------------------------------------------------

  resolve(tIn: number, reducedMotion: boolean): Resolution {
    const t = clamp(Number.isNaN(tIn) ? 0 : tIn, 0, VIDEO_DURATION);
    const episode = episodeAt(t);
    const sceneProgress = saturate((t - episode.start) / (episode.end - episode.start));
    const base = { t, episode, sceneProgress, globalProgress: t / VIDEO_DURATION };

    const composite = compositeAt(t);
    if (composite) {
      const linear = windowProgress(t, composite.start, composite.end);
      const mode = reducedMotion ? "mix" : composite.mode;
      const transition: ActiveTransition = {
        id: composite.id,
        from: composite.from,
        to: composite.to,
        fromSet: setOfEpisode(composite.from),
        toSet: setOfEpisode(composite.to),
        start: composite.start,
        end: composite.end,
        mode,
        linear,
        k: ease(composite.ease, linear),
        reduced: reducedMotion && composite.mode !== "mix",
      };
      return {
        ...base,
        transition,
        config: composite,
        views: [
          { set: transition.fromSet, local: this.localFor(transition.fromSet, t, "outgoing", transition, mode === "over") },
          { set: transition.toSet, local: this.localFor(transition.toSet, t, "incoming", transition, false) },
        ],
      };
    }

    if (reducedMotion) {
      const cam = cameraInsideAt(t);
      if (cam) {
        const set = setOfEpisode(cam.from);
        const linear = windowProgress(t, cam.start, cam.end);
        const transition: ActiveTransition = {
          id: `${cam.id}:reduced`,
          from: cam.from,
          to: cam.to,
          fromSet: set,
          toSet: set,
          start: cam.start,
          end: cam.end,
          mode: "mix",
          linear,
          k: ease("smooth", linear),
          reduced: true,
        };
        return {
          ...base,
          transition,
          config: cam,
          views: [
            { set, local: this.localFor(set, cam.start, "outgoing", transition, false) },
            { set, local: this.localFor(set, cam.end, "incoming", transition, false) },
          ],
        };
      }
    }

    return { ...base, transition: null, config: null, views: [{ set: episode.set, local: this.localFor(episode.set, t, "solo", null, false) }] };
  }

  /** SceneLocal of `set` at video time t. */
  localFor(set: SceneSetId, t: number, role: ViewRole, transition: ActiveTransition | null, transparent: boolean): SceneLocal {
    const ep = episodeAt(t, episodesOfSet(set));
    const raw = (t - ep.start) / (ep.end - ep.start);
    const span = setSpan(set);
    const cam = cameraOfSetAt(set, t);
    return {
      set,
      t,
      episode: ep.id,
      progress: saturate(raw),
      rawProgress: raw,
      setProgress: saturate((t - span.start) / (span.end - span.start)),
      role,
      transition,
      episodeBlend: cam ? { from: cam.from, to: cam.to, k: ease(cam.ease, windowProgress(t, cam.start, cam.end)) } : null,
      transparentBackground: transparent,
    };
  }

  private tuningOf(cfg: TransitionConfig): ResolvedTuning {
    let r = this.tunings.get(cfg);
    if (!r) {
      const tun: Required<TransitionTuning> = { ...TRANSITION_TUNING_DEFAULTS, ...cfg.tuning };
      r = { tun, dipColor: new Color(tun.dipColor), backA: new Color(tun.backA) };
      this.tunings.set(cfg, r);
    }
    return r;
  }

  /**
   * Composite parameters for the active transition: SceneConfig tuning, derived
   * curves (edge / offset / reveal / fade from k), then scene overrides (outgoing
   * scene first, incoming scene wins). The returned object is reused every frame.
   * An override that throws is reported through `onError(index, error)` (index into
   * `scenes`) and skipped.
   */
  transitionParams(
    res: Resolution,
    frame: FrameState,
    scenes: readonly (NatureScene | null)[],
    onError?: (index: number, error: unknown) => void,
  ): TransitionParams | null {
    const tr = res.transition;
    const cfg = res.config;
    if (!tr || !cfg) return null;
    const { tun, dipColor, backA } = this.tuningOf(cfg);
    const k = tr.k;
    const margin = tun.raggedness * 1.45 + tun.softness * 2;
    const params = this.params;
    params.mode = tr.mode;
    params.k = k;
    // a reduced-motion crossfade (standing in for a wipe / slide / over / camera move)
    // dips a little through the dip colour: less double exposure halfway
    params.dip = tr.reduced ? REDUCED_MOTION_DIP : tr.mode === "mix" ? tun.dip : 0;
    params.dipColor.copy(dipColor);
    params.edge = lerp(-margin, 1 + margin, k);
    params.raggedness = tun.raggedness;
    params.softness = tun.softness;
    params.noiseScale = tun.noiseScale;
    params.direction = tun.direction;
    params.edgeDarken = tun.edgeDarken;
    params.offset = k;
    params.seamSoftness = tun.seamSoftness;
    params.seamDarkness = tun.seamDarkness;
    params.seamBlurPx = tun.seamBlurPx;
    params.seamWave = tun.seamWave;
    params.follow = tun.follow;
    params.reveal = smoothstep(tun.revealSpan[0], tun.revealSpan[1], k);
    params.opacityA = 1 - smoothstep(tun.fadeSpan[0], tun.fadeSpan[1], k);
    params.backA.copy(backA);
    const mode = params.mode;
    for (let i = 0; i < scenes.length; i++) {
      const scene = scenes[i];
      if (!scene?.getTransitionOverride) continue;
      try {
        const o = scene.getTransitionOverride(tr, frame);
        if (o) applyOverride(params, o);
      } catch (error) {
        onError?.(i, error);
      }
    }
    params.mode = mode;
    return params;
  }

  /**
   * The transition of `res` as it stands at story time `t` (clamped to its window): same
   * window, mode and curve, progress at `t`. The engine's fade-in runs a composite from the
   * side where the picture that fades in has no weight to the real time.
   */
  transitionAt(res: Resolution, t: number): ActiveTransition | null {
    const tr = res.transition;
    const cfg = res.config;
    if (!tr || !cfg) return null;
    const linear = windowProgress(t, tr.start, tr.end);
    // a camera window is a transition only as a reduced-motion crossfade, eased "smooth" (resolve)
    return { ...tr, linear, k: ease(cfg.kind === "camera" ? "smooth" : cfg.ease, linear) };
  }

  // -------------------------------------------------------------------------
  // Lifecycle of scene sets
  // -------------------------------------------------------------------------

  state(set: SceneSetId): SetState {
    return this.entries.get(set)?.state ?? "idle";
  }

  states(): Partial<Record<SceneSetId, SetState>> {
    const out: Partial<Record<SceneSetId, SetState>> = {};
    for (const [id, e] of this.entries) out[id] = e.state;
    return out;
  }

  scene(set: SceneSetId): NatureScene | null {
    const e = this.entries.get(set);
    return e && e.state === "ready" ? e.scene : null;
  }

  readyScenes(): NatureScene[] {
    const out: NatureScene[] = [];
    for (const e of this.entries.values()) if (e.state === "ready" && e.scene) out.push(e.scene);
    return out;
  }

  /** Every view of `res` is ready (or failed for good). */
  settled(res: Resolution): boolean {
    for (const v of res.views) {
      const s = this.state(v.set);
      if (s !== "ready" && s !== "failed") return false;
    }
    return true;
  }

  /**
   * Start what `res` needs; once that is ready, preload the neighbour in the direction
   * of travel (+1 forward, −1 backward), then — nothing else preparing — the resident
   * sets; release sets more than one step away from the current one (never a resident
   * set). `preload` false holds every preload back (capture mode); `allow` holds back a
   * single one until it returns true (the engine: not while a transition that does not
   * show that set plays or is about to start, so a set's build never starts in the middle
   * of one). Neighbours and resident sets that exist are kept either way; a set still
   * preparing only while `wantsPreparing` (see the header): the others are released and
   * their builds cancelled.
   */
  sync(res: Resolution, preload: boolean, direction: 1 | -1 = 1, allow?: (set: SceneSetId) => boolean): void {
    if (this.disposed) return;
    const now = performance.now();
    const keep = this.wanted;
    const views = this.viewSets;
    keep.clear();
    views.clear();
    for (const v of res.views) {
      this.ensure(v.set);
      keep.add(v.set);
      views.add(v.set);
    }
    const idx = SCENE_SET_ORDER.indexOf(res.episode.set);
    for (let i = idx - 1; i <= idx + 1; i++) if (i >= 0 && i < SCENE_SET_ORDER.length) keep.add(SCENE_SET_ORDER[i]);
    for (const set of RESIDENT_SETS) keep.add(set);
    if (preload && this.settled(res)) {
      const next = idx + direction;
      if (next >= 0 && next < SCENE_SET_ORDER.length) {
        const set = SCENE_SET_ORDER[next];
        const state = this.state(set);
        // one already preparing (requested on the way) becomes the preload
        if (state === "preparing" || (state === "idle" && (!allow || allow(set)))) this.ensure(set, true);
      }
      // the resident sets at idle: after the neighbour, one build at a time
      if (!this.busy()) {
        for (const set of RESIDENT_SETS) {
          if (this.state(set) === "idle" && (!allow || allow(set))) {
            this.ensure(set, true);
            break;
          }
        }
      }
    }
    for (const [id, e] of this.entries) {
      if (e.state !== "preparing") {
        if (!keep.has(id)) this.release(id);
        continue;
      }
      if (views.has(id)) {
        if (Number.isNaN(e.viewSince)) e.viewSince = now;
        else if (now - e.viewSince >= PASSING_MS) e.kept = true;
      } else e.viewSince = Number.NaN;
      if (!this.wantsPreparing(id, e)) this.release(id);
    }
  }

  /** A set still preparing stays: on screen, resident, or kept (preload / on screen long enough) while a neighbour. */
  private wantsPreparing(id: SceneSetId, e: SetEntry): boolean {
    return this.viewSets.has(id) || RESIDENT_SETS.includes(id) || (e.kept && this.wanted.has(id));
  }

  /**
   * Start preparing `set` unless it exists. `preload`: requested by the preload rule (kept
   * while preparing as long as it stays a neighbour); for a set already preparing, marks it so.
   */
  ensure(set: SceneSetId, preload = false): void {
    let entry = this.entries.get(set);
    if (entry && entry.state !== "idle") {
      if (preload && entry.state === "preparing") entry.kept = true;
      return;
    }
    if (!entry) {
      entry = { state: "idle", scene: null, generation: this.generation, ctx: null, kept: false, startedAt: 0, viewSince: Number.NaN, releasedAt: Number.NaN };
      this.entries.set(set, entry);
    }
    const e = entry;
    e.state = "preparing";
    e.kept = preload;
    e.startedAt = performance.now();
    const gen = this.generation;
    const stale = () => this.disposed || gen !== this.generation || this.entries.get(set) !== e;
    // a fresh context per prepare: it is also the build's cancel token (release → cancelSlices)
    const ctx = this.opts.contextFor(set);
    e.ctx = ctx;
    void (async () => {
      try {
        const scene = await this.opts.factories[set]();
        if (stale()) {
          scene.dispose();
          this.ended(set, e, "cancelled");
          return;
        }
        e.scene = scene;
        await scene.prepare(ctx);
        if (stale()) {
          scene.dispose();
          this.ended(set, e, "cancelled");
          return;
        }
        // COLOR_0 (baked AO) must exist before the first draw (and compile) of each geometry
        prepareSilvaGeometry(scene.scene);
        if (!this.wantsPreparing(set, e)) {
          // the story moved on while it was building (e.g. a jump): do not compile / warm it
          this.drop(set, e);
          this.ended(set, e, "dropped");
          return;
        }
        if (this.opts.warm) await this.opts.warm(set, scene, stale);
        if (stale()) {
          scene.dispose();
          this.ended(set, e, "cancelled");
          return;
        }
        e.state = "ready";
        this.ended(set, e, "ready");
        this.opts.onReady?.(set, scene);
      } catch (error) {
        try {
          e.scene?.dispose();
        } catch {
          // partially built scene, nothing more to free
        }
        e.scene = null;
        if (stale()) {
          // released while preparing: the cancelled build unwound (SliceCancelled) or failed late
          this.ended(set, e, "cancelled");
          return;
        }
        e.state = "failed";
        e.error = error;
        this.ended(set, e, "failed");
        this.report(set, error, "failed to prepare");
      }
    })();
  }

  private ended(set: SceneSetId, e: SetEntry, outcome: PrepareOutcome): void {
    e.ctx = null;
    const now = performance.now();
    this.opts.onPrepareEnd?.(set, outcome, now - e.startedAt, Number.isNaN(e.releasedAt) ? null : now - e.releasedAt);
  }

  /**
   * A ready set threw while it was being updated or rendered: dispose it and keep it
   * failed (logged once) so the other views keep rendering. It is tried again only
   * after it has been released (out of range) and is needed again.
   */
  fail(set: SceneSetId, error: unknown): void {
    const e = this.entries.get(set);
    if (!e || e.state === "failed") return;
    const scene = e.scene;
    e.state = "failed";
    e.scene = null;
    e.error = error;
    try {
      scene?.dispose();
    } catch {
      // half-broken scene, nothing more to free
    }
    this.report(set, error, "failed while rendering and is disabled");
  }

  private report(set: SceneSetId, error: unknown, what: string): void {
    if (!this.reported.has(set)) {
      this.reported.add(set);
      console.error(`[nature] scene set "${set}" ${what}`, error);
    }
    this.opts.onError?.(set, error);
  }

  private drop(set: SceneSetId, e: SetEntry): void {
    if (this.entries.get(set) === e) this.entries.delete(set);
    try {
      e.scene?.dispose();
    } catch {
      // ignore
    }
    e.scene = null;
    this.opts.onRelease?.(set);
  }

  release(set: SceneSetId): void {
    const e = this.entries.get(set);
    if (!e) return;
    this.entries.delete(set);
    if (e.state === "preparing") {
      // still building: stop it at its next slice boundary; its prepare sees stale() and
      // disposes whatever the partial build had created
      e.releasedAt = performance.now();
      if (e.ctx) cancelSlices(e.ctx);
    } else if (e.scene && e.state === "ready") e.scene.dispose();
    this.opts.onRelease?.(set);
  }

  /** Drop every prepared set (quality or seed change); they re-prepare on demand. */
  invalidateAll(): void {
    this.generation++;
    for (const id of [...this.entries.keys()]) this.release(id);
  }

  /**
   * Drop every set except `keep` and the resident sets (adaptive quality step-down: the
   * sets on screen and the resident ones stay as built, the others prepare again with the
   * new budgets when needed). A set still preparing outside them is dropped too (its build
   * is cancelled, its prepare sees it is stale).
   */
  invalidateExcept(keep: ReadonlySet<SceneSetId>): void {
    for (const id of [...this.entries.keys()]) if (!keep.has(id) && !RESIDENT_SETS.includes(id)) this.release(id);
  }

  /** A set is being prepared (build, compile, warm-up): frame times say little now. */
  busy(): boolean {
    for (const e of this.entries.values()) if (e.state === "preparing") return true;
    return false;
  }

  dispose(): void {
    this.invalidateAll();
    this.disposed = true;
  }
}
