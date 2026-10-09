/**
 * URL API (CLAUDE.md): `/?t=12.5` or `/?scene=arch&progress=0.5`, plus `time`, `seed`,
 * `quality`, `capture`, `debug`, `layers`, `wind`, `post`, `autoplay`.
 * Extra engine flags: `dof=0`, `bloom=0`, `vignette=0`, `tonemap=aces|agx|neutral`,
 * `parallax=1`, `motion=reduce|full`.
 */
import { DEFAULT_SEED, VIDEO_DURATION, isEpisodeId, timeForEpisode } from "../SceneConfig";
import type { EpisodeId, QualityLevel, ToneMappingName } from "../types";
import { QUALITY_LEVELS, TONE_MAPPINGS } from "../types";

export interface UrlState {
  /** Explicit story time (video seconds). */
  t: number | null;
  scene: EpisodeId | null;
  progress: number | null;
  /** Fixed ambient clock (seconds). */
  time: number | null;
  seed: number;
  quality: QualityLevel | null;
  capture: boolean;
  debug: boolean;
  /** null = all layers visible. */
  layers: string[] | null;
  wind: boolean;
  post: boolean;
  dof: boolean;
  bloom: boolean;
  vignette: boolean;
  autoplay: boolean;
  toneMapping: ToneMappingName | null;
  parallax: boolean;
  /** null = follow prefers-reduced-motion. */
  reducedMotion: boolean | null;
}

function parseNumber(value: string | null): number | null {
  if (value === null || value.trim() === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function parseBool(value: string | null, fallback: boolean): boolean {
  if (value === null) return fallback;
  const v = value.trim().toLowerCase();
  if (v === "" || v === "1" || v === "true" || v === "yes" || v === "on") return true;
  if (v === "0" || v === "false" || v === "no" || v === "off") return false;
  return fallback;
}

export function parseUrlState(search: string): UrlState {
  const q = new URLSearchParams(search.startsWith("?") ? search.slice(1) : search);
  const sceneRaw = q.get("scene");
  const qualityRaw = q.get("quality");
  const toneRaw = q.get("tonemap") ?? q.get("toneMapping");
  const layersRaw = q.get("layers");
  const motionRaw = q.get("motion");
  const seed = parseNumber(q.get("seed"));

  return {
    t: parseNumber(q.get("t")),
    scene: sceneRaw && isEpisodeId(sceneRaw) ? sceneRaw : null,
    progress: parseNumber(q.get("progress")),
    time: parseNumber(q.get("time")),
    seed: seed === null ? DEFAULT_SEED : Math.trunc(seed),
    quality: qualityRaw && (QUALITY_LEVELS as readonly string[]).includes(qualityRaw) ? (qualityRaw as QualityLevel) : null,
    capture: parseBool(q.get("capture"), false),
    debug: parseBool(q.get("debug"), false),
    layers:
      layersRaw === null || layersRaw.trim() === "" || layersRaw === "all"
        ? null
        : layersRaw
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean),
    wind: parseBool(q.get("wind"), true),
    post: parseBool(q.get("post"), true),
    dof: parseBool(q.get("dof"), true),
    bloom: parseBool(q.get("bloom"), true),
    vignette: parseBool(q.get("vignette"), true),
    autoplay: parseBool(q.get("autoplay"), false),
    toneMapping: toneRaw && (TONE_MAPPINGS as readonly string[]).includes(toneRaw) ? (toneRaw as ToneMappingName) : null,
    parallax: parseBool(q.get("parallax"), false),
    reducedMotion: motionRaw === "reduce" ? true : motionRaw === "full" ? false : null,
  };
}

/** Story time requested by the URL (`t` wins over `scene`+`progress`), clamped to 0–59. */
export function requestedTime(state: UrlState): number | null {
  if (state.t !== null) return Math.min(VIDEO_DURATION, Math.max(0, state.t));
  if (state.scene !== null) return timeForEpisode(state.scene, state.progress ?? 0);
  return null;
}
