#!/usr/bin/env node
/**
 * Deterministic screenshots of the running app through system Edge (fallback: Chrome)
 * with playwright-core. No browser download.
 *
 *   npm run capture -- --t 12.5 --out docs/captures/web/arch.png
 *   npm run capture -- --scene arch --progress 0.5
 *   npm run capture -- --series "8,8.4,8.8" --out docs/captures/web/t01.png   (→ t01_8.png, t01_8.4.png, …)
 *
 * Options: --t, --scene + --progress, --time (2), --seed (134), --quality (high),
 * --width (1440), --height (1020), --query "post=0&dof=0", --out, --url
 * (http://localhost:3000), --wait <ms> (extra delay after ready), --series "<t,t,…>",
 * --timeout <ms> (120000), --channel msedge|chrome, --headed, --overlay,
 * --jpeg [quality] (82): save the accepted picture as a JPEG instead of a PNG (the
 * stability check still compares PNG shots; the JPEG is taken right after the pair).
 *
 * Waits for `window.__NATURE_READY__ === true`, prints the engine's WebGL renderer
 * string (`__NATURE__.stats().gpu`) and page errors.
 *
 * Stability: every shot is compared with a second one 3 frames later (decoded pixels,
 * RGB). The pair passes when no pixel differs by more than 1/255 and fewer than 0.01 %
 * of the pixels differ at all; the number of differing pixels and the largest
 * difference are printed whenever the two are not byte-identical. The GPU here renders
 * an occasional single frame with ±1/255 in a few (up to ~250) pixels, the same pixels
 * every time and with every engine version, so a pair outside the tolerance is retried
 * (up to 5 shots, 3 frames apart): one deviating frame is reported and tolerated, a
 * picture that keeps changing fails the run. The later shot of the accepted pair is saved.
 *
 * Exits non-zero on any failure: bad arguments (e.g. `--t abc`, an unknown --scene),
 * timeout, a scene set that failed, an uncaught page error, a `[nature]` console error,
 * an engine in error state, an unstable picture.
 *
 * --overlay: keep the HTML overlay visible. The page is opened without `capture=1` and
 * without `t=` (the engine is not pinned); the window is scrolled to the position of
 * --t (y = t / 59 × max scroll, the engine's own mapping) and the shot is taken once the
 * engine has followed the scroll (t == target, ready) and the overlay's one-off intro
 * animations have finished. CSS scroll-driven animations are left running
 * (Playwright's animations: "disabled" would fast-forward them).
 *   npm run capture -- --overlay --t 12.5 --out docs/captures/web/overlay_12.5.png
 */
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { inflateSync } from "node:zlib";
import { chromium } from "playwright-core";

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("--")) continue;
    const eq = a.indexOf("=");
    if (eq > 0) {
      out[a.slice(2, eq)] = a.slice(eq + 1);
    } else {
      const key = a.slice(2);
      const next = argv[i + 1];
      if (next === undefined || next.startsWith("--")) out[key] = true;
      else {
        out[key] = next;
        i++;
      }
    }
  }
  return out;
}

function fail(message) {
  console.error(`[capture] ${message}`);
  process.exit(1);
}

const args = parseArgs(process.argv.slice(2));

/** A numeric option: the default when absent, a finite number when given, an error otherwise. */
function num(name, fallback, { min = -Infinity, integer = false } = {}) {
  const raw = args[name];
  if (raw === undefined) return fallback;
  const value = typeof raw === "string" && raw.trim() !== "" ? Number(raw) : NaN;
  if (!Number.isFinite(value)) fail(`--${name} must be a number, got ${raw === true ? "no value" : `"${raw}"`}`);
  if (value < min) fail(`--${name} must be ≥ ${min}, got ${value}`);
  if (integer && !Number.isInteger(value)) fail(`--${name} must be an integer, got ${value}`);
  return value;
}

function str(name, fallback) {
  const raw = args[name];
  if (raw === undefined) return fallback;
  if (raw === true || String(raw).trim() === "") fail(`--${name} needs a value`);
  return String(raw);
}

const opt = {
  url: str("url", "http://localhost:3000").replace(/\/+$/, ""),
  width: num("width", 1440, { min: 1, integer: true }),
  height: num("height", 1020, { min: 1, integer: true }),
  time: num("time", 2, { min: 0 }),
  seed: num("seed", 134, { integer: true }),
  quality: str("quality", "high"),
  query: args.query !== undefined ? str("query", "").replace(/^[?&]/, "") : "",
  wait: num("wait", 0, { min: 0 }),
  timeout: num("timeout", 120000, { min: 1 }),
  channel: args.channel !== undefined ? str("channel", null) : null,
  headed: args.headed === true,
  overlay: args.overlay === true || args.overlay === "1" || args.overlay === "true",
  jpeg: args.jpeg === undefined ? null : args.jpeg === true ? 82 : num("jpeg", 82, { min: 1, integer: true }),
};
if (opt.jpeg !== null && opt.jpeg > 100) fail(`--jpeg quality must be 1–100, got ${opt.jpeg}`);
if (!["high", "medium", "low"].includes(opt.quality)) fail(`--quality must be high, medium or low, got "${opt.quality}"`);

const scene = args.scene !== undefined ? str("scene", null) : null;
const progress = num("progress", 0);
if (args.progress !== undefined && scene === null) fail("--progress needs --scene");

const seriesLabels = args.series !== undefined
  ? str("series", "")
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean)
  : null;
const series = seriesLabels ? seriesLabels.map((s) => (s === "" ? NaN : Number(s))) : null;
if (series && (series.length === 0 || series.some((v) => !Number.isFinite(v)))) fail(`--series must be a comma separated list of numbers, got "${args.series}"`);
if (series && scene !== null) fail("--series and --scene cannot be combined");

const tArg = num("t", null);
if (tArg !== null && scene !== null) fail("--t and --scene cannot be combined");
const firstT = series ? series[0] : tArg;
const label = series ? "series" : scene ? `${scene}_${progress}` : firstT !== null ? String(firstT) : "0";
const outArg = args.out !== undefined ? str("out", null) : `docs/captures/web/capture_${label}.png`;

function buildUrl(t) {
  const p = new URLSearchParams();
  if (opt.overlay) {
    // overlay mode: no pin, no capture flag; the story position comes from scrolling
    p.set("time", String(opt.time));
    p.set("seed", String(opt.seed));
    p.set("quality", opt.quality);
    const extra = opt.query ? `&${opt.query}` : "";
    return `${opt.url}/?${p.toString()}${extra}`;
  }
  if (t !== null && t !== undefined) p.set("t", String(t));
  else if (scene) {
    p.set("scene", scene);
    p.set("progress", String(progress));
  }
  p.set("time", String(opt.time));
  p.set("seed", String(opt.seed));
  p.set("quality", opt.quality);
  p.set("capture", "1");
  const extra = opt.query ? `&${opt.query}` : "";
  return `${opt.url}/?${p.toString()}${extra}`;
}

function seriesPath(base, t) {
  const ext = path.extname(base) || ".png";
  const stem = base.slice(0, base.length - path.extname(base).length);
  return `${stem}_${t}${ext}`;
}

const GPU_ARGS = [
  "--enable-gpu",
  "--ignore-gpu-blocklist",
  "--use-gl=angle",
  "--use-angle=d3d11",
  "--enable-webgl",
  "--force-color-profile=srgb",
  "--hide-scrollbars",
  "--disable-renderer-backgrounding",
  "--disable-background-timer-throttling",
  "--disable-backgrounding-occluded-windows",
];

async function launch() {
  const channels = opt.channel ? [opt.channel] : ["msedge", "chrome"];
  let lastError = null;
  for (const channel of channels) {
    try {
      const browser = await chromium.launch({ channel, headless: !opt.headed, args: GPU_ARGS });
      return { browser, channel };
    } catch (err) {
      lastError = err;
      console.warn(`[capture] could not launch ${channel}: ${String(err).split("\n")[0]}`);
    }
  }
  throw lastError ?? new Error("no browser channel available");
}

async function waitReady(page) {
  await page.waitForFunction(() => window.__NATURE_READY__ === true, null, { timeout: opt.timeout, polling: 100 });
  if (opt.wait > 0) await page.waitForTimeout(opt.wait);
}

/** --overlay: scroll the window to the position of t (or of --scene/--progress); returns t. */
async function scrollToStory(page, t) {
  return page.evaluate(
    ({ t, scene, progress }) => {
      let target = t;
      if (target === null && scene) {
        const sec = document.querySelector(`section[data-section="${CSS.escape(scene)}"]`);
        const a = Number(sec?.dataset.start);
        const b = Number(sec?.dataset.end);
        if (!Number.isFinite(a) || !Number.isFinite(b)) throw new Error(`unknown scene section "${scene}"`);
        target = a + (b - a) * Math.min(1, Math.max(0, progress));
      }
      target = Math.min(59, Math.max(0, target ?? 0));
      const max = Math.max(0, document.documentElement.scrollHeight - window.innerHeight);
      window.scrollTo({ top: (target / 59) * max, left: 0, behavior: "instant" });
      return target;
    },
    { t: t ?? null, scene, progress },
  );
}

/** --overlay: engine followed the scroll and settled; fonts loaded; load intro finished. */
async function waitOverlayReady(page, t) {
  await page.evaluate(() => document.fonts.ready.then(() => true));
  await page.waitForFunction(
    (t) => {
      const api = window.__NATURE__;
      if (!api) return false;
      const s = api.state();
      return Math.abs(s.targetT - t) < 0.05 && s.t === s.targetT && window.__NATURE_READY__ === true;
    },
    t,
    { timeout: opt.timeout, polling: 100 },
  );
  let settled = true;
  try {
    await page.waitForFunction(
      () => document.getAnimations().every((a) => a.timeline !== document.timeline || a.playState !== "running"),
      null,
      { timeout: 15000, polling: 100 },
    );
  } catch {
    settled = false;
    console.warn("[capture] warning: time-based animations still running after 15 s");
  }
  if (opt.wait > 0) await page.waitForTimeout(opt.wait);
  return settled;
}

const nextFrames = (page, n) =>
  page.evaluate(
    (n) =>
      new Promise((resolve) => {
        let left = n;
        const step = () => (--left <= 0 ? resolve(true) : requestAnimationFrame(step));
        requestAnimationFrame(step);
      }),
    n,
  );

const MAX_SHOTS = 5;
/** Largest per-channel difference and share of differing pixels a stable pair may show. */
const TOLERANCE = { maxLevel: 1, maxShare: 0.0001 };

/** Decodes the 8-bit RGB / RGBA non-interlaced PNGs that Chromium screenshots are. */
function decodePng(buf) {
  if (buf.length < 8 || buf.readUInt32BE(0) !== 0x89504e47) throw new Error("screenshot is not a PNG");
  let pos = 8;
  let width = 0;
  let height = 0;
  let channels = 0;
  const idat = [];
  while (pos + 8 <= buf.length) {
    const len = buf.readUInt32BE(pos);
    const type = buf.toString("latin1", pos + 4, pos + 8);
    const data = buf.subarray(pos + 8, pos + 8 + len);
    if (type === "IHDR") {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
      const depth = data[8];
      const colour = data[9];
      const interlace = data[12];
      if (depth !== 8 || interlace !== 0 || (colour !== 2 && colour !== 6)) throw new Error(`unsupported PNG (depth ${depth}, colour type ${colour}, interlace ${interlace})`);
      channels = colour === 6 ? 4 : 3;
    } else if (type === "IDAT") idat.push(data);
    else if (type === "IEND") break;
    pos += 12 + len;
  }
  const raw = inflateSync(Buffer.concat(idat));
  const stride = width * channels;
  const out = Buffer.alloc(stride * height);
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)];
    const src = y * (stride + 1) + 1;
    const dst = y * stride;
    for (let x = 0; x < stride; x++) {
      const a = x >= channels ? out[dst + x - channels] : 0;
      const b = y > 0 ? out[dst - stride + x] : 0;
      const c = x >= channels && y > 0 ? out[dst - stride + x - channels] : 0;
      let v = raw[src + x];
      if (filter === 1) v += a;
      else if (filter === 2) v += b;
      else if (filter === 3) v += (a + b) >> 1;
      else if (filter === 4) {
        const p = a + b - c;
        const pa = Math.abs(p - a);
        const pb = Math.abs(p - b);
        const pc = Math.abs(p - c);
        v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      }
      out[dst + x] = v & 255;
    }
  }
  return { width, height, channels, data: out };
}

/** RGB difference of two screenshots: differing pixels, largest channel difference (0–255). */
function pixelDiff(pngA, pngB) {
  if (pngA.equals(pngB)) return { pixels: 0, max: 0, total: 1 };
  const a = decodePng(pngA);
  const b = decodePng(pngB);
  const total = a.width * a.height;
  if (a.width !== b.width || a.height !== b.height) return { pixels: total, max: 255, total };
  let pixels = 0;
  let max = 0;
  for (let p = 0; p < total; p++) {
    const ia = p * a.channels;
    const ib = p * b.channels;
    const d = Math.max(Math.abs(a.data[ia] - b.data[ib]), Math.abs(a.data[ia + 1] - b.data[ib + 1]), Math.abs(a.data[ia + 2] - b.data[ib + 2]));
    if (d > 0) {
      pixels++;
      if (d > max) max = d;
    }
  }
  return { pixels, max, total };
}

const withinTolerance = (d) => d.max <= TOLERANCE.maxLevel && d.pixels < d.total * TOLERANCE.maxShare;
const describeDiff = (d) => (d.pixels === 0 ? "identical" : `${d.pixels} px differ (${((100 * d.pixels) / d.total).toFixed(4)} %), max ${d.max}/255`);

/**
 * Screenshots 3 frames apart until a pair is within TOLERANCE (at most MAX_SHOTS shots).
 * Returns the later shot of that pair and every comparison made.
 */
async function stableShot(page, shotOpts) {
  let prev = await page.screenshot(shotOpts);
  const diffs = [];
  for (let n = 2; n <= MAX_SHOTS; n++) {
    await nextFrames(page, 3);
    const next = await page.screenshot(shotOpts);
    const d = pixelDiff(prev, next);
    diffs.push(d);
    if (withinTolerance(d)) return { image: next, diffs, stable: true };
    prev = next;
  }
  return { image: prev, diffs, stable: false };
}

async function engineState(page) {
  return page.evaluate(() => {
    const api = window.__NATURE__;
    if (!api) return null;
    const s = api.state();
    const st = api.stats();
    return {
      t: s.t,
      episode: s.episode,
      transition: s.transition,
      sets: s.sets,
      status: s.status,
      ready: window.__NATURE_READY__ === true,
      calls: st.drawCalls,
      tris: st.triangles,
      passes: st.passes,
      gpu: st.gpu,
    };
  });
}

async function main() {
  const started = Date.now();
  const { browser, channel } = await launch();
  const pageErrors = [];
  const consoleErrors = [];
  const problems = [];
  try {
    const context = await browser.newContext({
      viewport: { width: opt.width, height: opt.height },
      deviceScaleFactor: 1,
      reducedMotion: "no-preference",
      colorScheme: "dark",
    });
    const page = await context.newPage();
    page.on("console", (msg) => {
      if (msg.type() !== "error") return;
      const text = msg.text();
      if (text.startsWith("[nature]")) pageErrors.push(`console.error: ${text}`);
      else consoleErrors.push(`console.error: ${text}`);
    });
    page.on("pageerror", (err) => pageErrors.push(`pageerror: ${err.message}`));
    page.on("crash", () => pageErrors.push("page crashed"));

    const url = buildUrl(firstT);
    console.log(`[capture] ${channel} ${opt.width}x${opt.height} → ${url}`);
    await page.goto(url, { waitUntil: "load", timeout: opt.timeout });
    if (scene) {
      const known = await page.evaluate(
        (scene) => document.querySelector(`section[data-section="${CSS.escape(scene)}"][data-kind="episode"]`) !== null,
        scene,
      );
      if (!known) throw new Error(`unknown --scene "${scene}" (no episode section with that id)`);
    }

    const shots = series ? series.map((t, i) => ({ t, file: seriesPath(outArg, seriesLabels[i]) })) : [{ t: firstT, file: outArg }];
    for (let i = 0; i < shots.length; i++) {
      const shot = shots[i];
      const t0 = Date.now();
      let overlayInfo = "";
      let overlaySettled = true;
      if (opt.overlay) {
        const target = await scrollToStory(page, shot.t);
        overlaySettled = await waitOverlayReady(page, target);
        const visible = await page.evaluate(() =>
          [...document.querySelectorAll("[data-overlay-stage]")]
            .filter((el) => getComputedStyle(el).visibility === "visible")
            .map((el) => el.getAttribute("data-overlay-stage")),
        );
        const path_ = await page.evaluate(() => document.querySelector("[data-nature-overlay]")?.getAttribute("data-overlay-path") ?? "?");
        overlayInfo = `  overlay(${path_}): ${visible.join("+") || "none"}  scrollY=${await page.evaluate(() => Math.round(window.scrollY))}`;
      } else {
        if (i > 0) {
          await page.evaluate((t) => window.__NATURE__.setT(t), shot.t);
        }
        await waitReady(page);
      }
      const state = await engineState(page);
      if (i === 0 && state?.gpu) {
        console.log(`[capture] UNMASKED_RENDERER_WEBGL: ${state.gpu}`);
        if (/swiftshader|llvmpipe|software/i.test(state.gpu)) console.warn("[capture] warning: software rasterizer, not the real GPU");
      }
      mkdirSync(path.dirname(path.resolve(shot.file)), { recursive: true });
      const shotOpts = { type: "png", animations: opt.overlay ? "allow" : "disabled", caret: "hide" };
      // the same picture 3 frames later, or the capture is not deterministic
      const taken = await stableShot(page, shotOpts);
      if (opt.jpeg !== null) {
        writeFileSync(shot.file, await page.screenshot({ ...shotOpts, type: "jpeg", quality: opt.jpeg }));
      } else {
        writeFileSync(shot.file, taken.image);
      }
      const tr = state?.transition ? ` ${state.transition.id} ${state.transition.mode} k=${state.transition.k.toFixed(3)}` : "";
      console.log(
        `[capture] ${shot.file}  t=${state?.t ?? "?"} ${state?.episode ?? ""}${tr}  calls=${state?.calls ?? "?"} tris=${state?.tris ?? "?"}  ready in ${Date.now() - t0} ms${overlayInfo}`,
      );
      const after = await engineState(page);
      if (taken.diffs.some((d) => d.pixels > 0)) {
        console.log(`[capture] stability (shots 3 frames apart): ${taken.diffs.map(describeDiff).join("; ")}${taken.stable ? " — within tolerance" : ""}`);
      }
      if (!taken.stable) {
        const hint = opt.overlay && !overlaySettled ? " (overlay animations were still running)" : "";
        problems.push(
          `${shot.file}: the picture kept changing — no two of ${MAX_SHOTS} shots 3 frames apart within ±${TOLERANCE.maxLevel}/255 on < ${TOLERANCE.maxShare * 100} % of the pixels — not stable / deterministic${hint}`,
        );
      }
      if (after && !after.ready) problems.push(`${shot.file}: __NATURE_READY__ dropped back to false right after the shot`);
      const failed = Object.entries(after?.sets ?? {}).filter(([, s]) => s === "failed").map(([k]) => k);
      if (failed.length) problems.push(`${shot.file}: scene set(s) failed: ${failed.join(", ")}`);
      if (after?.status === "error") problems.push(`${shot.file}: engine status "error"`);
    }
    await context.close();
  } catch (err) {
    problems.push(`failed: ${err instanceof Error ? err.message : String(err)}`);
  } finally {
    await browser.close();
  }
  if (consoleErrors.length) {
    console.log(`[capture] console errors (${consoleErrors.length}, not fatal):`);
    for (const e of consoleErrors) console.log(`  ${e}`);
  }
  if (pageErrors.length) {
    console.log(`[capture] page errors (${pageErrors.length}):`);
    for (const e of pageErrors) console.log(`  ${e}`);
    problems.push(`${pageErrors.length} page error(s)`);
  }
  for (const p of problems) console.error(`[capture] FAIL ${p}`);
  console.log(`[capture] ${problems.length ? "failed" : "done"} in ${((Date.now() - started) / 1000).toFixed(1)} s`);
  process.exit(problems.length ? 1 : 0);
}

main().catch((err) => fail(err instanceof Error ? err.stack ?? err.message : String(err)));
