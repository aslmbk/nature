#!/usr/bin/env node
/**
 * Side-by-side comparison sheet: reference | capture | 50 % blend (optionally a
 * fourth panel with the amplified absolute difference), drawn with a 2D canvas in
 * headless system Edge / Chrome through playwright-core — no native image libraries,
 * no browser download.
 *
 *   node tools/compare.mjs --ref docs/nature-webgl-reference/frames/01_hero_clean.png \
 *     --cap docs/captures/web/grove_0.5.png --out docs/captures/web/compare/grove_0.5.png
 *
 * Options: --ref, --cap, --out (required); --width <panel px> (720; height follows the
 * capture's aspect), --blend <0–1> (0.5), --diff (adds |ref − cap| × 3), --label "text"
 * (caption on the capture panel), --channel msedge|chrome.
 * Both images are scaled to the panel size, so a 1440×1020 capture and a reference of
 * any size line up as long as they have the same framing.
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { chromium } from "playwright-core";

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("--")) continue;
    const eq = a.indexOf("=");
    if (eq > 0) out[a.slice(2, eq)] = a.slice(eq + 1);
    else {
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

function fail(msg) {
  console.error(`[compare] ${msg}`);
  process.exit(1);
}

const args = parseArgs(process.argv.slice(2));
if (!args.ref || !args.cap || !args.out) fail('usage: node tools/compare.mjs --ref <png> --cap <png> --out <png> [--width 720] [--blend 0.5] [--diff] [--label "…"]');

const dataUrl = (file) => {
  const ext = path.extname(file).toLowerCase();
  const mime = ext === ".jpg" || ext === ".jpeg" ? "image/jpeg" : ext === ".webp" ? "image/webp" : "image/png";
  return `data:${mime};base64,${readFileSync(file).toString("base64")}`;
};

const opt = {
  ref: dataUrl(String(args.ref)),
  cap: dataUrl(String(args.cap)),
  width: Number(args.width ?? 720),
  blend: Math.min(1, Math.max(0, Number(args.blend ?? 0.5))),
  diff: args.diff === true,
  labels: ["reference", String(args.label ?? path.basename(String(args.cap))), `blend ${Math.round(Number(args.blend ?? 0.5) * 100)} %`, "|diff| × 3"],
};

async function launch() {
  const channels = args.channel ? [String(args.channel)] : ["msedge", "chrome"];
  let last = null;
  for (const channel of channels) {
    try {
      return await chromium.launch({ channel, headless: true });
    } catch (err) {
      last = err;
    }
  }
  throw last ?? new Error("no browser channel available");
}

const browser = await launch();
try {
  const page = await browser.newPage();
  const png = await page.evaluate(async (o) => {
    const load = (src) =>
      new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error("image decode failed"));
        img.src = src;
      });
    const [ref, cap] = await Promise.all([load(o.ref), load(o.cap)]);
    const w = o.width;
    const h = Math.round((w * cap.naturalHeight) / cap.naturalWidth);
    const panels = o.diff ? 4 : 3;
    const gap = 4;
    const canvas = document.createElement("canvas");
    canvas.width = panels * w + (panels - 1) * gap;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#111";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingQuality = "high";
    const x = (i) => i * (w + gap);
    ctx.drawImage(ref, x(0), 0, w, h);
    ctx.drawImage(cap, x(1), 0, w, h);
    ctx.drawImage(ref, x(2), 0, w, h);
    ctx.globalAlpha = o.blend;
    ctx.drawImage(cap, x(2), 0, w, h);
    ctx.globalAlpha = 1;
    if (o.diff) {
      const a = ctx.getImageData(x(0), 0, w, h);
      const b = ctx.getImageData(x(1), 0, w, h);
      const d = ctx.createImageData(w, h);
      for (let i = 0; i < d.data.length; i += 4) {
        for (let c = 0; c < 3; c++) d.data[i + c] = Math.min(255, Math.abs(a.data[i + c] - b.data[i + c]) * 3);
        d.data[i + 3] = 255;
      }
      ctx.putImageData(d, x(3), 0);
    }
    ctx.font = "13px system-ui, sans-serif";
    for (let i = 0; i < panels; i++) {
      const label = o.labels[i];
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = "rgba(0,0,0,0.6)";
      ctx.fillRect(x(i) + 8, h - 28, tw + 12, 20);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, x(i) + 14, h - 13);
    }
    return canvas.toDataURL("image/png");
  }, opt);
  const out = path.resolve(String(args.out));
  mkdirSync(path.dirname(out), { recursive: true });
  writeFileSync(out, Buffer.from(png.split(",")[1], "base64"));
  console.log(`[compare] ${out}`);
} finally {
  await browser.close();
}
