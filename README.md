# Silva — a scroll-driven WebGL nature experience

A 59-second scroll story through a mossy grove, a dark canyon, a glowing orb, a fuzzy ball
looping around a broken branch, a carved stone, a growing canopy and a moss-hill finale.
Built with Next.js 16 (App Router, TypeScript) and three.js, with an imperative WebGL engine
in `src/nature/` and an HTML overlay in `src/components/overlay/`. All 3D models are made
in Blender by scripts in `assets-src/blender/`; all textures are generated in-house. There are
no third-party textures, models or HDRIs.

The piece is a recreation of the visual result of a reference video (the pack lives in
`docs/nature-webgl-reference/`, dev-only). The target was the picture, the motion and the
feel, not a pixel-exact copy: own name (**Silva**), own emblem (**the seed**,
`assets-src/emblem/seed.svg`), own copy and layout.

## Run

```bash
npm install
npm run dev          # http://localhost:3000
npm run build && npm run start
npm run lint && npm run typecheck
```

Deterministic 1440×1020 screenshots through the system Edge/Chrome (nothing is downloaded):

```bash
npm run capture -- --t 12.5 --out docs/captures/web/arch.png
```

Options: `--time <s>` (ambient clock), `--series "8,8.4,8.8"`, `--quality low|medium|high`,
`--query "post=0"`, `--overlay`, `--jpeg 82`, `--url http://localhost:3100` (another server).
Comparison sheets: `node tools/compare.mjs`.

## What is on the page

Video time `t` (0–59 s) is the common coordinate; scroll progress = `t / 59`.

| t (s) | Episode | Scene set | What you see |
|---|---|---|---|
| 0–8 | hero | grove | overgrown arm top-left, a clover mound, blurred foreground |
| 8–10 | T01 | grove | the camera descends past the big mossy mass (no fade) |
| 10–15.5 | arch | grove | two mossy trunks with bark windows, a low branch |
| 15.5–21 | canyon | canyon | a dark cleft: lit rock walls, a teal glow deep inside, light shafts with dust |
| 21–25 | oracle | oracle | a glowing orb with particles |
| 25–31 | streams | oracle | the orb with a loop and streams |
| 31–37 | branch | branch | a J-branch with a broken end, a fuzzy ball on a closed loop |
| 37–43.3 | stone | stone | a diagonal rock with the seed emblem carved into it |
| 43.3–52.8 | canopy, canopyClose | canopy | a canopy grows from the bottom, then closer |
| 52.8–59 | finale | finale | moss hills, a standing stone with the seed cut through it |

Transitions: a continuous descent (hero→arch), a dark cover (arch→canyon), a dark crossfade
(canyon→oracle), a ragged organic wipe (streams→branch), slides with soft blurred edges
(branch→stone, stone→canopy), the canopy growing on black, leaves parting towards the edges
(canopyClose→finale). Reduced motion (the OS setting or `?motion=reduce`, both identical)
turns every transition into a crossfade and the overlay into sequential fades.

## URL API

`/?t=12.5` or `/?scene=arch&progress=0.5`, plus `time=2` (ambient clock), `seed=134`,
`quality=high|medium|low`, `capture=1` (fixed time, overlay hidden, `window.__NATURE_READY__`),
`debug=1` (panel), `layers=wood,moss,grass,leaves`, `wind=0`, `post=0`, `autoplay=1`,
`dof=0`, `bloom=0`, `vignette=0`, `tonemap=aces|agx|neutral`, `motion=reduce|full`,
`adapt=<ms>` / `adapt=0` (adaptive quality threshold / off), `overlay=fallback`.
Same URL ⇒ same picture: all randomness comes from the seed, the pose is a pure function of
progress, wind and ambient motion are pure functions of the ambient clock.
`window.__NATURE__` exposes `state()`, `stats()`, `setT`, `setScene`, `setTime`, `setQuality`,
`setSeed`, `setFlags`.

## Engine in one paragraph

One `WebGLRenderer`, one `requestAnimationFrame` loop, one clock. `SceneDirector` maps the
episode to a scene set, preloads the neighbours in the travel direction, keeps the grove
resident (Home, About and the logo land on it at once) and lets at most two sets render, only
inside a transition window. Scenes are built deterministically from Blender GLBs plus
code-scattered vegetation (`src/nature/vegetation/`: grass blades, kit leaves, moss fuzz, baked
per-instance shading). Builds are step generators paced by the frame loop (8–24 ms per frame
depending on what is on screen), so a scene prepares while the current one keeps rendering,
and a build the story has passed is cancelled at its next step. A set that turns ready while it
is on screen fades in over 0.45 s instead of popping. Post stack: DOF, transition compositing in
linear HDR, bloom, grade/vignette, one tone map. Shader programs are compiled and warmed
through a queue inside the frame loop, never during a transition. Auto quality starts from
the GPU string and steps down (never up) when the frame-time p95 stays above 20 ms: high →
medium → low → low at DPR 1.0; `quality=` pins it. Without WebGL 2 the page shows a still
poster of the hero scene (`public/nature/poster.jpg`, our own render) with a small status chip.

Deployment note: the only server-side code is `src/app/api/nature-assets/route.ts`, a listing of
`public/nature/` (size and mtime) that the engine uses to probe for assets without 404s and to
cache-bust rebuilt files with `?v=<mtime>`. It needs a Node runtime (`next start` or a Node host).
On a static host the engine falls back to HEAD probes and plain URLs, so the page still works.
The overlay's `?motion=` flag is applied by a tiny script inserted into `<head>` at render time.

## Assets (all own work, see `public/nature/ASSET_SOURCES.md`)

| File | Size | Triangles | Built by |
|---|---|---|---|
| `grove.glb` | 7.1 MB | 240,566 | `assets-src/blender/build_grove.py` |
| `branch.glb` | 5.7 MB | 212,428 | `build_branch.py` + `branchlib.py` |
| `canyon.glb` | 4.2 MB | 106,639 | `build_canyon.py` |
| `stone.glb` | 4.1 MB | 138,504 | `build_stone.py` + `stonelib.py` |
| `canopy.glb` | 6.4 MB | 192,080 | `build_canopy.py` + `canopylib.py` |
| `finale.glb` | 7.6 MB | 209,007 | `build_finale.py` + `stonelib.py` |
| `kit.glb` | 52 KB | 950 | `build_kit.py` (leaves, flowers, sprigs for instancing) |
| `textures/*.webp` | 7.0 MB total | — | `build_textures.py` (bark, rock, moss, kit; tileable, OpenGL normals) |

Headless build of any asset (Blender 5.1):

```bash
"/c/Program Files/Blender Foundation/Blender 5.1/blender.exe" --background --factory-startup --python assets-src/blender/build_branch.py -- --export --validate --render --compare
```

`assets-src/blender/nature.blend` keeps every scene set in its own Blender scene for
inspection; `-- --save` rebuilds one set into it (run the saves one after another, never in
parallel). Preview renders and comparison sheets go to `docs/captures/blender/`.

## How it was verified (honestly)

- Every scene was tuned against captures (`docs/captures/web/`, comparison sheets in
  `docs/captures/web/compare/`) and judged by eye at 1440×1020, then on phones and tablets.
- A measurement pass (`docs/captures/qa/REPORT.md`, 2026-10-08, RTX 2060 laptop, Edge 154,
  ANGLE D3D11, development build) found steady-state frames of 1.1–7.6 ms at high quality
  1440×1020, no missed vsync at 60 Hz at any quality or at 1920×1080 / DPR 2, and one-off
  freezes of 0.7–1.1 s from shader compilation, which were then removed.
- A final UX review (`docs/captures/review/REPORT.md`, 2026-10-09, 30 contact sheets and 9
  transition strips, same machine, dev build) found no blocker, no broken interaction (2,280
  link/button hit tests) and a clean console. At reading pace the frame interval was p50 16.7 /
  p95 17.0 ms with 17 hitches of 50–117 ms per pass while the next scene was built; after the
  fixes below a repeat of the same pass showed 2 frames above 50 ms and none above 100 ms.
- Fixes made on that review, each with before/after proof in `docs/captures/review/fixes/`:
  no bare frame at an adaptive step-down; builds paced by the frame loop; scenes fade in
  instead of popping after jumps and on first load; the grove kept resident; stale builds
  cancelled without leaks; phones render `low` at DPR 1.5 (sharper) with a last DPR fallback;
  overlay copy no longer slides under the top bar; reduced-motion sections fade in sequence;
  canopy and streams copy readable over the picture; 11 px text floor and a 16 px email field
  everywhere; the no-WebGL-2 chip covers no copy; the canyon lifted from ~90 % black to a
  readable cleft (+2 draw calls, GPU cost within run-to-run noise); the hero and the branch
  recomposed for phone portrait (desktop and tablet pixel-identical).
- The production build (`npm run build` + `next start`) was captured locally at 4, 12.5, 34 and
  56 s: pixel-identical to the dev server (0 px differ), same draw calls and triangle counts.
- Reduced motion, tab visibility (the ambient clock never jumps), WebGL context loss and the
  no-WebGL-2 fallback were exercised; the URL API gives identical pictures for identical URLs
  (±1/255 GPU flicker on a handful of pixels).
- Not measured: real phones and tablets (only Chromium device emulation on the desktop GPU),
  other GPUs, Safari and Firefox, a deployed host. No frame rate in the reports is estimated;
  every number comes from a listed run.

`docs/captures/` is large (≈ 1 GB of PNG/JPEG) and dev-only; nothing under `docs/` is a runtime asset.

## Known gaps and deliberate differences

- Moss and fur read as fine strokes rather than the reference's velvet; the canopy leaves are
  coarser than the video's. Both are instance-budget trade-offs (grove ≈ 500k plant instances
  at high; canopy 96k leaves).
- The green photos inside the reference's interface cards were not reproduced, nor were the
  "Grades" orbit pill and the logo shadow on the canopy.
- Phones: only emulated. On 375×667 the branch view hits its field-of-view cap and the J is
  smaller; 320 px wide was not captured; landscape phones (844×390) are not tuned; on very
  short phones the finale card touches the top of the standing stone.
- A jump made several seconds after Home/End to a scene the story passed waits up to ≈ 0.7 s
  longer than before, because passed builds are now cancelled instead of finishing in the
  background (the trade for no hitches right after landing).
- The overlay follows raw scroll while the 3D is damped by 0.16 s, so copy leads the picture by
  up to ≈ 140 px at 1440×1020 during fast scrolls.
- The start scene transfers ≈ 9.4 MB of 3D assets (grove + kit + seven textures). Meshopt
  compression of the GLBs would roughly halve it; not done.
- Plant instance counts are 2–10× the reference pack's suggested budgets at every quality. They
  run far under a 60 Hz frame on the test GPU; on weaker GPUs the adaptive auto quality steps
  down, which could not be tested on real hardware.
- Dev-only: the `debug=1` panel covers the top-right buttons; one React dev warning ("state
  update on a component that hasn't mounted yet") appeared once in ~70 page loads and could
  not be traced.

## Layout

```
src/app/                 layout, page, globals.css, icon.svg (the seed), api/nature-assets
src/components/          NatureStage (mounts the engine, fallback poster), ScrollTrack, overlay/, debug/
src/nature/              NatureExperience (renderer, frame loop, warm queue, adaptive quality, fade-in),
                         SceneDirector (residency, builds, transitions), SceneConfig (timeline, looks,
                         transitions, quality presets), CameraRig, AssetRegistry, core/, vegetation/,
                         scenes/, rendering/, debug/
public/nature/           models/*.glb, textures/*.webp, poster.jpg, ASSET_SOURCES.md
assets-src/              blender/ (build scripts, nature.blend), emblem/seed.svg
tools/                   capture.mjs, compare.mjs
docs/                    reference pack (dev-only), captures (web, blender, qa, review)
```

`CLAUDE.md` holds the working rules and the engine API for scene authors.
