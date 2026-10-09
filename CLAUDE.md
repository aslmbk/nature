# Silva — scroll-driven WebGL nature experience (Next.js + Three.js)

Recreation of the visual result shown in the video reference pack at `docs/nature-webgl-reference/`.
Read `AGENT_BRIEF.md` there first, then `01`–`04`, and actually open the images in `frames/`, `details/`, `motion/`, `boards/`.
The target is the picture and the motion (silhouette, plants, light, wind, transitions) — not the original site's HTML, brand or code.

## Hard rules

- Nothing from `docs/` is a runtime asset. Never import it in the app or copy it to `public/`.
- No "Pyko" name, logo or copy. Own name: **Silva**. Own emblem: **the seed** — an upright almond (two arcs meeting in points at top and bottom, width = 0.5 × height) split by a narrow slanted channel into two unequal halves, corners rounded. Exact construction and paths: `assets-src/emblem/seed.svg` (the single source for the stone carving, the finale cut-outs and the UI logo).
- Models are made in Blender by scripts in `assets-src/blender/` and exported as GLB. Geometry generated in code is allowed only for: grass blades, particles, the orb, orbits/loops, backgrounds, and placeholders that are clearly marked `PLACEHOLDER`.
- No third-party textures, models or HDRIs. Everything is made here. npm packages are fine.
- One `WebGLRenderer`, one `requestAnimationFrame` loop, one clock. No per-component RAF, no second canvas.
- No `git` write commands (no commit, push, stash, checkout of files). No deploy.
- Reports are honest: say what was actually run and looked at. No invented FPS. If the browser could not be opened, say so.

## Stack and commands

Next.js 16 (App Router, TypeScript, `src/`), `three` pinned to an exact version, plain CSS modules. No React Three Fiber, no GSAP, no Tailwind. Package manager: npm.

- `npm run dev` — dev server on http://localhost:3000 (one shared instance; do not start a second one)
- `npm run build`, `npm run lint`, `npm run typecheck`
- `npm run capture -- --t 12.5 --out docs/captures/web/arch.png` — deterministic 1440×1020 screenshot through system Edge/Chrome (no browser download)
- Headless Blender: `"/c/Program Files/Blender Foundation/Blender 5.1/blender.exe" --background --factory-startup --python assets-src/blender/<script>.py`

## Layout

```
src/app/                    layout, page, globals.css
src/components/             NatureStage (client, mounts the engine), overlay/, debug/
src/nature/                 the engine (imperative Three.js, no React inside)
  NatureExperience.ts       renderer, lifecycle, the single frame loop
  SceneDirector.ts          episode ↔ scene set, poses, transitions
  SceneConfig.ts            timeline, per-episode look, quality presets
  CameraRig.ts  AssetRegistry.ts  types.ts  core/
  vegetation/               SurfaceScatter, GrassBatch, LeafBatch, MossLayer, WindField, recipes
  scenes/                   GroveScene, CanyonScene, OracleScene, BranchScene, StoneScene, CanopyScene, FinaleScene
  rendering/                Materials, PostStack, TransitionPass, Background
  debug/                    DebugControls, CaptureState
public/nature/models/       *.glb          public/nature/textures/   *.webp
public/nature/ASSET_SOURCES.md             one section per asset: script, date, "own work"
assets-src/blender/         common.py, build_*.py, nature.blend
tools/                      capture.mjs and other dev scripts
docs/captures/web/          our screenshots        docs/captures/blender/   Blender preview renders
```

## Episodes, scene sets, timeline

Video time `t` (seconds, 0–59) is the common coordinate: global scroll progress = `t / 59`. Reference frame timestamps are in `docs/nature-webgl-reference/reference-manifest.json`.

| Episode (`scene=`) | Scene set (runtime class) | Video s | Reference frames |
|---|---|---|---|
| `hero` | grove | 0–8 | 01, 02; details 01, 02; clip 06 |
| (T01, inside grove) | grove | 8–10 | 03; `motion/01` |
| `arch` | grove | 10–15.5 | 04, 05; detail 03 |
| `canyon` | canyon | 15.5–21 | 06 |
| `oracle` | oracle | 21–25 | 07, 08; detail 04 |
| `streams` | oracle | 25–31 | 09; detail 05; clip 08 |
| `branch` | branch | 31–37 | 10, 11, 12; details 06, 07; clip 09 |
| `stone` | stone | 37–43.3 | 13, 14 |
| `canopy` | canopy | 43.3–48.5 | 15, 16; detail 08; `motion/03` |
| `canopyClose` | canopy | 48.5–52.8 | 17 |
| `finale` | finale | 52.8–59 | 18, 19, 20; details 09, 10; `motion/04` |

Transitions (approximate windows in video seconds):

- hero → arch, 8.0–10.0: one continuous world, the camera descends past the big overgrown mass (`motion/01`). No fade.
- arch → canyon, 14.6–16.4: light dims, a dark lower form covers the frame, canyon opens out of darkness.
- canyon → oracle, 20.3–21.5: dark crossfade. oracle → streams, 24.5–26: same scene, pose change.
- streams → branch, 30.4–31.4: the dark scene leaves upward with a ragged organic edge, the light branch scene is revealed from below (`motion/02`).
- branch → stone, 37.0–39.5: branch scene slides up with a soft blurred lower edge, the dark stone scene rises from below (frame 13).
- stone → canopy, 43.0–44.3: stone slides up, the canopy grows from bottom-centre on black (`motion/03`).
- canopy → canopyClose, 48–49.5: same scene, closer. canopyClose → finale, 52.4–53.9: leaves part towards the frame edges, the finale appears in the middle (`motion/04`).

At most two scene sets render at once, and only inside a transition window.

## URL API (must keep working)

`/?t=12.5` or `/?scene=arch&progress=0.5` (local 0–1), plus `time=2` (ambient clock, seconds), `seed=134`, `quality=high|medium|low`, `capture=1` (no smoothing, fixed time, overlay hidden, sets `window.__NATURE_READY__ = true` once assets are loaded and the frame is rendered), `debug=1` (panel), `layers=wood,moss,grass,leaves,...`, `wind=0`, `post=0`, `autoplay=1` (plays the 59 s at video pace).

Without `quality=` (and outside `capture=1`) quality is auto: the device default, one level lower on integrated / mobile GPUs (Intel HD/UHD/Iris, Mali, Adreno, PowerVR, Apple on touch; software renderers → low), and it may step down while running (frame-interval p95 > 20 ms over 2 s of steady frames, never up, never near a transition): high → medium → low, then — only where it shrinks the drawing buffer — a last step that caps the DPR at 1.0 inside low (at most three steps; reason "adapt: … → low dpr 1.0 (step 3)", `stats().quality.adapt.dprCap`); `quality=` pins it (`setQuality()` too, which lifts that cap). All presets cap the DPR at 1.5 and differ by pixel budget (3.2 / 2.2 / 1.4 MP), MSAA (4 / 2 / 0) and DOF, so a step down never raises the pixel count. A set that turns ready while on screen fades in over 0.45 s instead of popping (first load, jumps; `state().fading`), the canvas holds the last picture for ≤ 0.3 s while the story passes unprepared sets, and `grove` stays resident (released only by a quality or seed change through the API). `adapt=<ms>` sets that threshold (`adapt=1` forces the step-downs for tests), `adapt=0` turns it off; `state().qualityReason`, `stats().quality`.

Same URL ⇒ same picture. All randomness comes from the seed. Pose is a pure function of progress; wind and ambient motion are a pure function of `timeSec`. Nothing accumulates frame to frame.

## Asset contract (Blender → GLB → engine)

Units are metres at real plant scale (grass blade 4–9 cm, clover 1.5–3 cm, moss fuzz 1–2 cm). Blender is Z-up, export with `export_yup=True`. Cameras in Blender look roughly along +Y. Vertical FOV 35° unless there is a reason. Frame is 1440×1020.
Rough frame height at the focus plane: hero ≈ 1.1 m, arch ≈ 3 m, branch ≈ 1.5 m, finale ≈ 2.5 m.

| GLB | Episodes | Contents |
|---|---|---|
| `grove.glb` | hero, arch | hero masses, the T01 mass, arch trunks, lower branch, cameras |
| `canyon.glb` | canyon | 2–3 rock fragments |
| `branch.glb` | branch | J-branch with broken end, lower-right fragment, ball path anchors |
| `stone.glb` | stone | diagonal rock with recessed seed emblem |
| `canopy.glb` | canopy, canopyClose | twig clusters (leaves are instanced in code from the kit) |
| `finale.glb` | finale | moss hills, standing stone with cut-through seed emblem, far dark forms |
| `kit.glb` | all | leaves, flowers, sprigs for instancing |

Node name prefix decides material and scatter recipe:

- `wood_*` bark; `moss_*` moss cushion or overgrown mass (dense vegetation is scattered on it); `rock_*` rock (triplanar); `stone_*` hero stone; `emblem_*` emblem floor; `far_*` distant dark silhouettes.
- Suffix `__fg` = near-camera foreground, expected out of focus.
- `cam_<episode>_<pose>` cameras: `main` required; `in`, `out`, `p1…pn` optional path poses.
- `key_<episode>` empty: its local −Z is the suggested key light direction. `anchor_*` empties for code-driven objects.

Vertex data on meshes:

- `COLOR_0` (linear RGBA): R = vegetation density 0–1, G = vegetation length multiplier 0–1, B = baked AO/cavity (1 = open), A = 1. The engine never uses it as albedo.
- `TEXCOORD_0` on wood: U around the trunk (integer number of wraps), V along the grain, 1 UV unit = 0.5 m, texels roughly square. Rock and stone are shaded triplanar; any sane UV is fine.
- Transforms and modifiers applied. Normals exported. No materials needed beyond a named placeholder (`mat_wood`, `mat_moss`, `mat_rock`, `mat_stone`, `mat_emblem`, `mat_far`).

Kit meshes: origin at the attachment point, Blender +Z is the growth axis, must look right under a random yaw around it. `COLOR_0`: R = wind flex weight (0 root → 1 tip), G = per-vertex shade, B = 1 on leaf blade / 0 on stem.

Textures are tileable on both axes, WebP, OpenGL normal convention: `bark_basecolor`, `bark_normal`, `bark_orm` (R = AO, G = roughness, B = 0), same for `rock_*` (2048²) and `moss_*` (1024²). Bark grain runs along V.

Budgets: ≤ 250k triangles per scene GLB, ≤ 120k per hero mesh, GLB ≤ 8 MB before compression.
Every output has a headless-runnable build script; `nature.blend` keeps each scene set in its own Blender scene/collection for inspection.

## Verification

Compare at 1440×1020. Capture with `npm run capture`, open the PNG and the matching reference frame, and judge in this order: silhouette and framing, occupied areas, light/dark, colour temperature, plant size, edge softness, micro detail. Our HTML overlay is hidden in captures; ignore the reference's HTML.
"Not done" signals: green cylinders, uniform fur, flat pasted plants, identical leaves on a grid, a swaying trunk, a blurry frame, beauty that comes only from bloom.

## Working alongside other agents

Several agents work in this directory at the same time. Edit only the files your brief gives you. If someone else's file breaks the build, wait and retry, then report it — do not fix it. Only one agent at a time drives the live Blender through MCP; everyone else uses headless Blender. Append your own section to `public/nature/ASSET_SOURCES.md`; do not rewrite other sections.

## Engine API for scene authors

- A scene set is one file in `src/nature/scenes/`, registered in `scenes/index.ts` (`SCENE_FACTORIES`, lazy `import()`). Rewrite the file to replace its PLACEHOLDER; the engine core does not change. Contract types: `src/nature/types.ts`.
- Easiest: `class FooScene extends BaseScene` (`core/BaseScene.ts`), implement `build(ctx)`; optional `animate(frame, local, look)`, `getLook(frame, local, base)`, `getTransitionOverride(tr, frame)`, `debugInfo()`.
- `build` runs once per seed/quality and must be deterministic: all randomness from `ctx.rng("label")`. `animate` must be a pure function of `local.t` (story) and `frame.timeSec` (ambient clock): no state carried between frames, never integrate `deltaSec`.
- `SceneLocal`: `t` (video s of this view — differs from `frame.t` only in reduced-motion crossfades), `episode`, `progress` 0–1, `rawProgress` (< 0 incoming, > 1 outgoing), `setProgress`, `role`, `transition`, `episodeBlend` (inside hero→arch, oracle→streams, canopy→canopyClose), `transparentBackground`.
- `SceneContext`: `assets` (`tryGltf("models/x.glb")` → null when missing, no 404; `loadGltf`, `tryTexture`, `natureTexture("moss_normal")`), `quality` (budgets `grassBlades`, `leaves`, `particles`, …), `wind` (shared uniforms: reference them, never copy), `layers.assign(obj, "moss")`, `reportInstances("grass", n)`, `reducedMotion`, `capture`, `debug`.
- Camera (`this.rig`): `addPosesFromObject(gltf.scene)` turns `cam_<episode>_<pose>` into poses `<episode>_<pose>`; `CameraRig.lookAtPose(pos, target, fov)` for code poses; `setTrack([{ t: 31, pose: "branch_main" }, { t: 37, pose: "branch_out", ease: "inOutSine" }])` in video seconds. Vertical fov is kept on wider screens; narrow screens keep `rig.fit.minHorizontalFraction` of the width and shift the lens towards `rig.fit.subjectX`.
- Lights (`this.lights`): key / hemisphere fill / rim driven by `look.lights`; aim with `setKeyFromObject(key_<episode> empty)` or `setKeyDirection(dir, target)`; per-scene multipliers in `lights.scale`.
- Materials (`rendering/Materials.ts`): `createBarkMaterial`, `createRockMaterial`, `createMossBaseMaterial` (triplanar, metric), `createFarMaterial`, `createEmblemMaterial`. Albedo × baked AO from `COLOR_0.b`. Geometry without COLOR_0 gets a white one when the set is prepared; for meshes created later call `ensureColorAttribute(geometry)` before their first draw (otherwise AO reads 0 = black).
- Looks: `EPISODE_LOOKS` in `SceneConfig.ts` (sRGB hex; background stops are pre-compensated for ACES so the empty background column hits the measured palette). In `getLook` mutate `base` and return it (e.g. `base.exposure *= 0.5`). Fog colour/near/far come from the look; DOF distances are metres from the camera (`nearStart` fully blurred … `nearEnd` sharp); `bloom.strength` 0 skips bloom.
- Transitions: windows, modes (mix / wipe / slide / over) and tuning in `SceneConfig.TRANSITIONS`; slide `follow` 1 = B attached below A, 0 = B stays in place. In mode `over` the outgoing set renders on a transparent clear: no opaque full-screen backdrop there. In `wipe` and `slide` each picture gets its own look's vignette (bands, corners) at its own position before compositing (a sliding picture takes its bottom band along to the seam); `mix` / `over` use the blended look's. Reduced motion turns every other transition into a `mix` with a 0.4 dip through the dip colour.
- Warm-up: before a set is shown the engine compiles its programs (incl. shadow depth variants), lets them link, does their first use, uploads its textures and, when it is not on screen, draws it once off-screen — a few pieces per frame inside the one frame loop, never while a transition that does not show the set plays or is ≤ 0.5 s away (`stats().warm`). A set that renders an extra `Scene` of its own (an off-screen layer) returns it from `warmTargets()` (`{ scene, camera?, offscreen? }`, default off-screen) so it is warmed with the set.
- Debug / tests: `?debug=1` (lil-gui), `window.__NATURE__` (`state()`, `stats()`, `setT`, `setScene`, `setTime`, `setQuality`, `setSeed`, `setFlags`), `window.__NATURE_READY__`. Extra URL flags: `dof=0`, `bloom=0`, `vignette=0`, `tonemap=aces|agx|neutral`, `motion=reduce|full`, `parallax=1`. Capture extras: `--series "8,8.4,8.8"` (→ `<out>_<t>.png`, one page), `--query "post=0"`, `--wait 500`, `--jpeg 82` (JPEG output); it prints the WebGL renderer string.
- Vegetation (`vegetation/`): a recipe in `recipes.ts` is plain data — target meshes (`targets.include` name prefixes / RegExps, optional `targetWeights`) and layers: `grass` (procedural blades: length, width, tilt/bend, colours root→tip, stiffness) or `kit` (kit.glb items with weights, scale, lift, `pale`, clusters). Budgets are shares of `ctx.quality.grassBlades` / `leaves`; density = `COLOR_0.R^densityPower` × area × view importance; per layer `targets` (subset of meshes) and `patches` (noise patches) are optional.
  Pile shading (`PlantShade.ts`; per layer, all off by default: a layer without these options renders exactly as before): `surfaceShade: { strength: 1, darkest: 0.2, soft: 0.45, fill: 0.6, edge: 0.6, tips: 0.25, dir?, radial? }` scales every direct light by a soft terminator of the base-surface normal (the shadow side of a mass stays dark, a back light only reaches the outline) and bends the sky / ground fill normal towards it by `fill`; plants on faces seen edge-on (`edge`) and blade tips (`tips`) are partly spared; `dir: [x, y, z]` instead darkens the albedo by a fixed world direction (the branch look; `plantUniforms(material).uSilvaSurfDir` may move), `radial` takes the base normal from the batch origin (fur on a ball). `clumpAo: { strength, size, depth, gaps: 0.3, variation: 0.12, dome: 0 }`: Voronoi clumps of `size` (default: a grass layer's `tufts` cells, else 0.04 m) with dark cores towards the roots (`depth`: grass 1.3 × canopy, kit 0.8 × plant height), darker borders and a brightness per clump; `dome` (with `surfaceShade`) shades each clump as a small dome. `baseAo` also takes `{ strength, power: 1, open: 1 }` (min(1, COLOR_0.B / open)^power: crevices darker, open faces unchanged); `translucencyFloor` (0.12) is the back-light glow on plants whose base faces the camera; grass `comb: { dir, strength, lay: 0 }` lays the blades towards a direction (hair); `receiveShadow` (true): plants receive the key's shadow map, they never cast. Everything is baked per instance at scatter time (+3 floats base normal, +4 clump; no extra draw calls, no rng draws), per frame only uniforms. Tune `darkest` / `fill` first, then `clumpAo`, and lift the layer colours to win back the mean brightness (examples: `heroMeadow`, `archMoss`).
  Layer extras (optional, defaults unchanged): `count` instead of `budget` asks for a fixed number of plants at every quality (still × `densityScale` × `budgetScale`); `importance: { facingMin, facingFull }` overrides the build's view importance for that layer; kit `upright` (0–1, per layer or item) pulls the growth axis towards world up after `upBias` / `tiltJitter`, and `tiltRange: [min, max]` (rad) replaces `tiltJitter` by a tilt in that range towards a random side (`upright: 1, tiltRange: [0, 0.25]` stands `flower_star` up as cups; the random draws are the same either way); `clumpAo.fine: { size, strength, depth?, gaps?, variation? }` adds a second clump scale inside the first (e.g. 2 cm sponge in 20 cm domes; +3 floats, the occlusions multiply). `veg.report(ctx, prefix)` drops a prefix equal to the set id (`ctx.reportInstances` adds it), so `"stone."` in the stone set reports `stone.fuzz`, not `stone.stone.fuzz`. All plant varyings are `centroid`: under MSAA an edge-on blade no longer throws single firefly pixels (up to ~16 000× white before), so no scene-side radiance clamp is needed for them. `await buildVegetationAsync(ctx, recipe, opts)` is the same build (same random draws, byte-identical instances, identical pixels) run as a step generator (`vegetation/slices.ts`); `buildVegetation` stays synchronous. While the engine runs, build work is frame-paced: after each frame is drawn the frame loop posts one task that runs one job's steps (newest job first) for that frame's budget — 24 ms while only the background is drawn (first load, a jump to an unprepared set), 12 ms while a set the views need is still preparing or the next transition window shows a preparing set, otherwise 8 ms — so no build slice ever lands inside a frame (`yieldEveryMs` applies only without an engine: plain macrotask slices). A build belongs to its `ctx`: when the director releases a set that is still preparing (the story only passed it, or a preload is left behind), its queued jobs are dropped at the next slice boundary, `ctx.rng()` / `ctx.seed` / `ctx.reportInstances()` throw `SliceCancelled`, the prepare unwinds and the director disposes the partial scene (`stats().builds` logs every outcome). Scene authors: no module-level state a cancelled build leaves half-done; `parent` / the target meshes must not move until the promise resolves; keep heavy work inside steps (≈ 1 ms between `yield`s); pass `ctx` as the owner of your own `runSliced(steps, ms, ctx)`; let `SliceCancelled` propagate; everything a build creates hangs under `this.scene` or goes through `this.own()` so `dispose()` frees a partial build; no timers or RAF loops of your own.
- In `build(ctx)`: `const kit = await loadKit(ctx);` then `const veg = buildVegetation(ctx, recipe, { meshes, parent, views, kit, label })` — `views` are `ScatterView`s of the camera poses that will see the plants (plants are scattered once, only where they are seen); `veg.report(ctx, "label.")` → stats / debug panel. Re-use `foregroundMoss` for `*__fg` masses.
- New recipe: copy `heroMeadow` / `archMoss`, keep real plant sizes (never rescale per pose), check the captures at the episode's frames; wind comes from the shared `ctx.wind` uniforms automatically (`wind=0` freezes it). Plant materials: `GrassBatch` / `LeafBatch` (shared `PlantMaterial` translucency; back light only glows on silhouettes).
