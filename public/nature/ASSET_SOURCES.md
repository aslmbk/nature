# Asset sources

One section per shipped asset. Every asset here is own work made in this repository;
no third-party textures, models, scans or HDRIs are used.

## textures/bark_basecolor.webp

- Script: `assets-src/blender/build_textures.py` (helpers: `assets-src/blender/proclib.py`), headless Blender 5.1.2
- Date: 2026-10-04
- Source: own work, generated procedurally
- Notes: 2048², sRGB, tileable on both axes, grain along V, tile = 0.5 m

## textures/bark_normal.webp

- Script: `assets-src/blender/build_textures.py` (helpers: `assets-src/blender/proclib.py`), headless Blender 5.1.2
- Date: 2026-10-04
- Source: own work, generated procedurally
- Notes: 2048², linear data, OpenGL convention (+Y up), tileable

## textures/bark_orm.webp

- Script: `assets-src/blender/build_textures.py` (helpers: `assets-src/blender/proclib.py`), headless Blender 5.1.2
- Date: 2026-10-04
- Source: own work, generated procedurally
- Notes: 2048², linear data, R = AO, G = roughness, B = 0, tileable

## textures/rock_basecolor.webp

- Script: `assets-src/blender/build_textures.py` (helpers: `assets-src/blender/proclib.py`), headless Blender 5.1.2
- Date: 2026-10-04
- Source: own work, generated procedurally
- Notes: 2048², sRGB, tileable, no directional grain (for triplanar use), tile = 1 m

## textures/rock_normal.webp

- Script: `assets-src/blender/build_textures.py` (helpers: `assets-src/blender/proclib.py`), headless Blender 5.1.2
- Date: 2026-10-04
- Source: own work, generated procedurally
- Notes: 2048², linear data, OpenGL convention (+Y up), tileable

## textures/rock_orm.webp

- Script: `assets-src/blender/build_textures.py` (helpers: `assets-src/blender/proclib.py`), headless Blender 5.1.2
- Date: 2026-10-04
- Source: own work, generated procedurally
- Notes: 2048², linear data, R = AO, G = roughness, B = 0, tileable

## textures/moss_basecolor.webp

- Script: `assets-src/blender/build_textures.py` (helpers: `assets-src/blender/proclib.py`), headless Blender 5.1.2
- Date: 2026-10-04
- Source: own work, generated procedurally
- Notes: 1024², sRGB, tileable, tile = 0.3 m

## textures/moss_normal.webp

- Script: `assets-src/blender/build_textures.py` (helpers: `assets-src/blender/proclib.py`), headless Blender 5.1.2
- Date: 2026-10-04
- Source: own work, generated procedurally
- Notes: 1024², linear data, OpenGL convention (+Y up), tileable

## textures/moss_orm.webp

- Script: `assets-src/blender/build_textures.py` (helpers: `assets-src/blender/proclib.py`), headless Blender 5.1.2
- Date: 2026-10-04
- Source: own work, generated procedurally
- Notes: 1024², linear data, R = AO, G = roughness, B = 0, tileable

## models/kit.glb

- Script: `assets-src/blender/build_kit.py` (helpers: `assets-src/blender/proclib.py`), headless Blender 5.1.2
- Date: 2026-10-04
- Source: own work, generated procedurally
- Notes: 16 instancing meshes (leaves, flowers, sprigs, seed head, canopy leaves and twigs), one node each at the origin, material `mat_kit`, COLOR_0 = flex / shade / blade mask

## textures/kit_basecolor.webp

- Script: `assets-src/blender/build_kit.py` (helpers: `assets-src/blender/proclib.py`), headless Blender 5.1.2
- Date: 2026-10-04
- Source: own work, generated procedurally
- Notes: 1024², sRGB, 4 × 4 atlas of 256 px cells used by `models/kit.glb` (not tileable)

## models/grove.glb

- Script: `assets-src/blender/build_grove.py` (helpers: `assets-src/blender/common.py`), headless Blender 5.1.2; scene `grove` in `assets-src/blender/nature.blend`
- Date: 2026-10-04
- Source: own work, generated procedurally (signed-distance masses meshed with Blender's bundled OpenVDB, swept trunks, moss cushions; seed 134, deterministic)
- Notes: hero / T01 / arch world: 12 meshes (`moss_*`, `wood_*`, two `__fg`), 213,935 triangles, 6.2 MB; COLOR_0 = density / length / AO; wood TEXCOORD_0 along the grain (1 unit = 0.5 m); 7 cameras `cam_hero_*` / `cam_arch_*` (35° vertical FOV), empties `key_hero` / `key_arch`; no textures, lights or animation
- grove.glb — revision 2 (2026-10-08): same script and seed, own work, rebuilt headless in Blender 5.1.2 (`--export --validate --render --compare`; the `grove` scene in `nature.blend` is not refreshed yet: run the script with `--save` once the live Blender session is free). Arch trunks: mostly bare flowing bark with thick, lumpy moss cushions (packed 2.5–7.5 cm pillows, soft irregular borders) over the top and outer side, a big bare oval on the left trunk, a bare window on the right one (moss ≈ 48 % of the left trunk, 57 % of the right, 14 % of the low branch). T01 mass: rounded crest and a lobed keyhole edge fitted in the engine's 8.4 s / 9.1 s camera poses; `cam_hero_p1` / `cam_hero_p2` moved back to ~2 m from the wall. `moss_arch_ground` / `moss_arch_near__fg`: packed moss hummocks (8–22 cm across) on low swells instead of smooth blobs. 12 meshes, 241,648 triangles, 6.8 MB; node names, cameras, empties, COLOR_0 and UV conventions unchanged
- grove.glb — revision 3 (2026-10-09): same script and seed, own work, rebuilt headless in Blender 5.1.2 (`--export --validate --render --compare`, no `--save`: the `grove` scene in `nature.blend` still holds an older revision; run the script with `--save` once the live Blender session is free). Hero (frame 01): the top-left mass is a thick diagonal arm with a defined lower edge and a rounded nose, the left column a narrow neck over its foot (outlines = the frame-01 silhouette eroded by the measured plant fringe; the column's outline carries a short pile in COLOR_0). T01: an asymmetric diagonal left flank at 8.4 s instead of a dome; a bulge with a crease above it on the keyhole edge (9.1-9.6 s) with a denser, slightly longer pile in COLOR_0. Arch (frame 04, details/03): ragged cushion outlines (about +-3.5 cm), a rolled lip overhanging the bark by 2-5 cm, a thin band of plants on the bark along the lip (COLOR_0.R 0.7 -> 0 within 3 cm, 0 under the moss) and contact occlusion in COLOR_0.B below it; the right trunk's bare window has a narrower, noisier edge. Exit (frame 05): `moss_arch_near__fg` has an uneven top and the window is a wide V instead of a narrow slit; the low branch's bend sits about 40 px further left in frame 04 and its leg continues below frame 04 as the left side of the window. Duplicate triangles left by the SDF decimation are removed (the exporter's "mesh is not valid" warnings are gone). 12 meshes, 240,566 triangles (largest `moss_crown_mass` 45,997), 6.8 MB (7,141,464 bytes); node names, cameras, empties, COLOR_0 and UV conventions unchanged; a headless rebuild is byte-identical

## models/branch.glb

- Script: `assets-src/blender/build_branch.py` with `assets-src/blender/branchlib.py` (branch-only helpers on top of `assets-src/blender/common.py`), headless Blender 5.1.2; scene `branch` in `assets-src/blender/nature.blend` (the .blend was not re-saved in the 2026-10-08 / 2026-10-09 geometry revisions: built headless without `--save`)
- Date: 2026-10-09 (geometry revision 3: stumps and torn fibres in their own mesh with fresh-wood marking, wider bark channel through the bend and on to the broken end, ragged channel border); revision 2: 2026-10-08 (twisted log, ragged bark channel, lumpy moss, splintered stumps, tapered fragment)
- Source: own work, generated procedurally (swept logs with a variable grain twist, rope-like strands with cross-fissures and deep grain-following cracks, splintered fracture caps; swept stumps with oval / kidney sections, furrowed bark and torn tops; moss coats with Voronoi clumps, billows and a sagging underside; a moss annulus over the broken rim; seed 134, deterministic: a headless rebuild is byte-identical)
- Notes: branch episode (video 31-37 s): 6 meshes (`wood_branch_j` 87,296 tris, `wood_branch_stumps` 25,056 = the two stumps and all loose / torn fibres, `moss_branch_j` 57,744 incl. the end annulus and the stumps' moss sheaths and cushions, `wood_branch_frag` 29,952, `moss_branch_frag` 11,100, `moss_ball` 1,280 at the origin), 212,428 triangles, 5.7 MB; COLOR_0 = density / length / AO (wood: R 0.4-0.8 on the bark within ~3 cm of the moss border and in the deep cracks, up to 0.6 in small patches, 0 on clean bark and under the moss; `wood_branch_stumps`: G = 0 only on fresh wood - the breaks, the splinter lips and the torn fibre ends, which also have R = 0 and B 0.9-1.0 - and G >= 0.3 everywhere else; R >= 0.6 in the moss spots on the stumps' upper faces); wood TEXCOORD_0 U around (integer wraps, follows the twisted grain), V along the grain (1 unit = 0.5 m); 3 cameras `cam_branch_main` / `cam_branch_p1` / `cam_branch_out` (35 deg vertical FOV, one position, tilt only), empty `key_branch` (local -Z = glTF (-0.12, -0.70, -0.70) normalised, extras `light_dir_gltf`), 12 empties `anchor_ball_0..11` (closed loop of the fuzzy ball, see the `anchor_ball_0` extras; loop 0.505 m clear of every mesh); cameras, empties and `moss_ball` identical to revision 2 (names, transforms, extras); no textures, lights or animation. The reference frames were only used to fit silhouettes, the bark / moss split and the ball path (comparison sheets in `docs/captures/blender/`, silhouette IoU vs frame 10: 0.888 main / 0.855 p1); nothing from `docs/` is in the GLB

## models/finale.glb

- Script: `assets-src/blender/build_finale.py` (helpers: `assets-src/blender/stonelib.py`, `assets-src/blender/common.py`), headless Blender 5.1.2; scene `finale` in `assets-src/blender/nature.blend` is written by the script's `--save` (revision 3 was built headless without `--save`: `nature.blend` was not re-saved with it)
- Date: 2026-10-09 (revision 3: mound pile, far trunks, portrait bank)
- Source: own work, generated procedurally (signed-distance moss plateau carved into a pile of round mounds with 10–15 cm bumps on top — rolling-ball closing, grey opening, the crest clamped to the line traced on frame 19 — standing stone, near mound and far trunks / rock lumps meshed with Blender's bundled OpenVDB, a heightfield moss bank under the near mound; the Silva seed emblem built from the construction in `assets-src/emblem/seed.svg` and cut through the stone with an exact boolean; seed 134, deterministic)
- Notes: finale episode (video 52.8–59 s): 5 meshes (`stone_finale` with the seed cut through it, `moss_finale_hills`, `moss_finale_near__fg`, `far_finale_left`, `far_finale_right`), 209,007 triangles (`moss_finale_hills` 119,999, `stone_finale` 57,528, `moss_finale_near__fg` 13,480 = near mound 7,000 + bank 6,480, far forms 9,000 each), 7.6 MB; `moss_finale_hills` is a pile of 68 round mounds 25–60 cm across and 8–20 cm high with dark creases between them and 820 bumps on top, its crest within 0.005 frame heights (low-pass) of the line traced on frame 19; the far forms are leaning trunks merged with rock lumps 7–10 m behind the stone; the bank continues the near mound towards and behind the cameras under the lowest 35° bottom-edge plane of the engine's camera track, so 9:19.5 portrait frames (60° and 69.5° vertical FOV) show moss down to the bottom edge while the 1440×1020 frames only change where background showed under the lip (54.0–55.6 s); COLOR_0 = density / length / AO (hills: R ≥ 0.6 on every vertex a camera pose sees, sparse only down the sagging front face, B about 0.3–0.5 in the creases and 1 on the crowns; near mound and bank: R ≈ 0.2, B ≤ 0.18, under the island R = 0 and the island's key shadow in B; stone: R = moss in the crack and at the foot); box-projected TEXCOORD_0 (1 unit = 0.5 m); 3 cameras `cam_finale_in` / `main` / `out` (35° vertical FOV, extras focus distance, frame height, video time, `dolly_from_in` on main), empty `key_finale`; no textures, lights or animation

## models/stone.glb

- Script: `assets-src/blender/build_stone.py` (helpers: `assets-src/blender/stonelib.py`, `assets-src/blender/common.py`), headless Blender 5.1.2; scene `stone` in `assets-src/blender/nature.blend` is written by the script's `--save` (revision 3 was built headless without `--save`: `nature.blend` was not re-saved with it)
- Date: 2026-10-09 (revision 3: clean recess lip, top-right edge raised into the frame corner)
- Source: own work, generated procedurally (signed-distance rock fragment whose right face turns away in one smooth roll, with chipped edges, cracks and shallow pits, its top-right edge raised so it runs on into the frame corner, meshed with Blender's bundled OpenVDB; the Silva seed from `assets-src/emblem/seed.svg`'s construction recessed 4 cm with a 4 mm chamfer by an exact boolean, the polygons along the rim triangulated and given analytic corner normals (face / chamfer / walls), its floor split into its own mesh; moss strip of signed-distance cushions, thick on the left edge and the corner, a thin fringe on the right; seed 134, deterministic)
- Notes: stone episode (video 37–43.3 s): 3 meshes (`rock_stone_slab`, `emblem_stone` = recess floor with planar TEXCOORD_0 in the emblem plane, `moss_stone_edge`), 138,504 triangles, 4.1 MB; COLOR_0 = density / length / AO (rock: B ≥ 0.95 within 3 mm of the recess lip; emblem floor: contact shadow along the walls in B, never below 0.3; moss strip R 0.65 on the fringe to 1 on the cushion); rock UVs box-projected (1 unit = 0.5 m); 3 cameras `cam_stone_in` / `main` / `out` (pedestal moves only, extras incl. `composition_shift_fh` and the measured `exit_track`), empty `key_stone` (extras `light_target_gltf`, `falloff_radius_m`); no textures, lights or animation

## models/canyon.glb

- Script: `assets-src/blender/build_canyon.py` (helpers: `assets-src/blender/canopylib.py`, `assets-src/blender/common.py`), headless Blender 5.1.2; scene `canyon` in `assets-src/blender/nature.blend`
- Date: 2026-10-08
- Source: own work, generated procedurally (signed-distance rock masses broken into jointed, layered slabs and meshed with Blender's bundled OpenVDB, moss cushions; seed 134, deterministic: a headless rebuild is byte-identical)
- Notes: canyon episode (video 15.5–21 s): 5 meshes (`rock_canyon_right`, `rock_canyon_left`, `rock_canyon_low__fg`, `moss_canyon_right`, `moss_canyon_left`), 106,639 triangles, 4.2 MB; COLOR_0 = vegetation density (ledges in the plant zones only) / length / AO; 3 cameras `cam_canyon_in` / `cam_canyon_main` / `cam_canyon_out` (35° vertical FOV), empty `key_canyon`; no textures, lights or animation. The reference frames were only used to fit silhouettes (comparison sheets in `docs/captures/blender/`); nothing from `docs/` is in the GLB

## models/canopy.glb

- Script: `assets-src/blender/build_canopy.py` (helpers: `assets-src/blender/canopylib.py`, `assets-src/blender/common.py`), headless Blender 5.1.2; scene `canopy` in `assets-src/blender/nature.blend`
- Date: 2026-10-08
- Source: own work, generated procedurally (branching twig tubes grown towards a lumpy crown surface, volumetric AO from a leaf-density grid; seed 134, deterministic: a headless rebuild is byte-identical)
- Notes: canopy / canopyClose episodes (video 43.3–52.8 s): twig skeleton only, no leaves (the engine instances `kit.glb` leaves on it): 46 clusters `wood_canopy_c00`…`wood_canopy_c45` (origin at the cluster base; extras `reveal_order`, `depth`, `radius_m`), `wood_canopy_scaffold`, dark shells `far_canopy_inner` / `far_canopy_outer`; 192,080 triangles, 6.4 MB; COLOR_0 = leaf density / leaf size / AO; wood TEXCOORD_0 along the grain (1 unit = 0.5 m); 5 cameras `cam_canopy_in` / `cam_canopy_main` / `cam_canopy_out` / `cam_canopyClose_main` / `cam_canopyClose_out` (35° vertical FOV), empty `key_canopy`; no textures, lights or animation. The previews in `docs/captures/blender/canopy_*` show stand-in leaves generated for the previews only; the reference frames were only used to fit the crown outline; nothing from `docs/` is in the GLB

## poster.jpg

- Script: `tools/capture.mjs` (`node tools/capture.mjs --t 0.5 --time 0.5 --jpeg 82 --out public/nature/poster.jpg`), the running app rendered by headless system Edge on the local GPU
- Date: 2026-10-09
- Source: own work, a screenshot of this project's own hero render (grove scene set, video t = 0.5 s, ambient time 0.5 s, seed 134, quality high, 1440×1020, `capture=1` so no HTML overlay); nothing from `docs/` is in it
- Notes: 1440×1020 JPEG, quality 82, 196 KB; the still that `src/components/NatureStage.tsx` shows (object-fit: cover) when WebGL 2 is unavailable or the engine cannot load. Re-render it with the same command after the hero changes
