/**
 * Vegetation recipes: plain data describing what grows where. A recipe names its
 * target surfaces (node-name prefixes / names / regexes of the scene GLB) and a list
 * of layers:
 *  - `grass` layers (GrassBatch): procedural blades — length / width / tilt / bend /
 *    droop / lean field / outliers, colours root → tip, dry tips, stiffness, budget as
 *    a share of `quality.grassBlades`;
 *  - `kit` layers (LeafBatch): instanced kit.glb items with weights, scale ranges,
 *    lift, tint / pale variation, rim preference, clusters, budget as a share of
 *    `quality.leaves`.
 * Density everywhere = COLOR_0.R ^ densityPower × the views' importance × area;
 * blade length follows COLOR_0.G. Any layer may narrow its surfaces (`targets`, a
 * subset of the recipe's meshes) and grow in noise patches (`patches`). Build one
 * with `buildVegetation()` (Vegetation.ts).
 *
 * Sizes are real plant sizes in metres (CLAUDE.md: blades 4–9 cm, clover 1.5–3 cm,
 * moss fuzz 1–2 cm). Never rescale plants per camera pose.
 */
import type { GrassLayerSpec } from "./GrassBatch";
import type { KitLayerSpec } from "./LeafBatch";

export type VegetationLayerSpec = GrassLayerSpec | KitLayerSpec;

export type NameMatcher = string | RegExp;

export interface VegetationRecipe {
  name: string;
  /**
   * Target meshes. A string matches a node-name prefix (`"moss_"`) or an exact name;
   * a RegExp is tested against the node name. `exclude` wins over `include`.
   */
  targets: { include: NameMatcher[]; exclude?: NameMatcher[] };
  /** Per-target density multiplier by name match (first match wins, default 1). */
  targetWeights?: [NameMatcher, number][];
  layers: VegetationLayerSpec[];
  /** Multiplies every layer budget (default 1). */
  densityScale?: number;
}

export function matchesName(name: string, m: NameMatcher): boolean {
  return typeof m === "string" ? name === m || name.startsWith(m) : m.test(name);
}

// ---------------------------------------------------------------------------
// hero meadow — S01 / T01 masses: long tangled blades, broad pale leaves on
// petioles above the grass, a few sprigs, dry seed heads along the rims
// ---------------------------------------------------------------------------

export const heroMeadow: VegetationRecipe = {
  name: "heroMeadow",
  targets: { include: ["moss_crown_mass", "moss_left_column", "moss_hero_upper", "moss_hero_near__fg"] },
  layers: [
    {
      // a tangled mat (details 01 / 02): thin blades leave the surface anywhere inside a
      // wide cone, lean out of small swirling tufts and curl, a third of them back across
      // themselves; many short ones inside, a few long pale ones on the outlines
      kind: "grass",
      name: "grass",
      // thinner blades than before, more of them: the same cover of the masses
      budget: 3.2,
      length: [0.04, 0.1],
      lengthSkew: 1.4,
      lengthMask: 0.6,
      // hair-thin stalks (detail 01): ≈ 1.5–2.5 px at the hero's 1.1 m frame height
      width: [0.0016, 0.0028],
      tipWidth: 0.12,
      segments: 5,
      cone: 0.7,
      tilt: [0.15, 0.8],
      bend: [1.1, 2.6],
      bendLength: 0.7,
      maxAngle: 2.5,
      bendFlip: 0.2,
      twist: 1.0,
      droop: 0.3,
      lean: { strength: 0.1, scale: 5 },
      tufts: { size: 0.04, strength: 0.6, swirl: 0.7 },
      // detail 01: the overhang's pile flows down and to the left instead of netting
      comb: { dir: [-1, -0.75, 0.2], strength: 0.4, lay: 0.2 },
      outliers: { fraction: 0.02, scale: 1.7, rimBoost: 2.5, pale: 0.55 },
      // soft, desaturated olive-yellow-green (frame 01 mound #788752), many pale / dry stalks
      colors: { root: "#4E5628", mid: "#728240", tip: "#A4B466", tip2: "#8EA050", dry: "#C0C496" },
      dryFraction: 0.1,
      brightnessJitter: 0.22,
      rootAo: 0.7,
      canopy: 0.03,
      stiffness: [0.75, 1.25],
      roughness: 0.75,
      translucency: 1.6,
      roundness: 0.55,
      densityPower: 1.5,
      // a fringe of blades on the outlines (the leaves keep off them)
      rimBias: 0.8,
      clusterSize: 0.07,
      chunks: 10,
      // pile shading (PlantShade.ts): the mats follow the masses' form, tufts have dark cores
      surfaceShade: { darkest: 0.3, soft: 0.45, fill: 0.7, edge: 0.7, tips: 0.6 },
      clumpAo: { strength: 0.25, gaps: 0.15, variation: 0.15, dome: 0.9 },
      translucencyFloor: 0.03,
      baseAo: { strength: 0.9, power: 2, open: 0.92 },
    },
    {
      // the dark short underlayer of the mat: fills the ground between the long blades
      kind: "grass",
      name: "fuzz",
      budget: 0.4,
      length: [0.012, 0.028],
      lengthMask: 0.3,
      width: [0.0014, 0.0024],
      segments: 3,
      cone: 0.8,
      tilt: [0.2, 0.9],
      bend: [0.4, 1.6],
      bendFlip: 0.4,
      droop: 0.2,
      lean: { strength: 0.1, scale: 9 },
      tufts: { size: 0.025, strength: 0.5, swirl: 0.8 },
      outliers: { fraction: 0, scale: 1 },
      colors: { root: "#474C26", mid: "#686F3A", tip: "#8E9650", tip2: "#808A46", dry: "#A09C6C" },
      dryFraction: 0.02,
      rootAo: 0.45,
      canopy: 0.02,
      stiffness: [1.2, 1.8],
      translucency: 0.5,
      densityPower: 1,
      chunks: 4,
      surfaceShade: { darkest: 0.15, soft: 0.4, fill: 0.7, edge: 0.5, tips: 0.1 },
      clumpAo: { strength: 0.3, gaps: 0.25, variation: 0.15, dome: 0.9 },
      translucencyFloor: 0.03,
    },
    {
      // frames 01 / 02: matte sage leaves on thin petioles above the mat, tilted every
      // way, the ones facing the sky pale to whitish; few and small on the low mound and
      // the column (the arm has its own layer, armLeaves)
      kind: "kit",
      name: "leaves",
      layer: "leaves",
      viewTags: ["hero"],
      // ≈ 290 on the mound, 60 on the column (as before the arm got its own layer)
      budget: 0.022,
      targets: ["moss_crown_mass", "moss_left_column"],
      targetWeights: [
        ["moss_crown_mass", 0.42],
        // frame 01: the left column is mostly dark grass
        ["moss_left_column", 0.5],
      ],
      // lighter, warmer sage than the atlas (frame 01: #8FA582 mid, whitish-sage tops),
      // slightly glossy
      items: [
        { name: "leaf_round_a", weight: 2, scale: [0.6, 1.05], lift: [0.02, 0.04], tint: "#FAF0B4", pale: { chance: 0.85, amount: [0.3, 0.9], upFacing: 0.85 } },
        { name: "leaf_round_b", weight: 2, scale: [0.6, 1.1], lift: [0.02, 0.04], tint: "#FAF0B4", pale: { chance: 0.85, amount: [0.3, 0.9], upFacing: 0.85 } },
        { name: "leaf_round_c", weight: 1, scale: [0.55, 0.95], lift: [0.012, 0.03], tint: "#FAF0B4", pale: { chance: 0.85, amount: [0.3, 0.9], upFacing: 0.85 } },
        { name: "leaf_serrated_a", weight: 2.5, scale: [0.6, 1.05], lift: [0.02, 0.04], tint: "#FAF0B4", pale: { chance: 0.75, amount: [0.3, 0.85], upFacing: 0.85 } },
        { name: "leaf_serrated_b", weight: 2.5, scale: [0.6, 1.1], lift: [0.02, 0.04], tint: "#FAF0B4", pale: { chance: 0.75, amount: [0.3, 0.85], upFacing: 0.85 } },
        { name: "sprig_a", weight: 0.5, scale: [0.7, 1.05], lift: [0.01, 0.025] },
      ],
      upBias: 0.1,
      tiltJitter: 1.1,
      scaleSkew: 1.5,
      saturation: 0.85,
      // never hanging under the overhang's chin (the lit grass fringe is there)
      facing: { min: -0.9, max: -0.5 },
      cluster: { size: [1, 3], radius: 0.035, sameSpecies: 0.6 },
      patches: { scale: 5, threshold: 0.38, softness: 0.3 },
      densityPower: 1.5,
      stiffness: [0.8, 1.3],
      flutter: 1.2,
      translucency: 0.8,
      roughness: 0.55,
      brightness: [0.95, 1.4],
      baseAo: 0.6,
      chunks: 2,
      surfaceShade: { darkest: 0.4, soft: 0.5, fill: 0.35, edge: 0.5, tips: 0 },
      translucencyFloor: 0.05,
    },
    {
      // the arm (frame 01, detail 01): its body is combed grass; smaller sage leaves sit
      // mostly along the lower edge and around the nose (the outlines of the hero views)
      kind: "kit",
      name: "armLeaves",
      layer: "leaves",
      viewTags: ["hero"],
      budget: 0.028,
      targets: ["moss_hero_upper"],
      rimBias: 4,
      rimPower: 1.5,
      rimWidth: 0.6,
      items: [
        { name: "leaf_round_a", weight: 2, scale: [0.5, 0.85], lift: [0.015, 0.035], tint: "#FAF0B4", pale: { chance: 0.7, amount: [0.25, 0.8], upFacing: 0.85 } },
        { name: "leaf_round_b", weight: 2, scale: [0.5, 0.9], lift: [0.015, 0.035], tint: "#FAF0B4", pale: { chance: 0.7, amount: [0.25, 0.8], upFacing: 0.85 } },
        { name: "leaf_serrated_a", weight: 3, scale: [0.5, 0.85], lift: [0.015, 0.035], tint: "#FAF0B4", pale: { chance: 0.6, amount: [0.25, 0.75], upFacing: 0.85 } },
        { name: "leaf_serrated_b", weight: 3, scale: [0.5, 0.9], lift: [0.015, 0.035], tint: "#FAF0B4", pale: { chance: 0.6, amount: [0.25, 0.75], upFacing: 0.85 } },
      ],
      upBias: 0.1,
      tiltJitter: 1.1,
      scaleSkew: 1.5,
      saturation: 0.85,
      facing: { min: -0.9, max: -0.5 },
      cluster: { size: [1, 3], radius: 0.035, sameSpecies: 0.6 },
      densityPower: 1.5,
      stiffness: [0.8, 1.3],
      flutter: 1.2,
      translucency: 0.8,
      roughness: 0.55,
      brightness: [0.95, 1.4],
      baseAo: 0.6,
      chunks: 1,
      surfaceShade: { darkest: 0.4, soft: 0.5, fill: 0.35, edge: 0.5, tips: 0 },
      translucencyFloor: 0.05,
    },
    {
      // the wall the camera descends past (T01, frame 03): many smaller sage leaves over
      // the whole face, lighter than the dark grass between them even in its shade
      kind: "kit",
      name: "wallLeaves",
      layer: "leaves",
      viewTags: ["wall"],
      budget: 1.0,
      targets: ["moss_crown_mass", "moss_left_column"],
      targetScale: [["moss_crown_mass", 0.8]],
      // the left column (the keyhole's other side, also in the hero frames) keeps a few
      targetWeights: [["moss_left_column", 0.2]],
      // only on faces turned to the (lowered, right-shifted) wall views: the keyhole's edge,
      // the tops of its lobes and the chin bulge (lit grass in frame 03) stay grass
      importance: { facingMin: 0.35, facingFull: 0.75 },
      rimBias: -0.9,
      rimWidth: 0.5,
      items: [
        { name: "leaf_round_a", weight: 1.5, scale: [0.6, 1.0], lift: [0.015, 0.035], tint: "#FFF0B8", pale: { chance: 0.5, amount: [0.25, 0.6], upFacing: 0.6 } },
        { name: "leaf_round_b", weight: 1.5, scale: [0.6, 1.05], lift: [0.015, 0.035], tint: "#FFF0B8", pale: { chance: 0.5, amount: [0.25, 0.6], upFacing: 0.6 } },
        { name: "leaf_serrated_a", weight: 3, scale: [0.6, 1.0], lift: [0.015, 0.035], tint: "#FFF0B8", pale: { chance: 0.45, amount: [0.25, 0.6], upFacing: 0.6 } },
        { name: "leaf_serrated_b", weight: 3, scale: [0.6, 1.05], lift: [0.015, 0.035], tint: "#FFF0B8", pale: { chance: 0.45, amount: [0.25, 0.6], upFacing: 0.6 } },
      ],
      upBias: 0.35,
      tiltJitter: 0.85,
      scaleSkew: 1.4,
      saturation: 0.8,
      cluster: { size: [1, 4], radius: 0.04, sameSpecies: 0.6 },
      patches: { scale: 4, threshold: 0.35, softness: 0.35 },
      densityPower: 1.5,
      stiffness: [0.8, 1.3],
      flutter: 1.2,
      translucency: 0.6,
      roughness: 0.8,
      brightness: [1.15, 1.6],
      baseAo: 0.7,
      chunks: 2,
      surfaceShade: { darkest: 0.4, soft: 0.5, fill: 0.35, edge: 0.5, tips: 0 },
      translucencyFloor: 0.05,
    },
    {
      // tiny pale dots along the silhouettes: dry seed heads and white buds on thin stalks
      // (frame 01, detail 01), mostly where the masses turn away from the lens
      kind: "kit",
      name: "rims",
      layer: "flowers",
      viewTags: ["hero"],
      // the column at the crown's weight: the arm and the mound keep their count
      budget: 0.088,
      targetWeights: [
        ["moss_crown_mass", 0.3],
        ["moss_left_column", 0.3],
      ],
      // the buds and seed heads of the mound and the column stay low in the grass (frame
      // 01: nothing sticks up above their outlines but leaves and a few pale blades)
      targetScale: [
        ["moss_crown_mass", 0.45],
        ["moss_left_column", 0.45],
      ],
      rimBias: 14,
      rimPower: 2,
      items: [
        { name: "seedhead_a", weight: 3, scale: [0.8, 1.3], lift: [0.01, 0.03], upBias: 0.4 },
        { name: "flower_white_small", weight: 2.5, scale: [1.0, 1.6], lift: [0.03, 0.06] },
      ],
      upBias: 0.3,
      tiltJitter: 0.55,
      scaleSkew: 1.3,
      cluster: { size: [2, 5], radius: 0.025, sameSpecies: 0.7 },
      densityPower: 1,
      stiffness: [0.6, 1.0],
      flutter: 0.6,
      translucency: 0.5,
      roughness: 0.85,
      brightness: [1.1, 1.4],
      baseAo: 0.3,
      chunks: 2,
    },
  ],
};

// ---------------------------------------------------------------------------
// foreground mass (`*__fg`) — out of focus, close to the lens, in the shade of the
// frame's lower edge: only short, near-black green growth that hugs the surface
// (anything taller would tower over the frame) and a few faint dark leaf shapes
// breaking the blurred edge (detail 02)
// ---------------------------------------------------------------------------

export const foregroundMoss: VegetationRecipe = {
  name: "foregroundMoss",
  targets: { include: [/__fg$/] },
  layers: [
    {
      kind: "grass",
      name: "fgGrass",
      budget: 0.05,
      length: [0.014, 0.032],
      lengthMask: 0.6,
      width: [0.0018, 0.003],
      segments: 3,
      cone: 0.7,
      tilt: [0.2, 0.9],
      bend: [0.3, 1.4],
      bendFlip: 0.4,
      droop: 0.4,
      lean: { strength: 0.15, scale: 6 },
      tufts: { size: 0.03, strength: 0.5, swirl: 0.7 },
      outliers: { fraction: 0.02, scale: 1.4 },
      colors: { root: "#28301A", mid: "#56633C", tip: "#8A9470", tip2: "#7C8862", dry: "#8C8A70" },
      dryFraction: 0.02,
      rootAo: 0.4,
      stiffness: [1.0, 1.5],
      translucency: 0.15,
      densityPower: 1.2,
      chunks: 3,
      surfaceShade: { darkest: 0.25, soft: 0.5, fill: 0.7, edge: 0.4, tips: 0.2 },
      clumpAo: { strength: 0.25, gaps: 0.25, variation: 0.15, dome: 0.6 },
    },
    {
      kind: "kit",
      name: "fgLeaves",
      layer: "leaves",
      // few: their blurred shapes break the band's edge only here and there (detail 02)
      budget: 0.014,
      items: [
        { name: "leaf_round_a", weight: 3, scale: [0.55, 0.85], lift: [0.0, 0.006] },
        { name: "leaf_round_b", weight: 3, scale: [0.6, 0.9], lift: [0.0, 0.006] },
        { name: "leaf_serrated_a", weight: 2, scale: [0.55, 0.8], lift: [0.0, 0.004] },
      ],
      upBias: 0.15,
      tiltJitter: 0.6,
      saturation: 0.45,
      cluster: { size: [2, 4], radius: 0.03, sameSpecies: 0.7 },
      densityPower: 1.2,
      stiffness: [1.0, 1.4],
      flutter: 0.8,
      translucency: 0.1,
      roughness: 0.9,
      brightness: [0.7, 0.9],
      chunks: 2,
      surfaceShade: { darkest: 0.3, soft: 0.5, fill: 0.5, edge: 0.4, tips: 0 },
    },
  ],
};

// ---------------------------------------------------------------------------
// arch moss — S02 cushions on the trunks: short dense blades, clover, tiny white
// flowers, the odd sprig; sparse clover spots on the bare wood next to the moss
// ---------------------------------------------------------------------------

export const archMoss: VegetationRecipe = {
  name: "archMoss",
  targets: { include: ["moss_arch_left", "moss_arch_right", "moss_arch_low", "moss_arch_ground", "moss_arch_near__fg", "wood_arch_"] },
  targetWeights: [
    ["wood_", 0.6],
    ["moss_arch_ground", 0.7],
    // frame 04: the foreground band stays dark under the blur — only a few plants there
    ["moss_arch_near__fg", 0.12],
  ],
  layers: [
    {
      kind: "grass",
      name: "grass",
      // fine grass over the cushions (detail 03: blades of 15–30 px at ≈ 340 px/m)
      budget: 1.0,
      length: [0.03, 0.065],
      lengthSkew: 1.1,
      lengthMask: 0.8,
      width: [0.0016, 0.0026],
      tipWidth: 0.15,
      segments: 4,
      tilt: [0.15, 0.8],
      bend: [0.2, 1.1],
      twist: 0.7,
      droop: 0.45,
      normalJitter: 0.4,
      lean: { strength: 0.45, scale: 7 },
      outliers: { fraction: 0.05, scale: 1.6 },
      colors: { root: "#2A340F", mid: "#627628", tip: "#C0CC5E", tip2: "#A2B64C", dry: "#C8BC84" },
      dryFraction: 0.05,
      brightnessJitter: 0.16,
      rootAo: 0.3,
      baseAo: { strength: 0.9, power: 1.6, open: 0.9 },
      stiffness: [0.8, 1.3],
      roughness: 0.62,
      translucency: 0.9,
      densityPower: 1.3,
      // more blades where the cushions turn away from the camera: fuzzy outlines (detail 03)
      rimBias: 3,
      clusterSize: 0.05,
      chunks: 6,
      surfaceShade: { darkest: 0.22, soft: 0.45, fill: 0.6, edge: 0.6, tips: 0.3 },
      clumpAo: { strength: 0.3, size: 0.035, gaps: 0.25, variation: 0.15, dome: 0.8 },
      translucencyFloor: 0.03,
    },
    {
      // the dense short fuzz that makes the cushions soft instead of clay (detail 03):
      // 0.8–2 cm blades, tangled in tiny tufts, light tips, thicker on the outlines
      kind: "grass",
      name: "fuzz",
      budget: 1.2,
      length: [0.01, 0.022],
      lengthSkew: 1.2,
      lengthMask: 0.5,
      width: [0.0011, 0.0018],
      tipWidth: 0.2,
      segments: 3,
      cone: 0.9,
      tilt: [0.1, 0.7],
      bend: [0.3, 1.2],
      bendFlip: 0.4,
      twist: 0.6,
      droop: 0.2,
      lean: { strength: 0.05, scale: 9 },
      tufts: { size: 0.02, strength: 0.4, swirl: 0.6 },
      colors: { root: "#1E2A0C", mid: "#58722A", tip: "#BED066", tip2: "#A8C257", dry: "#B4AE7A" },
      dryFraction: 0.03,
      brightnessJitter: 0.18,
      rootAo: 0.35,
      canopy: 0.012,
      baseAo: { strength: 0.9, power: 1.6, open: 0.9 },
      stiffness: [0.9, 1.4],
      roughness: 0.7,
      translucency: 0.9,
      densityPower: 1.2,
      rimBias: 2.0,
      clusterSize: 0.04,
      chunks: 4,
      surfaceShade: { darkest: 0.22, soft: 0.45, fill: 0.6, edge: 0.6, tips: 0.2 },
      clumpAo: { strength: 0.3, gaps: 0.25, variation: 0.15, dome: 0.8 },
      translucencyFloor: 0.03,
    },
    {
      kind: "kit",
      name: "clover",
      layer: "leaves",
      budget: 2.0,
      // ≈ 2× the kit size: frame 04 / detail 03 show the clover at 15–25 px (≈ 5–7 cm);
      // on the trunks' cushions — the low branch carries a grass tuft (frame 04), the
      // ground and the foreground stay dark moss (frames 04 / 05)
      targetWeights: [
        ["moss_arch_ground", 0.08],
        ["moss_arch_low", 0.2],
        ["moss_arch_near__fg", 0.0],
        ["wood_", 0.6],
      ],
      items: [
        { name: "leaf_clover", weight: 6, scale: [1.7, 2.5], lift: [0.01, 0.035], tint: "#FFF0A0", pale: { chance: 0.45, amount: [0.15, 0.5], upFacing: 0.6 } },
        { name: "leaf_round_b", weight: 0.8, scale: [0.8, 1.15], lift: [0.005, 0.025], pale: { chance: 0.2, amount: [0.2, 0.6] } },
        { name: "sprig_a", weight: 0.35, scale: [1.2, 1.7], lift: [0.0, 0.01] },
      ],
      upBias: 0.35,
      tiltJitter: 0.4,
      cluster: { size: [3, 6], radius: 0.03, sameSpecies: 0.85 },
      densityPower: 1.3,
      stiffness: [0.9, 1.4],
      flutter: 1,
      saturation: 0.85,
      translucency: 0.7,
      roughness: 0.8,
      brightness: [1.0, 1.45],
      baseAo: 0.5,
      chunks: 3,
      // frame 04: the cushions' sides turned from the key fall into deep shade
      surfaceShade: { darkest: 0.15, soft: 0.45, fill: 0.5, edge: 0.4, tips: 0 },
      translucencyFloor: 0.04,
    },
    {
      kind: "kit",
      name: "flowers",
      layer: "flowers",
      budget: 0.7,
      targetWeights: [
        ["moss_arch_ground", 0.1],
        ["moss_arch_near__fg", 0.0],
        ["wood_", 0.6],
      ],
      items: [
        { name: "flower_white_small", weight: 5, scale: [1.0, 1.6], lift: [0.015, 0.04] },
        { name: "flower_star", weight: 0.6, scale: [0.7, 1.05], lift: [0.005, 0.02] },
      ],
      upBias: 0.5,
      tiltJitter: 0.35,
      cluster: { size: [1, 4], radius: 0.03, sameSpecies: 0.9 },
      densityPower: 1.5,
      stiffness: [0.7, 1.1],
      flutter: 0.7,
      translucency: 0.5,
      roughness: 0.7,
      brightness: [0.9, 1.1],
      baseAo: 0.6,
      chunks: 2,
    },
  ],
};
