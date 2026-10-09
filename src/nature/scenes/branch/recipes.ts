/**
 * Vegetation of the branch set (episode `branch`, frames 10–12, details 06 / 07).
 *
 * branchMoss — the moss coat of the J-branch and the lower-right fragment:
 *  - `fuzz`   a dense, fine, olive-yellow pile 1–2 cm long on every moss cushion: thin
 *             blades, dark roots, pale slightly translucent tips; crowded on the
 *             silhouettes so the outline is hairy, not a shell (detail 06);
 *  - `tufts`  short grass tufts 2–4 cm in patches on the cushions (frame 10 right
 *             edge, the fragment's crest);
 *  - `woodFuzz` moss on the bark where COLOR_0.R > 0: the band along the moss border, the
 *             deep cracks, the moss spots on the stumps (`wood_branch_stumps`);
 *  - `flowers` tiny white flower heads dotting the moss, more on the outlines;
 *  - `sprigs` a few clover leaves and sprigs.
 *  The pile follows the form of the lumpy cushions (PlantShade `surfaceShade`: the key
 *  only reaches the faces turned to it, the undersides see the ground), its clumps have
 *  dark cores and dark olive gaps between them (`clumpAo`, on the fuzz's own tuft cells)
 *  and the crevices of the cushions stay dark (`baseAo` on the baked cushion AO).
 * ballFur — hair on the `moss_ball` core (6 cm radius): 1.6–3 cm long but lying over
 * (≈ 1–1.5 cm of pile), uneven, combed in locks towards the ball's own −Y pole, parting
 * around +Y (`comb`; detail 07). The coat is lit from BALL_LIGHT (`surfaceShade` with a
 * fixed world direction, base normal from the ball's centre): dark inside on the side
 * away from it, a light fringe all round. Wider hairs on `low` so the fewer hairs still
 * read as fur.
 *
 * Real plant sizes in metres; budgets are shares of `quality.grassBlades` /
 * `quality.leaves` (Vegetation.ts).
 */
import type { QualityLevel } from "../../types";
import type { VegetationRecipe } from "../../vegetation/recipes";

/**
 * Towards the light that shapes the ball's coat (world): from the upper right and a little
 * from behind — frame 10: bright upper right and fringe, the lower-left two thirds a dark
 * green inside.
 */
export const BALL_LIGHT: [number, number, number] = [0.65, 0.7, -0.3];

export const branchMoss: VegetationRecipe = {
  name: "branchMoss",
  targets: { include: ["moss_branch_", "wood_branch_"] },
  layers: [
    {
      kind: "grass",
      name: "fuzz",
      targets: ["moss_branch_"],
      budget: 2.4,
      length: [0.007, 0.015],
      lengthSkew: 1.1,
      lengthMask: 0.5,
      width: [0.0004, 0.0008],
      tipWidth: 0.2,
      segments: 2,
      cone: 0.55,
      tilt: [0.25, 0.9],
      bend: [0.5, 1.5],
      bendFlip: 0.5,
      twist: 0.8,
      droop: 0.3,
      lean: { strength: 0.35, scale: 14 },
      // small cushions: blades lean out of 2.5 cm tufts, partings between them (clumpy pile)
      tufts: { size: 0.025, strength: 0.7, swirl: 0.6 },
      // a few longer pale hairs breaking the outline
      outliers: { fraction: 0.05, scale: 1.5, rimBoost: 8, pale: 0.45 },
      colors: { root: "#343C0C", mid: "#9AA040", tip: "#E2E09A", tip2: "#D0D07E", dry: "#E4E0B4" },
      dryFraction: 0.05,
      brightnessJitter: 0.3,
      rootAo: 0.5,
      canopy: 0.008,
      baseAo: { strength: 0.9, power: 2, open: 0.92 },
      stiffness: [1.3, 2.0],
      roughness: 0.6,
      translucency: 1.0,
      translucencyFloor: 0.03,
      roundness: 0.6,
      densityPower: 1.2,
      rimBias: 3,
      rimPower: 1.5,
      clusterSize: 0.03,
      chunks: 8,
      // pile shading: the key only reaches the faces of the lumpy cushions turned to it,
      // the undersides see the ground; 4.5 cm clumps (two or three of the 2.5 cm tufts) with
      // dark cores, dark olive gaps and a brightness of their own, each shaded as a dome
      surfaceShade: { darkest: 0.15, soft: 0.4, fill: 0.7, edge: 0.6, tips: 0.3 },
      clumpAo: { strength: 0.6, size: 0.045, gaps: 0.5, variation: 0.2, dome: 1.0 },
    },
    {
      kind: "grass",
      name: "tufts",
      targets: ["moss_branch_"],
      budget: 0.06,
      length: [0.02, 0.038],
      lengthSkew: 1.3,
      lengthMask: 0.6,
      width: [0.0012, 0.002],
      tipWidth: 0.1,
      segments: 4,
      cone: 0.4,
      tilt: [0.1, 0.5],
      bend: [0.4, 1.4],
      bendFlip: 0.3,
      droop: 0.6,
      lean: { strength: 0.2, scale: 8 },
      tufts: { size: 0.025, strength: 0.7, swirl: 0.4 },
      outliers: { fraction: 0.03, scale: 1.3 },
      colors: { root: "#323E10", mid: "#86963A", tip: "#D4D888", tip2: "#C0C870", dry: "#DEDAA8" },
      dryFraction: 0.08,
      rootAo: 0.35,
      baseAo: { strength: 0.85, power: 2, open: 0.92 },
      stiffness: [0.9, 1.4],
      roughness: 0.62,
      translucency: 0.8,
      translucencyFloor: 0.03,
      densityPower: 2,
      patches: { scale: 7, threshold: 0.58, softness: 0.25 },
      rimBias: 1,
      clusterSize: 0.05,
      chunks: 4,
      surfaceShade: { darkest: 0.25, soft: 0.45, fill: 0.6, edge: 0.6, tips: 0.4 },
      clumpAo: { strength: 0.35, gaps: 0.2, variation: 0.12 },
    },
    {
      kind: "grass",
      name: "woodFuzz",
      targets: ["wood_branch_"],
      budget: 0.35,
      length: [0.008, 0.018],
      lengthMask: 0.3,
      width: [0.0006, 0.001],
      tipWidth: 0.15,
      segments: 3,
      cone: 0.6,
      tilt: [0.2, 0.7],
      bend: [0.3, 1.2],
      bendFlip: 0.4,
      droop: 0.3,
      tufts: { size: 0.008, strength: 0.5, swirl: 0.6 },
      outliers: { fraction: 0, scale: 1 },
      colors: { root: "#2A320C", mid: "#6E7E2A", tip: "#BCC660", tip2: "#A6B44C", dry: "#C9C08E" },
      dryFraction: 0.04,
      rootAo: 0.4,
      baseAo: 0.85,
      stiffness: [1.3, 2.0],
      translucency: 1.0,
      translucencyFloor: 0.03,
      densityPower: 1,
      chunks: 3,
      surfaceShade: { darkest: 0.3, soft: 0.5, fill: 0.5, edge: 0.5, tips: 0.2 },
    },
    {
      kind: "kit",
      name: "flowers",
      layer: "flowers",
      targets: ["moss_branch_"],
      budget: 0.2,
      items: [{ name: "flower_white_small", weight: 1, scale: [0.55, 0.85], lift: [0.002, 0.008] }],
      upBias: 0.35,
      tiltJitter: 0.45,
      scaleSkew: 1.3,
      cluster: { size: [3, 8], radius: 0.035, sameSpecies: 1 },
      densityPower: 1.5,
      patches: { scale: 5, threshold: 0.35, softness: 0.3 },
      rimBias: 3,
      stiffness: [0.8, 1.2],
      flutter: 0.6,
      translucency: 0.5,
      translucencyFloor: 0.05,
      roughness: 0.7,
      brightness: [0.92, 1.12],
      baseAo: 0.5,
      chunks: 3,
      // white heads in the shade of a cushion stay grey, not lit
      surfaceShade: { darkest: 0.35, soft: 0.5, fill: 0.5, edge: 0.6, tips: 0 },
    },
    {
      kind: "kit",
      name: "sprigs",
      layer: "leaves",
      targets: ["moss_branch_"],
      budget: 0.025,
      items: [
        { name: "leaf_clover", weight: 4, scale: [0.55, 0.85], lift: [0.0, 0.006], tint: "#E8E6B0" },
        { name: "sprig_a", weight: 1, scale: [0.4, 0.6], lift: [0.0, 0.004] },
        { name: "sprig_b", weight: 1, scale: [0.4, 0.6], lift: [0.0, 0.004] },
      ],
      upBias: 0.3,
      tiltJitter: 0.4,
      saturation: 0.8,
      cluster: { size: [2, 5], radius: 0.02, sameSpecies: 0.8 },
      densityPower: 1.5,
      patches: { scale: 6, threshold: 0.5, softness: 0.3 },
      stiffness: [0.9, 1.4],
      flutter: 0.8,
      translucency: 0.7,
      translucencyFloor: 0.05,
      roughness: 0.8,
      brightness: [0.75, 1.05],
      baseAo: 0.75,
      chunks: 3,
      surfaceShade: { darkest: 0.3, soft: 0.5, fill: 0.5, edge: 0.5, tips: 0 },
    },
  ],
};

/** Hair of the fuzzy ball (target: the `moss_ball` core, radius ≈ 6 cm). */
export function ballFur(level: QualityLevel): VegetationRecipe {
  // fewer hairs on lower presets: make them a little thicker so the coat stays closed
  const width: [number, number] = level === "low" ? [0.0012, 0.0019] : level === "medium" ? [0.001, 0.0016] : [0.0008, 0.0014];
  return {
    name: "ballFur",
    targets: { include: ["moss_ball"] },
    layers: [
      {
        kind: "grass",
        name: "fur",
        layer: "grass",
        budget: 0.4,
        length: [0.016, 0.03],
        lengthSkew: 1.0,
        lengthMask: 0.7,
        width,
        tipWidth: 0.2,
        segments: 3,
        normalJitter: 0.2,
        // brushed: the hairs lie over at 45–75° in combed locks (detail 07, frame 10:
        // long light streaks, the dark interior shows where the coat parts)
        tilt: [0.8, 1.3],
        bend: [0.3, 0.8],
        bendFlip: 0.1,
        twist: 0.5,
        droop: 0.3,
        lean: { strength: 0.4, scale: 10 },
        tufts: { size: 0.025, strength: 0.45, swirl: 0.3 },
        // combed towards the ball's own −Y pole (it tumbles, so this is just the coat's
        // pattern): long streaks over most of it, the coat parts around the +Y pole
        comb: { dir: [0, -1, 0], strength: 0.8, lay: 0.6 },
        outliers: { fraction: 0.06, scale: 1.3 },
        colors: { root: "#202C06", mid: "#7A9A1C", tip: "#D8E86A", tip2: "#C6DA52", dry: "#E4E6A4" },
        dryFraction: 0.03,
        brightnessJitter: 0.3,
        rootAo: 0.6,
        canopy: 0.008,
        baseAo: 0.5,
        stiffness: [0.9, 1.4],
        roughness: 0.6,
        translucency: 2.0,
        roundness: 0.6,
        densityPower: 1,
        clusterSize: 0.02,
        chunks: 2,
        // the coat's light and dark: albedo × mix(0.1, 1, terminator(n·BALL_LIGHT)) with n
        // from the ball's centre (soft over ±0.35 of n·l); hairs on the outline keep their
        // colour (light fringe), so the dark side reads as a dark green inside, not a black half
        surfaceShade: { radial: true, dir: BALL_LIGHT, darkest: 0.1, soft: 0.35, fill: 0, edge: 1, tips: 0 },
      },
    ],
  };
}
