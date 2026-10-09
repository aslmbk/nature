/**
 * Vegetation of the canyon set (episode `canyon`, frame 06, 19.8 s).
 *
 * canyonIslands — the leafy plant islands on the rock fragments. Frame 06: dense clumps
 * of small leaves on short stems along the cracks and on the ledge lips of the lit
 * right face (2–4 cm leaves: 8–14 px at 4.5 m) with bare rock between them, sprigs, a
 * few clover and tiny white flowers, thin grass blades breaking the silhouettes, a grass
 * clump at the right border, short moss fuzz under the leaves; sprigs along the lit
 * left edge; dark, out-of-focus growth on the near form at the bottom left.
 *
 * Where: COLOR_0.R of the moss pads (`moss_canyon_*`) and of the rocks. For the rocks
 * CanyonScene passes scatter-only copies whose R adds frame 06's overgrowth inside the
 * asset's plant zones, in patches, kept to the faces the main camera sees
 * (canyon/growth.ts); `grass_canyon_right` is such a copy for the grass clump only.
 *  - `fuzz`    short dark moss pile (1–2.5 cm) under every island;
 *  - `leaves`  small leaves on petioles (kit round / serrated at 0.42–0.78 of the
 *              modelled 4–7 cm), clover, sprigs, in dense clusters (12–28 within 6 cm),
 *              turned every which way: the ones on a lip hang over it;
 *  - `flowers` a few tiny white flowers;
 *  - `blades`  thin long blades (7–18 cm), sparse, crowded on the silhouettes;
 *  - `clump`   the grass clump at the right border and the darker grass below it.
 * Real plant sizes in metres; budgets are shares of `quality.grassBlades` /
 * `quality.leaves` (Vegetation.ts).
 */
import type { VegetationRecipe } from "../../vegetation/recipes";

/** Warm olive multiplier on the kit atlas's sage leaves (frame 06: #3C4100 → #97A24F under the key). */
const LEAF_TINT = "#E8E0A0";

/** The islands (pads, rock growth); `grass_canyon_*` (the right-border clump) only grows `clump`. */
const ISLANDS = ["moss_canyon_", "rock_canyon_"];

export const canyonIslands: VegetationRecipe = {
  name: "canyonIslands",
  targets: { include: ["moss_canyon_", "rock_canyon_", "grass_canyon_"] },
  // the near form is dark and out of focus: fewer plants; most of the pads lie on ledge
  // tops above the frame or behind the lips: the rock's own ledges get a larger share
  targetWeights: [
    ["rock_canyon_low__fg", 0.5],
    ["moss_canyon_", 0.7],
  ],
  layers: [
    {
      // the dark green base of every island: a short tangled pile (1–2.5 cm) on the pads
      // and where the rock grows plants, the leaves sit in it
      kind: "grass",
      name: "fuzz",
      targets: ISLANDS,
      budget: 0.45,
      length: [0.01, 0.025],
      lengthSkew: 1.2,
      lengthMask: 0.5,
      width: [0.0012, 0.002],
      tipWidth: 0.2,
      segments: 3,
      cone: 0.85,
      tilt: [0.15, 0.8],
      bend: [0.3, 1.3],
      bendFlip: 0.4,
      twist: 0.6,
      droop: 0.4,
      lean: { strength: 0.1, scale: 10 },
      tufts: { size: 0.025, strength: 0.5, swirl: 0.6 },
      outliers: { fraction: 0.04, scale: 1.6, rimBoost: 4, pale: 0.1 },
      colors: { root: "#141A06", mid: "#38461A", tip: "#7E8C3A", tip2: "#6C7C30", dry: "#9C9466" },
      dryFraction: 0.03,
      brightnessJitter: 0.25,
      rootAo: 0.3,
      canopy: 0.012,
      baseAo: 0.8,
      stiffness: [1.1, 1.6],
      roughness: 0.7,
      translucency: 0.8,
      densityPower: 1.5,
      rimBias: 1.5,
      clusterSize: 0.04,
      chunks: 4,
    },
    {
      // small olive leaves on short stems in dense clusters (one little plant each),
      // turned every which way; on a lip they hang over it
      kind: "kit",
      name: "leaves",
      layer: "leaves",
      targets: ISLANDS,
      budget: 0.8,
      items: [
        { name: "leaf_round_a", weight: 2, scale: [0.45, 0.72], lift: [0.003, 0.012], tint: LEAF_TINT },
        { name: "leaf_round_b", weight: 2, scale: [0.5, 0.78], lift: [0.003, 0.012], tint: LEAF_TINT },
        { name: "leaf_round_c", weight: 1, scale: [0.42, 0.65], lift: [0.002, 0.01], tint: LEAF_TINT },
        { name: "leaf_serrated_a", weight: 2, scale: [0.45, 0.72], lift: [0.003, 0.012], tint: LEAF_TINT },
        { name: "leaf_serrated_b", weight: 2, scale: [0.5, 0.78], lift: [0.003, 0.012], tint: LEAF_TINT },
        { name: "leaf_clover", weight: 3, scale: [0.8, 1.2], lift: [0.0, 0.006], tint: LEAF_TINT },
        { name: "sprig_a", weight: 1.2, scale: [0.6, 1.0], lift: [0.0, 0.004], tint: LEAF_TINT },
        { name: "sprig_b", weight: 1.0, scale: [0.6, 1.0], lift: [0.0, 0.004], tint: LEAF_TINT },
      ],
      upBias: 0.25,
      tiltJitter: 0.9,
      scaleSkew: 1.3,
      saturation: 0.75,
      cluster: { size: [12, 28], radius: 0.06, sameSpecies: 0.8 },
      densityPower: 1.8,
      rimBias: 1.5,
      stiffness: [0.8, 1.3],
      flutter: 1,
      translucency: 0.7,
      roughness: 0.8,
      brightness: [0.45, 1.05],
      baseAo: 0.7,
      chunks: 3,
    },
    {
      kind: "kit",
      name: "flowers",
      layer: "flowers",
      targets: ISLANDS,
      budget: 0.02,
      items: [{ name: "flower_white_small", weight: 1, scale: [0.5, 0.8], lift: [0.004, 0.012] }],
      upBias: 0.6,
      tiltJitter: 0.35,
      cluster: { size: [1, 3], radius: 0.025, sameSpecies: 1 },
      densityPower: 1.5,
      stiffness: [0.7, 1.1],
      flutter: 0.7,
      translucency: 0.5,
      roughness: 0.7,
      brightness: [0.8, 1.0],
      baseAo: 0.6,
      chunks: 2,
    },
    {
      // grass breaking the outlines: long thin blades in loose tufts, leaning out and
      // drooping (the upper-left outline of the right face, the clump at the right border)
      kind: "grass",
      name: "blades",
      targets: ISLANDS,
      budget: 0.04,
      length: [0.07, 0.18],
      lengthSkew: 1.4,
      lengthMask: 0.6,
      width: [0.0025, 0.0045],
      tipWidth: 0.12,
      segments: 6,
      cone: 0.6,
      tilt: [0.2, 0.8],
      bend: [0.5, 1.8],
      bendLength: 0.6,
      maxAngle: 2.3,
      bendFlip: 0.3,
      twist: 0.9,
      droop: 0.6,
      lean: { strength: 0.3, scale: 5 },
      tufts: { size: 0.06, strength: 0.65, swirl: 0.5 },
      outliers: { fraction: 0.06, scale: 1.4, rimBoost: 3, pale: 0.3 },
      colors: { root: "#232C0C", mid: "#5A6820", tip: "#B4BA58", tip2: "#9AA244", dry: "#C6B880" },
      dryFraction: 0.1,
      brightnessJitter: 0.22,
      rootAo: 0.4,
      baseAo: 0.7,
      stiffness: [0.7, 1.1],
      roughness: 0.7,
      translucency: 1.0,
      densityPower: 1.8,
      rimBias: 5,
      rimPower: 1.5,
      clusterSize: 0.07,
      chunks: 3,
    },
    {
      // the grass clump at the right border (frame 06): long blades leaning out and
      // drooping, lit olive-yellow at the tips, dark at the roots; darker grass below it
      kind: "grass",
      name: "clump",
      targets: ["grass_canyon_"],
      budget: 0.08,
      length: [0.06, 0.17],
      lengthSkew: 1.2,
      lengthMask: 0.5,
      width: [0.0025, 0.004],
      tipWidth: 0.15,
      segments: 5,
      cone: 0.7,
      tilt: [0.3, 0.9],
      bend: [0.6, 1.8],
      bendLength: 0.6,
      maxAngle: 2.3,
      bendFlip: 0.25,
      twist: 0.8,
      droop: 0.7,
      lean: { strength: 0.35, scale: 4 },
      tufts: { size: 0.05, strength: 0.7, swirl: 0.5 },
      outliers: { fraction: 0.05, scale: 1.35, rimBoost: 3, pale: 0.25 },
      colors: { root: "#1A2207", mid: "#4A581A", tip: "#A6AE52", tip2: "#8C983E", dry: "#BDB078" },
      dryFraction: 0.12,
      brightnessJitter: 0.25,
      rootAo: 0.35,
      baseAo: 0.7,
      stiffness: [0.8, 1.2],
      roughness: 0.7,
      translucency: 1.0,
      densityPower: 1.3,
      rimBias: 1.5,
      clusterSize: 0.06,
      chunks: 3,
    },
  ],
};
