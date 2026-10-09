/**
 * Vegetation of the finale set (episode `finale`, frames 18–20, detail 09).
 *
 * finaleMoss — the moss plateau and its hills (`moss_finale_hills`):
 *  - `fuzz`    the moss pile: moss shoots 0.6–1.5 cm long and 2.5–4.5 mm wide (a shoot
 *              with its leaves, not a grass blade), green roots, bright lime tips (the
 *              highlights; the darker moss base between them is the depth), leaning out
 *              of 9 cm tufts; spread evenly over the mounds of finale.glb rev 3, their
 *              steep flanks included (bare flanks showed the moss base as black holes) —
 *              only overhangs stay bare; the far sides of the mounds dark olive, not
 *              black; sparse where COLOR_0.R is low;
 *  - `tufts`   short grass 2.5–5 cm in patches on the crests (frame 19: a few longer
 *              blades break the hill outlines);
 *  - `flowers` the white star flowers (`flower_star`), standing up as cups, single or in
 *              pairs on the tops near the crest, few on the outlines — a fixed number per
 *              scene (≈ 19 visible in ours at 54.3: a sparse few, as frame 19 reads), not a
 *              share of the leaf budget, so every preset keeps the same composition;
 *  - `sprigs`  a few feathery sprigs among the flowers (frame 19 left, detail 09).
 * finaleStoneMoss — a little moss fuzz where `stone_finale` carries COLOR_0.R (the crack
 * on the front left and the foot inside the contact ring).
 * finaleNearMoss — the near mound (`moss_finale_near__fg`): out of focus, in the frame's
 * dark lower band; only a short, near-black fuzz that roughens its blurred top edge.
 * finaleBankMoss — the bank in front of the island (same mesh) for narrow screens only.
 *
 * Real plant sizes in metres; grass budgets are shares of `quality.grassBlades`.
 */
import type { QualityPreset } from "../../types";
import type { VegetationRecipe } from "../../vegetation/recipes";

/** Star flowers / sprigs scattered over the whole plateau (all poses together). */
const FLOWER_COUNT = 60;
const SPRIG_COUNT = 40;

export function finaleMoss(q: QualityPreset): VegetationRecipe {
  // budgets of the composition elements as shares of this preset's leaf budget
  const flowers = FLOWER_COUNT / Math.max(1, q.leaves);
  const sprigs = SPRIG_COUNT / Math.max(1, q.leaves);
  // fewer shoots on lower presets: make them a little wider so the pile stays closed
  const width: [number, number] = q.level === "low" ? [0.0036, 0.006] : q.level === "medium" ? [0.003, 0.005] : [0.0025, 0.0045];
  return {
    name: "finaleMoss",
    targets: { include: ["moss_finale_hills"] },
    layers: [
      {
        kind: "grass",
        name: "fuzz",
        // 195k shoots on high (the flanks take their share since rev 3)
        budget: 2.6,
        length: [0.006, 0.015],
        lengthSkew: 1.0,
        lengthMask: 0.5,
        width,
        tipWidth: 0.5,
        segments: 2,
        // mostly upright shoots: a shoot that curls over shows its underside to the lens,
        // which only the dark ground fill reaches (black specks in the pile)
        cone: 0.5,
        tilt: [0.1, 0.55],
        bend: [0.2, 0.7],
        bendFlip: 0.3,
        // twisted, rounded and translucent (below): fewer shoots lie exactly edge-on to the
        // key, whose unlit side came out as near-black specks in the lit pile
        twist: 1.6,
        droop: 0.2,
        lean: { strength: 0.12, scale: 9 },
        // shoots lean out of 9 cm tufts: soft cushions (frame 19, detail 09), not a lawn
        tufts: { size: 0.09, strength: 0.6, swirl: 0.5 },
        // a few longer pale shoots on the outlines
        outliers: { fraction: 0.03, scale: 1.5, rimBoost: 6, pale: 0.25 },
        // a rich green with lime tips (frame 19 / detail 09), not a khaki felt
        colors: { root: "#4A6418", mid: "#8CAE34", tip: "#C4E274", tip2: "#B2D464", dry: "#D0D8A0" },
        dryFraction: 0.015,
        brightnessJitter: 0.1,
        rootAo: 0.45,
        canopy: 0.01,
        // soft: COLOR_0.B is ≈ 1 on the mound crowns and 0.34–0.51 in the gaps between them;
        // a strong AO there turns every gap black
        baseAo: { strength: 0.45, power: 2, open: 0.78 },
        // the pile follows the mounds: faces turned to the key (upper left) lit, the far
        // sides soft and dark olive (lit domes, not black flanks)
        surfaceShade: { darkest: 0.4, soft: 0.6, fill: 0.6, edge: 0.55, tips: 0.2 },
        // a light clump occlusion only (the cushions are modelled since rev 2)
        clumpAo: { strength: 0.15, gaps: 0.08, variation: 0.12 },
        stiffness: [1.3, 1.9],
        roughness: 0.8,
        translucency: 1.5,
        roundness: 1.0,
        densityPower: 1.6,
        // shoots everywhere but under overhangs: the mounds' steep flanks need them most
        facing: { min: -0.4, max: 0.15 },
        // a little extra on the crest outlines (hairy against the dark); at 2.2 the outlines
        // of every cushion piled up pale shoots, the quilt lines
        rimBias: 0.6,
        rimPower: 1.5,
        clusterSize: 0.05,
        chunks: 12,
      },
      {
        kind: "grass",
        name: "tufts",
        budget: 0.05,
        length: [0.025, 0.05],
        lengthSkew: 1.3,
        lengthMask: 0.7,
        width: [0.0012, 0.002],
        tipWidth: 0.1,
        segments: 4,
        cone: 0.35,
        tilt: [0.1, 0.5],
        bend: [0.4, 1.3],
        bendFlip: 0.3,
        droop: 0.5,
        lean: { strength: 0.25, scale: 6 },
        tufts: { size: 0.04, strength: 0.7, swirl: 0.4 },
        outliers: { fraction: 0.04, scale: 1.3 },
        colors: { root: "#2E3A0E", mid: "#7C962A", tip: "#D2DE80", tip2: "#BCCC66", dry: "#DCD8A6" },
        dryFraction: 0.1,
        rootAo: 0.4,
        baseAo: 0.85,
        surfaceShade: { darkest: 0.25, soft: 0.5, fill: 0.6, edge: 0.6, tips: 0.35 },
        stiffness: [0.9, 1.4],
        roughness: 0.65,
        translucency: 0.9,
        densityPower: 2,
        facing: { min: 0.1, max: 0.6 },
        patches: { scale: 4, threshold: 0.6, softness: 0.25 },
        rimBias: 2,
        clusterSize: 0.05,
        chunks: 4,
      },
      {
        kind: "kit",
        name: "flowers",
        layer: "flowers",
        budget: flowers,
        items: [{ name: "flower_star", weight: 1, scale: [1.3, 1.9], lift: [0.014, 0.028] }],
        upBias: 0.3,
        tiltJitter: 0.5,
        // standing up as cups (frame 19: white rosettes on the tops, not stars lying flat)
        upright: 0.85,
        tiltRange: [0.05, 0.35],
        scaleSkew: 1.2,
        cluster: { size: [1, 2], radius: 0.08, sameSpecies: 1 },
        densityPower: 2,
        // on the tops, a few on the outlines: seen at a grazing angle a crest packs a lot of
        // surface into a few pixels, so an even scatter would crown every ridge with flowers
        facing: { min: 0.45, max: 0.85 },
        rimBias: 0.4,
        stiffness: [0.9, 1.3],
        flutter: 0.5,
        translucency: 0.45,
        roughness: 0.6,
        brightness: [1.0, 1.2],
        baseAo: 0.25,
        rootAo: 0.6,
        chunks: 2,
      },
      {
        kind: "kit",
        name: "sprigs",
        layer: "leaves",
        budget: sprigs,
        items: [
          { name: "sprig_a", weight: 1, scale: [0.55, 0.85], lift: [0.004, 0.01] },
          { name: "sprig_b", weight: 1, scale: [0.5, 0.8], lift: [0.004, 0.01] },
        ],
        upBias: 0.6,
        tiltJitter: 0.35,
        saturation: 0.85,
        cluster: { size: [1, 3], radius: 0.03, sameSpecies: 0.8 },
        densityPower: 2,
        facing: { min: 0.1, max: 0.6 },
        rimBias: 1,
        patches: { scale: 2.2, threshold: 0.42, softness: 0.4 },
        stiffness: [0.9, 1.4],
        flutter: 0.8,
        translucency: 0.7,
        roughness: 0.8,
        brightness: [0.85, 1.1],
        baseAo: 0.6,
        chunks: 2,
      },
    ],
  };
}

/** Moss fuzz on the stone where COLOR_0.R marks it (crack, foot). */
export const finaleStoneMoss: VegetationRecipe = {
  name: "finaleStoneMoss",
  targets: { include: ["stone_finale"] },
  layers: [
    {
      kind: "grass",
      name: "stoneFuzz",
      budget: 0.03,
      length: [0.005, 0.012],
      lengthMask: 0.3,
      width: [0.0012, 0.002],
      tipWidth: 0.35,
      segments: 2,
      cone: 0.7,
      tilt: [0.2, 0.8],
      bend: [0.2, 0.9],
      bendFlip: 0.4,
      droop: 0.4,
      tufts: { size: 0.012, strength: 0.5, swirl: 0.5 },
      outliers: { fraction: 0, scale: 1 },
      colors: { root: "#2A3010", mid: "#56642A", tip: "#8E9C50", tip2: "#808E44", dry: "#A8A684" },
      dryFraction: 0.03,
      rootAo: 0.45,
      baseAo: 0.9,
      stiffness: [1.4, 2.0],
      roughness: 0.75,
      translucency: 0.6,
      // only where the crack and the foot carry a clear mark
      densityPower: 3,
      chunks: 2,
    },
  ],
};

/** Colours of the near mound's fuzz (sRGB; FinaleScene blends them for narrow screens). */
export const NEAR_FUZZ_COLORS = { root: "#0C0F05", mid: "#1C240C", tip: "#344018", tip2: "#2C3814", dry: "#3A3C2C" };

/** Near-camera mound: short, near-black fuzz (out of focus, in the dark lower band). */
export const finaleNearMoss: VegetationRecipe = {
  name: "finaleNearMoss",
  targets: { include: [/__fg$/] },
  layers: [
    {
      kind: "grass",
      name: "fgFuzz",
      budget: 0.04,
      length: [0.012, 0.03],
      lengthMask: 0.4,
      width: [0.002, 0.0035],
      tipWidth: 0.3,
      segments: 2,
      cone: 0.7,
      tilt: [0.2, 0.8],
      bend: [0.3, 1.2],
      bendFlip: 0.4,
      droop: 0.4,
      lean: { strength: 0.15, scale: 6 },
      tufts: { size: 0.04, strength: 0.5, swirl: 0.6 },
      outliers: { fraction: 0.02, scale: 1.4 },
      colors: NEAR_FUZZ_COLORS,
      dryFraction: 0.02,
      rootAo: 0.5,
      baseAo: 0.6,
      stiffness: [1.2, 1.6],
      translucency: 0.15,
      densityPower: 0.5,
      chunks: 3,
    },
  ],
};

/**
 * The bank in front of the island (`moss_finale_near__fg`, finale.glb rev 3; COLOR_0.R ≈ 0.2
 * on the bank, 0 under the island) as narrow screens see it: a dark olive moss pile with
 * lit tips, out of focus. Scattered for portrait views only and drawn only on narrow
 * screens (FinaleScene): landscape frames never show it.
 */
export const finaleBankMoss: VegetationRecipe = {
  name: "finaleBankMoss",
  targets: { include: [/__fg$/] },
  layers: [
    {
      kind: "grass",
      name: "bankFuzz",
      budget: 0.25,
      length: [0.012, 0.028],
      lengthSkew: 1.0,
      lengthMask: 0.4,
      width: [0.002, 0.0038],
      tipWidth: 0.4,
      segments: 2,
      cone: 0.55,
      tilt: [0.1, 0.6],
      bend: [0.2, 0.8],
      bendFlip: 0.3,
      twist: 1.4,
      droop: 0.25,
      lean: { strength: 0.15, scale: 6 },
      tufts: { size: 0.1, strength: 0.6, swirl: 0.5 },
      outliers: { fraction: 0.02, scale: 1.4 },
      colors: { root: "#1A240A", mid: "#3E5018", tip: "#7E9238", tip2: "#6E8230", dry: "#8A8C5C" },
      dryFraction: 0.02,
      brightnessJitter: 0.12,
      rootAo: 0.45,
      canopy: 0.015,
      baseAo: { strength: 0.5, power: 1.5, open: 0.6 },
      surfaceShade: { darkest: 0.4, soft: 0.6, fill: 0.6, edge: 0.55, tips: 0.25 },
      stiffness: [1.2, 1.7],
      roughness: 0.8,
      translucency: 0.8,
      roundness: 1.0,
      densityPower: 0.4,
      facing: { min: -0.3, max: 0.2 },
      chunks: 4,
    },
  ],
};
