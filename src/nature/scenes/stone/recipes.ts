/**
 * Vegetation of the stone set (episode `stone`, frame 14).
 *
 * stoneMoss — the strip of bright moss along the lit top edge of the slab
 * (`moss_stone_edge`: a thick cushion on the left and over the corner, COLOR_0.R ≈
 * 0.9–1; right of the corner a single edge row, R ≈ 0.63–0.72, slightly longer strands)
 * and a few sprigs on the bare rock where the slab's COLOR_0.R marks the rim under the
 * moss:
 *  - `fuzz`    a dense, short, fine pile (0.7–1.6 cm) in small tufts: dark roots, light
 *              yellow-green translucent tips, crowded on the outlines so the cushion
 *              reads as a frizzy carpet, not a rubber skin (frame 14, the top-left
 *              cushion); density R^8, so the low-R right fringe keeps only a sparse pile;
 *  - `stalks`  thin longer stalks (2–3.5 cm, longer where COLOR_0.G asks for it) sticking
 *              out of the carpet, mostly on the silhouette, spread by area (not by R):
 *              on the right fringe they are the single strands against the black;
 *  - `shoots`  tiny branched shoots (kit sprigs at moss size, 1.5–2.5 cm) dotting the
 *              carpet: the star-like tips that catch the light;
 *  - `rockSprigs` a handful of short blades on up-facing rock where COLOR_0.R > 0 (the
 *              rim under the moss strip; not on the slab's lower-left foot).
 * Shorter and denser than the arch moss (archMoss); real plant sizes in metres, budgets
 * are shares of `quality.grassBlades` / `quality.leaves` (Vegetation.ts).
 */
import type { VegetationRecipe } from "../../vegetation/recipes";

export const stoneMoss: VegetationRecipe = {
  name: "stoneMoss",
  targets: { include: ["moss_stone_", "rock_stone_"] },
  layers: [
    {
      kind: "grass",
      name: "fuzz",
      targets: ["moss_stone_"],
      budget: 0.75,
      length: [0.005, 0.013],
      lengthSkew: 1.1,
      lengthMask: 0.55,
      width: [0.0022, 0.0036],
      tipWidth: 0.4,
      segments: 3,
      cone: 0.75,
      tilt: [0.1, 0.75],
      bend: [0.3, 1.3],
      bendFlip: 0.45,
      twist: 0.7,
      droop: 0.25,
      lean: { strength: 0.12, scale: 12 },
      tufts: { size: 0.05, strength: 0.75, swirl: 0.6 },
      outliers: { fraction: 0.04, scale: 1.7, rimBoost: 6, pale: 0.15 },
      colors: { root: "#2E4410", mid: "#7FAE44", tip: "#D2EC8C", tip2: "#BCDC78", dry: "#D6D490" },
      dryFraction: 0.02,
      brightnessJitter: 0.22,
      rootAo: 0.35,
      canopy: 0.01,
      baseAo: 0.8,
      // cushions: tufts shaded as small domes, dark gaps between them (PlantShade.ts)
      surfaceShade: { darkest: 0.18, soft: 0.45, fill: 0.6, edge: 0.6, tips: 0.35 },
      clumpAo: { strength: 0.85, gaps: 0.7, variation: 0.3, dome: 1.0 },
      translucencyFloor: 0.05,
      stiffness: [1.2, 1.8],
      roughness: 0.6,
      translucency: 1.0,
      roundness: 0.6,
      densityPower: 8.0,
      rimBias: 1.6,
      clusterSize: 0.04,
      chunks: 4,
    },
    {
      kind: "grass",
      name: "stalks",
      targets: ["moss_stone_"],
      budget: 0.05,
      length: [0.02, 0.035],
      lengthSkew: 1.2,
      lengthMask: 1.0,
      width: [0.0006, 0.001],
      tipWidth: 0.35,
      segments: 4,
      cone: 0.45,
      tilt: [0.05, 0.45],
      bend: [0.1, 0.6],
      bendFlip: 0.3,
      twist: 0.4,
      droop: 0.15,
      lean: { strength: 0.2, scale: 8 },
      colors: { root: "#2A3A10", mid: "#74A036", tip: "#D0EA90", tip2: "#B8DC7A", dry: "#D6D2A0" },
      dryFraction: 0.08,
      brightnessJitter: 0.18,
      rootAo: 0.45,
      canopy: 0.012,
      baseAo: 0.7,
      stiffness: [0.9, 1.3],
      roughness: 0.6,
      translucency: 1.0,
      densityPower: 0,
      rimBias: 6,
      rimPower: 1.5,
      clusterSize: 0.05,
      chunks: 2,
    },
    {
      kind: "kit",
      name: "shoots",
      layer: "leaves",
      targets: ["moss_stone_"],
      budget: 0.22,
      items: [
        { name: "sprig_a", weight: 3, scale: [0.22, 0.36], lift: [0.0, 0.004], tint: "#E4F2A0" },
        { name: "sprig_b", weight: 2, scale: [0.2, 0.32], lift: [0.0, 0.004], tint: "#E4F2A0" },
      ],
      upBias: 0.25,
      tiltJitter: 0.55,
      scaleSkew: 1.2,
      saturation: 1.0,
      cluster: { size: [2, 5], radius: 0.015, sameSpecies: 0.8 },
      densityPower: 1.0,
      rimBias: 1.5,
      stiffness: [1.0, 1.5],
      flutter: 0.6,
      translucency: 0.9,
      roughness: 0.7,
      brightness: [0.95, 1.3],
      baseAo: 0.6,
      rootAo: 0.5,
      chunks: 2,
    },
    {
      kind: "grass",
      name: "rockSprigs",
      layer: "grass",
      targets: ["rock_stone_"],
      budget: 0.012,
      length: [0.012, 0.03],
      lengthSkew: 1.3,
      lengthMask: 0,
      width: [0.0008, 0.0014],
      tipWidth: 0.2,
      segments: 3,
      cone: 0.5,
      tilt: [0.1, 0.6],
      bend: [0.3, 1.0],
      twist: 0.6,
      droop: 0.2,
      tufts: { size: 0.03, strength: 0.6, swirl: 0.5 },
      colors: { root: "#24340A", mid: "#5E8C1E", tip: "#B0DA4C", tip2: "#94C23A", dry: "#C6C27A" },
      dryFraction: 0.05,
      rootAo: 0.4,
      baseAo: 0.8,
      stiffness: [1.0, 1.5],
      roughness: 0.65,
      translucency: 0.9,
      densityPower: 1.5,
      facing: { min: 0.1, max: 0.45 },
      clusterSize: 0.05,
      chunks: 2,
    },
  ],
};
