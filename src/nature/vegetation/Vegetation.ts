/**
 * Recipe → instanced vegetation. Usage from a scene's `build(ctx)`:
 *
 *   const kit = await loadKit(ctx);                       // kit.glb meshes + atlas (shared)
 *   const world = await ctx.assets.instance("models/grove.glb"); // own copy of the node tree
 *   if (!world) return;                                   // (or tryGltf + gltf.scene.clone(true);
 *   this.scene.add(world);                                //  never add the cached gltf.scene itself)
 *   const veg = buildVegetation(ctx, heroMeadow, {
 *     root: world,                                         // where target meshes are looked up
 *     parent: world,                                       // plants are added (and scattered) here
 *     views: [view(rig.pose("hero_main")), ...],           // where the budget should go
 *     kit, label: "hero",
 *   });
 *   veg.report(ctx);                                       // instance counts → stats / debug panel
 *
 * `await buildVegetationAsync(ctx, recipe, opts)` does the same work time-sliced (slices.ts):
 * identical random draws and instance data, but the work is cut between scatter batches and
 * chunk writes and paced by the engine's frame loop (its per-frame budget, a frame rendered
 * between any two stretches; without a running engine ≈ `yieldEveryMs` slices), so a set
 * can prepare while the current episode keeps rendering. The build belongs to `ctx`: when the
 * set is released while it prepares, it stops at its next slice boundary and the promise
 * rejects with `SliceCancelled` (at once when it is started after the release).
 *
 * Budgets: every layer takes `layer.budget × recipe.densityScale × opts.budgetScale`
 * of `ctx.quality.grassBlades` (grass) or `ctx.quality.leaves` (kit); a layer with
 * `count` takes `count × recipe.densityScale × opts.budgetScale` plants at every quality.
 * Deterministic: all randomness from `ctx.rng(label/layer)`.
 */
import { Group, Vector3, type Frustum, type Mesh, type Object3D, type Texture } from "three";
import type { GLTF } from "three/examples/jsm/loaders/GLTFLoader.js";
import { hashString } from "../core/rng";
import type { SceneContext } from "../types";
import { fbm3 } from "./noise";
import { GrassBatch } from "./GrassBatch";
import { LeafBatch, kitMeshes } from "./LeafBatch";
import { matchesName, type VegetationRecipe } from "./recipes";
import { runSliced, runSync, type Steps } from "./slices";
import { SurfaceSet, type ScatterView, type ViewImportanceOptions } from "./SurfaceScatter";

export interface VegetationKit {
  meshes: Map<string, Mesh>;
  map: Texture | null;
}

/** kit.glb + its atlas through the shared AssetRegistry (null meshes when missing). */
export async function loadKit(ctx: SceneContext): Promise<VegetationKit> {
  const [gltf, map] = await Promise.all([ctx.assets.tryGltf("models/kit.glb"), ctx.assets.natureTexture("kit_basecolor")]);
  return { meshes: kitMeshes(gltf as GLTF | null), map };
}

export interface BuildVegetationOptions {
  /** Explicit target meshes; otherwise the recipe targets are looked up under `root`. */
  meshes?: Mesh[];
  root?: Object3D;
  /** The vegetation group is added to this object; scatter happens in its space. */
  parent: Object3D;
  /** Camera poses (in `parent` space) whose visible surfaces get the plants. */
  views: ScatterView[];
  kit: VegetationKit;
  /** rng / noise namespace, e.g. "hero" or "hero-fg". */
  label: string;
  /** Multiplies every layer budget (default 1). */
  budgetScale?: number;
  /** View importance of the scatter (a layer's own `importance` overrides it). */
  importance?: ViewImportanceOptions;
}

export interface BuildVegetationAsyncOptions extends BuildVegetationOptions {
  /**
   * Longest stretch of build work between two yields to the browser (ms, default 8) when no
   * engine frame loop paces the slices; a running engine uses its own per-frame budget.
   */
  yieldEveryMs?: number;
}

export interface VegetationBuild {
  readonly group: Group;
  readonly grass: GrassBatch[];
  readonly kit: LeafBatch[];
  /** Instance counts per layer name. */
  readonly counts: Record<string, number>;
  /** Every chunk mesh with its instance count (visibility statistics). */
  readonly chunks: { mesh: Mesh; count: number; layer: string }[];
  /**
   * Report counts as `${prefix}${layer}` through ctx.reportInstances (which adds the set id;
   * a prefix that repeats it, e.g. "stone." in the stone set, is dropped).
   */
  report(ctx: SceneContext, prefix?: string): void;
  /** Instances in chunks intersecting `frustum` (approximate visible count). */
  visibleCount(frustum: Frustum): number;
}

/** Find the recipe's target meshes under `root` (traversal order = deterministic). */
export function selectTargets(root: Object3D, recipe: VegetationRecipe): Mesh[] {
  const out: Mesh[] = [];
  root.traverse((o) => {
    const m = o as Mesh;
    if (!m.isMesh) return;
    const inc = recipe.targets.include.some((x) => matchesName(m.name, x));
    const exc = (recipe.targets.exclude ?? []).some((x) => matchesName(m.name, x));
    if (inc && !exc) out.push(m);
  });
  return out;
}

/** Scatter and build the recipe's plants now (synchronous). */
export function buildVegetation(ctx: SceneContext, recipe: VegetationRecipe, opts: BuildVegetationOptions): VegetationBuild {
  return runSync(vegetationSteps(ctx, recipe, opts));
}

/**
 * `buildVegetation`, time-sliced: the same deterministic work in the same order (identical
 * random draws and instance data), a stretch per frame within the engine's budget
 * (slices.ts; without a running engine, a macrotask whenever a slice has run for
 * `yieldEveryMs`, default 8 ms). As with the
 * synchronous call, the vegetation group is added to `parent` first and the layers are
 * added to it as they are finished; `parent` and the target meshes must not move until the
 * promise resolves. Concurrent builds (e.g. `Promise.all`) give the same plants; only the
 * creation order of their three.js objects interleaves. Owned by `ctx` (cancelled with the
 * set's prepare: rejects with `SliceCancelled`, see slices.ts).
 */
export function buildVegetationAsync(ctx: SceneContext, recipe: VegetationRecipe, opts: BuildVegetationAsyncOptions): Promise<VegetationBuild> {
  return runSliced(vegetationSteps(ctx, recipe, opts), opts.yieldEveryMs ?? 8, ctx);
}

/** The build as pausable steps (slices.ts): checkpoints sit between triangles, samples, plants and chunks. */
function* vegetationSteps(ctx: SceneContext, recipe: VegetationRecipe, opts: BuildVegetationOptions): Steps<VegetationBuild> {
  const group = new Group();
  group.name = `SilvaVegetation:${recipe.name}:${opts.label}`;
  opts.parent.add(group);
  const meshes = opts.meshes ?? (opts.root ? selectTargets(opts.root, recipe) : []);
  const grass: GrassBatch[] = [];
  const kit: LeafBatch[] = [];
  const counts: Record<string, number> = {};
  const chunks: { mesh: Mesh; count: number; layer: string }[] = [];

  if (meshes.length > 0) {
    const set = yield* SurfaceSet.steps(meshes, opts.parent);
    // view importance / silhouette factor per set of view tags, importance options and rim width (memoized)
    const importanceByTags = new Map<string, Float32Array>();
    const rimByKey = new Map<string, Float32Array>();
    const viewData = function* (tags: string[] | undefined, rimWidth: number, own: ViewImportanceOptions | undefined): Steps<{ importance: Float32Array; rim: Float32Array }> {
      const key = tags && tags.length ? [...tags].sort().join("|") : "*";
      const views = key === "*" ? opts.views : opts.views.filter((w) => w.tag !== undefined && (tags as string[]).includes(w.tag));
      const ik = own ? `${key}#${own.facingMin ?? ""}/${own.facingFull ?? ""}` : key;
      let importance = importanceByTags.get(ik);
      if (!importance) {
        importance = yield* set.viewImportanceSteps(views, own ?? opts.importance);
        importanceByTags.set(ik, importance);
      }
      const rk = `${key}@${rimWidth}`;
      let rim = rimByKey.get(rk);
      if (!rim) {
        rim = yield* set.rimFactorSteps(views, rimWidth);
        rimByKey.set(rk, rim);
      }
      return { importance, rim };
    };
    const density = set.channel(0);
    const meshWeight = meshes.map((m) => {
      const hit = recipe.targetWeights?.find(([x]) => matchesName(m.name, x));
      return hit ? hit[1] : 1;
    });
    const scale = (recipe.densityScale ?? 1) * (opts.budgetScale ?? 1);
    const seed = hashString(`${ctx.seed}|${recipe.name}|${opts.label}`);

    for (const layer of recipe.layers) {
      const { importance, rim } = yield* viewData(layer.viewTags, layer.rimWidth ?? 0.55, layer.importance);
      const power = layer.densityPower ?? 1.5;
      const rimBias = layer.rimBias ?? 0;
      const rimPower = layer.rimPower ?? 1;
      const facing = layer.facing;
      const only = layer.targets;
      const meshOn = meshes.map((m) => (only ? only.some((x) => matchesName(m.name, x)) : true));
      if (!meshOn.some(Boolean)) continue;
      // per-layer density weights by target (first match wins), else the recipe's
      const layerWeight = meshes.map((m, mi) => {
        const hit = layer.targetWeights?.find(([x]) => matchesName(m.name, x));
        return hit ? hit[1] : meshWeight[mi];
      });
      const weights = new Float32Array(set.count);
      const patches = layer.patches;
      const patchSeed = hashString(`${seed}|${layer.name}|patches`);
      for (let t = 0; t < set.count; t++) {
        if ((t & 2047) === 2047) yield;
        const mi = set.meshOf[t];
        if (!meshOn[mi]) continue;
        const d = Math.pow(Math.max(0, density[t]), power);
        // rimBias > 0 crowds the outlines, < 0 keeps the layer off them
        const r = rimBias !== 0 ? rimBias * (rimPower === 1 ? rim[t] : Math.pow(rim[t], rimPower)) : 0;
        let w = set.areas[t] * d * importance[t] * layerWeight[mi] * Math.max(0, 1 + r);
        if (facing && w > 0) w *= smooth01((set.meanNormal(t, tmpN).y - facing.min) / Math.max(1e-4, facing.max - facing.min));
        if (patches && w > 0) {
          set.centroid(t, tmpC);
          const n = fbm3(tmpC.x * patches.scale, tmpC.y * patches.scale, tmpC.z * patches.scale, patchSeed);
          const h = 0.5 * (patches.softness ?? 0.3);
          w *= smooth01((n - (patches.threshold - h)) / (2 * h));
        }
        weights[t] = w;
      }
      const rng = ctx.rng(`${opts.label}/${recipe.name}/${layer.name}`);
      if (layer.kind === "grass") {
        const n = Math.round((layer.count ?? ctx.quality.grassBlades * (layer.budget ?? 0)) * scale);
        if (n <= 0) continue;
        const samples = yield* set.sampleSteps(rng, n, weights);
        const sampleRim = new Float32Array(samples.count);
        for (let i = 0; i < samples.count; i++) sampleRim[i] = rim[samples.tri[i]];
        const batch = yield* GrassBatch.steps(ctx, layer, samples, rng.fork("blades"), seed + grass.length * 101, sampleRim);
        group.add(batch.group);
        grass.push(batch);
        counts[layer.name] = (counts[layer.name] ?? 0) + batch.count;
        for (const m of batch.meshes) chunks.push({ mesh: m, count: instanceCountOf(m), layer: layer.name });
      } else {
        const n = Math.round((layer.count ?? ctx.quality.leaves * (layer.budget ?? 0)) * scale);
        if (n <= 0 || opts.kit.meshes.size === 0) continue;
        const cl = layer.cluster ?? { size: [1, 4] as [number, number], radius: 0.025 };
        const meanMembers = (cl.size[0] + cl.size[1]) / 2;
        const centres = yield* set.sampleSteps(rng, Math.ceil((n / meanMembers) * 1.15), weights);
        const centreRim = new Float32Array(centres.count);
        for (let i = 0; i < centres.count; i++) centreRim[i] = rim[centres.tri[i]];
        let centreScale: Float32Array | null = null;
        if (layer.targetScale?.length) {
          const meshScale = meshes.map((m) => layer.targetScale?.find(([x]) => matchesName(m.name, x))?.[1] ?? 1);
          centreScale = new Float32Array(centres.count);
          for (let i = 0; i < centres.count; i++) centreScale[i] = meshScale[set.meshOf[centres.tri[i]]];
        }
        const batch = yield* LeafBatch.steps(
          ctx,
          layer,
          opts.kit.meshes,
          opts.kit.map,
          { centres, rim: centreRim, scale: centreScale, count: n },
          rng.fork("plants"),
          seed + 7 + kit.length * 131,
        );
        group.add(batch.group);
        kit.push(batch);
        counts[layer.name] = (counts[layer.name] ?? 0) + batch.count;
        for (const m of batch.meshes) chunks.push({ mesh: m, count: instanceCountOf(m), layer: layer.name });
      }
    }
  }

  return {
    group,
    grass,
    kit,
    counts,
    chunks,
    report(c: SceneContext, prefix = "") {
      // c.reportInstances stores `${set}.${key}`: drop a prefix that repeats the set id
      const own = `${c.set}.`;
      const p = prefix.startsWith(own) ? prefix.slice(own.length) : prefix;
      for (const [k, v] of Object.entries(counts)) c.reportInstances(`${p}${k}`, v);
    },
    visibleCount(frustum: Frustum) {
      let n = 0;
      for (const ch of chunks) {
        if (!ch.mesh.visible || !isVisibleInTree(ch.mesh)) continue;
        ch.mesh.updateWorldMatrix(true, false);
        if (frustum.intersectsObject(ch.mesh)) n += ch.count;
      }
      return n;
    },
  };
}

const tmpC = new Vector3();
const tmpN = new Vector3();

function smooth01(x: number): number {
  const c = Math.min(1, Math.max(0, x));
  return c * c * (3 - 2 * c);
}

function instanceCountOf(mesh: Mesh): number {
  const g = mesh.geometry as { instanceCount?: number };
  return typeof g.instanceCount === "number" && Number.isFinite(g.instanceCount) ? g.instanceCount : 0;
}

function isVisibleInTree(o: Object3D): boolean {
  let p: Object3D | null = o;
  while (p) {
    if (!p.visible) return false;
    p = p.parent;
  }
  return true;
}
