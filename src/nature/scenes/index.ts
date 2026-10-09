/**
 * Scene set registry. Each set is a separate chunk, loaded when the SceneDirector
 * first needs it (current set + next one preloaded). To replace a placeholder,
 * rewrite its file; the engine core does not change.
 */
import type { SceneFactory, SceneSetId } from "../types";

export const SCENE_FACTORIES: Record<SceneSetId, SceneFactory> = {
  grove: () => import("./GroveScene").then((m) => new m.GroveScene()),
  canyon: () => import("./CanyonScene").then((m) => new m.CanyonScene()),
  oracle: () => import("./OracleScene").then((m) => new m.OracleScene()),
  branch: () => import("./BranchScene").then((m) => new m.BranchScene()),
  stone: () => import("./StoneScene").then((m) => new m.StoneScene()),
  canopy: () => import("./CanopyScene").then((m) => new m.CanopyScene()),
  finale: () => import("./FinaleScene").then((m) => new m.FinaleScene()),
};
