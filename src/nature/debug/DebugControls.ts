/**
 * Developer panel (lil-gui), loaded only with `debug=1` (never in capture).
 * Shows episode / t / local progress / seed / quality / DPR / GPU and frame stats,
 * and drives the engine through `DebugTarget`.
 */
import GUI, { type Controller } from "lil-gui";
import type { QualityLevel, ToneMappingName } from "../types";
import { QUALITY_LEVELS, TONE_MAPPINGS } from "../types";
import { VIDEO_DURATION } from "../SceneConfig";

export type DebugToggle = "post" | "dof" | "bloom" | "vignette" | "wind" | "wireframe" | "parallax";

export interface DebugInfo {
  episode: string;
  t: number;
  localProgress: number;
  timeSec: number;
  frozen: boolean;
  seed: number;
  quality: QualityLevel;
  toneMapping: ToneMappingName;
  dpr: number;
  gpu: string;
  frameMsAvg: number;
  frameMsP95: number;
  cpuMsAvg: number;
  drawCalls: number;
  triangles: number;
  instances: Record<string, number>;
  passes: string[];
  sets: string;
  toggles: Record<DebugToggle, boolean>;
}

export interface DebugTarget {
  info(): DebugInfo;
  setT(t: number): void;
  followScroll(): void;
  setFrozen(frozen: boolean, time: number): void;
  setSeed(seed: number): void;
  setQuality(level: QualityLevel): void;
  setToneMapping(name: ToneMappingName): void;
  setToggle(name: DebugToggle, value: boolean): void;
  layerNames(): string[];
  isLayerVisible(name: string): boolean;
  setLayerVisible(name: string, visible: boolean): void;
}

export class DebugControls {
  private readonly gui: GUI;
  private readonly view = {
    episode: "",
    t: 0,
    local: "",
    seed: 134,
    quality: "high" as QualityLevel,
    toneMapping: "aces" as ToneMappingName,
    frozen: false,
    time: 2,
    frame: "",
    cpu: "",
    calls: "",
    tris: "",
    instances: "",
    dpr: "",
    gpu: "",
    passes: "",
    sets: "",
    followScroll: () => this.target.followScroll(),
  };
  private readonly toggles: Record<DebugToggle, boolean> = {
    post: true,
    dof: true,
    bloom: true,
    vignette: true,
    wind: true,
    wireframe: false,
    parallax: false,
  };
  private readonly readOnly: Controller[] = [];
  private tController: Controller;
  private timeController: Controller;
  private draggingT = false;
  private lastRefresh = 0;

  constructor(private readonly target: DebugTarget) {
    this.gui = new GUI({ title: "Silva debug", width: 320 });
    this.gui.domElement.style.zIndex = "50";
    const info = target.info();
    Object.assign(this.toggles, info.toggles);
    this.view.seed = info.seed;
    this.view.quality = info.quality;
    this.view.toneMapping = info.toneMapping;
    this.view.frozen = info.frozen;
    this.view.time = info.timeSec;

    const story = this.gui.addFolder("Story");
    this.readOnly.push(story.add(this.view, "episode").disable());
    this.tController = story
      .add(this.view, "t", 0, VIDEO_DURATION, 0.01)
      .name("t (video s)")
      .onChange((v: number) => {
        this.draggingT = true;
        target.setT(v);
      })
      .onFinishChange(() => {
        this.draggingT = false;
      });
    this.readOnly.push(story.add(this.view, "local").name("local progress").disable());
    story.add(this.view, "followScroll").name("follow scroll again");
    story
      .add(this.view, "frozen")
      .name("freeze time")
      .onChange((v: boolean) => target.setFrozen(v, this.view.time));
    this.timeController = story
      .add(this.view, "time", 0, 120, 0.01)
      .name("time (s)")
      .onChange((v: number) => {
        if (this.view.frozen) target.setFrozen(true, v);
      });

    const render = this.gui.addFolder("Render");
    render.add(this.view, "quality", [...QUALITY_LEVELS]).onChange((v: QualityLevel) => target.setQuality(v));
    render.add(this.view, "toneMapping", [...TONE_MAPPINGS]).name("tone mapping").onChange((v: ToneMappingName) => target.setToneMapping(v));
    render
      .add(this.view, "seed", 0, 100000, 1)
      .onFinishChange((v: number) => target.setSeed(Math.round(v)));
    for (const key of Object.keys(this.toggles) as DebugToggle[]) {
      render.add(this.toggles, key).onChange((v: boolean) => target.setToggle(key, v));
    }

    const layers = this.gui.addFolder("Layers");
    const layerState: Record<string, boolean> = {};
    for (const name of target.layerNames()) {
      layerState[name] = target.isLayerVisible(name);
      layers.add(layerState, name).onChange((v: boolean) => target.setLayerVisible(name, v));
    }
    layers.close();

    const stats = this.gui.addFolder("Stats");
    this.readOnly.push(
      stats.add(this.view, "frame").name("frame ms avg/p95").disable(),
      stats.add(this.view, "cpu").name("cpu ms avg").disable(),
      stats.add(this.view, "calls").name("draw calls").disable(),
      stats.add(this.view, "tris").name("triangles").disable(),
      stats.add(this.view, "instances").disable(),
      stats.add(this.view, "dpr").name("DPR").disable(),
      stats.add(this.view, "gpu").name("GPU").disable(),
      stats.add(this.view, "passes").disable(),
      stats.add(this.view, "sets").name("scene sets").disable(),
    );
  }

  /** Refresh displayed values (throttled to ~4 Hz). */
  update(now: number): void {
    if (now - this.lastRefresh < 250) return;
    this.lastRefresh = now;
    const i = this.target.info();
    this.view.episode = i.episode;
    if (!this.draggingT) {
      this.view.t = Math.round(i.t * 100) / 100;
      this.tController.updateDisplay();
    }
    this.view.local = i.localProgress.toFixed(3);
    if (!this.view.frozen) {
      this.view.time = Math.round(i.timeSec * 100) / 100;
      this.timeController.updateDisplay();
    }
    this.view.frame = `${i.frameMsAvg.toFixed(2)} / ${i.frameMsP95.toFixed(2)}`;
    this.view.cpu = i.cpuMsAvg.toFixed(2);
    this.view.calls = String(i.drawCalls);
    this.view.tris = i.triangles.toLocaleString("en-US");
    const inst = Object.entries(i.instances);
    this.view.instances = inst.length ? inst.map(([k, v]) => `${k}=${v}`).join(" ") : "—";
    this.view.dpr = i.dpr.toFixed(2);
    this.view.gpu = i.gpu;
    this.view.passes = i.passes.join(" → ");
    this.view.sets = i.sets;
    for (const c of this.readOnly) c.updateDisplay();
  }

  dispose(): void {
    this.gui.destroy();
  }
}
