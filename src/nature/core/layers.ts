import type { Object3D } from "three";

/**
 * Named debug layers mapped onto THREE.Layers bits. Objects assigned to a layer are
 * shown or hidden through the camera layer mask (URL `layers=wood,moss` or the debug
 * panel). This never touches `object.visible`, so scenes stay free to animate it.
 * Unassigned objects stay on bit 0 and are always visible.
 */
export const STANDARD_LAYERS = [
  "wood",
  "moss",
  "grass",
  "leaves",
  "flowers",
  "rock",
  "stone",
  "emblem",
  "far",
  "particles",
  "orb",
  "fx",
  "placeholder",
] as const;

export class LayerRegistry {
  private readonly bits = new Map<string, number>();

  constructor() {
    STANDARD_LAYERS.forEach((name) => this.bit(name));
  }

  /** Bit index (1–31) of a layer; new names are allocated on first use. */
  bit(name: string): number {
    let b = this.bits.get(name);
    if (b === undefined) {
      b = this.bits.size + 1;
      if (b > 31) throw new Error(`LayerRegistry: too many layers (adding "${name}")`);
      this.bits.set(name, b);
    }
    return b;
  }

  names(): string[] {
    return [...this.bits.keys()];
  }

  /** Put `object` (and by default all its descendants) on layer `name` only. */
  assign(object: Object3D, name: string, recursive = true): void {
    const b = this.bit(name);
    if (recursive) object.traverse((o) => o.layers.set(b));
    else object.layers.set(b);
  }

  /** Camera mask for a set of visible layer names (null = all). Bit 0 is always on. */
  maskFor(visible: ReadonlySet<string> | null): number {
    if (visible === null) return 0xffffffff | 0;
    let mask = 1;
    for (const [name, b] of this.bits) if (visible.has(name)) mask |= 1 << b;
    return mask;
  }
}
