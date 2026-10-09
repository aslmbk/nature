/**
 * Default three-light setup driven by LookParams.lights: key (directional, slightly
 * warm, from above/side), fill (hemisphere, grey-green on light scenes) and a weak
 * rim (directional from behind). Optional for scenes; BaseScene creates one.
 */
import { Color, DirectionalLight, Group, HemisphereLight, Object3D, Vector3 } from "three";
import type { LookParams } from "../types";

const tmp = new Vector3();

export class LightRig {
  readonly group = new Group();
  readonly key = new DirectionalLight(0xffffff, 2);
  readonly fill = new HemisphereLight(0xffffff, 0x222222, 1);
  readonly rim = new DirectionalLight(0xffffff, 0.8);
  /** Multipliers on top of the look hints (per-scene artistic scale). */
  scale = { key: 1, fill: 1, rim: 1 };

  constructor() {
    this.group.name = "SilvaLightRig";
    this.key.name = "key";
    this.rim.name = "rim";
    this.fill.name = "fill";
    this.key.position.set(2.5, 4, 3);
    this.rim.position.set(-3, 2.5, -4);
    this.group.add(this.key, this.key.target, this.rim, this.rim.target, this.fill);
  }

  /** Point the key light along `direction` (world), aimed at `target`. */
  setKeyDirection(direction: Vector3, target = new Vector3(), distance = 6): void {
    this.key.target.position.copy(target);
    this.key.position.copy(target).addScaledVector(tmp.copy(direction).normalize(), -distance);
  }

  /** Use a `key_<episode>` empty: its local −Z is the light direction. */
  setKeyFromObject(empty: Object3D, target = new Vector3()): void {
    empty.updateMatrixWorld(true);
    const dir = new Vector3(0, 0, -1).transformDirection(empty.matrixWorld);
    this.setKeyDirection(dir, target);
  }

  setRimDirection(direction: Vector3, target = new Vector3(), distance = 6): void {
    this.rim.target.position.copy(target);
    this.rim.position.copy(target).addScaledVector(tmp.copy(direction).normalize(), -distance);
  }

  apply(look: LookParams): void {
    const l = look.lights;
    this.key.color.copy(l.keyColor);
    this.key.intensity = l.keyIntensity * this.scale.key;
    this.fill.color.copy(l.fillSky);
    (this.fill.groundColor as Color).copy(l.fillGround);
    this.fill.intensity = l.fillIntensity * this.scale.fill;
    this.rim.color.copy(l.rimColor);
    this.rim.intensity = l.rimIntensity * this.scale.rim;
  }
}
