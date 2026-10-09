/**
 * Screen-space background: vertical 3-stop gradient + optional radial glow from the
 * top, drawn as the first thing inside the scene render (same MSAA pass, no depth).
 * In post mode it writes linear HDR into the scene target; when the scene is
 * rendered straight to the canvas (`post=0`) it applies tone mapping, the sRGB
 * transfer and output dithering itself.
 */
import {
  ACESFilmicToneMapping,
  AgXToneMapping,
  BufferGeometry,
  Color,
  Float32BufferAttribute,
  GLSL3,
  Mesh,
  NeutralToneMapping,
  RawShaderMaterial,
  Vector2,
  type ToneMapping,
} from "three";
import type { LookParams } from "../types";
import { GLSL_COMMON } from "./FullscreenPass";

const VERTEX = /* glsl */ `
precision highp float;
in vec3 position;
out vec2 vUv;
void main() {
  vUv = position.xy * 0.5 + 0.5;
  gl_Position = vec4(position.xy, 1.0, 1.0);
}
`;

const FRAGMENT = /* glsl */ `
precision highp float;
uniform vec3 uTop;
uniform vec3 uMid;
uniform vec3 uBottom;
uniform float uMidPos;
uniform vec3 uGlowColor;
uniform float uGlowIntensity;
uniform float uGlowRadius;
uniform vec2 uGlowCenter;
uniform float uAspect;
uniform float uExposure;
in vec2 vUv;
out highp vec4 outColor;
${GLSL_COMMON}
#ifdef OUTPUT_TRANSFORM
#include <tonemapping_pars_fragment>
#include <colorspace_pars_fragment>
#endif
void main() {
  float y = vUv.y;
  vec3 c;
  if (y < uMidPos) {
    c = mix(uBottom, uMid, smoothstep(0.0, max(uMidPos, 1e-4), y));
  } else {
    c = mix(uMid, uTop, smoothstep(uMidPos, 1.0, y));
  }
  vec2 d = (vUv - uGlowCenter) * vec2(uAspect, 1.0);
  float r = max(uGlowRadius, 1e-3);
  c += uGlowColor * uGlowIntensity * exp(-dot(d, d) / (r * r));
  vec4 color = vec4(c * uExposure, 1.0);
#ifdef OUTPUT_TRANSFORM
  #if defined(ACES_FILMIC_TONE_MAPPING)
    color.rgb = ACESFilmicToneMapping(color.rgb);
  #elif defined(AGX_TONE_MAPPING)
    color.rgb = AgXToneMapping(color.rgb);
  #elif defined(NEUTRAL_TONE_MAPPING)
    color.rgb = NeutralToneMapping(color.rgb);
  #endif
  color = sRGBTransferOETF(color);
  color.rgb += (ign(gl_FragCoord.xy) - 0.5) / 255.0;
#endif
  outColor = color;
}
`;

export class Background {
  readonly mesh: Mesh<BufferGeometry, RawShaderMaterial>;
  private readonly material: RawShaderMaterial;
  private outputKey = "";

  constructor() {
    const geometry = new BufferGeometry();
    geometry.setAttribute("position", new Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));
    this.material = new RawShaderMaterial({
      name: "SilvaBackground",
      glslVersion: GLSL3,
      vertexShader: VERTEX,
      fragmentShader: FRAGMENT,
      uniforms: {
        uTop: { value: new Color() },
        uMid: { value: new Color() },
        uBottom: { value: new Color() },
        uMidPos: { value: 0.5 },
        uGlowColor: { value: new Color() },
        uGlowIntensity: { value: 0 },
        uGlowRadius: { value: 0.5 },
        uGlowCenter: { value: new Vector2(0.5, 1) },
        uAspect: { value: 1 },
        uExposure: { value: 1 },
        toneMappingExposure: { value: 1 },
      },
      depthTest: false,
      depthWrite: false,
    });
    this.mesh = new Mesh(geometry, this.material);
    this.mesh.name = "SilvaBackground";
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = -1e9;
    this.mesh.layers.enableAll();
    this.mesh.castShadow = false;
    this.mesh.receiveShadow = false;
  }

  /**
   * Update uniforms. `toCanvas` selects the variant with tone mapping + sRGB output
   * (post=0); otherwise the gradient is written linear and `exposure` is left to the
   * compositor.
   */
  setLook(look: LookParams, aspect: number, toCanvas: boolean, toneMapping: ToneMapping): void {
    const u = this.material.uniforms;
    const bg = look.background;
    (u.uTop.value as Color).copy(bg.top);
    (u.uMid.value as Color).copy(bg.mid);
    (u.uBottom.value as Color).copy(bg.bottom);
    u.uMidPos.value = bg.midPosition;
    (u.uGlowColor.value as Color).copy(bg.glow.color);
    u.uGlowIntensity.value = bg.glow.intensity;
    u.uGlowRadius.value = bg.glow.radius;
    (u.uGlowCenter.value as Vector2).set(bg.glow.x, bg.glow.y);
    u.uAspect.value = aspect;
    u.uExposure.value = toCanvas ? look.exposure : 1;
    u.toneMappingExposure.value = 1;

    const key = toCanvas ? `canvas:${toneMapping}` : "linear";
    if (key !== this.outputKey) {
      this.outputKey = key;
      const defines: Record<string, string> = {};
      if (toCanvas) {
        defines.OUTPUT_TRANSFORM = "";
        if (toneMapping === ACESFilmicToneMapping) defines.ACES_FILMIC_TONE_MAPPING = "";
        else if (toneMapping === AgXToneMapping) defines.AGX_TONE_MAPPING = "";
        else if (toneMapping === NeutralToneMapping) defines.NEUTRAL_TONE_MAPPING = "";
      }
      this.material.defines = defines;
      this.material.needsUpdate = true;
    }
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    this.material.dispose();
  }
}
