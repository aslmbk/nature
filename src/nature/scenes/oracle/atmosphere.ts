/**
 * Screen-space air of the oracle scene set, added on top of the engine background
 * (which carries the wide soft glow at the top centre, see EPISODE_LOOKS):
 *
 *  - beam: the narrow green shaft from the top, slightly left of centre (frames 07–09:
 *    ~150 px wide at the top, widening and fading out by y ≈ 0.4);
 *  - field: a very wide, faint elliptical lift of the dark green in the middle of the
 *    frame, so the edges fall off to near black (measured columns: #030502 at the
 *    edges, #0A1208 in the middle).
 *
 * A fullscreen triangle at the far plane: additive, depth-tested (passes only where
 * nothing opaque was drawn), first in the blend layer (layer.ts). Code-generated
 * background, no textures.
 */
import { BufferGeometry, Color, Float32BufferAttribute, Mesh, ShaderMaterial, Vector2, type IUniform } from "three";
import { OUTPUT_TAIL } from "./glsl";
import { additiveKeepAlpha } from "./layer";

const VERTEX = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = position.xy * 0.5 + 0.5;
  gl_Position = vec4(position.xy, 1.0, 1.0);
}
`;

const FRAGMENT = /* glsl */ `
uniform float uAspect;
uniform vec3 uBeamColor;
uniform float uBeamX;
uniform float uBeamWidth;
uniform float uBeamLength;
uniform vec3 uFieldColor;
uniform vec2 uFieldCenter;
uniform vec2 uFieldRadius;
uniform float uOpacity;
varying vec2 vUv;
void main() {
  float fromTop = 1.0 - vUv.y;
  float w = uBeamWidth * (1.0 + fromTop * 1.6);
  float dx = (vUv.x - uBeamX) * uAspect;
  float beam = exp(-dx * dx / (w * w)) * exp(-fromTop / uBeamLength) * (uBeamWidth / w);
  vec2 d = (vUv - uFieldCenter) * vec2(uAspect, 1.0) / uFieldRadius;
  float field = exp(-dot(d, d));
  vec3 c = (uBeamColor * beam + uFieldColor * field) * uOpacity;
  gl_FragColor = vec4(c, 1.0);
  ${OUTPUT_TAIL}
}
`;

export interface Atmosphere {
  mesh: Mesh<BufferGeometry, ShaderMaterial>;
  uniforms: {
    uAspect: IUniform<number>;
    uBeamX: IUniform<number>;
    uBeamWidth: IUniform<number>;
    uBeamLength: IUniform<number>;
    uBeamColor: IUniform<Color>;
    uFieldColor: IUniform<Color>;
    uFieldCenter: IUniform<Vector2>;
    uFieldRadius: IUniform<Vector2>;
    uOpacity: IUniform<number>;
  };
}

export function buildAtmosphere(): Atmosphere {
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));
  const uniforms = {
    uAspect: { value: 1440 / 1020 },
    uBeamX: { value: 0.428 },
    uBeamWidth: { value: 0.066 },
    uBeamLength: { value: 0.15 },
    uBeamColor: { value: new Color(0.027, 0.0675, 0.021) },
    uFieldColor: { value: new Color(0.0061, 0.0095, 0.005) },
    uFieldCenter: { value: new Vector2(0.52, 0.48) },
    uFieldRadius: { value: new Vector2(0.5, 0.55) },
    uOpacity: { value: 1 },
  };
  const material = additiveKeepAlpha(
    new ShaderMaterial({
      name: "OracleAtmosphere",
      uniforms,
      vertexShader: VERTEX,
      fragmentShader: FRAGMENT,
      transparent: true,
      depthTest: true,
      depthWrite: false,
    }),
  );
  const mesh = new Mesh(geometry, material);
  mesh.name = "oracle_atmosphere";
  mesh.frustumCulled = false;
  mesh.renderOrder = -1e8;
  return { mesh, uniforms };
}
