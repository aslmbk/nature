/**
 * The soft yellow-green haze along the top of the stone frame (frame 14: out-of-focus
 * light at the top edge — two broad yellow-olive patches around x ≈ 0.2 and x ≈ 0.75 of
 * the width, greyer light in the top corners, a dim band between them). Measured in
 * linear light it falls as (1 − d / 0.12)^1.6 with the distance d below the edge (in
 * frame heights); the colours below are fitted through the grade's ACES and corner
 * vignette to the measured top row (#474A41 · #585D2F · #222219 · #65663F · #42443F at
 * x = 40 / 280 / 600 / 1080 / 1400 px).
 *
 * A fullscreen triangle at the far plane: additive, depth-tested (it only shows where
 * nothing opaque was drawn), right after the engine background. Code-generated
 * background, no textures. Its slow drift is a pure function of the ambient clock.
 */
import { AdditiveBlending, BufferGeometry, Color, Float32BufferAttribute, Mesh, ShaderMaterial, Vector4, type IUniform } from "three";

const VERTEX = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = position.xy * 0.5 + 0.5;
  gl_Position = vec4(position.xy, 1.0, 1.0);
}
`;

const FRAGMENT = /* glsl */ `
uniform float uAspect;
uniform float uTime;
uniform float uDepth;
uniform float uPower;
uniform float uOpacity;
uniform vec3 uBandColor;
uniform vec3 uBlobColor;
uniform vec3 uEdgeColor;
uniform vec4 uBlobA;
uniform vec4 uBlobB;
uniform vec4 uEdgeA;
uniform vec4 uEdgeB;
varying vec2 vUv;
// patch: x centre (0–1 of the width), half width (frame heights), strength, phase
float hazePatch(vec4 b, float x) {
  float cx = b.x + 0.012 * sin(uTime * 0.09 + b.w);
  float d = (x - cx) * uAspect / b.y;
  return b.z * (1.0 + 0.08 * sin(uTime * 0.13 + 1.9 * b.w)) * exp(-d * d);
}
void main() {
  float fromTop = 1.0 - vUv.y;
  float band = pow(clamp(1.0 - fromTop / uDepth, 0.0, 1.0), uPower);
  vec3 c = uBandColor
    + uBlobColor * (hazePatch(uBlobA, vUv.x) + hazePatch(uBlobB, vUv.x))
    + uEdgeColor * (hazePatch(uEdgeA, vUv.x) + hazePatch(uEdgeB, vUv.x));
  gl_FragColor = vec4(c * band * uOpacity, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

export interface Haze {
  mesh: Mesh<BufferGeometry, ShaderMaterial>;
  uniforms: {
    uAspect: IUniform<number>;
    uTime: IUniform<number>;
    uDepth: IUniform<number>;
    uPower: IUniform<number>;
    uOpacity: IUniform<number>;
    uBandColor: IUniform<Color>;
    uBlobColor: IUniform<Color>;
    uEdgeColor: IUniform<Color>;
    uBlobA: IUniform<Vector4>;
    uBlobB: IUniform<Vector4>;
    uEdgeA: IUniform<Vector4>;
    uEdgeB: IUniform<Vector4>;
  };
}

/** Linear HDR colours (before exposure / grade). */
export function buildHaze(): Haze {
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));
  const uniforms = {
    uAspect: { value: 1440 / 1020 },
    uTime: { value: 0 },
    uDepth: { value: 0.122 },
    uPower: { value: 1.6 },
    uOpacity: { value: 1 },
    uBandColor: { value: new Color(0.012, 0.013, 0.012) },
    uBlobColor: { value: new Color(0.1, 0.103, 0.033) },
    uEdgeColor: { value: new Color(0.1, 0.108, 0.1) },
    uBlobA: { value: new Vector4(0.195, 0.2, 0.86, 0.0) },
    uBlobB: { value: new Vector4(0.75, 0.2, 1.17, 2.1) },
    uEdgeA: { value: new Vector4(-0.03, 0.25, 0.65, 4.4) },
    uEdgeB: { value: new Vector4(1.03, 0.25, 0.64, 5.3) },
  };
  const material = new ShaderMaterial({
    name: "StoneHaze",
    uniforms,
    vertexShader: VERTEX,
    fragmentShader: FRAGMENT,
    blending: AdditiveBlending,
    transparent: true,
    depthTest: true,
    depthWrite: false,
    fog: false,
  });
  const mesh = new Mesh(geometry, material);
  mesh.name = "stone_haze";
  mesh.frustumCulled = false;
  mesh.renderOrder = -1e8;
  return { mesh, uniforms };
}
