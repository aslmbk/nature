import {
  BufferGeometry,
  Float32BufferAttribute,
  GLSL3,
  Mesh,
  OrthographicCamera,
  RawShaderMaterial,
  type IUniform,
  type WebGLRenderer,
  type WebGLRenderTarget,
} from "three";

/** Vertex shader shared by every fullscreen pass (one oversized triangle). */
export const FULLSCREEN_VERTEX = /* glsl */ `
precision highp float;
in vec3 position;
out vec2 vUv;
void main() {
  vUv = position.xy * 0.5 + 0.5;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

/** Common GLSL helpers (hash, value noise, fbm, interleaved gradient noise). */
export const GLSL_COMMON = /* glsl */ `
float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
float valueNoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  float a = hash12(i);
  float b = hash12(i + vec2(1.0, 0.0));
  float c = hash12(i + vec2(0.0, 1.0));
  float d = hash12(i + vec2(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
float fbm(vec2 p) {
  float s = 0.0;
  float a = 0.5;
  for (int i = 0; i < 5; i++) {
    s += a * valueNoise(p);
    p = mat2(1.6, 1.2, -1.2, 1.6) * p + 17.13;
    a *= 0.5;
  }
  return s / 0.96875;
}
float ign(vec2 fragCoord) {
  return fract(52.9829189 * fract(dot(fragCoord, vec2(0.06711056, 0.00583715))));
}
`;

let sharedGeometry: BufferGeometry | null = null;
let sharedUsers = 0;

function acquireTriangle(): BufferGeometry {
  if (!sharedGeometry) {
    sharedGeometry = new BufferGeometry();
    sharedGeometry.setAttribute("position", new Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));
  }
  sharedUsers++;
  return sharedGeometry;
}

function releaseTriangle(): void {
  sharedUsers--;
  if (sharedUsers <= 0 && sharedGeometry) {
    sharedGeometry.dispose();
    sharedGeometry = null;
    sharedUsers = 0;
  }
}

const camera = new OrthographicCamera(-1, 1, 1, -1, 0, 1);

export interface PassMaterialOptions {
  name: string;
  fragmentShader: string;
  uniforms: Record<string, IUniform>;
  defines?: Record<string, string | number>;
}

/** GLSL3 RawShaderMaterial with the shared fullscreen vertex shader. */
export function createPassMaterial(opts: PassMaterialOptions): RawShaderMaterial {
  return new RawShaderMaterial({
    name: opts.name,
    glslVersion: GLSL3,
    vertexShader: FULLSCREEN_VERTEX,
    fragmentShader: opts.fragmentShader,
    uniforms: opts.uniforms,
    defines: opts.defines ?? {},
    depthTest: false,
    depthWrite: false,
  });
}

/** A fullscreen triangle drawing one material into a target (or the canvas). */
export class FullscreenPass {
  readonly mesh: Mesh<BufferGeometry, RawShaderMaterial>;

  constructor(readonly material: RawShaderMaterial) {
    this.mesh = new Mesh(acquireTriangle(), material);
    this.mesh.frustumCulled = false;
  }

  render(renderer: WebGLRenderer, target: WebGLRenderTarget | null): void {
    renderer.setRenderTarget(target);
    renderer.render(this.mesh, camera);
  }

  dispose(): void {
    this.material.dispose();
    releaseTriangle();
  }
}
