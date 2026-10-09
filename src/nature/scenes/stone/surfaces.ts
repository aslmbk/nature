/**
 * Shader tweaks of the stone set on top of the shared factories (wrapped
 * `onBeforeCompile`, own program cache keys; `rendering/Materials.ts` and the plant
 * materials stay untouched):
 *
 *  - `addMossLight`   the moss strip's own light: an extra directional term along the
 *                     key (Lambert with optional wrap, unshadowed, fading with the
 *                     distance from the pool's centre) and one from the top light that
 *                     only the moss materials see — the pool's cookie keeps the rock
 *                     under the strip dark and the top light stays off the rock's
 *                     bumps (frame 14) while the moss on the edge is lit, a stand-in for
 *                     light linking; then everything the material reflects (direct +
 *                     indirect) is scaled by a shared gain: the moss ramps up from dark
 *                     (38.5 s, frame 13: no moss reads on the rim) to full (≈ 40.5 s)
 *                     while the rock under it is already lit;
 *  - `addSandGrain`   the recess floor of the seed (frame 14 / detail crop): a fine,
 *                     slightly sparkling sand grain — albedo speckle and a tiny bump
 *                     from metric object-space value noise (≈ 2–3 mm grains), no texture;
 *                     optional warm bounce light (the floor in the lip's shadow);
 *  - `addRecessTint`  the recess walls and chamfer (part of the rock mesh) take the
 *                     floor's warm golden-beige where they lie below a height above the
 *                     floor, inside the ellipse round the seed and turned away from the
 *                     plane's normal (frame 14: the walls read golden, not grey; the face
 *                     round the recess keeps the rock), plus a warm bounce light that
 *                     falls off from the floor to the lip (the walls in shadow read deep
 *                     brown-gold, not black); optionally without the rock's normal map
 *                     there (the lip's chamfer at the key's terminator broke up into
 *                     black and lit pixels);
 *  - `dropShadowNormalOffset` no normal offset in the shadow lookup of the recess floor
 *                     (it never casts; the offset along its rounded edge pushed the
 *                     lookup into the slab: a lit hairline at the foot of the walls);
 *  - `addDensityShade` a darker base where COLOR_0.R asks for few plants (the thin right
 *                     fringe of the moss strip: single strands on a dark edge).
 */
import { Color, Matrix4, Vector3, Vector4, type IUniform, type MeshStandardMaterial, type WebGLProgramParametersWithUniforms, type WebGLRenderer } from "three";

const NOISE = /* glsl */ `
float stHash13(vec3 p3) {
  p3 = fract(p3 * 0.1031);
  p3 += dot(p3, p3.zyx + 31.32);
  return fract((p3.x + p3.y) * p3.z);
}
float stNoise(vec3 p) {
  vec3 i = floor(p);
  vec3 f = fract(p);
  vec3 u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(mix(stHash13(i), stHash13(i + vec3(1, 0, 0)), u.x), mix(stHash13(i + vec3(0, 1, 0)), stHash13(i + vec3(1, 1, 0)), u.x), u.y),
    mix(mix(stHash13(i + vec3(0, 0, 1)), stHash13(i + vec3(1, 0, 1)), u.x), mix(stHash13(i + vec3(0, 1, 1)), stHash13(i + vec3(1, 1, 1)), u.x), u.y),
    u.z);
}
`;

function chain(material: MeshStandardMaterial, tag: string, patch: (shader: WebGLProgramParametersWithUniforms) => void): void {
  const prev = material.onBeforeCompile;
  material.onBeforeCompile = (shader: WebGLProgramParametersWithUniforms, renderer: WebGLRenderer) => {
    prev.call(material, shader, renderer);
    patch(shader);
  };
  const baseKey = material.customProgramCacheKey();
  material.customProgramCacheKey = () => `${baseKey}|${tag}`;
  material.needsUpdate = true;
}

/** Shared uniforms of the moss light (one set for all moss materials of the scene). */
export interface MossLightUniforms {
  /** Overall gain of everything the moss reflects (0–1 ramp). */
  uStoneLightGain: IUniform<number>;
  /** Colour × irradiance of the moss-only key (linear). */
  uStoneMossKey: IUniform<Color>;
  /** Direction towards the moss-only key, view space (update per frame). */
  uStoneMossDir: IUniform<Vector3>;
  /** Wrap of the term: light reaching round the tufts (0 = Lambert). */
  uStoneMossWrap: IUniform<number>;
  /** Centre of the moss key's fall-off, view space (update per frame). */
  uStoneMossCenter: IUniform<Vector3>;
  /** Radius (m) of the fall-off: × exp(−(d / r)²). */
  uStoneMossRadius: IUniform<number>;
  /** Colour × irradiance of the moss-only top light (linear). */
  uStoneMossTop: IUniform<Color>;
  /** Direction towards the moss-only top light, view space (update per frame). */
  uStoneMossTopDir: IUniform<Vector3>;
}

export function createMossLightUniforms(): MossLightUniforms {
  return {
    uStoneLightGain: { value: 1 },
    uStoneMossKey: { value: new Color(0, 0, 0) },
    uStoneMossDir: { value: new Vector3(0, 1, 0) },
    uStoneMossWrap: { value: 0.3 },
    uStoneMossCenter: { value: new Vector3(0, 0, -1) },
    uStoneMossRadius: { value: 1e3 },
    uStoneMossTop: { value: new Color(0, 0, 0) },
    uStoneMossTopDir: { value: new Vector3(0, 1, 0) },
  };
}

/**
 * The moss-only key and top light and the moss light gain (see file header) on
 * `material`. `translucency` (thin blades / leaves): share of the top light that also
 * shows through a blade lit from behind (two-sided term; 0 for the opaque strip).
 */
export function addMossLight(material: MeshStandardMaterial, light: MossLightUniforms, translucency = 0): void {
  const transl = Math.max(0, translucency).toFixed(3);
  chain(material, `stone-moss-light-${transl}`, (shader) => {
    if (!shader.fragmentShader.includes("#include <opaque_fragment>")) {
      console.warn(`[stone] moss light: anchor missing in ${material.name}`);
      return;
    }
    Object.assign(shader.uniforms, light);
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>
uniform float uStoneLightGain;
uniform vec3 uStoneMossKey;
uniform vec3 uStoneMossDir;
uniform float uStoneMossWrap;
uniform vec3 uStoneMossCenter;
uniform float uStoneMossRadius;
uniform vec3 uStoneMossTop;
uniform vec3 uStoneMossTopDir;`,
      )
      .replace(
        "#include <opaque_fragment>",
        `{
    float mossNl = (dot(normal, uStoneMossDir) + uStoneMossWrap) / (1.0 + uStoneMossWrap);
    float mossD = length(-vViewPosition - uStoneMossCenter) / uStoneMossRadius;
    outgoingLight += uStoneMossKey * BRDF_Lambert(diffuseColor.rgb) * saturate(mossNl) * exp(-mossD * mossD);
    float mossTop = dot(normal, uStoneMossTopDir);
    outgoingLight += uStoneMossTop * BRDF_Lambert(diffuseColor.rgb) * (saturate(mossTop) + ${transl} * saturate(-mossTop));
  }
  outgoingLight *= uStoneLightGain;
#include <opaque_fragment>`,
      );
  });
}

export interface DensityShadeOptions {
  /** COLOR_0.R where the darkening starts to lift / is gone. */
  from: number;
  to: number;
  /** Albedo factor at R ≤ from. */
  dark: number;
}

/**
 * Darker base where the mesh asks for few plants (COLOR_0.R low): on the thin right
 * fringe of the moss strip only single strands stand on a dark edge (frame 14), the base
 * lumps must not read as lit green skin between them.
 */
export function addDensityShade(material: MeshStandardMaterial, o: DensityShadeOptions): void {
  const uniforms = { uDensShade: { value: new Vector3(o.from, o.to, o.dark) } };
  chain(material, "stone-density-shade", (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", "#include <common>\nvarying float vDensShade;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\n  vDensShade = color.r;");
    shader.fragmentShader = shader.fragmentShader
      .replace("#include <common>", "#include <common>\nvarying float vDensShade;\nuniform vec3 uDensShade;")
      .replace("#include <color_fragment>", "#include <color_fragment>\n  diffuseColor.rgb *= mix(uDensShade.z, 1.0, smoothstep(uDensShade.x, uDensShade.y, vDensShade));");
  });
}

export interface SandGrainOptions {
  /** Grain cells per metre (default 380 ≈ 2.6 mm). */
  frequency?: number;
  /** 0–1 albedo speckle contrast (default 0.35). */
  contrast?: number;
  /** Bump height in metres of one grain (default 0.0006). */
  bump?: number;
  /** Bounce light (linear radiance per unit albedo; shared, updated per frame). */
  bounce?: IUniform<Color>;
}

/** Fine sand grain on the emblem floor (object-space noise; the GLB transforms are applied, so it is metric). */
export function addSandGrain(material: MeshStandardMaterial, opts: SandGrainOptions = {}): void {
  const uniforms = {
    uSandFreq: { value: opts.frequency ?? 380 },
    uSandContrast: { value: opts.contrast ?? 0.35 },
    uSandBump: { value: opts.bump ?? 0.0006 },
    uSandBounce: opts.bounce ?? { value: new Color(0, 0, 0) },
  };
  chain(material, "stone-sand", (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", "#include <common>\nvarying vec3 vSandPos;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\n  vSandPos = transformed;");
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>
varying vec3 vSandPos;
uniform float uSandFreq;
uniform float uSandContrast;
uniform float uSandBump;
uniform vec3 uSandBounce;
${NOISE}
float sandHeight(vec3 p) {
  return stNoise(p * uSandFreq) * 0.65 + stNoise(p * uSandFreq * 2.3 + 7.1) * 0.35;
}`,
      )
      .replace(
        "#include <color_fragment>",
        `#include <color_fragment>
  float sandH = sandHeight(vSandPos);
  float sandSparkle = smoothstep(0.78, 0.95, stNoise(vSandPos * uSandFreq * 1.7 + 3.3));
  diffuseColor.rgb *= 1.0 - uSandContrast + 2.0 * uSandContrast * sandH + 0.25 * sandSparkle;
  float sandBroad = stNoise(vSandPos * 22.0);
  diffuseColor.rgb *= 0.92 + 0.16 * sandBroad;`,
      )
      .replace(
        "#include <normal_fragment_maps>",
        `#include <normal_fragment_maps>
  {
    // bump from the grain height (screen-space derivatives, perturbNormalArb)
    float h = sandH * uSandBump;
    vec2 dh = vec2(dFdx(h), dFdy(h));
    vec3 sx = dFdx(-vViewPosition);
    vec3 sy = dFdy(-vViewPosition);
    vec3 r1 = cross(sy, normal);
    vec3 r2 = cross(normal, sx);
    float det = dot(sx, r1);
    vec3 grad = sign(det) * (dh.x * r1 + dh.y * r2);
    normal = normalize(abs(det) * normal - grad);
  }`,
      )
      .replace("#include <emissivemap_fragment>", "#include <emissivemap_fragment>\n  totalEmissiveRadiance += uSandBounce * diffuseColor.rgb;");
  });
}

/**
 * No normal offset in the shadow lookup of a receiver that never casts (the recess floor:
 * `castShadow` false, so it cannot shadow itself and needs no acne guard). The key's
 * `normalBias` along the floor's rounded edge normals pushed the lookup into the slab: a
 * lit hairline along the foot of the shadowed walls.
 */
export function dropShadowNormalOffset(material: MeshStandardMaterial): void {
  chain(material, "stone-no-shadow-offset", (shader) => {
    shader.vertexShader = shader.vertexShader.replace(
      "#include <shadowmap_vertex>",
      "#define transformNormalByInverseViewMatrix(n, v) vec3(0.0)\n#include <shadowmap_vertex>\n#undef transformNormalByInverseViewMatrix",
    );
  });
}

export interface RecessTintOptions {
  /** Emblem plane: a point on the floor and its unit normal (object space of the rock = world, transforms applied). */
  point: Vector3;
  normal: Vector3;
  /** Outline in the plane: centre, unit axes and half sizes (m) of the ellipse round the seed. */
  center: Vector3;
  axisU: Vector3;
  axisV: Vector3;
  halfU: number;
  halfV: number;
  /** Height (m) above the floor where the tint starts to fade / is gone (walls and chamfer below it). */
  depth: [number, number];
  /** Albedo of the walls (sRGB hex). */
  color: string;
  /** Bounce light on the walls (linear radiance per unit albedo; shared, updated per frame). */
  bounce: IUniform<Color>;
  /** Share of the bounce left at the top of the wall (it falls off away from the lit floor; default 0.3). */
  bounceTop?: number;
  /** Height (m) above the floor the bounce falls off to (default `depth[1]`). */
  wallTop?: number;
  /**
   * Share of the rock's normal map taken off the tinted walls and chamfer (0 keeps it,
   * 1 = the smooth geometric normal; default 0). The 45° chamfer of the lip sits close to
   * the key's terminator: with the rock's relief its pixels flip between lit and black.
   */
  flatNormal?: number;
}

/**
 * Golden-beige recess walls on the rock material (see file header). Only rock that lies
 * inside the ellipse round the seed, below `depth` above the floor and is not turned to
 * the plane's normal (walls ⟂, the 45° chamfer) is tinted: the face round the recess keeps
 * the rock even where its relief dips low.
 */
export function addRecessTint(material: MeshStandardMaterial, o: RecessTintOptions): void {
  // plane → local frame: x along U, y along V, z along the normal (height above the floor)
  const toPlane = new Matrix4().makeBasis(o.axisU, o.axisV, o.normal).setPosition(o.center).invert();
  const floorHeight = new Vector3().copy(o.point).applyMatrix4(toPlane).z;
  const uniforms = {
    uRecessToPlane: { value: toPlane },
    uRecessFloor: { value: floorHeight },
    uRecessHalf: { value: new Vector3(o.halfU, o.halfV, 0) },
    uRecessDepth: { value: new Vector4(o.depth[0], o.depth[1], o.bounceTop ?? 0.3, o.wallTop ?? o.depth[1]) },
    uRecessColor: { value: new Color(o.color) },
    uRecessBounce: o.bounce,
    uRecessFlat: { value: o.flatNormal ?? 0 },
  };
  chain(material, "stone-recess-3", (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", "#include <common>\nvarying vec3 vRecessPos;\nvarying vec3 vRecessNormal;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\n  vRecessPos = transformed;\n  vRecessNormal = objectNormal;");
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>
varying vec3 vRecessPos;
varying vec3 vRecessNormal;
uniform mat4 uRecessToPlane;
uniform float uRecessFloor;
uniform vec3 uRecessHalf;
uniform vec4 uRecessDepth;
uniform vec3 uRecessColor;
uniform vec3 uRecessBounce;
uniform float uRecessFlat;`,
      )
      .replace(
        "#include <color_fragment>",
        `#include <color_fragment>
  float recessWall;
  float recessUp;
  {
    vec3 rp = (uRecessToPlane * vec4(vRecessPos, 1.0)).xyz;
    float inside = 1.0 - smoothstep(0.96, 1.04, length(rp.xy / uRecessHalf.xy));
    float above = rp.z - uRecessFloor;
    float facing = abs(normalize(mat3(uRecessToPlane) * vRecessNormal).z);
    recessUp = clamp(above / uRecessDepth.w, 0.0, 1.0);
    recessWall = inside * (1.0 - smoothstep(uRecessDepth.x, uRecessDepth.y, above)) * step(-0.006, above) * (1.0 - smoothstep(0.86, 0.96, facing));
    diffuseColor.rgb = mix(diffuseColor.rgb, uRecessColor, recessWall);
  }`,
      )
      .replace(
        "#include <emissivemap_fragment>",
        `#include <emissivemap_fragment>
  totalEmissiveRadiance += recessWall * mix(1.0, uRecessDepth.z, recessUp) * uRecessBounce * diffuseColor.rgb;
  normal = normalize(mix(normal, nonPerturbedNormal, recessWall * uRecessFlat));`,
      );
  });
}
