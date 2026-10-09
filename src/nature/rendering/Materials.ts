/**
 * Surface material factories (all MeshStandardMaterial, metalness 0).
 *
 * - Textures come from `public/nature/textures/{bark,rock,moss}_{basecolor,normal,orm}.webp`
 *   when present (basecolor sRGB, normal/ORM as data; ORM: R = AO, G = roughness);
 *   flat fallback colours otherwise. Textures are registry-owned; materials belong to
 *   the scene that created them (dispose them with the scene).
 * - Every factory multiplies albedo by the baked AO in `COLOR_0.b` (vertex colour is
 *   never used as albedo). Geometry without a `color` attribute gets a constant
 *   white one (AO = 1) when its scene set is prepared (`prepareSilvaGeometry`).
 * - Rock and moss are triplanar (object space, metric scale) via onBeforeCompile.
 */
import {
  BufferAttribute,
  Color,
  FrontSide,
  MeshStandardMaterial,
  Vector2,
  type BufferGeometry,
  type ColorRepresentation,
  type Mesh,
  type Object3D,
  type Side,
  type Texture,
  type WebGLProgramParametersWithUniforms,
} from "three";
import type { AssetRegistry } from "../AssetRegistry";

export interface SurfaceOptions {
  /** Albedo when textures are missing; tint (multiplied) when they exist. */
  color?: ColorRepresentation;
  roughness?: number;
  /** Texture repeats per metre (triplanar) or UV repeat (bark). */
  scale?: number;
  /** Strength of the baked AO from COLOR_0.b (default 1). */
  aoStrength?: number;
  /** Strength of the ORM texture AO on indirect light (default 1). */
  textureAoStrength?: number;
  /** Load textures if present (default true). */
  useTextures?: boolean;
  /** Triplanar projection space (rock / moss), default "object". */
  space?: "object" | "world";
  side?: Side;
}

// ---------------------------------------------------------------------------
// Baked AO (COLOR_0.b)
// ---------------------------------------------------------------------------

const AO_CHECKED = new WeakSet<BufferGeometry>();
const AO_WARNED = new WeakSet<BufferGeometry>();

/**
 * Give geometry without COLOR_0 a constant white colour attribute (AO = 1).
 * Must run before the geometry is first rendered: three.js uploads buffers and
 * builds the vertex array object on the first draw and does not notice attributes
 * added later. The engine calls `prepareSilvaGeometry(scene)` right after a scene
 * set's `prepare()`; call this yourself for meshes created after that.
 */
export function ensureColorAttribute(geometry: BufferGeometry): void {
  if (AO_CHECKED.has(geometry)) return;
  AO_CHECKED.add(geometry);
  if (geometry.getAttribute("color")) return;
  const position = geometry.getAttribute("position");
  if (!position) return;
  geometry.setAttribute("color", new BufferAttribute(new Float32Array(position.count * 4).fill(1), 4));
}

/** Run `ensureColorAttribute` on every mesh under `root` that uses a factory material. */
export function prepareSilvaGeometry(root: Object3D): void {
  root.traverse((o) => {
    const mesh = o as Mesh;
    if (!mesh.isMesh || !mesh.geometry || !mesh.material) return;
    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    if (materials.some((m) => m.userData?.silvaUniforms)) ensureColorAttribute(mesh.geometry);
  });
}

/** Late safety net: too late to bind for this geometry, so tell the author. */
function checkColorAttribute(geometry: BufferGeometry): void {
  if (AO_CHECKED.has(geometry) || geometry.getAttribute("color")) return;
  if (!AO_WARNED.has(geometry)) {
    AO_WARNED.add(geometry);
    if (process.env.NODE_ENV !== "production") {
      console.warn(
        `[nature] geometry "${geometry.name || geometry.uuid}" has no COLOR_0 and was added after prepare(); ` +
          "call ensureColorAttribute(geometry) before it is first rendered (baked AO reads 0 otherwise).",
      );
    }
  }
}

const AO_VERTEX_PARS = /* glsl */ `
#if !defined(USE_COLOR) && !defined(USE_COLOR_ALPHA)
attribute vec4 color;
#endif
varying float vSilvaAo;
`;

const AO_FRAGMENT_PARS = /* glsl */ `
varying float vSilvaAo;
uniform float uSilvaAoStrength;
`;

// ---------------------------------------------------------------------------
// Triplanar (Golus whiteout blend), object space with metric scale by default
// ---------------------------------------------------------------------------

const TP_VERTEX_PARS = /* glsl */ `
varying vec3 vTpPos;
varying vec3 vTpNormal;
varying vec3 vTpBX;
varying vec3 vTpBY;
varying vec3 vTpBZ;
`;

const TP_VERTEX_MAIN = /* glsl */ `
#ifdef TP_WORLD
  vec4 tpW = vec4(transformed, 1.0);
  #ifdef USE_INSTANCING
    tpW = instanceMatrix * tpW;
  #endif
  tpW = modelMatrix * tpW;
  vTpPos = tpW.xyz;
  vec3 tpWn = objectNormal;
  #ifdef USE_INSTANCING
    tpWn = mat3(instanceMatrix) * tpWn;
  #endif
  vTpNormal = normalize(mat3(modelMatrix) * tpWn);
  vTpBX = normalize(mat3(viewMatrix) * vec3(1.0, 0.0, 0.0));
  vTpBY = normalize(mat3(viewMatrix) * vec3(0.0, 1.0, 0.0));
  vTpBZ = normalize(mat3(viewMatrix) * vec3(0.0, 0.0, 1.0));
#else
  vec3 tpScale = vec3(length(modelMatrix[0].xyz), length(modelMatrix[1].xyz), length(modelMatrix[2].xyz));
  vTpPos = transformed * tpScale;
  vTpNormal = objectNormal;
  mat3 tpN = normalMatrix;
  #ifdef USE_INSTANCING
    tpN = tpN * mat3(instanceMatrix);
  #endif
  vTpBX = normalize(tpN * vec3(1.0, 0.0, 0.0));
  vTpBY = normalize(tpN * vec3(0.0, 1.0, 0.0));
  vTpBZ = normalize(tpN * vec3(0.0, 0.0, 1.0));
#endif
`;

const TP_FRAGMENT_PARS = /* glsl */ `
uniform float uTpScale;
uniform float uTpSharpness;
uniform float uTpAoStrength;
uniform sampler2D tTpColor;
uniform sampler2D tTpNormal;
uniform sampler2D tTpOrm;
varying vec3 vTpPos;
varying vec3 vTpNormal;
varying vec3 vTpBX;
varying vec3 vTpBY;
varying vec3 vTpBZ;
struct TpFrame { vec3 w; vec3 s; vec3 n; vec2 uvX; vec2 uvY; vec2 uvZ; };
TpFrame tpSetup() {
  TpFrame f;
  f.n = normalize(vTpNormal);
  vec3 w = pow(abs(f.n), vec3(uTpSharpness));
  f.w = w / max(w.x + w.y + w.z, 1e-5);
  f.s = vec3(f.n.x < 0.0 ? -1.0 : 1.0, f.n.y < 0.0 ? -1.0 : 1.0, f.n.z < 0.0 ? -1.0 : 1.0);
  vec3 p = vTpPos * uTpScale;
  f.uvX = vec2(p.z * f.s.x, p.y);
  f.uvY = vec2(p.x * f.s.y, p.z);
  f.uvZ = vec2(-p.x * f.s.z, p.y);
  return f;
}
vec4 tpSample(sampler2D tex, TpFrame f) {
  return texture2D(tex, f.uvX) * f.w.x + texture2D(tex, f.uvY) * f.w.y + texture2D(tex, f.uvZ) * f.w.z;
}
vec3 tpNormal(TpFrame f) {
  vec3 tx = texture2D(tTpNormal, f.uvX).xyz * 2.0 - 1.0;
  vec3 ty = texture2D(tTpNormal, f.uvY).xyz * 2.0 - 1.0;
  vec3 tz = texture2D(tTpNormal, f.uvZ).xyz * 2.0 - 1.0;
  tx.x *= f.s.x;
  ty.x *= f.s.y;
  tz.x *= -f.s.z;
  vec3 an = abs(f.n);
  tx = vec3(tx.xy + f.n.zy, an.x * tx.z);
  ty = vec3(ty.xy + f.n.xz, an.y * ty.z);
  tz = vec3(tz.xy + f.n.xy, an.z * tz.z);
  tx.z *= f.s.x;
  ty.z *= f.s.y;
  tz.z *= f.s.z;
  return normalize(tx.zyx * f.w.x + ty.xzy * f.w.y + tz.xyz * f.w.z);
}
float tpHash3(vec3 p) {
  p = fract(p * 0.1031);
  p += dot(p, p.zyx + 31.32);
  return fract((p.x + p.y) * p.z);
}
float tpNoise3(vec3 p) {
  vec3 i = floor(p);
  vec3 f = fract(p);
  vec3 u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(mix(tpHash3(i), tpHash3(i + vec3(1, 0, 0)), u.x), mix(tpHash3(i + vec3(0, 1, 0)), tpHash3(i + vec3(1, 1, 0)), u.x), u.y),
    mix(mix(tpHash3(i + vec3(0, 0, 1)), tpHash3(i + vec3(1, 0, 1)), u.x), mix(tpHash3(i + vec3(0, 1, 1)), tpHash3(i + vec3(1, 1, 1)), u.x), u.y),
    u.z);
}
`;

interface InjectOptions {
  triplanar: boolean;
  defines: string[];
  uniforms: Record<string, { value: unknown }>;
  aoStrength: number;
}

function inject(material: MeshStandardMaterial, opts: InjectOptions): void {
  const aoUniform = { value: opts.aoStrength };
  material.userData.silvaUniforms = { uSilvaAoStrength: aoUniform, ...opts.uniforms };
  const defineBlock = opts.defines.map((d) => `#define ${d}`).join("\n");
  material.onBeforeCompile = (shader: WebGLProgramParametersWithUniforms) => {
    shader.uniforms.uSilvaAoStrength = aoUniform;
    for (const [k, v] of Object.entries(opts.uniforms)) shader.uniforms[k] = v;

    let vs = shader.vertexShader;
    vs = vs.replace("#include <common>", `#include <common>\n${defineBlock}\n${AO_VERTEX_PARS}${opts.triplanar ? TP_VERTEX_PARS : ""}`);
    vs = vs.replace("#include <color_vertex>", "#include <color_vertex>\n  vSilvaAo = color.b;");
    if (opts.triplanar) vs = vs.replace("#include <begin_vertex>", `#include <begin_vertex>\n${TP_VERTEX_MAIN}`);
    shader.vertexShader = vs;

    let fs = shader.fragmentShader;
    fs = fs.replace("#include <common>", `#include <common>\n${defineBlock}\n${AO_FRAGMENT_PARS}${opts.triplanar ? TP_FRAGMENT_PARS : ""}`);
    fs = fs.replace("#include <color_fragment>", "#include <color_fragment>\n  diffuseColor.rgb *= mix(1.0, vSilvaAo, uSilvaAoStrength);");
    if (opts.triplanar) {
      fs = fs.replace(
        "#include <map_fragment>",
        /* glsl */ `
  TpFrame tp = tpSetup();
  #ifdef TP_COLOR
    diffuseColor.rgb *= tpSample(tTpColor, tp).rgb;
  #endif
  #ifdef TP_PROCEDURAL
    float tpN = tpNoise3(vTpPos * 1.7) * 0.65 + tpNoise3(vTpPos * 7.3) * 0.35;
    diffuseColor.rgb *= mix(0.72, 1.12, tpN);
  #endif
  #ifdef TP_ORM
    vec4 tpOrm = tpSample(tTpOrm, tp);
  #endif`,
      );
      fs = fs.replace(
        "#include <roughnessmap_fragment>",
        `#include <roughnessmap_fragment>
  #ifdef TP_ORM
    roughnessFactor *= tpOrm.g;
  #endif`,
      );
      fs = fs.replace(
        "#include <normal_fragment_maps>",
        `#ifdef TP_NORMAL
    normal = normalize(mat3(vTpBX, vTpBY, vTpBZ) * tpNormal(tp));
  #endif`,
      );
      fs = fs.replace(
        "#include <aomap_fragment>",
        `#include <aomap_fragment>
  #ifdef TP_ORM
    reflectedLight.indirectDiffuse *= mix(1.0, tpOrm.r, uTpAoStrength);
  #endif`,
      );
    }
    shader.fragmentShader = fs;
  };
  const key = `silva-surface|${opts.triplanar ? "tp" : "uv"}|${opts.defines.join(",")}`;
  material.customProgramCacheKey = () => key;
  material.onBeforeRender = (_renderer, _scene, _camera, geometry) => checkColorAttribute(geometry);
}

/** Uniforms injected into a factory material (e.g. to animate aoStrength). */
export function silvaUniforms(material: MeshStandardMaterial): Record<string, { value: unknown }> {
  return (material.userData.silvaUniforms ?? {}) as Record<string, { value: unknown }>;
}

interface TextureSet {
  basecolor: Texture | null;
  normal: Texture | null;
  orm: Texture | null;
}

async function loadSet(assets: AssetRegistry, prefix: "bark" | "rock" | "moss", use: boolean): Promise<TextureSet> {
  if (!use) return { basecolor: null, normal: null, orm: null };
  const [basecolor, normal, orm] = await Promise.all([
    assets.natureTexture(`${prefix}_basecolor`),
    assets.natureTexture(`${prefix}_normal`),
    assets.natureTexture(`${prefix}_orm`),
  ]);
  return { basecolor, normal, orm };
}

function withRepeat(tex: Texture | null, repeat: number): Texture | null {
  if (!tex || repeat === 1) return tex;
  const clone = tex.clone();
  clone.repeat.set(repeat, repeat);
  clone.needsUpdate = true;
  return clone;
}

// ---------------------------------------------------------------------------
// Factories
// ---------------------------------------------------------------------------

/** Bark on UV-mapped wood (TEXCOORD_0: U around the trunk, V along the grain, 1 unit = 0.5 m). */
export async function createBarkMaterial(assets: AssetRegistry, opts: SurfaceOptions = {}): Promise<MeshStandardMaterial> {
  const set = await loadSet(assets, "bark", opts.useTextures ?? true);
  const repeat = opts.scale ?? 1;
  const hasMap = !!set.basecolor;
  const mat = new MeshStandardMaterial({
    name: "SilvaBark",
    color: new Color(opts.color ?? (hasMap ? 0xffffff : "#5E4D3B")),
    roughness: opts.roughness ?? (set.orm ? 1 : 0.88),
    metalness: 0,
    side: opts.side ?? FrontSide,
    map: withRepeat(set.basecolor, repeat),
    normalMap: withRepeat(set.normal, repeat),
    // glTF UVs without tangents: three derives the tangent frame from UV derivatives;
    // flip Y like GLTFLoader does for derivative tangents.
    normalScale: new Vector2(1, -1),
    roughnessMap: withRepeat(set.orm, repeat),
    aoMap: withRepeat(set.orm, repeat),
    aoMapIntensity: opts.textureAoStrength ?? 1,
  });
  inject(mat, { triplanar: false, defines: [], uniforms: {}, aoStrength: opts.aoStrength ?? 1 });
  return mat;
}

async function createTriplanar(
  assets: AssetRegistry,
  prefix: "rock" | "moss",
  name: string,
  fallback: string,
  defaultScale: number,
  defaultRoughness: number,
  opts: SurfaceOptions,
): Promise<MeshStandardMaterial> {
  const set = await loadSet(assets, prefix, opts.useTextures ?? true);
  const defines: string[] = [];
  if (opts.space === "world") defines.push("TP_WORLD");
  if (set.basecolor) defines.push("TP_COLOR");
  else defines.push("TP_PROCEDURAL");
  if (set.normal) defines.push("TP_NORMAL");
  if (set.orm) defines.push("TP_ORM");
  const mat = new MeshStandardMaterial({
    name,
    color: new Color(opts.color ?? (set.basecolor ? 0xffffff : fallback)),
    roughness: opts.roughness ?? (set.orm ? 1 : defaultRoughness),
    metalness: 0,
    side: opts.side ?? FrontSide,
  });
  inject(mat, {
    triplanar: true,
    defines,
    aoStrength: opts.aoStrength ?? 1,
    uniforms: {
      uTpScale: { value: opts.scale ?? defaultScale },
      uTpSharpness: { value: 4 },
      uTpAoStrength: { value: opts.textureAoStrength ?? 1 },
      tTpColor: { value: set.basecolor },
      tTpNormal: { value: set.normal },
      tTpOrm: { value: set.orm },
    },
  });
  return mat;
}

/** Rock, triplanar (default 0.5 repeats per metre for the 2048² rock textures). */
export function createRockMaterial(assets: AssetRegistry, opts: SurfaceOptions = {}): Promise<MeshStandardMaterial> {
  return createTriplanar(assets, "rock", "SilvaRock", "#7A7263", 0.5, 0.82, opts);
}

/** Dense moss base under scattered vegetation, triplanar (default 2 repeats per metre). */
export function createMossBaseMaterial(assets: AssetRegistry, opts: SurfaceOptions = {}): Promise<MeshStandardMaterial> {
  return createTriplanar(assets, "moss", "SilvaMossBase", "#46561C", 2, 0.92, opts);
}

/** Distant dark silhouettes (`far_*`): flat, rough, low contrast. */
export function createFarMaterial(opts: SurfaceOptions = {}): MeshStandardMaterial {
  const mat = new MeshStandardMaterial({
    name: "SilvaFar",
    color: new Color(opts.color ?? "#141A18"),
    roughness: opts.roughness ?? 1,
    metalness: 0,
    side: opts.side ?? FrontSide,
  });
  inject(mat, { triplanar: false, defines: [], uniforms: {}, aoStrength: opts.aoStrength ?? 1 });
  return mat;
}

/** Seed emblem floor (`emblem_*`): warm beige-gold, matte. */
export function createEmblemMaterial(opts: SurfaceOptions = {}): MeshStandardMaterial {
  const mat = new MeshStandardMaterial({
    name: "SilvaEmblem",
    color: new Color(opts.color ?? "#B89B68"),
    roughness: opts.roughness ?? 0.6,
    metalness: 0,
    side: opts.side ?? FrontSide,
  });
  inject(mat, { triplanar: false, defines: [], uniforms: {}, aoStrength: opts.aoStrength ?? 1 });
  return mat;
}
