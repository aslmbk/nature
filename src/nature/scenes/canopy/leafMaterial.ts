/**
 * Canopy leaf material and instanced leaf meshes.
 *
 * One `MeshStandardMaterial` (double sided, the kit atlas as albedo) patched in
 * `onBeforeCompile`; three.js keeps doing the lights, fog and tone mapping. Every leaf is
 * one instance of a kit leaf (`InstancedBufferGeometry`, 10 triangles) with the packed
 * per-instance data of `foliage.ts`. Per cluster (crown clusters 0–45, the two far shells
 * 46 / 47) two vec4 uniforms carry the story state, so the reveal and the exit never
 * touch the instance buffers:
 *
 *   uClA[c] = (base xyz, positional scale)   leaf root = base + rel × scale + offset
 *   uClB[c] = (offset xyz, grow progress)    leaves unfurl from the cluster base outwards
 *
 * Exit (canopyClose → finale): a hole opens around the view axis. Each leaf root moves
 * outwards so that its screen radius ρ becomes √(ρ² + R²) (area preserving: no leaf is
 * lost or piled up, the middle opens first, the frame edges last); R gets a few smooth
 * angular lobes (no circle) and a lumpy per-cluster jitter (the crown leaves in clumps).
 * Leaves pushed outwards grow a little, as if the lens passed them.
 *
 * Light: the key reaching a leaf is scaled by its baked transmittance (`aLook.w`, see
 * foliage.ts), the hemisphere fill by the baked AO of its twig; light reaching the back
 * of a blade shows through it (thin-sheet translucency imitation, warm yellow-green).
 * Undersides are darker and greyer. No shadow maps.
 */
import {
  Color,
  DoubleSide,
  InstancedBufferGeometry,
  InstancedInterleavedBuffer,
  InterleavedBufferAttribute,
  Mesh,
  MeshStandardMaterial,
  StaticDrawUsage,
  Vector3,
  Vector4,
  type BufferGeometry,
  type IUniform,
  type Texture,
  type WebGLProgramParametersWithUniforms,
} from "three";
import { WIND_GLSL, bindWindUniforms, type WindUniforms } from "../../vegetation/WindField";
import { LEAF_ATTRIBUTES, LEAF_STRIDE } from "./foliage";

/** Uniform array length: 46 crown clusters + 2 far shells (+ spare). */
export const MAX_CLUSTERS = 48;

export interface LeafUniforms {
  [name: string]: IUniform;
  /** Per cluster: base xyz (crown space), positional scale. */
  uClA: IUniform<Vector4[]>;
  /** Per cluster: extra offset xyz, grow progress 0–1. */
  uClB: IUniform<Vector4[]>;
  /** Exit: camera position / forward / right / up (crown space). */
  uExitCam: IUniform<Vector3>;
  uExitF: IUniform<Vector3>;
  uExitRight: IUniform<Vector3>;
  uExitUp: IUniform<Vector3>;
  /** Exit: relative amplitudes of the 3 / 5 / 9-fold lobes of the hole, phase. */
  uExitLobe: IUniform<Vector4>;
  /** Exit: x hole radius R (0 = off), y = tan(fov/2) × frame diagonal (ρ normaliser), z = size boost 0–1. */
  uExit: IUniform<Vector4>;
  /** Wind amount (0 = still). */
  uSway: IUniform<number>;
  /** Strength of the light shining through the blades. */
  uTransl: IUniform<number>;
  /** Albedo of the undersides relative to the tops. */
  uBackShade: IUniform<number>;
  /** Key light left in fully shaded leaves (0–1). */
  uKeyFloor: IUniform<number>;
  /** Exponent on the baked key transmittance (contrast of the crown's depth). */
  uKeyGamma: IUniform<number>;
  /** Story factor on that exponent: < 1 while the crown grows (fewer leaves in the way). */
  uShadeK: IUniform<number>;
  /** Darkening of the leaves in the crown's hollow (0 none … 1 the close-up's dark middle). */
  uHollow: IUniform<number>;
  /** Albedo saturation (1 = atlas) and gain. */
  uLeafSat: IUniform<number>;
  uLeafGain: IUniform<number>;
  /** Colour of the light shining through a blade (multiplies the albedo). */
  uTranslTint: IUniform<Color>;
}

export function createLeafUniforms(): LeafUniforms {
  return {
    uClA: { value: Array.from({ length: MAX_CLUSTERS }, () => new Vector4(0, 0, 0, 1)) },
    uClB: { value: Array.from({ length: MAX_CLUSTERS }, () => new Vector4(0, 0, 0, 1)) },
    uExitCam: { value: new Vector3() },
    uExitF: { value: new Vector3(0, 0, -1) },
    uExitRight: { value: new Vector3(1, 0, 0) },
    uExitUp: { value: new Vector3(0, 1, 0) },
    uExitLobe: { value: new Vector4(0.16, 0.1, 0.06, 0) },
    uExit: { value: new Vector4(0, 1, 0.35, 0) },
    uSway: { value: 1 },
    uTransl: { value: 0.55 },
    uBackShade: { value: 0.62 },
    uKeyFloor: { value: 0.06 },
    uKeyGamma: { value: 1.8 },
    uShadeK: { value: 1 },
    uHollow: { value: 0 },
    uLeafSat: { value: 0.8 },
    uLeafGain: { value: 1 },
    uTranslTint: { value: new Color(0.95, 1.05, 0.7) },
  };
}

const VERTEX_PARS = /* glsl */ `
#if !defined( USE_COLOR ) && !defined( USE_COLOR_ALPHA )
attribute vec4 color;
#endif
attribute vec3 aRel;
attribute vec3 aAxis;
attribute vec3 aSide;
attribute vec4 aMisc;
attribute vec4 aLook;
attribute vec4 aAnim;
attribute float aHollow;
uniform vec4 uClA[ ${MAX_CLUSTERS} ];
uniform vec4 uClB[ ${MAX_CLUSTERS} ];
uniform vec3 uExitCam;
uniform vec3 uExitF;
uniform vec3 uExitRight;
uniform vec3 uExitUp;
uniform vec4 uExitLobe;
uniform vec4 uExit;
uniform float uSway;
uniform float uHollow;
varying vec3 vLeafTint;
varying float vLeafKey;
varying float vLeafAo;
varying float vLeafTransl;
${WIND_GLSL}
`;

/** Replaces `#include <beginnormal_vertex>`; defines objectNormal and silvaLeafPos. */
const VERTEX_MAIN = /* glsl */ `
  int leafCluster = int( aMisc.y + 0.5 );
  vec4 leafClA = uClA[ leafCluster ];
  vec4 leafClB = uClB[ leafCluster ];
  // leaves unfurl from the cluster base outwards while the cluster grows
  float leafGrow = clamp( ( leafClB.w - 0.55 * aAnim.y ) / 0.45, 0.0, 1.0 );
  leafGrow = leafGrow * leafGrow * ( 3.0 - 2.0 * leafGrow );
  vec3 leafRoot = leafClA.xyz + aRel * leafClA.w + leafClB.xyz;
  vec3 leafAxis = aAxis;
  vec3 leafSide = aSide;
  vec3 leafN = cross( leafSide, leafAxis );
  // wind: the spray sways with its arm, the branch a little, the blade flutters
  vec3 leafSway = silvaWindSway( leafRoot, aMisc.z, 1.0 );
  leafRoot += leafSway * ( aMisc.w * 3.0 + length( aRel ) * 0.12 ) * uSway;
  float leafFl = silvaFlutter( aMisc.z, 1.7 ) * uWindFlutter * uWindStrength * 0.9 * uSway;
  float leafC = cos( leafFl );
  float leafS = sin( leafFl );
  vec3 leafAxis2 = leafAxis * leafC + leafN * leafS;
  leafN = leafN * leafC - leafAxis * leafS;
  leafAxis = leafAxis2;
  // exit: an area-preserving hole opens around the view axis
  float leafBoost = 1.0;
  if ( uExit.x > 0.0 ) {
    vec3 ev = leafRoot - uExitCam;
    float ed = max( dot( ev, uExitF ), 0.05 );
    vec3 ep = ev - uExitF * ed;
    float er = length( ep );
    float erho = er / ( ed * uExit.y );
    float eTh = atan( dot( ep, uExitUp ), dot( ep, uExitRight ) + 1e-6 );
    float eLobe = 1.0 + uExitLobe.x * sin( 3.0 * eTh + uExitLobe.w ) + uExitLobe.y * sin( 5.0 * eTh + 1.7 * uExitLobe.w + 2.1 ) + uExitLobe.z * sin( 9.0 * eTh + 0.6 * uExitLobe.w + 4.0 );
    float eR = max( 0.0, uExit.x * eLobe + aAnim.z * min( 1.0, uExit.x * 4.0 ) );
    float erho2 = sqrt( erho * erho + eR * eR );
    vec3 edir = er > 1e-5 ? ep / er : normalize( leafSide - uExitF * dot( leafSide, uExitF ) + vec3( 1e-4, 0.0, 0.0 ) );
    leafRoot = uExitCam + uExitF * ed + edir * ( erho2 * ed * uExit.y );
    leafBoost = mix( 1.0, min( erho2 / max( erho, 1e-3 ), 2.5 ), uExit.z );
  }
  float leafScale = aMisc.x * leafGrow * leafBoost;
  vec3 silvaLeafPos = leafRoot + ( leafSide * position.x + leafAxis * position.y + leafN * position.z ) * leafScale;
  vec3 objectNormal = normalize( leafSide * normal.x + leafAxis * normal.y + leafN * normal.z );
  // kit COLOR_0.g: per-vertex shade (darker at the petiole and along the midrib)
  vLeafTint = aLook.rgb * mix( 0.8, 1.0, color.g );
  // the hollow: its leaves lose the key and most of the fill
  float leafHollow = clamp( uHollow * aHollow * 1.6, 0.0, 1.0 );
  vLeafKey = aLook.w * ( 1.0 - leafHollow );
  vLeafAo = aAnim.x * ( 1.0 - 0.75 * leafHollow );
  vLeafTransl = aAnim.w;
`;

const FRAGMENT_PARS = /* glsl */ `
varying vec3 vLeafTint;
varying float vLeafKey;
varying float vLeafAo;
varying float vLeafTransl;
uniform float uTransl;
uniform float uBackShade;
uniform float uKeyFloor;
uniform float uKeyGamma;
uniform float uShadeK;
uniform float uLeafSat;
uniform float uLeafGain;
uniform vec3 uTranslTint;
`;

const FRAGMENT_COLOR = /* glsl */ `
  {
    vec3 leafAlb = diffuseColor.rgb * vLeafTint;
    float leafL = dot( leafAlb, vec3( 0.2126, 0.7152, 0.0722 ) );
    leafAlb = mix( vec3( leafL ), leafAlb, uLeafSat ) * uLeafGain;
    // undersides: darker and a little greyer
    if ( !gl_FrontFacing ) leafAlb = mix( vec3( dot( leafAlb, vec3( 0.2126, 0.7152, 0.0722 ) ) ), leafAlb, 0.8 ) * uBackShade;
    diffuseColor.rgb = leafAlb;
  }
`;

/** Key light × baked transmittance, physical BRDF, light through the blade. */
const DIRECT = /* glsl */ `
void RE_Direct_SilvaLeaf( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
  IncidentLight leafLight = directLight;
  leafLight.color *= mix( uKeyFloor, 1.0, pow( clamp( vLeafKey, 1e-4, 1.0 ), uKeyGamma * uShadeK ) );
  RE_Direct_Physical( leafLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
  float back = saturate( dot( -geometryNormal, leafLight.direction ) );
  float into = pow( saturate( dot( -geometryViewDir, leafLight.direction ) ), 3.0 );
  reflectedLight.directDiffuse += leafLight.color * BRDF_Lambert( material.diffuseContribution ) * uTranslTint * ( vLeafTransl * uTransl * back * ( 0.6 + 1.4 * into ) );
}
#undef RE_Direct
#define RE_Direct RE_Direct_SilvaLeaf
`;

const AO = /* glsl */ `
#include <aomap_fragment>
  reflectedLight.indirectDiffuse *= vLeafAo;
  reflectedLight.indirectSpecular *= vLeafAo;
`;

export function createLeafMaterial(map: Texture | null, uniforms: LeafUniforms, wind: WindUniforms): MeshStandardMaterial {
  const material = new MeshStandardMaterial({
    name: "SilvaCanopyLeaf",
    color: map ? 0xffffff : 0x5f8a3a,
    map,
    roughness: 0.4,
    metalness: 0,
    side: DoubleSide,
  });
  material.onBeforeCompile = (shader: WebGLProgramParametersWithUniforms) => {
    bindWindUniforms(shader.uniforms, wind);
    for (const [k, v] of Object.entries(uniforms)) shader.uniforms[k] = v;
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", `#include <common>\n${VERTEX_PARS}`)
      .replace("#include <beginnormal_vertex>", VERTEX_MAIN)
      .replace("#include <begin_vertex>", "vec3 transformed = silvaLeafPos;");
    shader.fragmentShader = shader.fragmentShader
      .replace("#include <common>", `#include <common>\n${FRAGMENT_PARS}`)
      .replace("#include <color_fragment>", `#include <color_fragment>\n${FRAGMENT_COLOR}`)
      .replace("#include <lights_physical_pars_fragment>", `#include <lights_physical_pars_fragment>\n${DIRECT}`)
      .replace("#include <aomap_fragment>", AO);
  };
  material.customProgramCacheKey = () => `silva-canopy-leaf|${map ? "map" : "flat"}`;
  return material;
}

/**
 * Instanced mesh of one kit leaf with packed instance data (LEAF_STRIDE floats each).
 * The kit attributes are cloned (10 vertices): disposing this geometry must never
 * delete the GPU buffers of the shared kit.
 */
export function createLeafMesh(name: string, kitGeometry: BufferGeometry, data: Float32Array, count: number, material: MeshStandardMaterial): Mesh {
  const geometry = new InstancedBufferGeometry();
  geometry.name = name;
  if (kitGeometry.index) geometry.setIndex(kitGeometry.index.clone());
  for (const attr of ["position", "normal", "uv", "color"]) {
    const a = kitGeometry.getAttribute(attr);
    if (a) geometry.setAttribute(attr, a.clone());
  }
  const buffer = new InstancedInterleavedBuffer(data, LEAF_STRIDE, 1);
  buffer.setUsage(StaticDrawUsage);
  for (const [attr, [offset, size]] of Object.entries(LEAF_ATTRIBUTES)) {
    geometry.setAttribute(attr, new InterleavedBufferAttribute(buffer, size, offset));
  }
  geometry.instanceCount = count;
  const mesh = new Mesh(geometry, material);
  mesh.name = name;
  mesh.frustumCulled = false;
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  return mesh;
}
