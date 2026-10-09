/**
 * GLSL shared by the oracle scene set: hash / value noise, rotations and the
 * point-sprite model used by every particle layer (orb, streams, dust).
 *
 * Sprite model: a particle has a world-space extent and a linear colour that is
 * its peak brightness at that nominal size. The vertex part converts it to pixels,
 * clamps tiny sprites to `uMinPx` (dimming them so the energy stays the same) and
 * grows a defocus disc for particles off the focus distance (near ones much more
 * than far ones), again energy-conserving: near particles become big, dim, flat
 * bokeh discs, far ones stay small and sharp.
 */

export const NOISE_GLSL = /* glsl */ `
float oHash13(vec3 p3) {
  p3 = fract(p3 * 0.1031);
  p3 += dot(p3, p3.zyx + 31.32);
  return fract((p3.x + p3.y) * p3.z);
}
float oNoise(vec3 p) {
  vec3 i = floor(p);
  vec3 f = fract(p);
  vec3 u = f * f * (3.0 - 2.0 * f);
  float n000 = oHash13(i);
  float n100 = oHash13(i + vec3(1.0, 0.0, 0.0));
  float n010 = oHash13(i + vec3(0.0, 1.0, 0.0));
  float n110 = oHash13(i + vec3(1.0, 1.0, 0.0));
  float n001 = oHash13(i + vec3(0.0, 0.0, 1.0));
  float n101 = oHash13(i + vec3(1.0, 0.0, 1.0));
  float n011 = oHash13(i + vec3(0.0, 1.0, 1.0));
  float n111 = oHash13(i + vec3(1.0, 1.0, 1.0));
  return mix(
    mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
    mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y),
    u.z
  );
}
// three decorrelated noises in [-1, 1]
vec3 oNoise3(vec3 p) {
  return vec3(
    oNoise(p),
    oNoise(p + vec3(31.7, 11.3, 5.9)),
    oNoise(p + vec3(-7.1, 23.9, 17.3))
  ) * 2.0 - 1.0;
}
mat3 oRotY(float a) {
  float c = cos(a);
  float s = sin(a);
  return mat3(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c);
}
mat3 oRotX(float a) {
  float c = cos(a);
  float s = sin(a);
  return mat3(1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c);
}
`;

/** Uniforms + varyings + `oSprite()` for particle vertex shaders. */
export const SPRITE_VERTEX_GLSL = /* glsl */ `
uniform float uTime;    // ambient clock (s)
uniform float uScale;   // drawing-buffer px per metre at distance 1
uniform float uMinPx;   // smallest sprite (px)
uniform float uMaxPx;   // largest sprite (px, below the GL point size limit)
uniform float uFocus;   // focus distance (m)
uniform float uCocNear; // defocus disc (px) of a particle at half the focus distance
uniform float uCocFar;  // same for far particles (much weaker)
varying vec3 vColor;
varying float vSharp;
varying float vHalo;

void oSprite(vec4 mv, float sizeWorld, vec3 col, float halo) {
  float z = max(0.05, -mv.z);
  float nominal = sizeWorld * uScale / z;
  float s0 = max(nominal, uMinPx);
  float dz = z - uFocus;
  float coc = (dz < 0.0 ? uCocNear : uCocFar) * abs(dz) / z;
  float S = clamp(sqrt(s0 * s0 + coc * coc), 1.0, uMaxPx);
  float sharp = clamp((s0 * s0) / (S * S), 0.0, 1.0);
  // integral of the sprite profile in px^2 (gaussian core / flat disc)
  float area = mix(0.6, 0.112, sharp) * S * S;
  float energy = 0.112 * nominal * nominal;
  gl_PointSize = S;
  vColor = col * (energy / max(area, 1e-4));
  vSharp = sharp;
  vHalo = halo;
}
`;

export const SPRITE_FRAGMENT = /* glsl */ `
varying vec3 vColor;
varying float vSharp;
varying float vHalo;
void main() {
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(c, c);
  if (r2 > 1.0) discard;
  float core = exp(-r2 * 7.0) + vHalo * exp(-r2 * 2.0);
  float disc = (1.0 - smoothstep(0.55, 1.0, r2)) * (0.72 + 0.4 * r2);
  float shape = mix(disc, core, vSharp);
  gl_FragColor = vec4(vColor * shape, 1.0);
  // no-ops in the linear HDR pass; applied only when rendering straight to the canvas (post=0)
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

/** Shared output tail for custom surface shaders (tone mapping only in post=0). */
export const OUTPUT_TAIL = /* glsl */ `
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
`;
