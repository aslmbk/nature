"""build_textures.py - tileable PBR texture sets for Silva (bark, rock, moss).

Run from the repo root (headless):
  "/c/Program Files/Blender Foundation/Blender 5.1/blender.exe" --background --factory-startup \
      --python assets-src/blender/build_textures.py [-- --only bark,rock,moss --no-render]

Outputs (public/nature/textures/):
  bark_basecolor/normal/orm.webp  2048^2   grain along V, 1 UV unit = 0.5 m
  rock_basecolor/normal/orm.webp  2048^2   isotropic, for triplanar use
  moss_basecolor/normal/orm.webp  1024^2   dense dark clumpy moss base
Base colour is sRGB; normal (OpenGL, +Y = +V = up in the image) and ORM
(R = AO, G = roughness, B = 0) are raw linear data.

Method: every map is derived from a synthetic height field built with numpy
from periodic spectral noise (FFT-filtered white noise) and periodic Worley
noise, so all maps tile exactly by construction. Normal = gradient of height,
AO = horizon-based cosine-weighted occlusion of the height, roughness and
colour are functions of height/cavity plus independent periodic noise.
Deterministic: fixed seeds, no time or platform dependence.

Previews (docs/captures/blender/): tex_<set>_tiles.png (2x2 contact sheet),
tex_<set>_seam.png (1:1 crop over the tile corner), tex_<set>_lit.png (EEVEE).
"""

import os
import sys
import time
import math

import numpy as np

sys.dont_write_bytecode = True          # keep __pycache__ out of assets-src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proclib as pl  # noqa: E402

import bpy  # noqa: E402

# WebP quality (100 = lossless in Blender's writer). Lossy WebP is 4:2:0, which
# caps normal-map PSNR around 29 dB at mip 0 whatever the quality; at mip 2
# (how bark/rock are usually sampled) the error is ~1-2 deg. Lossless normals
# would be ~6 MB each, so lossy it is; heights are lightly band-limited first.
QUALITY = {"basecolor": 90, "normal": 92, "orm": 90}


def log(*a):
    print("[textures]", *a, flush=True)


# ============================================================================
# BARK
# ============================================================================

def build_bark(n=2048, seed=4101):
    """Old weathered trunk: long fibrous ridges and deep grooves along V,
    strands that split and merge, slight waviness. Tile = 0.5 m x 0.5 m."""
    rng = np.random.default_rng(seed)
    tile_mm = 500.0
    px = tile_mm / n                    # ~0.24 mm per texel
    sm = pl.smoothstep

    # --- flow: gentle sway of the grain that also varies across U ----------
    w_sway = pl.spectral_noise(n, rng, pl.lowpass_filter(2.5, 1.8))
    w_wig = pl.spectral_noise(n, rng, pl.lowpass_filter(3.5, 6.0))
    du = (0.011 * n) * w_sway + (0.0022 * n) * w_wig
    w_crk = pl.spectral_noise(n, rng, pl.lowpass_filter(16.0, 5.0))

    def fibre(ku, kv, sharp=1.0, warp=None):
        f = pl.spectral_noise(n, rng, pl.fibre_filter(ku, kv, sharp))
        return pl.standardize(pl.warp_u(f, du if warp is None else warp))

    # --- major strands: zero-crossings of strongly anisotropic noise ---------
    # Groove width AND depth vary along the grain: where the strand between
    # two zero-crossings is weak (|F| < local width) the two grooves fuse into
    # one, where it grows they split around a new strand -> split/merge.
    du_crk = du + 0.0025 * n * w_crk
    F1 = pl.standardize(0.85 * fibre(7.5, 1.0, 0.8, du_crk) + 0.55 * fibre(14.0, 1.8, 1.0, du_crk))
    W1 = 0.13 + 0.30 * sm(-1.2, 1.7, fibre(5.0, 1.3))      # groove width (in F units)
    M1 = 0.30 + 0.70 * sm(-1.4, 0.8, fibre(3.0, 0.8))      # groove depth
    g1 = np.exp(-(F1 / W1) ** 2)
    g1_core = np.exp(-(F1 / (0.35 * W1)) ** 2)
    crown1 = 1.0 - np.exp(-(F1 / 1.1) ** 2)                # rounded strand cross-section

    # --- secondary strands, fibres, weathered grain --------------------------
    G2 = fibre(24.0, 2.6)
    W2 = 0.10 + 0.22 * sm(-1.0, 1.5, fibre(7.0, 1.6))
    M2 = sm(-0.5, 1.3, fibre(6.0, 1.4))
    G3 = fibre(66.0, 7.5)
    G4 = fibre(220.0, 34.0)
    grain = pl.spectral_noise(n, rng, pl.band_filter(420.0, 0.5))
    brk = sm(-0.9, 0.9, pl.spectral_noise(n, rng, pl.fibre_filter(36.0, 10.0)))   # broken fibres

    h = np.zeros((n, n), np.float32)
    h += 3.2 * crown1 - (7.5 * g1 + 4.0 * g1_core) * M1
    h += 0.8 * (1.0 - np.exp(-(G2 / 0.9) ** 2)) - 2.6 * np.exp(-(G2 / W2) ** 2) * M2
    h += (0.32 * (1.0 - np.exp(-(G3 / 0.6) ** 2)) - 0.38 * np.exp(-(G3 / 0.18) ** 2)) * (0.3 + 0.7 * brk)
    h += 0.11 * G4 + 0.05 * grain
    h += 1.3 * pl.spectral_noise(n, rng, pl.lowpass_filter(3.0, 2.0))
    h = h.astype(np.float32)
    crack = np.clip(g1 * M1, 0.0, 1.0)
    core = np.clip(np.exp(-(F1 / (0.5 * W1)) ** 2) * M1 + 0.6 * np.exp(-(G2 / (0.6 * W2)) ** 2) * M2, 0, 1)
    plate = pl.standardize(np.sign(F1) * fibre(6.0, 0.9) + 0.7 * fibre(11.0, 1.2))

    # --- derived maps --------------------------------------------------------
    nrm = pl.normal_from_height(pl.blur(h, 0.8), px, strength=0.8)
    ao = np.clip(pl.horizon_ao(h, px), 0.0, 1.0)
    t = pl.normalize01(h, 1.0, 99.7)
    cav = sm(0.05, 0.5, t)                          # 0 in cracks -> 1 on strands
    top = sm(0.6, 0.95, t)                          # worn strand tops

    # --- colour (sRGB authoring values; albedo, no lighting baked) ----------
    c_deep = pl.hexrgb("#211913")
    c_wall = pl.hexrgb("#5c4735")
    c_mid = pl.hexrgb("#80674b")
    c_ridge = pl.hexrgb("#9c8061")
    c_worn = pl.hexrgb("#b4997c")
    c_grey = pl.hexrgb("#958b7d")
    c_red = pl.hexrgb("#9a6d4f")

    col = pl.lerp(c_wall, c_mid, sm(0.12, 0.5, t))
    col = pl.lerp(col, c_ridge, sm(0.45, 0.85, t))
    tone = sm(-1.0, 1.0, plate)                                    # strand-to-strand tone
    col = pl.lerp(col, c_red, 0.38 * tone * cav)
    weather = sm(-0.2, 1.2, pl.spectral_noise(n, rng, pl.lowpass_filter(6.0, 2.5)))
    col = pl.lerp(col, c_grey, (0.15 + 0.40 * weather) * sm(0.35, 0.9, t) * (1.0 - tone))
    col = pl.lerp(col, c_worn, 0.45 * top * sm(-0.2, 1.4, G3) * brk)
    col = col * (1.0 + 0.06 * (plate * cav)[..., None])
    streak = 0.045 * G3 * brk + 0.035 * G4 + 0.035 * grain + 0.03 * fibre(30.0, 2.5)
    col = col * (1.0 + streak[..., None])
    # groove floors darker, deep cores near-black (crisp, not a painted shadow)
    col = pl.lerp(col, c_wall * 0.8, 0.45 * np.clip(crack, 0, 1))
    col = pl.lerp(col, c_deep, 0.9 * core)
    base = np.clip(col, 0.0, 1.0)

    rough = 0.935 - 0.16 * top - 0.03 * cav + 0.02 * G4 + 0.01 * grain
    rough = np.clip(rough, 0.70, 0.95)
    orm = np.stack([ao, rough, np.zeros_like(ao)], -1)
    stats = {"height_mm_p1_p99": np.percentile(h, [1, 99]).round(2).tolist(),
             "flow_slope_p99": float(np.percentile(np.abs(np.gradient(du, axis=0)), 99)),
             "ao_p1_p50": np.percentile(ao, [1, 50]).round(3).tolist()}
    return {"basecolor": base, "normal": pl.encode_normal(nrm), "orm": orm, "height": h, "stats": stats}


# ============================================================================
# ROCK
# ============================================================================

def build_rock(n=2048, seed=5203):
    """Matte, eroded, slightly pitted stone; isotropic (no grain) so it can be
    used triplanar. Neutral light grey-beige for tinting. Designed at
    1 tile ~ 1 m but has detail at every scale from ~10 cm down to ~1 mm."""
    rng = np.random.default_rng(seed)
    tile_mm = 1000.0
    px = tile_mm / n
    sm = pl.smoothstep

    wu = pl.spectral_noise(n, rng, pl.band_filter(5.0, 0.9))
    wv = pl.spectral_noise(n, rng, pl.band_filter(5.0, 0.9))
    WU = 0.012 * n * wu
    WV = 0.012 * n * wv

    def warped(filt):
        return pl.standardize(pl.warp_uv(pl.spectral_noise(n, rng, filt), WU, WV))

    broad = pl.spectral_noise(n, rng, pl.fbm_filter(3.4, kmin=5.0, kmax=200.0))   # gentle undulation
    lumps = pl.spectral_noise(n, rng, pl.fbm_filter(2.6, kmin=8.0, kmax=220.0))   # weathered lumps
    # weathering cups: overlapping shallow bowls whose rims meet in soft ridges
    KR = rng.uniform(0.75, 1.25, 27 * 27).astype(np.float32)
    KD = rng.uniform(0.3, 1.0, 27 * 27).astype(np.float32)
    cups, _ = pl.cell_union(n, rng, 27, lambda d, p: 1.0 - KD[p] * (1.0 - np.minimum(d / KR[p], 1.0) ** 2),
                            mode="min", reach=2, du=0.3 * WU, dv=0.3 * WV)
    cups = pl.blur(cups, 2.0)
    gran = pl.spectral_noise(n, rng, pl.fbm_filter(2.0, kmin=120.0, kmax=420.0))   # sandy grain
    grain = pl.spectral_noise(n, rng, pl.band_filter(650.0, 0.6))                   # colour only

    # pores / solution pits and a few shallow hollows (Worley, radius per point)
    R = rng.uniform(0.08, 0.38, 64 * 64).astype(np.float32)
    A = (rng.random(64 * 64) < 0.38).astype(np.float32) * (0.4 + 0.6 * R / 0.38)
    pits, _ = pl.cell_union(n, rng, 64, lambda d, p: np.clip(1.0 - (d / R[p]) ** 2, 0, 1) ** 1.2 * A[p],
                            mode="max", reach=1, du=0.4 * WU, dv=0.4 * WV)
    Rm = rng.uniform(0.2, 0.5, 11 * 11).astype(np.float32)
    Am = (rng.random(11 * 11) < 0.35).astype(np.float32)
    hollows, _ = pl.cell_union(n, rng, 11, lambda d, p: np.clip(1.0 - (d / Rm[p]) ** 2, 0, 1) ** 2.0 * Am[p],
                               mode="max", reach=1, du=WU, dv=WV)

    crk = warped(pl.band_filter(3.5, 0.5))
    crack_mask = sm(0.5, 1.3, pl.spectral_noise(n, rng, pl.band_filter(2.5, 0.6)))
    crack = pl.blur(np.exp(-(crk / 0.03) ** 2) * crack_mask, 1.0)

    h = np.zeros((n, n), np.float32)
    h += 3.5 * broad + 1.3 * lumps + 1.5 * cups + 0.10 * gran
    h -= 2.2 * hollows + 1.1 * pits + 1.5 * crack
    h = h.astype(np.float32)

    nrm = pl.normal_from_height(pl.blur(h, 1.1), px, strength=0.85)
    ao = np.clip(pl.horizon_ao(h, px, steps=(1, 2, 3, 5, 8, 12, 18, 27, 40, 60, 90)), 0, 1)
    curv = pl.curvature(h, 4.0)
    curv = curv / (np.percentile(np.abs(curv), 98) + 1e-6)
    exposed = sm(0.1, 0.9, curv)
    sheltered = sm(0.1, 0.9, -curv)

    c_base = pl.hexrgb("#978f7f")
    c_warm = pl.hexrgb("#9f917b")
    c_cool = pl.hexrgb("#8e8c84")
    c_light = pl.hexrgb("#a59e8f")
    c_dirt = pl.hexrgb("#70695b")
    tint = warped(pl.fbm_filter(2.4, kmin=3.0, kmax=40.0))
    mott = pl.spectral_noise(n, rng, pl.fbm_filter(1.6, kmin=8.0, kmax=160.0))
    col = pl.lerp(c_base, c_warm, 0.5 * sm(-0.3, 1.8, tint))
    col = pl.lerp(col, c_cool, 0.5 * sm(-0.3, 1.8, -tint))
    col = col * (1.0 + 0.035 * mott[..., None])
    col = pl.lerp(col, c_light, 0.30 * exposed)
    col = pl.lerp(col, c_dirt, np.clip(0.30 * sheltered + 0.7 * pits + 0.22 * hollows + 0.35 * crack, 0, 0.8))
    dark_spk = sm(1.8, 2.7, pl.spectral_noise(n, rng, pl.band_filter(380.0, 0.5)))
    lite_spk = sm(1.9, 2.8, pl.spectral_noise(n, rng, pl.band_filter(500.0, 0.5)))
    col = col * (1.0 + (0.045 * grain + 0.03 * gran)[..., None])
    col = col * (1.0 - 0.22 * dark_spk[..., None]) * (1.0 + 0.10 * lite_spk[..., None])
    base = np.clip(col, 0, 1)

    rough = 0.79 - 0.07 * exposed + 0.06 * sheltered + 0.07 * np.clip(pits + hollows + crack, 0, 1)
    rough += 0.02 * gran
    rough = np.clip(rough, 0.65, 0.90)
    orm = np.stack([ao, rough, np.zeros_like(ao)], -1)
    stats = {"height_mm_p1_p99": np.percentile(h, [1, 99]).round(2).tolist(),
             "ao_p1_p50": np.percentile(ao, [1, 50]).round(3).tolist()}
    return {"basecolor": base, "normal": pl.encode_normal(nrm), "orm": orm, "height": h, "stats": stats}


# ============================================================================
# MOSS
# ============================================================================

def build_moss(n=1024, seed=6311):
    """Dense dark-green clumpy moss carpet: cushions, clumps, tufts and fuzz.
    Designed at 1 tile ~ 0.3 m; no feature stands out, so 3-4 tiles do not
    show a grid."""
    rng = np.random.default_rng(seed)
    tile_mm = 300.0
    px = tile_mm / n
    sm = pl.smoothstep

    wu = pl.spectral_noise(n, rng, pl.band_filter(8.0, 0.8))
    wv = pl.spectral_noise(n, rng, pl.band_filter(8.0, 0.8))
    WU, WV = 0.016 * n * wu, 0.016 * n * wv

    def domes(cells, r_lo, r_hi, h_lo, warp=1.0):
        """Max-union of round domes (continuous; creases where domes meet)."""
        R = rng.uniform(r_lo, r_hi, cells * cells).astype(np.float32)
        H = rng.uniform(h_lo, 1.0, cells * cells).astype(np.float32)
        T = rng.standard_normal(cells * cells).astype(np.float32)
        reach = 2 if r_hi > 1.0 else 1
        d, cid = pl.cell_union(n, rng, cells, lambda dd, p: np.sqrt(np.clip(1.0 - (dd / R[p]) ** 2, 0, 1)) * H[p],
                               mode="max", reach=reach, du=warp * WU, dv=warp * WV)
        return d, T[cid]

    cush, cush_t = domes(12, 0.95, 1.3, 0.65)        # cushions ~2.5 cm
    clmp, clmp_t = domes(28, 0.9, 1.25, 0.55)        # clumps ~1 cm
    # individual shoots seen from above: small 5-6 armed rosettes (~2 mm)
    SR = rng.uniform(0.62, 0.92, 150 * 150).astype(np.float32)
    SH = rng.uniform(0.3, 1.0, 150 * 150).astype(np.float32)
    SK = rng.choice([5.0, 6.0], 150 * 150).astype(np.float32)
    SP = rng.uniform(0, 2 * np.pi, 150 * 150).astype(np.float32)

    def rosette(d, p, dx, dy):
        r_eff = SR[p] * (0.68 + 0.32 * np.cos(SK[p] * np.arctan2(dy, dx) + SP[p]))
        return np.sqrt(np.clip(1.0 - (d / r_eff) ** 2, 0, 1)) * SH[p]
    shoot, sid = pl.cell_union(n, rng, 150, rosette, mode="max", reach=1, with_offset=True)
    shoot_t = rng.standard_normal(150 * 150).astype(np.float32)[sid]
    micro = pl.spectral_noise(n, rng, pl.band_filter(300.0, 0.5))   # colour only

    macro = 5.2 * cush + 2.7 * clmp
    h = (macro + 0.7 * shoot).astype(np.float32)
    nrm = pl.normal_from_height(pl.blur(h, 0.9), px, strength=1.0)
    ao = np.clip(pl.horizon_ao(h, px, steps=(1, 2, 3, 5, 8, 12, 18, 26, 38)), 0, 1)
    t = pl.normalize01(macro, 1.0, 99.5)

    c_valley = pl.hexrgb("#2d3b13")
    c_body = pl.hexrgb("#36491a")
    c_top = pl.hexrgb("#435a1d")
    c_tip = pl.hexrgb("#6c862a")
    c_yel = pl.hexrgb("#56621c")
    c_blue = pl.hexrgb("#2f4924")
    c_brown = pl.hexrgb("#453f20")

    # macro height only nudges the colour: the relief lives in the normal map,
    # so cushions do not print a repeating light/dark layout into the albedo
    col = pl.lerp(c_valley, c_body, sm(0.0, 0.35, t))
    col = pl.lerp(col, c_top, 0.6 * sm(0.45, 1.0, t))
    hue = 0.35 * cush_t + 0.45 * clmp_t + 0.45 * shoot_t
    col = pl.lerp(col, c_yel, 0.30 * sm(0.2, 1.8, hue))
    col = pl.lerp(col, c_blue, 0.30 * sm(0.2, 1.8, -hue))
    dead = sm(1.7, 2.6, pl.spectral_noise(n, rng, pl.band_filter(40.0, 0.5)))
    col = pl.lerp(col, c_brown, 0.22 * dead)
    # stipple of shoots: lit rosette tips, dark gaps between them
    tipw = sm(0.5, 0.95, shoot) * sm(0.1, 0.6, t)
    col = pl.lerp(col, c_tip, 0.5 * tipw)
    col = col * (1.0 - 0.34 * (1.0 - sm(0.0, 0.3, shoot)))[..., None]
    col = col * (1.0 + 0.07 * micro[..., None])
    base = np.clip(col, 0, 1)

    rough = 0.94 + 0.05 * (1.0 - t) - 0.06 * tipw + 0.01 * micro
    rough = np.clip(rough, 0.85, 1.0)
    orm = np.stack([ao, rough, np.zeros_like(ao)], -1)
    stats = {"height_mm_p1_p99": np.percentile(h, [1, 99]).round(2).tolist(),
             "ao_p1_p50": np.percentile(ao, [1, 50]).round(3).tolist()}
    return {"basecolor": base, "normal": pl.encode_normal(nrm), "orm": orm, "height": h, "stats": stats}


# ============================================================================
# writing + checking
# ============================================================================

def write_set(name, maps, out_dir):
    """Write the three WebPs, read them back, report size and error, and
    return the DECODED maps (what the engine will actually sample)."""
    decoded = {}
    for key, cs in (("basecolor", "sRGB"), ("normal", "Non-Color"), ("orm", "Non-Color")):
        p = os.path.join(out_dir, "%s_%s.webp" % (name, key))
        pl.save_image(p, maps[key], "WEBP", QUALITY[key], cs)
        back = pl.load_image(p, cs)
        ref = np.round(np.clip(maps[key], 0, 1) * 255) / 255
        decoded[key] = back
        msg = "%-24s %7.1f KB  psnr %.1f dB (mip2 %.1f dB)" % (
            os.path.basename(p), os.path.getsize(p) / 1024.0, pl.psnr(back, ref),
            pl.psnr(pl.downsample(back, 4), pl.downsample(ref, 4)))
        if key == "normal":
            def unit(a):
                a = a * 2.0 - 1.0
                return a / np.linalg.norm(a, axis=-1, keepdims=True)
            ang = np.degrees(np.arccos(np.clip((unit(back) * unit(ref)).sum(-1), -1, 1)))
            msg += "  angular err mean %.2f deg" % ang.mean()
        log(msg)
    return decoded


def seam_report(img):
    """Percentile rank of the wrap-around step among all interior column
    (u) / row (v) steps. A seam would rank at ~100; tileable maps rank anywhere."""
    a = img.astype(np.float32)
    if a.ndim == 3:
        a = a.mean(-1)
    steps_u = np.abs(np.diff(a, axis=1)).mean(axis=0)
    steps_v = np.abs(np.diff(a, axis=0)).mean(axis=1)
    wrap_u = np.abs(a[:, 0] - a[:, -1]).mean()
    wrap_v = np.abs(a[0] - a[-1]).mean()
    return 100.0 * (steps_u < wrap_u).mean(), 100.0 * (steps_v < wrap_v).mean()


def lambert_preview(base, nrm_enc, ao, light=(-0.45, 0.55, 0.70)):
    nrm = nrm_enc * 2.0 - 1.0
    L = np.asarray(light, np.float32)
    L = L / np.linalg.norm(L)
    lam = np.clip((nrm * L).sum(-1), 0.0, 1.0)
    lin = pl.srgb_to_linear(base)
    shade = 0.9 * lam + 0.22 * ao
    return pl.linear_to_srgb(lin * shade[..., None] * 1.15)


def contact_sheet(name, maps, cap_dir):
    base, nrm, orm = maps["basecolor"], maps["normal"], maps["orm"]
    n = base.shape[0]
    f = max(1, (2 * n) // 1024)
    lit = lambert_preview(base, nrm, orm[..., 0])
    panels = [("%s basecolor 2x2" % name, base), ("%s normal 2x2" % name, nrm),
              ("%s orm 2x2 (r=ao g=rough)" % name, orm), ("%s lambert 2x2" % name, lit)]
    tiles = []
    for label, img in panels:
        t = pl.downsample(pl.tile2(img), f)[::-1].copy()      # top-down for layout
        t[:22, :] *= 0.35
        pl.draw_text(t, label, 6, 4, 2)
        tiles.append(t)
    sheet = np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0)
    pl.save_image(os.path.join(cap_dir, "tex_%s_tiles.png" % name), sheet[::-1], "PNG")
    # 1:1 crop centred on the corner where four tiles meet
    c = 384
    big_b = pl.tile2(base)
    big_l = pl.tile2(lit)
    crop_b = big_b[n - c:n + c, n - c:n + c]
    crop_l = big_l[n - c:n + c, n - c:n + c]
    seam = np.concatenate([crop_b[::-1], crop_l[::-1]], 1).copy()
    seam[:22, :] *= 0.35
    pl.draw_text(seam, "%s 1:1 at tile corner (centre)" % name, 6, 4, 2)
    pl.save_image(os.path.join(cap_dir, "tex_%s_seam.png" % name), seam[::-1], "PNG")


BUILDERS = {"bark": build_bark, "rock": build_rock, "moss": build_moss}


# ============================================================================
# lit previews (EEVEE, headless) - they read the delivered WebP files
# ============================================================================

def _reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _use_nodes(idblock):
    """use_nodes is deprecated in 5.x (always on) but still needed to build the
    default tree on some datablocks; set it quietly."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        idblock.use_nodes = True


def _render_setup(w, h, world_rgb, world_strength, samples=48):
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE"
    sc.eevee.taa_render_samples = samples
    sc.render.resolution_x = w
    sc.render.resolution_y = h
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "None"
    world = bpy.data.worlds.new("preview_world")
    sc.world = world
    _use_nodes(world)
    bg = world.node_tree.nodes.get("Background")
    bg.inputs[0].default_value = (*world_rgb, 1.0)
    bg.inputs[1].default_value = world_strength
    return sc


def _sun(name, rot_deg, energy, color=(1.0, 0.97, 0.93), angle_deg=3.0):
    ld = bpy.data.lights.new(name, "SUN")
    ld.energy = energy
    ld.color = color
    ld.angle = math.radians(angle_deg)
    ob = bpy.data.objects.new(name, ld)
    bpy.context.scene.collection.objects.link(ob)
    ob.rotation_euler = [math.radians(a) for a in rot_deg]
    return ob


def _camera(loc, target, fov_deg=35.0):
    from mathutils import Vector
    cd = bpy.data.cameras.new("cam")
    cd.sensor_fit = "VERTICAL"
    cd.angle_y = math.radians(fov_deg)
    ob = bpy.data.objects.new("cam", cd)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = loc
    ob.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = ob
    return ob


def _pbr_material(name, tex_dir, set_name, ao_mix=0.6):
    mat = bpy.data.materials.new(name)
    _use_nodes(mat)
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")

    def img(key, cs):
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image = bpy.data.images.load(os.path.join(tex_dir, "%s_%s.webp" % (set_name, key)))
        node.image.colorspace_settings.name = cs
        node.interpolation = "Cubic"
        return node

    bc, nm, orm = img("basecolor", "sRGB"), img("normal", "Non-Color"), img("orm", "Non-Color")
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(orm.outputs["Color"], sep.inputs["Color"])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    ins = {s.identifier: s for s in mix.inputs}
    outs = {s.identifier: s for s in mix.outputs}
    ins["Factor_Float"].default_value = ao_mix
    nt.links.new(bc.outputs["Color"], ins["A_Color"])
    nt.links.new(sep.outputs["Red"], ins["B_Color"])
    nt.links.new(outs["Result_Color"], bsdf.inputs["Base Color"])
    nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
    nmap = nt.nodes.new("ShaderNodeNormalMap")
    nmap.space = "TANGENT"
    nmap.uv_map = "UVMap"
    nt.links.new(nm.outputs["Color"], nmap.inputs["Color"])
    nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Metallic"].default_value = 0.0
    return mat


def _mesh_object(name, verts, faces, loop_uv_fn, mat):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    uv = me.uv_layers.new(name="UVMap")
    uvs = np.zeros((len(me.loops), 2), np.float32)
    lv = np.zeros(len(me.loops), np.int64)
    me.loops.foreach_get("vertex_index", lv)
    lp = np.zeros(len(me.loops), np.int64)
    for p in me.polygons:
        lp[p.loop_start:p.loop_start + p.loop_total] = p.index
    uvs[:] = loop_uv_fn(np.asarray(verts, np.float32)[lv], lv, lp)
    uv.data.foreach_set("uv", uvs.ravel())
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
    me.materials.append(mat)
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def _render_to(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return path


def _compose(paths, labels, out_path):
    imgs = [pl.load_image(p, "sRGB")[::-1].copy() for p in paths]
    for im, lab in zip(imgs, labels):
        im[:22, :] *= 0.35
        pl.draw_text(im, lab, 6, 4, 2)
    sheet = np.concatenate(imgs, 1)
    pl.save_image(out_path, sheet[::-1], "PNG")
    for p in paths:
        os.remove(p)


def preview_bark(tex_dir, cap_dir):
    _reset()
    _render_setup(900, 1000, (0.30, 0.34, 0.28), 0.55)
    mat = _pbr_material("bark_preview", tex_dir, "bark")
    r, hgt, seg, rings, tile = 0.16, 1.3, 240, 12, 0.5
    wraps = max(1, round(2 * math.pi * r / tile))          # integer wraps, ~square texels
    verts = []
    for j in range(rings + 1):
        for i in range(seg + 1):
            a = 2 * math.pi * i / seg + math.pi / 2          # i = 0 at the back (+Y)
            verts.append((r * math.cos(a), r * math.sin(a), hgt * j / rings))
    faces = [(j * (seg + 1) + i, j * (seg + 1) + i + 1, (j + 1) * (seg + 1) + i + 1, (j + 1) * (seg + 1) + i)
             for j in range(rings) for i in range(seg)]

    def uvf(co, lv, lp):
        i = lv % (seg + 1)
        return np.stack([wraps * i / seg, co[:, 2] / tile], -1)
    _mesh_object("bark_cylinder", verts, faces, uvf, mat)
    _sun("key", (52, 0, 128), 4.2)                           # raking from the right
    _sun("rim", (60, 0, -60), 1.2, (0.85, 0.92, 1.0))
    tmp = []
    _camera((0.0, -2.35, 0.65), (0.0, 0.0, 0.65))
    tmp.append(_render_to(os.path.join(cap_dir, "_bark_a.png")))
    _camera((0.0, -0.62, 0.78), (0.0, 0.0, 0.70))
    tmp.append(_render_to(os.path.join(cap_dir, "_bark_b.png")))
    _compose(tmp, ["bark on r=16cm cylinder, 2 wraps, v along length",
                   "close-up (texture u-wrap at centre line)"],
             os.path.join(cap_dir, "tex_bark_lit.png"))


def preview_rock(tex_dir, cap_dir):
    import bmesh
    _reset()
    _render_setup(900, 900, (0.10, 0.11, 0.12), 0.6)
    mat = _pbr_material("rock_preview", tex_dir, "rock")
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=6, radius=0.45)
    rng = np.random.default_rng(77)
    ks = rng.normal(size=(9, 3))
    ks = ks / np.linalg.norm(ks, axis=1, keepdims=True) * rng.uniform(2.0, 9.0, (9, 1))
    ph = rng.uniform(0, 2 * np.pi, 9)
    am = 0.10 / rng.uniform(1.0, 2.5, 9)
    for v in bm.verts:
        p = np.array(v.co)
        v.co = v.co * float(1.0 + np.sum(am * np.sin(ks @ p + ph)))
    verts = [tuple(v.co) for v in bm.verts]
    faces = [tuple(x.index for x in f.verts) for f in bm.faces]
    bm.free()
    vv = np.asarray(verts, np.float32)
    fn = []
    for f in faces:
        a, b, c = vv[f[0]], vv[f[1]], vv[f[2]]
        fn.append(np.cross(b - a, c - a))
    fn = np.asarray(fn)
    tile = 0.5

    def uvf(co, lv, lp):                                     # per-face box projection
        n = fn[lp]
        ax = np.argmax(np.abs(n), axis=1)
        s = np.sign(n[np.arange(len(n)), ax])
        u = np.where(ax == 0, co[:, 1] * s, np.where(ax == 1, -co[:, 0] * s, co[:, 0]))
        v = np.where(ax == 2, co[:, 1] * s, co[:, 2])
        return np.stack([u / tile, v / tile], -1)
    _mesh_object("rock_lumpy", verts, faces, uvf, mat)
    _sun("key", (48, 0, -38), 4.5, (1.0, 0.96, 0.9))
    _sun("fill", (70, 0, 140), 0.6, (0.8, 0.88, 1.0))
    tmp = []
    _camera((0.0, -1.85, 0.25), (0.0, 0.0, 0.0))
    tmp.append(_render_to(os.path.join(cap_dir, "_rock_a.png")))
    _camera((-0.12, -0.78, 0.22), (-0.05, 0.0, 0.08))
    tmp.append(_render_to(os.path.join(cap_dir, "_rock_b.png")))
    _compose(tmp, ["rock on lumpy sphere, box-projected uv, 1 tile = 0.5 m",
                   "rock close-up"], os.path.join(cap_dir, "tex_rock_lit.png"))


def preview_moss(tex_dir, cap_dir):
    _reset()
    _render_setup(1000, 800, (0.30, 0.34, 0.28), 0.55)
    mat = _pbr_material("moss_preview", tex_dir, "moss")
    res, size, tile = 220, 1.3, 0.3
    rng = np.random.default_rng(9)
    lumps = [(rng.uniform(-0.45, 0.45), rng.uniform(-0.45, 0.45), rng.uniform(0.08, 0.2), rng.uniform(0.03, 0.08))
             for _ in range(9)]
    verts = []
    for j in range(res + 1):
        for i in range(res + 1):
            x = size * (i / res - 0.5)
            y = size * (j / res - 0.5)
            rr = math.hypot(x, y)
            z = 0.30 * math.exp(-(rr / 0.40) ** 2)
            for lx, ly, lr, lh in lumps:
                z += lh * math.exp(-((x - lx) ** 2 + (y - ly) ** 2) / lr ** 2)
            z *= max(0.0, 1.0 - (rr / (0.5 * size)) ** 6)
            verts.append((x, y, z))
    faces = [(j * (res + 1) + i, j * (res + 1) + i + 1, (j + 1) * (res + 1) + i + 1, (j + 1) * (res + 1) + i)
             for j in range(res) for i in range(res)]

    def uvf(co, lv, lp):
        return np.stack([co[:, 0] / tile, co[:, 1] / tile], -1)
    _mesh_object("moss_mound", verts, faces, uvf, mat)
    _sun("key", (50, 0, 35), 4.0)
    _sun("rim", (65, 0, 200), 1.0, (0.9, 0.95, 1.0))
    tmp = []
    _camera((0.0, -1.75, 1.05), (0.0, 0.0, 0.08))
    tmp.append(_render_to(os.path.join(cap_dir, "_moss_a.png")))
    _camera((0.18, -0.62, 0.52), (0.05, -0.05, 0.2))
    tmp.append(_render_to(os.path.join(cap_dir, "_moss_b.png")))
    _compose(tmp, ["moss mound 1.3 m, planar uv, 1 tile = 0.3 m (4+ tiles)", "moss close-up"],
             os.path.join(cap_dir, "tex_moss_lit.png"))


PREVIEWS = {"bark": preview_bark, "rock": preview_rock, "moss": preview_moss}


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    only = None
    render = "--no-render" not in argv
    generate = "--preview-only" not in argv
    out_dir = pl.TEX_DIR
    cap_dir = pl.CAPTURE_DIR
    for i, a in enumerate(argv):
        if a == "--only":
            only = argv[i + 1].split(",")
        if a == "--out":
            out_dir = argv[i + 1]
        if a == "--cap":
            cap_dir = argv[i + 1]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cap_dir, exist_ok=True)
    names = [k for k in BUILDERS if not only or k in only]
    for name in names if generate else []:
        t0 = time.time()
        maps = BUILDERS[name]()
        log(name, "generated in %.1fs" % (time.time() - t0), maps.get("stats"))
        for key in ("basecolor", "normal", "orm"):
            su, sv = seam_report(maps[key])
            log("  wrap-step rank %-9s u %5.1f%%  v %5.1f%%  (a seam would be ~100%%)" % (key, su, sv))
        decoded = write_set(name, maps, out_dir)
        contact_sheet(name, decoded, cap_dir)
        mean = (maps["basecolor"].reshape(-1, 3).mean(0) * 255).round()
        log("  basecolor mean sRGB", mean.tolist())
        del maps, decoded
    if render:
        for name in names:
            t0 = time.time()
            PREVIEWS[name](out_dir, cap_dir)
            log("preview", name, "rendered in %.1fs" % (time.time() - t0))
    log("done")


if __name__ == "__main__":
    main()
