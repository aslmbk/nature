"""
build_canopy.py - the 'canopy' scene set (S08-S09, video 43.3-52.8 s): the twig skeleton of a compact
crown on black, exported to public/nature/models/canopy.glb (contract: CLAUDE.md "Asset contract").
The GLB carries no leaves: the engine instances kit.glb leaves / twigs on the twig meshes, steered by
their COLOR_0 (R leaf density, G leaf size, B baked volumetric AO).

Run headless (from the repo root):
    "/c/Program Files/Blender Foundation/Blender 5.1/blender.exe" --background --factory-startup \
        --python assets-src/blender/build_canopy.py -- --export --validate [--render [cams]] [--compare] [--save]

Run inside a live Blender session (e.g. through MCP):
    import sys, importlib; sys.path.insert(0, r"<repo>/assets-src/blender")
    import build_canopy; importlib.reload(build_canopy); build_canopy.main(["--save"])

Options:
    --export            write public/nature/models/canopy.glb
    --validate          re-read / re-import the GLB and check it (common.validate_glb + canopy checks);
                        report: docs/captures/blender/canopy_validate.json
    --render [cams]     previews WITH STAND-IN LEAVES (generated here for the previews only, never
                        exported) -> docs/captures/blender/canopy_<cam>.png
    --compare           comparison sheets vs frames 15 / 16 / 17 and details/08 ->
                        docs/captures/blender/canopy_cmp_*.png (+ the reveal order, the bare skeleton)
    --percent N         preview size in % of 1440x1020 (default 50)
    --save              save nature.blend (headless: opens the existing file first so the other scene
                        sets are kept; only the 'canopy' scene / collection is rebuilt)
    --no-ao             skip the volumetric AO (B = 1; fast iteration)

Deterministic (seed 134). World units: metres, Blender Z-up, the cameras look along +Y.

The crown (frames 15-17, details/08, motion/03): roughly circular on black, ~80 % of the frame width at
46.5 s, ragged outline, lumpy lit masses with dark gaps, a dark middle in the close-up (frame 17).
The leaf-bearing twig ends lie on a crown surface defined in screen terms of cam_canopy_main: the
outline measured on frame 16 (CROWN_R16), a dome bulging B_DOME towards the camera, lobes of +-10 cm
(LOBE_*) and a hollow just below the centre (RECESS: set back, fewer leaves). 46 unequal branching
clusters (3 levels of twigs, 2-8 mm thick) whose main stems point back to a hidden centre H behind the
crown: 18 on a ring along the outline, 22 spread over the disc (best-candidate sampling), 6 of those
pushed towards the camera (they lean to the camera). Each main stem ends on the surface; 4-9 side twigs
fan from its outer half to surface points around it, 3-6 twiglets per side twig. A scaffold of 7
limbs + connectors joins every cluster base to H (no leaves).
Each cluster is its own node wood_canopy_cNN with the origin at its base, so the engine can scale it
in from there; node extras: reveal_order (0 first ... 1 last: inner / lower first), depth (0 nearest the
camera ... 1 farthest), radius_m (farthest twig point from the origin) and a few helpers.
"""

import importlib
import json
import math
import os
import sys
import tempfile
import time

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import common as C  # noqa: E402
import canopylib as L  # noqa: E402

importlib.reload(C)
importlib.reload(L)

ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
GLB_PATH = os.path.join(ROOT, "public", "nature", "models", "canopy.glb")
BLEND_PATH = os.path.join(HERE, "nature.blend")
CAPTURES = os.path.join(ROOT, "docs", "captures", "blender")
REF = os.path.join(ROOT, "docs", "nature-webgl-reference")      # comparison sheets only (never shipped)
TMP = os.path.join(tempfile.gettempdir(), "silva_canopy_build")
SCENE_NAME = "canopy"
COLL_NAME = "canopy"
SEED = 134
MAX_TRIS = 250_000
MAX_BYTES = 8 * 1024 * 1024

# ------------------------------------------------------------------------------------------
# crown frame: O = centre of the crown's silhouette (world origin), the cameras look along +Y,
# H = hidden centre the clusters radiate from (behind the silhouette plane, a little low).
# ------------------------------------------------------------------------------------------
O = np.array([0.0, 0.0, 0.0])
H = np.array([0.0, 0.95, -0.35])

# frame 16 (46.5 s): crown outline radius (frame heights) around (0.505, 0.54) every 10 deg
# (0 = right, 90 = up), measured on the hole-filled leaf mask (the crown runs out of the frame at the
# bottom, 250-270 deg: lower bounds). Frame 15 (44.3 s) for reference: 0.27-0.42, mean 0.36.
CROWN_R16 = [0.58, 0.512, 0.552, 0.564, 0.54, 0.512, 0.468, 0.504, 0.52, 0.512, 0.52, 0.52, 0.504, 0.52,
             0.488, 0.536, 0.544, 0.444, 0.416, 0.46, 0.408, 0.456, 0.52, 0.496, 0.46, 0.488, 0.47, 0.47,
             0.4, 0.46, 0.452, 0.48, 0.532, 0.512, 0.504, 0.508]
CROWN_UV16 = (0.505, 0.54)
LEAF_MARGIN = 0.03        # frame heights the leaves / kit twigs reach beyond the skeleton tips (~9 cm)

# cameras (pitch / yaw / roll in degrees, common.cam_quat). The crown centre O is framed at (u, v)
# at view depth d (the focus extras are set later from the leafy twigs). in: frame 15 (44.3 s, the
# crown fully grown, ~0.6 of the frame width; before that it grows from bottom-centre, motion/03,
# reveal_order); main: frame 16; out: start of the move closer (48-49.5 s); close: frame 17, the crown
# fills the frame; close_out: closer still (the leaves part towards the frame edges in the finale
# transition).
CAMS = {
    "cam_canopy_in": dict(uv=(0.505, 0.535), d=5.55, pitch=0.0, yaw=0.0, roll=0.0, t=44.3),
    "cam_canopy_main": dict(uv=(0.51, 0.53), d=4.85, pitch=0.0, yaw=0.0, roll=0.0, t=46.5),
    "cam_canopy_out": dict(uv=(0.51, 0.53), d=4.55, pitch=0.0, yaw=0.0, roll=0.0, t=48.5),
    "cam_canopyClose_main": dict(uv=(0.49, 0.55), d=3.1, pitch=-2.0, yaw=0.0, roll=0.0, t=50.5),
    "cam_canopyClose_out": dict(uv=(0.49, 0.55), d=2.8, pitch=-2.0, yaw=0.0, roll=0.0, t=52.8),
}
# key: from above-front, slightly left (direction the light travels, Blender world)
KEY_DIR = (0.26, 0.52, -0.81)

# crown surface, in screen terms of cam_canopy_main: outline radius r_out(theta) = r16 - LEAF_MARGIN
# (frame heights); the front of the dome bulges B_DOME towards the camera from the silhouette plane
# y = 0, with a hollow just below the centre: set back and with fewer leaves (the dark middle of the
# close-up, frame 17).
B_DOME = 0.62
RECESS = dict(uv=(0.505, 0.56), r=0.22, depth=0.3, leaves=0.7)
RHO_MAX = 1.0             # twig targets beyond the outline are pulled back (protrusions keep 60 % of the excess)
N_RIM, N_INNER = 18, 28   # 46 clusters
N_FRONT = 6               # inner clusters pushed towards the camera (they lean to the camera)
LOBE_AMP, LOBE_FREQ = 0.1, 7.0   # lumps of the crown surface: +-10 cm, ~0.14 frame heights across
NL = C.Noise3(SEED + 3)

SIDES = (6, 5, 4)                 # tube sides per level (main stem, twigs, twiglets)
RING_SPACING = (0.022, 0.018, 0.016)
R_MIN = 0.001                     # 2 mm thick at the finest
R_BASE = (0.0034, 0.0040)         # cluster main stem radius at its base (6.8-8 mm)

# assumed engine leaf load for the volumetric AO: ~60k leaves of ~2.2 cm^2 (kit canopy leaves)
LEAF_AREA_TOTAL = 13.0            # m^2
PREVIEW_LEAVES = 140000


def cam(name):
    c = CAMS[name]
    loc = L.cam_loc_for(O, c["uv"][0], c["uv"][1], c["d"], c["pitch"], c["yaw"], c["roll"])
    return L.Cam(name, loc, c["pitch"], c["yaw"], c["roll"], focus=c["d"], t=c["t"])


def main_cam():
    return cam("cam_canopy_main")


def r16(theta_deg):
    """Outline radius (frame heights) of frame 16 at screen angle theta (deg, 0 = right, 90 = up)."""
    a = np.asarray(theta_deg, float) % 360.0
    xs = np.arange(0, 370, 10)
    ys = np.array(CROWN_R16 + [CROWN_R16[0]])
    return np.interp(a, xs, ys)


# ------------------------------------------------------------------------------------------
# layout
# ------------------------------------------------------------------------------------------

def r_out(theta_deg):
    return r16(theta_deg) - LEAF_MARGIN


def to_polar(u, v):
    """Screen (u, v) of the main camera -> (theta deg, rho = fraction of the outline radius)."""
    x = (np.asarray(u, float) - CROWN_UV16[0]) * C.ASPECT
    y = -(np.asarray(v, float) - CROWN_UV16[1])
    th = np.degrees(np.arctan2(y, x)) % 360.0
    return th, np.hypot(x, y) / r_out(th)


def from_polar(th, rho):
    rr = rho * r_out(th)
    return (CROWN_UV16[0] + rr * np.cos(np.radians(th)) / C.ASPECT,
            CROWN_UV16[1] - rr * np.sin(np.radians(th)))


def dome_y(u, v):
    """Y of the crown surface seen at (u, v) by the main camera (y = 0: silhouette plane)."""
    th, rho = to_polar(u, v)
    y = -B_DOME * np.sqrt(np.clip(1.0 - np.minimum(rho, 1.0) ** 2, 0.0, 1.0))
    return y + RECESS["depth"] * hollow(u, v)


def hollow(u, v):
    """0..1 weight of the central hollow at screen (u, v) of the main camera."""
    dist = np.hypot((np.asarray(u, float) - RECESS["uv"][0]) * C.ASPECT, np.asarray(v, float) - RECESS["uv"][1])
    return np.exp(-(dist / RECESS["r"]) ** 2)


def lobe(u, v):
    """Large lumps of the crown surface (about -1..1) in screen space of the main camera."""
    x = (np.asarray(u, float) - CROWN_UV16[0]) * C.ASPECT * LOBE_FREQ
    y = -(np.asarray(v, float) - CROWN_UV16[1]) * LOBE_FREQ
    return NL.fbm(np.stack([x, y, np.full_like(x, 0.37)], -1), 3) * 1.6


def surf(u, v, dy=0.0, mc=None):
    """World point on the crown surface seen at (u, v) by the main camera, pushed back by dy (m);
    the lobes bulge towards the camera."""
    mc = mc or main_cam()
    fwd = mc.basis()[2]
    depth = (dome_y(u, v) - LOBE_AMP * np.clip(lobe(u, v), -1.0, 1.0) + dy - mc.loc[1]) / fwd[1]
    return mc.S(float(u), float(v), float(depth))


def fan_target(u0, v0, ang, dist_m, dy, mc):
    """Surface point dist_m (metres at the local depth) from screen point (u0, v0) in screen direction
    ang (rad, 0 = right, + = up); points beyond RHO_MAX are pulled back towards the outline."""
    depth = float(dome_y(u0, v0)) - mc.loc[1]
    fh = 2.0 * L.TV * depth
    u = u0 + dist_m / fh * math.cos(ang) / C.ASPECT
    v = v0 - dist_m / fh * math.sin(ang)
    th, rho = to_polar(u, v)
    if rho > RHO_MAX:
        u, v = from_polar(th, RHO_MAX + 0.6 * (rho - RHO_MAX))
    return surf(u, v, dy, mc), (float(u), float(v))


def layout(rng):
    """Cluster centres in screen space of the main camera: N_RIM on a ring just inside the outline,
    N_INNER spread over the disc by best-candidate sampling (N_FRONT of them pushed towards the
    camera). Each gets its surface point T (main stem tip), its base on the line H -> T and a fan
    radius (metres on the crown surface) its twigs spread over."""
    mc = main_cam()
    pts, kinds = [], []
    off = rng.uniform(0, 360.0 / N_RIM)
    for k in range(N_RIM):
        th = off + (k + rng.uniform(-0.3, 0.3)) * 360.0 / N_RIM
        u, v = from_polar(th, rng.uniform(0.86, 1.0))
        pts.append(((u - CROWN_UV16[0]) * C.ASPECT, -(v - CROWN_UV16[1])))
        kinds.append("rim")
    for k in range(N_INNER):
        best, bd = None, -1.0
        for _ in range(24):
            u, v = from_polar(rng.uniform(0, 360.0), 0.82 * math.sqrt(rng.uniform(0.0, 1.0)))
            x, y = (u - CROWN_UV16[0]) * C.ASPECT, -(v - CROWN_UV16[1])
            d = min(math.hypot(x - a, y - b) for a, b in pts)
            if d > bd:
                best, bd = (x, y), d
        pts.append(best)
        kinds.append("inner")
    inner = [i for i in range(len(pts)) if kinds[i] == "inner"]
    rho_in = {i: float(to_polar(CROWN_UV16[0] + pts[i][0] / C.ASPECT, CROWN_UV16[1] - pts[i][1])[1]) for i in inner}
    cand = [i for i in inner if 0.08 < rho_in[i] < 0.72 and
            float(hollow(CROWN_UV16[0] + pts[i][0] / C.ASPECT, CROWN_UV16[1] - pts[i][1])) < 0.45]
    for i in rng.choice(cand, size=min(N_FRONT, len(cand)), replace=False):
        kinds[int(i)] = "front"
    clusters = []
    for i, (x, y) in enumerate(pts):
        kind = kinds[i]
        u0, v0 = CROWN_UV16[0] + x / C.ASPECT, CROWN_UV16[1] - y
        if kind == "rim":
            dy, l1, fan = rng.uniform(-0.06, 0.10), rng.uniform(0.62, 0.85), rng.uniform(0.27, 0.36)
        elif kind == "front":
            dy, l1, fan = rng.uniform(-0.32, -0.16), rng.uniform(0.50, 0.70), rng.uniform(0.18, 0.26)
        else:
            dy, l1, fan = rng.uniform(-0.05, 0.05), rng.uniform(0.48, 0.68), rng.uniform(0.22, 0.32)
        tip = surf(u0, v0, dy, mc)
        base = tip + C.normalize(H - tip) * l1          # main stem of length l1 pointing back to H
        th, rho = to_polar(u0, v0)
        clusters.append(dict(kind=kind, uv=(float(u0), float(v0)), theta=float(th), rho=float(rho), tip=tip,
                             base=base, fan=fan))
    return clusters


def rotate_toward(v, w, ang):
    """Rotate unit vector v towards w by ang radians (in their plane)."""
    v = C.normalize(np.asarray(v, float))
    w = np.asarray(w, float)
    p = C.normalize(w - (w @ v) * v)
    return C.normalize(v * math.cos(ang) + p * math.sin(ang))


def grow_cluster(rng, cl, mc):
    """Three levels: the main stem from the base up to the surface point T (leaning to the camera for
    the inner clusters), 4-9 side twigs from its outer half to points of the crown surface around T
    (a fan over the surface), 2-5 twiglets on each, alternating sides, ending a little in front of
    the surface."""
    base, T = cl["base"], cl["tip"]
    u0, v0 = cl["uv"]
    L1 = float(np.linalg.norm(T - base))
    d0 = rotate_toward(C.normalize(T - base), rng.normal(size=3), math.radians(rng.uniform(2, 7)))
    X1 = L.grow_path(rng, base, d0, L1 * rng.uniform(1.0, 1.04), 7, 0.08,
                     lambda p, d, f: 0.3 * C.normalize(T - p))
    tw1 = L.Twig(X1, np.zeros(len(X1)), 0)
    r0 = rng.uniform(*R_BASE)
    tw1.r = np.maximum(r0 * (1.0 - 0.62 * tw1.s / tw1.L), R_MIN * 1.1)
    twigs = [tw1]
    n2 = int(np.clip(round(cl["fan"] / 0.055 * rng.uniform(0.85, 1.15)), 4, 9))
    a0 = rng.uniform(0, 2 * math.pi)
    for j in range(n2):
        ang = a0 + (j + rng.uniform(-0.3, 0.3)) * 2 * math.pi / n2
        tgt, _ = fan_target(u0, v0, ang, cl["fan"] * rng.uniform(0.55, 1.0), rng.normal() * 0.04, mc)
        s = rng.uniform(0.5, 0.92) * tw1.L
        p, t, r = tw1.at(s)
        d = rotate_toward(C.normalize(tgt - p), t, math.radians(rng.uniform(5, 15)))
        L2 = float(np.linalg.norm(tgt - p))
        X2 = L.grow_path(rng, p, d, L2 * 1.03, 5, 0.09, lambda pp, dd, f, g=tgt: 0.35 * C.normalize(g - pp))
        tw2 = L.Twig(X2, np.zeros(len(X2)), 1, parent=0, s_parent=s, base_dist=s)
        r2 = max(0.62 * r, R_MIN * 1.2)
        tw2.r = np.maximum(r2 * (1.0 - 0.5 * tw2.s / tw2.L), R_MIN)
        twigs.append(tw2)
        i2 = len(twigs) - 1
        n3 = int(np.clip(round(tw2.L / 0.05 * rng.uniform(0.8, 1.2)), 3, 6))
        side = 1.0 if rng.random() < 0.5 else -1.0
        for k in range(n3):
            s3 = (0.42 + 0.52 * (k + rng.uniform(0.2, 0.8)) / n3) * tw2.L
            p3, t3, r3 = tw2.at(s3)
            u3, v3, _ = mc.project(p3)
            side = -side
            ang3 = ang + side * rng.uniform(0.35, 1.2)
            tgt3, _ = fan_target(float(u3), float(v3), ang3, tw2.L * rng.uniform(0.22, 0.42),
                                 -abs(rng.normal()) * 0.035, mc)
            d3 = rotate_toward(C.normalize(tgt3 - p3), t3, math.radians(rng.uniform(5, 20)))
            L3 = float(np.linalg.norm(tgt3 - p3))
            X3 = L.grow_path(rng, p3, d3, L3, 3, 0.12, lambda pp, dd, f, g=tgt3: 0.3 * C.normalize(g - pp))
            tw3 = L.Twig(X3, np.zeros(len(X3)), 2, parent=i2, s_parent=s3, base_dist=s + s3)
            tw3.r = np.full(len(X3), max(min(0.62 * r3, 0.0013), R_MIN))
            tw3.r = np.maximum(tw3.r * (1.0 - 0.15 * tw3.s / tw3.L), R_MIN)
            twigs.append(tw3)
    return twigs


def scaffold(rng, clusters):
    """Limbs from H to 7 fork points (k-means of the base directions), connectors to every base, and a
    trunk stub leaving backwards / down out of view."""
    bases = np.array([c["base"] for c in clusters])
    dirs = C.normalize(bases - H)
    k = 7
    # deterministic farthest-point init + a few Lloyd steps on the unit sphere
    cent = [dirs[int(np.argmax(dirs[:, 2]))]]
    for _ in range(k - 1):
        d2 = np.min([1.0 - dirs @ c for c in cent], axis=0)
        cent.append(dirs[int(np.argmax(d2))])
    cent = np.array(cent)
    for _ in range(12):
        lab = np.argmax(dirs @ cent.T, axis=1)
        cent = np.array([C.normalize(dirs[lab == g].mean(0)) if np.any(lab == g) else cent[g] for g in range(k)])
    lab = np.argmax(dirs @ cent.T, axis=1)
    twigs, forks = [], []
    zup = np.array([0.0, 0.0, 1.0])
    for g in range(k):
        mem = np.nonzero(lab == g)[0]
        if not len(mem):
            forks.append(H.copy())
            continue
        mb = bases[mem].mean(0)
        F = H + (mb - H) * 0.58
        forks.append(F)
        X = L.grow_path(rng, H, C.normalize(F - H), float(np.linalg.norm(F - H)), 4, 0.05,
                        lambda p, d, f, F=F: 0.35 * C.normalize(F - p))
        tw = L.Twig(X, np.zeros(len(X)), 0)
        tw.r = 0.013 - 0.004 * tw.s / tw.L
        twigs.append(tw)
        for i in mem:
            B = bases[i]
            X = L.grow_path(rng, F, C.normalize(B - F), float(np.linalg.norm(B - F)), 4, 0.06,
                            lambda p, d, f, B=B: 0.4 * C.normalize(B - p) + 0.02 * zup)
            X[-1] = B                                   # end exactly at the cluster base
            tw = L.Twig(X, np.zeros(len(X)), 1)
            rb = clusters[i]["r0"] + 0.0006
            tw.r = 0.0085 + (rb - 0.0085) * tw.s / tw.L
            twigs.append(tw)
    trunk_end = H + np.array([0.05, 0.75, -0.85])
    X = L.grow_path(rng, H, C.normalize(trunk_end - H), float(np.linalg.norm(trunk_end - H)), 4, 0.04,
                    lambda p, d, f: 0.3 * C.normalize(trunk_end - p))
    tw = L.Twig(X, np.zeros(len(X)), 0)
    tw.r = 0.024 + 0.012 * tw.s / tw.L
    twigs.append(tw)
    return twigs, lab, np.array(forks)


# ------------------------------------------------------------------------------------------
# COLOR_0
# ------------------------------------------------------------------------------------------

LEVEL_R = np.array([0.55, 0.8, 1.0])      # max leaf density per level (thick stems carry fewer leaves per area)


def ramp_twig(level, f):
    """Leaf density along one twig (f = fraction of its length): main stems only on their outer 30 %,
    side twigs and twiglets rising from 40 % to 80 % of their length, full to the tip."""
    return np.where(np.asarray(level) == 0, C.smoothstep(0.55, 0.85, f), C.smoothstep(0.40, 0.80, f))


def ramp_cluster(path, path_max):
    """No leaves on the inner part of a cluster (path length from the cluster base)."""
    return C.smoothstep(0.16, 0.48, np.asarray(path, float) / path_max)


def leaf_patch(P, N3, cl_index, mc):
    """Leaf density modulation: denser on the lobes of the crown surface, sparser in the creases
    between them, plus a small per-cluster variation."""
    u, v, _ = mc.project(P)
    lob = C.smoothstep(-0.35, 0.25, lobe(u, v))
    small = C.smoothstep(-0.35, 0.35, N3.fbm(np.asarray(P) * 2.6 + 3.7 * cl_index, 3))
    return (0.45 + 0.55 * lob) * (0.75 + 0.25 * small) * (1.0 - RECESS["leaves"] * hollow(u, v))


def leaf_density(twigs, info, cl_rich, N3, cl_index, mc):
    """R: 0 at the cluster base, rising to 1 along the last 60 % of every twig (and only in the outer
    part of the cluster); lower on the thick stems (the engine scatters by area); uneven patches."""
    tw_level = info["level"]
    s = info["s"]
    Ltw = info["L"]
    ti = info["twig"]
    base_dist = np.array([twigs[k].base_dist for k in range(len(twigs))])[ti]
    path_max = max(tw.base_dist + tw.L for tw in twigs)
    f_tw = s / np.maximum(Ltw, 1e-6)
    ramp_tw = ramp_twig(tw_level, f_tw)
    ramp_cl = ramp_cluster(base_dist + s, path_max)
    lvl = LEVEL_R[np.clip(tw_level, 0, 2)]
    P = info["world"]
    R = ramp_tw * ramp_cl * lvl * leaf_patch(P, N3, cl_index, mc) * cl_rich
    G = (1.0 - 0.45 * C.smoothstep(0.55, 1.0, f_tw)) * np.array([1.0, 0.95, 0.88])[np.clip(tw_level, 0, 2)]
    return np.clip(R, 0.0, 1.0), G


def density_grid(samples_P, samples_w):
    lo = samples_P.min(0) - 0.4
    hi = samples_P.max(0) + 0.4
    g = L.DensityGrid(lo, hi, 0.05)
    w = np.asarray(samples_w, float)
    w = w / max(w.sum(), 1e-9) * LEAF_AREA_TOTAL / g.voxel ** 3      # leaf area per m^3
    g.splat(samples_P, w)
    g.blur(2)
    return g


def volumetric_ao(grid, P, use_ao=True):
    if not use_ao:
        return np.ones(len(P))
    out_dir = C.normalize(np.asarray(P, float) - H)
    # extinction of randomly oriented leaves: 0.5 x leaf area density
    return grid.occlusion(P, out_dir, sigma=0.5, length=1.6, n_dirs=10, steps=12, start=0.03)


# ------------------------------------------------------------------------------------------
# build
# ------------------------------------------------------------------------------------------

def build_cameras(coll):
    out = {}
    for name in CAMS:
        c = cam(name)
        extras = dict(focus_distance_m=round(c.focus, 3), frame_height_at_focus_m=round(c.frame_height(), 3),
                      video_time_s=c.t)
        out[name] = C.make_camera(name, coll, tuple(c.loc), c.q, focus=c.focus, extras=extras)
    return out


def set_focus(coll, clusters):
    """Focus of every camera = median view depth of the leafy twig points (R > 0.3) inside its frame;
    the frame height at that depth goes into the extras as well."""
    P = np.vstack([cl["info"]["world"][cl["R"] > 0.3] for cl in clusters])
    out = {}
    for name in CAMS:
        c = cam(name)
        u, v, z = c.project(P)
        sel = (u > 0) & (u < 1) & (v > 0) & (v < 1) & (z > 0.1)
        f = round(float(np.median(z[sel])), 2)
        o = coll.objects[name]
        o.data.dof.focus_distance = f
        o["focus_distance_m"] = f
        o["frame_height_at_focus_m"] = round(c.frame_height(f), 3)
        o["crown_centre_distance_m"] = round(c.focus, 3)
        out[name] = f
    return out


def build_keys(coll):
    d = C.normalize(np.asarray(KEY_DIR, float))
    C.make_key_empty("key_canopy", coll, tuple(O - d * 3.0), tuple(d),
                     extras=dict(note="from above-front, slightly left"))


def far_inner(coll, mat, N3):
    """Dark lumpy cap behind the cluster bases: the backdrop of the gaps inside the crown."""
    c = H + np.array([0.0, 0.55, 0.0])
    sdf = C.SDF(c - np.array([1.5, 0.7, 1.4]), c + np.array([1.5, 0.7, 1.4]), 0.04)
    sdf.ellipsoid(c, (1.12, 0.36, 1.02), None, k=0.0)
    sdf.ellipsoid(c + np.array([0.35, -0.1, 0.4]), (0.55, 0.3, 0.5), None, k=0.2)
    sdf.ellipsoid(c + np.array([-0.45, -0.08, -0.3]), (0.6, 0.3, 0.55), None, k=0.2)
    sdf.noise(N3, 0.12, 1.6, octaves=3, offset=(2.2, 9.1, 4.4))
    return sdf.to_object("far_canopy_inner", coll, mat, target_tris=1600, smooth_iter=2)


def far_outer(coll, mat, N3):
    """Dim surrounding foliage ~2.4 m behind the crown (frames 15 / 16): ragged dark patches on a
    shallow dish around the crown (annulus 1.3-3.9 m), meshed low-poly; COLOR_0.R = sparse dim leaves."""
    yc = H[1] + 2.4
    sdf = C.SDF((-4.3, yc - 0.6, -3.7), (4.3, yc + 1.0, 3.7), 0.06)
    sl, P = sdf._box(sdf.origin, sdf.origin + np.array(sdf.shape) * sdf.voxel)
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    zz = (z + 0.1) / 0.85
    r = np.hypot(x, zz)
    dish = yc + 0.10 * (r - 2.5) + 0.3 * N3.fbm(np.stack([x * 0.7, np.full_like(x, 3.3), z * 0.7], -1), 2)
    n = N3.fbm(np.stack([x * 1.05 + 17.0, np.full_like(x, 41.0), z * 1.2 - 5.0], -1), 3)
    thr = -0.02 + 0.22 * C.smoothstep(2.9, 3.9, r)
    d = np.maximum.reduce([np.abs(y - dish) - 0.05, (thr - n) * 0.6, 1.3 - r, r - 3.9])
    sdf.d[sl] = d.astype(np.float32)
    return sdf.to_object("far_canopy_outer", coll, mat, target_tris=2400, smooth_iter=2)


def build(use_ao=True):
    t0 = time.time()
    scene = C.ensure_scene(SCENE_NAME)
    coll = C.ensure_collection(scene, COLL_NAME)
    C.purge_collection(coll)
    mat_wood = C.preview_material("mat_wood", C.PALETTE["wood"], 0.85)
    mat_far = C.preview_material("mat_far", "#141A18", 1.0)
    N3 = C.Noise3(SEED)
    rng = np.random.default_rng(SEED)
    build_cameras(coll)
    build_keys(coll)
    mc = main_cam()

    clusters = layout(rng)
    all_twigs = []
    for i, cl in enumerate(clusters):
        crng = np.random.default_rng(SEED * 1000 + i)
        cl["twigs"] = grow_cluster(crng, cl, mc)
        cl["r0"] = float(cl["twigs"][0].r[0])
        cl["rich"] = crng.uniform(0.85, 1.0)
        all_twigs.extend(cl["twigs"])
    sc_twigs, groups, forks = scaffold(np.random.default_rng(SEED + 5), clusters)

    # ---- meshes
    objs = []
    for i, cl in enumerate(clusters):
        name = f"wood_canopy_c{i:02d}"
        V, q, tr, uv, info = L.tube_mesh(cl["twigs"], SIDES, RING_SPACING, cl["base"])
        info["world"] = info["centre"] + cl["base"]
        obj = L.mesh_object(name, coll, V, q, tr, uv, mat_wood, location=cl["base"])
        cl["obj"], cl["info"] = obj, info
        objs.append(obj)
    V, q, tr, uv, sinfo = L.tube_mesh(sc_twigs, (8, 6), (0.03, 0.025), O, base_cap_levels=())
    sinfo["world"] = sinfo["centre"] + O
    scaf = L.mesh_object("wood_canopy_scaffold", coll, V, q, tr, uv, mat_wood, location=O)
    fi = far_inner(coll, mat_far, N3)
    fo = far_outer(coll, mat_far, N3)
    for o in (fi, fo):
        C.box_uvs(o.data, 1.0)
    C.log(f"geometry: {len(clusters)} clusters, {len(all_twigs)} twigs ({time.time() - t0:.1f}s)")

    # ---- COLOR_0: R / G first (they define the leaf volume), then the volumetric AO
    t = time.time()
    samp_P, samp_w = [], []
    leafy_m = 0.0                       # integral of R along all twigs (leaf-bearing length, m)
    for i, cl in enumerate(clusters):
        R, G = leaf_density(cl["twigs"], cl["info"], cl["rich"], N3, i, mc)
        cl["R"], cl["G"] = R, G
        path_max = max(x.base_dist + x.L for x in cl["twigs"])
        for tw in cl["twigs"]:
            Xs = L.resample_polyline(tw.X, 0.01)
            ss = np.linspace(0, tw.L, len(Xs))
            w = (ramp_twig(tw.level, ss / max(tw.L, 1e-6)) * ramp_cluster(tw.base_dist + ss, path_max) *
                 LEVEL_R[min(tw.level, 2)] * cl["rich"] * leaf_patch(Xs, N3, i, mc))
            samp_P.append(Xs)
            samp_w.append(w)
            leafy_m += float(w.sum()) * tw.L / max(len(Xs) - 1, 1)
    samp_P = np.vstack(samp_P)
    samp_w = np.concatenate(samp_w)
    grid = density_grid(samp_P, samp_w)
    stats = dict(R=[], G=[], B=[])
    for i, cl in enumerate(clusters):
        Pw = cl["info"]["world"]
        ao = volumetric_ao(grid, Pw, use_ao)
        B = np.clip(0.14 + 0.86 * ao, 0.0, 1.0)
        C.write_color0(cl["obj"].data, cl["R"], cl["G"], B)
        cl["B"] = B
    # scaffold: bare wood (no leaves), deep inside
    aos = volumetric_ao(grid, sinfo["world"], use_ao)
    C.write_color0(scaf.data, np.zeros(len(aos)), np.full(len(aos), 0.5), np.clip(0.1 + 0.8 * aos, 0, 1))
    # far shells
    for o, kind in ((fi, "inner"), (fo, "outer")):
        co, nrm, ed, _ = C.mesh_arrays(o.data)
        if kind == "inner":
            R = np.zeros(len(co))
            B = np.clip(0.3 + 0.3 * C.smoothstep(-0.3, 0.4, N3.fbm(co * 1.5 + 2.0, 2)), 0, 1) * volumetric_ao(grid, co, use_ao) ** 0.5
            G = np.full(len(co), 0.5)
        else:
            R = np.clip(0.55 * C.smoothstep(-0.1, 0.45, N3.fbm(co * 1.3 + 8.0, 3)), 0, 1)
            G = np.clip(0.7 + 0.2 * N3.fbm(co * 2.0 + 1.0, 2), 0.3, 1.0)
            B = np.clip(0.35 + 0.2 * C.smoothstep(-0.3, 0.4, N3.fbm(co * 1.1 + 5.0, 2)), 0, 1)
        C.write_color0(o.data, R, G, B)
    C.log(f"colours ({time.time() - t:.1f}s), leaf-bearing twig length (integral of R ds) {leafy_m:.1f} m")

    focus = set_focus(coll, clusters)
    C.log("focus (median depth of the leafy twigs):", focus)
    # ---- extras: reveal order, depth, radius
    right, up, fwd = mc.basis()
    cen = np.array([cl["info"]["world"].mean(0) for cl in clusters])
    u, v, z = mc.project(cen)
    rho = np.hypot((u - CROWN_UV16[0]) * C.ASPECT, v - CROWN_UV16[1]) / 0.5
    zz = cen[:, 2]
    zn = (zz - zz.min()) / max(zz.max() - zz.min(), 1e-6)
    jit = np.array([np.random.default_rng(SEED + 77 + i).uniform(-1, 1) for i in range(len(clusters))])
    score = 0.62 * np.clip(rho, 0, 1.3) + 0.38 * zn + 0.10 * jit
    rank = np.argsort(np.argsort(score))
    reveal = rank / max(len(clusters) - 1, 1)
    depth = (z - z.min()) / max(z.max() - z.min(), 1e-6)
    for i, cl in enumerate(clusters):
        o = cl["obj"]
        co, _, _, _ = C.mesh_arrays(o.data)
        rad = float(np.linalg.norm(co, axis=1).max())
        axis = C.normalize(cl["twigs"][0].X[-1] - cl["twigs"][0].X[0])
        o["reveal_order"] = round(float(reveal[i]), 4)
        o["depth"] = round(float(depth[i]), 4)
        o["radius_m"] = round(rad, 4)
        o["axis_gltf"] = [round(x, 4) for x in C.blender_to_gltf_dir(axis)]
        o["screen_uv_main"] = [round(float(u[i]), 4), round(float(v[i]), 4)]
        o["view_depth_m"] = round(float(z[i]), 3)
        o["kind"] = cl["kind"]
        o["twig_count"] = len(cl["twigs"])
        o["limb"] = int(groups[i])
        cl["reveal"], cl["depth"], cl["radius"] = float(reveal[i]), float(depth[i]), rad
    scaf["role"] = "scaffold: limbs from the hidden centre to every cluster base (no leaves, COLOR_0.R = 0)"
    scaf["reveal_note"] = ("wood_canopy_cNN: origin = cluster base on a scaffold connector. reveal_order = rank of "
                           "0.62 x screen distance of the cluster centroid from the crown centre (cam_canopy_main, "
                           "1 = crown radius) + 0.38 x height (low first) + 0.10 x seeded jitter, normalised to 0..1 "
                           "(inner / lower first). depth = centroid view depth from cam_canopy_main, 0 nearest .. 1 "
                           "farthest. radius_m = farthest vertex from the origin.")
    fi["role"] = "dark backdrop of the gaps inside the crown"
    fo["role"] = "dim surrounding foliage ~3 m behind the crown (frames 15 / 16); R = sparse dim leaves"
    total = sum(C.tri_count(o) for o in coll.objects if o.type == "MESH")
    C.log(f"canopy built: {total} tris, {time.time() - t0:.1f}s")
    return scene, coll, dict(clusters=clusters, grid=grid, scaffold=scaf, far=(fi, fo), sinfo=sinfo, leafy_m=leafy_m)


# ------------------------------------------------------------------------------------------
# previews with stand-in leaves (never exported)
# ------------------------------------------------------------------------------------------

LEAF_COLS = [C.hex_to_linear(h) for h in ("#5a9a3a", "#4a7f34", "#79a63f", "#4f8c34", "#8fc06a")]
LEAF_W = [0.32, 0.30, 0.22, 0.12, 0.04]
BARK = C.hex_to_linear("#4a3c2c")
FAR_COL = C.hex_to_linear("#0f170d")
KEY_E, FILL_E = 2.4, 0.12


def leaf_quads(P, A, Wd, Sz):
    """Diamond leaf blades (base, two sides, tip) -> (V, tris)."""
    m = len(P)
    w = 0.46 * Sz
    V = np.stack([P, P + A * (0.38 * Sz)[:, None] + Wd * (0.5 * w)[:, None], P + A * Sz[:, None],
                  P + A * (0.38 * Sz)[:, None] - Wd * (0.5 * w)[:, None]], 1).reshape(-1, 3)
    ids = np.arange(m)[:, None] * 4
    return V, np.concatenate([ids + np.array([0, 1, 2]), ids + np.array([0, 2, 3])], 0)


def far_leaves(obj, rng, count=7000):
    """Dim stand-in leaves on far_canopy_outer (density COLOR_0.R x area, a loose layer around the shell),
    for the previews only."""
    co, _, _, tri = C.mesh_arrays(obj.data)
    co = co + np.array(obj.location)
    col = C.read_color0(obj.data)
    a, b, c = co[tri[:, 0]], co[tri[:, 1]], co[tri[:, 2]]
    w = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1) * col[tri, 0].mean(1)
    if w.sum() <= 0:
        return None
    pick = rng.choice(len(tri), size=count, p=w / w.sum())
    r1, r2 = np.sqrt(rng.random(count)), rng.random(count)
    P = (a[pick] * (1 - r1)[:, None] + b[pick] * (r1 * (1 - r2))[:, None] + c[pick] * (r1 * r2)[:, None] +
         rng.normal(size=(count, 3)) * 0.1)
    nrm = C.normalize(np.array([0.0, -1.0, 0.4]) + 0.8 * rng.normal(size=(count, 3)))
    axis = rng.normal(size=(count, 3))
    axis = C.normalize(axis - np.sum(axis * nrm, 1)[:, None] * nrm)
    Sz = 0.028 * rng.uniform(0.8, 1.2, count)
    V, tris = leaf_quads(P, axis, C.normalize(np.cross(nrm, axis)), Sz)
    ci = rng.choice(len(LEAF_COLS), size=count, p=np.array(LEAF_W) / sum(LEAF_W))
    ldir = -C.normalize(np.asarray(KEY_DIR, float))
    light = 0.05 + 0.16 * np.abs(nrm @ ldir) * rng.uniform(0.3, 1.0, count)
    return V, tris, np.repeat(np.array(LEAF_COLS)[ci] * light[:, None], 4, axis=0)


def stand_in_leaves(clusters, grid, rng, count, reveal_max=1.0):
    """Leaves for the previews: positions along the twigs with density R, size 2.6 cm x G, a short
    petiole, blades turned outwards / up, double-sided. Returns (V, tris, per-vertex rgb)."""
    items = []
    for cl in clusters:
        if cl["reveal"] > reveal_max + 1e-9:
            continue
        info = cl["info"]
        for k, tw in enumerate(cl["twigs"]):
            sel = (info["twig"] == k)
            s = info["s"][sel]
            R = cl["R"][sel]
            G = cl["G"][sel]
            o = np.argsort(s)
            items.append((tw, s[o], R[o], G[o]))
    # expected leaves per twig = lambda * integral(R ds)
    integ = np.array([float(np.sum(0.5 * (R[1:] + R[:-1]) * np.diff(s))) if len(s) > 1 else 0.0
                      for (_, s, R, _) in items])
    lam = count / max(integ.sum(), 1e-9)
    P, A, Wd, Nn, Sz = [], [], [], [], []
    zup = np.array([0.0, 0.0, 1.0])
    for (tw, s, R, G), I in zip(items, integ):
        n = rng.poisson(lam * I)
        if n == 0 or len(s) < 2:
            continue
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (R[1:] + R[:-1]) * np.diff(s))])
        if cdf[-1] <= 0:
            continue
        q = (np.arange(n) + rng.random(n)) / n * cdf[-1]
        sq = np.interp(q, cdf, s)
        gq = np.interp(sq, s, G)
        pts = np.stack([np.interp(sq, tw.s, tw.X[:, j]) for j in range(3)], 1)
        tan = C.normalize(np.stack([np.interp(sq, tw.s, tw.T[:, j]) for j in range(3)], 1))
        rad = C.normalize(pts - H)
        pdir = rng.normal(size=(n, 3)) + 1.4 * rad + 0.45 * zup
        pdir = C.normalize(pdir - 0.7 * np.sum(pdir * tan, 1)[:, None] * tan)
        size = 0.028 * gq * rng.uniform(0.85, 1.15, n)
        base = pts + pdir * rng.uniform(0.004, 0.016, n)[:, None]
        nrm = C.normalize(0.9 * rad + 0.5 * zup + 0.75 * rng.normal(size=(n, 3)))
        axis = C.normalize(pdir + 0.4 * tan)
        axis = C.normalize(axis - np.sum(axis * nrm, 1)[:, None] * nrm)
        wdir = C.normalize(np.cross(nrm, axis))
        P.append(base)
        A.append(axis)
        Wd.append(wdir)
        Nn.append(nrm)
        Sz.append(size)
    P, A, Wd, Nn, Sz = (np.vstack(P), np.vstack(A), np.vstack(Wd), np.vstack(Nn), np.concatenate(Sz))
    m = len(P)
    V, tris = leaf_quads(P, A, Wd, Sz)
    # lighting (linear): albedo x (fill x AO x sky + key x (front + 0.35 back translucency) x shadow)
    ci = rng.choice(len(LEAF_COLS), size=m, p=np.array(LEAF_W) / sum(LEAF_W))
    alb = np.array(LEAF_COLS)[ci] * rng.uniform(0.78, 1.15, m)[:, None]
    ldir = -C.normalize(np.asarray(KEY_DIR, float))
    ndl = Nn @ ldir
    front = np.clip(ndl, 0, 1)
    back = np.clip(-ndl, 0, 1) * 0.35
    sh = grid.transmittance(P, ldir, sigma=0.5, length=2.4, steps=16, start=0.02)
    ao = grid.occlusion(P, C.normalize(P - H), sigma=0.5, length=1.6, n_dirs=8, steps=10, start=0.03)
    sky = 0.55 + 0.45 * np.clip(Nn[:, 2], -1, 1)
    light = FILL_E * ao * sky + KEY_E * (front + back) * sh
    rgb = alb * light[:, None]
    return V, tris, np.repeat(rgb, 4, axis=0), m


def preview_colours(info):
    """Twig / scaffold / far colours for the flat previews (linear)."""
    out = {}
    ldir = -C.normalize(np.asarray(KEY_DIR, float))
    grid = info["grid"]
    for cl in info["clusters"]:
        o = cl["obj"]
        co, nrm, _, _ = C.mesh_arrays(o.data)
        P = co + np.array(o.location)
        sh = grid.transmittance(P, ldir, sigma=0.5, length=2.4, steps=12, start=0.01)
        lit = np.clip(nrm @ ldir, 0, 1) * sh
        out[o.name] = np.asarray(BARK)[None, :] * ((FILL_E + KEY_E * lit) * cl["B"])[:, None]
    sc = info["scaffold"]
    co, nrm, _, _ = C.mesh_arrays(sc.data)
    col = C.read_color0(sc.data)
    b2 = col[:, 2] ** 2
    out[sc.name] = np.asarray(BARK)[None, :] * (FILL_E * b2 + 0.3 * KEY_E * np.clip(nrm @ ldir, 0, 1) * b2)[:, None]
    for o in info["far"]:
        co, nrm, _, _ = C.mesh_arrays(o.data)
        col = C.read_color0(o.data)
        out[o.name] = np.asarray(FAR_COL)[None, :] * ((0.15 + 0.25 * np.clip(nrm @ ldir, 0, 1)) * col[:, 2])[:, None]
    return out


def render_previews(scene, coll, info, cams=None, percent=50, suffix="", reveal_max=1.0, leaves=True,
                    outdir=None, prefix="canopy_", hide=()):
    names = cams or list(CAMS.keys())
    cols = preview_colours(info)
    L.setup_flat(scene, percent=percent, bg=(0.0, 0.0, 0.0))
    out = []
    rng = np.random.default_rng(SEED + 999)
    with L.FlatPreview(scene, "canopy_preview") as pv:
        for name, rgb in cols.items():
            if name in hide:
                continue
            o = coll.objects[name]
            if name.startswith("wood_canopy_c") and o.get("reveal_order", 0.0) > reveal_max + 1e-9:
                continue
            L.preview_copy(o, pv.coll, rgb)
        if leaves:
            V, tris, rgb, m = stand_in_leaves(info["clusters"], info["grid"], rng, PREVIEW_LEAVES * reveal_max ** 0.5,
                                              reveal_max)
            L.preview_mesh("PREVIEW_stand_in_leaves", pv.coll, V, [tris], rgb)
            C.log(f"preview: {m} stand-in leaves (reveal <= {reveal_max})")
            fl = None if "far_canopy_outer" in hide else far_leaves(info["far"][1], rng)
            if fl is not None:
                L.preview_mesh("PREVIEW_stand_in_far_leaves", pv.coll, fl[0], [fl[1]], fl[2])
        for n in names:
            path = os.path.join(outdir or CAPTURES, f"{prefix}{n}{suffix}.png")
            C.render_camera(scene, coll.objects[n], path)
            out.append(path)
    return out


# ------------------------------------------------------------------------------------------
# comparison sheets
# ------------------------------------------------------------------------------------------

def crown_mask(rgb):
    """Leaf mask of a crown on black: green-ish pixels above a luminance floor, box-blurred, filled."""
    a = rgb[..., :3]
    lum = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
    green = (a[..., 1] > a[..., 0] * 1.02) & (a[..., 1] > a[..., 2] * 1.05)
    leaf = (lum > 0.05) & green
    m = L.box_blur(leaf.astype(float), 5) > 0.35
    return m


def silhouette(m):
    """Hole-filled silhouette at quarter resolution, upsampled back (fast flood fill)."""
    h, w = m.shape
    q = m[:h // 2 * 2, :w // 2 * 2].reshape(h // 2, 2, w // 2, 2).mean((1, 3)) > 0.5
    f = L.fill_holes(q)
    up = np.repeat(np.repeat(f, 2, 0), 2, 1)
    out = np.zeros_like(m)
    out[:up.shape[0], :up.shape[1]] = up
    return out


def compare_sheets(scene, coll, info, percent=50):
    os.makedirs(TMP, exist_ok=True)
    stats, paths = {}, []
    # our silhouettes: stand-in leaves + twigs (far shells hidden), alpha masks
    mask_paths = {}
    with L.FlatPreview(scene, "canopy_mask") as pv:
        rng = np.random.default_rng(SEED + 999)
        V, tris, rgb, m = stand_in_leaves(info["clusters"], info["grid"], rng, PREVIEW_LEAVES)
        leaves = L.preview_mesh("PREVIEW_stand_in_leaves", pv.coll, V, [tris], rgb)
        objs = [leaves] + [L.preview_copy(cl["obj"], pv.coll, np.ones((len(cl["obj"].data.vertices), 3)))
                           for cl in info["clusters"]]
        mask_paths = L.render_masks(scene, [coll.objects[n] for n in
                                            ("cam_canopy_in", "cam_canopy_main", "cam_canopyClose_main")],
                                    TMP, objs, percent)
    for key, camn, ref_rel in (("in", "cam_canopy_in", "frames/15_canopy_entry.png"),
                               ("main", "cam_canopy_main", "frames/16_canopy.png"),
                               ("close", "cam_canopyClose_main", "frames/17_canopy_close.png")):
        ref = L.down2(L.load_rgba(os.path.join(REF, ref_rel)))
        h, w = ref.shape[:2]
        rsil = silhouette(crown_mask(ref))
        alpha = L.load_rgba(mask_paths[camn])[..., 3] > 0.5
        osil = silhouette(L.box_blur(alpha.astype(float), 6) > 0.25)
        valid = np.ones((h, w), bool)
        valid[:int(0.09 * h)] = False                           # nav bar
        pp = os.path.join(CAPTURES, f"canopy_{camn}.png")
        prev = L.load_rgba(pp)[..., :3] if os.path.exists(pp) else np.zeros_like(ref[..., :3])
        ov = ref[..., :3] * 0.55
        ov[L.outline(rsil, 1)] = (0.2, 0.95, 1.0)
        ov[L.outline(osil, 1)] = (1.0, 0.92, 0.1)
        L.grid(ov)
        path = os.path.join(CAPTURES, f"canopy_cmp_{key}.png")
        L.save_rgb(np.concatenate([ref[..., :3], prev, ov], 1), path)
        paths.append(path)
        st = dict(iou_silhouette=round(L.iou(rsil, osil, valid), 4),
                  coverage_ref=round(float(rsil[valid].mean()), 4), coverage_ours=round(float(osil[valid].mean()), 4))
        # leaf-pixel share inside the reference silhouette (how 'full' the crown reads)
        rl = crown_mask(ref)
        pl = alpha
        st["leaf_share_in_ref_silhouette"] = [round(float(rl[rsil & valid].mean()), 3),
                                              round(float(pl[rsil & valid].mean()), 3)]
        ys, xs = np.nonzero(osil & valid)
        ry, rx = np.nonzero(rsil & valid)
        if len(xs) and len(rx):
            st["bbox_u_ref"] = [round(rx.min() / w, 3), round(rx.max() / w, 3)]
            st["bbox_u_ours"] = [round(xs.min() / w, 3), round(xs.max() / w, 3)]
            st["bbox_v_ref"] = [round(ry.min() / h, 3), round(ry.max() / h, 3)]
            st["bbox_v_ours"] = [round(ys.min() / h, 3), round(ys.max() / h, 3)]
        stats[key] = st
    # detail: details/08 = frame 16 crop [1020, 210, 1430, 650] vs the same crop of a full-size preview
    full = render_previews(scene, coll, info, ["cam_canopy_main"], percent=100, outdir=TMP, prefix="full_")[0]
    fp = L.load_rgba(full)[..., :3]
    r16 = L.load_rgba(os.path.join(REF, "frames/16_canopy.png"))[..., :3]
    x0, y0, x1, y1 = 1020, 210, 1430, 650
    path = os.path.join(CAPTURES, "canopy_cmp_detail.png")
    L.save_rgb(np.concatenate([r16[y0:y1, x0:x1], fp[y0:y1, x0:x1]], 1), path)
    paths.append(path)
    # all poses in a strip
    panels = []
    for n in CAMS:
        p = os.path.join(CAPTURES, f"canopy_{n}.png")
        if os.path.exists(p):
            im = L.load_rgba(p)[..., :3].copy()
            L.grid(im)
            panels.append(L.down2(im))
    if panels:
        path = os.path.join(CAPTURES, "canopy_cmp_poses.png")
        L.save_rgb(np.concatenate(panels, 1), path)
        paths.append(path)
    # reveal order: clusters with reveal_order <= 0.25 / 0.5 / 0.75 / 1 seen from cam_canopy_in
    panels = []
    for rm in (0.25, 0.5, 0.75, 1.0):
        p = render_previews(scene, coll, info, ["cam_canopy_in"], percent=25, outdir=TMP,
                            prefix=f"reveal_{int(rm * 100)}_", reveal_max=rm, hide=("far_canopy_outer",))[0]
        panels.append(L.load_rgba(p)[..., :3])
    path = os.path.join(CAPTURES, "canopy_cmp_reveal.png")
    L.save_rgb(np.concatenate(panels, 1), path)
    paths.append(path)
    # bare skeleton (no stand-in leaves) from the main camera
    p = render_previews(scene, coll, info, ["cam_canopy_main"], percent=percent, outdir=TMP, prefix="bare_",
                        leaves=False)[0]
    bare = L.load_rgba(p)[..., :3]
    bare = np.clip(bare * 4.0, 0, 1)                    # brightened: the twigs are dark
    path = os.path.join(CAPTURES, "canopy_cmp_skeleton.png")
    L.save_rgb(bare, path)
    paths.append(path)
    C.log("compare:", json.dumps(stats))
    return paths, stats


# ------------------------------------------------------------------------------------------
# validation
# ------------------------------------------------------------------------------------------

CAM_RE_OK = ("cam_canopy_in", "cam_canopy_main", "cam_canopy_out", "cam_canopyClose_main", "cam_canopyClose_out")


def validate_canopy(rep, coll, info):
    errs, out = [], {}
    # common.validate_glb only accepts lower-case episode ids; canopyClose is an episode id of the
    # engine (SceneConfig / CameraRig accept [A-Za-z]+): replace that error by an exact check
    rep["errors"] = [e for e in rep["errors"] if not any(e.endswith(n) for n in CAM_RE_OK[3:])]
    names = [n["name"] for n in rep["nodes"]]
    for n in CAM_RE_OK + ("key_canopy", "wood_canopy_scaffold", "far_canopy_inner", "far_canopy_outer"):
        if n not in names:
            errs.append(f"missing node {n}")
    cl_nodes = sorted(n for n in names if n.startswith("wood_canopy_c") and n[13:].isdigit())
    out["cluster_count"] = len(cl_nodes)
    if not 30 <= len(cl_nodes) <= 60:
        errs.append(f"{len(cl_nodes)} clusters (want 30-60)")
    if any(len(n) != len("wood_canopy_c00") for n in cl_nodes):
        errs.append("cluster names are not wood_canopy_cNN")
    js, binb = C.read_glb(rep["path"])
    nodes = {n.get("name"): n for n in js["nodes"]}
    reveal, depth = [], []
    for n in cl_nodes:
        ex = nodes[n].get("extras") or {}
        for k in ("reveal_order", "depth", "radius_m"):
            if k not in ex:
                errs.append(f"{n}: extras without {k}")
        reveal.append(ex.get("reveal_order", -1))
        depth.append(ex.get("depth", -1))
        if "rotation" in nodes[n] or "scale" in nodes[n]:
            errs.append(f"{n}: node has rotation / scale (only a translation to the base is expected)")
        # origin at the base: the first ring of the main stem surrounds the origin
        mesh = js["meshes"][nodes[n]["mesh"]]
        pos = C.glb_accessor(js, binb, mesh["primitives"][0]["attributes"]["POSITION"])
        dmin = float(np.linalg.norm(pos, axis=1).min())
        if dmin > 0.006:
            errs.append(f"{n}: origin {dmin * 1000:.1f} mm from the nearest vertex (base not at the origin)")
    out["reveal_order_range"] = [min(reveal), max(reveal)] if reveal else None
    out["depth_range"] = [min(depth), max(depth)] if depth else None
    if reveal and (abs(min(reveal)) > 1e-6 or abs(max(reveal) - 1.0) > 1e-6 or len(set(reveal)) != len(reveal)):
        errs.append("reveal_order is not a 0..1 ranking")
    for cname in CAM_RE_OK:
        ex = (rep["cameras"].get(cname) or {}).get("extras") or {}
        for k in ("focus_distance_m", "frame_height_at_focus_m", "video_time_s"):
            if k not in ex:
                errs.append(f"{cname}: extras without {k}")
    # COLOR_0 ranges (from the GLB) and thickness of the twigs
    rng_tab = {}
    allR = []
    for n, m in rep["meshes"].items():
        c0 = m["attributes"].get("COLOR_0", {})
        rng_tab[n] = dict(min=c0.get("min"), max=c0.get("max"), mean=c0.get("mean"))
    out["color0"] = rng_tab
    radii = np.concatenate([np.concatenate([tw.r for tw in cl["twigs"]]) for cl in info["clusters"]])
    out["twig_diameter_mm"] = [round(float(radii.min()) * 2000, 2), round(float(radii.max()) * 2000, 2)]
    if radii.min() * 2 < 0.0019 or radii.max() * 2 > 0.0081:
        errs.append(f"twig diameters {out['twig_diameter_mm']} mm outside 2-8 mm")
    for cl in info["clusters"]:
        allR.append(cl["R"])
    allR = np.concatenate(allR)
    out["R_share_gt_0.5"] = round(float((allR > 0.5).mean()), 3)
    out["tris_by_kind"] = {k: int(sum(v["tris"] for n, v in rep["meshes"].items() if n.startswith(k)))
                           for k in ("wood_canopy_c", "wood_canopy_scaffold", "far_canopy")}
    return dict(errors=errs, **out)


def save_blend():
    bpy.ops.wm.save_as_mainfile(filepath=BLEND_PATH, check_existing=False, compress=True)
    C.log("saved", BLEND_PATH)


def parse_args(argv):
    a = dict(export=False, validate=False, render=None, percent=50, save=False, ao=True, compare=False)
    i = 0
    while i < len(argv):
        x = argv[i]
        if x == "--export":
            a["export"] = True
        elif x == "--validate":
            a["validate"] = True
        elif x == "--render":
            a["render"] = []
            while i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                i += 1
                a["render"].append(argv[i])
        elif x == "--compare":
            a["compare"] = True
        elif x == "--percent":
            i += 1
            a["percent"] = int(argv[i])
        elif x == "--save":
            a["save"] = True
        elif x == "--no-ao":
            a["ao"] = False
        i += 1
    return a


def main(argv=None):
    a = parse_args(argv or [])
    if bpy.app.background and a["save"] and os.path.exists(BLEND_PATH):
        bpy.ops.wm.open_mainfile(filepath=BLEND_PATH)   # keep the other scene sets
    scene, coll, info = build(use_ao=a["ao"])
    result = dict(tris=sum(C.tri_count(o) for o in coll.objects if o.type == "MESH"),
                  clusters=len(info["clusters"]))
    if a["export"]:
        C.export_glb(scene, GLB_PATH)
        result["glb"] = GLB_PATH
        result["glb_bytes"] = os.path.getsize(GLB_PATH)
    if a["validate"]:
        rep = C.validate_glb(GLB_PATH, max_total_tris=MAX_TRIS, max_bytes=MAX_BYTES)
        extra = validate_canopy(rep, coll, info)
        rep["canopy"] = extra
        result["validate"] = dict(errors=rep["errors"] + extra["errors"], warnings=rep["warnings"][:6],
                                  n_warnings=len(rep["warnings"]), totals=rep["totals"])
        os.makedirs(CAPTURES, exist_ok=True)
        with open(os.path.join(CAPTURES, "canopy_validate.json"), "w") as f:
            json.dump(rep, f, indent=1, default=str)
    if a["render"] is not None or a["compare"]:
        result["renders"] = render_previews(scene, coll, info, a["render"] or None, a["percent"])
    if a["compare"]:
        result["compare"], result["compare_stats"] = compare_sheets(scene, coll, info, a["percent"])
    if a["save"]:
        save_blend()
    C.log("result", json.dumps(result, default=str))
    return result


if __name__ == "__main__":
    _argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    main(_argv)
