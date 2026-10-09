"""
build_grove.py - the 'grove' scene set: S01 hero masses, the T01 overgrown mass the camera descends
past, and the S02 mossy trunk arch - one continuous world - exported to
public/nature/models/grove.glb (contract: CLAUDE.md "Asset contract").

Run headless (from the repo root):
    "/c/Program Files/Blender Foundation/Blender 5.1/blender.exe" --background --factory-startup \
        --python assets-src/blender/build_grove.py -- --export --validate [--render] [--save]

Run inside a live Blender session (e.g. through MCP):
    import sys, importlib; sys.path.insert(0, r"<repo>/assets-src/blender")
    import build_grove; importlib.reload(build_grove); build_grove.main(["--export", "--validate"])

Options:
    --export            write public/nature/models/grove.glb
    --glb PATH          export to / validate PATH instead (scratch builds; the validate report then
                        goes next to it as <name>_validate.json)
    --validate          re-read / re-import the GLB and check it against the contract
    --render [cams]     Workbench previews -> docs/captures/blender/grove_<cam>.png (all cams if none given)
    --percent N         preview size in % of 1440x1020 (default 50)
    --save              save nature.blend (headless: opens the existing file first so other
                        scene sets are kept, then replaces only the 'grove' scene content)
    --no-ao             skip the ray-traced AO (B channel = cavity only; fast iteration)
    --compare [keys]    comparison sheets reference | preview | outline -> grove_cmp_<key>.png
                        (all SHEETS keys if none given; T01 keys also print a sky IoU; the motion/01
                        panels are compared at the engine's camera track pose of their time)
    --sheet-percent N   size of the --compare sheets (default: --percent)
    --outdir DIR        where --render / --compare write (default docs/captures/blender; the
                        per-view previews and masks go to DIR/_grove_views, or to the system temp
                        folder when DIR is docs/captures/blender)

For quick iteration build(only=[...]) builds a subset (keys of MASS_BUILDERS / TRUNK_BUILDERS),
previews only - an export needs the full set.

Everything is deterministic (seed 134). World units: metres, Blender Z-up, cameras look along +Y.
"""

import json
import math
import os
import sys
import tempfile
import time
import importlib

import bpy
import numpy as np
from mathutils import Quaternion

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import common as C  # noqa: E402

importlib.reload(C)

ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
GLB_PATH = os.path.join(ROOT, "public", "nature", "models", "grove.glb")
BLEND_PATH = os.path.join(HERE, "nature.blend")
CAPTURES = os.path.join(ROOT, "docs", "captures", "blender")
SCENE_NAME = "grove"
COLL_NAME = "grove"
SEED = 134

TV = math.tan(math.radians(C.FOV_V_DEG) / 2.0)
TH = TV * C.ASPECT

# ------------------------------------------------------------------------------------------
# cameras: key poses the engine interpolates, in scroll order
#   cam_hero_main -> cam_hero_out -> cam_hero_p1 -> cam_hero_p2 -> cam_arch_in -> cam_arch_main -> cam_arch_out
# (t = matching video second; cam_hero_out holds until ~8.0 s, the descent runs 8.0 -> 10.0 s)
# focus = distance (m) along the view axis to the focus plane; frame height there = 0.6306 * focus
# ------------------------------------------------------------------------------------------
CAMS = {
    "cam_hero_main": dict(loc=(-0.60, -2.65, 4.70), pitch=-4.0, yaw=0.0, roll=0.0, focus=1.75, t=0.5),
    "cam_hero_out": dict(loc=(-0.615, -2.56, 4.665), pitch=-4.1, yaw=-1.0, roll=-2.0, focus=1.65, t=5.5,
                         hold=8.0),
    "cam_hero_p1": dict(loc=(-0.20, -3.72, 3.66), pitch=0.0, yaw=0.0, roll=0.0, focus=2.05, t=8.8),
    "cam_hero_p2": dict(loc=(-0.20, -3.72, 3.38), pitch=0.0, yaw=0.0, roll=0.0, focus=2.1, t=9.6),
    "cam_arch_in": dict(loc=(0.25, -4.85, 1.54), pitch=8.4, yaw=-1.1, roll=-3.2, focus=3.9, t=10.0),
    "cam_arch_main": dict(loc=(0.30, -4.75, 1.55), pitch=0.0, yaw=0.0, roll=0.0, focus=4.75, t=12.5),
    "cam_arch_out": dict(loc=(0.33, -4.40, 1.47), pitch=-15.5, yaw=-0.6, roll=3.2, focus=4.2, t=15.2),
}

# suggested key-light directions (direction the light travels, Blender world coords)
KEYS = {
    "key_hero": dict(loc=(-0.2, -1.2, 5.2), dir=(0.25, 0.45, -0.86)),
    "key_arch": dict(loc=(0.6, -1.0, 3.2), dir=(-0.55, 0.30, -0.78)),
}


def cam_q(name):
    c = CAMS[name]
    return C.cam_quat(c["pitch"], c["yaw"], c["roll"])


_POSES = {}


def pose_of(p):
    """(loc, quat) of a camera name, or of the engine camera at video time p (track replica)."""
    if isinstance(p, str):
        return np.asarray(CAMS[p]["loc"], float), cam_q(p)
    key = round(float(p), 4)
    if key not in _POSES:
        loc, q, _ = track_pose(key)
        _POSES[key] = (np.asarray(loc, float), q)
    return _POSES[key]


def S(cam, u, v, d):
    """World point at frame position (u, v) and view depth d for camera `cam` (name or video time)."""
    loc, q = pose_of(cam)
    return C.screen_to_world(loc, q, u, v, d)


def du(d):
    return 1.0 / (2.0 * d * TH)   # u per metre at depth d


def dv(d):
    return 1.0 / (2.0 * d * TV)   # v per metre at depth d


def edge_lumps(sdf, cam, edge, depths, radii, side, k, squash=(1.0, 1.0, 1.0), seed=0, jitter=0.25):
    """Ellipsoid lumps whose screen outline touches a silhouette polyline `edge` [(u, v), ...].

    side=+1: mass lies to the left of the polyline direction (in screen metric space), -1 right.
    """
    rng = np.random.default_rng(seed)
    E = np.asarray(edge, float)
    n = len(E)
    depths = np.broadcast_to(np.asarray(depths, float), (n,))
    radii = np.broadcast_to(np.asarray(radii, float), (n,))
    for i in range(n):
        u, v = E[i]
        d = depths[i]
        a = E[max(i - 1, 0)]
        b = E[min(i + 1, n - 1)]
        # tangent in metric screen space (x right, y down)
        tx = (b[0] - a[0]) / du(d)
        ty = (b[1] - a[1]) / dv(d)
        L = math.hypot(tx, ty) or 1.0
        tx, ty = tx / L, ty / L
        nx, ny = (ty * side, -tx * side)       # rotate +-90 deg
        r = radii[i] * (1.0 + jitter * rng.uniform(-1, 1))
        cu = u + nx * r * du(d)
        cv = v + ny * r * dv(d)
        c = S(cam, cu, cv, d)
        sq = np.asarray(squash) * (1.0 + 0.2 * rng.uniform(-1, 1, 3))
        sdf.ellipsoid(c, r * sq, k=k)


def fill_screen(sdf, cam, poly, depth, r, spacing, seed, *, depth_jit=0.08, r_jit=0.25,
                squash=(1.2, 0.8, 1.0), k=0.1, zrange=None, box=(-0.1, -0.1, 1.1, 1.1)):
    """Lumps on a jittered hex-ish grid (metric `spacing` at `depth`) covering the screen polygon
    `poly` of camera `cam`, limited to the uv `box` and world z range."""
    rng = np.random.default_rng(seed)
    V = np.array([((2 * u - 1) * TH, (2 * v - 1) * TV) for (u, v) in poly], float)
    su, sv = spacing * du(depth), spacing * dv(depth)
    for j, v in enumerate(np.arange(box[1], box[3] + 1e-9, sv)):
        for u in np.arange(box[0] + (0.5 * su if j % 2 else 0.0), box[2] + 1e-9, su):
            uu = u + 0.3 * su * rng.uniform(-1, 1)
            vv = v + 0.3 * sv * rng.uniform(-1, 1)
            if C.poly_sdf(np.array([[(2 * uu - 1) * TH, (2 * vv - 1) * TV]]), V)[0] > 0:
                continue
            d = depth + depth_jit * rng.uniform(-1, 1)
            P = S(cam, uu, vv, d)
            if zrange is not None and not (zrange[0] <= P[2] <= zrange[1]):
                continue
            rr = r * (1.0 + r_jit * rng.uniform(-1, 1))
            sdf.ellipsoid(P, rr * np.asarray(squash) * (1.0 + 0.15 * rng.uniform(-1, 1, 3)), k=k)


def clip(sdf, cam, poly, k=0.03, weight=None, margin=0.0, invert=False):
    """Visual-hull clamp of `sdf` to the screen polygon `poly` of camera `cam` (name or video time;
    invert: keep out)."""
    loc, q = pose_of(cam)
    sdf.clip_view(loc, q, poly, margin=margin, k=k, weight=weight, invert=invert)


def zband(z0, z1, fade=0.1):
    """Weight 1 for world z in [z0, z1], fading to 0 over `fade` outside."""
    return lambda P: C.smoothstep(z0 - fade, z0, P[:, 2]) * (1.0 - C.smoothstep(z1, z1 + fade, P[:, 2]))


def right_of(edge, far=6.0):
    """Screen region right of a top->bottom edge polyline (edge extended vertically off-frame)."""
    e = [tuple(p) for p in edge]
    return [(e[0][0], -far)] + e + [(e[-1][0], far), (far, far), (far, -far)]


def left_of(edge, far=6.0):
    e = [tuple(p) for p in edge]
    return [(e[0][0], -far)] + e + [(e[-1][0], far), (-far, far), (-far, -far)]


def _px(edge):
    return np.asarray(edge, float) * [C.FRAME_W, C.FRAME_H]


def inset(edge, side, amount):
    """Offset a screen polyline towards its mass side (side as in edge_lumps) by `amount` frame
    widths, isotropic in pixels."""
    P = _px(edge)
    n = len(P)
    out = []
    for i in range(n):
        a, b = P[max(i - 1, 0)], P[min(i + 1, n - 1)]
        t = (b - a) / max(np.hypot(*(b - a)), 1e-9)
        nrm = np.array([t[1] * side, -t[0] * side])
        out.append(P[i] + nrm * amount * C.FRAME_W)
    return [tuple(x) for x in np.asarray(out) / [C.FRAME_W, C.FRAME_H]]


def resample(edge, step_px):
    """Polyline resampled at a uniform pixel spacing."""
    P = _px(edge)
    L = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    s = np.linspace(0.0, L[-1], max(2, int(round(L[-1] / step_px)) + 1))
    Q = np.stack([np.interp(s, L, P[:, 0]), np.interp(s, L, P[:, 1])], 1)
    return [tuple(x) for x in Q / [C.FRAME_W, C.FRAME_H]]


def fill_region(sdf, cam, poly, depth, r, spacing, seed, *, depth_jit=0.08, r_jit=0.25,
                squash=(1.2, 0.8, 1.0), k=0.1, zrange=None, box=(-0.1, -0.1, 1.1, 1.1), keep_in=0.8):
    """Like fill_screen, but lump centres stay keep_in * r (screen, at `depth`) inside the polygon,
    so the lumps round off towards the outline instead of being sliced by a clamp."""
    rng = np.random.default_rng(seed)
    V = np.array([((2 * u - 1) * TH, (2 * v - 1) * TV) for (u, v) in poly], float)
    su, sv = spacing * du(depth), spacing * dv(depth)
    for j, v in enumerate(np.arange(box[1], box[3] + 1e-9, sv)):
        for u in np.arange(box[0] + (0.5 * su if j % 2 else 0.0), box[2] + 1e-9, su):
            uu = u + 0.3 * su * rng.uniform(-1, 1)
            vv = v + 0.3 * sv * rng.uniform(-1, 1)
            d = depth + depth_jit * rng.uniform(-1, 1)
            rr = r * (1.0 + r_jit * rng.uniform(-1, 1))
            jit = rng.uniform(-1, 1, 3)
            # poly_sdf works in tan units (x / depth): r at depth d is r / d
            if C.poly_sdf(np.array([[(2 * uu - 1) * TH, (2 * vv - 1) * TV]]), V)[0] > -keep_in * rr / d:
                continue
            P = S(cam, uu, vv, d)
            if zrange is not None and not (zrange[0] <= P[2] <= zrange[1]):
                continue
            sdf.ellipsoid(P, rr * np.asarray(squash) * (1.0 + 0.15 * jit), k=k)


# ------------------------------------------------------------------------------------------
# hero (S01) - anchored to cam_hero_main.  Silhouettes are the *base* surface (no plants);
# the engine's grass / leaves stand out of it by a fringe that was measured on revision 2 in the
# engine (silhouette of the capture vs the Blender base, normal distance): ~100 px on the
# overhang's right flank, ~90 px on its lower edge, ~105-120 px at the neck and the column's top,
# ~70 px lower down the column and ~45 px at its foot (1 cm ~ 9 px at 1.8 m).  The revision-3
# outlines below are the frame-01 silhouette (plant tips) eroded by that fringe (the overhang's
# upper-left side, new in revision 3, by ~85 px).
# ------------------------------------------------------------------------------------------

# crown of the big mass = lower-right mound of frame 01 (top silhouette, left -> right)
CROWN_EDGE = [(0.395, 0.80), (0.405, 0.76), (0.425, 0.728), (0.455, 0.705), (0.495, 0.69),
              (0.545, 0.68), (0.595, 0.676), (0.645, 0.684), (0.695, 0.696), (0.745, 0.709),
              (0.80, 0.722), (0.855, 0.736), (0.91, 0.75), (0.965, 0.768), (1.02, 0.792), (1.08, 0.82)]
# upper-left overhang (revision 3): a thick diagonal arm from the top edge down to the left edge
# (frame 01, details/01), not a blob filling the corner: the top-left corner above it is empty.
# Lower / right silhouette: right flank from the top edge down to the rounded nose (u ~0.23,
# v ~0.29), then the straight lower edge where the combed pile hangs, down to the left edge.
# Listed from the left frame edge up to the top edge (mass on the left of the direction).
ARM_LR = [(-0.011, 0.549), (0.013, 0.518), (0.025, 0.492), (0.037, 0.473), (0.053, 0.460), (0.072, 0.446),
          (0.088, 0.427), (0.106, 0.412), (0.122, 0.398), (0.142, 0.389), (0.163, 0.375), (0.179, 0.355),
          (0.193, 0.340), (0.208, 0.330), (0.228, 0.320), (0.245, 0.292), (0.252, 0.260), (0.259, 0.233),
          (0.270, 0.207), (0.279, 0.178), (0.287, 0.151), (0.292, 0.123), (0.298, 0.094), (0.305, 0.066),
          (0.308, 0.035), (0.309, 0.0), (0.31, -0.06)]
# upper-left silhouette of the arm, from the left frame edge up to the top edge (mass on the right)
ARM_UL = [(-0.016, 0.266), (0.010, 0.260), (0.032, 0.255), (0.054, 0.246), (0.074, 0.231), (0.091, 0.213),
          (0.109, 0.198), (0.126, 0.182), (0.145, 0.167), (0.163, 0.150), (0.177, 0.126), (0.184, 0.099),
          (0.188, 0.070), (0.188, 0.040), (0.188, 0.0), (0.188, -0.06)]
# left column right silhouette: from the left frame edge (the neck gap above it closes with the
# plants of the arm and the column) out to the lower-left bulge and down behind the near band
LEFT_EDGE = [(-0.012, 0.588), (0.004, 0.607), (0.016, 0.631), (0.024, 0.658), (0.033, 0.684), (0.045, 0.708),
             (0.060, 0.727), (0.077, 0.744), (0.095, 0.759), (0.114, 0.772), (0.133, 0.783), (0.153, 0.792),
             (0.172, 0.801), (0.186, 0.821), (0.187, 0.849), (0.180, 0.876), (0.169, 0.902), (0.157, 0.925),
             (0.145, 0.949)]

# allowed screen regions (visual-hull clamps) seen from cam_hero_main; generous off-frame
HERO_CROWN_POLY = CROWN_EDGE + [(3.0, 1.0), (3.0, 6.0), (-3.0, 6.0), (-3.0, 1.08), (0.30, 1.04),
                                (0.37, 0.93), (0.39, 0.85)]
# the arm: the band between its two silhouettes, open towards the top and the left (off-frame)
HERO_UPPER_POLY = ARM_UL + [(0.188, -3.0), (0.31, -3.0)] + ARM_LR[::-1] + [(-3.0, 4.5), (-3.0, 0.95)]
HERO_COLUMN_POLY = [(-3.0, 0.50), (-0.03, 0.535)] + LEFT_EDGE + [(0.12, 1.02), (0.36, 1.06), (0.36, 6.0),
                                                                  (-3.0, 6.0)]

# T01 (8.0-10.0 s): the camera descends past the big mass (left: the wavy column, right: the
# wall, light keyhole between).  Traced as the light-background boundary (plant tips) in the view
# at 9.1 s (frame 03), which is the master view: motion/01 at 8.8 / 9.6 s shows the same edges
# shifted down 0.12 / up 0.09 frame heights (pure vertical camera move at constant distance,
# measured on the recess, the bulges and the lit grass patch), and cam_hero_p1 / _p2 reproduce
# that move.  The tops (v < 0.1) come from 8.6 / 8.8 s and the chin from 9.6 / 9.8 s.
T01_VIEW = 9.1
T01_FRINGE = 0.032            # base surface this far (frame widths) inside the plant tips
T01_RIM_DEPTH = 2.15          # m from the camera: rim of the wall edge (front face ~1.95 m)
T01_COL_DEPTH = 2.3           # rim of the column edge
T01_WALL_SKY = [(0.60, -0.22), (0.52, -0.17), (0.46, -0.11), (0.41, -0.05), (0.37, 0.01), (0.335, 0.055),
                (0.305, 0.10), (0.285, 0.15), (0.275, 0.20), (0.275, 0.26), (0.28, 0.31), (0.295, 0.35),
                (0.32, 0.39), (0.345, 0.43), (0.36, 0.48), (0.37, 0.53), (0.38, 0.565), (0.385, 0.60),
                (0.375, 0.635), (0.355, 0.68), (0.335, 0.72), (0.315, 0.76), (0.295, 0.80), (0.275, 0.84),
                (0.258, 0.88), (0.25, 0.92), (0.256, 0.955), (0.28, 0.98), (0.33, 0.995), (0.45, 1.0),
                (0.58, 1.005), (0.66, 1.03), (0.74, 1.10)]
T01_CHIN = 26                 # index of the chin's lowest-left point in T01_WALL_SKY
T01_COL_SKY = [(0.17, -0.02), (0.15, 0.03), (0.135, 0.09), (0.12, 0.15), (0.10, 0.20), (0.09, 0.25),
               (0.088, 0.30), (0.09, 0.35), (0.105, 0.40), (0.13, 0.45), (0.15, 0.50), (0.165, 0.55),
               (0.175, 0.60), (0.17, 0.65), (0.15, 0.70), (0.125, 0.75), (0.10, 0.80), (0.08, 0.85),
               (0.06, 0.89), (0.04, 0.95), (0.02, 1.02)]
# 8.4 s (motion/01): the mass rises as an asymmetric diagonal flank from the lower left up to its
# crest right of the centre and falls slowly to the right - base outline in the engine's 8.4 s
# view (px; the plant tips stand ~45 px out of it on the flank, 2.1-2.5 m away, and ~70-100 px on
# the wall's top on the right, 1.1-1.5 m away; inside the glass card, 550-890 px, interpolated).
# Applied to the crown part (z > 3.55) at every depth, except the hero crest band (crest_guard),
# which keeps frame 01's outline: its left end stays a little proud of this flank at 8.4 s.
T01_FLANK_84 = [(x / C.FRAME_W, y / C.FRAME_H) for (x, y) in [
    (470, 6000), (470, 900), (490, 760), (515, 690), (555, 640), (600, 600), (650, 540), (720, 470),
    (800, 425), (880, 400), (1000, 430), (1100, 470), (1200, 510), (1300, 560), (1440, 610),
    (1800, 660), (1800, 6000)]]
# 9.1 s (frame 03): the lit chin - a bulge between the upper lobe and the recess that stands out of
# the wall towards the lens, its top turned up to the light from behind the wall (the T01 key)
# (v in the 9.1 s view, inset from the base edge (frame widths), view depth m, radius m)
T01_CHIN_LUMPS = [(0.46, 0.035, 1.98, 0.1), (0.5, 0.04, 1.9, 0.12), (0.54, 0.055, 1.88, 0.11)]
# its pile: dense, a little longer (COLOR_0 R / G lifted in this 9.1 s view box (u0, v0, u1, v1),
# depth < m; a longer pile pushed the keyhole edge ~25 px into the opening at y 450-600)
T01_CHIN_PILE = ((0.3, 0.4, 0.52, 0.62), 2.25, 0.1, 0.15)
# the crease under the upper lobe, above the chin: lets the high key reach the chin's top
T01_CREASE = [(0.41, 0.04, 0.18), (0.425, 0.11, 0.16)]       # (v, inset, half-length m) in the 9.1 s view


def t01_wall(upto=None):
    """Base-surface outline of the wall's keyhole edge in the 9.1 s view (upto: last index of
    T01_WALL_SKY to use; T01_CHIN = down to the chin's lowest left point)."""
    return resample(inset(T01_WALL_SKY[:None if upto is None else upto + 1], +1, T01_FRINGE), 36.0)


def t01_column():
    return resample(inset(T01_COL_SKY, -1, T01_FRINGE), 36.0)


def t01_weight(z0, z1, z1_front, z0_front=None, y_front=(-1.32, -1.12), fade=0.08):
    """Clamp weight for the T01 views: world z in [z0, z1] everywhere, and up to z1_front in front
    of y_front (the T01 faces), so the hero masses further back keep their own silhouettes."""
    zb = zband(z0, z1, fade)
    zf = zband(z0 if z0_front is None else z0_front, z1_front, fade)
    return lambda P: np.maximum(zb(P), (1.0 - C.smoothstep(y_front[0], y_front[1], P[:, 1])) * zf(P))


# seen from cam_arch_main: the overhang (bottom of the big mass) stays above this line
ARCH_OVERHANG_POLY = [(-6.0, -6.0), (6.0, -6.0), (6.0, 0.05), (1.0, 0.06), (0.86, 0.08), (0.80, 0.14),
                      (0.775, 0.235), (0.755, 0.265), (0.735, 0.22), (0.71, 0.14), (0.68, 0.085),
                      (0.64, 0.058), (0.55, 0.05), (0.45, 0.055), (0.35, 0.06), (0.25, 0.065),
                      (0.15, 0.075), (0.05, 0.085), (-6.0, 0.10)]
ARCH_COLUMN_POLY = [(-6.0, -6.0), (6.0, -6.0), (6.0, 0.04), (0.3, 0.06), (0.15, 0.08), (0.08, 0.10),
                    (0.04, 0.14), (0.0, 0.20), (-6.0, 0.30)]


# extra depth (m, along cam_hero_main rays) of the crown's left end vs the frame-01 u of its crest:
# it recedes so that from the T01 poses it sits behind the wall's upper lobe, and at 8.4 s its
# crest reads as a long diagonal flank rising to the right instead of a dome's rounded corner
CROWN_RECEDE = [(0.395, 1.05), (0.43, 1.0), (0.50, 0.7), (0.59, 0.2), (0.68, 0.0)]


def crown_recede(u):
    """Extra depth (m, along cam_hero_main rays) of the crown's left end."""
    R = np.asarray(CROWN_RECEDE)
    return np.interp(np.asarray(u, float), R[:, 0], R[:, 1])


def crest_v(u):
    E = np.asarray(CROWN_EDGE)
    return float(np.interp(u, E[:, 0], E[:, 1]))


def _surface_depth(sdf, cam, u, v, d0, d1, step=0.005):
    """View depth of the first surface crossing along the ray through (u, v) of `cam`, or None."""
    ds = np.arange(d0, d1, step)
    P = np.array([S(cam, u, v, d) for d in ds])
    f = sdf.sample(P)
    k = np.nonzero((f[:-1] > 0) & (f[1:] <= 0))[0]
    if not len(k):
        return None
    i = int(k[0])
    return float(ds[i] + step * f[i] / max(f[i] - f[i + 1], 1e-9))


def build_crown_mass(coll, mat, N3):
    """moss_crown_mass: hero lower-right mound (its crown) + the big T01 mass + top-right arch overhang."""
    sdf = C.SDF((-1.35, -2.35, 1.95), (2.75, 0.65, 4.62), 0.02)
    # crown along the hero silhouette.  Its left end recedes along the same view rays (deeper and
    # lower, same outline from cam_hero_main/_out) so that from the T01 poses it sits behind the
    # wall's upper lobe instead of overhanging the keyhole
    E = np.asarray(CROWN_EDGE)
    d0 = np.linspace(1.95, 2.1, len(E))
    d1 = d0 + crown_recede(E[:, 0])
    r0 = np.array([0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.17, 0.17, 0.16, 0.16, 0.16, 0.17, 0.17, 0.18, 0.19, 0.2])
    edge_lumps(sdf, "cam_hero_main", CROWN_EDGE, depths=d1, radii=r0 * d1 / d0, side=-1, k=0.09,
               squash=(1.25, 1.1, 0.85), seed=11)
    # front face of the mound (seen v ~0.70-0.87): lumps whose tops stay below the crest
    for (u, d, r, extra) in [(0.45, 1.75, 0.15, 0.0), (0.53, 1.72, 0.17, 0.0), (0.62, 1.72, 0.18, 0.0),
                             (0.72, 1.75, 0.18, 0.0), (0.82, 1.8, 0.18, 0.0), (0.93, 1.85, 0.19, 0.0),
                             (1.05, 1.9, 0.2, 0.0), (0.50, 1.55, 0.17, 0.07), (0.66, 1.52, 0.2, 0.07),
                             (0.84, 1.58, 0.2, 0.07), (1.0, 1.62, 0.2, 0.07)]:
        d = d + float(crown_recede(u))
        vc = crest_v(u) + 0.035 + extra + 0.95 * r * dv(d)
        sdf.ellipsoid(S("cam_hero_main", u, vc, d), (r * 1.25, r, r * 0.95), k=0.1)
    # fill under the crown, joining the body
    for (c, r) in [((-0.45, -0.5, 3.95), (0.34, 0.42, 0.3)), ((0.15, -0.75, 4.05), (0.55, 0.5, 0.32)),
                   ((0.75, -0.55, 3.95), (0.6, 0.55, 0.35)), ((1.35, -0.4, 3.8), (0.55, 0.5, 0.4))]:
        sdf.ellipsoid(c, r, k=0.15)
    # main body of the T01 mass
    sdf.ellipsoid((0.55, -0.75, 3.40), (1.3, 0.8, 0.72), k=0.2)
    # T01 front wall the camera descends past (8.0-10.0 s), modelled in the master view (9.1 s):
    # rim lumps along the keyhole edge (base surface T01_FRINGE inside the traced plant tips:
    # top-left shoulder, upper lobe, lit shoulder, recess, chin) + body lumps behind it
    wall = t01_wall()
    edge_lumps(sdf, T01_VIEW, wall, depths=T01_RIM_DEPTH, radii=0.12, side=+1, k=0.08,
               squash=(1.0, 1.15, 1.1), seed=14)
    # second, larger row a little inside: the rim rolls over into the face instead of a step
    edge_lumps(sdf, T01_VIEW, resample(inset(T01_WALL_SKY, +1, T01_FRINGE + 0.05), 48.0), depths=2.02,
               radii=0.15, side=+1, k=0.09, squash=(1.0, 1.1, 1.0), seed=15)
    fill_region(sdf, T01_VIEW, right_of(wall), 2.05, 0.2, 0.2, 12, zrange=(2.85, 4.3),
                box=(0.2, -0.3, 1.15, 1.08), keep_in=1.1)
    # the lit chin (frame 03): a bulge standing out of the face between the upper lobe and the
    # recess; the 9.1 s clamp keeps its outline on the keyhole edge
    W = np.asarray(wall)
    for (v, inn, d, r) in T01_CHIN_LUMPS:
        u = float(np.interp(v, W[:, 1], W[:, 0])) + inn
        sdf.ellipsoid(S(T01_VIEW, u, v, d), (r * 1.3, r * 0.8, r * 0.7), k=0.08)
    # underside behind the chin: one convex belly from the chin (front, z ~2.9) back to the
    # hanging underside (z ~2.55) - seen from below at 9.8-10.0 s
    for (u, d, r) in [(0.36, 2.2, 0.2), (0.47, 2.22, 0.23), (0.59, 2.25, 0.25), (0.72, 2.3, 0.26),
                      (0.86, 2.35, 0.26), (1.0, 2.4, 0.26)]:
        c = S(T01_VIEW, u, 1.0, d)
        sdf.ellipsoid((c[0], c[1] + 0.12, 2.84), (r * 1.3, r * 1.25, 0.18), k=0.12)
    # right part, sitting on the right arch trunk
    for (c, r) in [((1.55, -0.55, 3.35), (0.6, 0.6, 0.55)), ((2.05, -0.15, 3.05), (0.5, 0.5, 0.45)),
                   ((2.25, 0.25, 2.75), (0.35, 0.4, 0.35))]:
        sdf.ellipsoid(c, r, k=0.15)
    # hanging underside (the dark overhang at the top of the arch frame)
    for (c, r) in [((-0.25, -1.3, 2.72), (0.3, 0.3, 0.2)), ((0.25, -1.2, 2.72), (0.36, 0.34, 0.22)),
                   ((0.8, -1.0, 2.70), (0.4, 0.38, 0.22)), ((1.35, -0.8, 2.66), (0.36, 0.36, 0.24))]:
        sdf.ellipsoid(c, r, k=0.12)
    # mossy drip where the overhang meets the right trunk (frame 04 top right)
    for (u, v, d, r) in [(0.70, 0.07, 4.0, 0.2), (0.725, 0.13, 4.05, 0.18), (0.745, 0.19, 4.1, 0.16),
                         (0.76, 0.235, 4.15, 0.13)]:
        sdf.ellipsoid(S("cam_arch_main", u, v, d), (r * 1.1, r, r * 1.2), k=0.1)
    # folds: short, tilted creases in the T01 front wall (placed in the 9.1 s view, on its face)
    # (broad, shallow troughs found on the surface along the view ray: soft undulation of the carpet,
    # not knife cuts; radii: along the crease, depth, across)
    for (u, v, r, ang) in [(0.66, 0.28, (0.28, 0.035, 0.06), 18.0), (0.86, 0.52, (0.24, 0.035, 0.05), -14.0),
                           (0.74, 0.82, (0.22, 0.035, 0.05), -25.0)]:
        d = _surface_depth(sdf, T01_VIEW, u, v, 1.4, 2.6)
        if d is None:
            continue
        C.log(f"T01 fold at ({u}, {v}): surface at view depth {d:.3f} m")
        c = S(T01_VIEW, u, v, d + 0.022)
        a = math.radians(ang)
        R = np.array([[math.cos(a), 0.0, -math.sin(a)], [0.0, 1.0, 0.0], [math.sin(a), 0.0, math.cos(a)]])
        sdf.carve_ellipsoid(c, r, rot=R, k=0.05)
    # the crease under the upper lobe, above the lit chin (frame 03): opens the key's way to the
    # chin's top
    for (v, inn, half) in T01_CREASE:
        u = float(np.interp(v, W[:, 1], W[:, 0])) + inn
        d = _surface_depth(sdf, T01_VIEW, u, v, 1.4, 2.6)
        if d is None:
            continue
        C.log(f"T01 crease at ({u:.3f}, {v}): surface at view depth {d:.3f} m")
        sdf.carve_ellipsoid(S(T01_VIEW, u, v, d + 0.03), (half, 0.08, 0.045), k=0.05)
    # lumps, then the visual-hull clamps, then fine lumps (organic silhouettes)
    sdf.noise(N3, 0.045, 2.6, octaves=3, offset=(3.1, 0.4, 1.7))
    clip(sdf, "cam_hero_main", HERO_CROWN_POLY)
    # (clamped down to the chin only: the belly under it stays round)
    clip(sdf, T01_VIEW, right_of(t01_wall(T01_CHIN)), weight=t01_weight(2.82, 4.12, 4.7, y_front=(-0.85, -0.7)))
    # 8.4 s: the diagonal flank - every depth of the crown part, but not the hero crest band
    hl, hq = pose_of("cam_hero_main")
    hr, hu, hf = C.cam_basis(hq)
    d1_edge = d1

    def crest_guard(P):
        rel = P - hl
        z = np.maximum(rel @ hf, 1e-3)
        u = 0.5 + 0.5 * (rel @ hr) / (z * TH)
        v = 0.5 - 0.5 * (rel @ hu) / (z * TV)
        below = v - np.interp(u, E[:, 0], E[:, 1])
        dd = np.abs(z - np.interp(u, E[:, 0], d1_edge))
        return (1.0 - C.smoothstep(0.06, 0.1, below)) * (1.0 - C.smoothstep(0.3, 0.45, dd))

    zb84 = zband(3.55, 4.75)
    clip(sdf, 8.4, T01_FLANK_84, weight=lambda P: zb84(P) * (1.0 - crest_guard(P)))
    clip(sdf, "cam_arch_main", ARCH_OVERHANG_POLY)
    sdf.noise(N3, 0.016, 7.5, octaves=2, offset=(9.3, 2.1, 0.2))
    obj = sdf.to_object("moss_crown_mass", coll, mat, target_tris=46000)
    return obj, sdf


def build_left_column(coll, mat, N3):
    """moss_left_column: hero left volume (neck + lower-left bulge) continuing down as the wavy
    T01 column and wrapping the top of the left arch trunk."""
    sdf = C.SDF((-4.0, -1.75, 2.3), (-0.55, 1.1, 4.95), 0.02)
    # hero: the narrow column of frame 01 / 02 - from the neck at the left frame edge (v ~0.59)
    # out to the lower-left bulge (u ~0.2, v ~0.83) and down behind the near band
    E = np.asarray(LEFT_EDGE)
    s_e = np.linspace(0.0, 1.0, len(E))
    edge_lumps(sdf, "cam_hero_main", LEFT_EDGE, depths=np.interp(s_e, [0.0, 1.0], [1.88, 1.68]),
               radii=np.interp(s_e, [0.0, 0.3, 0.7, 1.0], [0.1, 0.12, 0.16, 0.17]), side=-1,
               k=0.08, squash=(1.0, 1.1, 1.0), seed=21)
    # body of the column behind the silhouette, off-frame left; it meets the arm off-frame
    # (the neck at the left frame edge is closed by the plants only, as in frame 01)
    for (u, v, d, r) in [(-0.12, 0.50, 2.05, 0.22), (-0.10, 0.62, 1.9, 0.22), (-0.06, 0.75, 1.8, 0.24),
                         (0.0, 0.86, 1.7, 0.24), (-0.12, 0.9, 1.9, 0.28), (0.03, 0.98, 1.6, 0.22),
                         (-0.07, 0.56, 1.95, 0.14)]:
        sdf.ellipsoid(S("cam_hero_main", u, v, d), (r, r * 1.1, r), k=0.12)
    # T01: the wavy column left of the keyhole, modelled in the master view (9.1 s)
    col = t01_column()
    edge_lumps(sdf, T01_VIEW, col, depths=T01_COL_DEPTH, radii=0.12, side=-1, k=0.08,
               squash=(1.0, 1.15, 1.1), seed=24)
    fill_region(sdf, T01_VIEW, left_of(col), T01_COL_DEPTH - 0.05, 0.2, 0.2, 22, zrange=(2.75, 4.35),
                box=(-0.35, -0.3, 0.3, 1.1), keep_in=1.1)
    # lower end: bends left and wraps the top of the left arch trunk
    path = [(-1.45, -0.80, 3.30), (-1.62, -0.62, 3.0), (-1.95, -0.32, 2.86), (-2.35, 0.0, 2.88),
            (-2.75, 0.25, 3.02), (-3.15, 0.45, 3.2), (-3.4, 0.55, 3.45)]
    sdf.tube(path, [0.30, 0.32, 0.36, 0.42, 0.46, 0.46, 0.4], k=0.15)
    sdf.noise(N3, 0.04, 3.0, octaves=3, offset=(1.3, 5.4, 2.2))
    clip(sdf, "cam_hero_main", HERO_COLUMN_POLY)
    clip(sdf, T01_VIEW, left_of(col), weight=t01_weight(2.8, 4.05, 4.5))
    clip(sdf, "cam_arch_main", ARCH_COLUMN_POLY)
    sdf.noise(N3, 0.016, 8.0, octaves=2, offset=(4.3, 1.4, 7.2))
    obj = sdf.to_object("moss_left_column", coll, mat, target_tris=30000)
    return obj, sdf


def build_upper_arm(coll, mat, N3):
    """moss_hero_upper: the thick overgrown arm of frame 01 / details/01 - it hangs into the frame
    from above (left of the centre), runs diagonally down to the left edge and leaves the top-left
    corner empty; its right flank turns at a rounded nose into the straight lower edge where the
    combed pile hangs.  In 3D a limb ~0.25 m thick (screen) and ~0.35 m deep, 1.9-2.15 m from
    cam_hero_main, receding (deeper) towards the left edge and upwards out of the frame."""
    sdf = C.SDF((-2.35, -1.8, 4.25), (-0.3, 0.55, 6.15), 0.018)
    cam = "cam_hero_main"
    # lower / right silhouette: lumps rolling over the edge; the nose (u ~0.20-0.25) is the
    # nearest, biggest part, the lower edge recedes towards the left frame edge
    lr = np.asarray(ARM_LR[1:-1])
    s_lr = np.linspace(0.0, 1.0, len(lr))                  # 0 = left frame edge ... 1 = top edge
    d_lr = np.interp(s_lr, [0.0, 0.45, 0.55, 1.0], [2.08, 1.93, 1.9, 1.98])
    r_lr = np.interp(s_lr, [0.0, 0.4, 0.5, 0.6, 1.0], [0.085, 0.085, 0.11, 0.1, 0.085])
    edge_lumps(sdf, cam, [tuple(x) for x in lr], depths=d_lr, radii=r_lr, side=+1, k=0.06,
               squash=(1.0, 1.25, 1.0), seed=31)
    # upper-left silhouette (dark side, mostly seen against the black top band)
    ul = np.asarray(ARM_UL[1:-1])
    s_ul = np.linspace(0.0, 1.0, len(ul))
    edge_lumps(sdf, cam, [tuple(x) for x in ul], depths=np.interp(s_ul, [0.0, 1.0], [2.12, 2.02]),
               radii=0.08, side=-1, k=0.06, squash=(1.0, 1.25, 1.0), seed=32)
    # body: lumps filling the band (front, rounded towards the camera along the middle), and a
    # back layer that gives the limb its depth
    fill_region(sdf, cam, HERO_UPPER_POLY, 1.97, 0.1, 0.09, 33, squash=(1.05, 1.3, 1.0), k=0.08,
                box=(-0.08, -0.1, 0.34, 0.56), keep_in=0.9)
    fill_region(sdf, cam, HERO_UPPER_POLY, 2.15, 0.13, 0.12, 34, squash=(1.0, 1.2, 1.0), k=0.1,
                box=(-0.1, -0.15, 0.34, 0.56), keep_in=1.0)
    # continuation out of the frame: up (rising and receding) and out over the left edge
    for (u, v, d, r) in [(0.255, -0.08, 2.02, 0.12), (0.26, -0.2, 2.1, 0.13), (0.27, -0.34, 2.2, 0.14),
                         (0.28, -0.5, 2.3, 0.15), (-0.03, 0.39, 2.12, 0.13), (-0.08, 0.42, 2.2, 0.15),
                         (-0.14, 0.46, 2.28, 0.17), (-0.2, 0.5, 2.35, 0.18)]:
        sdf.ellipsoid(S(cam, u, v, d), (r, r * 1.3, r), k=0.1)
    sdf.noise(N3, 0.03, 3.4, octaves=3, offset=(7.1, 3.3, 0.9))
    clip(sdf, cam, HERO_UPPER_POLY)
    sdf.noise(N3, 0.012, 9.0, octaves=2, offset=(2.2, 8.1, 4.4))
    obj = sdf.to_object("moss_hero_upper", coll, mat, target_tris=16000)
    return obj, sdf


def build_hero_fg(coll, mat, N3):
    """moss_hero_near__fg: dark out-of-focus near foliage along the bottom of frame 01."""
    sdf = C.SDF((-1.55, -2.2, 3.95), (0.5, -1.3, 4.6), 0.02)
    rng = np.random.default_rng(41)
    for u in np.linspace(-0.12, 1.12, 9):
        d = 0.95 + 0.12 * rng.uniform(-1, 1)
        r = 0.13 + 0.03 * rng.uniform(-1, 1)
        v_top = 0.89 + 0.015 * rng.uniform(-1, 1)
        c = S("cam_hero_main", u, v_top + 0.7 * r * dv(d), d)
        sdf.ellipsoid(c, (r * 1.4, r, r * 0.7), k=0.08)     # flat: stays out of the T01 frames
    # the light gap between the left bulge and the crown's left end: the band starts higher there
    for (u, v_top, d, r) in [(0.19, 0.80, 1.1, 0.12), (0.26, 0.795, 1.12, 0.13), (0.33, 0.80, 1.1, 0.12)]:
        sdf.ellipsoid(S("cam_hero_main", u, v_top + 0.7 * r * dv(d), d), (r * 1.4, r, r * 0.7), k=0.08)
    sdf.noise(N3, 0.03, 4.0, octaves=3, offset=(0.7, 0.7, 0.7))
    # keep it out of the T01 frames (it is the hero's near foreground only)
    # (frozen at the revision-1 cam_hero_p1 pose; the engine hides this object from 8.7 s anyway)
    sdf.clip_view(np.array((-0.40, -2.95, 3.85)), C.cam_quat(-2.0, 0.0, 0.0),
                  [(-0.3, -0.02), (1.3, -0.02), (1.3, 1.3), (-0.3, 1.3)], margin=0.0, k=0.03, invert=True)
    obj = sdf.to_object("moss_hero_near__fg", coll, mat, target_tris=5000)
    return obj, sdf


# ------------------------------------------------------------------------------------------
# arch (S02) - anchored to cam_arch_main
# ------------------------------------------------------------------------------------------

# left trunk: base bottom-left, rising and bending up-left out of the frame's top-left corner.
# Centred on the bare oval (frame 04: oval u 0.09-0.22, right silhouette incl. moss 0.29 / 0.31 /
# 0.34 at v 0.5 / 0.6 / 0.7, rounded mossy shoulder at v ~0.30-0.34 for u 0.1-0.2)
LEFT_TRUNK_UVD = [(0.255, 1.12, 4.3), (0.215, 0.95, 4.35), (0.18, 0.80, 4.4), (0.145, 0.70, 4.45),
                  (0.115, 0.605, 4.5), (0.085, 0.51, 4.55), (0.025, 0.44, 4.6), (-0.05, 0.37, 4.65),
                  (-0.12, 0.28, 4.72), (-0.18, 0.185, 4.8), (-0.24, 0.08, 4.88)]
LEFT_TRUNK_R = [0.70, 0.62, 0.57, 0.55, 0.53, 0.52, 0.50, 0.48, 0.46, 0.44, 0.42, 0.42]
LEFT_TRUNK_END = (-3.2, 0.42, 3.15)          # inside moss_left_column

# right trunk: wide mossy upper part from the top-right, narrow bare stem down the right edge
RIGHT_TRUNK_UVD = [(1.08, 1.15, 4.7), (1.045, 0.90, 4.68), (1.035, 0.78, 4.66), (1.018, 0.69, 4.64),
                   (0.995, 0.60, 4.6), (0.955, 0.50, 4.5), (0.925, 0.40, 4.4), (0.905, 0.28, 4.3),
                   (0.90, 0.14, 4.22), (0.90, 0.0, 4.15)]
RIGHT_TRUNK_R = [0.36, 0.30, 0.28, 0.28, 0.30, 0.37, 0.44, 0.48, 0.50, 0.50, 0.48]
RIGHT_TRUNK_END = (1.85, -0.45, 3.2)         # inside moss_crown_mass (smooth continuation upwards)

# bark windows (frame 04, details/03): ragged cushion outlines (contour shifted by ~+-3.5 cm of
# 11 / 5.5 cm noise) and a rolled lip overhanging the bark by 2-5 cm
ARCH_RAG = (0.035, 9.0)
ARCH_LIP = (0.02, 0.05)

# rev 3: the bend sits ~40 px further left (frame 04: the leg's right edge at x ~1100) and the leg
# runs on down-right below frame 04 - from cam_arch_out it is the left side of the exit window,
# within ~10 px of frame 05 from y 510 to 830 px (rev 2: ~30-60 px right of it, ending at y ~720)
LOW_UVD = [(0.36, 0.92, 4.3), (0.44, 0.915, 4.35), (0.52, 0.905, 4.4), (0.61, 0.895, 4.45),
           (0.68, 0.90, 4.45), (0.72, 0.93, 4.4), (0.74, 1.0, 4.35), (0.765, 1.09, 4.3), (0.80, 1.17, 4.25),
           (0.835, 1.25, 4.2), (0.865, 1.32, 4.15)]
LOW_R = [0.22, 0.20, 0.19, 0.18, 0.18, 0.19, 0.21, 0.23, 0.25, 0.27, 0.28]


def _ellipse_window(px, py, cx, cy, ha, hb, ang_deg, noise=None, soft=(0.85, 1.15)):
    a = math.radians(ang_deg)
    dx, dy = px - cx, py - cy
    pa = dx * math.cos(a) + dy * math.sin(a)
    pb = -dx * math.sin(a) + dy * math.cos(a)
    e = np.sqrt((pa / ha) ** 2 + (pb / hb) ** 2)
    if noise is not None:
        e = e + noise
    return 1.0 - C.smoothstep(soft[0], soft[1], e)


def _arch_screen(P):
    c = CAMS["cam_arch_main"]
    u, v, z = C.project_to_screen(c["loc"], cam_q("cam_arch_main"), P.reshape(-1, 3))
    return u.reshape(P.shape[:-1]) * C.FRAME_W, v.reshape(P.shape[:-1]) * C.FRAME_H, z.reshape(P.shape[:-1])


def _facing(sw, cam="cam_arch_main"):
    c = np.asarray(CAMS[cam]["loc"], float)
    v = C.normalize(sw.smooth - c)
    return -np.einsum("...k,...k->...", sw.nsmooth, v)


def _locate(sw, px, py):
    """Grid (ring, seg) of the camera-facing trunk vertex nearest to pixel (px, py)."""
    X, Y, _ = _arch_screen(sw.smooth)
    f = _facing(sw)
    d = (X - px) ** 2 + (Y - py) ** 2 + (f < 0.2) * 1e9
    i, j = np.unravel_index(np.argmin(d), d.shape)
    return i, j


def _egg_window(px, py, cx, cy, ha, hb, ang_deg, egg=0.0, noise=None, soft=(0.85, 1.15)):
    """Elliptic screen window (1 inside); egg > 0 widens it towards +long axis."""
    a = math.radians(ang_deg)
    dx, dy = px - cx, py - cy
    pa = dx * math.cos(a) + dy * math.sin(a)
    pb = -dx * math.sin(a) + dy * math.cos(a)
    hb_e = hb * (1.0 + egg * np.clip(pa / ha, -1.0, 1.0))
    e = np.sqrt((pa / ha) ** 2 + (pb / hb_e) ** 2)
    if noise is not None:
        e = e + noise
    return 1.0 - C.smoothstep(soft[0], soft[1], e)


def _grid_lerp(A, fi, fj, wrap_m):
    """Bilinear sample of a trunk-grid array A (N, M[, k]) at fractional ring indices fi (n,) x
    segment indices fj (m,) (wrapped around M when wrap_m, else clamped to M - 1 + 1 columns)."""
    A = np.asarray(A, float)
    N = A.shape[0]
    i0 = np.minimum(np.floor(fi).astype(int), N - 2)
    ti = (fi - i0)[:, None]
    jf = np.floor(fj).astype(int)
    tj = (fj - jf)[None, :]
    if wrap_m:
        j0 = jf % wrap_m
        j1 = (j0 + 1) % wrap_m
    else:
        j0 = np.minimum(jf, A.shape[1] - 2)
        j1 = j0 + 1
        tj = (fj - j0)[None, :]
    if A.ndim == 3:
        ti, tj = ti[..., None], tj[..., None]
    a0 = A[i0][:, j0] * (1.0 - tj) + A[i0][:, j1] * tj
    a1 = A[i0 + 1][:, j0] * (1.0 - tj) + A[i0 + 1][:, j1] * tj
    return a0 * (1.0 - ti) + a1 * ti


def _grid_grad(F, P):
    """Metric surface gradient (N, M, 3; units of F per metre) of a trunk-grid field F (N, M) on the
    grid positions P (N, M, 3): rings along axis 0 (clamped), wrapped around axis 1."""
    F = np.asarray(F, float)
    dFi = np.empty_like(F)
    dPi = np.empty_like(P)
    dFi[1:-1] = 0.5 * (F[2:] - F[:-2])
    dPi[1:-1] = 0.5 * (P[2:] - P[:-2])
    dFi[0], dFi[-1] = F[1] - F[0], F[-1] - F[-2]
    dPi[0], dPi[-1] = P[1] - P[0], P[-1] - P[-2]
    dFj = 0.5 * (np.roll(F, -1, 1) - np.roll(F, 1, 1))
    dPj = 0.5 * (np.roll(P, -1, 1) - np.roll(P, 1, 1))
    # g = a dPi + b dPj with g . dPi = dFi and g . dPj = dFj (first fundamental form)
    E = (dPi * dPi).sum(-1)
    Fm = (dPi * dPj).sum(-1)
    G = (dPj * dPj).sum(-1)
    det = np.maximum(E * G - Fm * Fm, 1e-18)
    a = (G * dFi - Fm * dFj) / det
    b = (E * dFj - Fm * dFi) / det
    return a[..., None] * dPi + b[..., None] * dPj


def lumpy_cushions(name, coll, sw, mask, *, thickness, material=None, seed=0, up=1.0,
                   bump_r=(0.025, 0.075), bump_h=0.42, fill=0.6, border=(0.28, 0.62), edge_exp=0.75,
                   edge_noise=0.12, tuck=0.012, keep_above=None, sag=0.0, flatten=0.15, rag=None,
                   lip=None):
    """Thick, lumpy moss cushions hugging a swept trunk where `mask` (N, M) is high.

    The cushion is a sheet over the bark (clear of its highest ridge) of `fill` x thickness, topped
    by a field of rounded pillows (radius bump_r = 2.5-7.5 cm by default, height ~bump_h x r) on two
    jittered lattices (large, then small in between), combined by max so the top reads as packed
    cushion moss.  Its border is the mask plus noise (irregular), where the sheet rolls over a short
    rounded edge and tucks `tuck` m under the bark relief.  The trunk grid is resampled `up` times
    (bilinear) so the pillows are resolved.  The bark relief under the cushion (never seen) is
    flattened to `flatten` x, so the cushion meets an even surface at its border (a clean rolled
    edge, not a comb where the sheet cuts the ridges).
    rag (m, freq): ragged outlines - the border is shifted by fbm noise of `freq` (1/m) scaled to about
    +-m metres along the surface (the noise is multiplied by the mask's metric gradient, so the
    contour moves by the same distance on sharp and on soft mask edges).
    lip (min, max): a rolled-over lip - the lower part of the border band (under the cushion's rounded
    edge) is pulled back under the cushion by min..max metres along the surface, so the edge overhangs
    the bark it grows from (a dark undercut, the AO of the bark under it, plants hanging over it);
    the visible front of the edge stays on the mask contour (the bark windows keep their framing).
    Returns (obj, info) like common.moss_cushions; info["bark_dist"] (N, M) is the signed distance
    (m) of the bark grid from the lip front (+ on the bare bark), for the bark's COLOR_0."""
    rng = np.random.default_rng(seed)
    noise = C.Noise3(seed + 307)
    N, M = sw.N, sw.M

    def ragged(mk, P):
        out = mk + edge_noise * noise.fbm(P * 3.5, 2)          # irregular, but not spiky, borders
        if rag is not None:
            out = out + rag[0] * np.linalg.norm(_grid_grad(mk, P), axis=-1) * noise.fbm(P * rag[1] + 23.0, 2)
        return out

    mn0 = ragged(np.asarray(mask, float), sw.smooth)
    if flatten is not None and sw.relief is not None:
        kf = 1.0 - (1.0 - flatten) * C.smoothstep(border[0] - 0.16, border[0] + 0.02, mn0)
        drop = sw.relief * (1.0 - kf)
        sw.relief = sw.relief * kf
        sw.pos = sw.pos - sw.nsmooth * drop[..., None]
        if sw.obj is not None:
            me_b = sw.obj.data
            co_b = np.empty(len(me_b.vertices) * 3, np.float32)
            me_b.vertices.foreach_get("co", co_b)
            co_b = co_b.reshape(-1, 3)
            co_b[:N * M] = sw.pos.reshape(-1, 3)
            me_b.vertices.foreach_set("co", co_b.ravel())
            me_b.update()
    nr = int(round((N - 1) * up)) + 1
    nc = int(round(M * up))
    fi = np.linspace(0.0, N - 1, nr)
    fj = np.arange(nc) * (M / nc)
    base = _grid_lerp(sw.smooth, fi, fj, M)
    nrm = C.normalize(_grid_lerp(sw.nsmooth, fi, fj, M))
    rel = _grid_lerp(sw.relief, fi, fj, M) if sw.relief is not None else np.zeros((nr, nc))
    m = _grid_lerp(mask, fi, fj, M)
    T = _grid_lerp(np.broadcast_to(np.asarray(thickness, float), (N, M)), fi, fj, M)
    uvg = _grid_lerp(sw.uv, fi, np.append(fj, float(M)), 0)          # (nr, nc + 1, 2), seam column
    mn = ragged(m, base)
    prof = C.smoothstep(border[0], border[1], mn) ** edge_exp
    # pillow centres: jittered lattices in (arc length, arc around), large first, then small
    s_ring = np.interp(fi, np.arange(N), sw.s)
    rad = np.interp(fi, np.arange(N), sw.radius)
    cen, rr, hf_l = [], [], []
    for (lo, hi), sp, hf in (((0.6 * bump_r[1], bump_r[1]), 2.0 * bump_r[1], 1.0),
                             ((bump_r[0], 0.55 * bump_r[1]), 1.2 * bump_r[1], 0.65)):
        for k, s0 in enumerate(np.arange(0.0, sw.length, sp)):
            i_f = float(np.interp(s0 + sp * 0.3 * rng.uniform(-1, 1), s_ring, fi))
            r_here = float(np.interp(i_f, fi, rad))
            ncol = max(3, int(round(2 * math.pi * r_here / sp)))
            for j in range(ncol):
                j_f = ((j + 0.5 * (k % 2) + 0.45 * rng.uniform(-1, 1)) / ncol * M) % M
                cen.append((i_f, j_f))
                rr.append(rng.uniform(lo, hi))
                hf_l.append(hf * rng.uniform(0.6, 1.25))
    cen = np.asarray(cen)
    rr = np.asarray(rr)
    hf_l = np.asarray(hf_l)
    ci = np.clip(cen[:, 0], 0, N - 1 - 1e-6)
    P = np.array([_grid_lerp(sw.smooth, np.array([a]), np.array([b]), M)[0, 0] for a, b in cen])
    mk = np.array([_grid_lerp(mask, np.array([a]), np.array([b]), M)[0, 0] for a, b in zip(ci, cen[:, 1])])
    sel = mk > border[0] - 0.05
    P, rr = P[sel], rr[sel]
    hk = bump_h * rr * hf_l[sel]
    Q = base.reshape(-1, 3)
    h = np.zeros(len(Q))
    p2 = (P * P).sum(1)
    for a in range(0, len(Q), 2000):
        q = Q[a:a + 2000]
        x2 = ((q * q).sum(1)[:, None] + p2[None, :] - 2.0 * q @ P.T) / (rr[None, :] ** 2)
        h[a:a + 2000] = (np.clip(1.0 - x2, 0.0, 1.0) ** 0.7 * hk[None, :]).max(1)
    h = h.reshape(nr, nc) + T * 0.3 * noise.fbm(base * 2.5 + 11.0, 2)      # slow swell under the pillows
    inside = prof > 0.5
    clear = max(float(np.percentile(rel[inside], 99)) if inside.any() else sw.relief_max, 0.0)
    off = (rel - tuck) * (1.0 - prof) + (clear + T * fill + h) * prof
    # rise out of the (flattened) bark over the first part of the border band: the cushion edge is
    # a rolled lip that follows the mask contour smoothly across several grid cells (a lip switched
    # on within one cell made the contour of a binary field: a stair-step along the grid)
    off += (tuck + 0.004) * C.smoothstep(0.0, 0.45, prof) * (1.0 - prof)
    pos = base + nrm * off[..., None]
    if sag:
        pos[..., 2] -= sag * T * prof * np.clip(-nrm[..., 2], 0, 1)
    if lip is not None:
        # rolled lip: the lower border band (prof < 0.5, the rounded edge's underside and the tucked
        # skirt) is pulled back under the cushion, the front (prof ~0.45) stays where it was
        g3 = _grid_grad(mn, base)
        gl = np.linalg.norm(g3, axis=-1)
        outward = -g3 / np.maximum(gl, 1e-9)[..., None]
        L = lip[0] + (lip[1] - lip[0]) * C.smoothstep(-0.25, 0.25, noise.fbm(base * 6.0 + 41.0, 2))
        L = L * C.smoothstep(1.5, 4.5, gl)          # real edges only (band narrower than ~8-23 cm)
        pos = pos - outward * (L * (1.0 - C.smoothstep(0.12, 0.5, prof)))[..., None]
    vid = np.arange(nr * nc).reshape(nr, nc)
    i0, j0 = np.meshgrid(np.arange(nr - 1), np.arange(nc), indexing="ij")
    j1 = (j0 + 1) % nc
    quads = np.stack([vid[i0, j0], vid[i0, j1], vid[i0 + 1, j1], vid[i0 + 1, j0]], -1).reshape(-1, 4)
    # one extra cell around the cushion: its outer vertices are tucked under the bark, so the mesh
    # boundary never shows (a steep mask would otherwise leave a raised stair-step edge)
    k_above = border[0] - 0.02 if keep_above is None else keep_above
    keep = (mn.reshape(-1)[quads] > k_above).any(1)
    quads = quads[keep]
    if not len(quads):
        return None, None
    a_i, a_j = np.divmod(quads[:, 0], nc)
    c_i = np.divmod(quads[:, 2], nc)[0]
    uv_l = np.stack([uvg[a_i, a_j], uvg[a_i, a_j + 1], uvg[c_i, a_j + 1], uvg[c_i, a_j]], 1).reshape(-1, 2)
    used = np.unique(quads)
    remap = -np.ones(nr * nc, np.int64)
    remap[used] = np.arange(len(used))
    me = C.mesh_from_numpy(name, pos.reshape(-1, 3)[used], remap[quads], smooth=True)
    C.set_uvs(me, uv_l)
    obj = C.object_from_mesh(name, me, coll, material)
    # signed distance of the bark grid from the lip front (the contour where prof = 0.45)
    xs = np.linspace(0.0, 1.0, 2001)
    c_front = border[0] + (border[1] - border[0]) * float(np.interp(0.45, C.smoothstep(0.0, 1.0, xs) ** edge_exp, xs))
    g0 = np.linalg.norm(_grid_grad(mn0, sw.smooth), axis=-1)
    bark_dist = np.clip((c_front - mn0) / np.maximum(g0, 1e-6), -0.5, 0.5)
    info = dict(offset=off.reshape(-1)[used], shape=prof.reshape(-1)[used], thickness=T.reshape(-1)[used],
                relief=rel.reshape(-1)[used], clear=clear, used=used, bumps=int(len(rr)), bark_dist=bark_dist,
                lip=lip)
    return obj, info


def moss_coverage(sw, mask, level=0.4):
    """Area fraction of the trunk surface under moss (mask above `level`)."""
    w = np.broadcast_to(sw.radius[:, None], mask.shape)
    return float((w * (mask > level)).sum() / w.sum())


def build_left_trunk(coll, mat_wood, mat_moss, N3):
    ctrl = [S("cam_arch_main", *p) for p in LEFT_TRUNK_UVD] + [np.array(LEFT_TRUNK_END)]
    common = dict(seed=SEED + 1, ring_spacing=0.036, segments=112, twist=0.9, twist_noise=0.15,
                  profile=dict(lobes=[(2, 0.10, 0.4), (3, 0.05, 1.3), (5, 0.018, 0.2)], var=0.6, drift=0.5),
                  radius_noise=0.05, lump=0.035, caps=("flat", "flat"))
    probe = C.sweep_trunk("wood_arch_left", coll, ctrl, LEFT_TRUNK_R, build_object=False, **common)
    # knot under the exposed oval so the grain flows around it (details/03)
    i, j = _locate(probe, 0.165 * C.FRAME_W, 0.70 * C.FRAME_H)
    k_s = probe.s[i] / probe.length
    k_th = float(probe.grain[i, j])
    sw = C.sweep_trunk("wood_arch_left", coll, ctrl, LEFT_TRUNK_R, material=mat_wood,
                       ridges=dict(count=19, depth=0.022, groove=0.22, meander=0.45, layers=2, var=0.6),
                       folds=[(k_th + 2.2, 0.28, 0.10, 0.0, 0.35), (k_th - 2.0, 0.22, 0.08, 0.0, 0.25)],
                       knots=[(k_s + 0.035, k_th + 0.15, 0.12, 0.05, 0.35), (k_s - 0.05, k_th - 0.35, 0.08, 0.03, 0.25)],
                       **common)
    # ---- moss mask (frame 04, details/03): thick cushions over the top and down the outer
    # (opening-facing) side, a mossy band left of the oval; bare: the big oval of flowing bark, the
    # strip along the left frame edge, the inner side and the back (a few noise patches only)
    X, Y, _ = _arch_screen(sw.smooth)
    f = _facing(sw)
    n = sw.nsmooth
    big = N3.fbm(sw.smooth * 1.4 + 3.0, 3)
    small = N3.fbm(sw.smooth * 5.0 + 9.0, 2)
    front = C.smoothstep(-0.05, 0.3, f)
    top = C.smoothstep(0.05, 0.55, n[..., 2])
    outer = C.smoothstep(-0.1, 0.45, n[..., 0])
    m = 0.08 + 0.3 * big + 0.1 * small + 0.62 * front + 0.4 * top + 0.3 * outer
    oval = _egg_window(X, Y, 0.165 * 1440, 0.697 * 1020, 150, 80, 76.0, egg=0.22, noise=0.12 * small,
                       soft=(0.78, 1.22))
    strip = (1.0 - C.smoothstep(50, 105, X)) * C.smoothstep(150, 210, Y) * (1.0 - C.smoothstep(360, 420, Y))
    m -= 1.6 * front * oval + 1.2 * front * strip
    # inner side (under the bend) and back: bare apart from patches
    m -= 0.55 * (1.0 - front) * (1.0 - top) * (1.0 - 0.6 * outer) * C.smoothstep(-0.3, 0.3, 0.45 - big)
    m = C.blur_grid(np.clip(m, 0, 1), 3)     # soft, non grid-aligned cushion borders
    thick = 0.042 * (0.8 + 0.45 * C.smoothstep(-0.4, 0.6, big)) * (sw.radius / 0.6)[:, None]
    cush, info = lumpy_cushions("moss_arch_left", coll, sw, m, thickness=thick, material=mat_moss,
                                seed=SEED + 2, up=1.5, bump_r=(0.025, 0.075), bump_h=0.42, sag=0.25,
                                rag=ARCH_RAG, lip=ARCH_LIP)
    C.log(f"moss_arch_left: coverage {moss_coverage(sw, m):.2f} of the trunk, {info['bumps']} pillows")
    return sw, m, cush, info


def build_right_trunk(coll, mat_wood, mat_moss, N3):
    ctrl = [S("cam_arch_main", *p) for p in RIGHT_TRUNK_UVD] + [np.array(RIGHT_TRUNK_END)]
    sw = C.sweep_trunk("wood_arch_right", coll, ctrl, RIGHT_TRUNK_R, material=mat_wood, seed=SEED + 3,
                       ring_spacing=0.038, segments=96, twist=-0.75, twist_noise=0.12,
                       profile=dict(lobes=[(2, 0.13, 1.1), (3, 0.05, 2.0), (4, 0.02, 0.7)], var=0.6, drift=0.6),
                       ridges=dict(count=17, depth=0.024, groove=0.2, meander=0.5, layers=2, var=0.6),
                       folds=[(0.8, 0.25, 0.09, 0.0, 0.3)], radius_noise=0.06, lump=0.04, caps=("flat", "flat"))
    X, Y, _ = _arch_screen(sw.smooth)
    f = _facing(sw)
    n = sw.nsmooth
    big = N3.fbm(sw.smooth * 1.5 + 13.0, 3)
    small = N3.fbm(sw.smooth * 5.0 + 19.0, 2)
    left = np.clip(-n[..., 0], 0, 1)
    right = np.clip(n[..., 0], 0, 1)
    vv = Y / 1020
    m = 0.52 + 0.4 * big + 0.12 * small + 0.55 * left * C.smoothstep(0.08, 0.18, vv) \
        * (1.0 - C.smoothstep(0.6, 0.68, vv)) + 0.2 * n[..., 2]
    # top: merges into the overhang
    m += 0.6 * (1.0 - C.smoothstep(0.03, 0.11, vv))
    # lower stem: moss on its front-right, bare dark wood on the left edge
    m += (0.3 * right + 0.15 * np.clip(f, 0, 1) - 0.3 * left) * C.smoothstep(0.6, 0.68, vv) \
        * (1.0 - C.smoothstep(0.82, 0.9, vv))
    # bare window of flowing bark: it scales the clipped mask, so the cushion border spans the
    # whole soft edge of the window (a subtracted window left a one-cell, stair-stepped border)
    # (rev 3: a narrower soft edge and finer noise - a ragged outline with a real lip, not a soft
    # painted oval; frame 04)
    fine = N3.fbm(sw.smooth * 11.0 + 7.0, 2)
    bare = _ellipse_window(X, Y, 1330, 280, 192, 110, 52.0, noise=0.2 * small + 0.12 * fine, soft=(0.84, 1.16))
    bare = C.blur_grid(C.smoothstep(0.05, 0.3, f) * bare, 6)
    m = np.clip(m, 0.0, 1.0) * (1.0 - bare)
    # back (never seen from the arch cameras): mostly bare, a few patches
    m -= 0.5 * (1.0 - C.smoothstep(-0.05, 0.3, f)) * (1.0 - C.smoothstep(0.05, 0.55, n[..., 2])) \
        * C.smoothstep(-0.3, 0.3, 0.45 - big)
    m = C.blur_grid(np.clip(m, 0, 1), 4)     # soft, non grid-aligned cushion borders
    thick = 0.04 * (0.7 + 0.6 * C.smoothstep(-0.4, 0.6, big))
    cush, info = lumpy_cushions("moss_arch_right", coll, sw, m, thickness=thick, material=mat_moss,
                                seed=SEED + 4, up=1.0, bump_r=(0.02, 0.055), bump_h=0.4, sag=0.4,
                                border=(0.22, 0.7), rag=ARCH_RAG, lip=ARCH_LIP)
    C.log(f"moss_arch_right: coverage {moss_coverage(sw, m):.2f} of the trunk, {info['bumps']} pillows")
    return sw, m, cush, info


def build_low_branch(coll, mat_wood, mat_moss, N3):
    ctrl = [S("cam_arch_main", *p) for p in LOW_UVD]
    sw = C.sweep_trunk("wood_arch_low", coll, ctrl, LOW_R, material=mat_wood, seed=SEED + 5,
                       ring_spacing=0.032, segments=64, twist=0.5, twist_noise=0.1,
                       profile=dict(lobes=[(2, 0.16, 0.0), (3, 0.04, 0.9)], var=0.5, drift=0.4),
                       ridges=dict(count=11, depth=0.012, groove=0.22, meander=0.4, layers=2, var=0.5),
                       radius_noise=0.05, lump=0.03, caps=("flat", "flat"))
    X, Y, _ = _arch_screen(sw.smooth)
    n = sw.nsmooth
    small = N3.fbm(sw.smooth * 6.0 + 29.0, 2)
    tuft = C.smoothstep(0.545 * 1440, 0.585 * 1440, X) * (1.0 - C.smoothstep(0.70 * 1440, 0.735 * 1440, X))
    m = (0.2 + 0.25 * small + 1.0 * tuft * C.smoothstep(0.1, 0.5, n[..., 2]) +
         0.35 * C.smoothstep(0.3, 0.8, n[..., 2]) * C.smoothstep(0.0, 0.5, N3.fbm(sw.smooth * 2.0 + 41, 2)))
    m = C.blur_grid(np.clip(m, 0, 1), 4)     # soft, non grid-aligned cushion borders
    cush, info = lumpy_cushions("moss_arch_low", coll, sw, m, thickness=0.022, material=mat_moss,
                                seed=SEED + 6, up=1.5, bump_r=(0.015, 0.04), bump_h=0.4)
    C.log(f"moss_arch_low: coverage {moss_coverage(sw, m):.2f} of the branch, {info['bumps']} pillows")
    return sw, m, cush, info


# seen from cam_arch_out (frame 05): the window right of the low branch's descending leg and left of
# the right stem stays open (ground / near forms are kept out of it).  Rev 3: a wide V down to
# y ~880 px like frame 05 (rev 2 left a narrow vertical slit between the near band and its right
# part); the left side cuts the near band ~30 px inside the low branch's leg, which forms the
# window's left side as in frame 05; the right side sits ~30 px outside the reference edge for the
# plant fringe and the soft cut, and its top leaves frame 04's bottom-right corner open (frame 04:
# haze down to the bottom edge there)
ARCH_OUT_WINDOW = [(x / C.FRAME_W, y / C.FRAME_H) for (x, y) in [
    (1123, 306), (1339, 255), (1375, 400), (1460, 430), (1460, 560), (1450, 600), (1440, 640), (1428, 700),
    (1392, 800), (1335, 885), (1272, 960), (1238, 860), (1200, 785), (1155, 704), (1100, 612), (1068, 510),
    (1060, 439)]]
# top of the near band in the cam_arch_main view, before its hummocks (which add ~30-50 px): frame 04's
# blurred band dips under the left trunk's foot (x ~200-650 px) and runs just under the low branch and
# its tuft (the tuft stays whole); an uneven line - rev 2 had a straight top at v ~0.885, which at
# 15.2 s (seen against the trunk base, the same line ~455 px higher) read as a horizontal shelf
ARCH_FG_TOP = [(x / C.FRAME_W, (y + 40) / C.FRAME_H) for (x, y) in [
    (-800, 880), (-200, 885), (0, 888), (70, 882), (140, 893), (210, 905), (280, 916), (350, 912), (420, 925), (490, 928),
    (560, 918), (620, 910), (680, 902), (740, 904), (800, 906), (870, 900), (940, 904), (1010, 899),
    (1070, 903), (1120, 915), (1180, 940), (1260, 965), (1700, 1000), (2300, 1000)]]


def hummocks(sdf, seed, r_range=(0.05, 0.2), h_ratio=(0.28, 0.45), pack=0.85, exp=0.6, weight=None,
             n_cand=40000, passes=3):
    """Packed moss hummocks on an SDF mass (ground and near forms of the arch).

    Pillows of radius r_range are seeded on the current surface by dart throwing (large ones first,
    smaller ones in the gaps), each `h_ratio` x r high, combined by max (creases between them) and
    pushed out of the surface.  Turns a smooth clay blob into lumpy cushion moss.
    weight(P) -> [0, 1] scales the pillow heights by region.  Returns the number of pillows."""
    rng = np.random.default_rng(seed)
    vx = sdf.voxel
    idx = np.nonzero(np.abs(sdf.d) < 0.6 * vx)
    P = sdf.origin + np.stack(idx, -1) * vx
    P = P[rng.choice(len(P), min(n_cand, len(P)), replace=False)]
    r0, r1 = r_range
    edges = np.geomspace(r1, r0, passes + 1)
    cell = 2.0 * r1
    grid = {}
    cen, rad = [], []
    for k in range(passes):
        lo, hi = edges[k + 1], edges[k]
        rs = rng.uniform(lo, hi, len(P))
        for p, r in zip(P.tolist(), rs.tolist()):
            key = (int(p[0] // cell), int(p[1] // cell), int(p[2] // cell))
            ok = True
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for j in grid.get((key[0] + dx, key[1] + dy, key[2] + dz), ()):
                            c = cen[j]
                            lim = pack * (r + rad[j])
                            if (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 + (p[2] - c[2]) ** 2 < lim * lim:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                grid.setdefault(key, []).append(len(cen))
                cen.append(p)
                rad.append(r)
    cen = np.asarray(cen, float)
    rad = np.asarray(rad, float)
    hk = rad * rng.uniform(h_ratio[0], h_ratio[1], len(rad))
    if weight is not None:
        hk = hk * np.clip(np.asarray(weight(cen), float), 0.0, 1.0)
    bump = np.zeros(sdf.shape, np.float32)
    for c, r, h in zip(cen, rad, hk):
        if h <= 0.25 * vx:
            continue
        sl, Q = sdf._box(c - r, c + r)
        if sl is None:
            continue
        x2 = ((Q - c) ** 2).sum(-1) / (r * r)
        b = (np.clip(1.0 - x2, 0.0, 1.0) ** exp * h).astype(np.float32)
        bump[sl] = np.maximum(bump[sl], b)
    sdf.d -= bump
    return len(cen)


def arch_visible(sdf, cams=("cam_arch_in", "cam_arch_main", "cam_arch_out"), margin=(0.3, 0.2)):
    """weight(P) for hummocks: 1 on surface that faces one of the arch cameras inside its frame
    (+ margin, wide screens see more at the sides), 0 on the hidden back / underside, so the
    triangle budget goes to relief that is seen."""
    def w(P):
        P = np.asarray(P, float)
        e = 1.5 * sdf.voxel
        g = np.stack([sdf.sample(P + e * ax) - sdf.sample(P - e * ax) for ax in np.eye(3)], -1)
        n = C.normalize(g)
        out = np.zeros(len(P))
        for cam in cams:
            loc, q = pose_of(cam)
            u, v, z = C.project_to_screen(loc, q, P)
            inside = (C.smoothstep(-margin[0] - 0.1, -margin[0], u) * (1.0 - C.smoothstep(1 + margin[0], 1.1 + margin[0], u))
                      * C.smoothstep(-margin[1] - 0.1, -margin[1], v) * (1.0 - C.smoothstep(1 + margin[1], 1.1 + margin[1], v))
                      * (z > 0.3))
            facing = (n * C.normalize(np.asarray(loc, float) - P)).sum(1)
            out = np.maximum(out, inside * C.smoothstep(-0.3, 0.1, facing))
        return out
    return w


def build_arch_ground(coll, mat, N3):
    """moss_arch_ground: mossy ground between and around the trunk bases (seen in frame 05):
    the same macro forms as revision 1, now covered by packed moss hummocks (5-20 cm)."""
    sdf = C.SDF((-3.2, -2.3, -1.2), (3.2, 1.2, 0.75), 0.018)
    rng = np.random.default_rng(51)
    for u in np.linspace(-0.15, 1.15, 12):
        d = 4.0 + 0.4 * rng.uniform(-1, 1)
        c = S("cam_arch_main", u, 1.08 + 0.03 * rng.uniform(-1, 1), d)
        r = 0.45 + 0.1 * rng.uniform(-1, 1)
        sdf.ellipsoid(c, (r * 1.3, r * 1.1, r * 0.7), k=0.25)
    for u in np.linspace(0.0, 1.0, 7):
        d = 3.2 + 0.3 * rng.uniform(-1, 1)
        c = S("cam_arch_main", u, 1.12, d)
        r = 0.4 + 0.1 * rng.uniform(-1, 1)
        sdf.ellipsoid(c, (r * 1.3, r, r * 0.7), k=0.25)
    sdf.noise(N3, 0.06, 1.8, octaves=3, offset=(5.5, 1.1, 3.3))
    vis = arch_visible(sdf)
    n0 = hummocks(sdf, 52, r_range=(0.12, 0.24), h_ratio=(0.12, 0.2), pack=0.8, passes=2, weight=vis)
    n1 = hummocks(sdf, 53, r_range=(0.04, 0.11), h_ratio=(0.28, 0.42), pack=0.85, passes=3, weight=vis)
    sdf.noise(N3, 0.008, 10.0, octaves=2, offset=(1.5, 6.1, 0.3))
    clip(sdf, "cam_arch_out", ARCH_OUT_WINDOW, invert=True, k=0.05)
    obj = sdf.to_object("moss_arch_ground", coll, mat, target_tris=20500, smooth_iter=1)
    C.log(f"moss_arch_ground: {n0} swells + {n1} hummocks")
    return obj, sdf


def build_arch_fg(coll, mat, N3):
    """moss_arch_near__fg: near, dark, out-of-focus forms along the bottom of frame 04, covered by
    moss hummocks (6-20 cm) so their soft top edge is lumpy."""
    sdf = C.SDF((-2.6, -3.0, -0.6), (3.0, -1.0, 1.35), 0.022)
    rng = np.random.default_rng(61)
    tu, tv = np.array(ARCH_FG_TOP).T
    for u in np.linspace(-0.12, 1.12, 10):
        d = 2.9 + 0.35 * rng.uniform(-1, 1)
        r = 0.3 + 0.07 * rng.uniform(-1, 1)
        v_top = float(np.interp(u, tu, tv)) - 0.015 + 0.012 * rng.uniform(-1, 1)
        c = S("cam_arch_main", u, v_top + r * dv(d), d)
        sdf.ellipsoid(c, (r * 1.35, r, r * 1.05), k=0.15)
        sdf.ellipsoid(c + np.array([0, 0, -0.45]), (r * 1.4, r * 1.1, r * 0.9), k=0.15)
    # smaller swells between them: an uneven top line instead of a row of equal domes
    for u in rng.uniform(-0.05, 1.05, 7):
        d = 2.75 + 0.3 * rng.uniform(-1, 1)
        r = 0.17 + 0.05 * rng.uniform(-1, 1)
        v_top = float(np.interp(u, tu, tv)) + rng.uniform(-0.015, 0.02)
        sdf.ellipsoid(S("cam_arch_main", u, v_top + r * dv(d), d), (r * 1.3, r, r * 1.1), k=0.12)
    sdf.noise(N3, 0.04, 2.5, octaves=3, offset=(2.5, 3.5, 4.5))
    # the top line itself (frame 04), then the hummocks on it
    clip(sdf, "cam_arch_main", ARCH_FG_TOP + [(1.7, 3.0), (-0.6, 3.0)], k=0.04)
    vis = arch_visible(sdf)
    n0 = hummocks(sdf, 62, r_range=(0.12, 0.24), h_ratio=(0.12, 0.2), pack=0.8, passes=2, weight=vis)
    n1 = hummocks(sdf, 63, r_range=(0.05, 0.12), h_ratio=(0.25, 0.38), pack=0.85, passes=3, weight=vis)
    sdf.noise(N3, 0.008, 9.0, octaves=2, offset=(0.5, 2.5, 7.5))
    clip(sdf, "cam_arch_out", ARCH_OUT_WINDOW, invert=True, k=0.05)
    obj = sdf.to_object("moss_arch_near__fg", coll, mat, target_tris=9000, smooth_iter=1)
    C.log(f"moss_arch_near__fg: {n0} swells + {n1} hummocks")
    return obj, sdf


# ------------------------------------------------------------------------------------------
# COLOR_0: R density, G length, B AO/cavity, A 1
# ------------------------------------------------------------------------------------------

def hero_outline_weight(co, nrm, zone_uv=(-0.1, 0.5, 0.34, 1.06), max_depth=2.3):
    """1 on faces that form an outline in the hero views (the engine's rim: |n . to_cam| < 0.55 for
    cam_hero_main / cam_hero_out), limited to a cam_hero_main screen zone (u0, v0, u1, v1) and
    to view depths below max_depth (the hero part of a mass, not its T01 / arch parts)."""
    best = np.zeros(len(co))
    for cname in ("cam_hero_main", "cam_hero_out"):
        to_cam = np.asarray(CAMS[cname]["loc"], float) - co
        to_cam /= np.maximum(np.linalg.norm(to_cam, axis=1, keepdims=True), 1e-9)
        best = np.maximum(best, 1.0 - np.minimum(1.0, np.abs(np.einsum("ij,ij->i", nrm, to_cam)) / 0.55))
    loc, q = pose_of("cam_hero_main")
    right, up, fwd = C.cam_basis(q)
    rel = co - loc
    z = np.maximum(rel @ fwd, 1e-3)
    u = 0.5 + 0.5 * (rel @ right) / (z * TH)
    v = 0.5 - 0.5 * (rel @ up) / (z * TV)
    u0, v0, u1, v1 = zone_uv
    zone = (C.smoothstep(u0 - 0.04, u0, u) * (1.0 - C.smoothstep(u1, u1 + 0.04, u)) *
            C.smoothstep(v0 - 0.04, v0, v) * (1.0 - C.smoothstep(v1, v1 + 0.04, v)) *
            (1.0 - C.smoothstep(max_depth - 0.15, max_depth, z)))
    return best * zone


def view_zone(co, view, box, max_depth, soft=0.03):
    """1 inside the screen box (u0, v0, u1, v1) of `view` (camera name or video time) and
    nearer than max_depth, fading out over `soft` (screen) / 0.1 m."""
    loc, q = pose_of(view)
    right, up, fwd = C.cam_basis(q)
    rel = co - loc
    z = np.maximum(rel @ fwd, 1e-3)
    u = 0.5 + 0.5 * (rel @ right) / (z * TH)
    v = 0.5 - 0.5 * (rel @ up) / (z * TV)
    u0, v0, u1, v1 = box
    return (C.smoothstep(u0 - soft, u0, u) * (1.0 - C.smoothstep(u1, u1 + soft, u)) *
            C.smoothstep(v0 - soft, v0, v) * (1.0 - C.smoothstep(v1, v1 + soft, v)) *
            (1.0 - C.smoothstep(max_depth - 0.1, max_depth, z)))


def color_moss_mass(obj, sdf, bvh, N3, cams, ao_dist, fg=False, use_ao=True, outline_trim=None,
                    boost=None):
    """outline_trim (r, g): on the hero outline faces (hero_outline_weight) scale the density by
    (1 - r) and the plant length by (1 - g): a short, tidy pile on that outline.
    boost (view, box, max_depth, dR, dG): lift density / length inside a view_zone."""
    me = obj.data
    co, nrm, ed, _ = C.mesh_arrays(me)
    cx_l = sdf.convexity(co, 0.10)     # + crest / - crevice (m), large scale
    cx_s = sdf.convexity(co, 0.035)
    ao = C.ray_ao(co, nrm, bvh, n_rays=12, max_dist=ao_dist) if use_ao else np.ones(len(co))
    cam_pos = [CAMS[c]["loc"] for c in cams]
    rim = C.rim_factor(co, nrm, cam_pos)
    big = N3.fbm(co * 0.8 + 17.0, 3)
    crev = C.smoothstep(0.0, 0.03, -cx_l)
    crest = C.smoothstep(0.0, 0.03, cx_l)
    R = 0.88 + 0.12 * big - 0.18 * crev - 0.1 * (1.0 - ao)
    R = np.clip(R, 0.7, 1.0)
    down = C.smoothstep(0.2, 0.8, -nrm[:, 2])          # hanging grass under overhangs
    G = 0.42 + 0.28 * crest - 0.22 * crev + 0.3 * rim + 0.12 * down + 0.08 * big
    if fg:
        R = np.clip(R * 0.95, 0.7, 1.0)
        G = np.clip(G, 0.3, 0.8)
    if outline_trim is not None:
        w = hero_outline_weight(co, nrm)
        R = R * (1.0 - outline_trim[0] * w)
        G = G * (1.0 - outline_trim[1] * w)
    if boost is not None:
        w = view_zone(co, boost[0], boost[1], boost[2])
        R = np.clip(R + boost[3] * w, 0.0, 1.0)
        G = G + boost[4] * w
    B = ao * (0.88 + 0.12 * np.clip(0.5 + cx_s / 0.02, 0, 1))
    C.write_color0(me, R, np.clip(G, 0.08, 1.0), B)
    C.box_uvs(me)


def color_wood(sw, mask, bvh, N3, use_ao=True, info=None, halo=(0.7, 0.03)):
    """Bark COLOR_0: sparse clover spots near the cushions; with the cushion info a thin halo of
    plants on the bark along the lip front (R = halo[0] at the front, 0 at halo[1] m out on the bark,
    0 under the cushion)."""
    me = sw.obj.data
    co, nrm, ed, _ = C.mesh_arrays(me)
    nv = sw.N * sw.M
    ao = C.ray_ao(co, nrm, bvh, n_rays=12, max_dist=0.45) if use_ao else np.ones(len(co))
    cx = C.surface_convexity(co, nrm, ed, iterations=3)
    R = np.zeros(len(co))
    m = mask.reshape(-1)
    spots = C.smoothstep(0.25, 0.65, N3.fbm(co[:nv] * 9.0 + 77.0, 2))
    near_border = C.smoothstep(0.12, 0.3, m) * (1.0 - C.smoothstep(0.3, 0.42, m))
    R[:nv] = np.clip(0.35 * spots * near_border, 0, 0.35)
    if info is not None and "bark_dist" in info:
        d = np.asarray(info["bark_dist"], float).reshape(-1)
        ring = C.smoothstep(-0.01, 0.0, d) * (1.0 - C.smoothstep(0.0, halo[1], d))
        R[:nv] = np.maximum(R[:nv] * C.smoothstep(0.0, 0.01, d), halo[0] * ring)
        # contact occlusion next to the cushion wall (12 AO rays under-sample the first few cm);
        # wider and deeper where the cushion hangs above the bark (details/03: the shade under the lip)
        gd = _grid_grad(np.asarray(info["bark_dist"], float), sw.smooth).reshape(-1, 3)
        below = C.smoothstep(0.0, 0.7, -C.normalize(gd)[:, 2])
        w = 0.06 + 0.06 * below
        ao[:nv] = ao[:nv] * (1.0 - (0.3 + 0.15 * below) * (1.0 - C.smoothstep(0.0, 1.0, d / w)))
    G = np.full(len(co), 0.35)
    G[:nv] += 0.2 * spots
    B = ao * (0.85 + 0.15 * np.clip(0.5 + cx / 0.006, 0, 1))
    C.write_color0(me, R, G, B)


def color_cushion(obj, info, bvh, N3, cams, use_ao=True):
    me = obj.data
    co, nrm, ed, _ = C.mesh_arrays(me)
    ao = C.ray_ao(co, nrm, bvh, n_rays=12, max_dist=0.35) if use_ao else np.ones(len(co))
    cx = C.surface_convexity(co, nrm, ed, iterations=6)
    rim = C.rim_factor(co, nrm, [CAMS[c]["loc"] for c in cams])
    vis = info["offset"] - info["relief"]           # cushion height above the local bark surface
    edge = C.smoothstep(0.0, 0.45, vis / np.maximum(info["thickness"], 1e-4))
    big = N3.fbm(co * 1.2 + 5.0, 3)
    R = np.clip((0.8 + 0.18 * big) * edge, 0, 1)
    R[vis <= 0.0] = 0.0
    G = 0.4 + 0.35 * np.clip(cx / 0.01, -1, 1) + 0.28 * rim + 0.1 * big
    G *= 0.55 + 0.45 * edge
    if info.get("lip") is not None:
        # the rolled lip carries a dense, long fringe that hangs over the bark (details/03)
        p = info["shape"]
        fringe = C.smoothstep(0.18, 0.35, p) * (1.0 - C.smoothstep(0.6, 0.85, p)) * (vis > 0.0)
        R = np.maximum(R, 0.9 * fringe)
        G = np.maximum(G, 0.95 * fringe)
    B = ao * (0.88 + 0.12 * np.clip(0.5 + cx / 0.01, 0, 1))
    C.write_color0(me, R, np.clip(G, 0.08, 1.0), B)


# ------------------------------------------------------------------------------------------
# build
# ------------------------------------------------------------------------------------------

def build_cameras(coll):
    out = {}
    for name, c in CAMS.items():
        frame_h = 2.0 * c["focus"] * TV
        extras = dict(focus_distance_m=round(c["focus"], 3), frame_height_at_focus_m=round(frame_h, 3),
                      video_time_s=c["t"])
        if "hold" in c:
            extras["hold_until_video_s"] = c["hold"]
        out[name] = C.make_camera(name, coll, c["loc"], cam_q(name), focus=c["focus"], extras=extras)
    for name, k in KEYS.items():
        C.make_key_empty(name, coll, k["loc"], k["dir"])
    return out


MASS_BUILDERS = dict(crown="build_crown_mass", column="build_left_column", upper="build_upper_arm",
                     herofg="build_hero_fg", ground="build_arch_ground", archfg="build_arch_fg")
TRUNK_BUILDERS = dict(left="build_left_trunk", right="build_right_trunk", low="build_low_branch")


def build(use_ao=True, only=None):
    """Build the grove set; only: optional list of builder keys (MASS_BUILDERS / TRUNK_BUILDERS) for
    quick iteration (previews only - an export needs the full set)."""
    t0 = time.time()
    scene = C.ensure_scene(SCENE_NAME)
    coll = C.ensure_collection(scene, COLL_NAME)
    C.purge_collection(coll)
    _POSES.clear()
    mat_moss = C.preview_material("mat_moss", C.PALETTE["moss"], 0.95)
    mat_wood = C.preview_material("mat_wood", C.PALETTE["wood"], 0.85)
    N3 = C.Noise3(SEED)

    build_cameras(coll)
    masses = {}
    for key, fname in MASS_BUILDERS.items():
        if only is not None and key not in only:
            continue
        fn = globals()[fname]
        t = time.time()
        obj, sdf = fn(coll, mat_moss, N3)
        masses[obj.name] = (obj, sdf)
        C.log(f"{obj.name}: {C.tri_count(obj)} tris ({time.time() - t:.1f}s)")
    trunks = {}
    for key, fname in TRUNK_BUILDERS.items():
        if only is not None and key not in only:
            continue
        fn = globals()[fname]
        t = time.time()
        sw, m, cush, info = fn(coll, mat_wood, mat_moss, N3)
        trunks[sw.obj.name] = (sw, m, cush, info)
        C.log(f"{sw.obj.name}: {C.tri_count(sw.obj)} tris + {cush.name}: {C.tri_count(cush)} tris "
              f"({time.time() - t:.1f}s)")

    # the SDF decimation can leave a duplicated triangle (moss_crown_mass, moss_arch_near__fg: one
    # each; the glTF exporter warned "not valid"): drop such faces before the colours
    meshes = [o for o in coll.objects if o.type == "MESH"]
    for o in meshes:
        if o.data.validate():
            C.log(f"{o.name}: invalid geometry removed ({C.tri_count(o)} tris)")

    # ---- vertex colours (AO against the whole set)
    t = time.time()
    bvh = C.build_bvh(meshes) if use_ao else None
    hero_cams = ["cam_hero_main", "cam_hero_out", "cam_hero_p1", "cam_hero_p2"]
    arch_cams = ["cam_arch_in", "cam_arch_main", "cam_arch_out"]
    for name, (obj, sdf) in masses.items():
        cams = arch_cams if name.startswith("moss_arch") else hero_cams + (["cam_arch_in", "cam_arch_main"] if "crown" in name or "column" in name else [])
        # frame 01: the left column carries a short, tidy pile on its outline
        trim = (0.5, 1.0) if name == "moss_left_column" else None
        # frame 03: the lit chin carries a dense, long pile
        boost = (T01_VIEW,) + T01_CHIN_PILE if name == "moss_crown_mass" else None
        color_moss_mass(obj, sdf, bvh, N3, cams, ao_dist=0.45 if "arch" in name else 0.3,
                        fg=name.endswith("__fg"), use_ao=use_ao, outline_trim=trim, boost=boost)
    for name, (sw, m, cush, info) in trunks.items():
        color_wood(sw, m, bvh, N3, use_ao=use_ao, info=info if name != "wood_arch_low" else None)
        color_cushion(cush, info, bvh, N3, arch_cams, use_ao=use_ao)
    C.log(f"colours ({time.time() - t:.1f}s)")

    total = sum(C.tri_count(o) for o in coll.objects if o.type == "MESH")
    C.log(f"grove built: {total} tris, {time.time() - t0:.1f}s")
    return scene, coll


def render_previews(scene, coll, cams=None, percent=50, suffix="", outdir=CAPTURES):
    C.setup_preview_render(scene, percent=percent)
    out = []
    for name in (cams or list(CAMS.keys())):
        cam = coll.objects.get(name)
        path = os.path.join(outdir, f"grove_{name}{suffix}.png")
        C.render_camera(scene, cam, path)
        out.append(path)
    return out


def save_blend():
    bpy.ops.wm.save_as_mainfile(filepath=BLEND_PATH, check_existing=False, compress=True)
    C.log("saved", BLEND_PATH)


# ------------------------------------------------------------------------------------------
# replica of the engine's camera track (src/nature/scenes/GroveScene.ts buildTrack, TRACK):
# hero_main hold -> hero_out (inOutSine 0.7 -> 2.2 s), hold to 8.0 s, then one monotone cubic
# Hermite (Fritsch-Butland tangents, start at rest, end tangent x0.6) per glTF component through
# hero_out (8.0), p1 (8.95), a key 42 % of the way from p1 to p2 still looking like p1 (9.15),
# p2 (9.72), arch_in (10.0), arch_main (12.5), arch_out (15.2) and an exit pose (16.4 s).  Used to
# model and preview the in-between frames (8.4 s, 9.1 s = frame 03, ...) exactly as the engine
# frames them (revision 3: rev 2 still used the older 8.8 / 9.6 s keys without the 9.15 s key).
# ------------------------------------------------------------------------------------------
TRACK_KEYS = [(8.0, "cam_hero_out"), (8.95, "cam_hero_p1"), (9.15, "p12"), (9.72, "cam_hero_p2"),
              (10.0, "cam_arch_in"), (12.5, "cam_arch_main"), (15.2, "cam_arch_out")]
TRACK_P12_FRACTION = 0.42
TRACK_ARCH_OUT_FOCUS = 2.6      # GroveScene: min(arch_out focus extra, 2.6)
TRACK_EXIT_T = 16.4
_CORR = Quaternion((math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0))     # exporter's camera correction


def _q_to_gltf(q):
    """Blender camera rotation -> glTF node rotation (+Y up export incl. the camera correction)."""
    s = q @ _CORR
    return Quaternion((s.w, s.x, s.z, -s.y))


def _q_from_gltf(g):
    s = Quaternion((g.w, g.x, -g.z, g.y))
    return s @ _CORR.inverted()


def _key_vec(loc, q, focus):
    g = _q_to_gltf(q)
    return np.array([loc[0], loc[2], -loc[1], g.x, g.y, g.z, g.w, C.FOV_V_DEG, focus], float)


def _monotone_tangents(times, values):
    n = len(times)
    out = np.zeros_like(values)
    for i in range(n):
        if i == 0:
            out[i] = 0.0                                                   # startAtRest
        elif i == n - 1:
            out[i] = (values[i] - values[i - 1]) / (times[i] - times[i - 1]) * 0.6
        else:
            h0, h1 = times[i] - times[i - 1], times[i + 1] - times[i]
            s0 = (values[i] - values[i - 1]) / h0
            s1 = (values[i + 1] - values[i]) / h1
            with np.errstate(divide="ignore", invalid="ignore"):
                m = 3.0 * (h0 + h1) / ((2.0 * h1 + h0) / s0 + (h1 + 2.0 * h0) / s1)
            out[i] = np.where(s0 * s1 <= 0.0, 0.0, m)
    return out


def track_pose(t):
    """(loc, quat, focus) of the engine camera at video time t (Blender coordinates)."""
    if t <= 0.7:
        c = CAMS["cam_hero_main"]
        return np.array(c["loc"], float), cam_q("cam_hero_main"), c["focus"]
    if t < 8.0:
        a, b = CAMS["cam_hero_main"], CAMS["cam_hero_out"]
        s = min(1.0, (t - 0.7) / 1.5)
        e = 0.5 - 0.5 * math.cos(math.pi * s)
        q = cam_q("cam_hero_main").slerp(cam_q("cam_hero_out"), e)
        return (np.array(a["loc"]) * (1 - e) + np.array(b["loc"]) * e, q,
                a["focus"] * (1 - e) + b["focus"] * e)
    times, vals = [], []
    for (tk, name) in TRACK_KEYS:
        times.append(tk)
        if name == "p12":
            a, b = CAMS["cam_hero_p1"], CAMS["cam_hero_p2"]
            f = TRACK_P12_FRACTION
            loc = np.asarray(a["loc"], float) * (1.0 - f) + np.asarray(b["loc"], float) * f
            vals.append(_key_vec(loc, cam_q("cam_hero_p1"), a["focus"] + (b["focus"] - a["focus"]) * f))
            continue
        c = CAMS[name]
        focus = min(c["focus"], TRACK_ARCH_OUT_FOCUS) if name == "cam_arch_out" else c["focus"]
        vals.append(_key_vec(c["loc"], cam_q(name), focus))
    # exit pose: arch_out + 0.45 (arch_out - arch_main), slerp(main, out, 1.35)
    am, ao = np.array(CAMS["cam_arch_main"]["loc"]), np.array(CAMS["cam_arch_out"]["loc"])
    qm, qo = _q_to_gltf(cam_q("cam_arch_main")), _q_to_gltf(cam_q("cam_arch_out"))
    if qm.dot(qo) < 0:
        qo.negate()
    # three's slerp with t > 1 extrapolates along the great circle (qa * (qa^-1 qb)^t)
    d = qm.inverted() @ qo
    dax, dang = d.to_axis_angle()
    qe = qm @ Quaternion(dax, dang * 1.35)
    pe = ao + 0.45 * (ao - am)
    loc_e = np.array([pe[0], pe[2], -pe[1]])
    times.append(TRACK_EXIT_T)
    vals.append(np.array([*loc_e, qe.x, qe.y, qe.z, qe.w, C.FOV_V_DEG, 2.2], float))
    vals = np.array(vals)
    for i in range(1, len(vals)):                                          # one hemisphere
        if np.dot(vals[i, 3:7], vals[i - 1, 3:7]) < 0:
            vals[i, 3:7] *= -1.0
    times = np.array(times)
    tan = _monotone_tangents(times, vals)
    t = min(t, TRACK_EXIT_T)
    i = int(np.clip(np.searchsorted(times, t) - 1, 0, len(times) - 2))
    h = times[i + 1] - times[i]
    s = (t - times[i]) / h
    s2, s3 = s * s, s * s * s
    v = ((2 * s3 - 3 * s2 + 1) * vals[i] + (s3 - 2 * s2 + s) * h * tan[i] + (-2 * s3 + 3 * s2) * vals[i + 1] +
         (s3 - s2) * h * tan[i + 1])
    g = Quaternion((v[6], v[3], v[4], v[5])).normalized()
    return np.array([v[0], -v[2], v[1]]), _q_from_gltf(g), float(v[8])


# ------------------------------------------------------------------------------------------
# previews, silhouette masks and comparison sheets (Blender only: numpy + bpy images).
# References are read for comparison only - never shipped.
# ------------------------------------------------------------------------------------------
REF = os.path.join(ROOT, "docs", "nature-webgl-reference")
M01 = os.path.join(REF, "motion", "01_foliage_transition.jpg")
M01_T = [8.0, 8.4, 8.8, 9.2, 9.6, 10.0]           # panels of motion/01 (3 x 2, 480 x 340 each)

# key -> (reference, pose): pose = camera name or video time on the track
SHEETS = {
    "hero_main": ("frames/01_hero_clean.png", "cam_hero_main"),
    "hero_out": ("frames/02_hero.png", "cam_hero_out"),
    "t8.4": (("m01", 8.4), 8.4),
    # (keys keep their rev-2 names; the panels are matched at the engine track pose of their time)
    "hero_p1": (("m01", 8.8), 8.8),
    "t9.1": ("frames/03_foliage_wipe.png", 9.1),
    "hero_p2": (("m01", 9.6), 9.6),
    "arch_in": (("m01", 10.0), "cam_arch_in"),
    "arch_main": ("frames/04_trunk_arch.png", "cam_arch_main"),
    "arch_out": ("frames/05_trunk_exit.png", "cam_arch_out"),
}
SKY_SHEETS = ("t8.4", "hero_p1", "t9.1", "hero_p2")     # reference sky outline + IoU (T01)


def _load_rgba(path):
    img = bpy.data.images.load(path, check_existing=False)
    w, h = img.size
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    return px.reshape(h, w, 4)[::-1].copy()          # top-down rows


def _save_rgb(arr, path):
    h, w = arr.shape[:2]
    img = bpy.data.images.new("silva_sheet_tmp", width=w, height=h, alpha=False)
    rgba = np.ones((h, w, 4), np.float32)
    rgba[..., :3] = np.clip(arr[..., :3], 0, 1)
    img.pixels.foreach_set(rgba[::-1].ravel())
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)
    return path


def _resize(a, h, w):
    """Bilinear resize of an (H, W, C) or (H, W) array."""
    H, W = a.shape[:2]
    ys = np.clip((np.arange(h) + 0.5) * H / h - 0.5, 0, H - 1)
    xs = np.clip((np.arange(w) + 0.5) * W / w - 0.5, 0, W - 1)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, H - 1)
    x1 = np.minimum(x0 + 1, W - 1)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    if a.ndim == 3:
        fy, fx = fy[..., None], fx[..., None]
    top = a[y0][:, x0] * (1 - fx) + a[y0][:, x1] * fx
    bot = a[y1][:, x0] * (1 - fx) + a[y1][:, x1] * fx
    return top * (1 - fy) + bot * fy


def _box(a, k):
    """Mean over a k x k window (edge-clamped), via an integral image."""
    r = k // 2
    p = np.pad(a.astype(np.float64), r, mode="edge")
    s = np.pad(np.cumsum(np.cumsum(p, 0), 1), ((1, 0), (1, 0)))
    H, W = a.shape
    return (s[k:k + H, k:k + W] - s[:H, k:k + W] - s[k:k + H, :W] + s[:H, :W]) / (k * k)


def reference_image(ref):
    if isinstance(ref, tuple):                        # motion/01 panel at time ref[1]
        i = M01_T.index(ref[1])
        a = _load_rgba(M01)
        c, r = i % 3, i // 3
        return a[370 * r:370 * r + 340, 480 * c:480 * c + 480, :3]
    return _load_rgba(os.path.join(REF, ref))[..., :3]


def sky_mask(rgb):
    """Smooth, low-saturation background of a reference frame (T01: the light keyhole) at the
    resolution given (use ~720 px wide or the 480 px panels). Iteration aid only."""
    a = rgb * 255.0
    G, B = a[..., 1], a[..., 2]
    L = 0.299 * a[..., 0] + 0.587 * G + 0.114 * B
    m1 = _box(L, 7)
    std = np.sqrt(np.maximum(_box(L * L, 7) - m1 * m1, 0.0))
    m = (B > 0.74 * G) & (std < 4.0) & (G > 14.0)
    return _box(m.astype(np.float64), 7) > 0.5


def _outline(mask, width=1):
    m = mask.astype(bool)
    e = np.zeros_like(m)
    for dy in range(-width, width + 1):
        for dx in range(-width, width + 1):
            if dx or dy:
                e |= np.roll(np.roll(m, dy, 0), dx, 1) != m
    return e & m


def _grid(img, step=0.1, col=(0.3, 0.45, 1.0), major=(1.0, 0.25, 0.25)):
    h, w = img.shape[:2]
    for i in range(1, int(round(1 / step))):
        c = major if abs(i * step - 0.5) < 1e-6 else col
        x = int(round(i * step * w))
        y = int(round(i * step * h))
        img[:, x, :3] = img[:, x, :3] * 0.4 + np.array(c) * 0.6
        img[y, :, :3] = img[y, :, :3] * 0.4 + np.array(c) * 0.6


def _pose_camera(coll_tmp, key, pose):
    if isinstance(pose, str):
        c = CAMS[pose]
        loc, q = c["loc"], cam_q(pose)
    else:
        loc, q, _ = track_pose(pose)
    return C.make_camera("tmp_" + key.replace(".", "_"), coll_tmp, tuple(loc), q)


def _pose_time(pose):
    return CAMS[pose]["t"] if isinstance(pose, str) else float(pose)


def place_hero_fg(coll, t):
    """moss_hero_near__fg as the engine shows it at video time t (GroveScene.animate): 5 cm lower
    than modelled, riding down with the camera from 8.0 s while dropping 0.5 m, hidden from 8.7 s.
    t=None resets it (export)."""
    fg = coll.objects.get("moss_hero_near__fg")
    if fg is None:
        return
    if t is None:
        fg.location = (0.0, 0.0, 0.0)
        fg.hide_render = False
        return
    if t <= 8.0:
        fg.location = (0.0, 0.0, -0.05)
        fg.hide_render = False
    elif t < 8.7:
        p0 = track_pose(8.0)[0]
        p1 = track_pose(t)[0]
        s = float(C.smoothstep(8.0, 8.65, t))
        d = p1 - p0
        fg.location = (d[0], d[1], d[2] - 0.05 - 0.5 * s)
        fg.hide_render = False
    else:
        fg.hide_render = True


def render_sheet_views(scene, keys, outdir, percent=50):
    """Preview (Workbench) + silhouette mask of every sheet pose -> outdir/view_<key>.png, mask_<key>.png."""
    tmp = bpy.data.collections.new("grove_tmp_views")
    scene.collection.children.link(tmp)
    coll = bpy.data.collections.get(COLL_NAME)
    sh = scene.display.shading
    out = {}
    try:
        cams = {k: _pose_camera(tmp, k, SHEETS[k][1]) for k in keys}
        C.setup_preview_render(scene, percent=percent)
        for k in keys:
            place_hero_fg(coll, _pose_time(SHEETS[k][1]))
            p = os.path.join(outdir, f"view_{k}.png")
            C.render_camera(scene, cams[k], p)
            out[k] = [p]
        prev = (scene.render.film_transparent, sh.light, sh.color_type, sh.show_cavity,
                scene.render.image_settings.color_mode)
        scene.render.film_transparent = True
        scene.render.image_settings.color_mode = "RGBA"
        sh.light, sh.color_type, sh.single_color, sh.show_cavity = "FLAT", "SINGLE", (1.0, 1.0, 1.0), False
        for k in keys:
            place_hero_fg(coll, _pose_time(SHEETS[k][1]))
            p = os.path.join(outdir, f"mask_{k}.png")
            C.render_camera(scene, cams[k], p)
            out[k].append(p)
        (scene.render.film_transparent, sh.light, sh.color_type, sh.show_cavity,
         scene.render.image_settings.color_mode) = prev
    finally:
        place_hero_fg(coll, None)
        for o in list(tmp.objects):
            C.remove_object(o.name)
        bpy.data.collections.remove(tmp)
    return out


def compare_sheets(scene, coll, outdir=CAPTURES, keys=None, percent=50):
    """grove_cmp_<key>.png: reference | Workbench preview | reference x0.55 + our base silhouette
    (yellow; plants add their fringe outside it) + for T01 the reference sky edge (cyan) + 0.1 grid."""
    keys = keys or list(SHEETS.keys())
    vdir = os.path.join(outdir, "_grove_views")
    if os.path.normcase(os.path.normpath(outdir)) == os.path.normcase(os.path.normpath(CAPTURES)):
        vdir = os.path.join(tempfile.gettempdir(), "silva_grove_views")    # keep captures/ clean
    os.makedirs(vdir, exist_ok=True)
    views = render_sheet_views(scene, keys, vdir, percent)
    W, H = int(C.FRAME_W * percent / 100), int(C.FRAME_H * percent / 100)
    out, stats = [], {}
    for k in keys:
        ref = reference_image(SHEETS[k][0])
        ref_s = _resize(ref, H, W)
        prev = _load_rgba(views[k][0])[..., :3]
        mask = _load_rgba(views[k][1])[..., 3] > 0.5
        ov = ref_s * 0.55
        if k in SKY_SHEETS:
            small = ref if ref.shape[1] <= 720 else _resize(ref, 510, 720)
            sky = _resize(sky_mask(small).astype(np.float32), H, W) > 0.5
            ov[_outline(sky)] = (0.2, 0.95, 1.0)
            valid = np.zeros_like(sky)
            valid[int(0.12 * H):int(0.82 * H)] = True
            occ_ref = ~sky & valid
            occ_our = mask & valid
            stats[k] = dict(iou=round(float((occ_ref & occ_our).sum()) / max(float((occ_ref | occ_our).sum()), 1.0), 4),
                            ours_minus_ref=round(float((occ_our & ~occ_ref).sum()) / valid.sum(), 4),
                            ref_minus_ours=round(float((occ_ref & ~occ_our).sum()) / valid.sum(), 4))
        ov[_outline(mask)] = (1.0, 0.92, 0.1)
        _grid(ov)
        path = os.path.join(outdir, f"grove_cmp_{k}.png")
        _save_rgb(np.concatenate([ref_s, prev, ov], 1), path)
        out.append(path)
    C.log("compare:", json.dumps(stats))
    return out, stats


def parse_args(argv):
    a = dict(export=False, validate=False, render=None, percent=50, save=False, ao=True, compare=None,
             outdir=CAPTURES, sheet_percent=None, glb=GLB_PATH)
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
            a["compare"] = []
            while i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                i += 1
                a["compare"].append(argv[i])
        elif x == "--outdir":
            i += 1
            a["outdir"] = argv[i]
        elif x == "--glb":
            i += 1
            a["glb"] = argv[i]
        elif x == "--percent":
            i += 1
            a["percent"] = int(argv[i])
        elif x == "--sheet-percent":
            i += 1
            a["sheet_percent"] = int(argv[i])
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
    scene, coll = build(use_ao=a["ao"])
    result = dict(tris=sum(C.tri_count(o) for o in coll.objects if o.type == "MESH"))
    glb = a["glb"]
    if a["export"]:
        C.export_glb(scene, glb)
        result["glb"] = glb
        result["glb_bytes"] = os.path.getsize(glb)
    if a["validate"]:
        rep = C.validate_glb(glb)
        result["validate"] = dict(errors=rep["errors"], warnings=rep["warnings"], totals=rep["totals"])
        # the report of the shipped GLB goes to captures/; a scratch GLB's report sits next to it
        if os.path.normcase(os.path.abspath(glb)) == os.path.normcase(GLB_PATH):
            vpath = os.path.join(CAPTURES, "grove_validate.json")
        else:
            vpath = os.path.splitext(glb)[0] + "_validate.json"
        os.makedirs(os.path.dirname(vpath), exist_ok=True)
        with open(vpath, "w") as f:
            json.dump(rep, f, indent=1, default=str)
    if a["render"] is not None:
        result["renders"] = render_previews(scene, coll, a["render"] or None, a["percent"], outdir=a["outdir"])
    if a["compare"] is not None:
        result["compare"], result["compare_stats"] = compare_sheets(
            scene, coll, a["outdir"], a["compare"] or None,
            a["percent"] if a["sheet_percent"] is None else a["sheet_percent"])
    if a["save"]:
        save_blend()
    C.log("result", json.dumps(result, default=str))
    return result


if __name__ == "__main__":
    _argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    main(_argv)
