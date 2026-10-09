"""
build_canyon.py - the 'canyon' scene set (S03, video 15.5-21 s): an almost black gorge framed by three
jointed rock fragments - a narrow edge on the left, a big lit face at the upper right with sparse plant
islands on its ledges, a dark form rising from the bottom-left corner - exported to
public/nature/models/canyon.glb (contract: CLAUDE.md "Asset contract").

Run headless (from the repo root):
    "/c/Program Files/Blender Foundation/Blender 5.1/blender.exe" --background --factory-startup \
        --python assets-src/blender/build_canyon.py -- --export --validate [--render [cams]] [--compare] [--save]

Run inside a live Blender session (e.g. through MCP):
    import sys, importlib; sys.path.insert(0, r"<repo>/assets-src/blender")
    import build_canyon; importlib.reload(build_canyon); build_canyon.main(["--save"])

Options:
    --export            write public/nature/models/canyon.glb
    --validate          re-read / re-import the GLB and check it (common.validate_glb + canyon checks);
                        report: docs/captures/blender/canyon_validate.json
    --render [cams]     flat-lit previews -> docs/captures/blender/canyon_<cam>.png (all cams if none)
    --compare           comparison sheet vs frame 06 -> docs/captures/blender/canyon_cmp_main.png (+ _in/_out
                        previews side by side)
    --percent N         preview size in % of 1440x1020 (default 50)
    --save              save nature.blend (headless: opens the existing file first so the other scene
                        sets are kept; only the 'canyon' scene / collection is rebuilt)
    --no-ao             skip the ray-traced AO (B = cavity only; fast iteration)

Deterministic (seed 134). World units: metres, Blender Z-up, the cameras look along +Y.

Framing (frame 06, 19.8 s; the HTML cards of the reference cover most of the lower half and are
ignored): the centre is black; the right fragment's lit face fills the upper right from ~70 % of the
width (its left silhouette runs from u 0.69 at the top to u 0.79 at 45 % height, measured), the left
fragment shows a narrow lit edge (u <= 0.095) between ~20 % and 45 % height and continues down the left
border, the near fragment rises from the bottom-left corner (~25 % x 25 % of the frame). The previews
are flat colour renders (Lambert from key_canyon with ray-traced shadows x baked AO; rock ledges that
carry plants are tinted green by their COLOR_0.R): they show placement and light/dark, not the
engine's materials.
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
GLB_PATH = os.path.join(ROOT, "public", "nature", "models", "canyon.glb")
BLEND_PATH = os.path.join(HERE, "nature.blend")
CAPTURES = os.path.join(ROOT, "docs", "captures", "blender")
REF = os.path.join(ROOT, "docs", "nature-webgl-reference")      # comparison sheets only (never shipped)
TMP = os.path.join(tempfile.gettempdir(), "silva_canyon_build")
SCENE_NAME = "canyon"
COLL_NAME = "canyon"
SEED = 134
MAX_TRIS = 120_000

# ------------------------------------------------------------------------------------------
# cameras: main = frame 06 (19.8 s). in = 16.4 s, end of the arch -> canyon dip: the canyon opens out
# of the dark a little closer and lower (looking slightly more up); out = 21.2 s, inside the dark
# crossfade to the oracle: the slow pull back continues. Frame height at the focus plane ~2.8 m.
# ------------------------------------------------------------------------------------------
MAIN_LOC = (0.0, -4.4, 0.0)
CAMS = {
    "cam_canyon_in": dict(loc=(0.04, -4.0, -0.13), pitch=5.0, yaw=0.4, roll=0.0, focus=4.05, t=16.4),
    "cam_canyon_main": dict(loc=MAIN_LOC, pitch=3.0, yaw=0.0, roll=0.0, focus=4.45, t=19.8),
    "cam_canyon_out": dict(loc=(-0.03, -4.58, 0.07), pitch=2.2, yaw=-0.2, roll=0.0, focus=4.6, t=21.2),
}
# key: narrow, from the upper right and a little in front: lights the slab flanks and ledges of the
# right face (its camera-facing planes only dimly), grazes the inner edge of the left fragment whose
# face turns away, leaves the centre and the near form dark.
KEY_DIR = (-0.42, 0.45, -0.79)          # direction the light travels (Blender world)

# ------------------------------------------------------------------------------------------
# fragments in frame-06 screen space of cam_canyon_main: visual-hull polygons (u, v) (the mass never
# leaves them) and view depths (m) of their front faces.
# ------------------------------------------------------------------------------------------
RIGHT_POLY = [(0.692, -0.40), (0.694, 0.0), (0.703, 0.08), (0.714, 0.15), (0.729, 0.22), (0.746, 0.29),
              (0.762, 0.35), (0.777, 0.41), (0.790, 0.47), (0.800, 0.56), (0.814, 0.68), (0.830, 0.82),
              (0.848, 1.0), (0.87, 1.4), (1.8, 1.4), (1.8, -0.4)]
LEFT_POLY = [(-0.8, -0.4), (0.004, -0.4), (0.008, 0.0), (0.016, 0.10), (0.030, 0.18), (0.046, 0.24),
             (0.063, 0.30), (0.080, 0.37), (0.092, 0.42), (0.096, 0.47), (0.092, 0.56), (0.083, 0.66),
             (0.070, 0.78), (0.060, 0.92), (0.055, 1.4), (-0.8, 1.4)]
LOW_POLY = [(-0.8, 0.705), (0.0, 0.705), (0.050, 0.716), (0.100, 0.742), (0.150, 0.786), (0.190, 0.832),
            (0.222, 0.888), (0.247, 0.950), (0.262, 1.02), (0.27, 1.4), (-0.8, 1.4)]

# plant islands (frame 06: green on the right fragment at its upper-left edge, on a ledge at mid-left,
# at the top right; a little on the left edge and on top of the near form): soft screen rectangles
# (u0, v0, u1, v1, weight) of cam_canyon_main. COLOR_0.R = ledge (upward faces) x zone x patches.
ZONES = {
    "rock_canyon_right": [(0.680, -0.10, 0.800, 0.27, 1.0), (0.770, 0.25, 0.875, 0.50, 0.9),
                          (0.875, -0.10, 1.10, 0.34, 1.0), (0.890, 0.30, 1.10, 0.52, 0.65)],
    "rock_canyon_left": [(-0.10, 0.27, 0.105, 0.48, 0.55), (-0.10, 0.62, 0.095, 0.90, 0.45)],
    "rock_canyon_low__fg": [(-0.10, 0.68, 0.30, 1.10, 0.7)],
}
# the near form is in shadow in frame 06 (dark, out of focus): its baked AO is scaled down
DARKEN = {"rock_canyon_low__fg": 0.24, "rock_canyon_left": 0.6}
# moss islands: a few flat cushions on the ledges inside these zones (frame 06's green islands)
MOSS = {
    "moss_canyon_right": dict(rock="rock_canyon_right", count=26, spacing=0.11, radius=(0.05, 0.13),
                              thick=(0.022, 0.042), tris=9000),
    "moss_canyon_left": dict(rock="rock_canyon_left", count=4, spacing=0.12, radius=(0.035, 0.07),
                             thick=(0.010, 0.018), tris=2000),
}


def cam(name):
    c = CAMS[name]
    return L.Cam(name, c["loc"], c["pitch"], c["yaw"], c["roll"], focus=c["focus"], t=c["t"])


MAIN = None


def main_cam():
    global MAIN
    if MAIN is None:
        MAIN = cam("cam_canyon_main")
    return MAIN


def S(u, v, d):
    return main_cam().S(u, v, d)


# ------------------------------------------------------------------------------------------
# SDF helpers
# ------------------------------------------------------------------------------------------

def sd_round_box(q, b, r):
    d = np.abs(q) - (np.asarray(b, float) - r)
    return (np.linalg.norm(np.maximum(d, 0.0), axis=-1) + np.minimum(np.max(d, axis=-1), 0.0) - r)


def round_box(sdf, c, half, R, r=0.08, k=0.0, mode="union"):
    """Rounded box (half extents `half`, local axes = columns of R) into the field."""
    c = np.asarray(c, float)
    half = np.asarray(half, float)
    R = np.asarray(R, float)
    ext = np.abs(R) @ half
    m = 2.0 * k + 3.0 * sdf.voxel
    sdf._combine(c - ext - m, c + ext + m, lambda P: sd_round_box((P - c) @ R, half, r), k, mode)


def rot(yaw_deg=0.0, tilt_x_deg=0.0, tilt_y_deg=0.0):
    """Rotation matrix (columns = local axes): yaw about Z, then tilt about local X and Y."""
    cz, sz = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1.0]])
    cx, sx = math.cos(math.radians(tilt_x_deg)), math.sin(math.radians(tilt_x_deg))
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    cy, sy = math.cos(math.radians(tilt_y_deg)), math.sin(math.radians(tilt_y_deg))
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return Rz @ Rx @ Ry


def view_clamp(sdf, poly, margin=0.0, k=0.03):
    c = main_cam()
    sdf.clip_view(c.loc, c.q, poly, margin=margin, k=k)


# ------------------------------------------------------------------------------------------
# fragments
# ------------------------------------------------------------------------------------------

def face_box(sdf, p_face, normal, half_w, half_h, depth, yaw=0.0, tilt=0.0, r=0.06, k=0.0):
    """Rounded box whose front face (half extents half_w x half_h) lies at p_face, facing `normal`
    (turned by yaw about Z and tilted about its horizontal axis, degrees); `depth` m deep."""
    n = C.normalize(np.asarray(normal, float))
    R0 = C.rot_from_axes(x_axis=np.cross((0.0, 0.0, 1.0), n), z_axis=(0.0, 0.0, 1.0))    # x along face, y = -n
    Rm = rot(yaw, tilt, 0.0)
    R = Rm @ R0
    fwd = R @ np.array([0.0, 1.0, 0.0])          # into the rock
    c = np.asarray(p_face, float) + fwd * (0.5 * depth)
    round_box(sdf, c, (half_w, 0.5 * depth, half_h), R, r=r, k=k)


def along_joint_fade(sdf, P, joint_normal):
    """Fade the joint insets where the surface runs parallel to the joint planes (the side walls of
    the slabs): there the slab index flickers with tiny moves and terraces into a 'zipper'."""
    n = L.sdf_normals(sdf, P)
    return 1.0 - 0.9 * C.smoothstep(0.78, 0.94, np.abs(n @ C.normalize(np.asarray(joint_normal, float))))


def build_right(coll, mat, N3):
    """Tall column at the upper right, undercut below ~45 % of the frame (its lower face recedes into
    the shadow of the upper part). Coarse fissures split the face into a few big slabs; inside them
    steep blades leaning right (traces ~70 deg from horizontal), each a flat, slightly wedge-shaped
    slab set back by its own amount, with narrow joints; cross joints every ~0.3 m break the blades
    into steps (ledges for the plants). Only the
    camera side is modelled (the field is cut 1 m behind the face and above / below every frame)."""
    face_n = (-0.12, -0.993, 0.0)
    sdf = C.SDF((0.50, -0.55, -1.62), (2.55, 1.08, 1.98), 0.013)
    for (x, y, z, hw, hh, yaw, tilt) in (
            (1.55, -0.05, 1.45, 1.05, 0.62, 6, -4), (1.48, -0.11, 0.55, 1.10, 0.52, -5, 4),
            (1.66, 0.20, -0.36, 1.00, 0.42, 8, -10), (1.78, 0.40, -1.25, 1.00, 0.60, -4, -12)):
        face_box(sdf, (x, y, z), face_n, hw, hh, 1.2, yaw, tilt, r=0.07, k=0.10)
    # buttresses / bulges on the face
    for (x, y, z, r3, yaw, k) in ((0.95, -0.20, 0.62, (0.24, 0.22, 0.62), 12, 0.10),
                                  (1.25, -0.24, 1.30, (0.30, 0.20, 0.50), -18, 0.10),
                                  (1.98, -0.18, 0.55, (0.34, 0.20, 0.50), 6, 0.10),
                                  (1.30, 0.02, -0.20, (0.26, 0.20, 0.45), -8, 0.10)):
        sdf.ellipsoid((x, y, z), r3, rot(yaw, 0, 14), k=k)
    sdf.clip_plane((0.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    view_clamp(sdf, RIGHT_POLY, k=0.02)
    sdf.noise(N3, 0.03, 1.2, octaves=3, offset=(3.1, 0.7, 5.2))
    L.joint_insets(sdf, [
        dict(normal=(0.92, 0.05, 0.38), spacing=0.44, plate=0.15, groove=0.05, groove_w=0.03,
             warp=(0.45, 0.7), power=1.2),
        dict(normal=(0.94, 0.0, 0.34), spacing=0.14, plate=0.08, wedge=0.04, groove=0.015, groove_w=0.028,
             warp=(0.6, 1.4), power=1.6, block=0.06),
        dict(normal=(0.30, -0.25, 0.92), spacing=0.30, plate=0.025, groove=0.015, groove_w=0.028,
             warp=(0.45, 0.8)),
    ], seed=SEED + 11, band=0.45, amp_fn=lambda P: along_joint_fade(sdf, P, (0.94, 0.0, 0.34)))
    sdf.noise(N3, 0.006, 9.0, octaves=3, offset=(9.1, 2.2, 1.3), ridged=True)
    # (no finer grain here: at 1.3 cm voxels it aliases into 'teeth' on lit ridges; the engine's
    # triplanar rock normal map carries the fine detail)
    return sdf.to_object("rock_canyon_right", coll, mat, target_tris=56000, smooth_iter=1)


def build_left(coll, mat, N3):
    """Narrow edge on the left: a tall slab of gently dipping layers (stacked blocks whose tops catch
    the key); its face turns away from the key, so mostly the inner edge reads."""
    face_n = (-0.62, -0.785, 0.0)
    sdf = C.SDF((-2.05, -1.62, -1.45), (-0.98, -0.30, 1.85), 0.012)
    for (x, y, z, hw, hh, yaw, tilt) in (
            (-1.58, -1.02, 1.35, 0.42, 0.55, -6, 4), (-1.50, -1.05, 0.55, 0.42, 0.42, 5, -5),
            (-1.52, -0.98, -0.25, 0.45, 0.50, -4, 3), (-1.58, -1.02, -1.05, 0.48, 0.55, 3, -4)):
        face_box(sdf, (x, y, z), face_n, hw, hh, 1.0, yaw, tilt, r=0.06, k=0.08)
    sdf.clip_plane((0.0, -0.4, 0.0), (0.0, 1.0, 0.0))
    view_clamp(sdf, LEFT_POLY, k=0.02)
    sdf.noise(N3, 0.025, 1.4, octaves=3, offset=(7.3, 1.9, 0.4))
    # gently dipping layers: stacked plates whose tops catch the key, broken by steep joints
    L.joint_insets(sdf, [
        dict(normal=(0.26, 0.1, 0.96), spacing=0.13, plate=0.07, wedge=0.05,
             warp=(0.6, 1.3), power=1.5, block=0.06),
        dict(normal=(0.88, -0.38, -0.28), spacing=0.26, warp=(0.3, 1.0)),
    ], seed=SEED + 21, band=0.3)
    sdf.noise(N3, 0.006, 9.0, octaves=3, offset=(2.6, 4.1, 7.7), ridged=True)
    return sdf.to_object("rock_canyon_left", coll, mat, target_tris=26000, smooth_iter=1)


def build_low(coll, mat, N3):
    """Near form rising from the bottom-left corner (out of focus): three stacked slabs leaning out
    of the corner, layered, moss on their tops (COLOR_0.R), kept dark (see color_rock)."""
    sdf = C.SDF((-1.55, -2.55, -1.05), (-0.35, -1.25, 0.05), 0.012)
    face_n = (-0.6, -0.8, 0.0)
    for (x, y, z, hw, hh, d, yaw, tilt, r) in ((-1.05, -1.95, -0.72, 0.48, 0.28, 0.7, -14, -8, 0.09),
                                               (-0.80, -2.05, -0.80, 0.30, 0.22, 0.6, -28, -4, 0.08),
                                               (-1.30, -1.80, -0.50, 0.32, 0.30, 0.6, -6, -10, 0.08)):
        face_box(sdf, (x, y, z), face_n, hw, hh, d, yaw, tilt, r=r, k=0.08)
    view_clamp(sdf, LOW_POLY, k=0.025)
    sdf.noise(N3, 0.035, 1.8, octaves=3, offset=(4.4, 6.6, 2.0))
    L.joint_insets(sdf, [
        dict(normal=(0.45, 0.25, 0.86), spacing=0.12, plate=0.06, wedge=0.02, bulge=0.015,
             warp=(0.5, 1.2), power=1.4, block=0.035),
        dict(normal=(0.7, -0.6, -0.3), spacing=0.28, warp=(0.3, 1.0)),
    ], seed=SEED + 31, band=0.2)
    sdf.noise(N3, 0.008, 9.0, octaves=3, offset=(0.9, 5.5, 3.3), ridged=True)
    return sdf.to_object("rock_canyon_low__fg", coll, mat, target_tris=15000, smooth_iter=1)


# ------------------------------------------------------------------------------------------
# COLOR_0
# ------------------------------------------------------------------------------------------

def zone_weight(name, P, N3, feather=0.035):
    c = main_cam()
    u, v, _ = c.project(P)
    w = np.zeros(len(P))
    # warp the zone borders a little so they are not screen-aligned rectangles
    wu = u + 0.025 * N3.fbm(P * 2.3 + 11.0, 2)
    wv = v + 0.025 * N3.fbm(P * 2.3 + 37.0, 2)
    for (u0, v0, u1, v1, wt) in ZONES.get(name, ()):
        z = (C.smoothstep(u0 - feather, u0 + feather, wu) * (1.0 - C.smoothstep(u1 - feather, u1 + feather, wu)) *
             C.smoothstep(v0 - feather, v0 + feather, wv) * (1.0 - C.smoothstep(v1 - feather, v1 + feather, wv)))
        w = np.maximum(w, wt * z)
    return w


def color_rock(obj, bvh, N3, use_ao=True):
    me = obj.data
    co, nrm, ed, _ = C.mesh_arrays(me)
    ao = C.ray_ao(co, nrm, bvh, n_rays=12, max_dist=0.6, bias=0.006) if use_ao else np.ones(len(co))
    cx = C.surface_convexity(co, nrm, ed, iterations=6)
    ledge = C.smoothstep(0.42, 0.78, nrm[:, 2])
    zone = zone_weight(obj.name, co, N3)
    patch = C.smoothstep(-0.12, 0.3, N3.fbm(co * 3.2 + 5.0, 3))
    R = ledge * zone * (0.25 + 0.75 * patch) * C.smoothstep(0.35, 0.8, ao)
    R = np.where(R < 0.04, 0.0, R)
    G = np.clip(0.32 + 0.3 * C.smoothstep(-0.3, 0.4, N3.fbm(co * 2.0 + 21.0, 2)) + 0.25 * zone, 0.15, 1.0)
    B = ao * (0.78 + 0.22 * np.clip(0.5 + cx / 0.02, 0.0, 1.0))
    if obj.name in DARKEN:
        B = B * DARKEN[obj.name]
    C.write_color0(me, R, G, B)
    return dict(R=R, G=G, B=B)


def build_moss(coll, rocks, mat, N3):
    """Moss islands on ledge points of the plant zones: Poisson-disc spaced seeds (deterministic), each
    a lumpy cluster of flat cushions hugging the rock (sub-cushions snapped to the rock surface)."""
    out = []
    for name, spec in MOSS.items():
        rock = rocks[spec["rock"]]
        co, nrm, _, _ = C.mesh_arrays(rock.data)
        col = C.read_color0(rock.data)
        score = col[:, 0] * C.smoothstep(0.55, 0.85, nrm[:, 2])
        cand = np.nonzero(score > 0.3)[0]
        if not len(cand):
            continue
        rng = np.random.default_rng(SEED + len(name))
        order = cand[np.argsort(-(score[cand] + 0.35 * rng.random(len(cand))))]
        pick = []
        for i in order:
            if all(np.linalg.norm(co[i] - co[j]) > spec["spacing"] for j in pick):
                pick.append(i)
            if len(pick) >= spec["count"]:
                break
        bvh = C.build_bvh([rock])
        pts, nrms, rads, ths = [], [], [], []
        for i in pick:
            r0 = rng.uniform(*spec["radius"])
            t0 = rng.uniform(*spec["thick"])
            n0 = C.normalize(nrm[i] * 0.7 + np.array([0.0, 0.0, 0.3]))
            ta, tb = L.perp_basis(n0)
            subs = [(np.zeros(2), 1.0)] + [(rng.normal(size=2) * 0.55, rng.uniform(0.45, 0.75))
                                           for _ in range(int(rng.integers(2, 5)))]
            for off, sc in subs:
                q = co[i] + (ta * off[0] + tb * off[1]) * r0
                hit = bvh.find_nearest(q.tolist())
                if hit[0] is None:
                    continue
                p = np.array(hit[0])
                n = C.normalize(np.array(hit[1]) * 0.6 + n0 * 0.4)
                t = t0 * rng.uniform(0.7, 1.0)
                pts.append(p + n * 0.3 * t)
                nrms.append(n)
                rads.append(r0 * sc)
                ths.append(t)
        sdf = L.blob_field(np.array(pts), np.array(nrms), np.array(rads), np.array(ths), voxel=0.007, noise=N3,
                           k=0.03, lumps=[(0.009, 18.0, 3), (0.003, 55.0, 2)])
        obj = sdf.to_object(name, coll, mat, target_tris=spec["tris"], smooth_iter=1)
        out.append(obj)
    return out


def color_moss(obj, bvh, N3, use_ao=True):
    me = obj.data
    co, nrm, ed, _ = C.mesh_arrays(me)
    ao = C.ray_ao(co, nrm, bvh, n_rays=12, max_dist=0.3, bias=0.004) if use_ao else np.ones(len(co))
    cx = C.surface_convexity(co, nrm, ed, iterations=5)
    up = C.smoothstep(-0.2, 0.6, nrm[:, 2])
    R = np.clip((0.55 + 0.4 * up) * (0.8 + 0.2 * C.smoothstep(-0.3, 0.3, N3.fbm(co * 6.0 + 3.0, 2))), 0, 1)
    G = np.clip(0.35 + 0.35 * C.smoothstep(-0.3, 0.4, N3.fbm(co * 4.0 + 9.0, 2)) + 0.1 * up, 0.15, 1.0)
    B = ao * (0.85 + 0.15 * np.clip(0.5 + cx / 0.008, 0.0, 1.0))
    C.write_color0(me, R, G, B)


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


def build_keys(coll):
    d = C.normalize(np.asarray(KEY_DIR, float))
    target = S(0.86, 0.22, 4.5)
    C.make_key_empty("key_canyon", coll, tuple(target - d * 2.5), tuple(d),
                     extras=dict(target_m=[round(float(x), 3) for x in C.blender_to_gltf_dir(target)],
                                 note="narrow key: right face + ledges, grazes the left edge, centre unlit"))


def build(use_ao=True):
    global MAIN
    MAIN = None
    t0 = time.time()
    scene = C.ensure_scene(SCENE_NAME)
    coll = C.ensure_collection(scene, COLL_NAME)
    C.purge_collection(coll)
    mat_rock = C.preview_material("mat_rock", C.PALETTE["stone"], 0.9)
    mat_moss = C.preview_material("mat_moss", C.PALETTE["moss"], 0.95)
    N3 = C.Noise3(SEED)
    build_cameras(coll)
    build_keys(coll)
    rocks = {}
    for fn in (build_right, build_left, build_low):
        t = time.time()
        o = fn(coll, mat_rock, N3)
        dropped, parts = L.drop_small_islands(o, 0.02)
        # relax the zig-zag the decimation leaves along thin lit ridges (2 Taubin steps, volume kept)
        co, _, ed, _ = C.mesh_arrays(o.data)
        C.set_vertex_positions(o.data, C.taubin_smooth(co, ed, iterations=2))
        C.box_uvs(o.data, 1.0)
        rocks[o.name] = o
        C.log(f"{o.name}: {C.tri_count(o)} tris, {parts} parts, {dropped} crumb vertices dropped "
              f"({time.time() - t:.1f}s)")
    t = time.time()
    bvh = C.build_bvh(list(rocks.values()))
    stats = {}
    for o in rocks.values():
        stats[o.name] = color_rock(o, bvh, N3, use_ao)
    mosses = build_moss(coll, rocks, mat_moss, N3)
    bvh2 = C.build_bvh(list(rocks.values()) + mosses) if use_ao else None
    for o in mosses:
        C.box_uvs(o.data, 0.5)
        color_moss(o, bvh2, N3, use_ao)
        C.log(f"{o.name}: {C.tri_count(o)} tris")
    C.log(f"colours + moss ({time.time() - t:.1f}s)")
    total = sum(C.tri_count(o) for o in coll.objects if o.type == "MESH")
    C.log(f"canyon built: {total} tris, {time.time() - t0:.1f}s")
    return scene, coll


# ------------------------------------------------------------------------------------------
# previews / comparison
# ------------------------------------------------------------------------------------------

ROCK_ALBEDO = C.hex_to_linear("#8E8371")
MOSS_ALBEDO = C.hex_to_linear("#4B5A1C")
PLANT_TINT = C.hex_to_linear("#3F5418")
KEY_E, FILL_E = 1.35, 0.06


def preview_colours(coll):
    """Per-vertex preview colours (linear): albedo x (fill x AO + key x Lambert x shadow)."""
    meshes = [o for o in coll.objects if o.type == "MESH"]
    bvh = C.build_bvh(meshes)
    out = {}
    for o in meshes:
        co, nrm, ed, _ = C.mesh_arrays(o.data)
        col = C.read_color0(o.data)
        # per-vertex shadow rays are binary: soften them over the 1-ring (else lit ridges turn into
        # saw teeth in these flat previews)
        sh = L.shadow_rays(bvh, co, nrm, KEY_DIR, bias=0.01)
        for _ in range(2):
            sh = 0.5 * sh + 0.5 * C.neighbor_mean(sh, ed, len(sh))
        lit = L.lambert(nrm, KEY_DIR, wrap=0.05) * sh
        sky = 0.5 + 0.5 * nrm[:, 2]
        if o.name.startswith("moss"):
            alb = np.broadcast_to(MOSS_ALBEDO, (len(co), 3))
        else:
            r = col[:, 0:1]
            alb = np.asarray(ROCK_ALBEDO)[None, :] * (1 - r) + np.asarray(PLANT_TINT)[None, :] * r
        # the engine multiplies the albedo by COLOR_0.B (baked AO): direct and ambient light alike
        light = (FILL_E * sky + KEY_E * lit) * col[:, 2]
        out[o.name] = alb * light[:, None]
    return out


def render_previews(scene, coll, cams=None, percent=50, suffix=""):
    setup_cams = [coll.objects[n] for n in (cams or list(CAMS.keys()))]
    cols = preview_colours(coll)
    L.setup_flat(scene, percent=percent, bg=(0.0, 0.0, 0.0))
    out = []
    with L.FlatPreview(scene, "canyon_preview") as pv:
        for name, rgb in cols.items():
            L.preview_copy(coll.objects[name], pv.coll, rgb)
        for co in setup_cams:
            path = os.path.join(CAPTURES, f"canyon_{co.name}{suffix}.png")
            C.render_camera(scene, co, path)
            out.append(path)
    return out


def reference_masks(ref):
    """Frame 06 (half size): lit mask (blurred luminance > 0.07) and visible mask (> 0.02), plus the
    valid area (outside the HTML: nav bar, headline / copy block, the three cards, the logo)."""
    a = ref[..., :3]
    lum = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
    lb = L.box_blur(lum, 3)
    h, w = lum.shape
    valid = np.ones((h, w), bool)
    for (u0, v0, u1, v1) in ((0.36, 0.02, 0.64, 0.10), (0.02, 0.03, 0.10, 0.09), (0.83, 0.02, 0.98, 0.09),
                             (0.27, 0.10, 0.73, 0.40), (0.04, 0.44, 0.34, 0.96), (0.35, 0.44, 0.65, 0.96),
                             (0.66, 0.44, 0.96, 0.96)):
        valid[int(v0 * h):int(v1 * h), int(u0 * w):int(u1 * w)] = False
    return lb > 0.07, lb > 0.02, valid


def compare_sheets(scene, coll, percent=50):
    """canyon_cmp_main: frame 06 | preview | frame 06 x0.55 with the reference lit area (cyan), our lit
    area (yellow), our geometry silhouette (orange) and a 0.1 grid; IoU of the lit areas outside the
    HTML. canyon_cmp_poses: in | main | out previews."""
    os.makedirs(TMP, exist_ok=True)
    ref = L.down2(L.load_rgba(os.path.join(REF, "frames/06_dark_canyon.png")))
    h, w = ref.shape[:2]
    rlit, rvis, valid = reference_masks(ref)
    masks = L.render_masks(scene, [coll.objects["cam_canyon_main"]], TMP,
                           [o for o in coll.objects if o.type == "MESH"], percent)
    geo = L.load_rgba(masks["cam_canyon_main"])[..., 3] > 0.5
    pp = os.path.join(CAPTURES, "canyon_cam_canyon_main.png")
    prev = L.load_rgba(pp)[..., :3] if os.path.exists(pp) else np.zeros_like(ref[..., :3])
    plum = 0.2126 * prev[..., 0] + 0.7152 * prev[..., 1] + 0.0722 * prev[..., 2]
    plit = L.box_blur(plum, 3) > 0.07
    ov = ref[..., :3] * 0.55
    ov[L.outline(geo)] = (1.0, 0.55, 0.1)
    ov[L.outline(rlit)] = (0.2, 0.95, 1.0)
    ov[L.outline(plit)] = (1.0, 0.92, 0.1)
    L.grid(ov)
    path = os.path.join(CAPTURES, "canyon_cmp_main.png")
    L.save_rgb(np.concatenate([ref[..., :3], prev, ov], 1), path)
    stats = dict(iou_lit=round(L.iou(rlit, plit, valid), 4), iou_visible_vs_geometry=round(L.iou(rvis, geo, valid), 4),
                 lit_share_ref=round(float(rlit[valid].mean()), 4), lit_share_ours=round(float(plit[valid].mean()), 4))
    # edges of the right face / left edge along rows (frame fractions)
    rows = {}
    for v in (0.05, 0.15, 0.25, 0.35, 0.42):
        y = int(v * h)
        r_ref = np.nonzero(rlit[y, int(0.6 * w):])[0]
        r_our = np.nonzero(plit[y, int(0.6 * w):])[0]
        l_ref = np.nonzero(rvis[y, :int(0.2 * w)])[0]
        l_our = np.nonzero(geo[y, :int(0.2 * w)])[0]
        rows[f"v={v}"] = dict(right_face_left_edge=[round((r_ref[0] + int(0.6 * w)) / w, 3) if len(r_ref) else None,
                                                    round((r_our[0] + int(0.6 * w)) / w, 3) if len(r_our) else None],
                              left_edge_right_extent=[round(l_ref[-1] / w, 3) if len(l_ref) else None,
                                                      round(l_our[-1] / w, 3) if len(l_our) else None])
    stats["rows_ref_vs_ours"] = rows
    panels = []
    for n in ("cam_canyon_in", "cam_canyon_main", "cam_canyon_out"):
        p = os.path.join(CAPTURES, f"canyon_{n}.png")
        im = L.load_rgba(p)[..., :3] if os.path.exists(p) else np.zeros_like(ref[..., :3])
        im = im.copy()
        L.grid(im)
        panels.append(im)
    p2 = os.path.join(CAPTURES, "canyon_cmp_poses.png")
    L.save_rgb(np.concatenate(panels, 1), p2)
    C.log("compare:", json.dumps(stats))
    return [path, p2], stats


# ------------------------------------------------------------------------------------------
# validation
# ------------------------------------------------------------------------------------------

def validate_canyon(rep, coll):
    errs, info = [], {}
    names = {n["name"] for n in rep["nodes"]}
    req = ["rock_canyon_right", "rock_canyon_left", "rock_canyon_low__fg", "cam_canyon_main", "cam_canyon_in",
           "cam_canyon_out", "key_canyon"] + list(MOSS.keys())
    for r in req:
        if r not in names:
            errs.append(f"missing node {r}")
    if rep["totals"]["tris"] > MAX_TRIS:
        errs.append(f"total tris {rep['totals']['tris']} > {MAX_TRIS}")
    for cname, c in rep["cameras"].items():
        ex = c.get("extras") or {}
        for k in ("focus_distance_m", "frame_height_at_focus_m", "video_time_s"):
            if k not in ex:
                errs.append(f"{cname}: extras without {k}")
    # COLOR_0 per mesh: R only on ledges of the plant zones of the rocks (share of vertices with R > 0)
    ranges = {}
    for o in coll.objects:
        if o.type != "MESH":
            continue
        col = C.read_color0(o.data)
        ranges[o.name] = dict(min=[round(float(x), 3) for x in col.min(0)], max=[round(float(x), 3) for x in col.max(0)],
                              mean=[round(float(x), 3) for x in col.mean(0)],
                              share_R_gt0=round(float((col[:, 0] > 0).mean()), 4))
        if o.name.startswith("rock") and (col[:, 0] > 0).mean() > 0.25:
            errs.append(f"{o.name}: vegetation density on {100 * (col[:, 0] > 0).mean():.0f} % of the vertices")
    info["color0"] = ranges
    # the centre column of the frame stays empty in every pose (no geometry between u 0.3 and 0.65)
    mesh_objs = [o for o in coll.objects if o.type == "MESH"]
    for cname in CAMS:
        c = cam(cname)
        hits = 0
        for o in mesh_objs:
            co, _, _, _ = C.mesh_arrays(o.data)
            u, v, z = c.project(co)
            hits += int(np.sum((z > 0.1) & (u > 0.3) & (u < 0.65) & (v > 0.0) & (v < 1.0)))
        info[f"{cname}_centre_vertices"] = hits
        if hits:
            errs.append(f"{cname}: {hits} vertices in the empty centre (u 0.3-0.65)")
    return dict(errors=errs, **info)


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
    scene, coll = build(use_ao=a["ao"])
    result = dict(tris=sum(C.tri_count(o) for o in coll.objects if o.type == "MESH"))
    if a["export"]:
        C.export_glb(scene, GLB_PATH)
        result["glb"] = GLB_PATH
        result["glb_bytes"] = os.path.getsize(GLB_PATH)
    if a["validate"]:
        rep = C.validate_glb(GLB_PATH, max_total_tris=MAX_TRIS)
        extra = validate_canyon(rep, coll)
        rep["canyon"] = extra
        result["validate"] = dict(errors=rep["errors"] + extra["errors"], warnings=rep["warnings"],
                                  totals=rep["totals"])
        os.makedirs(CAPTURES, exist_ok=True)
        with open(os.path.join(CAPTURES, "canyon_validate.json"), "w") as f:
            json.dump(rep, f, indent=1, default=str)
    if a["render"] is not None or a["compare"]:
        result["renders"] = render_previews(scene, coll, a["render"] or None, a["percent"])
    if a["compare"]:
        result["compare"], result["compare_stats"] = compare_sheets(scene, coll, a["percent"])
    if a["save"]:
        save_blend()
    C.log("result", json.dumps(result, default=str))
    return result


if __name__ == "__main__":
    _argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    main(_argv)
