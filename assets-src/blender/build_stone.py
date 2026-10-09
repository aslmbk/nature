"""
build_stone.py - the 'stone' scene set (S07, video 37-43.3 s): one big diagonal rock fragment in the lower
right of the frame, the Silva seed recessed 4 cm into its face with a 4 mm chamfer on the rim (the beige-gold
recess floor is its own node, the walls belong to the rock) and a lumpy moss strip along its upper edge that
thins to a fringe on the right, exported to public/nature/models/stone.glb (contract: CLAUDE.md "Asset
contract").

Run headless (from the repo root):
    "/c/Program Files/Blender Foundation/Blender 5.1/blender.exe" --background --factory-startup \
        --python assets-src/blender/build_stone.py -- --export --validate [--render] [--compare] [--save]

Run inside a live Blender session (e.g. through MCP):
    import sys, importlib; sys.path.insert(0, r"<repo>/assets-src/blender")
    import build_stone; importlib.reload(build_stone); build_stone.main(["--export", "--validate"])

Options:
    --export            write public/nature/models/stone.glb
    --validate          re-read / re-import the GLB, check it against the contract + stone checks
                        (report: docs/captures/blender/stone_validate.json)
    --render [cams]     lit EEVEE previews -> docs/captures/blender/stone_<cam>.png (all cams if none given)
    --compare           comparison sheets -> docs/captures/blender/stone_cmp_<what>.png (implies --render)
    --percent N         preview size in % of 1440x1020 (default 50)
    --save              save nature.blend (headless: opens the existing file first so the other scene sets
                        are kept; only the 'stone' scene / collection is rebuilt)
    --no-ao             skip the ray-traced AO (B channel = cavity only; fast iteration)
    --glb PATH          export / validate PATH instead of public/nature/models/stone.glb (iterations,
                        determinism checks)

Everything is deterministic (seed 134). World units: metres, Blender Z-up, cameras look along +Y.

Measured from the reference (1440x1020 px; frames 13 / 14, clip 10; brightened crops):
  * frame 14 (41.7 s, cam_stone_main): rock silhouette (left / upper edge, under the moss) -> ROCK_OUTLINE_UV;
    the moss strip runs along the upper edge from (915, 760) px to the right frame edge at (1440, 333) px,
    40-60 px wide, hanging over the steep upper-left edge; recessed glyph centre (1097, 742) px, ~260 px tall
    along its axis, leaning 16 deg clockwise; crease between the lit front face and the shaded right face at
    x ~1245 px; the rock's rounded underside fades out at y ~1050-1090 px (seen in the exit frames).
    Background near-black (#010101-#110F0B), lit face ~#94877A, recess floor ~#8C785C, shaded face ~#373027.
  * frame 13 (38.5 s, cam_stone_in): the same composition 0.291 frame heights lower (glyph top 925 vs 628 px)
    -> a pedestal camera 0.64 m higher. Frame 13 shows no moss on the edge yet (the moss is its own node).
  * clip 10 (43.0-43.6 s): the composition slides up; glyph bottom 870 -> 794 -> 580 -> 372 -> 216 -> 102
    -> 14 px every 0.1 s. 43.2 s = 0.284 frame heights -> cam_stone_out 0.62 m lower; the measured track
    is in its extras (exit_track: [video s, share of the frame height moved up]).
  * brief: frame height ~2.2 m at the focus plane -> camera 3.49 m from the glyph, 35 deg vertical fov.
  * revision 2 (frame 14 / brightened crops): the recess is deep enough for lit walls (4 cm, 4 mm chamfer);
    the face right of x ~1245 px rolls away in one smooth turn (CREASE_K) instead of a cut fissure; the moss
    is a thick cushion up the left edge and over the corner and only a thin fringe of strands along the
    upper edge (FRINGE_U); the face carries coarser relief with shallow pits (PITS).
  * revision 3: the upper edge right of x ~1220 px is raised to frame 14's line (moss top per column there:
    408 px at x 1240, 386 at 1320, 346 at 1400, 315 at 1439; the outline sits ~10 px under it, the fringe
    on top) and runs on into the top-right corner: the back plane (ROCK_DEPTH) moved back so the receding
    right face no longer ends in a shoulder at x ~1400 px. The AO along the recess rim ignores the recess
    (face-side vertices take the occlusion of the uncut rock with the field normal; LIP_B_MIN within 3 mm
    of the rim edge, CHAMFER_B_MIN on the chamfer, FLOOR_B_MIN on the floor along the walls). The dark
    dashes along the lip in the engine came from the normals, not from COLOR_0: the boolean leaves sub-mm
    sliver triangles in the face plane along the rim edge inside the cut chamfer n-gons, and these carried
    the chamfer's 45 deg normal (about 10 % of the face-plane corners on the rim edge in the GLB). The
    polygons along the rim are triangulated and get analytic corner normals (custom normals, fix_rim_normals):
    face plane n, chamfer (n - g) / |n - g|, walls -g (g = outward direction of the seed outline in the face
    plane), with sharp edges between the three.
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
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import common as C  # noqa: E402
import stonelib as SL  # noqa: E402

importlib.reload(C)
importlib.reload(SL)

ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
GLB_PATH = os.path.join(ROOT, "public", "nature", "models", "stone.glb")
BLEND_PATH = os.path.join(HERE, "nature.blend")
CAPTURES = os.path.join(ROOT, "docs", "captures", "blender")
REF = os.path.join(ROOT, "docs", "nature-webgl-reference")      # comparison sheets only (never shipped)
SVG = os.path.join(ROOT, "assets-src", "emblem", "seed.svg")
TMP = os.path.join(tempfile.gettempdir(), "silva_stone_build")
SCENE_NAME = "stone"
COLL_NAME = "stone"
SEED = 134

TV = math.tan(math.radians(C.FOV_V_DEG) / 2.0)
TH = TV * C.ASPECT

BG_HEX = "#050403"          # near-black, faintly warm (frame 14)

# ------------------------------------------------------------------------------------------
# cameras: one lens, pedestal moves only (the reference composition slides vertically)
# ------------------------------------------------------------------------------------------
FOCUS = 3.49                                    # m, camera -> glyph centre
FRAME_H = 2.0 * FOCUS * TV                      # 2.2 m at the focus plane
CAM_POSES = {
    "cam_stone_in": dict(shift=0.291, t=38.5),      # frame 13: composition 0.291 frame heights lower
    "cam_stone_main": dict(shift=0.0, t=41.7),      # frame 14
    "cam_stone_out": dict(shift=-0.284, t=43.2),    # clip 10: exit slide, 0.284 frame heights up
}
EXIT_TRACK = [(43.0, 0.0), (43.1, 0.0745), (43.2, 0.284), (43.3, 0.488), (43.4, 0.641), (43.5, 0.753),
              (43.6, 0.839)]
KEY_DIR = (0.55, 0.40, -0.73)                   # warm key from the upper left, in front (travel direction)
KEY_LOC = (-1.2, -2.6, 2.3)

# ------------------------------------------------------------------------------------------
# the rock
# ------------------------------------------------------------------------------------------
N_FRONT = C.normalize(np.array([-0.30, -0.88, 0.37]))     # lit front face: towards the camera, up, left
CREASE_DEG = 48.0           # the face right of the crease turns away by this much (dimmer under the key)
CREASE_K = 0.12             # rounding (m) of that turn: one smooth roll, no fissure (planar glyph face kept)
PITS = 110                  # shallow pits 2.4-5 cm across on the visible faces (away from the glyph)
ROLL_SLOPE = 0.6            # below / left of the glyph the face rolls away (convex boulder, darker lower left)
ROLL_FROM = 0.33            # ... starting this far (m) from the glyph centre
FACE_CONCAVE = 0.05         # the front face is slightly concave (m of sag per m^2 away from the glyph)
ROCK_DEPTH = 1.0            # back plane behind the front face (never seen; deep enough that the receding
                            # right face reaches the outline up to the top-right corner)
EDGE_R = 0.10               # rounding of the face edges (boulder-like roll-over)
EMBLEM_UV = (0.762, 0.727)  # glyph centre in frame 14 (on the focus plane)
# the dark fissure right of the glyph (frame 14, brightened): top edge -> curving right at the bottom
CREASE_PATH_UV = [(0.853, 0.47), (0.86, 0.56), (0.866, 0.65), (0.873, 0.74), (0.885, 0.82), (0.905, 0.89),
                  (0.935, 0.96), (0.97, 1.03), (1.0, 1.08)]
VOX = 0.012

# silhouette in cam_stone_main (u, v): underside (below the frame, seen in the exit pose), left edge,
# upper edge under the moss (moss outer edge + ~22 px), off-frame right
ROCK_OUTLINE_UV = [
    (1.30, 1.075), (1.0, 1.069), (0.903, 1.064), (0.799, 1.054), (0.694, 1.039), (0.639, 1.02), (0.604, 0.98),
    (0.594, 0.931), (0.599, 0.882), (0.608, 0.833), (0.615, 0.784), (0.625, 0.745), (0.6375, 0.711),
    (0.653, 0.68), (0.669, 0.632), (0.692, 0.588), (0.712, 0.549), (0.729, 0.51), (0.741, 0.48), (0.752, 0.462),
    (0.79, 0.468), (0.82, 0.468), (0.8455, 0.4422), (0.8656, 0.4157), (0.8892, 0.402), (0.917, 0.3892),
    (0.9448, 0.3696), (0.9726, 0.35), (1.0, 0.3186), (1.06, 0.268), (1.16, 0.183), (1.30, 0.065)]
MOSS_FROM_V = 0.76          # the moss strip covers the outline above this v (left edge) ...
MOSS_TO_U = 1.16            # ... up to this u (off frame right)
FRINGE_U = (0.79, 0.86)     # right of the corner the strip thins to a fringe (screen u, cam_stone_main)

# the seed: recessed into the front face
EMBLEM_IMG_H = 0.26         # projected tip-to-tip length in frame heights (frame 14 glyph: 260-270 px,
                            # ~45 % of the visible slab height); EMBLEM_H (m) is solved from it
EMBLEM_LEAN = 16.0          # deg clockwise in the image (frame 14 glyph axis)
EMBLEM_DEPTH = 0.04         # recess depth below the face (m)
EMBLEM_BEVEL = 0.004        # 45 deg chamfer on the recess rim (m)
EMBLEM_FLAT = 0.04          # the face is exactly planar within this distance of the seed outline
LIP_ZONE = 0.02             # face-side band (m outside the outline) whose AO ignores the recess
LIP_B_MIN = 0.45            # COLOR_0.B never lower within 3 mm outside the rim edge (face side)
CHAMFER_B_MIN = 0.35        # ... nor on the 45 deg chamfer
FLOOR_B_MIN = 0.3           # the recess floor's contact shadow along the walls never darker than this
RIM_ZONE = 0.012            # fix_rim_normals works within this distance of the seed outline (m)
RIM_PLANE_TOL = 0.0006      # triangles whose corners all lie this close to the face plane are face (m)

EMBLEM_H = 0.55             # starting value, replaced by solve_emblem()
CAM = {}                    # filled by cameras()
BASIS = {}                  # front plane basis


def cam_q():
    return C.cam_quat(0.0, 0.0, 0.0)


def cameras():
    for name, p in CAM_POSES.items():
        CAM[name] = dict(loc=(0.0, -FOCUS, p["shift"] * FRAME_H), t=p["t"], shift=p["shift"])
    return CAM


def ray_to_plane(u, v, point, normal, cam="cam_stone_main"):
    loc = np.asarray(CAM[cam]["loc"], float)
    d = C.screen_to_world(loc, cam_q(), u, v, 1.0) - loc
    n = np.asarray(normal, float)
    t = float((np.asarray(point, float) - loc) @ n) / float(d @ n)
    return loc + t * d


def setup_plane():
    loc = np.asarray(cameras()["cam_stone_main"]["loc"], float)
    P0 = C.screen_to_world(loc, cam_q(), EMBLEM_UV[0], EMBLEM_UV[1], FOCUS)
    ex = C.normalize(np.cross(np.array([0.0, 0.0, 1.0]), N_FRONT))
    ey = np.cross(N_FRONT, ex)
    BASIS.update(P0=P0, ex=ex, ey=ey)
    cp = np.array([ray_to_plane(u, v, P0, N_FRONT) for (u, v) in CREASE_PATH_UV])
    d = C.normalize(cp[5] - cp[1])
    m = C.normalize(np.cross(d, N_FRONT))
    if m[0] < 0:
        m = -m
    a = math.radians(CREASE_DEG)
    BASIS.update(crease=cp[1], crease_path=cp, n_right=C.normalize(N_FRONT * math.cos(a) + m * math.sin(a)))
    BASIS["outline3"] = np.array([ray_to_plane(u, v, P0, N_FRONT) for (u, v) in ROCK_OUTLINE_UV])
    BASIS["outline2"] = to2(BASIS["outline3"])
    return BASIS


def outline_tan(uv=None):
    """Screen polygon (u, v) -> tangent space of cam_stone_main (pitch 0: x right, z up)."""
    return np.array([((2.0 * u - 1.0) * TH, (2.0 * v - 1.0) * TV) for (u, v) in (uv or ROCK_OUTLINE_UV)])


def ray_hit(sdf, u, v, cam="cam_stone_main", t0=2.4, t1=5.6):
    """First surface point of `sdf` along the camera ray through (u, v) (sphere tracing) or None."""
    loc = np.asarray(CAM[cam]["loc"], float)
    d = C.normalize(C.screen_to_world(loc, cam_q(), u, v, 1.0) - loc)
    t = t0
    for _ in range(400):
        p = loc + d * t
        dist = float(sdf.sample(p[None])[0])
        if dist < 0.0005:
            return SL.project_to_surface(sdf, p[None])[0], d
        t += max(dist * 0.9, 0.0015)
        if t > t1:
            return None, d
    return None, d


def to2(P):
    d = np.atleast_2d(P) - BASIS["P0"]
    return np.stack([d @ BASIS["ex"], d @ BASIS["ey"]], -1)


def emblem_frame(rot, height=None):
    return SL.EmblemFrame(BASIS["P0"], N_FRONT, (0.0, 0.0, 1.0), height or EMBLEM_H, rot_deg=rot)


def image_length(frame):
    """Projected tip-to-tip length of the seed in cam_stone_main, in frame heights."""
    a = frame.world(np.array([[0.0, -50.0], [0.0, 50.0]]), np.zeros(2))
    u, v, _ = C.project_to_screen(CAM["cam_stone_main"]["loc"], cam_q(), a)
    return float(math.hypot((u[1] - u[0]) * C.ASPECT, v[1] - v[0]))


def image_lean(frame):
    """Lean of the seed axis in cam_stone_main, degrees clockwise (top to the right)."""
    a = frame.world(np.array([[0.0, -50.0], [0.0, 50.0]]), np.zeros(2))
    u, v, _ = C.project_to_screen(CAM["cam_stone_main"]["loc"], cam_q(), a)
    du, dv = (u[1] - u[0]) * C.ASPECT, -(v[1] - v[0])
    return math.degrees(math.atan2(du, dv))


def solve_emblem_rot():
    lo, hi = -10.0, 45.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if image_lean(emblem_frame(mid)) < EMBLEM_LEAN:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_emblem():
    """In-plane rotation for the measured lean, height (m) for the measured projected length."""
    global EMBLEM_H
    for _ in range(4):
        rot = solve_emblem_rot()
        EMBLEM_H *= EMBLEM_IMG_H / image_length(emblem_frame(rot))
    return solve_emblem_rot(), EMBLEM_H


def _ramp(x, w):
    """C1 ramp: 0 for x <= 0, x^2 / 2w up to w, then x - w / 2."""
    x = np.asarray(x, float)
    return np.where(x <= 0.0, 0.0, np.where(x < w, x * x / (2.0 * w), x - 0.5 * w))


def emblem_dist_m(P, frame):
    q, _h = frame.local(P)
    return SL.seed_sdf2(q) * frame.s


def build_rock(coll, mat, N3, frame):
    lo = np.array([0.15, -0.82, -2.2])
    hi = np.array([2.1, 1.4, 0.9])             # y: the receding right face reaches ~1.3 m at the top-right corner
    sdf = C.SDF(lo, hi, VOX)
    P0, crease, n_right = BASIS["P0"], BASIS["crease"], BASIS["n_right"]
    cam = np.asarray(CAM["cam_stone_main"]["loc"], float)
    V = outline_tan()

    def field(P):
        # outline as a cone from cam_stone_main (visual hull): the silhouette matches at every depth,
        # also where the faces right of the fissure recede
        rel = P - cam
        z = np.maximum(rel[:, 1], 0.05)
        d2 = C.poly_sdf(np.stack([rel[:, 0] / z, -rel[:, 2] / z], 1), V) * z
        d_face = SL.plane_dist(P, P0, N_FRONT, FACE_CONCAVE, P0)
        q2 = to2(P)
        t_dl = -(0.5 * q2[:, 0] + 0.866 * q2[:, 1])                    # down-left of the glyph
        d_face = d_face + ROLL_SLOPE * _ramp(t_dl - ROLL_FROM, 0.25)
        d_plane = (P - P0) @ N_FRONT
        w = 1.0 - C.smoothstep(EMBLEM_FLAT, EMBLEM_FLAT + 0.07, emblem_dist_m(P, frame))
        d_face = d_face + w * (d_plane - d_face)
        wx, wy = d2 + EDGE_R, d_face + EDGE_R
        f = np.minimum(np.maximum(wx, wy), 0.0) + np.hypot(np.maximum(wx, 0.0), np.maximum(wy, 0.0)) - EDGE_R
        f = C.smax(f, (P - crease) @ n_right, CREASE_K)
        return np.maximum(f, -d_plane - ROCK_DEPTH)

    nz = sdf.shape[2]
    _sl, Pall = sdf._box(lo, hi)
    for k0 in range(0, nz, 16):
        k1 = min(k0 + 16, nz)
        P = Pall[:, :, k0:k1]
        sdf.d[:, :, k0:k1] = field(P.reshape(-1, 3)).reshape(P.shape[:-1]).astype(np.float32)
    del Pall

    rng = np.random.default_rng(SEED + 301)

    def away(P, margin=EMBLEM_FLAT + 0.04):
        return C.smoothstep(margin, margin + 0.05, emblem_dist_m(P, frame))

    def visible(P):
        u, v, _ = C.project_to_screen(CAM["cam_stone_main"]["loc"], cam_q(), P)
        return ((u > 0.55) & (u < 1.08) & (v > 0.2) & (v < 1.32)).astype(float)

    # chipped edges (not on the open right face: a chip there reads as a crater)
    def edge_w(P):
        n = SL.sdf_normals(sdf, P)
        return (1.0 - np.abs(n @ N_FRONT)) ** 2 * (1.0 - C.smoothstep(0.6, 0.85, n @ n_right)) * away(P) * visible(P)

    pts, nrm = SL.surface_samples(sdf, 24, rng, weight=edge_w)
    for p, n in zip(pts, nrm):
        tilt = C.normalize(n + 0.45 * C.normalize(rng.normal(size=3)))
        SL.chip(sdf, p, tilt, depth=rng.uniform(0.008, 0.02), radius=rng.uniform(0.05, 0.12), k=0.005)
    # cracks: surface walks running down the face, stopped before the glyph
    starts = [(0.70, 0.56), (0.93, 0.47), (0.66, 0.86), (0.97, 0.80), (0.83, 0.97)]
    dirs = [(-0.3, -0.2, -1.0), (0.15, -0.1, -1.0), (-0.2, 0.0, -1.0), (0.4, 0.0, -1.0), (1.0, 0.2, -0.3)]
    cracks = []
    for (u, v), dvec in zip(starts, dirs):
        p = ray_to_plane(u, v, BASIS["P0"], N_FRONT)
        path = SL.surface_walk(sdf, p, dvec, int(rng.integers(9, 16)), 0.03, rng, jitter=0.35)
        keep = emblem_dist_m(path, frame) > EMBLEM_FLAT + 0.05
        path = path[: int(np.argmin(keep)) if not keep.all() else len(path)]
        if len(path) >= 3:
            r = np.linspace(0.008, 0.003, len(path))
            SL.carve_tube(sdf, path - 0.002 * SL.sdf_normals(sdf, path), r, k=0.003)
            cracks.append(path)
    # shallow pits (weathered stone, frame 14): round dents 2.4-5 cm across, 0.35-0.6 of their radius deep
    prng = np.random.default_rng(SEED + 302)
    pts, nrm = SL.surface_samples(sdf, PITS, prng, weight=lambda P: away(P, EMBLEM_FLAT + 0.04) * visible(P))
    for p, n in zip(pts, nrm):
        a = prng.uniform(0.012, 0.025)
        dep = prng.uniform(0.35, 0.6) * a
        rho = (a * a + dep * dep) / (2.0 * dep)
        sdf.sphere(p + n * (rho - dep), rho, k=0.004, mode="subtract")
    # weathering relief, faded out around the glyph (the recess needs a planar face)
    amp = lambda P: away(P, EMBLEM_FLAT)
    sdf.noise(N3, 0.02, 3.0, octaves=2, offset=(1.7, 4.2, 0.3), amp_fn=amp)
    sdf.noise(N3, 0.013, 6.0, octaves=3, offset=(3.3, 1.2, 7.7), amp_fn=amp)
    sdf.noise(N3, 0.006, 12.0, octaves=2, offset=(5.5, 2.2, 0.4), ridged=True, amp_fn=amp)
    sdf.noise(N3, 0.0025, 26.0, octaves=2, offset=(1.1, 9.4, 2.3), amp_fn=amp)
    obj = sdf.to_object("rock_stone_slab", coll, mat, target_tris=100000, smooth_iter=1)
    return obj, sdf, cracks


def cut_emblem(rock, coll, frame, mat_rock, mat_emblem):
    cutter, info = SL.emblem_cutter("silva_stone_cutter", coll, frame, mode="recess", depth=EMBLEM_DEPTH,
                                    bevel=EMBLEM_BEVEL, lip=0.004, above=0.06, ds=0.7, mats=(mat_rock, mat_emblem))
    n0 = C.tri_count(rock)
    SL.boolean_difference(rock, cutter, solver="EXACT", material_mode="TRANSFER")
    C.remove_object(cutter.name)
    floor = SL.split_by_material(rock, mat_emblem.name, "emblem_stone", coll, mat_emblem)
    SL.keep_largest_part(rock)
    rock.data.materials.clear()
    rock.data.materials.append(mat_rock)
    return floor, dict(tris_before=n0, tris_after=C.tri_count(rock),
                       floor_tris=C.tri_count(floor) if floor else 0, cutter=info)


def rim_classes(co_tri, frame):
    """Class of triangles near the rim from their corners' heights over the face plane: 0 face plane (all
    corners within RIM_PLANE_TOL), 1 chamfer (centroid above -EMBLEM_BEVEL), 2 walls."""
    h = frame.local(co_tri.reshape(-1, 3))[1].reshape(-1, 3)
    cls = np.where(h.mean(1) > -EMBLEM_BEVEL, 1, 2)
    cls[np.abs(h).max(1) < RIM_PLANE_TOL] = 0
    return cls


def rim_outward(P, frame, eps=0.0005):
    """Unit outward direction of the seed outline in the face plane at points P (gradient of the offset)."""
    g = np.zeros((len(P), 3))
    for k in range(3):
        e = np.zeros(3)
        e[k] = eps
        g[:, k] = (rim_coords(P + e, frame)[1] - rim_coords(P - e, frame)[1]) / (2.0 * eps)
    n = np.asarray(frame.n, float)
    g -= (g @ n)[:, None] * n[None, :]
    return g / np.maximum(np.linalg.norm(g, axis=1)[:, None], 1e-9)


def fix_rim_normals(obj, frame):
    """Normals along the recess rim. The boolean leaves sub-mm sliver triangles in the face plane along the
    rim edge inside the cut chamfer n-gons, flat-shaded with the chamfer's 45 deg normal: dark dashes along
    the lip under the engine's key light. Here the polygons touching the rim zone are triangulated, each
    triangle is classed by its corners' heights (rim_classes: face plane / chamfer / walls), the rim triangles
    are smooth with sharp edges between the classes, and every rim corner gets the analytic normal of its
    class (custom normals): face n, chamfer (n - g) / |n - g|, walls -g."""
    import bmesh
    me = obj.data
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    h, off = rim_coords(co, frame)
    near = (np.abs(off) < RIM_ZONE) & (h > -EMBLEM_DEPTH - RIM_ZONE) & (h < 0.004)
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    sel = [f for f in bm.faces if any(near[v.index] for v in f.verts)]
    bmesh.ops.triangulate(bm, faces=[f for f in sel if len(f.verts) > 3], quad_method="BEAUTY",
                          ngon_method="BEAUTY")
    bm.faces.ensure_lookup_table()
    bm.faces.index_update()
    bm.verts.ensure_lookup_table()
    rim_faces = [f for f in bm.faces if any(near[v.index] for v in f.verts)]
    tri_co = np.array([[v.co[:] for v in f.verts] for f in rim_faces])
    cls = rim_classes(tri_co, frame)
    fcls = {f.index: int(c) for f, c in zip(rim_faces, cls)}
    for f in rim_faces:
        f.smooth = True
    n_sharp = 0
    for e in bm.edges:
        lf = e.link_faces
        if len(lf) == 2 and lf[0].index in fcls and lf[1].index in fcls and fcls[lf[0].index] != fcls[lf[1].index]:
            e.smooth = False
            n_sharp += 1
    bm.to_mesh(me)
    bm.free()
    me.update()
    # analytic corner normals on the rim triangles
    npoly, nl = len(me.polygons), len(me.loops)
    lt = np.empty(npoly, np.int64)
    me.polygons.foreach_get("loop_total", lt)
    ls = np.empty(npoly, np.int64)
    me.polygons.foreach_get("loop_start", ls)
    lv = np.empty(nl, np.int64)
    me.loops.foreach_get("vertex_index", lv)
    poly = np.repeat(np.arange(npoly), lt)
    rim_poly = np.bincount(poly, weights=near[lv].astype(float), minlength=npoly) > 0
    pid = np.nonzero(rim_poly & (lt == 3))[0]
    loops = ls[pid][:, None] + np.arange(3)[None, :]
    cls = rim_classes(co[lv[loops]], frame)
    cn = np.empty(nl * 3)
    me.corner_normals.foreach_get("vector", cn)
    cn = cn.reshape(-1, 3)
    L = loops.ravel()
    V = lv[L]
    Cc = np.repeat(cls, 3)
    g = rim_outward(co[V], frame)
    n = np.asarray(frame.n, float)[None, :]
    nn = np.where((Cc == 0)[:, None], n, np.where((Cc == 1)[:, None], n - g, -g))
    nn /= np.linalg.norm(nn, axis=1)[:, None]
    cn[L] = nn
    me.normals_split_custom_set(cn.tolist())
    me.update()
    return dict(rim_tris=int(len(pid)), face=int((cls == 0).sum()), chamfer=int((cls == 1).sum()),
                walls=int((cls == 2).sum()), sharp_edges=n_sharp)


def lip_corner_normals(obj, frame):
    """Corner normals of the face-plane polygons on the rim edge (all corners within RIM_PLANE_TOL of the
    plane, one 3-5 mm outside the outline): how many turn more than 20 deg from the stone face (the chamfer
    normal is 45 deg)."""
    me = obj.data
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    h, off = rim_coords(co, frame)
    lipv = (np.abs(h) < 0.0005) & (off > EMBLEM_BEVEL - 0.001) & (off < EMBLEM_BEVEL + 0.001)
    nl, npoly = len(me.loops), len(me.polygons)
    lv = np.empty(nl, np.int64)
    me.loops.foreach_get("vertex_index", lv)
    cn = np.empty(nl * 3)
    me.corner_normals.foreach_get("vector", cn)
    cn = cn.reshape(-1, 3)
    lt = np.empty(npoly, np.int64)
    me.polygons.foreach_get("loop_total", lt)
    poly = np.repeat(np.arange(npoly), lt)
    hmax = np.zeros(npoly)
    np.maximum.at(hmax, poly, np.abs(h[lv]))
    m = lipv[lv] & (hmax < RIM_PLANE_TOL)[poly]
    ang = np.degrees(np.arccos(np.clip(cn[m] @ frame.n, -1.0, 1.0)))
    return dict(corners=int(m.sum()), over_20deg=int((ang > 20.0).sum()),
                max_deg=round(float(ang.max()), 1) if len(ang) else None)


# ------------------------------------------------------------------------------------------
# moss strip along the upper edge
# ------------------------------------------------------------------------------------------

def moss_path(rock_sdf, inset=0.012):
    """Surface points just inside the silhouette (cam_stone_main) along the outline part that carries
    moss, bottom-left -> top-right, with the view direction and the screen-inward direction (3D)."""
    sel = [i for i, (u, v) in enumerate(ROCK_OUTLINE_UV) if v < MOSS_FROM_V and u <= MOSS_TO_U + 1e-6]
    uv = np.array([ROCK_OUTLINE_UV[i] for i in sel])
    sc = np.stack([uv[:, 0] * C.ASPECT, uv[:, 1]], 1)            # isotropic screen units (frame heights)
    seg = np.linalg.norm(np.diff(sc, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.arange(0.0, s[-1], 0.0045)
    pts = np.stack([np.interp(t, s, sc[:, k]) for k in range(2)], 1)
    tan = C.normalize(np.gradient(pts, axis=0))
    inward = np.stack([-tan[:, 1], tan[:, 0]], 1)                 # rock side: below / right of the edge
    if inward[len(inward) // 2, 1] < 0:
        inward = -inward
    P, D, IN = [], [], []
    for (x, y), w in zip(pts, inward):
        q = (x + w[0] * inset) / C.ASPECT, y + w[1] * inset
        hit, d = ray_hit(rock_sdf, q[0], q[1])
        if hit is None:
            continue
        q2 = (x + w[0] * (inset + 0.01)) / C.ASPECT, y + w[1] * (inset + 0.01)
        hit2, _ = ray_hit(rock_sdf, q2[0], q2[1])
        P.append(hit)
        D.append(d)
        IN.append(C.normalize(hit2 - hit) if hit2 is not None else -N_FRONT)
    P = np.array(P)
    ss = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    return P, np.array(D), np.array(IN), ss / ss[-1]


def build_moss(coll, mat, N3, rock_sdf):
    path, view, inw, frac = moss_path(rock_sdf)
    lo = path.min(0) - 0.25
    hi = path.max(0) + 0.25
    sdf = C.SDF(lo, hi, 0.006)
    rng = np.random.default_rng(SEED + 311)
    lumps = 0
    # rows: (metres back along the view ray = onto the top side, metres down the face)
    rows = [(0.0, 0.0), (0.04, 0.0), (0.08, 0.0), (0.12, 0.0), (0.0, 0.04)]
    clump = 0.75 + 0.5 * np.clip(0.5 + N3.line(frac * 30.0, freq=1.0, offset=3.1), 0, 1)
    # right of the corner (FRINGE_U) the strip thins to a fringe: only the edge rows, small low lumps
    u_scr, _v, _z = C.project_to_screen(CAM["cam_stone_main"]["loc"], cam_q(), path)
    fringe = C.smoothstep(FRINGE_U[0], FRINGE_U[1], u_scr)
    row_cut = (2.0, 0.6, 0.12, 0.12, 0.08)                 # a row is dropped where fringe exceeds this
    for i in range(len(path)):
        taper = C.smoothstep(0.0, 0.05, frac[i]) * C.smoothstep(1.0, 0.93, frac[i])
        width = 0.35 + 0.65 * taper
        fr = float(fringe[i])
        for row, (back, inward) in enumerate(rows):
            if row == 4 and rng.uniform() > 0.4:           # some tufts hang over the front edge
                continue
            if row >= 2 and rng.uniform() > (0.75 if row == 2 else 0.5) * width:
                continue
            if fr > row_cut[row]:
                continue
            up_side = 0.025 if back > 0 else 0.0           # rows behind the edge: onto the top side
            up_side += 0.02 * fr                           # the fringe sits right on the edge
            p = path[i] + view[i] * back + inw[i] * (inward - up_side + rng.uniform(-0.008, 0.008))
            p = SL.project_to_surface(rock_sdf, p[None])[0]
            n = SL.sdf_normals(rock_sdf, p[None])[0]
            r = rng.uniform(0.022, 0.04) * (0.6 + 0.4 * width) * clump[i] * (1.0 - 0.6 * fr)
            prot = rng.uniform(0.03, 0.06) * width * clump[i] * (0.6 if row == 4 else 1.0) * (1.0 - 0.65 * fr)
            sdf.ellipsoid(p + n * (prot - r * 0.8), (r * 1.25, r * 1.25, r * 0.8),
                          rot=C.rot_from_axes(z_axis=n), k=0.012)
            lumps += 1
    sdf.noise(N3, 0.005, 28.0, octaves=2, offset=(2.2, 0.7, 5.1))
    sdf.noise(N3, 0.003, 60.0, octaves=1, offset=(7.1, 3.3, 1.9))
    obj = sdf.to_object("moss_stone_edge", coll, mat, target_tris=26000, smooth_iter=1)
    return obj, sdf, lumps


# ------------------------------------------------------------------------------------------
# COLOR_0: R density, G length, B AO/cavity, A 1
# ------------------------------------------------------------------------------------------

def rim_coords(co, frame):
    """Height above the face plane (m) and offset from the seed outline (m, + outside) per point."""
    q, h = frame.local(co)
    return h, SL.seed_sdf2(q) * frame.s


def lip_stats(co, B, frame):
    """COLOR_0.B along the recess rim, face side: vertices on the face plane (|h| < 1.5 mm) within 3 mm
    outside the rim edge (the 45 deg chamfer meets the face EMBLEM_BEVEL outside the outline); and a wider
    band (to 12 mm) and the chamfer for context."""
    h, off = rim_coords(co, frame)

    def st(m):
        b = B[m]
        if not len(b):
            return dict(n=0)
        return dict(n=int(len(b)), min=round(float(b.min()), 3), p10=round(float(np.percentile(b, 10)), 3),
                    p50=round(float(np.percentile(b, 50)), 3), below_0_3=int((b < 0.3).sum()),
                    below_0_45=int((b < 0.45).sum()))

    face = np.abs(h) < 0.0015
    return dict(lip_3mm=st(face & (off > EMBLEM_BEVEL - 0.0005) & (off < EMBLEM_BEVEL + 0.003)),
                lip_12mm=st(face & (off > EMBLEM_BEVEL - 0.0005) & (off < 0.012)),
                chamfer=st((h < -0.0005) & (h > -EMBLEM_BEVEL - 0.0005) & (off > -0.0005) & (off < EMBLEM_BEVEL)),
                walls=st((h <= -EMBLEM_BEVEL - 0.0005) & (h > -EMBLEM_DEPTH - 0.01) & (np.abs(off) < 0.003)))


def floor_edge_stats(co, B, frame):
    """COLOR_0.B of the recess floor (emblem_stone) within 6 mm of the walls."""
    _h, off = rim_coords(co, frame)
    m = off > -0.006
    b = B[m]
    if not len(b):
        return dict(n=0)
    return dict(n=int(len(b)), min=round(float(b.min()), 3), p10=round(float(np.percentile(b, 10)), 3),
                p50=round(float(np.percentile(b, 50)), 3), below_0_3=int((b < 0.3).sum()))


def color_rock(obj, bvh, N3, frame, cracks, use_ao=True, bvh_open=None, rsdf=None):
    """COLOR_0 of the rock: R moss in the cracks near the upper edge, G 0.3, B = AO x convexity. Around the
    recess rim the face-side AO comes from the uncut rock (bvh_open) with the field normal: rays into the cut
    and the normals averaged over the sharp rim edge read as black specks along the lip; B is floored at
    LIP_B_MIN within 3 mm of the rim edge and CHAMFER_B_MIN on the chamfer."""
    me = obj.data
    co, nrm, ed, _ = C.mesh_arrays(me)
    ao = C.ray_ao(co, nrm, bvh, n_rays=14, max_dist=0.35) if use_ao else np.ones(len(co))
    h, off = rim_coords(co, frame)
    face_lip = (h > -0.0005) & (off > -0.0005) & (off < LIP_ZONE)
    if use_ao and bvh_open is not None and rsdf is not None and face_lip.any():
        ao[face_lip] = C.ray_ao(co[face_lip], SL.sdf_normals(rsdf, co[face_lip]), bvh_open, n_rays=14,
                                max_dist=0.35)
    cx = C.surface_convexity(co, nrm, ed, iterations=5)
    # a little moss in the cracks close to the upper edge (R); nothing on the open face or the glyph
    d2 = C.poly_sdf(to2(co), BASIS["outline2"])
    near_top = 1.0 - C.smoothstep(0.03, 0.25, -d2)
    crack = np.zeros(len(co))
    for path in cracks:
        for p in path[::2]:
            dd = np.linalg.norm(co - p, axis=1)
            crack = np.maximum(crack, 1.0 - C.smoothstep(0.006, 0.02, dd))
    R = 0.55 * crack * near_top + 0.25 * near_top * C.smoothstep(0.3, 0.8, N3.fbm(co * 6.0, 2))
    R = np.clip(R, 0, 1)
    R[emblem_dist_m(co, frame) < 0.02] = 0.0
    G = np.full(len(co), 0.3)
    B = ao * (0.8 + 0.2 * np.clip(0.5 + cx / 0.006, 0, 1))
    lip3 = (np.abs(h) < 0.0015) & (off > EMBLEM_BEVEL - 0.0005) & (off < EMBLEM_BEVEL + 0.003)
    B[lip3] = np.maximum(B[lip3], LIP_B_MIN)
    cham = (h < -0.0005) & (h > -EMBLEM_BEVEL - 0.0005) & (off > -0.0005) & (off < EMBLEM_BEVEL)
    B[cham] = np.maximum(B[cham], CHAMFER_B_MIN)
    C.write_color0(me, R, G, B)
    C.box_uvs(me)


def color_emblem(obj, bvh, frame, use_ao=True):
    me = obj.data
    co, nrm, _, _ = C.mesh_arrays(me)
    ao = C.ray_ao(co, nrm, bvh, n_rays=16, max_dist=0.06) if use_ao else np.ones(len(co))
    q, _h = frame.local(co)
    edge = C.smoothstep(0.0, 0.012, -SL.seed_sdf2(q) * frame.s)          # contact shadow at the walls
    B = np.maximum(ao * (0.72 + 0.28 * edge), FLOOR_B_MIN)
    C.write_color0(me, np.zeros(len(co)), np.zeros(len(co)), B)
    # planar UVs in the emblem frame, 1 unit = 0.5 m
    uv = q * frame.s / C.UV_M_PER_UNIT
    lv = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", lv)
    C.set_uvs(me, uv[lv])


def color_moss(obj, sdf, bvh, N3, use_ao=True):
    me = obj.data
    co, nrm, ed, _ = C.mesh_arrays(me)
    ao = C.ray_ao(co, nrm, bvh, n_rays=10, max_dist=0.12) if use_ao else np.ones(len(co))
    cx = sdf.convexity(co, 0.02)
    crest = C.smoothstep(0.0, 0.006, cx)
    u_scr, _v, _z = C.project_to_screen(CAM["cam_stone_main"]["loc"], cam_q(), co)
    fringe = C.smoothstep(FRINGE_U[0], FRINGE_U[1], u_scr)        # sparser, longer strands on the fringe
    R = np.clip((0.9 + 0.1 * crest) * (1.0 - 0.3 * fringe), 0, 1)
    G = np.clip(0.62 + 0.28 * crest + 0.08 * N3.fbm(co * 9.0 + 4.0, 2) + 0.1 * fringe, 0.5, 0.92)
    B = ao * (0.85 + 0.15 * np.clip(0.5 + cx / 0.01, 0, 1))
    C.write_color0(me, R, G, B)
    C.box_uvs(me)


# ------------------------------------------------------------------------------------------
# build
# ------------------------------------------------------------------------------------------

def build_cameras(coll):
    for name, c in CAM.items():
        extras = dict(focus_distance_m=round(FOCUS, 3), frame_height_at_focus_m=round(FRAME_H, 3),
                      video_time_s=c["t"], composition_shift_fh=c["shift"])
        if name == "cam_stone_out":
            extras["exit_track"] = json.dumps(EXIT_TRACK)
        C.make_camera(name, coll, c["loc"], cam_q(), focus=FOCUS, extras=extras)
    P0 = BASIS["P0"]
    C.make_key_empty("key_stone", coll, KEY_LOC, KEY_DIR,
                     extras=dict(light_target_gltf=list(C.blender_to_gltf_dir(P0)), falloff_radius_m=1.1))


def build(use_ao=True):
    t0 = time.time()
    scene = C.ensure_scene(SCENE_NAME)
    coll = C.ensure_collection(scene, COLL_NAME)
    C.purge_collection(coll)
    setup_plane()
    rot, height = solve_emblem()
    frame = emblem_frame(rot)
    C.log(f"front plane through {np.round(BASIS['P0'], 3)}, emblem {height:.3f} m, rot {rot:.2f} deg in-plane"
          f" -> lean {image_lean(frame):.2f} deg, length {image_length(frame):.3f} frame heights")
    mat_rock = C.preview_material("mat_rock", C.PALETTE["stone"], 0.9)
    mat_moss = C.preview_material("mat_moss", C.PALETTE["moss"], 0.95)
    mat_emblem = C.preview_material("mat_emblem", "#B89A68", 0.8)
    N3 = C.Noise3(SEED + 700)
    build_cameras(coll)

    t = time.time()
    rock, rsdf, cracks = build_rock(coll, mat_rock, N3, frame)
    C.log(f"rock field + mesh: {C.tri_count(rock)} tris, {len(cracks)} cracks ({time.time() - t:.1f}s)")
    bvh_open = C.build_bvh([rock]) if use_ao else None          # the uncut rock (AO along the recess rim)
    t = time.time()
    floor, cut = cut_emblem(rock, coll, frame, mat_rock, mat_emblem)
    C.log(f"emblem recess: {cut} ({time.time() - t:.1f}s)")
    C.log(f"rim normals: lip corners before {lip_corner_normals(rock, frame)}, {fix_rim_normals(rock, frame)}, "
          f"after {lip_corner_normals(rock, frame)}")
    t = time.time()
    moss, msdf, lumps = build_moss(coll, mat_moss, N3, rsdf)
    C.log(f"moss_stone_edge: {C.tri_count(moss)} tris, {lumps} lumps ({time.time() - t:.1f}s)")

    t = time.time()
    bvh = C.build_bvh([rock, floor, moss]) if use_ao else None
    color_rock(rock, bvh, N3, frame, cracks, use_ao, bvh_open=bvh_open, rsdf=rsdf)
    color_emblem(floor, bvh, frame, use_ao)
    color_moss(moss, msdf, bvh, N3, use_ao)
    C.log(f"colours ({time.time() - t:.1f}s)")
    total = sum(C.tri_count(o) for o in coll.objects if o.type == "MESH")
    C.log(f"stone built: {total} tris, {time.time() - t0:.1f}s")
    return scene, coll, dict(cut=cut, frame=frame, rot=rot, rsdf=rsdf, msdf=msdf)


# ------------------------------------------------------------------------------------------
# validation extras
# ------------------------------------------------------------------------------------------

def glyph_box(coll, cam):
    co, _, _, _ = C.mesh_arrays(coll.objects["emblem_stone"].data)
    u, v, _ = C.project_to_screen(CAM[cam]["loc"], cam_q(), co)
    return [round(float(x), 4) for x in (u.min(), v.min(), u.max(), v.max())]


def validate_stone(rep, coll, info):
    errs, out = [], {}
    names = {n["name"] for n in rep["nodes"]}
    for req in ("rock_stone_slab", "moss_stone_edge", "emblem_stone", "cam_stone_in", "cam_stone_main",
                "cam_stone_out", "key_stone"):
        if req not in names:
            errs.append(f"missing node {req}")
    rock, floor, moss = (coll.objects.get(n) for n in ("rock_stone_slab", "emblem_stone", "moss_stone_edge"))
    frame = info["frame"]
    bvh_rock = C.build_bvh([rock])
    bvh_floor = C.build_bvh([floor])
    depths = {}
    for piece in SL.PIECES:
        c = SL.piece_centre(piece)
        ol = SL.seed_outline(piece, 0.0, step=2.0)
        probes = [c] + [c + 0.6 * (ol[j] - c) for j in np.linspace(0, len(ol) - 1, 6, endpoint=False).astype(int)]
        for k, qq in enumerate(probes):
            off = tuple(int(round(x)) for x in qq)
            p = frame.world(np.asarray(qq, float)[None], np.array([0.05]))[0]
            hr = bvh_rock.ray_cast(Vector(p), Vector(-frame.n), 0.2)
            hf = bvh_floor.ray_cast(Vector(p), Vector(-frame.n), 0.2)
            if hf[0] is None:
                errs.append(f"recess floor missing under {piece} {off}")
                continue
            if hr[0] is not None and hr[3] < hf[3]:
                errs.append(f"rock covers the recess floor at {piece} {off}")
            depths[f"{piece}{k}"] = round(float(hf[3] - 0.05), 4)
    # the face around the glyph is planar (clean recess rim)
    co, _, _, _ = C.mesh_arrays(rock.data)
    q, h = frame.local(co)
    dist = SL.seed_sdf2(q) * frame.s
    ring = (dist > 0.012) & (dist < EMBLEM_FLAT - 0.005) & (h > -0.004)
    flat_dev = float(np.abs(h[ring]).max()) if ring.any() else -1.0
    if flat_dev > 0.003:
        errs.append(f"face around the glyph not planar ({flat_dev:.4f} m)")
    # the moss sits on the rock
    mco, _, _, _ = C.mesh_arrays(moss.data)
    dm = info["rsdf"].sample(mco)
    touching = float((dm < 0.008).mean())
    if touching < 0.05:
        errs.append(f"moss strip floats ({touching:.3f} of its vertices near the rock)")
    if depths and min(depths.values()) < EMBLEM_DEPTH - 0.007:
        errs.append(f"recess shallower than {EMBLEM_DEPTH - 0.007:.3f} m ({min(depths.values()):.4f})")
    out.update(recess_depths_m=depths, recess_depth_range_m=[min(depths.values()), max(depths.values())] if depths else None,
               emblem_depth_m=EMBLEM_DEPTH, emblem_bevel_m=EMBLEM_BEVEL,
               face_planarity_m=round(flat_dev, 5), moss_touching_share=round(touching, 3),
               emblem_rot_deg=round(info["rot"], 3), emblem_height_m=EMBLEM_H,
               glyph_box_main=glyph_box(coll, "cam_stone_main"), glyph_box_in=glyph_box(coll, "cam_stone_in"),
               glyph_box_out=glyph_box(coll, "cam_stone_out"))
    rco, _, _, _ = C.mesh_arrays(rock.data)
    out["recess_rim_B"] = lip_stats(rco, C.read_color0(rock.data)[:, 2], frame)
    fco, _, _, _ = C.mesh_arrays(floor.data)
    out["recess_floor_edge_B"] = floor_edge_stats(fco, C.read_color0(floor.data)[:, 2], frame)
    out["top_edge_main_px"] = top_edge_px(coll)
    out["lip_corner_normals"] = lc = lip_corner_normals(rock, frame)
    if lc["over_20deg"]:
        errs.append(f"{lc['over_20deg']} face corners on the rim edge carry a normal > 20 deg off the face")
    return dict(errors=errs, **out)


def top_edge_px(coll, cols=(1200, 1240, 1280, 1320, 1360, 1400, 1420, 1439)):
    """Top edge of rock + moss per pixel column in cam_stone_main (px from the top, rendered mask), next to
    ROCK_OUTLINE_UV."""
    scene = bpy.context.scene
    objs = [o for o in (coll.objects.get("rock_stone_slab"), coll.objects.get("moss_stone_edge")) if o]
    p = SL.render_mask(scene, coll.objects["cam_stone_main"], os.path.join(TMP, "m_top_edge.png"), objs, 100)
    m = SL.load_rgba(p)[..., 0] > 0.5
    upper = [q for q in ROCK_OUTLINE_UV if q[1] < 0.47 and q[0] >= 0.752]
    out = {}
    for x in cols:
        ys = np.nonzero(m[:, x])[0]
        out[str(x)] = dict(ours=int(ys[0]) if len(ys) else None,
                           outline=round(float(np.interp((x + 0.5) / C.FRAME_W, [q[0] for q in upper],
                                                         [q[1] for q in upper]) * C.FRAME_H), 1))
    return out


# ------------------------------------------------------------------------------------------
# previews and comparison sheets
# ------------------------------------------------------------------------------------------

def preview_mats():
    return {
        "rock_": SL.preview_ao_material("prev_stone_rock", "#A1978A", 0.9),
        "moss_": SL.preview_ao_material("prev_stone_moss", "#5E7A26", 0.95),
        "emblem_": SL.preview_ao_material("prev_stone_emblem", "#C4A46E", 0.8),
    }


def lit(scene, percent):
    SL.setup_lit_preview(scene, KEY_DIR, BG_HEX, percent=percent, sun_strength=4.6, sun_color=(1.0, 0.9, 0.76),
                         fill_dir=(-0.7, 0.55, -0.3), fill_strength=0.1, world_strength=1.0)


def render_previews(scene, coll, cams=None, percent=50):
    lit(scene, percent)
    out = []
    try:
        with SL.PreviewMaterials(coll, preview_mats()):
            for name in (cams or list(CAM.keys())):
                path = os.path.join(CAPTURES, f"stone_{name}.png")
                C.render_camera(scene, coll.objects[name], path)
                out.append(path)
    finally:
        SL.remove_preview_lights(scene)
    return out


def reference_masks(rgb, gain=1.0):
    a = np.clip(rgb[..., :3] * gain, 0, 1) * 255.0
    H, W = a.shape[:2]
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    lum = a @ np.array([0.2126, 0.7152, 0.0722])
    sat = a.max(-1) - a.min(-1)
    green = (g > r * 1.15) & (g > b * 1.3) & (g > 50)
    rock = (lum > 24) & ~green & (sat < 75)
    gold = (r > 100) & (g > 75) & (b < 0.72 * r) & (r - g < 60) & (r > g)
    for m in (rock, green, gold):
        m[:, :int(0.55 * W)] = False
        m[:int(0.28 * H)] = False
    acc = np.zeros(rock.shape, np.float32)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            acc += np.roll(np.roll(rock, dy, 0), dx, 1)
    return acc > 12.5, green, gold


def _box(m, w, h):
    ys, xs = np.nonzero(m)
    if not len(xs):
        return None
    return [round(np.percentile(xs, 1) / w, 4), round(np.percentile(ys, 1) / h, 4),
            round(np.percentile(xs, 99) / w, 4), round(np.percentile(ys, 99) / h, 4)]


def compare_sheets(scene, coll, percent=50):
    os.makedirs(TMP, exist_ok=True)
    out, stats = [], {}
    rows = (("main", "cam_stone_main", "frames/14_rock_emblem.png", 1.8),
            ("in", "cam_stone_in", "frames/13_branch_exit.png", 1.8))
    rock = coll.objects["rock_stone_slab"]
    floor = coll.objects["emblem_stone"]
    moss = coll.objects["moss_stone_edge"]
    for key, cam, ref_rel, gain in rows:
        ref = SL.load_rgba(os.path.join(REF, ref_rel))
        prev_p = os.path.join(CAPTURES, f"stone_{cam}.png")
        prev = SL.load_rgba(prev_p) if os.path.exists(prev_p) else np.zeros_like(ref)
        h, w = prev.shape[:2]
        refs = SL.resize_to(ref, h, w)
        cam_o = coll.objects[cam]
        m_rock = SL.load_rgba(SL.render_mask(scene, cam_o, os.path.join(TMP, f"m_rock_{key}.png"),
                                             [rock, floor], percent, occluders=[moss]))[..., 0] > 0.5
        m_moss = SL.load_rgba(SL.render_mask(scene, cam_o, os.path.join(TMP, f"m_moss_{key}.png"),
                                             [moss], percent, occluders=[rock, floor]))[..., 0] > 0.5
        m_gold = SL.load_rgba(SL.render_mask(scene, cam_o, os.path.join(TMP, f"m_gold_{key}.png"),
                                             [floor], percent, occluders=[rock, moss]))[..., 0] > 0.5
        r_rock, r_moss, r_gold = reference_masks(refs, gain)
        ov = np.clip(refs[..., :3] * gain, 0, 1) * 0.55
        ov[SL.outline(r_rock)] = (0.2, 0.95, 1.0)
        ov[SL.outline(r_moss)] = (0.3, 1.0, 0.3)
        ov[SL.outline(m_rock)] = (1.0, 0.92, 0.1)
        ov[SL.outline(m_moss)] = (1.0, 0.45, 0.1)
        ov[SL.outline(m_gold)] = (1.0, 0.3, 0.9)
        SL.draw_grid(ov)
        SL.label(ov, f"{key} ref rock cyan moss green / ours rock yellow moss orange seed magenta", 6, 6)
        path = os.path.join(CAPTURES, f"stone_cmp_{key}.png")
        SL.save_rgb(np.concatenate([np.clip(refs[..., :3] * gain, 0, 1), prev[..., :3], ov], 1), path)
        out.append(path)
        inter = (r_rock & m_rock).sum()
        union = (r_rock | m_rock).sum()
        stats[key] = dict(rock_iou=round(float(inter) / max(float(union), 1.0), 4),
                          gold_box_ref=_box(r_gold, w, h), gold_box_ours=_box(m_gold, w, h),
                          moss_box_ref=_box(r_moss, w, h), moss_box_ours=_box(m_moss, w, h))
    # full-size crop around the glyph next to frame 14
    pfull = os.path.join(TMP, "prev_main_full.png")
    lit(scene, 100)
    try:
        with SL.PreviewMaterials(coll, preview_mats()):
            C.render_camera(scene, coll.objects["cam_stone_main"], pfull)
    finally:
        SL.remove_preview_lights(scene)
    full = SL.load_rgba(pfull)
    r14 = SL.load_rgba(os.path.join(REF, "frames/14_rock_emblem.png"))
    x0, y0, x1, y1 = 840, 380, 1440, 1020
    path = os.path.join(CAPTURES, "stone_cmp_detail.png")
    SL.save_rgb(np.concatenate([np.clip(r14[y0:y1, x0:x1, :3] * 1.3, 0, 1), full[y0:y1, x0:x1, :3]], 1), path)
    out.append(path)
    # the recess at 2x: frame 14 (brightened) / ours (walls, chamfer, floor)
    x0, y0, x1, y1 = 960, 580, 1240, 900
    H2, W2 = 2 * (y1 - y0), 2 * (x1 - x0)
    img = np.concatenate([SL.resize_to(np.clip(r14[y0:y1, x0:x1, :3] * 1.3, 0, 1), H2, W2),
                          SL.resize_to(full[y0:y1, x0:x1, :3], H2, W2)], 1)
    SL.label(img, f"frame 14 x1.3 / ours: recess {EMBLEM_DEPTH * 100:.0f} cm, chamfer {EMBLEM_BEVEL * 1000:.0f} mm", 6, 6)
    path = os.path.join(CAPTURES, "stone_cmp_recess.png")
    SL.save_rgb(img, path)
    out.append(path)
    out.append(seed_sheet(coll))
    C.log("compare:", json.dumps(stats))
    return out, stats


def seed_sheet(coll):
    """stone_cmp_seed: the recess walls of rock_stone_slab (vertices between face and floor, in seed units) and
    the floor vertices, over the outlines parsed from seed.svg (cyan) and our construction (yellow)."""
    frame = emblem_frame(solve_emblem_rot())
    co, _, _, _ = C.mesh_arrays(coll.objects["rock_stone_slab"].data)
    q, hgt = frame.local(co)
    sel = (np.abs(q[:, 0]) < 34) & (np.abs(q[:, 1]) < 56) & (hgt < -0.004) & (hgt > -EMBLEM_DEPTH + 0.003)
    fco, _, _, _ = C.mesh_arrays(coll.objects["emblem_stone"].data)
    fq, _fh = frame.local(fco)
    S_ = 6
    W, H = 70 * S_, 120 * S_
    img = np.full((H, W, 3), 0.08)

    def to_uv(P):
        return np.stack([(P[:, 0] + 35) / 70.0, (60 - P[:, 1]) / 120.0], 1)

    for a, b in to_uv(fq):
        SL.draw_dot(img, a, b, 0.5, (0.55, 0.38, 0.15))
    for P in SL.svg_seed_paths(SVG):
        SL.draw_polyline(img, to_uv(P), (0.2, 0.95, 1.0), closed=True, width=1.5)
    for piece in SL.PIECES:
        SL.draw_polyline(img, to_uv(SL.seed_outline(piece, 0.0, step=0.2)), (1.0, 0.9, 0.1), closed=True,
                         width=0.6)
    for a, b in to_uv(q[sel]):
        SL.draw_dot(img, a, b, 0.8, (1.0, 0.3, 0.3))
    SL.label(img, "svg cyan  ours yellow  walls red  floor brown", 6, 6, scale=2)
    path = os.path.join(CAPTURES, "stone_cmp_seed.png")
    SL.save_rgb(img, path)
    return path


def save_blend():
    bpy.ops.wm.save_as_mainfile(filepath=BLEND_PATH, check_existing=False, compress=True)
    C.log("saved", BLEND_PATH)


def parse_args(argv):
    a = dict(export=False, validate=False, render=None, percent=50, save=False, ao=True, compare=False, glb=None)
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
        elif x == "--glb":
            i += 1
            a["glb"] = argv[i]
        i += 1
    return a


def main(argv=None):
    a = parse_args(argv or [])
    if bpy.app.background and a["save"] and os.path.exists(BLEND_PATH):
        bpy.ops.wm.open_mainfile(filepath=BLEND_PATH)   # keep the other scene sets
    scene, coll, info = build(use_ao=a["ao"])
    result = dict(tris=sum(C.tri_count(o) for o in coll.objects if o.type == "MESH"),
                  cameras={k: dict(loc=[round(x, 3) for x in v["loc"]], t=v["t"]) for k, v in CAM.items()})
    glb = os.path.abspath(a["glb"]) if a["glb"] else GLB_PATH
    if a["export"]:
        C.export_glb(scene, glb)
        result["glb"] = glb
        result["glb_bytes"] = os.path.getsize(glb)
    if a["validate"]:
        rep = C.validate_glb(glb)
        extra = validate_stone(rep, coll, info)
        rep["stone"] = extra
        result["validate"] = dict(errors=rep["errors"] + extra["errors"], warnings=rep["warnings"],
                                  totals=rep["totals"], stone=extra)
        os.makedirs(CAPTURES, exist_ok=True)
        with open(os.path.join(CAPTURES, "stone_validate.json"), "w") as f:
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
