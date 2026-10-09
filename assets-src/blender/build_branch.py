"""
build_branch.py - the 'branch' scene set (S06, video 31-37 s): the J-shaped mossy log with its
splintered broken end, the lower-right fragment, the fuzzy-ball base mesh and the anchors of its
loop, exported to public/nature/models/branch.glb (contract: CLAUDE.md "Asset contract").

Run headless (from the repo root):
    "/c/Program Files/Blender Foundation/Blender 5.1/blender.exe" --background --factory-startup \
        --python assets-src/blender/build_branch.py -- --export --validate [--render] [--compare] [--save]

Run inside a live Blender session (e.g. through MCP):
    import sys, importlib; sys.path.insert(0, r"<repo>/assets-src/blender")
    import build_branch; importlib.reload(build_branch); build_branch.main(["--export", "--validate"])

Options:
    --export            write public/nature/models/branch.glb
    --out PATH          write / validate the GLB at PATH instead (scratch iterations)
    --captures DIR      previews, sheets and the validate report go to DIR (default docs/captures/blender)
    --validate          re-read / re-import the GLB, check it against the contract and the ball loop
                        (report: docs/captures/blender/branch_validate.json)
    --render [cams]     vertex-lit Workbench previews -> docs/captures/blender/branch_<cam>.png (all
                        cams if none): key light along key_branch + sky fill, baked AO, cast shadows
    --compare           comparison sheets -> docs/captures/blender/branch_cmp_<what>.png
    --percent N         preview size in % of 1440x1020 (default 50)
    --save              save nature.blend (headless: opens the existing file first so the other scene
                        sets are kept; only the 'branch' scene / collection is rebuilt)
    --no-ao             skip the ray-traced AO (B channel = cavity only; fast iteration)

Everything is deterministic (seed 134). World units: metres, Blender Z-up, cameras look along +Y.
Geometry helpers of this set (twisted bark relief, cracks, moss coats, lit previews): branchlib.py.

The log (revision 2): a round trunk with a variable grain twist whose moss coat wraps most of the
circumference and leaves an irregular bare-bark channel facing the lens (measured per control point
on frame 10: leaning to the inside of the hook at the top, centred in the descent, ~100 deg wide in
the bend, closing towards the broken end; its edge wanders with bays, tongues, moss islands and
holes), split at the top by a deep dark cleft. The bark is a few big rope-like strands with V creases
that wander, merge and split, plus deep grain-following cracks (some branching); the fine fibres are
left to the bark texture. The moss is a lumpy carpet (Voronoi clumps 3-8 cm, billows, broad mounds,
sagging underside) that thins out to a ragged edge and creeps into the cracks; a twig stub under the
arm stays bare. The broken end is a moss mound over the fracture with two stumps rising out of it and
torn fibres between them.

Revision 3 (geometry iteration 3):
  * the stumps and all loose / torn fibres are their own mesh `wood_branch_stumps` (bark material,
    same COLOR_0 semantics; their moss coats, base cushions and the end mound stay in moss_branch_j):
    a tall column (irregular oval twisted section, furrowed bark with cross-fissures and cracks,
    splintered top with spikes of different heights, a moss sheath on each side) and a split-off
    front wall ("chunk": kidney section, leaning out to the right, broken across its axis with a
    pale sliver), standing fibres at both rims. COLOR_0 on wood_branch_stumps: fresh wood (the breaks,
    the splinter lips, the fibres' torn ends) has R = 0, B 0.9-1.0 and G = 0 - G is >= 0.2 everywhere
    else, so G == 0 marks the pale fresh wood for the engine; R >= 0.6 in the moss spots on the upper
    faces and where the cushions climb the bark; B ~0.3 in the cracks and under the moss.
  * the J's end ring carries a moss annulus over the broken rim (dark open centre between the stumps,
    broken rim plates poking through); the J's coat only leaves the chunk's footprint open.
  * the bark channel: bare across the whole visible width at the apex of the bend (frame 10: only a
    thin moss fringe on the inner outline), a ragged inner border below it, and it now runs on along
    the lower arm to the broken end (dark bark under the chunk); square-ish tongues / bays 2-6 cm on
    its border; COLOR_0.R 0.4-0.8 on the bark within ~3 cm of the moss border and in the deep cracks
    (the engine's wood moss frays the border); short fissures across the strands every 5-13 cm.

Measured from the reference (frames 10-13 and clip motion/09, tracked frame by frame at 30 fps):
  * 31.2-33.6 s the whole branch picture slides up as one rigid layer (no parallax, no zoom) with a
    decelerating ease: at 31.5 s (frame 10) it sits 0.0549 frame heights lower than at rest, from
    ~33.6 s to 37.0 s it is static (frames 11 and 12 differ only by the ball). cam_branch_main ->
    cam_branch_p1 is therefore a pure 1.98 deg tilt; CAM_DRIFT is the measured progress curve (also
    written to the camera extras).
  * one fuzzy ball loops with a period of 2.367 s (71 video frames). Seen from the rest pose it rises
    on the left, crosses the top in front of the trunk, falls down the right side, leaves the frame
    at the bottom right, re-enters at the top centre ~0.15 s later, drops in front of the trunk and
    runs left in front of the arm back to the start. It is never hidden by the wood; its measured
    size (0.10-0.13 of the frame width) puts it 1.3-1.6 m from the lens, ~0.8 m in front of the J.
    BALL_TRACK / BALL_SIZE are those measurements; the anchors are fitted to them at build time.
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

importlib.reload(C)
import branchlib as BL  # noqa: E402

importlib.reload(BL)

ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
GLB_PATH = os.path.join(ROOT, "public", "nature", "models", "branch.glb")
BLEND_PATH = os.path.join(HERE, "nature.blend")
CAPTURES = os.path.join(ROOT, "docs", "captures", "blender")
REF = os.path.join(ROOT, "docs", "nature-webgl-reference")      # comparison sheets only (never shipped)
TMP = os.path.join(tempfile.gettempdir(), "silva_branch_build")
SCENE_NAME = "branch"
COLL_NAME = "branch"
SEED = 134

TV = math.tan(math.radians(C.FOV_V_DEG) / 2.0)
TH = TV * C.ASPECT

# ------------------------------------------------------------------------------------------
# cameras (scroll order main -> p1 -> out); all three share one position, only the tilt changes.
# focus = distance (m) along the view axis to the focus plane (the J): frame height 1.5 m there.
# ------------------------------------------------------------------------------------------
PITCH_MAIN = -6.0
DRIFT_FH = 0.0549                                                  # main -> rest, frame heights
DRIFT_DEG = math.degrees(math.atan(2.0 * DRIFT_FH * TV))           # 1.983 deg
OUT_DEG = 0.6                                                      # small extra tilt for the exit pose
CAM_LOC = (0.0, -2.35, 0.25)
CAMS = {
    "cam_branch_main": dict(loc=CAM_LOC, pitch=PITCH_MAIN, yaw=0.0, roll=0.0, focus=2.38, t=31.5),
    "cam_branch_p1": dict(loc=CAM_LOC, pitch=PITCH_MAIN - DRIFT_DEG, yaw=0.0, roll=0.0, focus=2.38, t=34.5,
                          settle=33.6, hold=37.0),
    "cam_branch_out": dict(loc=CAM_LOC, pitch=PITCH_MAIN - DRIFT_DEG - OUT_DEG, yaw=0.0, roll=0.0, focus=2.38,
                           t=38.5),
}
# measured progress main -> p1 (video s, 0..1) and the slide before 31.5 s (frame heights below rest)
CAM_DRIFT = [(31.5, 0.0), (31.6, 0.25), (31.7, 0.393), (31.8, 0.5), (31.9, 0.536), (32.0, 0.607),
             (32.2, 0.714), (32.4, 0.786), (32.6, 0.857), (32.8, 0.893), (33.0, 0.929), (33.2, 0.964),
             (33.6, 1.0)]
DRIFT_BEFORE_MAIN = [(31.2, 0.133), (31.267, 0.110), (31.333, 0.090), (31.4, 0.075), (31.467, 0.061),
                     (31.5, 0.055)]

# key light: from above, a little from the right, towards the scene = glTF (-0.12, -0.70, -0.70)
# normalised (the direction the light travels; frames 10-12: lit upper right of the log and of the
# ball, dark underside). Blender world = (x_gltf, -z_gltf, y_gltf).
KEYS = {"key_branch": dict(dir=(-0.12, 0.70, -0.70))}
KEY_DIR_BL = C.normalize(np.array(KEYS["key_branch"]["dir"], float))

# ------------------------------------------------------------------------------------------
# the J: centreline in frame-10 screen space of cam_branch_main (u, v, view depth m) + base radii
# (m, before cross-section lobes / bark / moss). It hangs from above the frame (slightly right of
# centre), descends with a slight S, bends down and to the left and curls up into the broken end.
# Control points 3-10 were fitted (edge distances along the centreline normals) so the base mesh
# silhouette sits ~10 px (at 1440) inside the frame-10 silhouette: room for the vegetation the
# engine grows on top.
# ------------------------------------------------------------------------------------------
J_UVD = [(0.506, -0.50, 2.80), (0.524, -0.16, 2.64), (0.541, 0.10, 2.53), (0.549, 0.280, 2.47),
         (0.5375, 0.451, 2.42), (0.514, 0.591, 2.37), (0.4525, 0.709, 2.32), (0.384, 0.803, 2.27),
         (0.290, 0.827, 2.24), (0.200, 0.838, 2.23), (0.136, 0.8235, 2.22), (0.095, 0.785, 2.20),
         (0.087, 0.722, 2.17)]
J_R = [0.245, 0.234, 0.221, 0.209, 0.195, 0.164, 0.1425, 0.1555, 0.131, 0.109, 0.098, 0.104, 0.104]
J_HOOK_UV = (0.30, 0.60)          # screen point inside the J: defines "inner side" of bend and arm
# grain twist rate (rad/m) per control point: twisted strands at the top (~17 deg spiral), the
# cracks run almost along the log through the bend (frame 10 / details 06)
J_TWIST = [1.2, 1.3, 1.4, 1.4, 1.15, 0.75, 0.5, 0.4, 0.4, 0.45, 0.6, 0.8, 0.8]
# bare-bark channel per control point, measured on frame 10 (bark vs moss pixels across the J):
# centre leans J_CH_OFF (rad) from the view direction towards the inside of the hook, half-width
# J_CH_W (rad); both wobble along the log, the edge is ragged (grain-aligned noise)
# (revision 3: frame 10 shows bare bark across the whole width at the apex of the bend (CP 6, only a
# thin pale fringe of the moss behind on the inner outline) and a ragged inner border at CP 7)
J_CH_OFF = [0.36, 0.36, 0.36, 0.34, 0.10, 0.08, -0.02, 0.30, 0.45, 0.55, 0.60, 0.60, 0.60]
J_CH_W = [0.50, 0.50, 0.52, 0.52, 0.50, 0.64, 1.62, 0.62, 0.50, 0.36, 0.36, 0.30, 0.22]
# the dark cleft at the top of the channel (frame 10 x 700-725, y 160-400): view-anchored angle
J_CLEFT = dict(angle=0.43, v=(-0.22, 0.42), depth=0.05, width=0.024)

# the broken end (revision 3; frame 10 x 20-215, y 550-850; details/06). Two stumps in their own
# mesh `wood_branch_stumps` (with the loose fibres); their moss is part of moss_branch_j.
#   tall  - a column at the left rim of the end ring, 0.25 m above it, leaning a little to the lens
#           and to the right; irregular, slightly oval, twisted section; furrowed bark (strands with
#           sparse cross-fissures and cracks); a jagged top with pale fresh wood and loose fibres of
#           different heights; a thick moss sheath on its right / back side (towards the chunk, lit
#           by the key), a second one up its left side, a cushion climbing its base, moss spots on
#           its upper left faces.
#   chunk - the split-off front wall of the log end: rises out of the J's front face ~0.1 m below
#           the ring, leans out to the right and splays (wider at the top, kidney section: convex
#           bark to the lens, hollow back), broken at a slant across its axis with a tall pale
#           sliver on the right; dark furrowed bark to the lens; moss up its back / right edge, a
#           cushion climbing its back, moss spots on its upper left face.
# Directions phi are angles in the end-ring plane from the lens side (0) towards camera right
# (+pi/2), the back (pi) and the left (-pi/2). Keys: ring = (a, b) x end radius along (camera
# right, camera forward) where the axis crosses the ring; root = (a, b, back): the axis starts
# `back` m before the end at (a, b) x the log radius there (inside the log); height above the
# ring (m); lean = top offset per metre of height along (right, forward); bow = mid-height offset
# (right, forward) m; radii root, ring, 1/3, 2/3, top (m); oval = (amp, long-axis angle from
# camera right towards forward (rad), turn over the length (rad)[, kidney amp: 3-lobe, hollow
# back]); lobes = small extra lobes of the sweep profile; bumps = (amp, 1/m); furrows = bark strands
# (count around, furrow depth m, meander, crown); cracks = deep cracks; jag = raggedness of the
# torn top; spikes = torn top (phi, height x top radius, width rad[, thick 0..1]);
# fibres = loose splinters at the top rim (phi, length m, outward lean, base radius m);
# coats = moss (phi, half-width rad, up to share of the length, thickness m, lump m);
# base = cushion around the base: (height above the ring m on the lens side, on the back side,
# thickness m, lump m); spots = moss spots on the bark (phi, strength).
J_STUMPS = [
    dict(name="tall", seed=21, ring=(-0.72, 0.12), root=(-0.62, 0.15, 0.09), height=0.25, lean=(0.12, -0.13),
         bow=(0.006, -0.004), radii=(0.040, 0.039, 0.035, 0.031, 0.028), oval=(0.12, 0.5, 1.1),
         lobes=[(3, 0.06, 0.4), (5, 0.035, 1.1)], lobe_twist=0.8, bumps=(0.07, 45.0), twist=1.6, segments=60,
         furrows=(7, 0.011, 0.5, 0.75), cracks=4, jag=0.25,
         spikes=[(-1.6, 0.9, 0.4), (0.5, 0.6, 0.45), (2.3, 1.5, 0.3, 0.4)],
         fibres=[(0.9, 0.04, 0.35, 0.005), (-1.0, 0.03, 0.5, 0.0045), (2.6, 0.05, 0.25, 0.005)],
         coats=[(2.0, 1.45, 0.86, 0.022, 0.014), (-1.75, 1.6, 0.97, 0.02, 0.012)],
         base=(0.012, 0.05, 0.011, 0.009), spots=(-1.1, 0.8)),
    dict(name="chunk", seed=22, ring=(0.36, -0.70), root=(0.20, -0.88, 0.11), height=0.10, lean=(0.6, 0.06),
         bow=(0.004, -0.006), radii=(0.034, 0.037, 0.040, 0.043, 0.046), oval=(0.26, 0.0, 0.3, 0.14),
         lobes=[(3, 0.07, 1.0), (4, 0.04, 0.3)], lobe_twist=0.3, bumps=(0.06, 35.0), twist=0.5, segments=76,
         furrows=(9, 0.013, 0.5, 0.6), cracks=5, jag=0.18,
         spikes=[(1.6, 0.4, 1.8, 0.8), (1.9, 1.0, 0.24, 0.3), (-2.4, 0.35, 0.4), (-0.6, 0.3, 0.5)],
         fibres=[(1.5, 0.034, 0.3, 0.0055), (2.2, 0.032, 0.3, 0.0045), (-1.2, 0.028, 0.5, 0.0045)],
         coats=[(2.4, 1.5, 0.85, 0.013, 0.010)],
         base=(0.0, 0.06, 0.011, 0.009), spots=(-0.7, 0.7)),
]
# torn fibres rising from the fracture between the stumps: (a, b) x end radius, length (m), lean
# (right, forward) per m, base radius (m)
J_FIBRES = [((-0.15, -0.05), 0.075, (0.15, -0.2), 0.008), ((0.05, 0.25), 0.06, (0.3, 0.1), 0.007),
            ((-0.35, 0.3), 0.085, (-0.1, 0.2), 0.008)]
# moss over the end face: a lumpy ring over the rim (the J's coat runs on over it into the stumps'
# bases, the broken rim plates poke through it), open in the middle between the stumps (the dark
# fracture of the log end): inner edge at rho (open, full) x the ring radius, ragged
J_END_MOSS = dict(rho=(0.40, 0.62), height=0.026, lump=0.018)

# the fragment: lower right, broken at its left (near) end, running right and down out of frame
F_UVD = [(0.636, 0.824, 2.10), (0.690, 0.866, 2.06), (0.775, 0.905, 2.08), (0.870, 0.952, 2.12),
         (0.975, 1.012, 2.18), (1.120, 1.090, 2.26)]
F_R = [0.070, 0.098, 0.122, 0.126, 0.128, 0.130]
F_TWIST = [0.9, 0.8, 0.6, 0.5, 0.5, 0.5]
# bark channel along the upper front (frame 10 x 1060-1300): centre F_CH_OFF rad from the view
# direction towards world up, half-width F_CH_W
F_CH_OFF = [1.0, 1.0, 0.95, 0.9, 0.9, 0.9]
F_CH_W = [-0.2, 0.02, 0.22, 0.24, 0.2, 0.15]

BALL_RADIUS = 0.06
BALL_FUR = 0.015          # fur the engine is assumed to grow (only used to size the loop)

# ------------------------------------------------------------------------------------------
# fuzzy ball: measured screen track in the rest pose (cam_branch_p1), phase-binned over 2.5 loops
# of clip 09 (blob detection after removing the slide): (loop phase, u, v). Phase 0 = 33.533 s.
# BALL_SIZE: measured blob width / frame width (smoothed bins). Gaps = ball not detectable
# (off frame, or over the mossy fragment / the dark top).
# ------------------------------------------------------------------------------------------
BALL_PERIOD = 2.367
BALL_T0 = 33.533
BALL_TRACK = [
    (0.000, 0.191, 0.733), (0.014, 0.181, 0.720), (0.028, 0.179, 0.685), (0.042, 0.178, 0.649),
    (0.056, 0.176, 0.612), (0.070, 0.180, 0.575), (0.085, 0.184, 0.542), (0.099, 0.192, 0.505),
    (0.113, 0.202, 0.472), (0.127, 0.212, 0.442), (0.141, 0.224, 0.416), (0.155, 0.236, 0.394),
    (0.169, 0.248, 0.373), (0.183, 0.262, 0.356), (0.197, 0.276, 0.341), (0.211, 0.291, 0.329),
    (0.225, 0.305, 0.319), (0.239, 0.319, 0.311), (0.254, 0.332, 0.305), (0.268, 0.345, 0.299),
    (0.282, 0.357, 0.293), (0.296, 0.369, 0.287), (0.310, 0.381, 0.283), (0.324, 0.394, 0.281),
    (0.338, 0.404, 0.286), (0.352, 0.416, 0.297), (0.408, 0.493, 0.332), (0.423, 0.512, 0.352),
    (0.437, 0.524, 0.363), (0.451, 0.543, 0.380), (0.465, 0.561, 0.396), (0.507, 0.609, 0.465),
    (0.521, 0.627, 0.494), (0.535, 0.645, 0.523), (0.549, 0.668, 0.559), (0.563, 0.690, 0.593),
    (0.577, 0.712, 0.627), (0.592, 0.735, 0.661), (0.606, 0.763, 0.692), (0.620, 0.801, 0.721),
    (0.634, 0.843, 0.745), (0.648, 0.872, 0.767), (0.662, 0.907, 0.796), (0.817, 0.526, 0.233),
    (0.831, 0.528, 0.282), (0.845, 0.528, 0.358), (0.859, 0.514, 0.428), (0.873, 0.508, 0.472),
    (0.887, 0.483, 0.520), (0.902, 0.430, 0.588), (0.916, 0.392, 0.626), (0.930, 0.363, 0.681),
    (0.944, 0.332, 0.707), (0.958, 0.294, 0.723), (0.972, 0.261, 0.728), (0.986, 0.223, 0.732),
    (1.000, 0.193, 0.735),
]
# extrapolated at the measured speed into the undetected stretches next to the hidden part
BALL_TRACK_EXTRA = [(0.676, 0.940, 0.820), (0.690, 0.972, 0.845), (0.789, 0.526, 0.121),
                    (0.803, 0.526, 0.177)]
BALL_SIZE = [(0.0, 0.105), (0.1, 0.117), (0.2, 0.118), (0.3, 0.113), (0.4, 0.108), (0.47, 0.115),
             (0.53, 0.128), (0.6, 0.131), (0.66, 0.122), (0.82, 0.106), (0.9, 0.11), (0.95, 0.105)]
# anchors: visible ones are fitted in (u, v) at the depth given by BALL_SIZE; the hidden arc
# (exit right -> above/behind the lens -> re-entry top centre) is fixed: 'uvd' in the rest pose or
# 'cam' = rest-pose camera space (x right, y up, z forward; m)
ANCHORS = [
    dict(phase=0.000), dict(phase=0.100), dict(phase=0.205), dict(phase=0.315), dict(phase=0.430),
    dict(phase=0.540), dict(phase=0.645),
    dict(phase=0.700, uvd=(1.10, 0.92, 1.30), hidden=True),
    dict(phase=0.748, cam=(0.25, 0.60, -0.05), hidden=True),
    dict(phase=0.795), dict(phase=0.865), dict(phase=0.935),
]

# preview albedo (linear): weathered bark, lit moss
ALBEDO = {"wood": C.hex_to_linear("#8F7B66"), "moss": C.hex_to_linear("#8A9C30")}


def cam_q(name):
    c = CAMS[name]
    return C.cam_quat(c["pitch"], c["yaw"], c["roll"])


def S(cam, u, v, d):
    """World point at frame position (u, v) and view depth d for camera `cam`."""
    c = CAMS[cam]
    return C.screen_to_world(c["loc"], cam_q(cam), u, v, d)


def cam_space_to_world(cam, p):
    right, up, fwd = C.cam_basis(cam_q(cam))
    return np.asarray(CAMS[cam]["loc"], float) + right * p[0] + up * p[1] + fwd * p[2]


def project(cam, P):
    c = CAMS[cam]
    P = np.asarray(P, float)
    u, v, z = C.project_to_screen(c["loc"], cam_q(cam), P.reshape(-1, 3))
    return u.reshape(P.shape[:-1]), v.reshape(P.shape[:-1]), z.reshape(P.shape[:-1])


def facing(sw, cam="cam_branch_main"):
    c = np.asarray(CAMS[cam]["loc"], float)
    v = C.normalize(sw.smooth - c)
    return -np.einsum("...k,...k->...", sw.nsmooth, v)


# ------------------------------------------------------------------------------------------
# ball loop: closed centripetal Catmull-Rom exactly as three.js CatmullRomCurve3(points, true,
# 'centripetal').getPoint(t); loop phase -> anchor segment is piecewise linear in loop_phase
# ------------------------------------------------------------------------------------------

def _cr_three(p0, p1, p2, p3, w):
    dt0 = float(np.sum((p0 - p1) ** 2)) ** 0.25
    dt1 = float(np.sum((p1 - p2) ** 2)) ** 0.25
    dt2 = float(np.sum((p2 - p3) ** 2)) ** 0.25
    if dt1 < 1e-4:
        dt1 = 1.0
    if dt0 < 1e-4:
        dt0 = dt1
    if dt2 < 1e-4:
        dt2 = dt1
    t1 = ((p1 - p0) / dt0 - (p2 - p0) / (dt0 + dt1) + (p2 - p1) / dt1) * dt1
    t2 = ((p2 - p1) / dt1 - (p3 - p1) / (dt1 + dt2) + (p3 - p2) / dt2) * dt1
    c2 = -3.0 * p1 + 3.0 * p2 - 2.0 * t1 - t2
    c3 = 2.0 * p1 - 2.0 * p2 + t1 + t2
    return p1 + t1 * w + c2 * w * w + c3 * w * w * w


def loop_point(points, phases, phi):
    """Ball position at loop phase phi: segment i with phase_i <= phi < phase_i+1, w = local
    fraction, then getPoint((i + w) / n) of the closed centripetal Catmull-Rom (three.js)."""
    n = len(points)
    phi = phi % 1.0
    ph = list(phases) + [1.0]
    i = max(k for k in range(n) if ph[k] <= phi + 1e-12)
    w = (phi - ph[i]) / max(ph[i + 1] - ph[i], 1e-9)
    P = [np.asarray(points[(i + k) % n], float) for k in (-1, 0, 1, 2)]
    return _cr_three(P[0], P[1], P[2], P[3], w)


def ball_phase(t):
    return ((t - BALL_T0) / BALL_PERIOD) % 1.0


def ball_depth(phase):
    """View depth (rest pose) at which the ball (radius + assumed fur) has the measured size."""
    ph = np.array([p for p, _ in BALL_SIZE] + [1.0 + BALL_SIZE[0][0]])
    sz = np.array([s for _, s in BALL_SIZE] + [BALL_SIZE[0][1]])
    size = float(np.interp(phase % 1.0, ph, sz))
    return float(np.clip(2.0 * (BALL_RADIUS + BALL_FUR) / (2.0 * TH * size), 1.2, 1.7))


def fit_anchors():
    """Anchor world positions so the loop reproduces BALL_TRACK from cam_branch_p1."""
    cam = "cam_branch_p1"
    trk = np.array(BALL_TRACK + BALL_TRACK_EXTRA)
    wts = np.array([1.0] * len(BALL_TRACK) + [0.5] * len(BALL_TRACK_EXTRA))
    order = np.argsort(trk[:, 0])
    trk, wts = trk[order], wts[order]
    phases = [a["phase"] for a in ANCHORS]
    vis = [k for k, a in enumerate(ANCHORS) if not a.get("hidden")]
    x0 = []
    for k in vis:
        ph = ANCHORS[k]["phase"]
        x0 += [np.interp(ph, trk[:, 0], trk[:, 1]), np.interp(ph, trk[:, 0], trk[:, 2])]
    x0 = np.array(x0, float)

    def points(xv):
        pts, j = [], 0
        for a in ANCHORS:
            if a.get("hidden"):
                if "uvd" in a:
                    pts.append(S(cam, *a["uvd"]))
                else:
                    pts.append(cam_space_to_world(cam, a["cam"]))
            else:
                pts.append(S(cam, xv[2 * j], xv[2 * j + 1], ball_depth(a["phase"])))
                j += 1
        return pts

    def resid(xv):
        pts = points(xv)
        P = np.array([loop_point(pts, phases, ph) for ph in trk[:, 0]])
        u, v, _ = project(cam, P)
        r = np.concatenate([(u - trk[:, 1]) * C.ASPECT * wts, (v - trk[:, 2]) * wts])
        return np.concatenate([r, 0.05 * (xv - x0)])

    x = x0.copy()
    for _ in range(15):
        r0 = resid(x)
        J = np.zeros((len(r0), len(x)))
        for i in range(len(x)):
            dx = np.zeros_like(x)
            dx[i] = 1e-4
            J[:, i] = (resid(x + dx) - r0) / 1e-4
        step = np.linalg.lstsq(J, -r0, rcond=None)[0]
        x = x + step
        if np.abs(step).max() < 1e-6:
            break
    pts = points(x)
    meas = np.array(BALL_TRACK)
    P = np.array([loop_point(pts, phases, ph) for ph in meas[:, 0]])
    u, v, _ = project(cam, P)
    err = np.hypot((u - meas[:, 1]) * C.ASPECT, v - meas[:, 2])
    return pts, dict(rms=float(np.sqrt(np.mean(err ** 2))), max=float(err.max()),
                     max_at_phase=float(meas[int(np.argmax(err)), 0]))


# ------------------------------------------------------------------------------------------
# geometry helpers
# ------------------------------------------------------------------------------------------

def ctrl_rings(sw, ctrl):
    """Ring index of every control point (nearest ring centre; non-decreasing)."""
    ctrl = np.asarray(ctrl, float)
    d = np.linalg.norm(sw.centres[None, :, :] - ctrl[:, None, :], axis=2)
    return np.maximum.accumulate(np.argmin(d, axis=1))


def per_ring(sw, ctrl, values):
    idx = ctrl_rings(sw, ctrl).astype(float)
    return np.interp(np.arange(sw.N, dtype=float), idx, np.asarray(values, float))


def grain_toward(sw, i, d):
    """Grain angle of the vertex of ring i whose radial direction is closest to world direction d."""
    dirs = C.normalize(sw.smooth[i] - sw.centres[i])
    return float(sw.grain[i, int(np.argmax(dirs @ np.asarray(d, float)))])


def rotate(vec, axis, a):
    axis = C.normalize(np.asarray(axis, float))
    vec = np.asarray(vec, float)
    return vec * math.cos(a) + np.cross(axis, vec) * math.sin(a) + axis * (axis @ vec) * (1.0 - math.cos(a))


def view_band(sw, cam, W, off, hook_uv=None, lat_dir=None, jitter=None):
    """Band on the camera side of a sweep. Returns (dist, centre, lat, tc): dist = angular distance
    (rad) of each grid vertex from the band edge (< 0 inside); the band centre leans by `off`
    (rad per ring) towards `lat` = the screen point hook_uv at the ring's depth (or lat_dir)."""
    loc = np.asarray(CAMS[cam]["loc"], float)
    X, T = sw.centres, sw.T
    tc = C.normalize(loc[None, :] - X)
    tc = C.normalize(tc - np.sum(tc * T, -1)[:, None] * T)
    if hook_uv is not None:
        _, _, depth = project(cam, X)
        lat = np.array([S(cam, hook_uv[0], hook_uv[1], float(d)) for d in depth]) - X
    else:
        lat = np.broadcast_to(np.asarray(lat_dir, float), X.shape).copy()
    lat = lat - np.sum(lat * T, -1)[:, None] * T
    lat = C.normalize(lat - np.sum(lat * tc, -1)[:, None] * tc)
    b = C.normalize(tc * np.cos(off)[:, None] + lat * np.sin(off)[:, None])
    dirs = C.normalize(sw.smooth - X[:, None, :])
    ang = np.arccos(np.clip(np.einsum("nmk,nk->nm", dirs, b), -1.0, 1.0))
    if jitter is not None:
        ang = ang + jitter
    return ang - np.asarray(W, float)[:, None], b, lat, tc


def channel_mask(sw, dist, f, crack, N3, seed_off=0.0, islands=0.9, holes=0.9, back=(-0.62, -0.40)):
    """Moss coverage (N, M) around a bare channel (dist < 0 inside): a ragged edge with moss
    islands on the bark near it, bark holes in the moss next to it, moss creeping into the cracks
    close to the moss, and no moss on the never-seen back. Returns (mask, big-scale noise)."""
    inside = -dist
    so = float(seed_off)
    core = C.smoothstep(-0.025, 0.025, dist)
    isl = C.smoothstep(0.12, 0.28, BL.grain_noise(sw, N3, 12.0, stretch=2.0, octaves=3, offset=41.0 + so))
    isl = isl * (1.0 - C.smoothstep(0.04, 0.32, inside))
    hol = C.smoothstep(0.18, 0.34, BL.grain_noise(sw, N3, 11.0, stretch=2.5, octaves=3, offset=63.0 + so))
    hol = hol * (1.0 - C.smoothstep(0.03, 0.14, dist))
    crk = C.smoothstep(0.3, 0.75, crack) * (1.0 - C.smoothstep(0.03, 0.18, inside))
    big = N3.fbm(sw.smooth * 1.6 + 3.0 + so, 3)
    small = N3.fbm(sw.smooth * 6.0 + 9.0 + so, 2)
    m = core + (1.0 - core) * np.maximum(islands * isl, 0.85 * crk) - holes * hol * core
    m = m - 1.4 * (1.0 - C.smoothstep(back[0], back[1], f + 0.1 * small))
    return m, big


# ------------------------------------------------------------------------------------------
# the J
# ------------------------------------------------------------------------------------------

J_SWEEP = dict(seed=SEED + 1, ring_spacing=0.013, segments=176, up_hint=(0.0, -1.0, 0.0), twist=2.4,
               twist_noise=0.25, profile=dict(lobes=[(2, 0.09, 0.3), (3, 0.05, 1.4), (5, 0.02, 0.2)], var=0.6,
                                              drift=0.5),
               radius_noise=0.045, lump=0.03)


def build_j(coll, mat_wood, mat_moss, N3, rng):
    cam = "cam_branch_main"
    right, up, fwd = C.cam_basis(cam_q(cam))
    ctrl = [S(cam, *p) for p in J_UVD]
    probe = C.sweep_trunk("wood_branch_j_part", coll, ctrl, J_R, build_object=False, **J_SWEEP)
    uu, vv, _ = project(cam, probe.centres)
    sn = probe.s / probe.length
    _, _, lat0, tc0 = view_band(probe, cam, per_ring(probe, ctrl, J_CH_W), per_ring(probe, ctrl, J_CH_OFF),
                                hook_uv=J_HOOK_UV)

    def s_at(v):
        return float(probe.s[int(np.argmax(vv >= v))])

    # knots: a swelling on the outer side of the bend, a broken twig stub under the arm (frame 10:
    # x~460, y~960) and a small knot near the end
    i_bend = int(np.argmax(vv >= 0.68))
    arm = np.nonzero((vv > 0.75) & (uu < 0.45) & (sn > 0.5))[0]
    i_stub = int(arm[np.argmin(np.abs(uu[arm] - 0.32))])
    i_e = int(arm[np.argmin(np.abs(uu[arm] - 0.17))])
    knots = [(sn[i_bend], grain_toward(probe, i_bend, -lat0[i_bend]), 0.10, 0.05, 0.4),
             (sn[i_stub], grain_toward(probe, i_stub, -up - 0.3 * fwd), 0.035, 0.6, 0.6),
             (sn[i_e], grain_toward(probe, i_e, tc0[i_e]), 0.06, 0.05, 0.3)]
    sw = C.sweep_trunk("wood_branch_j_part", coll, ctrl, J_R, build_object=False, knots=knots, **J_SWEEP)
    tw0 = np.array(sw.twist_s, float)                   # construction twist (knot angles live in it)
    BL.retwist(sw, per_ring(sw, ctrl, J_TWIST), N3, amp=0.12, freq=0.9, offset=3.3)

    # the channel: wobbling centre and width along the log
    W = per_ring(sw, ctrl, J_CH_W) * (1.0 + 0.22 * N3.line(sw.s, freq=2.4, offset=5.0))
    off = per_ring(sw, ctrl, J_CH_OFF) + 0.07 * N3.line(sw.s, freq=1.7, offset=9.0)
    _, b, lat, tc = view_band(sw, cam, W, off, hook_uv=J_HOOK_UV)

    # ---- bark: a few big rounded twisted strands (rope-like domes, Voronoi cells in grain space)
    # with deep V creases between them, long deep grain-following cracks (some branching), the
    # cleft, long cracks into the bend. No fine ridges: the bark texture carries the fibres.
    rel_s, crease = BL.strand_relief(sw, N3, n_around=14, depth=0.026, r_ref=0.18, meander=0.7, merge=1.0,
                                     crown=0.85, offset=101.0)
    cracks = BL.make_cracks(sw, rng, 10, depth=(0.016, 0.03), width=(0.010, 0.018), length=(0.5, 1.6),
                            meander=0.05, branch=0.4)
    cdir = tc * math.cos(J_CLEFT["angle"]) + lat * math.sin(J_CLEFT["angle"])
    th_cleft = BL.ring_angle(sw, cdir) + 0.04 * N3.line(sw.s, freq=3.0, offset=12.0)
    cleft = dict(theta=th_cleft, s0=s_at(J_CLEFT["v"][0]), s1=s_at(J_CLEFT["v"][1]), depth=J_CLEFT["depth"],
                 width=J_CLEFT["width"], sharp=1.15, taper=0.15, moff=21.0)
    cracks.append(cleft)
    i_b = int(np.argmax(vv >= 0.62))
    for k, a in enumerate((-0.34, 0.0, 0.3)):
        g = grain_toward(sw, i_b, BL.rotate(b[i_b], sw.T[i_b], a))
        cracks.append(dict(g0=g, s0=s_at(0.36) + 0.04 * k, s1=min(sw.length - 0.3, float(sw.s[i_b]) + 0.55 + 0.1 * k),
                           depth=0.022 - 0.003 * k, width=0.011, meander=0.04, mfreq=2.0, moff=31.0 + 7.0 * k,
                           sharp=1.5, taper=0.15))
    rel_c, crack_c = BL.crack_relief(sw, cracks, N3, r_ref=0.18)
    # (revision 3) short fissures across the strands every 5-13 cm (the strands read as long plates,
    # frame 10 / details 06); as fine as the 13 mm rings allow - the bark texture carries the rest
    rel_x, cross = BL.cross_fissures(sw, np.random.default_rng(SEED + 61), N3, n_around=14, spacing=(0.05, 0.13),
                                     depth=0.009, width=0.009, r_ref=0.18, meander=0.7, offset=101.0,
                                     extent=(0.2, 0.5), p=0.6, skew=0.8, crown=(0.1, 0.32), steep=0.4)
    rel = np.minimum(np.minimum(rel_s, rel_c), rel_s + rel_x) + BL.fibre_relief(sw, N3, spacing=0.02, amp=0.0004,
                                                                                plate_amp=0.0008, plate_freq=14.0,
                                                                                offset=0.0)
    crack = np.maximum(np.maximum(crack_c, 0.8 * crease), 0.6 * cross)
    # the log end under the stumps: a low jagged, sunken fracture (the stumps carry the height)
    # (revision 3: the lens side of the rim is seen now - broken bark plates of different heights
    # leaning out a little, not a clean cup edge)
    end_spec = dict(kind="splinter", seed=3.0, jag=0.28, sink=0.32, fibre=0.16, outer_rings=4,
                    splinters=[dict(dir=tuple(-0.3 * right + 0.95 * fwd), height=0.55, width=0.5, thick=0.3,
                                    lean=-0.05, tip=1.0, rag=0.5),
                               dict(dir=tuple(0.85 * right + 0.5 * fwd), height=0.4, width=0.4, thick=0.25, lean=0.1,
                                    tip=0.9, rag=0.5),
                               dict(dir=tuple(-0.55 * right - 0.85 * fwd), height=0.58, width=0.38, thick=0.3,
                                    lean=0.12, tip=0.8, rag=0.6),
                               dict(dir=tuple(-0.95 * right - 0.25 * fwd), height=0.45, width=0.3, thick=0.3, lean=0.08,
                                    tip=0.9, rag=0.6),
                               dict(dir=tuple(0.1 * right - 1.0 * fwd), height=0.22, width=0.3, thick=0.25, lean=0.15,
                                    tip=0.8, rag=0.6)])
    BL.finish_sweep(sw, rel, "wood_branch_j_part", coll, ("flat", end_spec), mat_wood, J_SWEEP["seed"])

    # ---- moss: everywhere but the ragged channel (and the cleft, the twig stub), over the broken
    # rim into the end mound. The channel edge: bays (2 / m), meanders (5 / m), small tongues (14 / m)
    # (revision 3: + square-ish tongues / bays 2-6 cm at 9 / m)
    tng = BL.grain_noise(sw, N3, 9.0, stretch=2.0, octaves=2, offset=47.0)
    jit = (0.16 * BL.grain_noise(sw, N3, 2.2, stretch=1.5, octaves=2, offset=37.0) +
           0.16 * BL.grain_noise(sw, N3, 5.0, stretch=3.0, octaves=3, offset=17.0) +
           0.08 * BL.grain_noise(sw, N3, 14.0, stretch=2.0, octaves=2, offset=29.0) +
           0.12 * np.sign(tng) * np.abs(tng) ** 0.6)
    dist, _, _, _ = view_band(sw, cam, W, off, hook_uv=J_HOOK_UV, jitter=jit)
    f = facing(sw)
    m, big = channel_mask(sw, dist, f, crack, N3)
    ks, kth, kr = knots[1][0], knots[1][1], knots[1][2]                # the twig stub stays bark
    dk = np.sqrt((sw.s[:, None] - ks * sw.length) ** 2 +
                 (sw.radius[:, None] * BL.wrap(sw.theta[None, :] - tw0[:, None] - kth)) ** 2)
    m -= 1.6 * (1.0 - C.smoothstep(1.2 * kr, 2.1 * kr, dk))
    xc = np.abs(BL.wrap(sw.theta[None, :] - th_cleft[:, None])) * sw.radius[:, None]
    env = C.smoothstep(cleft["s0"], cleft["s0"] + 0.1, sw.s) * (1.0 - C.smoothstep(cleft["s1"] - 0.1, cleft["s1"], sw.s))
    m -= 1.4 * env[:, None] * (1.0 - C.smoothstep(0.02, 0.04, xc))
    # the broken end: the coat runs up to the rim (the moss annulus over the end face starts there)
    # and leaves the J's bark bare only where the chunk (the split-off front wall) rises out of the
    # front face
    frame = _end_frame(sw, right, fwd)
    m -= end_opening(sw, frame, N3)
    m = C.blur_grid(np.clip(m, 0.0, 1.0), 2)
    dirs = C.normalize(sw.smooth - sw.centres[:, None, :])
    outer = -np.einsum("nmk,nk->nm", dirs, lat)
    side = 1.0 - C.smoothstep(0.0, 0.7, np.abs(f))
    rs = np.sqrt(sw.radius / 0.2)[:, None]
    base = (rs * (0.010 + 0.006 * C.smoothstep(-0.3, 0.6, big) + 0.026 * side) +
            0.005 * C.smoothstep(0.2, 0.8, outer))
    # full thickness up to the rim: the moss over the end face starts on the coat's last ring and
    # runs on from there (no dip, so no crease where the two meet)
    endf = np.ones((sw.N, 1))
    # the carpet thickens gradually away from the bark (no lip at the channel edge)
    ramp = C.smoothstep(0.35, 0.95, C.blur_grid(m, 6))
    coat, info = BL.moss_coat("moss_branch_j_part", coll, sw, m, base=base * endf, lump=0.02 * rs * endf, noise=N3,
                              material=mat_moss, sag=0.35, seed_off=0.0, ramp=ramp, rng=rng, clump=0.08)
    return dict(sw=sw, wood=dict(sw=sw, moss=m, crack=crack, chan=dist), moss=dict(obj=coat, info=info),
                right=right, up=up, fwd=fwd, end_frame=frame)


def _seg_dist(P, A, B):
    """Distance of points P (..., 3) to the segment A-B."""
    ab = B - A
    tt = np.clip(np.einsum("...k,k->...", P - A, ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(P - (A + tt[..., None] * ab), axis=-1)


def end_opening(sw, frame, N3):
    """Moss to take away from the J's coat at the broken end (N, M): the chunk's footprint on the
    front face (its lower body shows below the ring)."""
    c_e, t_e, r_e, rp, fp = frame
    L = float(sw.length)
    s2 = sw.s[:, None]
    cut = np.zeros(sw.smooth.shape[:2])
    chunk = next(sp for sp in J_STUMPS if sp["name"] == "chunk")
    cp, _ = stump_axis(chunk, sw, frame)
    seg = [cp[0], cp[1], cp[2], cp[2] + t_e * 0.03]
    near = s2 > L - 0.25
    dmin = np.full(sw.smooth.shape[:2], np.inf)
    for A, B in zip(seg[:-1], seg[1:]):
        dmin = np.minimum(dmin, _seg_dist(sw.smooth, np.asarray(A, float), np.asarray(B, float)))
    reach = 0.046 + 0.01 * N3.fbm(sw.smooth * 50.0 + 71.0, 2)
    cut = cut + 1.6 * near * (1.0 - C.smoothstep(reach - 0.01, reach + 0.004, dmin))
    return cut


# ------------------------------------------------------------------------------------------
# the broken end: stumps, torn fibres and their moss
# ------------------------------------------------------------------------------------------

def _end_frame(sw, right, fwd):
    ring = sw.N - 1
    c, t, r = sw.centres[ring], sw.T[ring], float(sw.radius[ring])
    rp = C.normalize(right - (right @ t) * t)
    fp = C.normalize(fwd - (fwd @ t) * t)
    fp = C.normalize(fp - (fp @ rp) * rp)
    return c, t, r, rp, fp


def _log_point(sw_log, s, a, b, rp, fp):
    """Point inside the log on the ring at arc length s, offset (a, b) x ring radius along the
    camera right / forward directions projected on that ring."""
    i = int(np.clip(np.searchsorted(sw_log.s, s), 0, sw_log.N - 1))
    t = sw_log.T[i]
    r1 = C.normalize(rp - (rp @ t) * t)
    f1 = C.normalize(fp - (fp @ t) * t)
    f1 = C.normalize(f1 - (f1 @ r1) * r1)
    return sw_log.centres[i] + float(sw_log.radius[i]) * (a * r1 + b * f1)


def _end_dir(phi, rp, fp):
    """Direction in the end-ring plane: phi from the lens side (0) towards camera right (+pi/2)."""
    return math.cos(phi) * (-fp) + math.sin(phi) * rp


def stump_axis(spec, sw_log, frame):
    """Control points (root inside the log, on the log's way to the ring, the ring crossing, then
    up with lean and bow) and radii of a stump."""
    c, t, r_end, rp, fp = frame
    a, b = spec["ring"]
    ra, rb, back = spec["root"]
    H = float(spec["height"])
    L = float(sw_log.length)
    P0 = _log_point(sw_log, L - back, ra, rb, rp, fp)
    Pm = _log_point(sw_log, L - 0.45 * back, 0.5 * (ra + a), 0.5 * (rb + b), rp, fp)
    P1 = c + r_end * (a * rp + b * fp)
    lean = spec["lean"][0] * rp + spec["lean"][1] * fp
    bow = spec.get("bow", (0.0, 0.0))
    bowv = bow[0] * rp + bow[1] * fp
    up = [P1 + t * h + lean * (h ** 1.3) / (H ** 0.3) + bowv * math.sin(math.pi * h / H)
          for h in (H / 3.0, 2.0 * H / 3.0, H)]
    r = spec["radii"]
    radii = [r[0], 0.5 * (r[0] + r[1]), r[1], r[2], r[3], r[4]]
    return [P0, Pm, P1] + up, radii


def build_stump(spec, sw_log, frame, coll, mat_wood, mat_moss, N3, rng):
    """A stump of the broken end (revision 3): its own sweep from inside the curled log up through
    the end ring; an irregular, slightly oval section whose long axis turns along the length,
    bulges and dents (an optional kidney lobe), furrowed bark (strands with sparse cross-fissures),
    a few deeper cracks and fibres; a jagged top (splinter cap: spikes of different heights, pale
    fracture); moss coats (sheaths / base cushion) as their own sheets. Returns (wood part, moss
    part or None, top frame)."""
    c, t, r_end, rp, fp = frame
    pts, radii = stump_axis(spec, sw_log, frame)
    seed = SEED + 100 + int(spec["seed"])
    name = f"wood_branch_stump_{spec['name']}"
    st = C.sweep_trunk(name, coll, pts, radii, build_object=False, seed=seed, ring_spacing=0.0045,
                       segments=int(spec["segments"]), up_hint=tuple(fp), twist=float(spec["twist"]), twist_noise=0.25,
                       profile=dict(lobes=spec["lobes"], var=0.45, drift=0.35, lobe_twist=spec.get("lobe_twist", 0.5)),
                       radius_noise=0.10, lump=0.05)
    L = float(st.length)
    i_ring = int(np.argmin(np.linalg.norm(st.centres - pts[2], axis=1)))
    s_ring = float(st.s[i_ring])
    # section: oval whose long axis starts along (camera right turned towards forward) and turns
    amp, ax, turn = spec["oval"][:3]
    d_ax = math.cos(ax) * rp + math.sin(ax) * fp
    th0 = BL.ring_angle(st, d_ax) + turn * st.s / L
    lobes = [(2, amp, th0)]
    if len(spec["oval"]) > 3:          # kidney: convex bark side to the lens, hollow back (a wall piece)
        lobes.append((3, spec["oval"][3], BL.ring_angle(st, -fp) + turn * st.s / L))
    BL.reshape_section(st, lobes, noise=N3, bumps=spec["bumps"][0], bump_freq=spec["bumps"][1],
                       bump_stretch=1.8, offset=float(seed))
    # bark: wavy furrows along the grain (ridges that wander, merge and split), short breaks across
    # the ridges, a few deeper cracks, fine fibres (frame 10: coarse, deeply furrowed dark bark)
    n_fu, dep_fu, crown_fu, mea_fu = spec["furrows"]
    rel_s, crease = BL.strand_relief(st, N3, n_around=n_fu, depth=dep_fu, r_ref=0.035, meander=mea_fu, merge=1.0,
                                     crown=crown_fu, offset=float(seed))
    rel_x, cross = BL.cross_fissures(st, rng, N3, n_around=n_fu, spacing=(0.025, 0.06), depth=0.55 * dep_fu,
                                     width=0.003, r_ref=0.035, meander=mea_fu, offset=float(seed), p=0.55)
    cracks = BL.make_cracks(st, rng, int(spec.get("cracks", 4)), depth=(0.006, 0.011), width=(0.003, 0.0055),
                            length=(0.06, 0.2), meander=0.15, branch=0.3)
    rel_c, crack_c = BL.crack_relief(st, cracks, N3, r_ref=0.035)
    rel = np.minimum(np.minimum(rel_s, rel_c), rel_s + rel_x) + BL.fibre_relief(
        st, N3, spacing=0.006, amp=0.0006, plate_amp=0.0008, plate_freq=40.0, offset=float(seed))
    crack = np.maximum(np.maximum(crack_c, 0.9 * crease), 0.7 * cross)
    # torn top
    splinters = []
    for sp in spec["spikes"]:
        d = _end_dir(sp[0], rp, fp)
        splinters.append(dict(dir=tuple(d), height=float(sp[1]), width=float(sp[2]),
                              thick=float(sp[3]) if len(sp) > 3 else 0.32, lean=0.06 * float(rng.uniform(-1, 1)),
                              tip=float(rng.uniform(0.6, 1.1)), rag=0.6))
    cap = dict(kind="splinter", seed=float(spec["seed"]), jag=float(spec.get("jag", 0.25)), sink=0.18, fibre=0.4,
               outer_rings=5, rho=(0.94, 0.86, 0.75, 0.62, 0.48, 0.33, 0.18), splinters=splinters)
    BL.finish_sweep(st, rel, name, coll, ("flat", cap), mat_wood, seed)

    # ---- moss: coats (sheath / fringe), the base cushion; ragged edges in grain space
    s2 = st.s[:, None]
    m = np.zeros((st.N, st.M))
    thick = np.zeros((st.N, st.M))
    lump = np.zeros((st.N, st.M))
    for k, (phi, W, top, th, lp) in enumerate(spec["coats"]):
        jit = (0.22 * BL.grain_noise(st, N3, 28.0, stretch=2.5, octaves=3, offset=float(seed) + 11.0 * k) +
               0.12 * BL.grain_noise(st, N3, 70.0, stretch=1.5, octaves=2, offset=float(seed) + 5.0 * k))
        dist = BL.dir_band(st, _end_dir(phi, rp, fp), W, jitter=jit)
        topk = top * L * (1.0 + 0.07 * N3.fbm(st.smooth * 30.0 + float(seed) + k, 2))
        mk = C.smoothstep(0.06, -0.06, dist) * (1.0 - C.smoothstep(topk - 0.03, topk, s2))
        mk = mk * C.smoothstep(s_ring - 0.035, s_ring - 0.01, s2)
        core = C.smoothstep(0.0, -0.9 * W, dist)
        m = np.maximum(m, mk)
        thick = np.maximum(thick, mk * th * (0.55 + 0.45 * core))
        lump = np.maximum(lump, mk * lp)
    h_front, h_back, th_b, lp_b = spec["base"]
    dirs = C.normalize(st.smooth - st.centres[:, None, :])
    back = np.clip(0.5 - 0.5 * np.einsum("nmk,k->nm", dirs, -fp), 0.0, 1.0)          # 0 lens side .. 1 back
    hb = s_ring + h_front + (h_back - h_front) * back ** 1.3
    hb = hb + 0.012 * N3.fbm(st.smooth * 45.0 + float(seed) + 3.0, 2) + 0.008 * BL.grain_noise(
        st, N3, 60.0, stretch=1.0, octaves=2, offset=float(seed) + 7.0)
    mb = (1.0 - C.smoothstep(hb - 0.012, hb, s2)) * C.smoothstep(s_ring - 0.035, s_ring - 0.012, s2)
    m = np.maximum(m, mb)
    thick = np.maximum(thick, mb * th_b * (0.6 + 0.4 * back))
    lump = np.maximum(lump, mb * lp_b)
    m = C.blur_grid(np.clip(m, 0.0, 1.0), 2)
    ramp = C.smoothstep(0.35, 0.95, C.blur_grid(m, 4))
    coat, info = BL.moss_coat(f"moss_branch_stump_{spec['name']}", coll, st, m, base=np.maximum(thick, 0.004),
                              lump=lump, noise=N3, material=mat_moss, sag=0.2, lump_freq=(30.0, 60.0), mound_freq=14.0,
                              seed_off=float(seed), ramp=ramp, rng=rng, clump=0.022, tuck=0.006)
    # moss spots on the bark (engine woodFuzz): upper faces turned towards spots[0]
    phi_s, k_s = spec["spots"]
    face = np.einsum("nmk,k->nm", st.nsmooth, C.normalize(_end_dir(phi_s, rp, fp) + 0.6 * t))
    patch = C.smoothstep(0.1, 0.45, N3.fbm(st.smooth * 55.0 + float(seed) + 41.0, 2))
    spots = k_s * patch * C.smoothstep(0.05, 0.45, face) * C.smoothstep(s_ring, s_ring + 0.03, s2)
    # the band just above the cushion edge (woodFuzz creeps up from the cushion)
    above = C.smoothstep(0.0, 0.02, hb - s2 + 0.025) * (1.0 - C.smoothstep(0.3, 0.6, m))
    wood = dict(sw=st, moss=m, crack=crack, chan=None, kind="stump", spots=spots, above=above, name=spec["name"])
    moss = dict(obj=coat, info=info) if coat is not None else None
    top = dict(c=st.centres[-1], t=st.T[-1], r=float(st.radius[-1]))
    return wood, moss, top


def build_fibre(name, P0, P1, length, lean_dir, r0, coll, mat_wood, N3, seed, kind="fibre", flat=0.45, segments=12):
    """A loose fibre / torn splinter: a thin flat twisted sliver from P0 (buried) through P1 out to
    a pointed tip `length` m further along the direction P0 -> P1, bent towards lean_dir."""
    P0 = np.asarray(P0, float)
    P1 = np.asarray(P1, float)
    d = C.normalize(P1 - P0)
    lv = np.asarray(lean_dir, float)
    pts = [P0, P1] + [P1 + d * h + lv * h * h / length for h in (0.5 * length, length)]
    radii = [r0, r0 * 0.9, r0 * 0.55, max(0.0012, r0 * 0.16)]
    up = tuple(lv) if np.linalg.norm(lv) > 1e-6 else (0.0, 0.0, 1.0)
    fsw = C.sweep_trunk(name, coll, pts, radii, build_object=False, seed=seed, ring_spacing=0.004, segments=segments,
                        up_hint=up, twist=1.4, twist_noise=0.3,
                        profile=dict(lobes=[(2, flat, 0.4 * seed), (3, 0.08, 1.0)], var=0.3, drift=0.3),
                        radius_noise=0.12, lump=0.06)
    rel = BL.fibre_relief(fsw, N3, spacing=0.004, amp=0.0003, plate_amp=0.0003, plate_freq=60.0, offset=float(seed))
    BL.finish_sweep(fsw, rel, name, coll, ("flat", "round"), mat_wood, seed)
    s_out = float(fsw.s[int(np.argmin(np.linalg.norm(fsw.centres - P1, axis=1)))])
    return dict(sw=fsw, moss=np.zeros((fsw.N, fsw.M)), crack=np.zeros((fsw.N, fsw.M)), chan=None, kind=kind,
                s_out=s_out)


def build_end(j, coll, mat_wood, mat_moss, N3, rng):
    """The broken end: a moss annulus over the broken rim, the two stumps with their moss and loose
    fibres at their tops, torn fibres out of the fracture. Returns wood parts (-> wood_branch_stumps)
    and moss parts (-> moss_branch_j)."""
    frame = j["end_frame"]
    c, t, r_end, rp, fp = frame
    wood, moss = [], []
    cr = J_END_MOSS

    def keep(rho, dirs):
        rho = np.asarray(rho, float)[:, None]
        rag = 0.12 * N3.fbm(dirs * 4.0 + 5.0, 2)[None, :] + 0.06 * N3.fbm(dirs * 11.0 + 9.0, 2)[None, :]
        return C.smoothstep(cr["rho"][0], cr["rho"][1], rho + rag)

    mo, mi = BL.end_crescent("moss_branch_end_mound", coll, j["sw"], j["moss"]["info"], keep=keep, height=cr["height"],
                             lump=cr["lump"], noise=N3, rng=rng, clump=0.03, rings=14, material=mat_moss, seed_off=7.0)
    if mo is not None:
        moss.append(dict(obj=mo, info=mi))
    for spec in J_STUMPS:
        w, m, top = build_stump(spec, j["sw"], frame, coll, mat_wood, mat_moss, N3, rng)
        wood.append(w)
        if m is not None:
            moss.append(m)
        # loose fibres standing at the top rim, leaning out
        tc, tt, tr = top["c"], top["t"], top["r"]
        for k, (phi, ln, out, r0) in enumerate(spec["fibres"]):
            d = _end_dir(phi, rp, fp)
            d = C.normalize(d - (d @ tt) * tt)
            P1 = tc + d * tr * 0.82 + tt * 0.004
            P0 = P1 - tt * 0.02
            seed = SEED + 300 + 10 * int(spec["seed"]) + k
            wood.append(build_fibre(f"wood_branch_fibre_{spec['name']}_{k}", P0, P1, ln, d * out, r0, coll, mat_wood,
                                    N3, seed, kind="fresh", flat=0.62))
    for k, ((a, b), ln, (lr, lf), r0) in enumerate(J_FIBRES):
        P0 = _log_point(j["sw"], j["sw"].length - 0.04, a, b, rp, fp)
        P1 = c + r_end * (a * rp + b * fp)
        wood.append(build_fibre(f"wood_branch_fibre_{k}", P0, P1, ln, lr * rp + lf * fp, r0, coll, mat_wood, N3,
                                SEED + 200 + k, kind="fibre", flat=0.35))
    return dict(wood=wood, moss=moss)


# ------------------------------------------------------------------------------------------
# the fragment
# ------------------------------------------------------------------------------------------

F_SWEEP = dict(seed=SEED + 7, ring_spacing=0.012, segments=144, up_hint=(0.0, 0.0, 1.0), twist=0.6,
               twist_noise=0.15, profile=dict(lobes=[(2, 0.10, 0.8), (3, 0.05, 0.2)], var=0.6, drift=0.5),
               radius_noise=0.05, lump=0.03)


def build_fragment(coll, mat_wood, mat_moss, N3, rng):
    cam = "cam_branch_main"
    right, up, fwd = C.cam_basis(cam_q(cam))
    ctrl = [S(cam, *p) for p in F_UVD]
    probe = C.sweep_trunk("wood_branch_frag", coll, ctrl, F_R, build_object=False, **F_SWEEP)
    uu, vv, _ = project(cam, probe.centres)
    sn = probe.s / probe.length
    zup = np.array([0.0, 0.0, 1.0])
    i_m = int(np.argmin(np.abs(uu - 0.79)))                 # the second moss mound sits on a swelling
    knots = [(sn[i_m], grain_toward(probe, i_m, zup - 0.3 * fwd), 0.09, 0.32, 0.4)]
    sw = C.sweep_trunk("wood_branch_frag", coll, ctrl, F_R, build_object=False, knots=knots, **F_SWEEP)
    BL.retwist(sw, per_ring(sw, ctrl, F_TWIST), N3, amp=0.1, freq=1.2, offset=13.0)
    rel_s, crease = BL.strand_relief(sw, N3, n_around=13, depth=0.016, r_ref=0.12, meander=0.45, merge=1.0,
                                     crown=0.85, offset=151.0)
    cracks = BL.make_cracks(sw, rng, 7, depth=(0.01, 0.02), width=(0.008, 0.014), length=(0.3, 0.9),
                            meander=0.06, branch=0.35)
    rel_c, crack_c = BL.crack_relief(sw, cracks, N3, r_ref=0.12)
    rel = np.minimum(rel_s, rel_c) + BL.fibre_relief(sw, N3, spacing=0.016, amp=0.0004, plate_amp=0.0007,
                                                     plate_freq=16.0, offset=50.0)
    crack = np.maximum(crack_c, 0.8 * crease)
    # the near end: broken at a slant (higher at the top), fibre spikes, sunken fracture
    start_spec = dict(kind="splinter", seed=9.0, jag=0.2, sink=0.16, fibre=0.16, outer_rings=4,
                      splinters=[dict(dir=tuple(up * 0.9 - fwd * 0.3), height=0.85, width=2.6, thick=0.9, lean=0.0,
                                      tip=1.0, rag=0.35),
                                 dict(dir=tuple(up * 0.8 - fwd * 0.6), height=1.25, width=0.4, thick=0.35, lean=0.1,
                                      tip=0.7, rag=0.5),
                                 dict(dir=tuple(up * 0.4 + right * 0.7), height=0.95, width=0.3, thick=0.3, lean=0.05,
                                      tip=0.8, rag=0.5),
                                 dict(dir=tuple(-up * 0.3 - fwd * 0.9), height=0.6, width=0.35, thick=0.3, lean=0.0,
                                      tip=0.8, rag=0.5)])
    BL.finish_sweep(sw, rel, "wood_branch_frag", coll, (start_spec, "flat"), mat_wood, F_SWEEP["seed"])
    # moss: top and front, a narrow ragged bark channel along the upper front, bare broken end,
    # bark on the underside
    W = per_ring(sw, ctrl, F_CH_W) * (1.0 + 0.25 * N3.line(sw.s, freq=3.0, offset=61.0))
    off = per_ring(sw, ctrl, F_CH_OFF) + 0.08 * N3.line(sw.s, freq=2.0, offset=67.0)
    jit = (0.14 * BL.grain_noise(sw, N3, 6.0, stretch=3.0, octaves=3, offset=77.0) +
           0.07 * BL.grain_noise(sw, N3, 16.0, stretch=2.0, octaves=2, offset=83.0))
    dist, _, _, _ = view_band(sw, cam, W, off, lat_dir=(0.0, 0.0, 1.0), jitter=jit)
    f = facing(sw)
    m, big = channel_mask(sw, dist, f, crack, N3, seed_off=31.0)
    n = sw.nsmooth
    # bare broken end only on the top-left corner (bark + pale wood, frame 10 x 885-960, y 770-850),
    # ragged; the front moss reaches the rim
    reach = 0.018 + 0.05 * C.smoothstep(0.15, 0.7, n[..., 2]) + 0.014 * N3.fbm(sw.smooth * 14.0 + 3.0, 2)
    m -= 1.5 * (1.0 - C.smoothstep(reach - 0.02, reach + 0.01, sw.s[:, None]))
    m -= 1.5 * (1.0 - C.smoothstep(0.004, 0.016, sw.s))[:, None]          # moss up to the broken rim
    m -= 1.2 * C.smoothstep(-0.38, -0.62, n[..., 2] + 0.08 * N3.fbm(sw.smooth * 8.0 + 5.0, 2))
    m = C.blur_grid(np.clip(m, 0.0, 1.0), 2)
    s_m = sw.s[i_m]
    mound = np.exp(-((sw.s - s_m) / 0.085) ** 2)[:, None] * C.smoothstep(0.0, 0.6, n[..., 2])
    side = 1.0 - C.smoothstep(0.0, 0.7, np.abs(f))
    rs = np.sqrt(sw.radius / 0.12)[:, None]
    base = rs * (0.012 + 0.007 * C.smoothstep(-0.4, 0.6, big) + 0.010 * side) + 0.03 * mound
    ramp = C.smoothstep(0.35, 0.95, C.blur_grid(m, 6))
    coat, info = BL.moss_coat("moss_branch_frag", coll, sw, m, base=base, lump=0.018 * rs, noise=N3,
                              material=mat_moss, sag=0.3, seed_off=31.0, ramp=ramp, rng=rng, clump=0.055)
    return dict(sw=sw, wood=dict(sw=sw, moss=m, crack=crack, chan=dist), moss=dict(obj=coat, info=info))


def build_ball(coll, mat, N3):
    """moss_ball: small irregular sphere at the origin (the engine grows fur and moves it)."""
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=4, radius=BALL_RADIUS)
    C._free_name(bpy.data.meshes, "moss_ball")
    me = bpy.data.meshes.new("moss_ball")
    bm.to_mesh(me)
    bm.free()
    co, _, _, _ = C.mesh_arrays(me)
    d = C.normalize(co)
    lump = 0.07 * N3.fbm(d * 1.6 + 5.0, 3) + 0.025 * N3.fbm(d * 5.0 + 9.0, 2)
    co = d * BALL_RADIUS * (1.0 + lump)[:, None]
    co[:, 2] *= 0.94
    C.set_vertex_positions(me, co)
    me.shade_smooth()
    return C.object_from_mesh("moss_ball", me, coll, mat)


def build_anchors(coll, pts):
    out = []
    for k, (a, p) in enumerate(zip(ANCHORS, pts)):
        name = f"anchor_ball_{k}"
        C.remove_object(name)
        e = bpy.data.objects.new(name, None)
        e.empty_display_type = "SPHERE"
        e.empty_display_size = BALL_RADIUS
        e.location = Vector(p)
        coll.objects.link(e)
        u, v, z = project("cam_branch_p1", np.asarray(p)[None, :])
        um, vm, _ = project("cam_branch_main", np.asarray(p)[None, :])
        on = bool(z[0] > 0.05 and 0.0 <= u[0] <= 1.0 and 0.0 <= v[0] <= 1.0)
        e["loop_index"] = k
        e["loop_phase"] = a["phase"]
        e["video_time_s"] = round(BALL_T0 + a["phase"] * BALL_PERIOD, 3)
        e["period_s"] = BALL_PERIOD
        e["depth_m"] = round(float(z[0]), 3)
        e["screen_uv_rest"] = [round(float(u[0]), 4), round(float(v[0]), 4)]
        e["screen_uv_main"] = [round(float(um[0]), 4), round(float(vm[0]), 4)]
        e["in_frame_rest"] = on
        if k == 0:
            e["loop_note"] = ("closed centripetal Catmull-Rom through anchor_ball_0..11 in index order "
                              "(three.js CatmullRomCurve3(points, true, 'centripetal')). Ball phase = "
                              "((t - %.3f) / %.3f) mod 1; phase -> segment i with loop_phase_i <= phase < "
                              "loop_phase_i+1, w = local fraction, position = getPoint((i + w) / 12). "
                              "Anchors 7-8 are the hidden arc (outside the frame / behind the lens)."
                              % (BALL_T0, BALL_PERIOD))
            e["ball_radius_m"] = BALL_RADIUS
            e["ball_fur_assumed_m"] = BALL_FUR
        out.append(e)
    return out


# ------------------------------------------------------------------------------------------
# COLOR_0: R density, G length, B AO/cavity, A 1
# ------------------------------------------------------------------------------------------

def wood_part_features(part, N3):
    """R (moss density for the engine's wood spots), G (length) and a darkening factor (x AO) for
    one sweep part (grid + caps), in the part's vertex order. R: up to 0.6 in the cracks and in
    small patches on the moss-facing side of the channel, 0.4-0.8 on the bark within ~3 cm of the
    moss border and in the deep cracks, 0 on clean bark and under the moss."""
    sw = part["sw"]
    nv_all = int(part["nv"])
    nv = sw.N * sw.M
    R = np.zeros(nv_all)
    G = np.full(nv_all, 0.3)
    dark = np.ones(nv_all)
    P = sw.pos.reshape(-1, 3)
    m = np.asarray(part["moss"], float).reshape(-1)
    crack = np.asarray(part["crack"], float).reshape(-1)
    under = C.smoothstep(0.3, 0.55, m)
    if part.get("chan") is not None:
        inside = -np.asarray(part["chan"], float).reshape(-1)
        near = 1.0 - C.smoothstep(0.04, 0.3, inside)
    else:
        near = np.full(nv, 0.6)
    patch = C.smoothstep(0.05, 0.35, N3.fbm(P * 16.0 + 77.0, 2))
    dots = C.smoothstep(0.2, 0.5, N3.fbm(P * 45.0 + 13.0, 2))
    r = 0.65 * crack ** 0.6 * (0.3 + 0.7 * patch) * (0.3 + 0.7 * near) + 0.55 * near * dots * patch
    if part.get("chan") is not None:
        # (revision 3) the bark within ~3 cm of the moss border (and around moss islands / in the
        # bays) carries a ragged frill of the engine's wood moss: the border is no clean cut
        rad = np.broadcast_to(np.asarray(sw.radius, float)[:, None], (sw.N, sw.M)).reshape(-1)
        im = inside * rad                                  # m into the channel (< 0: the moss side)
        band = C.smoothstep(-0.004, 0.002, im) * (1.0 - C.smoothstep(0.012, 0.035, im))
        mb = C.blur_grid(np.asarray(part["moss"], float).reshape(sw.N, sw.M), 3).reshape(-1)
        band = np.maximum(band, C.smoothstep(0.04, 0.3, mb))
        frill = C.smoothstep(-0.25, 0.35, N3.fbm(P * 30.0 + 5.0, 2))
        r = np.maximum(r, band * (0.42 + 0.36 * frill))
    # ... and moss deep in the cracks / fissures everywhere
    r = np.maximum(r, C.smoothstep(0.45, 0.85, crack) * (0.4 + 0.25 * patch))
    R[:nv] = np.clip(r * (1.0 - under), 0.0, 0.8)
    G[:nv] = 0.28 + 0.2 * patch + 0.1 * crack
    # dark creases / cracks, the bark in the shade of the moss along the channel edge, darker
    # weathered stumps
    dark[:nv] = 1.0 - 0.5 * np.clip(crack, 0.0, 1.0) ** 1.2
    if part.get("chan") is not None:
        dark[:nv] *= 1.0 - 0.25 * (1.0 - C.smoothstep(0.0, 0.1, inside))
    dark[:nv] *= float(part.get("dark_mul", 1.0))
    for end, meta in getattr(sw, "cap_meta", {}).items():
        a, b = meta["first"], meta["first"] + meta["count"]
        frac = meta["fracture"] & (meta["rho"] < 0.97)
        deep = np.clip(1.0 - meta["rho"], 0, 1)
        fd = float(part.get("fracture_dark", 0.45))          # the stumps' fresh breaks stay light
        d = np.where(frac, fd - 0.2 * deep, float(part.get("dark_mul", 1.0)))
        # splinter tops and the torn rim stay lighter (pale fibres catch the light)
        d = np.where(frac & (meta["splinter"] > 0.4) & (meta["rho"] > 0.75), np.maximum(d, 0.75), d)
        hi = meta["h"] > 0.7 * float(np.max(meta["h"]))
        d = np.where(hi, np.maximum(d, 0.9), d)
        dark[a:b] = d
        R[a:b] = 0.0
    return R, G, dark


def stump_part_features(part, N3):
    """COLOR_0 inputs of one part of wood_branch_stumps (vertex order of the part): R, G, a
    darkening factor (x AO) and the fresh-wood mask. Stumps: R >= 0.6 in the moss spots on the
    upper faces (spots) and in the band just above the base cushion (above), some in the
    fissures, 0 under the moss and on fresh wood; dark fissures and the bark under the moss
    (B ~0.3), weathered bark ~0.6-0.75; fresh breaks (the torn top's fracture faces and the lips
    of its splinters) and the loose fibres: B 0.9-1.0, G = 0 (no moss, and the marker of fresh
    wood: G is >= 0.2 everywhere else on this mesh)."""
    sw = part["sw"]
    nv_all = int(part["nv"])
    nv = sw.N * sw.M
    R = np.zeros(nv_all)
    G = np.full(nv_all, 0.32)
    dark = np.full(nv_all, 0.75)
    fresh = np.zeros(nv_all, bool)
    P = sw.pos.reshape(-1, 3)
    kind = part.get("kind", "stump")
    if kind == "stump":
        m = np.asarray(part["moss"], float).reshape(-1)
        crack = np.asarray(part["crack"], float).reshape(-1)
        under = C.smoothstep(0.3, 0.55, m)
        patch = C.smoothstep(0.0, 0.4, N3.fbm(P * 60.0 + 13.0, 2))
        spots = np.asarray(part["spots"], float).reshape(-1)
        above = np.asarray(part["above"], float).reshape(-1)
        r = np.maximum(np.clip(spots, 0.0, 1.0) * 0.95, 0.68 * above * (0.75 + 0.25 * patch))
        r = np.where(r > 0.05, np.maximum(r, 0.6 * C.smoothstep(0.05, 0.2, r)), r)
        r = np.maximum(r, 0.35 * crack ** 0.7 * patch)
        R[:nv] = np.clip(r * (1.0 - under), 0.0, 0.9)
        G[:nv] = 0.3 + 0.22 * patch + 0.1 * crack
        dark[:nv] = (0.78 - 0.08 * patch) * (1.0 - 0.6 * np.clip(crack, 0.0, 1.0) ** 1.2)
        dark[:nv] = dark[:nv] * (1.0 - 0.45 * under)
        meta = getattr(sw, "cap_meta", {}).get("end")
        if meta is not None:
            a, b = meta["first"], meta["first"] + meta["count"]
            M = sw.M
            k = np.arange(meta["count"])
            Hth = np.asarray(meta["rim_height"], float)[k % M]
            lip = meta["outer"] & (meta["h"] >= 0.72 * np.maximum(Hth, 1e-6))
            fr = meta["fracture"] | lip
            fresh[a:b] = fr
            dark[a:b] = np.where(fr, 1.0, 0.7)
            R[a:b] = 0.0
    else:
        # loose fibres: fresh where they stand out of the wood (the buried root stays bark)
        s_out = float(part.get("s_out", 0.0))
        out = (np.broadcast_to(sw.s[:, None], (sw.N, sw.M)).reshape(-1) > s_out + (0.0 if kind == "fresh" else 0.02))
        fresh[:nv] = out
        fresh[nv + 1:] = True                                    # the round tip cap
        dark[:nv] = np.where(out, 1.0, 0.6)
    G[fresh] = 0.0
    return R, G, dark, fresh


def color_stumps(obj, parts, bvh, N3, use_ao=True):
    me = obj.data
    co, nrm, ed, _ = C.mesh_arrays(me)
    feats = [stump_part_features(p, N3) for p in parts]
    R = np.concatenate([f[0] for f in feats])
    G = np.concatenate([f[1] for f in feats])
    dark = np.concatenate([f[2] for f in feats])
    fresh = np.concatenate([f[3] for f in feats])
    assert len(R) == len(co), (len(R), len(co))
    ao = C.ray_ao(co, nrm, bvh, n_rays=12, max_dist=0.2) if use_ao else np.ones(len(co))
    cx = C.surface_convexity(co, nrm, ed, iterations=3)
    B = ao * (0.8 + 0.2 * np.clip(0.5 + cx / 0.004, 0, 1)) * dark
    B = np.where(fresh, 0.9 + 0.1 * np.clip(ao, 0.0, 1.0) * np.clip(0.5 + cx / 0.004, 0.0, 1.0), B)
    C.write_color0(me, R, G, B)
    return dict(R=R, G=G, B=B, fresh=float(fresh.mean()))


def color_wood(obj, parts, bvh, N3, use_ao=True):
    me = obj.data
    co, nrm, ed, _ = C.mesh_arrays(me)
    feats = [wood_part_features(p, N3) for p in parts]
    R = np.concatenate([f[0] for f in feats])
    G = np.concatenate([f[1] for f in feats])
    dark = np.concatenate([f[2] for f in feats])
    assert len(R) == len(co), (len(R), len(co))
    ao = C.ray_ao(co, nrm, bvh, n_rays=12, max_dist=0.4) if use_ao else np.ones(len(co))
    cx = C.surface_convexity(co, nrm, ed, iterations=3)
    B = ao * (0.84 + 0.16 * np.clip(0.5 + cx / 0.006, 0, 1)) * dark
    C.write_color0(me, R, G, B)
    return dict(R=R, G=G, B=B)


def color_cushion(obj, info, bvh, N3, cams, use_ao=True, ao_groups=()):
    """ao_groups: [(first vertex, count, max_dist m)]: vertex ranges whose AO looks only that far
    (the small moss coats on the stumps: their own cavities, not the whole log end)."""
    me = obj.data
    co, nrm, ed, _ = C.mesh_arrays(me)
    assert len(info["offset"]) == len(co), (len(info["offset"]), len(co))
    ao = C.ray_ao(co, nrm, bvh, n_rays=12, max_dist=0.3) if use_ao else np.ones(len(co))
    if use_ao:
        for a, n, md in ao_groups:
            ao[a:a + n] = C.ray_ao(co[a:a + n], nrm[a:a + n], bvh, n_rays=12, max_dist=md)
    cx = C.surface_convexity(co, nrm, ed, iterations=6)
    locs = [CAMS[c]["loc"] for c in cams]
    rim = C.rim_factor(co, nrm, locs)
    fac = -np.einsum("ij,ij->i", nrm, C.normalize(co - np.asarray(locs[0], float)))
    vis = info["offset"] - info["relief"]
    edge = C.smoothstep(0.0, 0.45, vis / np.maximum(info["thickness"], 1e-4))
    big = N3.fbm(co * 1.4 + 5.0, 3)
    R = np.clip((0.82 + 0.18 * big) * (0.15 + 0.85 * edge), 0, 1)
    R[vis <= 0.0] = 0.0
    R *= 0.35 + 0.65 * C.smoothstep(-0.45, -0.15, fac)          # sparse on the never-seen back
    longs = C.smoothstep(0.35, 0.6, N3.fbm(co * 4.0 + 21.0, 2))  # a few longer tufts
    G = (0.26 + 0.06 * np.clip(cx / 0.008, -1, 1) + 0.1 * rim + 0.05 * big + 0.3 * longs +
         0.08 * np.clip(info["dome"], 0.0, 1.0))
    G *= 0.6 + 0.4 * edge
    B = ao * (0.88 + 0.12 * np.clip(0.5 + cx / 0.01, 0, 1))
    C.write_color0(me, R, np.clip(G, 0.08, 1.0), B)
    return dict(R=R, G=G, B=B)


def color_ball(obj, N3):
    me = obj.data
    co, nrm, ed, _ = C.mesh_arrays(me)
    d = C.normalize(co)
    big = N3.fbm(d * 2.0 + 41.0, 2)
    R = np.clip(0.94 + 0.06 * big, 0, 1)
    G = np.clip(0.5 + 0.12 * big + 0.06 * N3.fbm(d * 6.0 + 3.0, 2), 0.3, 0.75)
    B = np.clip(0.92 + 0.08 * big, 0, 1)
    C.write_color0(me, R, G, B)
    C.box_uvs(me, 0.25)


# ------------------------------------------------------------------------------------------
# build
# ------------------------------------------------------------------------------------------

def build_cameras(coll):
    out = {}
    for name, c in CAMS.items():
        frame_h = 2.0 * c["focus"] * TV
        extras = dict(focus_distance_m=round(c["focus"], 3), frame_height_at_focus_m=round(frame_h, 3),
                      video_time_s=c["t"])
        if name == "cam_branch_main":
            extras["slide_before_main_fh"] = [list(x) for x in DRIFT_BEFORE_MAIN]
        if "settle" in c:
            extras["settle_video_s"] = c["settle"]
            extras["drift_from_main"] = [list(x) for x in CAM_DRIFT]
        if "hold" in c:
            extras["hold_until_video_s"] = c["hold"]
        out[name] = C.make_camera(name, coll, c["loc"], cam_q(name), focus=c["focus"], extras=extras)
    return out


def build_keys(coll):
    centre = S("cam_branch_main", 0.42, 0.55, 2.35)
    for name, k in KEYS.items():
        d = C.normalize(np.asarray(k["dir"], float))
        C.make_key_empty(name, coll, tuple(centre - d * 1.6), tuple(d))


def build(use_ao=True):
    t0 = time.time()
    scene = C.ensure_scene(SCENE_NAME)
    coll = C.ensure_collection(scene, COLL_NAME)
    C.purge_collection(coll)
    mat_moss = C.preview_material("mat_moss", C.PALETTE["moss"], 0.95)
    mat_wood = C.preview_material("mat_wood", C.PALETTE["wood"], 0.85)
    N3 = C.Noise3(SEED)

    build_cameras(coll)
    build_keys(coll)
    t = time.time()
    j = build_j(coll, mat_wood, mat_moss, N3, np.random.default_rng(SEED + 11))
    end = build_end(j, coll, mat_wood, mat_moss, N3, np.random.default_rng(SEED + 12))
    wood_parts = [j["wood"]]
    stump_parts = end["wood"]
    moss_parts = [j["moss"]] + end["moss"]
    for p in wood_parts + stump_parts:       # the part objects are deleted by join_objects
        p["nv"] = len(p["sw"].obj.data.vertices)
    part_tris = {p["sw"].obj.name: C.tri_count(p["sw"].obj) for p in stump_parts}
    part_tris.update({p["obj"].name: C.tri_count(p["obj"]) for p in moss_parts})
    moss_nv = [len(p["obj"].data.vertices) for p in moss_parts]
    moss_names = [p["obj"].name for p in moss_parts]
    wood_j, _ = C.join_objects("wood_branch_j", coll, [p["sw"].obj for p in wood_parts], mat_wood)
    wood_s, _ = C.join_objects("wood_branch_stumps", coll, [p["sw"].obj for p in stump_parts], mat_wood)
    moss_j, _ = C.join_objects("moss_branch_j", coll, [p["obj"] for p in moss_parts], mat_moss)
    for p in wood_parts + stump_parts:
        p["sw"].obj = None
    C.log(f"wood_branch_j {C.tri_count(wood_j)} tris + wood_branch_stumps {C.tri_count(wood_s)} tris + moss_branch_j "
          f"{C.tri_count(moss_j)} tris (J length {j['sw'].length:.2f} m, {j['sw'].N} rings, {time.time() - t:.1f}s)")
    C.log("parts: " + json.dumps(part_tris))
    t = time.time()
    fr = build_fragment(coll, mat_wood, mat_moss, N3, np.random.default_rng(SEED + 13))
    wood_f, moss_f = fr["sw"].obj, fr["moss"]["obj"]
    fr["wood"]["nv"] = len(wood_f.data.vertices)
    C.log(f"wood_branch_frag {C.tri_count(wood_f)} tris + moss_branch_frag {C.tri_count(moss_f)} tris "
          f"({time.time() - t:.1f}s)")
    ball = build_ball(coll, mat_moss, N3)
    pts, fit = fit_anchors()
    build_anchors(coll, pts)
    C.log(f"ball loop vs measured track (rest pose): rms {fit['rms']:.4f}, max {fit['max']:.4f} frame heights "
          f"(at phase {fit['max_at_phase']:.3f})")

    t = time.time()
    meshes = [wood_j, wood_s, moss_j, wood_f, moss_f]
    bvh = C.build_bvh(meshes) if use_ao else None
    cams = list(CAMS.keys())
    cols = {}
    cols["wood_branch_j"] = color_wood(wood_j, wood_parts, bvh, N3, use_ao=use_ao)
    cols["wood_branch_stumps"] = color_stumps(wood_s, stump_parts, bvh, N3, use_ao=use_ao)
    # the stumps' own moss coats (and the end annulus): AO of their own cavities only
    first = np.concatenate([[0], np.cumsum(moss_nv)[:-1]])
    groups = [(int(a), int(n), 0.12) for a, n, nm in zip(first, moss_nv, moss_names) if "stump" in nm]
    groups += [(int(a), int(n), 0.18) for a, n, nm in zip(first, moss_nv, moss_names) if "end_mound" in nm]
    cols["moss_branch_j"] = color_cushion(moss_j, BL.concat_info([p["info"] for p in moss_parts]), bvh, N3, cams,
                                          use_ao=use_ao, ao_groups=groups)
    cols["wood_branch_frag"] = color_wood(wood_f, [fr["wood"]], bvh, N3, use_ao=use_ao)
    cols["moss_branch_frag"] = color_cushion(moss_f, fr["moss"]["info"], bvh, N3, cams, use_ao=use_ao)
    color_ball(ball, N3)
    C.log(f"colours ({time.time() - t:.1f}s)")
    total = sum(C.tri_count(o) for o in coll.objects if o.type == "MESH")
    C.log(f"branch built: {total} tris, {time.time() - t0:.1f}s")
    ranges = {k: {c: [round(float(np.min(v[c])), 3), round(float(np.max(v[c])), 3), round(float(np.mean(v[c])), 3)]
                  for c in ("R", "G", "B")} for k, v in cols.items()}
    ranges["wood_branch_stumps"]["fresh_share"] = round(cols["wood_branch_stumps"]["fresh"], 3)
    return scene, coll, dict(fit=fit, anchors=[list(map(float, p)) for p in pts], color_ranges=ranges,
                             part_tris=part_tris)


# ------------------------------------------------------------------------------------------
# previews, masks, comparison sheets (Blender only: numpy + bpy images)
# ------------------------------------------------------------------------------------------

def _anchor_points(coll):
    return [np.array(coll.objects[f"anchor_ball_{k}"].matrix_world.translation) for k in range(len(ANCHORS))]


def place_ball(coll, t):
    """Put moss_ball where the loop has it at video time t (previews only; reset before export)."""
    ball = coll.objects.get("moss_ball")
    if ball is None:
        return None
    if t is None:
        ball.location = (0.0, 0.0, 0.0)
        ball.hide_render = True
        return None
    p = loop_point(_anchor_points(coll), [a["phase"] for a in ANCHORS], ball_phase(t))
    ball.location = Vector(p)
    ball.hide_render = False
    return p


def reset_ball(coll):
    ball = coll.objects.get("moss_ball")
    if ball is not None:
        ball.location = (0.0, 0.0, 0.0)
        ball.hide_render = False


def _albedo(o, co, nrm):
    key = "wood" if o.name.startswith("wood_") else "moss"
    return np.broadcast_to(np.asarray(ALBEDO[key], float), co.shape)


class LitPreview:
    """Context: vertex-lit colours on every mesh of the set (key along key_branch with ray-cast
    shadows, sky fill, baked AO from COLOR_0.B), Workbench FLAT + attribute colours. Restores the
    COLOR_0 attribute as active / render colour on exit (the GLB is exported before previews)."""

    def __init__(self, scene, coll, percent):
        self.scene, self.coll, self.percent = scene, coll, percent

    def __enter__(self):
        reset_ball(self.coll)
        self.meshes = [o for o in self.coll.objects if o.type == "MESH"]
        branch = [o for o in self.meshes if o.name != "moss_ball"]
        bvh = C.build_bvh(branch)
        BL.bake_lit(branch, KEY_DIR_BL, bvh, _albedo)
        BL.bake_lit([o for o in self.meshes if o.name == "moss_ball"], KEY_DIR_BL, None, _albedo)
        BL.setup_lit_workbench(self.scene, percent=self.percent)
        return self

    def __exit__(self, *exc):
        BL.unbake_lit(self.meshes)
        reset_ball(self.coll)
        return False


def render_previews(scene, coll, cams=None, percent=50, suffix=""):
    out = []
    with LitPreview(scene, coll, percent):
        for name in (cams or list(CAMS.keys())):
            place_ball(coll, CAMS[name]["t"] if CAMS[name]["t"] < 37.0 else None)
            path = os.path.join(CAPTURES, f"branch_{name}{suffix}.png")
            C.render_camera(scene, coll.objects[name], path)
            out.append(path)
    return out


def render_masks(scene, coll, outdir, cams=None, percent=50, ball_t=None):
    """White silhouettes on a transparent film (alpha = coverage). ball_t: {cam: video time}."""
    sh = scene.display.shading
    prev = (scene.render.engine, scene.render.film_transparent, sh.light, sh.color_type, sh.show_cavity,
            scene.render.image_settings.color_mode, scene.render.resolution_percentage)
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.film_transparent = True
    scene.render.resolution_percentage = int(percent)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    sh.light = "FLAT"
    sh.color_type = "SINGLE"
    sh.single_color = (1.0, 1.0, 1.0)
    sh.show_cavity = False
    out = []
    try:
        for name in (cams or list(CAMS.keys())):
            bt = (ball_t or {}).get(name, CAMS[name]["t"] if CAMS[name]["t"] < 37.0 else None)
            place_ball(coll, bt)
            path = os.path.join(outdir, f"mask_{name}.png")
            C.render_camera(scene, coll.objects[name], path)
            out.append(path)
    finally:
        reset_ball(coll)
        (scene.render.engine, scene.render.film_transparent, sh.light, sh.color_type, sh.show_cavity,
         scene.render.image_settings.color_mode, scene.render.resolution_percentage) = prev
    return out


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


def _down2(a):
    h, w = a.shape[:2]
    return a[:h // 2 * 2, :w // 2 * 2].reshape(h // 2, 2, w // 2, 2, -1).mean((1, 3))


def _outline(mask, width=1):
    m = mask.astype(bool)
    e = np.zeros_like(m)
    for dy in range(-width, width + 1):
        for dx in range(-width, width + 1):
            if dx or dy:
                e |= np.roll(np.roll(m, dy, 0), dx, 1) != m
    return e & m


def reference_mask(rgb, top_cut=0.115):
    """Rough branch silhouette of frame 10 (light background): distance from a per-row background
    colour + saturation, majority-filtered. Iteration aid only (unreliable in the dark bottom)."""
    a = rgb[..., :3] * 255.0
    H, W = a.shape[:2]
    bg = np.zeros((H, 3))
    for y in range(H):
        v = y / H
        if v < 0.5:
            cols = np.r_[int(0.70 * W):int(0.86 * W), int(0.02 * W):int(0.12 * W)]
        elif v < 0.7:
            cols = np.r_[int(0.68 * W):int(0.86 * W), int(0.25 * W):int(0.38 * W)]
        else:
            cols = np.r_[int(0.56 * W):int(0.60 * W)]
        bg[y] = np.median(a[y, cols], 0)
    k = 15
    bgs = np.array([bg[max(0, y - k):y + k + 1].mean(0) for y in range(H)])
    d = np.sqrt(((a - bgs[:, None, :]) ** 2).sum(-1))
    sat = a.max(-1) - a.min(-1)
    m = ((d > 28) | (sat > 45)).astype(np.float32)
    acc = np.zeros_like(m)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            acc += np.roll(np.roll(m, dy, 0), dx, 1)
    m = acc > 12.5
    m[:int(top_cut * H)] = False
    return m


def edge_stats(rm, mm):
    """Silhouette edges (frame fractions) of reference vs base mesh along a few rows / columns."""
    h, w = rm.shape

    def row(mask, v, u0, u1):
        xs = np.nonzero(mask[int(v * h), int(u0 * w):int(u1 * w)])[0]
        return [round((xs[0] + int(u0 * w)) / w, 3), round((xs[-1] + int(u0 * w)) / w, 3)] if len(xs) else None

    def col(mask, u, v0, v1):
        ys = np.nonzero(mask[int(v0 * h):int(v1 * h), int(u * w)])[0]
        return [round((ys[0] + int(v0 * h)) / h, 3), round((ys[-1] + int(v0 * h)) / h, 3)] if len(ys) else None

    out = {}
    for v in (0.2, 0.3, 0.4, 0.5, 0.6, 0.65):
        out[f"J v={v} l/r"] = [row(rm, v, 0.33, 0.75), row(mm, v, 0.33, 0.75)]
    for u in (0.03, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5):
        out[f"J u={u} t/b"] = [col(rm, u, 0.55, 0.995), col(mm, u, 0.55, 0.995)]
    for u in (0.64, 0.7, 0.75, 0.8, 0.9, 0.97):
        out[f"F u={u} t/b"] = [col(rm, u, 0.66, 0.995), col(mm, u, 0.66, 0.995)]
    return out


def _grid(img, step=0.1, col=(0.3, 0.45, 1.0), major=(1.0, 0.25, 0.25)):
    h, w = img.shape[:2]
    for i in range(1, int(round(1 / step))):
        c = major if abs(i * step - 0.5) < 1e-6 else col
        x = int(round(i * step * w))
        y = int(round(i * step * h))
        img[:, x, :3] = img[:, x, :3] * 0.4 + np.array(c) * 0.6
        img[y, :, :3] = img[y, :, :3] * 0.4 + np.array(c) * 0.6


def _dot(img, u, v, r, col):
    h, w = img.shape[:2]
    cx, cy = u * w, v * h
    y0, y1 = int(max(0, cy - r - 1)), int(min(h, cy + r + 2))
    x0, x1 = int(max(0, cx - r - 1)), int(min(w, cx + r + 2))
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    sel = np.hypot(xx - cx, yy - cy) <= r
    img[y0:y1, x0:x1][sel, :3] = col


def _pad_to(a, h, w):
    out = np.zeros((h, w, a.shape[2]), np.float32)
    out[:a.shape[0], :a.shape[1]] = a[:h, :w]
    return out


def compare_sheets(scene, coll, percent=50):
    """branch_cmp_main / _p1: reference | lit preview | reference x0.55 + base silhouette (yellow) +
    reference silhouette (cyan, from frame 10) + 0.1 grid. branch_cmp_ball_path: loop (yellow),
    measured track (red, extrapolated pink), anchors (cyan) on the rest-pose preview and on frame 11.
    branch_cmp_broken_end / _fragment: crops of frame 10 next to the same crop of a full-size lit
    main preview. branch_cmp_channel: the bark channel - frame 11 (rest pose) next to the full-size
    p1 preview (upper trunk) and frame 10 next to the main preview (the bend). Needs the previews
    from render_previews()."""
    os.makedirs(TMP, exist_ok=True)
    masks = render_masks(scene, coll, TMP, ["cam_branch_main", "cam_branch_p1"], percent)
    m_main = _load_rgba(masks[0])[..., 3] > 0.5
    m_p1 = _load_rgba(masks[1])[..., 3] > 0.5
    ref10 = _down2(_load_rgba(os.path.join(REF, "frames/10_branch_entry.png")))
    rmask10 = reference_mask(ref10)
    out, stats = [], {}
    for key, cam, ref_rel, shift, msk in (("main", "cam_branch_main", "frames/10_branch_entry.png", 0.0, m_main),
                                         ("p1", "cam_branch_p1", "frames/11_branch.png", DRIFT_FH, m_p1)):
        ref = _down2(_load_rgba(os.path.join(REF, ref_rel)))
        pp = os.path.join(CAPTURES, f"branch_{cam}.png")
        prev = _load_rgba(pp) if os.path.exists(pp) else np.zeros_like(ref)
        h, w = ref.shape[:2]
        rm = np.roll(rmask10, -int(round(shift * h)), 0)
        if shift:
            rm[-int(round(shift * h)):] = False
        ov = ref[..., :3] * 0.55
        ov[_outline(rm)] = (0.2, 0.95, 1.0)
        ov[_outline(msk)] = (1.0, 0.92, 0.1)
        _grid(ov)
        pv = prev[..., :3] if prev.shape[:2] == ref.shape[:2] else np.zeros_like(ref[..., :3])
        path = os.path.join(CAPTURES, f"branch_cmp_{key}.png")
        _save_rgb(np.concatenate([ref[..., :3], pv, ov], 1), path)
        out.append(path)
        valid = np.ones_like(rm)
        valid[:int(0.12 * h)] = False
        valid[int(0.93 * h):] = False                  # reference mask unreliable in the dark bottom
        inter = (rm & msk & valid).sum()
        union = ((rm | msk) & valid).sum()
        stats[key] = dict(iou=round(float(inter) / max(float(union), 1.0), 4))
        if key == "main":
            stats["edges_main"] = edge_stats(rm, msk)
    # ball path on the rest pose
    p1 = _load_rgba(os.path.join(CAPTURES, "branch_cam_branch_p1.png"))
    ref11 = _down2(_load_rgba(os.path.join(REF, "frames/11_branch.png")))
    pts = _anchor_points(coll)
    phases = [a["phase"] for a in ANCHORS]
    loop = np.array([loop_point(pts, phases, ph) for ph in np.linspace(0, 1, 721)])
    lu, lv, lz = project("cam_branch_p1", loop)
    panels = []
    for base in (p1[..., :3].copy(), ref11[..., :3] * 0.7):
        for u, v, z in zip(lu, lv, lz):
            if z > 0.05:
                _dot(base, u, v, 1.2, (1.0, 0.9, 0.1))
        for ph, u, v in BALL_TRACK:
            _dot(base, u, v, 2.2, (1.0, 0.15, 0.15))
        for ph, u, v in BALL_TRACK_EXTRA:
            _dot(base, u, v, 2.2, (1.0, 0.55, 0.85))
        for pt in pts:
            u, v, z = project("cam_branch_p1", np.asarray(pt)[None, :])
            if z[0] > 0.05:
                _dot(base, u[0], v[0], 5.0, (0.15, 0.9, 1.0))
        _grid(base)
        panels.append(base)
    path = os.path.join(CAPTURES, "branch_cmp_ball_path.png")
    _save_rgb(np.concatenate(panels, 1), path)
    out.append(path)
    # full-size lit renders: broken end (details/06 = frame 10 [10, 455, 650, 950]), the fragment,
    # the channel (frame 11 = rest pose for the upper trunk, frame 10 for the bend)
    pfull = os.path.join(TMP, "prev_main_full.png")
    pfull1 = os.path.join(TMP, "prev_p1_full.png")
    with LitPreview(scene, coll, 100):
        place_ball(coll, 31.5)
        C.render_camera(scene, coll.objects["cam_branch_main"], pfull)
        place_ball(coll, 34.5)
        C.render_camera(scene, coll.objects["cam_branch_p1"], pfull1)
    full = _load_rgba(pfull)
    full1 = _load_rgba(pfull1)
    ref10f = _load_rgba(os.path.join(REF, "frames/10_branch_entry.png"))
    ref11f = _load_rgba(os.path.join(REF, "frames/11_branch.png"))
    for key, (x0, y0, x1, y1) in (("broken_end", (10, 455, 650, 950)), ("fragment", (840, 700, 1440, 1020))):
        path = os.path.join(CAPTURES, f"branch_cmp_{key}.png")
        _save_rgb(np.concatenate([ref10f[y0:y1, x0:x1, :3], full[y0:y1, x0:x1, :3]], 1), path)
        out.append(path)
    (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = (560, 60, 1010, 560), (400, 470, 850, 970)
    row_a = np.concatenate([ref11f[ay0:ay1, ax0:ax1, :3], full1[ay0:ay1, ax0:ax1, :3]], 1)
    row_b = np.concatenate([ref10f[by0:by1, bx0:bx1, :3], full[by0:by1, bx0:bx1, :3]], 1)
    gap = np.full((8, row_a.shape[1], 3), 0.1, np.float32)
    path = os.path.join(CAPTURES, "branch_cmp_channel.png")
    _save_rgb(np.concatenate([row_a, gap, row_b], 0), path)
    out.append(path)
    C.log("compare:", json.dumps(stats))
    return out, stats


def save_blend():
    bpy.ops.wm.save_as_mainfile(filepath=BLEND_PATH, check_existing=False, compress=True)
    C.log("saved", BLEND_PATH)


def validate_branch(rep, coll):
    """Extra checks: required nodes and anchors, ball mesh at the origin, loop clear of the branch
    meshes, hidden arc outside the frame of every branch pose."""
    errs, info = [], {}
    names = {n["name"] for n in rep["nodes"]}
    for k in range(len(ANCHORS)):
        if f"anchor_ball_{k}" not in names:
            errs.append(f"missing anchor_ball_{k}")
    for req in ("moss_ball", "wood_branch_j", "wood_branch_stumps", "moss_branch_j", "wood_branch_frag",
                "moss_branch_frag", "cam_branch_main", "cam_branch_p1", "cam_branch_out", "key_branch"):
        if req not in names:
            errs.append(f"missing node {req}")
    mb = rep["meshes"].get("moss_ball", {})
    if mb.get("transform"):
        errs.append("moss_ball node is not at the origin")
    pts = _anchor_points(coll)
    phases = [a["phase"] for a in ANCHORS]
    ph = np.linspace(0, 1, 2001)
    loop = np.array([loop_point(pts, phases, p) for p in ph])
    meshes = [o for o in coll.objects if o.type == "MESH" and o.name != "moss_ball"]
    bvh = C.build_bvh(meshes)
    near = [bvh.find_nearest(Vector(p)) for p in loop[::2]]
    k_min = int(np.argmin([n[3] for n in near]))
    dmin = near[k_min][3]
    if dmin < BALL_RADIUS + BALL_FUR + 0.02:
        errs.append(f"ball loop passes {dmin:.3f} m from the branch meshes")
    wood_bvh = C.build_bvh([o for o in meshes if o.name.startswith("wood_")])
    dwood = min(wood_bvh.find_nearest(Vector(p))[3] for p in loop[::2])
    # per mesh: the loop's closest approach (m), where on the loop (phase) and where on the mesh
    per_mesh = {}
    for o in meshes:
        ob = C.build_bvh([o])
        nn = [ob.find_nearest(Vector(p)) for p in loop[::2]]
        km = int(np.argmin([x[3] for x in nn]))
        u_, v_, _ = project("cam_branch_main", np.asarray(nn[km][0])[None, :])
        per_mesh[o.name] = dict(m=round(float(nn[km][3]), 4), phase=round(float(ph[::2][km]), 3),
                                at_px_main=[round(float(u_[0]) * C.FRAME_W), round(float(v_[0]) * C.FRAME_H)])
    k_hid = [k for k, a in enumerate(ANCHORS) if a.get("hidden")]
    ph0 = ANCHORS[k_hid[0]]["phase"] - 0.004
    ph1 = ANCHORS[k_hid[-1]]["phase"] + 0.03
    hidden = (ph > ph0) & (ph < ph1)
    vis_169 = {}
    for cam in CAMS:
        u, v, z = project(cam, loop)
        r_u = (BALL_RADIUS + BALL_FUR) / (2.0 * TH * np.maximum(z, 1e-3))
        r_v = (BALL_RADIUS + BALL_FUR) / (2.0 * TV * np.maximum(z, 1e-3))
        inside = (z > 0.05) & (u > -r_u) & (u < 1.0 + r_u) & (v > -r_v) & (v < 1.0 + r_v)
        if np.any(hidden & inside):
            errs.append(f"{cam}: hidden arc of the ball loop shows inside the 1440x1020 frame")
        wide = 0.5 * (16.0 / 9.0 / C.ASPECT - 1.0)       # extra width of a 16:9 screen (vertical fov kept)
        inside169 = (z > 0.05) & (u > -wide - r_u) & (u < 1.0 + wide + r_u) & (v > -r_v) & (v < 1.0 + r_v)
        vis_169[cam] = round(float(np.mean(inside169[hidden])), 3)
    u, v, z = project("cam_branch_p1", loop)
    info.update(loop_clearance_m=round(float(dmin), 3), loop_clearance_wood_m=round(float(dwood), 3),
                loop_clearance_exact_m=round(float(dmin), 4), loop_clearance_per_mesh=per_mesh,
                loop_closest_at_phase=round(float(ph[::2][k_min]), 3),
                loop_depth_m=[round(float(z[z > 0].min()), 3), round(float(z.max()), 3)],
                hidden_arc_phase=[round(ph0, 3), round(ph1, 3)], hidden_arc_share_visible_on_16x9=vis_169)
    return dict(errors=errs, **info)


def parse_args(argv):
    a = dict(export=False, validate=False, render=None, percent=50, save=False, ao=True, compare=False, out=None,
             captures=None)
    i = 0
    while i < len(argv):
        x = argv[i]
        if x == "--out":
            i += 1
            a["out"] = os.path.abspath(argv[i])
        elif x == "--captures":
            i += 1
            a["captures"] = os.path.abspath(argv[i])
        elif x == "--export":
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
    global GLB_PATH, CAPTURES
    a = parse_args(argv or [])
    if a["out"]:
        GLB_PATH = a["out"]
    if a["captures"]:
        CAPTURES = a["captures"]
        os.makedirs(CAPTURES, exist_ok=True)
    if bpy.app.background and a["save"] and os.path.exists(BLEND_PATH):
        bpy.ops.wm.open_mainfile(filepath=BLEND_PATH)   # keep the other scene sets
    scene, coll, info = build(use_ao=a["ao"])
    result = dict(tris=sum(C.tri_count(o) for o in coll.objects if o.type == "MESH"), ball_fit=info["fit"],
                  color_ranges=info["color_ranges"], part_tris=info["part_tris"],
                  mesh_tris={o.name: C.tri_count(o) for o in sorted(coll.objects, key=lambda o: o.name)
                             if o.type == "MESH"})
    if a["export"]:
        reset_ball(coll)
        C.export_glb(scene, GLB_PATH)
        result["glb"] = GLB_PATH
        result["glb_bytes"] = os.path.getsize(GLB_PATH)
    if a["validate"]:
        rep = C.validate_glb(GLB_PATH)
        extra = validate_branch(rep, coll)
        rep["branch"] = extra
        result["validate"] = dict(errors=rep["errors"] + extra["errors"], warnings=rep["warnings"],
                                  totals=rep["totals"], branch=extra)
        os.makedirs(CAPTURES, exist_ok=True)
        with open(os.path.join(CAPTURES, "branch_validate.json"), "w") as f:
            json.dump(rep, f, indent=1, default=str)
    if a["render"] is not None or a["compare"]:
        result["renders"] = render_previews(scene, coll, a["render"] or None, a["percent"])
    if a["compare"]:
        result["compare"], result["compare_stats"] = compare_sheets(scene, coll, a["percent"])
    if a["save"]:
        reset_ball(coll)
        save_blend()
    C.log("result", json.dumps(result, default=str))
    return result


if __name__ == "__main__":
    _argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    main(_argv)
