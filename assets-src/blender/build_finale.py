"""
build_finale.py - the 'finale' scene set (S10, video 52.8-59 s): a mini landscape of fluffy moss hills, a
standing stone on the central hill with the Silva seed emblem cut through it, a low near mound and dark far
forms, exported to public/nature/models/finale.glb (contract: CLAUDE.md "Asset contract").

Run headless (from the repo root):
    "/c/Program Files/Blender Foundation/Blender 5.1/blender.exe" --background --factory-startup \
        --python assets-src/blender/build_finale.py -- --export --validate [--render] [--compare] [--save]

Run inside a live Blender session (e.g. through MCP):
    import sys, importlib; sys.path.insert(0, r"<repo>/assets-src/blender")
    import build_finale; importlib.reload(build_finale); build_finale.main(["--export", "--validate"])

Options:
    --export            write public/nature/models/finale.glb
    --validate          re-read / re-import the GLB, check it against the contract + finale checks
                        (report: docs/captures/blender/finale_validate.json)
    --render [cams]     lit EEVEE previews -> docs/captures/blender/finale_<cam>.png (all cams if none given)
    --compare           comparison sheets -> docs/captures/blender/finale_cmp_<what>.png (implies --render)
    --percent N         preview size in % of 1440x1020 (default 50)
    --save              save nature.blend (headless: opens the existing file first so the other scene sets
                        are kept; only the 'finale' scene / collection is rebuilt)
    --no-ao             skip the ray-traced AO (B channel = cavity only; fast iteration)
    --glb PATH          export / validate PATH instead of public/nature/models/finale.glb (iterations,
                        determinism checks)

Everything is deterministic (seed 134). World units: metres, Blender Z-up, cameras look along +Y.

Measured from the reference (frames 18 / 19 / 20 and clip motion/11, stone silhouette tracked by hand on
2x crops, 1440x1020 px):
  * frame 19 (54.3 s, cam_finale_main): stone top at v 0.268, widest 0.1715 W (x 590-837 px) at v 0.44,
    visible bottom (moss line) at v ~0.545; moss crest line per column in MOSS_CREST.
  * frame 18 (53.2 s, cam_finale_in): stone 0.125 W wide, top at v 0.431 -> camera ~1.4x farther, higher.
  * frame 20 (57.5 s, cam_finale_out): stone 0.181 W wide, top at v 0.183 -> a bit closer and lower.
  * clip 11: 53.2 -> 54.0 s the stone grows 0.119 -> 0.173 W (dolly in, decelerating); 54.0 -> 55.1 s it
    only rises (0.336 -> 0.201) while the footer panel comes up; 57.5 s = frame 20. The far forms keep
    their screen position through the dolly (they read as a distant backdrop): they sit 7-10 m behind the
    stone here so the parallax stays small.
  * stone silhouette (frame 19, 2.45 mm per px at the stone): ~0.60 m wide at 35 % of its height, 0.38 m
    across the top, 0.69 m visible above the moss -> modelled from STONE_OUTLINE (the brief's 0.45 m width
    reads too narrow against frame 19; the picture wins).
  * moss (frame 19 / detail 09, revision 3): the hills are a pile of round mounds 30-60 cm across (bigger
    in the near rows, smaller towards the stone), 8-20 cm high, with dark gaps between them and 10-15 cm
    bumps on them; the crest outline is a row of mound tops and notches (MOSS_CREST_FINE: traced on frame
    19 every few px). Built as: the fitted base field (plateau, hills, stone seat) gets its narrow pits
    filled (rolling ball) and is clamped under the crest -> envelope E (1.2 cm voxels); mounds are seeded
    first along E's silhouette in cam_finale_main (the crest becomes a row of dome tops), then Poisson-
    thrown by depth rows (PILE_ROWS); the pile is carved into E from above as a height map
    (H = zE - w * smin(two shallowest dome carvings, gap floor)), so no overhang or cavity can form and
    nothing rises above E; a second rolling-ball closing (FILL2_BALL) removes slots, then the bumps
    (outwards only, warped elliptic footprints), a last clamp lets the tops rise at most LUMP_UP above the
    crest, and a grey opening of the field (POCKET_R) fills the 1-2 cm pockets left between the bumps.
    COLOR_0.B carries the gaps (about 0.3-0.5 deep in the creases), R >= 0.6 on every seen vertex.
  * far forms (revision 3, brightened frames 18-20): per side a few leaning tree trunks (the inner pair
    frames the stone in a V) merged with mossy rock lumps, 7-10 m behind the stone (FAR_TRUNKS / FAR_ROCKS).
  * portrait screens (revision 3, QA finding R4): on narrow screens the engine keeps >= 72 % of the
    reference width, so the vertical fov grows (69.5 deg at 375x812) and the cameras see below the island.
    The near mound continues as a coarse moss bank (a heightfield merged into moss_finale_near__fg) from
    behind cam_finale_in to under the island: BANK_MARGIN under the lowest 35 deg bottom-edge plane of the
    engine's camera track (finale/track.ts replayed in engine_track), never higher than BANK_TOP, and a floor
    at BANK_UNDER just under the island's cut bottom, so rays below the lip end on moss. The 1440x1020
    frames do not change except where they showed background before (a sliver under the lip, 54.0-55.6 s);
    moss_finale_hills, the stone and the near mound's own triangles and colours are untouched. The bank is
    coloured like the mound (R ~0.2, G 0.4, B <= 0.18 with a 1.5 m AO reach); R = 0 under the island (it
    lies inside the engine's near-mound scatter frusta and would take plants from the mound's budget) and
    the island's key shadow is baked into B there.
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
from mathutils.bvhtree import BVHTree

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import common as C  # noqa: E402
import stonelib as SL  # noqa: E402

importlib.reload(C)
importlib.reload(SL)

ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
GLB_PATH = os.path.join(ROOT, "public", "nature", "models", "finale.glb")
BLEND_PATH = os.path.join(HERE, "nature.blend")
CAPTURES = os.path.join(ROOT, "docs", "captures", "blender")
REF = os.path.join(ROOT, "docs", "nature-webgl-reference")      # comparison sheets only (never shipped)
SVG = os.path.join(ROOT, "assets-src", "emblem", "seed.svg")
TMP = os.path.join(tempfile.gettempdir(), "silva_finale_build")
SCENE_NAME = "finale"
COLL_NAME = "finale"
SEED = 134

TV = math.tan(math.radians(C.FOV_V_DEG) / 2.0)
TH = TV * C.ASPECT

BG_HEX = "#191F22"          # cold near-black background (measured, frame 19)

# ------------------------------------------------------------------------------------------
# the standing stone
# ------------------------------------------------------------------------------------------
Z_B = 0.30                  # moss crest level under the stone (m)
RING_H = 0.05               # moss contact ring: the stone is visible from Z_B + RING_H
SINK = 0.07                 # stone bottom sits SINK below the crest
STONE_X = 0.0
HALF_T = 0.125              # half thickness (0.25 m)
VIS0 = Z_B + RING_H         # z of the visible bottom (z_rel = 0)

# outline in the stone plane: (x m, z_rel m above the visible bottom), CCW, measured on frame 19
STONE_OUTLINE = np.array([
    (-0.292, -0.14), (-0.286, 0.0), (-0.296, 0.142), (-0.305, 0.203), (-0.298, 0.265), (-0.278, 0.326),
    (-0.262, 0.387), (-0.243, 0.448), (-0.230, 0.510), (-0.214, 0.571), (-0.190, 0.628), (-0.160, 0.664),
    (-0.118, 0.688), (-0.085, 0.696), (-0.04, 0.692), (0.03, 0.688), (0.10, 0.682), (0.150, 0.672),
    (0.180, 0.652), (0.200, 0.625), (0.215, 0.571), (0.242, 0.510), (0.266, 0.448), (0.285, 0.387),
    (0.300, 0.326), (0.298, 0.265), (0.290, 0.203), (0.281, 0.142), (0.285, 0.0), (0.292, -0.14),
])
STONE_TOP_REL = 0.696
STONE_WIDE_REL = 0.265      # z_rel of the widest row (frame 19: v 0.44)

EMBLEM_H = 0.42             # ~56 % of the stone height (brief: ~55 %)
EMBLEM_C = (0.018, 0.335)   # (x, z_rel) emblem centre on the front face
EMBLEM_ROT = 1.5            # deg clockwise
EMBLEM_FLAT = 0.05          # stone faces are flattened within this distance (m) around the emblem
# the split on the stone's front-left (x, z_rel), bottom -> top; the slab left of it sits 13 mm back
CRACK_LINE = [(-0.212, -0.15), (-0.205, 0.02), (-0.188, 0.16), (-0.198, 0.30), (-0.222, 0.42), (-0.236, 0.50),
              (-0.240, 0.56)]

# ------------------------------------------------------------------------------------------
# cameras: fixed pitch, positions solved so the stone hits the measured screen box
#   in (53.2 s, frame 18) -> main (54.3 s, frame 19) -> out (57.5 s, frame 20)
# ------------------------------------------------------------------------------------------
PITCH = -8.0
CAM_TARGETS = {
    # frame 18: stone box; flatter pitch so the island's front lip (lit moss -> dark drop, v 0.765 in the
    # frame) is not pushed too far down. The reference reaches main by a dolly plus a page scroll (the canvas
    # slides up 0.19 / 0.30 of the frame in frames 19 / 20); main and out emulate that scroll with a lower
    # camera, which a single pose pair can only approximate (the lip lands at v ~0.80 here, see the log).
    # A target may also carry lip_v (then the pitch is solved as well).
    "cam_finale_in": dict(top_v=0.431, width=0.125, t=53.2, pitch=-3.0),
    "cam_finale_main": dict(top_v=0.268, width=0.1715, t=54.3),
    "cam_finale_out": dict(top_v=0.183, width=0.181, t=57.5),
}
# measured progress in -> main (clip 11): stone width share of the way, video s
CAM_DOLLY = [(53.2, 0.0), (53.4, 0.27), (53.6, 0.55), (53.8, 0.72), (54.0, 0.92), (54.3, 1.0)]

# key light: upper right, in front (direction the light travels, Blender world)
KEYS = {"key_finale": dict(dir=(-0.46, 0.50, -0.73))}

# ------------------------------------------------------------------------------------------
# moss: reference crest line seen from cam_finale_main, (u, v)
#   MOSS_CREST: revisions 1-2 (frame 19, green mask top per column + by eye where flowers / the panel
#   interfere, 33 points); kept for comparison and the far forms' framing
#   MOSS_CREST_FINE: revision 3 (frame 19, first run of 6 moss pixels per column, white flowers and the
#   sprig at x 172-230 px bridged, narrow spikes opened (21 px), 9 px mean, a point every 8 px; the dark
#   left flank x < 120 px keeps the by-eye line); the clamp line of the hills
# ------------------------------------------------------------------------------------------
MOSS_CREST = [(-0.25, 0.70), (0.0, 0.66), (0.04, 0.648), (0.083, 0.625), (0.125, 0.598), (0.167, 0.594),
              (0.208, 0.572), (0.25, 0.552), (0.275, 0.53), (0.292, 0.511), (0.31, 0.53), (0.333, 0.553),
              (0.375, 0.570), (0.40, 0.562), (0.417, 0.555), (0.458, 0.553), (0.50, 0.549),
              (0.542, 0.53), (0.583, 0.523), (0.605, 0.56), (0.625, 0.606), (0.645, 0.575),
              (0.667, 0.555), (0.69, 0.566), (0.708, 0.576), (0.75, 0.612), (0.792, 0.629),
              (0.833, 0.616), (0.875, 0.645), (0.917, 0.653), (0.958, 0.649), (1.0, 0.655), (1.25, 0.68)]
MOSS_CREST_FINE = [
    (-0.25, 0.7), (0.0003, 0.6597), (0.0059, 0.6582), (0.0115, 0.6566), (0.017, 0.6549), (0.0226, 0.6532), (0.0281, 0.6516),
    (0.0337, 0.6499), (0.0392, 0.6481), (0.0448, 0.6454), (0.0503, 0.6425), (0.0559, 0.6395), (0.0615, 0.6365), (0.067, 0.6336),
    (0.0726, 0.6306), (0.0781, 0.6277), (0.0837, 0.6233), (0.0892, 0.6164), (0.0948, 0.609), (0.1003, 0.6001), (0.1059, 0.597),
    (0.1115, 0.597), (0.117, 0.5967), (0.1226, 0.5955), (0.1281, 0.5942), (0.1337, 0.5929), (0.1392, 0.5916), (0.1448, 0.5903),
    (0.1503, 0.589), (0.1559, 0.5877), (0.1615, 0.59), (0.167, 0.5953), (0.1726, 0.5979), (0.1781, 0.5972), (0.1837, 0.59),
    (0.1892, 0.5818), (0.1948, 0.5766), (0.2003, 0.5765), (0.2059, 0.5745), (0.2115, 0.5695), (0.217, 0.5641), (0.2226, 0.561),
    (0.2281, 0.5578), (0.2337, 0.5557), (0.2392, 0.5535), (0.2448, 0.5521), (0.2503, 0.5503), (0.2559, 0.5441), (0.2615, 0.5346),
    (0.267, 0.5295), (0.2726, 0.523), (0.2781, 0.5164), (0.2837, 0.5137), (0.2892, 0.5118), (0.2948, 0.5116), (0.3003, 0.5116),
    (0.3059, 0.5127), (0.3115, 0.5168), (0.317, 0.5201), (0.3226, 0.5274), (0.3281, 0.5353), (0.3337, 0.5431), (0.3392, 0.551),
    (0.3448, 0.5587), (0.3503, 0.5615), (0.3559, 0.5612), (0.3615, 0.5612), (0.367, 0.5617), (0.3726, 0.5627), (0.3781, 0.5637),
    (0.3837, 0.5648), (0.3892, 0.5658), (0.3948, 0.5668), (0.4003, 0.5679), (0.4059, 0.5665), (0.4115, 0.5646), (0.417, 0.5635),
    (0.4226, 0.5624), (0.4281, 0.5613), (0.4337, 0.5602), (0.4392, 0.5589), (0.4448, 0.557), (0.4503, 0.5551), (0.4559, 0.554),
    (0.4615, 0.5524), (0.467, 0.5504), (0.4726, 0.5502), (0.4781, 0.5502), (0.4837, 0.5508), (0.4892, 0.551), (0.4948, 0.5504),
    (0.5003, 0.5502), (0.5059, 0.5502), (0.5115, 0.5507), (0.517, 0.5473), (0.5226, 0.5425), (0.5281, 0.535), (0.5337, 0.5303),
    (0.5392, 0.5277), (0.5448, 0.5255), (0.5503, 0.5255), (0.5559, 0.5245), (0.5615, 0.5197), (0.567, 0.5194), (0.5726, 0.5195),
    (0.5781, 0.5213), (0.5837, 0.5242), (0.5892, 0.5362), (0.5948, 0.5532), (0.6003, 0.57), (0.6059, 0.58), (0.6115, 0.5806),
    (0.617, 0.5808), (0.6226, 0.5807), (0.6281, 0.5802), (0.6337, 0.5796), (0.6392, 0.579), (0.6448, 0.5773), (0.6503, 0.5678),
    (0.6559, 0.5614), (0.6615, 0.5584), (0.667, 0.5554), (0.6726, 0.5551), (0.6781, 0.5551), (0.6837, 0.5561), (0.6892, 0.5598),
    (0.6948, 0.5644), (0.7003, 0.569), (0.7059, 0.5736), (0.7115, 0.5791), (0.717, 0.5871), (0.7226, 0.5916), (0.7281, 0.5973),
    (0.7337, 0.601), (0.7392, 0.6043), (0.7448, 0.6083), (0.7503, 0.6116), (0.7559, 0.615), (0.7615, 0.6193), (0.767, 0.6237),
    (0.7726, 0.6278), (0.7781, 0.63), (0.7837, 0.6316), (0.7892, 0.6323), (0.7948, 0.6261), (0.8003, 0.6214), (0.8059, 0.6165),
    (0.8115, 0.6137), (0.817, 0.6137), (0.8226, 0.6137), (0.8281, 0.6137), (0.8337, 0.6155), (0.8392, 0.6205), (0.8448, 0.6256),
    (0.8503, 0.6305), (0.8559, 0.6342), (0.8615, 0.6366), (0.867, 0.6422), (0.8726, 0.6444), (0.8781, 0.6488), (0.8837, 0.6528),
    (0.8892, 0.656), (0.8948, 0.6592), (0.9003, 0.6592), (0.9059, 0.6583), (0.9115, 0.6583), (0.917, 0.6577), (0.9226, 0.6554),
    (0.9281, 0.6529), (0.9337, 0.6504), (0.9392, 0.649), (0.9448, 0.649), (0.9503, 0.649), (0.9559, 0.649), (0.9615, 0.649),
    (0.967, 0.6492), (0.9726, 0.6494), (0.9781, 0.6494), (0.9837, 0.6495), (0.9892, 0.65), (0.9948, 0.65), (0.9997, 0.65),
    (1.25, 0.68),
]
CREST_DROP = 0.006          # base mesh sits this far (frame heights) under the reference crest (fluff)

# moss (revision 3, frame 19 / detail 09): a pile of round mounds 26-55 cm across and 8-20 cm high with
# dark gaps between them, lit tops, shadowed sides; 10-15 cm bumps and 3-8 cm knobs on them
HILL_VOX = 0.016            # envelope field E: plateau, fitted hills, pit fill, crest clamp (the first fit)
FINE_VOX = 0.012            # pile field (E resampled, mounds carved into it, bumps + knobs)
FILL_BALL = 0.2             # rolling ball filling narrow pits / gaps between the hills of E (m)
FILL_MIN = 0.01             # fills shallower than this are left alone (m)
PILE_ROWS = ((-1.2, 0.26), (-0.8, 0.25), (-0.45, 0.22), (-0.1, 0.2), (0.4, 0.21))  # (plan y m, mound a m)
PILE_STONE = (0.55, 0.40, 0.72)  # stone ellipse half axes (m) and the mound size factor at the stone
PILE_JITTER = 0.15          # footprint semi-axis a = size field x U(1 - j, 1 + j)
PILE_ASPECT = (0.72, 1.0)   # footprint b / a, random direction
PILE_H = (0.68, 0.88)       # visible dome height / a ...
PILE_HCLIP = (0.08, 0.20)   # ... clipped to 8-20 cm
PILE_EPS = 0.18             # dome profile: elliptic arc, its foot rounded by this (0 = vertical at the rim)
PILE_SPACING = 0.85         # mound centres at least spacing * (a_i + a_j) apart (neighbours overlap a little)
PILE_SIL = 0.6              # along the crest (silhouette of E in cam_finale_main) mounds are seeded first,
                            # spacing * (a_i + a_j) apart: the crest is a row of dome tops with shallow notches
PILE_K = 0.03               # smooth union of neighbouring mounds: rounded crease bottoms (m)
PILE_CORE_K = 0.045         # smooth union with the gap floor (E inset by the local dome height)
PILE_HOLD = (1.05, 1.45)    # no carving inside the stone foot ellipse (STONE_FOOT_E x first), full from x second
PILE_EDGE = (0.1, 0.35)     # no carving within the first value (m) inside the island outline, full from the
                            # second (the rounded, undercut lip stays whole)
PILE_RISE = 0.0             # E is clamped this much (frame heights) under crest + CREST_DROP: the bumps
                            # (outward only) bring the crest line back
FILL2_BALL = 0.03           # rolling-ball closing of the pile: no slot or pit narrower than 6 cm (a bigger
                            # ball would fill the creases between the mounds)
FILL2_MIN = 0.005
CUSHION_R = (0.05, 0.075)   # second scale: bumps 10-15 cm across on the mounds (dart-throwing radius r)
CUSHION_SPACING = 0.8       # centres at least spacing * (r_i + r_j) apart
CUSHION_FOOT = 1.0          # bump footprint radius / r
CUSHION_H = (0.18, 0.34)    # bump height / r, times the patch factor (1-2.5 cm)
CUSHION_PATCH = (1.3, 0.55)  # patch noise frequency (1/m) and lowest height factor: calm and lumpy areas
CUSHION_ASPECT = (0.7, 1.0)  # bump footprint aspect (area kept), random direction
CUSHION_WARP = (9.0, 0.02)  # domain warp of the bump footprints (1/m, m)
CUSHION_EPS = 0.3           # bump profile: rounding where it meets the mound (0 = vertical edge)
CUSHION_K = 0.008           # smooth max between neighbouring bumps (m)
CUSHION_GAP = 0.35          # bump height factor down in the gaps (1 on the mound tops)
POCKET_R = 1                # last cleanup: grey opening of the field with a (2 r + 1)^3 voxel cube - pockets and
                            # slots narrower than ~2 r voxels (2.4 cm) between the bumps fill, nothing else moves
LUMP_UP = (0.012, 0.003)    # bumps may rise this much (frame heights) above the fitted crest, no more
                            # (second value: at the stone, u 0.37-0.63, where the crest is the moss line)
STONE_FOOT_E = (0.36, 0.20)  # stone foot ellipse (half axes x, y, m): low bumps there
KNOBS = ((11.0, 0.004, (4.1, 7.7, 1.3)), (23.0, 0.002, (9.2, 2.6, 5.9)))   # billow knobs (1/m, m, offset)
HILLS_TRIS = 120000         # moss_finale_hills budget (contract: <= 120k per mesh)
HIDDEN_KEEP = 0.25          # share of the full-res triangles kept where no camera pose sees the moss
DROP_FRONT = 0.3            # COLOR_0: vertices with drop above this are the undercut front face (stay dark)

# the island: a moss plateau (plan outline x/y in metres) with a rounded, sagging top edge and an undercut
# front face that falls away into the dark. Front lip: v ~0.83 in cam_finale_main (frame 19: moss to the
# canvas cut at v 0.81 on the right, ~0.75 where the island's left front corner recedes), v 0.765 in
# cam_finale_in (frame 18)
ISLAND_PLAN = np.array([
    (-2.95, 0.1), (-2.5, -0.3), (-1.95, -0.6), (-1.45, -0.80), (-0.95, -0.95), (-0.5, -1.06), (0.0, -1.12),
    (0.6, -1.15), (1.2, -1.19), (1.75, -1.16), (2.25, -0.95), (2.65, -0.6), (3.0, -0.1),
    (3.05, 0.55), (2.5, 1.15), (1.2, 1.35), (-0.3, 1.4), (-1.8, 1.25), (-2.7, 0.75)])
LIP_R = 0.13                # rounding radius of the island's top edge
UNDERCUT = 0.70             # the front face recedes this far over its first 0.5 m


def floor_top(y):
    """Height of the island floor between the hills (rises gently towards the stone)."""
    return 0.03 + 0.09 * C.smoothstep(-1.2, 0.1, y)


# where the lit top turns into the dark front face at x = 0 (50 deg round the lip)
LIP_PT = np.array([0.0, -1.12 + 0.03, float(floor_top(-1.09)) - 0.046])

# far forms (revision 3, brightened frames 18-20): distinct dark silhouettes on both sides - leaning tree
# trunks (the inner pair forms the V around the stone) and mossy rock lumps, 7-10 m behind the stone (small
# parallax through the dolly). Screen points in cam_finale_main (u, v) and depth behind the stone (m).
#   trunks: (top u, v, depth), (bottom u, v, depth), radius top, radius bottom (m)
FAR_TRUNKS = {
    "left": [((0.07, -0.32, 8.6), (0.20, 0.64, 8.0), 0.26, 0.37),     # outer trunk (frame 19 x ~140-270 px)
             ((0.18, -0.32, 7.7), (0.435, 0.66, 7.4), 0.25, 0.36),    # inner trunk: left arm of the V
             ((-0.07, -0.30, 9.6), (0.02, 0.72, 9.2), 0.42, 0.58)],   # at the frame edge, behind
    "right": [((0.866, -0.32, 7.7), (0.559, 0.66, 7.4), 0.25, 0.36),  # inner trunk: right arm of the V
              ((0.875, -0.32, 9.6), (0.835, 0.40, 9.3), 0.2, 0.27)],  # thin trunk behind the rock
}
#   rock lumps: (u, v, depth), half axes (m) - the right one the tall mossy rock at the frame edge (frame 19)
FAR_ROCKS = {
    "left": [((0.08, 0.66, 8.8), (1.5, 1.2, 1.1)), ((-0.04, 0.5, 9.4), (1.2, 1.1, 1.6))],
    "right": [((0.925, 0.40, 8.8), (1.45, 1.3, 2.5)), ((1.04, 0.12, 9.3), (1.3, 1.2, 1.9)),
              ((0.86, 0.66, 8.6), (1.1, 1.0, 0.9))],
}
FAR_TRIS = 9000             # per side (6-12k)

def cam_q(pitch=PITCH):
    return C.cam_quat(pitch, 0.0, 0.0)


CAMS = {}           # filled by solve_cameras(): name -> dict(loc, pitch, focus, t)


def project(cam, P):
    c = CAMS[cam]
    return C.project_to_screen(c["loc"], cam_q(c["pitch"]), np.atleast_2d(P))


def S(cam, u, v, d):
    c = CAMS[cam]
    return C.screen_to_world(c["loc"], cam_q(c["pitch"]), u, v, d)


def stone_world(x, z_rel, y=0.0):
    return np.array([STONE_X + x, y, VIS0 + z_rel])


def _stone_box_screen(loc, pitch):
    q = cam_q(pitch)
    top = stone_world(-0.085, STONE_TOP_REL, -0.02)
    left = stone_world(-0.300, STONE_WIDE_REL, -0.07)
    right = stone_world(0.298, STONE_WIDE_REL, -0.07)
    u, v, z = C.project_to_screen(loc, q, np.array([top, left, right]))
    return v[0], u[2] - u[1]


def _cam_residual(y, z, pitch, tg):
    loc = np.array([STONE_X, y, z])
    v, w = _stone_box_screen(loc, pitch)
    r = [v - tg["top_v"], w - tg["width"]]
    if "lip_v" in tg:
        _u, vl, _z = C.project_to_screen(loc, cam_q(pitch), LIP_PT[None])
        r.append(vl[0] - tg["lip_v"])
    return np.array(r)


def solve_cameras():
    """Camera locations (x = stone x) so the stone top v and widest width match each reference frame
    (and, for the entry pose, the island lip: pitch solved as well)."""
    for name, tg in CAM_TARGETS.items():
        x = np.array([-4.0, 1.0, tg.get("pitch", PITCH)])
        free = 3 if "lip_v" in tg else 2
        for _ in range(80):
            r = _cam_residual(x[0], x[1], x[2], tg)
            if np.abs(r).max() < 1e-8:
                break
            J = np.zeros((len(r), free))
            for k in range(free):
                h = 1e-4 if k < 2 else 1e-3
                x2 = x.copy()
                x2[k] += h
                J[:, k] = (_cam_residual(x2[0], x2[1], x2[2], tg) - r) / h
            step = np.linalg.lstsq(J, -r, rcond=None)[0]
            x[:free] += np.clip(step, -0.5, 0.5)
        loc = np.array([STONE_X, x[0], x[1]])
        pitch = float(x[2])
        right, up, fwd = C.cam_basis(cam_q(pitch))
        focus = float((stone_world(0.0, 0.33, -HALF_T) - loc) @ fwd)
        CAMS[name] = dict(loc=tuple(float(v) for v in loc), pitch=round(pitch, 4), focus=focus, t=tg["t"],
                          residual=float(np.abs(_cam_residual(x[0], x[1], x[2], tg)).max()),
                          lip_v=float(C.project_to_screen(loc, cam_q(pitch), LIP_PT[None])[1][0]))
    return CAMS


# ------------------------------------------------------------------------------------------
# stone
# ------------------------------------------------------------------------------------------

def _ramp(x, w):
    """C1 ramp: 0 for x <= 0, x^2 / 2w up to w, then x - w / 2."""
    x = np.asarray(x, float)
    return np.where(x <= 0.0, 0.0, np.where(x < w, x * x / (2.0 * w), x - 0.5 * w))


def _faces_natural(x, zr):
    """Front / back face y of the stone (before the emblem field is flattened): a boulder, thickest in the
    middle, the left third turning away from the viewer (broad side facet, in shade under the key from the
    right, frame 19 / detail 10), a narrower lit bevel on the right, receding towards top and foot."""
    # (the band around the emblem stays planar by construction: the facets start outside its field)
    t = HALF_T * (1.0 - 0.18 * (x / 0.3) ** 2)
    t = t - 0.25 * _ramp(zr - 0.60, 0.05) - 0.2 * _ramp(0.05 - zr, 0.05)
    yc = 0.006 + 0.018 * zr
    yf = yc - t + 0.55 * _ramp(-0.19 - x, 0.04) + 0.30 * _ramp(x - 0.235, 0.03)
    yb = yc + t - 0.25 * _ramp(-0.19 - x, 0.04) - 0.25 * _ramp(x - 0.235, 0.03)
    return yf, yb


Y_FRONT, Y_BACK = (float(v) for v in _faces_natural(np.array(EMBLEM_C[0]), np.array(EMBLEM_C[1])))


def emblem_frame():
    o = stone_world(EMBLEM_C[0], EMBLEM_C[1], Y_FRONT)
    return SL.EmblemFrame(o, (0.0, -1.0, 0.0), (0.0, 0.0, 1.0), EMBLEM_H, rot_deg=EMBLEM_ROT)


def emblem_dist_m(P, frame):
    """Distance (m, + outside) from the emblem outline, measured in the stone's front plane."""
    q, _h = frame.local(P)
    return SL.seed_sdf2(q) * frame.s


def build_stone(coll, mat, N3, frame):
    vox = 0.005
    lo = np.array([STONE_X - 0.36, -0.22, Z_B - SINK - 0.06])
    hi = np.array([STONE_X + 0.36, 0.22, VIS0 + STONE_TOP_REL + 0.05])
    sdf = C.SDF(lo, hi, vox)
    rr = 0.048                                    # edge rounding radius
    outline = STONE_OUTLINE.copy()
    crack = np.array(CRACK_LINE)

    def crack_x(zr):
        return np.interp(zr, crack[:, 1], crack[:, 0])

    def field(P):
        x = P[:, 0] - STONE_X
        zr = P[:, 2] - VIS0
        d2 = C.poly_sdf(np.stack([x, zr], 1), outline)
        yf, yb = _faces_natural(x, zr)
        # the split-off slab left of the crack sits a little back (frame 19 / detail 10: stepped crack)
        chunk = C.smoothstep(0.004, -0.004, x - crack_x(zr)) * (1.0 - C.smoothstep(0.40, 0.55, zr))
        yf = yf + 0.02 * chunk
        # emblem field: both faces exactly planar (clean chamfered cut-through)
        w = 1.0 - C.smoothstep(EMBLEM_FLAT, EMBLEM_FLAT + 0.06, emblem_dist_m(P, frame))
        yf = yf + w * (Y_FRONT - yf)
        yb = yb + w * (Y_BACK - yb)
        dy = np.maximum(yf - P[:, 1], P[:, 1] - yb)
        wx, wy = d2 + rr, dy + rr
        return np.minimum(np.maximum(wx, wy), 0.0) + np.hypot(np.maximum(wx, 0), np.maximum(wy, 0)) - rr

    sl, P = sdf._box(lo, hi)
    sdf.d[sl] = field(P.reshape(-1, 3)).reshape(P.shape[:-1]).astype(np.float32)
    # bottom: flat cut well inside the hill
    sdf.clip_plane((0, 0, Z_B - SINK), (0, 0, -1))

    rng = np.random.default_rng(SEED + 801)

    def away_from_emblem(P, margin=EMBLEM_FLAT + 0.03):
        return C.smoothstep(margin, margin + 0.04, emblem_dist_m(P, frame))

    # broad shallow facets on the convex top and shoulders (planar cuts, weathered boulder)
    def top_weight(P):
        n = SL.sdf_normals(sdf, P)
        return np.clip(n[:, 2], 0.0, 1.0) ** 2 * away_from_emblem(P) * (P[:, 2] > VIS0 + 0.40)

    pts, nrm = SL.surface_samples(sdf, 7, rng, weight=top_weight)
    for p, n in zip(pts, nrm):
        tilt = C.normalize(n + 0.2 * C.normalize(rng.normal(size=3)))
        SL.chip(sdf, p, tilt, depth=rng.uniform(0.005, 0.012), radius=rng.uniform(0.10, 0.18), k=0.008)
    # flake scars along the rims and sides (chipped edges), never on the flattened emblem field
    def rim_weight(P):
        n = SL.sdf_normals(sdf, P)
        side = 1.0 - np.abs(n[:, 1])
        return side ** 2 * away_from_emblem(P) * (P[:, 2] > Z_B + 0.03)

    pts, nrm = SL.surface_samples(sdf, 18, rng, weight=rim_weight)
    for p, n in zip(pts, nrm):
        tilt = C.normalize(n + 0.4 * C.normalize(rng.normal(size=3)))
        SL.chip(sdf, p, tilt, depth=rng.uniform(0.006, 0.015), radius=rng.uniform(0.05, 0.10), k=0.005)
    # the crack down the front-left (frame 19: a split chunk on the stone's left side) + one on the side
    crev = [stone_world(x, zr, -HALF_T - 0.002) for (x, zr) in CRACK_LINE[1:]]
    SL.carve_tube(sdf, crev, [0.010, 0.014, 0.016, 0.014, 0.011, 0.006], k=0.004)
    side_crack = [stone_world(-0.29, 0.12, -0.05), stone_world(-0.27, 0.30, -0.06), stone_world(-0.255, 0.42, -0.05)]
    SL.carve_tube(sdf, side_crack, [0.007, 0.009, 0.005], k=0.003)
    # weathering relief, faded out on the emblem field (clean chamfered cut)
    flat = lambda P: 0.12 + 0.88 * away_from_emblem(P, EMBLEM_FLAT)
    sdf.noise(N3, 0.012, 3.2, octaves=2, offset=(7.1, 0.3, 4.4), amp_fn=flat)
    sdf.noise(N3, 0.005, 7.0, octaves=3, offset=(3.3, 1.2, 7.7), amp_fn=flat)
    sdf.noise(N3, 0.002, 22.0, octaves=2, offset=(1.1, 9.4, 2.3), amp_fn=flat)
    sdf.noise(N3, 0.0025, 11.0, octaves=2, offset=(5.5, 2.2, 0.4), ridged=True, amp_fn=flat)
    obj = sdf.to_object("stone_finale", coll, mat, target_tris=52000, smooth_iter=1)
    return obj, sdf


def cut_emblem(stone, coll, frame, mat):
    cutter, info = SL.emblem_cutter("silva_finale_cutter", coll, frame, mode="through",
                                    thickness=Y_BACK - Y_FRONT, bevel=0.0035, lip=0.004, above=0.08,
                                    ds=0.7, mats=(mat, None))
    n0 = C.tri_count(stone)
    SL.boolean_difference(stone, cutter, solver="EXACT", material_mode="INDEX")
    C.remove_object(cutter.name)
    SL.keep_largest_part(stone)
    stone.data.materials.clear()
    stone.data.materials.append(mat)
    return dict(tris_before=n0, tris_after=C.tri_count(stone), cutter=info)


# ------------------------------------------------------------------------------------------
# moss hills
# ------------------------------------------------------------------------------------------

def stone_depth():
    c = CAMS["cam_finale_main"]
    _r, _u, fwd = C.cam_basis(cam_q(c["pitch"]))
    return float((np.array([STONE_X, 0.0, Z_B]) - np.asarray(c["loc"])) @ fwd)


_CREST_FINE = np.asarray(MOSS_CREST_FINE, float)


def crest_v(u):
    """Reference crest (frame 19, MOSS_CREST_FINE) at column(s) u."""
    v = np.interp(u, _CREST_FINE[:, 0], _CREST_FINE[:, 1])
    return float(v) if np.ndim(v) == 0 else v


def crest_v_r2(u):
    """The revision 1-2 crest polyline (MOSS_CREST), for comparison."""
    E = np.asarray(MOSS_CREST)
    v = np.interp(u, E[:, 0], E[:, 1])
    return float(v) if np.ndim(v) == 0 else v


def _island_plateau(sdf, N3):
    """Moss plateau: plan outline ISLAND_PLAN, floor at floor_top(y), rounded top edge (LIP_R), lumpy rim,
    the side / front faces recede downwards (UNDERCUT) so the mat sags over a dark undercut."""
    nx, ny, nz = sdf.shape
    vox = sdf.voxel
    xs = sdf.origin[0] + np.arange(nx) * vox
    ys = sdf.origin[1] + np.arange(ny) * vox
    zs = sdf.origin[2] + np.arange(nz) * vox
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    d2 = C.poly_sdf(np.stack([X.ravel(), Y.ravel()], 1), ISLAND_PLAN).reshape(nx, ny)
    d2 = d2 + 0.06 * N3.fbm(np.stack([X * 1.4, Y * 1.4, np.full_like(X, 3.7)], -1), 3)
    top = floor_top(Y)[..., None]
    for k0 in range(0, nz, 12):
        Z = zs[None, None, k0:k0 + 12]
        dz = Z - top
        dd = d2[..., None] + UNDERCUT * C.smoothstep(-0.02, -0.52, dz)
        wx, wy = dd + LIP_R, dz + LIP_R
        f = np.minimum(np.maximum(wx, wy), 0.0) + np.hypot(np.maximum(wx, 0.0), np.maximum(wy, 0.0)) - LIP_R
        sdf.d[:, :, k0:k0 + 12] = np.minimum(sdf.d[:, :, k0:k0 + 12], f.astype(np.float32))


def build_hills_base(N3):
    """The envelope E: the fitted moss field of the first build (plateau, hills along the frame-19 crest,
    contact ring, crest clamp) on the 1.6 cm grid, narrow pits between the hills filled before the clamp
    (fill_pits); the small cushion relief of revisions 1-2 is gone (the mound pile replaces it).
    Returns (sdf, contact ring centres, fill info)."""
    vox = HILL_VOX
    sdf = C.SDF((-3.15, -1.45, -0.62), (3.25, 1.55, 0.72), vox)
    ds = stone_depth()
    cam = "cam_finale_main"
    rng = np.random.default_rng(SEED + 811)
    _island_plateau(sdf, N3)

    def hill(u, dd, rx, ry, rz, k=0.09, lumps=3, lump=(0.3, 0.5)):
        """Moss cushion whose crest touches the reference crest line at column u, dd metres in front (-)
        of / behind (+) the stone depth; `lumps` sub-cushions on its flanks (never above the crest)."""
        top = S(cam, u, crest_v(u), ds + dd)
        c = top - np.array([0.0, 0.0, rz * 0.98])
        sdf.ellipsoid(c, (rx, ry, rz), k=k)
        for _ in range(lumps):
            a = rng.uniform(0.0, 2.0 * math.pi)
            el = rng.uniform(0.05, 0.75)
            dirn = np.array([math.cos(el) * math.cos(a), math.cos(el) * math.sin(a), math.sin(el)])
            p = c + dirn * np.array([rx, ry, rz]) * 0.86
            r = rng.uniform(*lump) * min(rx, ry, rz)
            r = min(r, max((top[2] - 0.02 - p[2]) / 0.85, 0.04))
            sdf.ellipsoid(p, (r * 1.2, r * 1.2, r * 0.85), k=0.045)
        return top

    # hills along the reference crest (frame 19), left -> right; dd < 0 = in front of the stone
    hill(-0.12, -0.75, 0.55, 0.45, 0.30)                    # out of frame left
    hill(0.035, -0.62, 0.42, 0.38, 0.30, lumps=4)           # A' outer left
    hill(0.15, -0.45, 0.46, 0.40, 0.36, lumps=4)            # A  big left
    hill(0.215, -0.22, 0.26, 0.30, 0.28, lumps=2)           # A-B shoulder
    # B: the pointy hill (round cone, its ridge running down to the right)
    tip = S(cam, 0.292, crest_v(0.292) + 0.004, ds + 0.10)
    base = tip.copy()
    base[2] = floor_top(base[1]) - 0.12
    base[0] -= 0.05
    sdf.round_cone(base, tip - np.array([0.0, 0.0, 0.045]), 0.40, 0.045, k=0.07)
    ridge = [tip - np.array([0.0, 0.0, 0.06]), S(cam, 0.33, crest_v(0.33) + 0.012, ds + 0.06),
             S(cam, 0.37, crest_v(0.37) + 0.012, ds + 0.0)]
    sdf.tube(ridge, [0.05, 0.09, 0.12], k=0.06)
    hill(0.255, -0.05, 0.2, 0.22, 0.2, lumps=1)              # B left shoulder
    hill(0.305, 0.08, 0.2, 0.22, 0.24, lumps=0)              # B round top (frame 19: a dome, not a spike)
    hill(0.385, -0.28, 0.24, 0.28, 0.22, lumps=2)           # valley floor B-C (in front)
    hill(0.47, 0.06, 0.55, 0.52, 0.34, k=0.12, lumps=3)     # C  central hill under the stone
    hill(0.585, -0.06, 0.2, 0.24, 0.2, k=0.07, lumps=1)     # C  right shoulder (moss up the stone side)
    hill(0.667, -0.30, 0.30, 0.32, 0.30, lumps=3)           # D
    hill(0.725, -0.12, 0.24, 0.30, 0.24, lumps=2)           # D-E
    hill(0.835, -0.48, 0.36, 0.36, 0.30, lumps=3)           # E
    hill(0.93, -0.62, 0.36, 0.36, 0.28, lumps=3)            # F
    hill(1.06, -0.7, 0.55, 0.45, 0.30)                      # F' (out of frame right)
    # hills behind the stone (seen from the entry pose) and low cushions on the floor in front
    for (u, dd, r) in [(0.42, 0.65, 0.32), (0.6, 0.7, 0.3), (0.25, 0.6, 0.3), (0.78, 0.55, 0.3), (0.1, 0.4, 0.3),
                       (0.92, 0.35, 0.3)]:
        p = S(cam, u, 0.62, ds + dd)
        p[2] = floor_top(p[1]) + 0.04 + 0.03 * rng.uniform(-1, 1)
        sdf.ellipsoid(p, (r * 1.25, r, r * 0.6), k=0.08)
    n_h = 0
    while n_h < 26:
        x = rng.uniform(-2.3, 2.5)
        y = rng.uniform(-1.0, -0.35)
        r = rng.uniform(0.08, 0.2)
        if C.poly_sdf(np.array([[x, y]]), ISLAND_PLAN)[0] > -(0.12 + r):
            continue
        z = floor_top(y) - r * rng.uniform(0.45, 0.65)
        sdf.ellipsoid((x, y, z), (r * 1.3, r * 1.1, r * 0.8), k=0.06)
        n_h += 1
    # contact ring around the stone foot (moss hugging the stone, a bit higher on the right)
    ring = []
    for a in np.linspace(0, 2 * math.pi, 28, endpoint=False):
        x = STONE_X + 0.33 * math.cos(a)
        y = 0.165 * math.sin(a)
        hgt = 0.045 + 0.03 * C.smoothstep(0.0, 0.3, x - STONE_X) + 0.01 * rng.uniform(-1, 1)
        ring.append((x, y, float(Z_B + hgt - 0.05)))
        sdf.ellipsoid(ring[-1], (0.09, 0.08, 0.07), k=0.05)
    # large-scale undulation only: the mounds are carved into this envelope (build_hills)
    sdf.noise(N3, 0.028, 2.0, octaves=2, offset=(2.1, 7.3, 0.4))
    # narrow pits / gaps between the hills (the engine showed them as black holes): filled first
    fill = fill_pits(sdf)
    # visual-hull clamp: E never rises above the reference crest (minus the fluff, minus the rise the bumps
    # add on top)
    crest_poly = [(float(u), float(v) + CREST_DROP + PILE_RISE) for (u, v) in MOSS_CREST_FINE] + \
        [(1.25, 3.0), (-0.25, 3.0)]
    c = CAMS[cam]
    sdf.clip_view(c["loc"], cam_q(c["pitch"]), crest_poly, k=0.025)
    return sdf, ring, fill


def fill_pits(sdf, ball=FILL_BALL, min_fill=FILL_MIN):
    """Fill the top surface where a ball of `ball` m rolling over it cannot reach (narrow pits and gaps
    between the hills; broad valleys stay). Only on the island (columns with moss), vertical slabs from
    6 cm below the old top up to the closed surface."""
    ztop, has = SL.top_heightfield(sdf)
    closed = SL.ball_closing(ztop, ball, sdf.voxel)
    depth = np.where(has, closed - ztop, 0.0)
    mask = has & (depth > min_fill)
    n = SL.fill_heightfield(sdf, closed, ztop - 0.06, mask, k=0.02)
    return dict(columns=int(n), max_fill_m=round(float(depth.max()), 3),
                over_5cm=int((depth > 0.05).sum()), over_10cm=int((depth > 0.10).sum()))


def cushion_zone(P):
    """Cushion weight: 1 on the island top and the hills, 0 down the undercut face below the lip."""
    P = np.atleast_2d(P)
    return C.smoothstep(-0.14, -0.05, P[:, 2] - floor_top(P[:, 1]))


def stone_foot(P):
    """1 inside the stone foot ellipse (STONE_FOOT_E), 0 from 1.6 x its size outwards."""
    P = np.atleast_2d(P)
    e = np.hypot((P[:, 0] - STONE_X) / STONE_FOOT_E[0], P[:, 1] / STONE_FOOT_E[1])
    return 1.0 - C.smoothstep(1.0, 1.6, e)


def copy_sdf(src):
    dst = C.SDF(src.origin, src.origin + (np.array(src.shape) - 1) * src.voxel, src.voxel)
    assert dst.shape == src.shape
    dst.d = src.d.copy()
    return dst


def pile_size(P):
    """Mound footprint semi-axis a (m) at plan positions P: larger in the near rows (PILE_ROWS), smaller at
    the stone (PILE_STONE)."""
    P = np.atleast_2d(P)
    R = np.asarray(PILE_ROWS, float)
    a = np.interp(P[:, 1], R[:, 0], R[:, 1])
    e = np.hypot((P[:, 0] - STONE_X) / PILE_STONE[0], P[:, 1] / PILE_STONE[1])
    return a * (PILE_STONE[2] + (1.0 - PILE_STONE[2]) * C.smoothstep(0.9, 1.7, e))


def pile_depth(P):
    """Depth of the gap floor under the envelope (m): the expected dome height at P."""
    return np.clip(0.5 * (PILE_H[0] + PILE_H[1]) * pile_size(P), *PILE_HCLIP)


class Mounds:
    """The mounds of the pile: plan centre (touch point p on the envelope's top), footprint semi-axes a, b
    (turned by phi), dome height h. Carved into the envelope from above (heightfield: no overhang, no
    cavity can form)."""

    def __init__(self, p, a, b, h, phi):
        self.p = np.asarray(p, float)
        self.a, self.b, self.h, self.phi = (np.asarray(x, float) for x in (a, b, h, phi))
        self.cos, self.sin = np.cos(self.phi), np.sin(self.phi)
        self.grid = SL.CellField(np.stack([self.p[:, 0], self.p[:, 1], np.zeros(len(self.p))], 1), cell=0.5)

    def __len__(self):
        return len(self.a)

    def carve2(self, XY):
        """The two smallest carving depths h x (1 - profile(q)) over the mounds at plan points XY (n, 2)
        (q = elliptic footprint distance / 1 at the rim); 9 where no mound is near."""
        P = np.stack([XY[:, 0], XY[:, 1], np.zeros(len(XY))], 1)
        n = len(P)
        g1 = np.full(n, 9.0)
        g2 = np.full(n, 9.0)
        for sel, cand in self.grid._groups(P):
            dx = P[sel, None, 0] - self.p[None, cand, 0]
            dy = P[sel, None, 1] - self.p[None, cand, 1]
            cs, sn = self.cos[None, cand], self.sin[None, cand]
            lx = (dx * cs + dy * sn) / self.a[None, cand]
            ly = (-dx * sn + dy * cs) / self.b[None, cand]
            G = self.h[None, cand] * (1.0 - mound_profile(np.sqrt(lx * lx + ly * ly)))
            rows = np.arange(len(sel))
            i = np.argmin(G, axis=1)
            g1[sel] = G[rows, i]
            if len(cand) > 1:
                G[rows, i] = np.inf
                g2[sel] = G.min(axis=1)
        return g1, g2


def mound_profile(q):
    """Dome profile over q (footprint distance / rim): 1 at the centre, 0 at the rim (elliptic arc, its foot
    rounded by PILE_EPS), continued linearly below 0 outside."""
    e = PILE_EPS
    k0 = math.sqrt(1.0 + e * e) - e
    inside = (np.sqrt(np.maximum(1.0 + e * e - q * q, 0.0)) - e) / k0
    return np.where(q < 1.0, inside, -(q - 1.0) / (e * k0))


def stone_hold(P):
    """1 at the stone foot (no carving: the stone stays seated in the fitted moss), 0 outside PILE_HOLD."""
    P = np.atleast_2d(P)
    e = np.hypot((P[:, 0] - STONE_X) / STONE_FOOT_E[0], P[:, 1] / STONE_FOOT_E[1])
    return 1.0 - C.smoothstep(PILE_HOLD[0], PILE_HOLD[1], e)


def silhouette_points(env, step_px=3):
    """Points of the envelope's silhouette in cam_finale_main: first hits of the rays just under the clamp
    line, ordered left -> right (on the island top, off the stone foot)."""
    cam = CAMS["cam_finale_main"]
    loc = np.asarray(cam["loc"], float)
    q = cam_q(cam["pitch"])
    us = (np.arange(0, C.FRAME_W, step_px) + 0.5) / C.FRAME_W
    vs = crest_v(us) + CREST_DROP + PILE_RISE + 0.004
    dirs = np.array([C.normalize(C.screen_to_world(loc, q, u, v, 1.0) - loc) for u, v in zip(us, vs)])
    t = np.full(len(us), 1.5)
    hit = np.zeros(len(us), bool)
    for _ in range(600):
        d = env.sample(loc + dirs * t[:, None])
        hit |= d < 0.002
        t = np.where(hit, t, t + np.clip(d * 0.9, 0.003, 0.05))
        if (hit | (t > 9.0)).all():
            break
    P = SL.project_to_surface(env, loc + dirs * t[:, None], iters=3)
    ok = hit & (cushion_zone(P) > 0.7) & (stone_hold(P) < 0.5)
    return P[ok]


def place_mounds(env, rng):
    """Mound touch points on the envelope: first a row along its silhouette in cam_finale_main (PILE_SIL,
    the crest), then dart throwing over the top (cushion zone, off the stone foot, upward-ish faces) in a
    seeded order; footprint semi-axes from pile_size x jitter, PILE_SPACING."""
    S0 = silhouette_points(env)
    a0 = pile_size(S0) * rng.uniform(1.0 - PILE_JITTER, 1.0 + PILE_JITTER, size=len(S0))
    # the crest's local peaks get a mound first (a dome top, never the notch between two mounds), highest
    # first; then the rest of the crest, highest first
    _u0, v0, _z0 = project("cam_finale_main", S0)
    n0, w = len(S0), 8
    vpad = np.pad(v0, w, mode="edge")
    vmin = np.min(np.stack([vpad[k:k + n0] for k in range(2 * w + 1)]), 0)
    order0 = np.lexsort((v0, ~(v0 <= vmin + 1e-9)))
    keep0 = []
    for i in order0:
        if all(np.linalg.norm(S0[i] - S0[j]) >= PILE_SIL * (a0[i] + a0[j]) for j in keep0):
            keep0.append(int(i))
    keep0 = sorted(keep0)
    S0, a0 = S0[keep0], a0[keep0]
    _idx, P = env.surface_points(env.voxel * 0.55)
    P = P[(cushion_zone(P) > 0.7) & (stone_hold(P) < 0.5)]
    order = rng.permutation(len(P))[:150000]
    P = SL.project_to_surface(env, P[order], iters=3)
    nrm = SL.sdf_normals(env, P)
    ok = nrm[:, 2] > 0.15
    P = P[ok]
    a = pile_size(P) * rng.uniform(1.0 - PILE_JITTER, 1.0 + PILE_JITTER, size=len(P))
    P, a = np.concatenate([S0, P]), np.concatenate([a0, a])
    keep = SL.poisson_surface(P, a, PILE_SPACING, cell=0.5, seeded=len(S0))
    P = P[keep]
    m = len(keep)
    a = a[keep]
    b = a * rng.uniform(*PILE_ASPECT, size=m)
    h = np.clip(a * rng.uniform(*PILE_H, size=m), *PILE_HCLIP)
    phi = rng.uniform(0.0, math.pi, size=m)
    M = Mounds(P, a, b, h, phi)
    M.silhouette = len(S0)
    return M


def pile_gap(env, P):
    """0 on the mound tops (at the envelope), 1 on the gap floor (pile_depth under it)."""
    P = np.atleast_2d(P)
    return np.clip(-env.sample(P) / pile_depth(P), 0.0, 1.0)


def apply_pile(sdf, env, M):
    """Carve the mound pile into the envelope from above: on E's top height map zE the surface drops by
    G = smin(smin of the two shallowest mound carvings, gap floor pile_depth) - the domes keep E's height at
    their centres, the gaps between them reach the floor; weighted by cushion_zone (none down the undercut
    faces) and 1 - stone_hold (the stone stays seated). Field: max(E, (z - H) x cos(slope of H)) - a
    height map, so no overhang or cavity can form, nothing rises above E."""
    zE, has = SL.top_heightfield(env)
    nx, ny = zE.shape
    vox = sdf.voxel
    X = env.origin[0] + np.arange(nx)[:, None] * vox + np.zeros((1, ny))
    Y = env.origin[1] + np.arange(ny)[None, :] * vox + np.zeros((nx, 1))
    ii, jj = np.nonzero(has)
    P = np.stack([X[ii, jj], Y[ii, jj], zE[ii, jj]], 1)
    w = cushion_zone(P) * (1.0 - stone_hold(P))
    # only where the column stays solid below the deepest cut (never through the undercut lip), and not at
    # the island's rim
    Dp = pile_depth(P)
    below = env.sample(np.stack([P[:, 0], P[:, 1], P[:, 2] - Dp - 0.04], 1))
    solid = np.zeros((nx, ny))
    solid[ii, jj] = (below < -0.005).astype(float)
    for _ in range(3):
        sp = np.pad(solid, 1, mode="edge")
        solid = sum(sp[1 + di:nx + 1 + di, 1 + dj:ny + 1 + dj] for di in (-1, 0, 1) for dj in (-1, 0, 1)) / 9.0
    w = w * C.smoothstep(0.55, 0.95, solid[ii, jj])
    w = w * C.smoothstep(PILE_EDGE[0], PILE_EDGE[1], -C.poly_sdf(P[:, :2], ISLAND_PLAN))
    g1, g2 = M.carve2(P[:, :2])
    G = C.smin(C.smin(g1, g2, PILE_K), Dp, PILE_CORE_K)
    carve = np.zeros((nx, ny))
    carve[ii, jj] = w * np.maximum(G, 0.0)
    H = np.where(has, zE - carve, zE)
    gx, gy = np.gradient(H, vox)
    cos_t = 1.0 / np.sqrt(1.0 + np.minimum(gx * gx + gy * gy, 16.0))
    act = has & (carve > 1e-5)
    zs = sdf.origin[2] + np.arange(sdf.shape[2]) * vox
    for i in range(nx):
        cols = np.nonzero(act[i])[0]
        if not len(cols):
            continue
        f = (zs[None, :] - H[i, cols, None]) * cos_t[i, cols, None]
        sdf.d[i, cols, :] = np.maximum(env.d[i, cols, :], f.astype(np.float32))
    return dict(columns=int(act.sum()), carve_max_m=round(float(carve.max()), 3),
                carve_mean_m=round(float(carve[act].mean()), 3) if act.any() else 0.0)


def knob_mean_abs(N3):
    """Mean |noise| (normalises the billow knobs to zero mean); fixed sample grid, deterministic."""
    g = np.stack(np.meshgrid(*[np.linspace(0.13, 9.71, 24)] * 3, indexing="ij"), -1).reshape(-1, 3)
    return float(np.abs(N3(g)).mean())


def place_cushions(sdf, rng, N3):
    """Cushion centres on the surface (cushion_zone > 0.5): dart throwing in a seeded order, radii in
    CUSHION_R; dome heights CUSHION_H x radius x a patch factor (calm and lumpy areas), lower at the stone
    foot; elongated footprints (CUSHION_ASPECT) in random directions."""
    _idx, P = sdf.surface_points(sdf.voxel * 0.55)
    P = P[cushion_zone(P) > 0.5]
    order = rng.permutation(len(P))[:80000]
    P = SL.project_to_surface(sdf, P[order], iters=3)
    radii = rng.uniform(*CUSHION_R, size=len(P))
    keep = SL.poisson_surface(P, radii, CUSHION_SPACING, cell=0.4)
    centres, r = P[keep], radii[keep]
    n = len(keep)
    lo = CUSHION_PATCH[1]
    patch = lo + (1.0 - lo) * C.smoothstep(-0.3, 0.3, N3(centres * CUSHION_PATCH[0] + np.array([5.3, 1.7, 8.1])))
    h = r * rng.uniform(*CUSHION_H, size=n) * patch * (1.0 - 0.9 * stone_foot(centres))
    ang = rng.uniform(0.0, math.pi, size=n)
    cells = SL.CellField(centres, cell=0.42)
    cells.r, cells.h = r, h
    cells.axis = np.stack([np.cos(ang), np.sin(ang), np.zeros(n)], 1)
    cells.asp = rng.uniform(*CUSHION_ASPECT, size=n)
    return cells


def cap_profile(q):
    """Dome profile over q = distance / footprint radius: 1 at the centre, 0 at q = 1 (finite slope),
    continued below 0 outside (so the smooth max ignores far domes)."""
    e = CUSHION_EPS
    k0 = math.sqrt(1.0 + e * e) - e
    inside = (np.sqrt(np.maximum(1.0 + e * e - q * q, 0.0)) - e) / k0
    return np.where(q < 1.0, inside, -(q - 1.0) / (e * k0))


def warp_xy(N3, P):
    """Horizontal domain warp of the cushion footprints (CUSHION_WARP)."""
    f, a = CUSHION_WARP
    Q = np.asarray(P, float) * f
    W = np.zeros((len(Q), 3))
    W[:, 0] = a * N3(Q + np.array([3.1, 9.4, 2.2]))
    W[:, 1] = a * N3(Q + np.array([7.7, 0.6, 4.9]))
    return P + W


def dome_field(cells, P, N3):
    """Height (m) of the cushion domes above the base at P: smooth max of the two highest dome caps and the
    ground (rounded creases where neighbours meet, ground level in the gaps); also the highest cap
    and the height of its dome. Footprints: elliptic (cells.axis / cells.asp), warped (warp_xy)."""
    Pw = warp_xy(N3, np.asarray(P, float))
    n = len(Pw)
    z1 = np.full(n, -1.0)
    z2 = np.full(n, -1.0)
    i1 = np.full(n, -1, np.int64)
    for sel, cand in cells._groups(Pw):
        dv = Pw[sel, None, :] - cells.c[None, cand, :]
        al = (dv * cells.axis[None, cand, :]).sum(-1)
        d2 = (dv * dv).sum(-1)
        s = cells.asp[None, cand]
        q = np.sqrt(np.maximum(al * al * s + (d2 - al * al) / s, 0.0)) / (CUSHION_FOOT * cells.r[None, cand])
        V = cells.h[None, cand] * cap_profile(q)
        a = np.argmax(V, axis=1)
        rows = np.arange(len(sel))
        z1[sel] = V[rows, a]
        i1[sel] = cand[a]
        if len(cand) > 1:
            V[rows, a] = -np.inf
            z2[sel] = V.max(axis=1)
    z = C.smax(C.smax(z1, z2, CUSHION_K), np.zeros(n), CUSHION_K)
    h1 = np.where(i1 >= 0, cells.h[np.maximum(i1, 0)], 1.0)
    return z, z1, h1


def knobs(N3, P, mean_abs):
    """Billow knobs 3-8 cm (rounded tops, creased between), outward only (mean = sum of the amplitudes)."""
    out = np.zeros(len(P))
    for f, a, off in KNOBS:
        out += a * np.abs(N3((P + np.asarray(off)) * f)) / mean_abs
    return out


def apply_cushions(sdf, cells, N3, env=None):
    """Outward displacement only (no hole can open): bumps + knobs (knobs weaker between the bumps), both
    lower down in the gaps of the mound pile (CUSHION_GAP, when the envelope `env` is given)."""
    mean_abs = knob_mean_abs(N3)

    def disp(P):
        z, _z1, h1 = dome_field(cells, P, N3)
        rel = np.clip(z / np.maximum(h1, 1e-6), 0.0, 1.0)
        kn = knobs(N3, P, mean_abs) * (0.3 + 0.7 * rel) * (1.0 - 0.8 * stone_foot(P))
        g = 1.0 if env is None else 1.0 - (1.0 - CUSHION_GAP) * C.smoothstep(0.2, 0.8, pile_gap(env, P))
        return cushion_zone(P) * (z + kn) * g

    hi = float(cells.h.max()) + 4.5 * sum(a for _f, a, _o in KNOBS) + CUSHION_K + 3.0 * sdf.voxel
    return SL.displace_band(sdf, disp, -3.0 * sdf.voxel, hi)


def vis_poses():
    """Camera poses that see the moss: in / main / out and halfway along the dolly and the drift."""
    out = [(np.asarray(CAMS[n]["loc"], float), CAMS[n]["pitch"]) for n in ("cam_finale_in", "cam_finale_main",
                                                                          "cam_finale_out")]
    for a, b in (("cam_finale_in", "cam_finale_main"), ("cam_finale_main", "cam_finale_out")):
        la, lb = np.asarray(CAMS[a]["loc"], float), np.asarray(CAMS[b]["loc"], float)
        out.append((0.5 * (la + lb), 0.5 * (CAMS[a]["pitch"] + CAMS[b]["pitch"])))
    return out


def visible_mask(co, nrm, bvh, margin=0.03):
    """Vertices seen (in frame, facing, unoccluded in `bvh`) from any of the vis_poses."""
    vis = np.zeros(len(co), bool)
    cast = bvh.ray_cast
    for loc, pitch in vis_poses():
        u, v, z = C.project_to_screen(loc, cam_q(pitch), co)
        cand = (u > -margin) & (u < 1.0 + margin) & (v > -margin) & (v < 1.0 + margin) & (z > 0.1) & ~vis
        cand &= np.einsum("ij,ij->i", nrm, loc[None, :] - co) > 0.0
        ids = np.nonzero(cand)[0]
        org = co[ids] + nrm[ids] * 0.003
        dvec = loc[None, :] - org
        dist = np.linalg.norm(dvec, axis=1)
        dirs = dvec / dist[:, None]
        orgl, dirl, distl = org.tolist(), dirs.tolist(), dist.tolist()
        for k, i in enumerate(ids.tolist()):
            if cast(orgl[k], dirl[k], distl[k])[0] is None:
                vis[i] = True
    return vis


def build_hills(coll, mat, N3):
    """moss_finale_hills: the envelope E (fitted hills, pits filled, clamped), resampled to FINE_VOX; the
    mound pile carved into it; the stone's contact ring; a small rolling-ball closing; 10-15 cm bumps +
    knobs (outward); the dome-top clamp; bottom clipped; decimated with the hidden parts thinned first."""
    t = time.time()
    base, ring, fill = build_hills_base(N3)
    C.log(f"hills base: {time.time() - t:.1f}s, pit fill {fill}")
    t = time.time()
    env = SL.resample_sdf(base, FINE_VOX)
    del base
    rng = np.random.default_rng(SEED + 812)
    M = place_mounds(env, rng)
    sdf = copy_sdf(env)
    carve = apply_pile(sdf, env, M)
    C.log(f"mound pile: {len(M)} mounds ({M.silhouette} along the crest; a {M.a.min():.3f}-{M.a.max():.3f} m, "
          f"h {M.h.min():.3f}-{M.h.max():.3f} m), carving {carve} ({time.time() - t:.1f}s)")
    # contact ring around the stone foot
    for c in ring:
        sdf.ellipsoid(c, (0.09, 0.08, 0.07), k=0.05)
    t = time.time()
    fill2 = fill_pits(sdf, FILL2_BALL, FILL2_MIN)
    C.log(f"pile closing ({FILL2_BALL} m ball): {fill2} ({time.time() - t:.1f}s)")
    t = time.time()
    rng = np.random.default_rng(SEED + 813)
    cells = place_cushions(sdf, rng, N3)
    n_band = apply_cushions(sdf, cells, N3, env)
    C.log(f"bumps: {len(cells.c)} cells (r {cells.r.min():.3f}-{cells.r.max():.3f} m, dome "
          f"{cells.h.min():.3f}-{cells.h.max():.3f} m), {n_band} band voxels ({time.time() - t:.1f}s)")
    # upper crest clamp: cushion tops may rise LUMP_UP above the fitted crest line, no further (keeps the
    # notches of the frame-19 crest open and the moss line on the stone where the first fit put it)
    cam = CAMS["cam_finale_main"]
    us = np.linspace(-0.25, 1.25, 301)
    upper = [(float(u), float(v)) for u, v in zip(us, upper_clamp_v(us))]
    sdf.clip_view(cam["loc"], cam_q(cam["pitch"]), upper + [(1.25, 3.0), (-0.25, 3.0)], k=0.015)
    sdf.clip_plane((0, 0, -0.55), (0, 0, -1))
    n_pockets = open_pockets(sdf)
    C.log(f"pocket opening ({POCKET_R} vox): {n_pockets} voxels filled")
    t = time.time()
    obj = sdf.to_object("moss_finale_hills", coll, mat, target_tris=None,
                        drop_cut_planes=[((0, 0, -0.55), (0, 0, -1))])
    loose = SL.keep_largest_part(obj, 0.02)            # no floating crumbs
    n_full = C.tri_count(obj)
    # decimation: the parts no pose sees first (down to HIDDEN_KEEP), then everything to the budget
    co, nrm, ed, _ = C.mesh_arrays(obj.data)
    vis = SL.dilate_mask(visible_mask(co, nrm, C.build_bvh([obj])), ed, 4)
    hidden = 1.0 - float(vis.mean())
    ratio = 1.0 - hidden * (1.0 - HIDDEN_KEEP)
    n_pass1 = n_full
    if ratio * n_full > HILLS_TRIS:
        n_pass1 = SL.decimate_vgroup(obj, ratio, (~vis).astype(float))
    n_final = C.decimate(obj, HILLS_TRIS)
    info = dict(fill=fill, pile_fill=fill2, mounds=len(M), bumps=len(cells.c), pocket_voxels=n_pockets,
                loose_faces_removed=loose,
                full_tris=n_full,
                hidden_share=round(hidden, 3), after_hidden_pass=n_pass1, tris=n_final)
    C.log(f"moss_finale_hills mesh: {info} ({time.time() - t:.1f}s)")
    return obj, sdf, env, M, cells, info


def upper_clamp_v(u):
    """Screen v (cam_finale_main) of the upper crest clamp: the fitted crest + CREST_DROP, minus LUMP_UP
    (less at the stone)."""
    u = np.asarray(u, float)
    at_stone = C.smoothstep(0.34, 0.37, u) * (1.0 - C.smoothstep(0.63, 0.66, u))
    return crest_v(u) + CREST_DROP - (LUMP_UP[0] + (LUMP_UP[1] - LUMP_UP[0]) * at_stone)


def open_pockets(sdf, r=POCKET_R):
    """Grey opening of the field (min filter, then max filter over a cube of half-size r voxels, separable):
    narrow peaks of the distance field - the centres of pockets and slots narrower than ~2 r voxels - drop
    below 0 and fill; on flat, convex and wide concave parts the field is unchanged; the field is never raised
    (no hole can open). Returns the number of voxels that turned solid."""
    d = sdf.d

    def filt(a, op):
        for ax in range(3):
            out = a.copy()
            n = a.shape[ax]
            for k in range(1, r + 1):
                lo = [slice(None)] * 3
                hi = [slice(None)] * 3
                lo[ax], hi[ax] = slice(0, n - k), slice(k, n)
                op(out[tuple(hi)], a[tuple(lo)], out=out[tuple(hi)])
                op(out[tuple(lo)], a[tuple(hi)], out=out[tuple(lo)])
            a = out
        return a

    o = np.minimum(filt(filt(d, np.minimum), np.maximum), d)
    n = int(((o < 0.0) & (d >= 0.0)).sum())
    sdf.d = o.astype(np.float32)
    return n


# ------------------------------------------------------------------------------------------
# portrait screens: the near mound continues as a moss bank towards and below the cameras
# ------------------------------------------------------------------------------------------

FIT_MIN_H = 0.72            # engine CameraRig fit.minHorizontalFraction / maxFov (the finale keeps the defaults)
FIT_MAX_FOV = 70.0
PORTRAIT_ASPECT = 375.0 / 812.0     # 9:19.5 phone
PORTRAIT_FOVS = (60.0, round(math.degrees(2.0 * min(math.atan(FIT_MIN_H * TH / PORTRAIT_ASPECT),
                                                    math.radians(FIT_MAX_FOV / 2.0))), 2))    # 60 / 69.52 deg
BANK_X = (-2.7, 2.7)        # plan extent (m): from behind cam_finale_in (y -5.43) to under the island
BANK_Y = (-5.7, 0.3)
BANK_STEP = 0.1             # heightfield grid (m), coarse: no 1440x1020 frame sees it
BANK_TOP = 0.32             # never higher (m): the portrait bottom edge meets the bank 1-1.3 m from the lens
BANK_MARGIN = 0.06          # under the lowest 35 deg bottom-edge plane of the camera track (m)
BANK_UNDER = -0.58          # floor just under the island's cut bottom (z -0.55): rays under the lip end here
BANK_DIP = 0.08             # mound relief: the gaps this far under the mound tops (m)
BANK_MOUND_R = (0.22, 0.45)  # mound footprint radius (m)
BANK_AO_DIST = 1.5          # AO reach (m): the floor under the island sees the island above it
ENGINE_KEY_DIR = (0.6, 0.25, -0.72)  # FinaleScene KEY_DIR (0.6, -0.72, -0.25) in Blender axes: the key's travel
BANK_SHADOW_B = 0.1         # COLOR_0.B factor where the island hides that key (baked shadow: the engine's
                            # near-mound material takes no shadow map; B scales its albedo, not its specular)
# finale/track.ts inputs (FinaleScene STONE_TOP / STONE_WIDE_CENTRE in Blender coordinates)
TRACK_STONE_TOP = (-0.085, -0.02, 1.046)
TRACK_STONE_CENTRE = (0.0, -0.07, 0.615)
TRACK_TOP_RISE = [(0.0, 0.0), (0.236, 0.055), (0.5, 0.209), (0.773, 0.65), (1.0, 1.0)]
TRACK_DRIFT_TAU = 0.63
TRACK_HZ = 30.0


def _mono_curve(points):
    """Monotone cubic (Fritsch-Carlson) through (x, y) samples, clamped outside (track.ts monotoneCurve)."""
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    n = len(points)
    d = [(ys[i + 1] - ys[i]) / max(1e-9, xs[i + 1] - xs[i]) for i in range(n - 1)]
    m = [0.0] * n
    m[0], m[n - 1] = d[0], d[n - 2]
    for i in range(1, n - 1):
        m[i] = 0.0 if d[i - 1] * d[i] <= 0 else 0.5 * (d[i - 1] + d[i])
    for i in range(n - 1):
        if abs(d[i]) < 1e-12:
            m[i] = m[i + 1] = 0.0
            continue
        a, b = m[i] / d[i], m[i + 1] / d[i]
        h = a * a + b * b
        if h > 9.0:
            tau = 3.0 / math.sqrt(h)
            m[i], m[i + 1] = tau * a * d[i], tau * b * d[i]

    def f(x):
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        i = 0
        while i < n - 2 and x > xs[i + 1]:
            i += 1
        h = xs[i + 1] - xs[i]
        s_ = (x - xs[i]) / h
        s2, s3 = s_ * s_, s_ * s_ * s_
        return ((2 * s3 - 3 * s2 + 1) * ys[i] + (s3 - 2 * s2 + s_) * h * m[i] + (-2 * s3 + 3 * s2) * ys[i + 1]
                + (s3 - s2) * h * m[i + 1])
    return f


def _bisect(f, target, lo, hi):
    """Bisection on a monotone f(s) = target over [lo, hi] (track.ts solve)."""
    a, b = lo, hi
    fa = f(a) - target
    inc = f(b) - target > fa
    for _ in range(48):
        mid = 0.5 * (a + b)
        v = f(mid) - target
        if v == 0:
            return mid
        if (v < 0) == inc:
            a = mid
        else:
            b = mid
    return 0.5 * (a + b)


def engine_track():
    """The engine's finale camera track (finale/track.ts, replayed): [(t, loc, pitch)] at 30 Hz from tIn to
    tOut. in -> main: the position share follows the stone width (CAM_DOLLY, 1 / distance), the rotation
    share is solved so the stone top follows TRACK_TOP_RISE; main -> out: exponential drift. All three
    poses are pitch-only (x = 0, no yaw / roll), so the slerp is a linear pitch blend."""
    ci, cm, co = (CAMS[n] for n in ("cam_finale_in", "cam_finale_main", "cam_finale_out"))
    t_in, t_main, t_out = ci["t"], cm["t"], co["t"]
    top = np.array(TRACK_STONE_TOP)[None]
    centre = np.array(TRACK_STONE_CENTRE)
    li, lm, lo = (np.array(c["loc"], float) for c in (ci, cm, co))
    p_in, p_main, p_out = ci["pitch"], cm["pitch"], co["pitch"]
    width = _mono_curve(CAM_DOLLY)
    d_in, d_main = np.linalg.norm(li - centre), np.linalg.norm(lm - centre)

    def pos_share(w):
        target = 1.0 / (1.0 / d_in + w * (1.0 / d_main - 1.0 / d_in))
        return _bisect(lambda s_: float(np.linalg.norm(li + (lm - li) * s_ - centre)), target, -0.2, 1.2)

    def screen_v(loc, pitch):
        return float(C.project_to_screen(loc, cam_q(pitch), top)[1][0])

    v_in, v_main = screen_v(li, p_in), screen_v(lm, p_main)
    rise = _mono_curve([(t_in + k * (t_main - t_in), r) for k, r in TRACK_TOP_RISE])
    norm = 1.0 - math.exp(-(t_out - t_main) / TRACK_DRIFT_TAU)
    poses = [(t_in, li, p_in)]
    n = max(1, int(math.floor((t_out - t_in) * TRACK_HZ + 0.5)))
    for i in range(1, n + 1):
        t = t_in + (t_out - t_in) * i / n
        if t <= t_main + 1e-6:
            loc = li + (lm - li) * pos_share(width(t))
            v_t = v_in + (v_main - v_in) * rise(t)
            k = _bisect(lambda kk: screen_v(loc, p_in + (p_main - p_in) * kk), v_t, -0.5, 1.5)
            poses.append((t, loc, p_in + (p_main - p_in) * k))
        else:
            k = min(1.0, max(0.0, (1.0 - math.exp(-(t - t_main) / TRACK_DRIFT_TAU)) / norm))
            poses.append((t, lm + (lo - lm) * k, p_main + (p_out - p_main) * k))
    poses[-1] = (t_out, lo, p_out)
    return poses


def track_cap(y, poses):
    """Height (m) of the lowest 35 deg bottom-edge plane over the camera track at plan depth y: nothing under
    it is inside any 1440x1020 (or wider) frame of the finale. The planes contain the camera x axis (no yaw /
    roll), so they do not depend on x."""
    y = np.asarray(y, float)
    out = np.full(y.shape, np.inf)
    for _t, loc, pitch in poses:
        pr = math.radians(pitch)
        dy = math.cos(pr) + TV * math.sin(pr)
        dz = math.sin(pr) - TV * math.cos(pr)
        out = np.minimum(out, loc[2] + (y - loc[1]) * dz / dy)
    return out


def build_near_bank(coll, mat, N3, poses):
    """Moss bank under the near mound (merged into moss_finale_near__fg after colouring): a heightfield over
    BANK_X x BANK_Y whose top is min(BANK_TOP, max(track cap - BANK_MARGIN, BANK_UNDER)) minus a relief of
    round mounds (dart-thrown, BANK_MOUND_R, gaps BANK_DIP deep) and a little fbm; the relief fades out
    under the island, where the bank is a flat floor just under the island's cut bottom."""
    xs = np.arange(BANK_X[0], BANK_X[1] + 1e-9, BANK_STEP)
    ys = np.arange(BANK_Y[0], BANK_Y[1] + 1e-9, BANK_STEP)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel()], 1)
    rng = np.random.default_rng(SEED + 822)
    cen, rad = [], []
    for _ in range(4000):
        p = np.array([rng.uniform(*BANK_X), rng.uniform(*BANK_Y)])
        r = rng.uniform(*BANK_MOUND_R)
        if all(math.hypot(*(p - c)) >= 0.8 * (r + rc) for c, rc in zip(cen, rad)):
            cen.append(p)
            rad.append(r)
    dome = np.zeros(len(P))
    for c, r in zip(cen, rad):
        q = np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]) / r
        dome = np.maximum(dome, np.sqrt(np.clip(1.0 - q * q, 0.0, 1.0)))
    nz = N3.fbm(np.stack([P[:, 0] * 2.2, P[:, 1] * 2.2, np.full(len(P), 7.3)], 1), 3)
    cap = track_cap(P[:, 1], poses)
    top = np.minimum(BANK_TOP, np.maximum(cap - BANK_MARGIN, BANK_UNDER))
    relief = 1.0 - C.smoothstep(-1.6, -1.15, P[:, 1])
    Z = top - (BANK_DIP * (1.0 - dome) + 0.012 * (0.5 + 0.5 * nz)) * relief
    V = np.stack([P[:, 0], P[:, 1], Z], 1)
    nx, ny = len(xs), len(ys)
    idx = np.arange(nx * ny).reshape(nx, ny)
    F = np.stack([idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()], 1)
    me = C.mesh_from_numpy("moss_finale_near_bank", V, F, smooth=True)
    obj = C.object_from_mesh("moss_finale_near_bank", me, coll, mat)
    over = Z > cap - 1e-6
    return obj, dict(verts=int(len(V)), tris=int(2 * len(F)), mounds=len(cen),
                     z_range=[round(float(Z.min()), 3), round(float(Z.max()), 3)],
                     above_track_cap=int(over.sum()),
                     above_track_cap_only_from_y=round(float(P[over, 1].min()), 2) if over.any() else None)


def bank_density_off_under_island(obj):
    """COLOR_0.R = 0 (no plants) where the bank runs under the island (y > -1.3 m, ramp from -1.5 m): nobody
    sees it there, yet it lies inside the engine's near-mound scatter frusta (main / out + 15 % margin, no
    occlusion test) and would draw plants from the mound's budget."""
    me = obj.data
    col = C.read_color0(me)
    co, _, _, _ = C.mesh_arrays(me)
    col[:, 0] *= 1.0 - C.smoothstep(-1.5, -1.3, co[:, 1])
    C.write_color0(me, col[:, 0], col[:, 1], col[:, 2], col[:, 3])


def near_scatter_weight(obj, first_vertex, density_power=0.5, margin=0.15, f_min=-0.3, f_full=0.15):
    """The engine's near-mound scatter weight per triangle (SurfaceScatter.viewImportance with the finale's
    near views main / out, 35 deg, 1440x1020, margin 15 %; x mean R^density_power x area): the share that falls
    on the appended bank (vertices from first_vertex on)."""
    me = obj.data
    co, _, _, tri = C.mesh_arrays(me)
    col = C.read_color0(me)
    P = co[tri]
    c = P.mean(1)
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    area = 0.5 * np.linalg.norm(n, axis=1)
    vn = np.empty(len(me.vertices) * 3)
    me.vertex_normals.foreach_get("vector", vn)
    nm = vn.reshape(-1, 3)[tri].sum(1)
    nm /= np.maximum(np.linalg.norm(nm, axis=1)[:, None], 1e-8)
    imp = np.zeros(len(tri))
    for name in ("cam_finale_main", "cam_finale_out"):
        cam = CAMS[name]
        loc = np.asarray(cam["loc"], float)
        right, up, fwd = C.cam_basis(cam_q(cam["pitch"]))
        rel = c - loc
        z = rel @ fwd
        tv = TV * (1.0 + margin)
        inside = (z > 0.02) & (np.abs(rel @ right) <= z * tv * C.ASPECT) & (np.abs(rel @ up) <= z * tv)
        to_cam = -rel / np.maximum(np.linalg.norm(rel, axis=1)[:, None], 1e-6)
        x = np.clip((np.einsum("ij,ij->i", nm, to_cam) - f_min) / (f_full - f_min), 0.0, 1.0)
        facing = x * x * (3.0 - 2.0 * x)
        imp = np.maximum(imp, np.where(inside, facing, 0.0))
    dens = np.clip(col[tri, 0].mean(1), 0.0, 1.0) ** density_power
    w = area * dens * imp
    bank = (tri >= first_vertex).all(1)
    return dict(bank_share=round(float(w[bank].sum() / max(w.sum(), 1e-12)), 6),
                bank_tris_with_weight=int((w[bank] > 0).sum()))


def bank_island_shadow(obj, bvh):
    """Bake the island's key shadow into the bank's COLOR_0.B (x BANK_SHADOW_B where a ray towards the engine's
    key light hits the island or the stone): the near-mound material takes no shadow map. The floor under the
    island shows through the sliver under the lip in the 1440x1020 frames (54.0-55.6 s); there the engine
    still draws the key's grazing specular on it (up to 17/255 at 54.3 s), which B does not scale."""
    me = obj.data
    co, _, _, _ = C.mesh_arrays(me)
    col = C.read_color0(me)
    L = -np.asarray(ENGINE_KEY_DIR, float)
    L /= np.linalg.norm(L)
    Lv = Vector(tuple(L))
    lit = np.ones(len(co))
    for i, p in enumerate(co.tolist()):
        if bvh.ray_cast(Vector(p) + Lv * 0.01, Lv, 20.0)[0] is not None:
            lit[i] = 0.0
    col[:, 2] *= BANK_SHADOW_B + (1.0 - BANK_SHADOW_B) * lit
    C.write_color0(me, col[:, 0], col[:, 1], col[:, 2], col[:, 3])
    return int((lit == 0).sum())


def merge_into(obj, other):
    """Append `other` (world space, COLOR_0, UVMap) to obj's mesh; the result keeps obj's name, collection and
    material, both sources are removed. Returns (new object, dict(first_vertex, vertices) of the appended
    part)."""
    Vs, faces, uvs, cols = [], [], [], []
    off = 0
    for o in (obj, other):
        me = o.data
        nv, nl, nf = len(me.vertices), len(me.loops), len(me.polygons)
        co = np.empty(nv * 3)
        me.vertices.foreach_get("co", co)
        Mw = np.array(o.matrix_world)
        co = co.reshape(-1, 3) @ Mw[:3, :3].T + Mw[:3, 3]
        lv = np.empty(nl, np.int64)
        me.loops.foreach_get("vertex_index", lv)
        lt = np.empty(nf, np.int64)
        me.polygons.foreach_get("loop_total", lt)
        ls = np.empty(nf, np.int64)
        me.polygons.foreach_get("loop_start", ls)
        uv = np.zeros(nl * 2, np.float32)
        me.uv_layers[C.UV_NAME].data.foreach_get("uv", uv)
        uv = uv.reshape(-1, 2)
        for k in np.unique(lt):
            sel = np.nonzero(lt == k)[0]
            ids = ls[sel][:, None] + np.arange(k)[None, :]
            faces.append(lv[ids] + off)
            uvs.append(uv[ids].reshape(-1, 2))
        Vs.append(co)
        cols.append(C.read_color0(me))
        off += nv
    name, coll = obj.name, obj.users_collection[0]
    mat = obj.data.materials[0] if len(obj.data.materials) else None
    first = len(Vs[0])
    me = C.mesh_from_numpy(name, np.vstack(Vs), faces, smooth=True)
    uv_all = np.vstack(uvs)
    if len(me.loops) != len(uv_all) or len(me.vertices) != off:
        raise RuntimeError(f"merge_into: mesh changed while merging ({len(me.vertices)} / {off} vertices, "
                           f"{len(me.loops)} / {len(uv_all)} loops)")
    C.set_uvs(me, uv_all)
    col = np.vstack(cols)
    C.write_color0(me, col[:, 0], col[:, 1], col[:, 2], col[:, 3])
    C.remove_object(other.name)
    new = C.object_from_mesh(name, me, coll, mat)
    return new, dict(first_vertex=int(first), vertices=int(off - first))


def _id_bvh(objs):
    """One BVH over the objects' world triangles + the object index of every triangle."""
    Vs, Fs, fo = [], [], []
    off = 0
    for k, o in enumerate(objs):
        co, _, _, tri = C.mesh_arrays(o.data)
        Mw = np.array(o.matrix_world)
        Vs.append(co @ Mw[:3, :3].T + Mw[:3, 3])
        Fs.append(tri + off)
        fo.append(np.full(len(tri), k))
        off += len(co)
    return BVHTree.FromPolygons(np.vstack(Vs).tolist(), np.vstack(Fs).tolist(), all_triangles=True), np.concatenate(fo)


def _ray_dirs(loc, pitch, fov_v, aspect, W, H):
    right, up, fwd = C.cam_basis(cam_q(pitch))
    tv = math.tan(math.radians(fov_v) / 2.0)
    U, Vv = np.meshgrid((np.arange(W) + 0.5) / W, (np.arange(H) + 0.5) / H)
    D = (fwd[None, None] + right[None, None] * ((2.0 * U - 1.0) * tv * aspect)[..., None]
         + up[None, None] * ((1.0 - 2.0 * Vv) * tv)[..., None])
    return D / np.linalg.norm(D, axis=-1)[..., None]


def _cast_ids(bvh, face_obj, loc, D):
    """Object index hit first by each ray (-1: background)."""
    o = Vector(tuple(float(x) for x in loc))
    rc = bvh.ray_cast
    ids = np.full(D.shape[:-1], -1, np.int64).ravel()
    for i, d in enumerate(D.reshape(-1, 3).tolist()):
        hit = rc(o, Vector(d), 200.0)
        if hit[0] is not None:
            ids[i] = face_obj[hit[2]]
    return ids.reshape(D.shape[:-1])


def bank_in_landscape(objs, bank, poses, W=288, H=204, every=4):
    """Where the bank shows in the 1440x1020 frames along the track (every 4th 30 Hz pose + the ends):
    pixels that hit it first, and what the same rays hit without it (must be background only)."""
    bvh_all, fo_all = _id_bvh(list(objs) + [bank])
    bvh_old, fo_old = _id_bvh(list(objs))
    kb = len(objs)
    sel = sorted(set(list(range(0, len(poses), every)) + [len(poses) - 1]))
    per, over, worst = {}, 0, (0, None)
    for i in sel:
        t, loc, pitch = poses[i]
        D = _ray_dirs(loc, pitch, C.FOV_V_DEG, C.ASPECT, W, H)
        ids = _cast_ids(bvh_all, fo_all, loc, D)
        m = ids == kb
        if m.any():
            old = _cast_ids(bvh_old, fo_old, loc, D[m][:, None, :])
            over += int((old >= 0).sum())
            per[f"{t:.2f}"] = round(float(m.mean()), 5)
            if m.mean() > worst[0]:
                worst = (float(m.mean()), round(t, 2))
    return dict(poses=len(sel), frames_with_bank=len(per), max_share=round(worst[0], 5), max_at_t=worst[1],
                bank_px_over_geometry=over, shares=per)


_PORTRAIT_IDS = {}


def portrait_coverage(coll, W=150, H=325):
    """Portrait 9:19.5 frames (PORTRAIT_FOVS) from the three poses: moss (moss_* meshes) per pixel. Per frame:
    the lowest crest row (lowest first-moss row over the columns), the smallest moss share of a row below it,
    the bottom row's share and the background ('holes') below the first moss pixel of each column."""
    objs = [o for o in coll.objects if o.type == "MESH"]
    bvh, fo = _id_bvh(objs)
    moss = np.array([o.name.startswith("moss_") for o in objs])
    out = {}
    for n in ("cam_finale_in", "cam_finale_main", "cam_finale_out"):
        c = CAMS[n]
        for fv in PORTRAIT_FOVS:
            ids = _cast_ids(bvh, fo, c["loc"], _ray_dirs(c["loc"], c["pitch"], fv, PORTRAIT_ASPECT, W, H))
            m = np.where(ids >= 0, moss[np.maximum(ids, 0)], False)
            first = np.where(m.any(0), np.argmax(m, 0), H)
            below = np.arange(H)[:, None] >= first[None, :]
            holes = below & (ids < 0)
            lowest = int(first.max())
            rows = m[lowest:].mean(1) if lowest < H else np.zeros(1)
            key = f"{n[11:]}_{fv:g}"
            out[key] = dict(fov_v=fv, lowest_crest_v=round(lowest / H, 3),
                            lowest_row_moss_share=round(float(rows.min()), 4),
                            bottom_row_moss_share=round(float(m[-1].mean()), 4),
                            moss_share_below_crest=round(float(m[lowest:].mean()), 4) if lowest < H else 0.0,
                            holes_share=round(float(holes.sum() / max(below.sum(), 1)), 5))
            _PORTRAIT_IDS[key] = (ids, [o.name for o in objs])
    return out


def portrait_sheet(scene, coll):
    """finale_cmp_portrait: lit previews of 9:19.5 portrait frames (60 deg, the engine's 69.5 deg) from the
    in / main / out poses (top row), with the object id maps of the coverage check (bottom row; background
    black)."""
    key = KEYS["key_finale"]["dir"]
    SL.setup_lit_preview(scene, key, BG_HEX, percent=100, sun_strength=4.2, sun_color=(1.0, 0.95, 0.86),
                         fill_dir=(0.5, 0.75, -0.3), fill_strength=0.2, world_strength=1.0)
    W, H = 300, 650
    scene.render.resolution_x, scene.render.resolution_y = W, H
    os.makedirs(TMP, exist_ok=True)
    cols = {"moss_finale_hills": (0.36, 0.6, 0.16), "moss_finale_near__fg": (0.5, 0.42, 0.2),
            "stone_finale": (0.75, 0.72, 0.64), "far_finale_left": (0.15, 0.2, 0.35),
            "far_finale_right": (0.15, 0.25, 0.42)}
    top, bottom = [], []
    try:
        with SL.PreviewMaterials(coll, preview_mats()):
            for fv in PORTRAIT_FOVS:
                for n in ("cam_finale_in", "cam_finale_main", "cam_finale_out"):
                    c = CAMS[n]
                    cam = C.make_camera("tmp_portrait", coll, c["loc"], cam_q(c["pitch"]), fov_v_deg=fv)
                    cam.data.sensor_fit = "VERTICAL"
                    cam.data.angle_y = math.radians(fv)
                    p = os.path.join(TMP, f"portrait_{n}_{fv:g}.png")
                    C.render_camera(scene, cam, p)
                    C.remove_object("tmp_portrait")
                    img = SL.load_rgba(p)[..., :3].copy()
                    SL.label(img, f"{n[11:]} {fv:g}", 4, 4)
                    top.append(np.pad(img, ((2, 2), (2, 2), (0, 0)), constant_values=1.0))
                    ids, names = _PORTRAIT_IDS.get(f"{n[11:]}_{fv:g}", (None, None))
                    idimg = np.zeros((H, W, 3))
                    if ids is not None:
                        small = np.zeros(ids.shape + (3,))
                        for k, nm in enumerate(names):
                            small[ids == k] = cols.get(nm, (1.0, 0.0, 1.0))
                        yi = (np.arange(H) * ids.shape[0] // H)
                        xi = (np.arange(W) * ids.shape[1] // W)
                        idimg = small[yi][:, xi]
                    bottom.append(np.pad(idimg, ((2, 2), (2, 2), (0, 0)), constant_values=1.0))
    finally:
        SL.remove_preview_lights(scene)
        scene.render.resolution_x, scene.render.resolution_y = C.FRAME_W, C.FRAME_H
    img = np.concatenate([np.concatenate(top, 1), np.concatenate(bottom, 1)], 0)
    path = os.path.join(CAPTURES, "finale_cmp_portrait.png")
    SL.save_rgb(img, path)
    return path


def build_near_fg(coll, mat, N3):
    """moss_finale_near__fg: low, dark, out-of-focus mound along the bottom edge of the main pose."""
    sdf = C.SDF((-2.6, -3.25, -0.55), (2.6, -1.45, 0.75), 0.03)
    rng = np.random.default_rng(SEED + 821)
    for u in np.linspace(-0.25, 1.25, 9):
        d = 1.75 + 0.25 * rng.uniform(-1, 1)
        p = S("cam_finale_main", u, 1.0 + 0.02 * rng.uniform(-1, 1), d)
        r = 0.32 + 0.06 * rng.uniform(-1, 1)
        edge = abs(u - 0.5) * 2.0
        top = p + np.array([0.0, 0.0, -0.05 + 0.04 * edge])
        sdf.ellipsoid(top - np.array([0, 0, r * 0.55]), (r * 1.5, r, r * 0.6), k=0.12)
    sdf.noise(N3, 0.035, 3.5, octaves=3, offset=(3.7, 0.2, 5.5))
    sdf.clip_plane((0, 0, -0.5), (0, 0, -1))
    obj = sdf.to_object("moss_finale_near__fg", coll, mat, target_tris=7000,
                        drop_cut_planes=[((0, 0, -0.5), (0, 0, -1))])
    return obj, sdf


# ------------------------------------------------------------------------------------------
# far forms
# ------------------------------------------------------------------------------------------

def build_far(coll, mat, N3, side):
    """far_finale_<side>: a few leaning trunks (slightly bent round-cone chains, flared at the base, bark
    ridges along them) merged with rock lumps, near-black (mat_far), same depth band and extents as before."""
    cam = "cam_finale_main"
    ds = stone_depth()
    sign = -1.0 if side == "left" else 1.0
    lo = np.array([-11.0 if side == "left" else 0.0, 5.0, -3.6])
    hi = np.array([0.0 if side == "left" else 11.0, 13.0, 9.5])
    sdf = C.SDF(lo, hi, 0.06)
    rng = np.random.default_rng(SEED + (831 if side == "left" else 832))
    axes = []
    for (top, bot, r_top, r_bot) in FAR_TRUNKS[side]:
        a = S(cam, top[0], top[1], ds + top[2])
        b = S(cam, bot[0], bot[1], ds + bot[2])
        b[2] = min(b[2], -0.4)                                   # the foot well below the island's floor
        n = 5
        pts, rad = [], []
        for i in range(n):
            t = i / (n - 1.0)
            p = a + (b - a) * t
            if 0 < i < n - 1:
                p = p + rng.normal(size=3) * np.array([0.12, 0.12, 0.05])
            pts.append(p)
            rad.append(r_top + (r_bot - r_top) * t ** 1.5)
        rad[-1] *= 1.25                                          # root flare
        sdf.tube(pts, rad, k=0.12)
        axes.append((a, b))
    for (c, r) in FAR_ROCKS[side]:
        p = S(cam, c[0], c[1], ds + c[2])
        sdf.ellipsoid(p, r, k=0.5)
        for _ in range(3):                                       # lumps on the boulder
            q = p + rng.uniform(-0.6, 0.6, 3) * np.asarray(r)
            rr = rng.uniform(0.35, 0.6) * min(r)
            sdf.ellipsoid(q, (rr * 1.2, rr, rr * 0.9), k=0.35)
    sdf.clip_plane((0, 0, -3.4), (0, 0, -1))
    # rock: broad dents and ridges; trunks: vertical bark ridges (anisotropic, stretched along z)
    sdf.noise(N3, 0.18, 0.7, octaves=3, offset=(1.7 + sign, 3.1, 0.5))
    sdf.noise(N3, 0.05, 2.2, octaves=2, offset=(4.1, 0.3 + sign, 7.7), aniso=(1.0, 1.0, 0.18))
    sdf.noise(N3, 0.025, 5.0, octaves=2, offset=(0.9, 6.6, 2.2 + sign), ridged=True)
    obj = sdf.to_object(f"far_finale_{side}", coll, mat, target_tris=FAR_TRIS, smooth_iter=2,
                        drop_cut_planes=[((0, 0, -3.4), (0, 0, -1))])
    return obj, sdf


# ------------------------------------------------------------------------------------------
# COLOR_0: R density, G length, B AO/cavity, A 1
# ------------------------------------------------------------------------------------------

def hills_masks(co, nrm, stone_sdf, bvh_vis):
    """Per-vertex masks of the hills: undercut-face weight (drop), inside / touching the stone, seen."""
    drop = C.smoothstep(-0.05, -0.22, co[:, 2] - floor_top(co[:, 1]))
    in_stone = stone_sdf.sample(co) < 0.004
    vis = visible_mask(co, nrm, bvh_vis)
    return drop, in_stone, vis


GAP_DARK = 0.5              # COLOR_0.B darkening on the gap floors of the pile (x AO): gaps ~0.3-0.5
GAP_B_MIN = 0.3             # seen moss never darker than this (no black specks)


def color_hills(obj, env, bvh, N3, cells, stone_sdf, bvh_vis, use_ao=True):
    """COLOR_0 of the hills. R: dense everywhere a pose sees moss (>= 0.6), a little thinner down in the
    gaps of the mound pile, sparse only down the undercut face below the lip and 0 under the stone. G: long
    on the mound crowns, short in the gaps. B: AO x gap darkening (gap floors ~0.3-0.5, open crowns 1)."""
    me = obj.data
    co, nrm, ed, _ = C.mesh_arrays(me)
    gap = pile_gap(env, co) * cushion_zone(co)                     # 0 mound tops ... 1 gap floor
    crease = C.smoothstep(0.1, 0.65, gap)
    crown = 1.0 - C.smoothstep(0.05, 0.35, gap)
    nb = np.abs(N3((co + np.asarray(KNOBS[0][2])) * KNOBS[0][0])) / knob_mean_abs(N3)
    knob_crease = 1.0 - C.smoothstep(0.08, 0.5, nb)                # small creases between the knobs
    ao = C.ray_ao(co, nrm, bvh, n_rays=16, max_dist=0.3) if use_ao else np.ones(len(co))
    ao = np.clip(ao / 0.92, 0.0, 1.0)
    big = N3.fbm(co * 1.1 + 31.0, 3)
    up = np.clip(nrm[:, 2], -1, 1)
    near_stone = 1.0 - C.smoothstep(0.38, 0.6, np.hypot((co[:, 0] - STONE_X) / 1.0, co[:, 1] / 0.55))
    # 0 on the island top / hills, 1 well down the sagging front and side faces (the dark drop)
    drop, in_stone, vis = hills_masks(co, nrm, stone_sdf, bvh_vis)
    R = 0.9 + 0.08 * crown - 0.2 * crease + 0.05 * big
    R = R * (1.0 - 0.9 * drop)
    R = np.where(~vis & (up < -0.3), R * 0.5, R)                   # undersides no pose sees: sparse
    R = np.maximum(R, near_stone * (1.0 - drop))
    seen_moss = vis & (drop < DROP_FRONT) & ~in_stone
    R = np.where(seen_moss, np.maximum(R, 0.62), R)
    R[in_stone] = 0.0                                              # under the stone's foot
    G = 0.5 + 0.3 * crown - 0.3 * crease + 0.06 * big + 0.1 * near_stone
    G = G * (1.0 - 0.5 * drop)
    # B: AO x gap darkening; the undercut lies in the island's own shadow (rays find nothing below it)
    B = ao ** 0.85 * (1.0 - GAP_DARK * crease) * (1.0 - 0.08 * knob_crease)
    B = np.maximum(B, crown * ao)                                  # open crowns stay 1
    B = np.where(seen_moss, np.maximum(B, GAP_B_MIN), B)
    B = B * (1.0 - 0.8 * drop)
    B = np.where(in_stone, B * 0.3, B)
    C.write_color0(me, np.clip(R, 0, 1), np.clip(G, 0.12, 0.92), np.clip(B, 0, 1))
    C.box_uvs(me)

    def q(x, m):
        return [round(float(np.percentile(x[m], p)), 3) for p in (10, 50, 90)] if m.any() else None

    gapm, side, crm = seen_moss & (crease > 0.8), seen_moss & (crease > 0.2) & (crease <= 0.8), \
        seen_moss & (crown > 0.8)
    steep = seen_moss & (nrm[:, 2] < 0.3)
    return dict(seen=int(vis.sum()), seen_moss=int(seen_moss.sum()),
                seen_moss_R_min=round(float(R[seen_moss].min()), 3) if seen_moss.any() else None,
                front_face=int((drop >= DROP_FRONT).sum()), in_stone=int(in_stone.sum()),
                B_p10_p50_p90=dict(crowns=q(B, crm), sides=q(B, side), gaps=q(B, gapm)),
                seen_moss_shares=dict(crowns=round(float(crm.sum() / seen_moss.sum()), 3),
                                      sides=round(float(side.sum() / seen_moss.sum()), 3),
                                      gaps=round(float(gapm.sum() / seen_moss.sum()), 3),
                                      normal_up_below_0_3=round(float(steep.sum() / seen_moss.sum()), 3)))


def color_simple(obj, bvh, N3, r_base, g_base, use_ao=True, ao_dist=0.5, noise_r=0.0, off=0.0, b_scale=1.0):
    me = obj.data
    co, nrm, ed, _ = C.mesh_arrays(me)
    ao = C.ray_ao(co, nrm, bvh, n_rays=10, max_dist=ao_dist) if use_ao else np.ones(len(co))
    cx = C.surface_convexity(co, nrm, ed, iterations=4)
    R = np.full(len(co), r_base) + noise_r * N3.fbm(co * 0.5 + off, 2)
    G = np.full(len(co), g_base)
    B = ao * (0.85 + 0.15 * np.clip(0.5 + cx / 0.02, 0, 1)) * b_scale
    C.write_color0(me, np.clip(R, 0, 1), np.clip(G, 0, 1), B)
    C.box_uvs(me)


def color_stone(obj, bvh, N3, frame, use_ao=True):
    me = obj.data
    co, nrm, ed, _ = C.mesh_arrays(me)
    ao = C.ray_ao(co, nrm, bvh, n_rays=14, max_dist=0.3) if use_ao else np.ones(len(co))
    cx = C.surface_convexity(co, nrm, ed, iterations=5)
    zr = co[:, 2] - VIS0
    crev_x = -0.205
    crevice = (1.0 - C.smoothstep(0.012, 0.05, np.abs(co[:, 0] - crev_x - 0.04 * zr))) * \
        C.smoothstep(-0.05, 0.05, zr) * (1.0 - C.smoothstep(0.42, 0.55, zr)) * (co[:, 1] < 0)
    foot = 1.0 - C.smoothstep(0.0, 0.08, zr)
    spots = C.smoothstep(0.35, 0.75, N3.fbm(co * 9.0 + 3.0, 2))
    R = np.clip(0.55 * crevice + 0.45 * foot + 0.12 * spots * C.smoothstep(0.5, -0.2, cx * 40), 0, 1)
    R[co[:, 2] < Z_B - 0.01] = 0.0
    G = 0.3 + 0.2 * crevice
    B = ao * (0.82 + 0.18 * np.clip(0.5 + cx / 0.006, 0, 1))
    C.write_color0(me, R, G, B)
    C.box_uvs(me)


# ------------------------------------------------------------------------------------------
# build
# ------------------------------------------------------------------------------------------

def build_cameras(coll):
    for name, c in CAMS.items():
        frame_h = 2.0 * c["focus"] * TV
        extras = dict(focus_distance_m=round(c["focus"], 3), frame_height_at_focus_m=round(frame_h, 3),
                      video_time_s=c["t"])
        if name == "cam_finale_main":
            extras["dolly_from_in"] = json.dumps(CAM_DOLLY)
        C.make_camera(name, coll, c["loc"], cam_q(c["pitch"]), focus=c["focus"], extras=extras)
    for name, k in KEYS.items():
        C.make_key_empty(name, coll, (STONE_X + 2.0, -2.2, 3.4), k["dir"])


def build(use_ao=True):
    t0 = time.time()
    scene = C.ensure_scene(SCENE_NAME)
    coll = C.ensure_collection(scene, COLL_NAME)
    C.purge_collection(coll)
    solve_cameras()
    for n, c in CAMS.items():
        C.log(f"{n}: loc {tuple(round(x, 3) for x in c['loc'])} pitch {c['pitch']} focus {c['focus']:.3f} m "
              f"(frame height {2 * c['focus'] * TV:.3f} m)")
    mat_moss = C.preview_material("mat_moss", C.PALETTE["moss"], 0.95)
    mat_stone = C.preview_material("mat_stone", C.PALETTE["stone"], 0.85)
    mat_far = C.preview_material("mat_far", "#0E1214", 1.0)
    N3 = C.Noise3(SEED + 900)
    build_cameras(coll)
    frame = emblem_frame()

    t = time.time()
    stone, ssdf = build_stone(coll, mat_stone, N3, frame)
    C.log(f"stone field + mesh: {C.tri_count(stone)} tris ({time.time() - t:.1f}s)")
    t = time.time()
    cut = cut_emblem(stone, coll, frame, mat_stone)
    C.log(f"emblem cut: {cut} ({time.time() - t:.1f}s)")
    t = time.time()
    hills, hsdf, env, mounds, cells, hinfo = build_hills(coll, mat_moss, N3)
    C.log(f"moss_finale_hills: {C.tri_count(hills)} tris ({time.time() - t:.1f}s)")
    fg, fsdf = build_near_fg(coll, mat_moss, N3)
    C.log(f"moss_finale_near__fg: {C.tri_count(fg)} tris")
    fars = []
    for side in ("left", "right"):
        t = time.time()
        o, _ = build_far(coll, mat_far, N3, side)
        fars.append(o)
        C.log(f"{o.name}: {C.tri_count(o)} tris ({time.time() - t:.1f}s)")

    t = time.time()
    bvh = C.build_bvh([stone, hills, fg]) if use_ao else None
    bvh_vis = C.build_bvh([stone, hills])
    color_stone(stone, bvh, N3, frame, use_ao)
    hcol = color_hills(hills, env, bvh, N3, cells, ssdf, bvh_vis, use_ao)
    C.log(f"hills COLOR_0: {hcol}")
    color_simple(fg, bvh, N3, 0.2, 0.4, use_ao, 0.5, 0.1, 11.0, b_scale=0.18)     # dark, sparse
    bvh_far = C.build_bvh(fars) if use_ao else None
    for o in fars:
        color_simple(o, bvh_far, N3, 0.15, 0.3, use_ao, 2.5, 0.25, 21.0)
    C.log(f"colours ({time.time() - t:.1f}s)")
    # portrait screens: the near mound's bank, coloured like the mound (its AO sees everything around it; the
    # hills, the stone and the mound keep the colours above), then merged into moss_finale_near__fg
    t = time.time()
    poses = engine_track()
    bank, binfo = build_near_bank(coll, mat_moss, N3, poses)
    bvh_bank = C.build_bvh([stone, hills, fg, bank]) if use_ao else None
    color_simple(bank, bvh_bank, N3, 0.2, 0.4, use_ao, BANK_AO_DIST, 0.1, 11.0, b_scale=0.18)
    bank_density_off_under_island(bank)
    binfo["shadowed_vertices"] = bank_island_shadow(bank, C.build_bvh([stone, hills]))
    binfo["landscape"] = bank_in_landscape([stone, hills, fg] + fars, bank, poses)
    fg, part = merge_into(fg, bank)
    binfo.update(part)
    binfo["near_scatter"] = near_scatter_weight(fg, part["first_vertex"])
    C.log(f"near bank: {binfo} ({time.time() - t:.1f}s)")
    total = sum(C.tri_count(o) for o in coll.objects if o.type == "MESH")
    C.log(f"finale built: {total} tris, {time.time() - t0:.1f}s")
    return scene, coll, dict(cut=cut, frame=frame, hsdf=hsdf, ssdf=ssdf, env=env, mounds=mounds, cells=cells,
                             hills=hinfo, hills_color=hcol, bank=binfo)


# ------------------------------------------------------------------------------------------
# validation extras
# ------------------------------------------------------------------------------------------

def validate_finale(rep, coll, info):
    errs, out = [], {}
    names = {n["name"] for n in rep["nodes"]}
    for req in ("moss_finale_hills", "moss_finale_near__fg", "stone_finale", "far_finale_left",
                "far_finale_right", "cam_finale_in", "cam_finale_main", "cam_finale_out", "key_finale"):
        if req not in names:
            errs.append(f"missing node {req}")
    stone = coll.objects["stone_finale"]
    co, nrm, _, _ = C.mesh_arrays(stone.data)
    frame = info["frame"]
    # holes: rays along the stone normal through each piece's centre must pass (no geometry)
    bvh = C.build_bvh([stone])
    holes = {}
    for piece in SL.PIECES:
        c = SL.piece_centre(piece)
        p = frame.world(c[None], np.array([0.2]))[0]
        hit = bvh.ray_cast(Vector(p), Vector(-frame.n), 0.6)
        holes[piece] = hit[0] is None
        if hit[0] is not None:
            errs.append(f"emblem hole '{piece}' is blocked at {tuple(round(x, 3) for x in hit[0])}")
    # channel bridge: a ray through the channel centre must hit stone
    p = frame.world(np.array([SL.SEED_CH_P]), np.array([0.2]))[0]
    hit = bvh.ray_cast(Vector(p), Vector(-frame.n), 0.6)
    if hit[0] is None:
        errs.append("channel bridge missing")
    # contact: the stone foot must be inside the moss all around (no floating)
    # (exception: where the frame-19 moss line dips beside the stone, the crest clamp leaves the stone free
    # down to that line in cam_finale_main - stone the frame shows; counted apart, never as a gap)
    hs = info["hsdf"]
    foot = co[(co[:, 2] < Z_B - 0.005) & (co[:, 2] > Z_B - SINK + 0.01)]
    inside = hs.sample(foot) < 0.0 if len(foot) else np.zeros(0, bool)
    fu, fv, _fz = project("cam_finale_main", foot) if len(foot) else (np.zeros(0),) * 3
    line_dv = fv - upper_clamp_v(fu)                     # < 0: above the clamp line (the frame shows stone)
    by_crest = (~inside) & (line_dv < 0.0)
    near_line = (~inside) & ~by_crest & (line_dv < 0.004)
    share = float(inside.mean()) if len(inside) else 0.0
    share_ok = float((inside | by_crest).mean()) if len(inside) else 0.0
    out["foot_outside"] = dict(n_band=int(len(foot)), outside=int((~inside).sum()), above_crest_line=int(by_crest.sum()),
                               within_0_004_fh_under_line=int(near_line.sum()),
                               other=int((~inside & ~by_crest & ~near_line).sum()),
                               deepest_outside_z=round(float(foot[~inside][:, 2].min()), 3) if (~inside).any() else None)
    if share_ok < 0.97:
        errs.append(f"stone foot only {share_ok:.3f} inside the moss")
    if share_ok < 0.999:
        errs.append(f"stone foot only {share_ok:.4f} inside the moss or above the crest line (want >= 0.999)")
    bad = foot[~inside]
    out["foot_outside_examples"] = np.round(bad[:: max(1, len(bad) // 8)][:8], 3).tolist()
    out.update(holes_open=holes, foot_inside_moss=round(share, 4), foot_inside_or_above_line=round(share_ok, 4), stone_min_z=round(float(co[:, 2].min()), 3),
               stone_dims=[round(float(x), 3) for x in (co.max(0) - co.min(0))],
               emblem_height_m=EMBLEM_H, emblem_share_of_stone=round(EMBLEM_H / (co[:, 2].max() - co[:, 2].min()), 3))
    # moss density: every hills vertex a pose sees carries fuzz (R >= 0.6), except down the undercut face
    # below the lip and under the stone
    hills = coll.objects["moss_finale_hills"]
    hco, hnrm, _, _ = C.mesh_arrays(hills.data)
    col = C.read_color0(hills.data)
    bvh_vis = C.build_bvh([stone, hills])
    drop, in_stone, vis = hills_masks(hco, hnrm, info["ssdf"], bvh_vis)
    seen = vis & (drop < DROP_FRONT) & ~in_stone
    low = seen & (col[:, 0] < 0.6)
    if low.any():
        errs.append(f"{int(low.sum())} seen moss vertices with COLOR_0.R < 0.6")
    front_seen = vis & (drop >= DROP_FRONT)
    # pits: closing of the final top surface with a flat 12 cm disc (the probe that found the 16 cm pit
    # of the first build); only columns whose top a pose sees
    ztop, has = SL.top_heightfield(hs)
    depth = np.where(has, SL.ball_closing(ztop, 0.12, hs.voxel, flat=True) - ztop, 0.0)
    ii, jj = np.nonzero(depth > 0.05)
    P = np.stack([hs.origin[0] + ii * hs.voxel, hs.origin[1] + jj * hs.voxel, ztop[ii, jj] + 0.01], 1)
    seen_col = visible_mask(P, np.tile([0.0, 0.0, 1.0], (len(P), 1)), bvh_vis) if len(P) else np.zeros(0, bool)
    dseen = depth[ii, jj][seen_col]
    # where the deepest seen ones are (main pose px, depth m; one per 15 cm)
    Ps = P[seen_col]
    picks = []
    for k in np.argsort(-dseen, kind="stable"):
        if len(picks) >= 6 or dseen[k] < 0.08:
            break
        if all(np.hypot(Ps[k, 0] - Ps[j, 0], Ps[k, 1] - Ps[j, 1]) >= 0.15 for j in picks):
            picks.append(int(k))
    cm = CAMS["cam_finale_main"]
    pits_px = []
    for k in picks:
        u, v, _z = C.project_to_screen(cm["loc"], cam_q(cm["pitch"]), Ps[k][None])
        pits_px.append([round(float(u[0]) * C.FRAME_W), round(float(v[0]) * C.FRAME_H), round(float(dseen[k]), 3)])
    out["pits_seen_deepest_main_px"] = pits_px
    # pools (the pit metric for the mound pile, whose creases the flat 12 cm closing above counts as well):
    # depth of the water that would stand on the top height map (priority flood over the island top), on
    # the columns a pose sees
    nx, ny = ztop.shape
    X = hs.origin[0] + np.arange(nx)[:, None] * hs.voxel + np.zeros((1, ny))
    Y = hs.origin[1] + np.arange(ny)[None, :] * hs.voxel + np.zeros((nx, 1))
    valid = has & (cushion_zone(np.stack([X, Y, ztop], -1).reshape(-1, 3)).reshape(nx, ny) > 0.5)
    pool = SL.depression_depth(np.where(has, ztop, -1.0), valid)
    ii, jj = np.nonzero(pool > 0.01)
    P = np.stack([hs.origin[0] + ii * hs.voxel, hs.origin[1] + jj * hs.voxel, ztop[ii, jj] + 0.01], 1)
    pseen = visible_mask(P, np.tile([0.0, 0.0, 1.0], (len(P), 1)), bvh_vis) if len(P) else np.zeros(0, bool)
    pd = pool[ii, jj][pseen]
    out.update(pools_seen_over_2cm=int((pd > 0.02).sum()), pools_seen_over_4cm=int((pd > 0.04).sum()),
               pools_seen_over_6cm=int((pd > 0.06).sum()), pools_seen_max_m=round(float(pd.max()), 3) if len(pd)
               else 0.0, pools_all_max_m=round(float(pool.max()), 3))
    if "mounds" in info:
        out["mounds"] = mound_stats(info["mounds"], bvh_vis)
    out.update(hills_seen_vertices=int(seen.sum()), hills_seen_R_min=round(float(col[seen, 0].min()), 3),
               hills_seen_R_mean=round(float(col[seen, 0].mean()), 3), hills_seen_lowR=int(low.sum()),
               hills_front_face_seen=int(front_seen.sum()),
               hills_front_face_R_mean=round(float(col[front_seen, 0].mean()), 3) if front_seen.any() else None,
               hills_seen_B_range=[round(float(col[seen, 2].min()), 3), round(float(col[seen, 2].max()), 3)],
               pits_seen_over_5cm=int(len(dseen)), pits_seen_over_8cm=int((dseen > 0.08).sum()),
               pits_seen_over_10cm=int((dseen > 0.10).sum()),
               pits_seen_max_m=round(float(dseen.max()), 3) if len(dseen) else 0.0,
               hills_build=info.get("hills"))
    # portrait screens (QA R4): moss to the bottom edge of a 9:19.5 frame at 60 deg and at the engine's fov;
    # the bank never in front of other geometry in a 1440x1020 frame of the track
    out["portrait"] = portrait_coverage(coll)
    for key, r in out["portrait"].items():
        if r["holes_share"] > 0.002 or r["lowest_row_moss_share"] < 0.99:
            errs.append(f"portrait {key}: moss does not fill the frame below the crest: {r}")
    bank = info.get("bank")
    if bank is not None:
        out["near_bank"] = bank
        if bank["landscape"]["bank_px_over_geometry"]:
            errs.append(f"near bank shows in front of other geometry in a 1440x1020 frame: {bank['landscape']}")
        if bank["near_scatter"]["bank_share"] > 0.001:
            errs.append(f"near bank takes plants from the near mound's scatter budget: {bank['near_scatter']}")
    return dict(errors=errs, **out)


def mound_stats(M, bvh):
    """Mound count, footprint histogram (diameter 2a and mean diameter a + b, cm; all / touch point seen from
    cam_finale_main), dome heights."""
    cam = CAMS["cam_finale_main"]
    loc = np.asarray(cam["loc"], float)
    u, v, z = C.project_to_screen(loc, cam_q(cam["pitch"]), M.p)
    seen = (u > 0) & (u < 1) & (v > 0) & (v < 1) & (z > 0.1)
    org = M.p + np.array([0.0, 0.0, 0.03])
    for i in np.nonzero(seen)[0]:
        dvec = loc - org[i]
        dist = float(np.linalg.norm(dvec))
        if bvh.ray_cast(Vector(org[i]), Vector(dvec / dist), dist)[0] is not None:
            seen[i] = False
    bins = [20, 25, 30, 35, 40, 45, 50, 55, 60]
    d2a = 200.0 * M.a
    dm = 100.0 * (M.a + M.b)

    def hist(x):
        h, _ = np.histogram(x, bins=bins)
        return {f"{bins[i]}-{bins[i + 1]}": int(h[i]) for i in range(len(h))}

    return dict(count=len(M), seen_main=int(seen.sum()),
                diam_2a_cm=dict(all=hist(d2a), seen_main=hist(d2a[seen])),
                diam_mean_cm=dict(all=hist(dm), seen_main=hist(dm[seen])),
                height_cm_p10_p50_p90=[round(float(np.percentile(100 * M.h, p)), 1) for p in (10, 50, 90)])


# ------------------------------------------------------------------------------------------
# previews and comparison sheets
# ------------------------------------------------------------------------------------------

def preview_mats():
    return {
        "moss_": SL.preview_ao_material("prev_finale_moss", "#5E7A26", 0.95),
        "stone_": SL.preview_ao_material("prev_finale_stone", "#ABA391", 0.85),
        "far_": SL.preview_ao_material("prev_finale_far", "#151B1E", 1.0),
    }


def render_previews(scene, coll, cams=None, percent=50, suffix=""):
    key = KEYS["key_finale"]["dir"]
    SL.setup_lit_preview(scene, key, BG_HEX, percent=percent, sun_strength=4.2, sun_color=(1.0, 0.95, 0.86),
                         fill_dir=(0.5, 0.75, -0.3), fill_strength=0.2, world_strength=1.0)
    out = []
    try:
        with SL.PreviewMaterials(coll, preview_mats()):
            for name in (cams or list(CAMS.keys())):
                path = os.path.join(CAPTURES, f"finale_{name}{suffix}.png")
                C.render_camera(scene, coll.objects[name], path)
                out.append(path)
    finally:
        SL.remove_preview_lights(scene)
    return out


def reference_stone_mask(rgb, gain=1.0):
    """Stone silhouette in a finale frame: light, low-saturation pixels in the central column."""
    a = np.clip(rgb[..., :3] * gain, 0, 1) * 255.0
    H, W = a.shape[:2]
    lum = a @ np.array([0.2126, 0.7152, 0.0722])
    sat = a.max(-1) - a.min(-1)
    m = (lum > 70) & (sat < 52)
    m[:, :int(0.36 * W)] = False
    m[:, int(0.64 * W):] = False
    m[:int(0.12 * H)] = False
    m[int(0.62 * H):] = False
    acc = np.zeros(m.shape, np.float32)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            acc += np.roll(np.roll(m, dy, 0), dx, 1)
    return acc > 12.5


def reference_crest(rgb, gain=1.0, v0=0.4):
    """Top of the green moss per column (frame fractions) or None."""
    a = np.clip(rgb[..., :3] * gain, 0, 1) * 255.0
    H, W = a.shape[:2]
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    m = (g > r * 1.05) & (g > b * 1.25) & (g > 45)
    out = []
    for x in range(0, W, max(W // 120, 1)):
        col = np.nonzero(m[int(v0 * H):, x])[0]
        out.append((x / W, (col[0] + int(v0 * H)) / H if len(col) else None))
    return out


def compare_sheets(scene, coll, percent=50):
    os.makedirs(TMP, exist_ok=True)
    out, stats = [], {}
    rows = (("main", "cam_finale_main", "frames/19_final_landscape.png", 1.0),
            ("in", "cam_finale_in", "frames/18_final_entry.png", 3.2),
            ("out", "cam_finale_out", "frames/20_footer.png", 1.0))
    for key, cam, ref_rel, gain in rows:
        ref = SL.load_rgba(os.path.join(REF, ref_rel))
        prev_p = os.path.join(CAPTURES, f"finale_{cam}.png")
        prev = SL.load_rgba(prev_p) if os.path.exists(prev_p) else np.zeros_like(ref)
        h, w = prev.shape[:2]
        refs = SL.resize_to(ref, h, w)
        occ = [coll.objects["moss_finale_hills"], coll.objects["moss_finale_near__fg"]]
        mstone = SL.load_rgba(SL.render_mask(scene, coll.objects[cam], os.path.join(TMP, f"m_stone_{key}.png"),
                                             [coll.objects["stone_finale"]], percent, occluders=occ))[..., 0] > 0.5
        mhill = SL.load_rgba(SL.render_mask(scene, coll.objects[cam], os.path.join(TMP, f"m_hill_{key}.png"),
                                            [coll.objects["moss_finale_hills"]], percent,
                                            occluders=[coll.objects["stone_finale"]]))[..., 0] > 0.5
        rstone = reference_stone_mask(refs, gain)
        ov = np.clip(refs[..., :3] * gain, 0, 1) * 0.6
        ov[SL.outline(rstone)] = (0.2, 0.95, 1.0)
        ov[SL.outline(mstone)] = (1.0, 0.92, 0.1)
        # our moss/stone top line vs the reference crest
        hh = mhill.copy()
        top_ours = []
        for x in range(0, w, max(w // 120, 1)):
            col = np.nonzero(hh[int(0.4 * h):, x])[0]
            top_ours.append((x / w, (col[0] + int(0.4 * h)) / h if len(col) else None))
        for (u, v) in reference_crest(refs, gain):
            if v is not None:
                SL.draw_dot(ov, u, v, 1.6, (0.2, 0.95, 1.0))
        for (u, v) in top_ours:
            if v is not None:
                SL.draw_dot(ov, u, v, 1.2, (1.0, 0.92, 0.1))
        SL.draw_grid(ov)
        SL.label(ov, f"{key} ref cyan / ours yellow", 6, 6)
        path = os.path.join(CAPTURES, f"finale_cmp_{key}.png")
        SL.save_rgb(np.concatenate([np.clip(refs[..., :3] * min(gain, 2.0), 0, 1), prev[..., :3], ov], 1), path)
        out.append(path)
        # stats: stone box
        def box(m):
            ys, xs = np.nonzero(m)
            if not len(xs):
                return None
            return [round(xs.min() / w, 4), round(xs.max() / w, 4), round(ys.min() / h, 4), round(ys.max() / h, 4)]
        inter = (rstone & mstone).sum()
        union = (rstone | mstone).sum()
        stats[key] = dict(stone_box_ref=box(rstone), stone_box_ours=box(mstone),
                          stone_iou=round(float(inter) / max(float(union), 1.0), 4))
    # full-size stone crop next to frame 19 (details/10 = frame 19 [560, 260, 870, 585])
    pfull = os.path.join(TMP, "prev_main_full.png")
    key = KEYS["key_finale"]["dir"]
    SL.setup_lit_preview(scene, key, BG_HEX, percent=100, sun_strength=4.2, sun_color=(1.0, 0.95, 0.86),
                         fill_dir=(0.5, 0.75, -0.3), fill_strength=0.2)
    try:
        with SL.PreviewMaterials(coll, preview_mats()):
            C.render_camera(scene, coll.objects["cam_finale_main"], pfull)
    finally:
        SL.remove_preview_lights(scene)
    full = SL.load_rgba(pfull)
    r19 = SL.load_rgba(os.path.join(REF, "frames/19_final_landscape.png"))
    x0, y0, x1, y1 = 520, 230, 910, 600
    path = os.path.join(CAPTURES, "finale_cmp_stone.png")
    SL.save_rgb(np.concatenate([r19[y0:y1, x0:x1, :3], full[y0:y1, x0:x1, :3]], 1), path)
    out.append(path)
    out.append(emblem_sheet(coll))
    out.append(far_sheet(scene, coll, percent))
    # crest line per pixel column (full resolution) in the three poses; main against the reference crest
    crests = {key: hill_crest(scene, coll, cam) for key, cam in (("main", "cam_finale_main"),
                                                                 ("in", "cam_finale_in"), ("out", "cam_finale_out"))}
    with open(os.path.join(TMP, "crest_new.json"), "w") as f:
        json.dump({k: [None if np.isnan(x) else float(x) for x in v] for k, v in crests.items()}, f)
    stats["crest_main"] = crest_stats(crests["main"])
    stats["crest_main_vs_r2_polyline"] = crest_stats(crests["main"], crest_v_r2)
    out.append(crest_sheet(full, r19, crests["main"]))
    path, stats["color0_main"] = color0_sheet(scene, coll)
    out.append(path)
    C.log("compare:", json.dumps(stats))
    return out, stats


def far_sheet(scene, coll, percent=50):
    """finale_cmp_far: the far forms' outline (orange; hills, stone and near mound occlude) over the
    brightened reference frames 18 / 19 / 20, and the brightened frame alone."""
    rows = (("cam_finale_in", "frames/18_final_entry.png", 6.0), ("cam_finale_main", "frames/19_final_landscape.png", 4.0),
            ("cam_finale_out", "frames/20_footer.png", 4.0))
    fars = [coll.objects["far_finale_left"], coll.objects["far_finale_right"]]
    occ = [coll.objects[n] for n in ("moss_finale_hills", "stone_finale", "moss_finale_near__fg")]
    panels = []
    for cam, ref_rel, gain in rows:
        m = SL.load_rgba(SL.render_mask(scene, coll.objects[cam], os.path.join(TMP, f"m_far_{cam}.png"), fars,
                                        percent, occluders=occ))[..., 0] > 0.5
        h, w = m.shape
        ref = SL.resize_to(SL.load_rgba(os.path.join(REF, ref_rel)), h, w)[..., :3]
        bright = np.clip(np.power(np.clip(ref, 0, 1), 0.6) * gain * 0.45, 0, 1)
        ov = bright.copy()
        ov[SL.outline(m, 2)] = (1.0, 0.55, 0.1)
        SL.label(ov, f"{cam[11:]}: far forms orange", 6, 6)
        panels.append(np.concatenate([bright, ov], 1))
    img = np.concatenate(panels, 0)
    path = os.path.join(CAPTURES, "finale_cmp_far.png")
    SL.save_rgb(img, path)
    return path


def hill_crest(scene, coll, cam):
    """Top of the visible moss per pixel column in `cam` (the stone occludes), frame fractions (NaN: none)."""
    p = SL.render_mask(scene, coll.objects[cam], os.path.join(TMP, f"crest_{cam}.png"),
                       [coll.objects["moss_finale_hills"]], 100, occluders=[coll.objects["stone_finale"]])
    m = SL.load_rgba(p)[..., 0] > 0.5
    h, w = m.shape
    v0 = int(0.3 * h)
    top = np.full(w, np.nan)
    for x in range(w):
        col = np.nonzero(m[v0:, x])[0]
        if len(col):
            top[x] = (col[0] + v0) / h
    return top


def _box_mean(a, n):
    """Moving average (window n columns) ignoring NaN."""
    ok = ~np.isnan(a)
    k = np.ones(n)
    s = np.convolve(np.where(ok, a, 0.0), k, mode="same")
    c = np.convolve(ok.astype(float), k, mode="same")
    return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def crest_stats(top, ref_fn=None):
    """Our crest (main pose) against the measured frame-19 crest (MOSS_CREST_FINE, or ref_fn): signed mean
    (+ = lower), mean and p95 |diff|, the largest |diff| of the 0.04 W moving averages of both lines (the
    overall line), lump amplitude (crest minus its moving average) - all in frame heights."""
    w = len(top)
    u = (np.arange(w) + 0.5) / w
    ref = np.asarray((ref_fn or crest_v)(u), float)
    sel = (u > 0.01) & (u < 0.99) & ~np.isnan(top)
    d = top - ref
    lp = _box_mean(top, int(0.04 * w))
    lpr = _box_mean(ref, int(0.04 * w))
    lump = top - lp
    return dict(mean=round(float(d[sel].mean()), 4), mean_abs=round(float(np.abs(d[sel]).mean()), 4),
                p95_abs=round(float(np.percentile(np.abs(d[sel]), 95)), 4),
                lowpass_max_abs=round(float(np.nanmax(np.abs((lp - lpr)[sel]))), 4),
                lowpass_over_0_01=round(float((np.abs(lp - lpr)[sel] > 0.01).mean()), 4),
                lump_rms=round(float(np.sqrt(np.nanmean(lump[sel] ** 2))), 4),
                lump_up_max=round(float(-np.nanmin(lump[sel])), 4))


def crest_sheet(full, r19, top):
    """finale_cmp_crest: frame 19 against our full-size main preview on the left hills (detail 09 = frame 19
    [0:605, 515:755]) and on the right hills; third row: ours with the measured crest (cyan) and our crest
    line (yellow)."""
    regions = [(0, 515, 605, 755), (835, 480, 1440, 720)]
    cols = []
    for (x0, y0, x1, y1) in regions:
        ref = r19[y0:y1, x0:x1, :3]
        ours = full[y0:y1, x0:x1, :3]
        ov = ours * 0.75
        W, H = x1 - x0, y1 - y0
        xs = np.arange(x0, x1)
        uu = (xs + 0.5) / C.FRAME_W
        SL.draw_polyline(ov, np.stack([(xs - x0) / W, (np.array([crest_v(x) for x in uu]) * C.FRAME_H - y0) / H], 1),
                         (0.2, 0.95, 1.0), width=0.8)
        for x in range(x0, x1, 2):
            if not np.isnan(top[x]):
                SL.draw_dot(ov, (x - x0) / W, (top[x] * C.FRAME_H - y0) / H, 1.0, (1.0, 0.92, 0.1))
        col = np.concatenate([ref, ours, ov], 0)
        cols.append(SL.resize_to(col, int(col.shape[0] * 1.5), int(col.shape[1] * 1.5)))
    img = np.concatenate(cols, 1)
    SL.label(img, "frame 19 / ours / crest: measured cyan, ours yellow", 6, 6)
    path = os.path.join(CAPTURES, "finale_cmp_crest.png")
    SL.save_rgb(img, path)
    return path


def color0_sheet(scene, coll, percent=50):
    """finale_cmp_color0: COLOR_0.R (fuzz density; red where < 0.6) and COLOR_0.B (AO) of the moss as
    cam_finale_main sees it (the stone black, the near mound and far forms hidden)."""
    hills, stone = coll.objects["moss_finale_hills"], coll.objects["stone_finale"]
    sh = scene.display.shading
    prev = (scene.render.engine, scene.render.film_transparent, sh.light, sh.color_type,
            scene.render.image_settings.color_mode, scene.render.resolution_percentage,
            scene.view_settings.view_transform)
    hidden = [o for o in coll.objects if o.type == "MESH" and o not in (hills, stone) and not o.hide_render]
    panels = []
    try:
        for o in hidden:
            o.hide_render = True
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.render.film_transparent = True
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.resolution_percentage = percent
        scene.view_settings.view_transform = "Standard"
        sh.light = "FLAT"
        sh.color_type = "VERTEX"
        for ch in (0, 2):
            for o in (hills, stone):
                me = o.data
                src = C.read_color0(me)
                val = src[:, ch] if o is hills else np.zeros(len(src))
                a = me.color_attributes.new("silva_diag", "FLOAT_COLOR", "POINT")
                a.data.foreach_set("color", np.stack([val, val, val, np.ones(len(val))], 1).astype(np.float32).ravel())
                me.color_attributes.active_color = a
            p = os.path.join(TMP, f"color0_{ch}.png")
            C.render_camera(scene, coll.objects["cam_finale_main"], p)
            img = SL.load_rgba(p)
            lin = C.srgb_to_linear(img[..., 0])
            alpha = img[..., 3] > 0.99
            rgb = np.full(img.shape[:2] + (3,), 0.12)
            g = np.clip(lin, 0, 1)
            rgb[alpha] = np.stack([g, g, g], -1)[alpha]
            if ch == 0:
                low = alpha & (lin < 0.6)
                rgb[low] = np.stack([0.4 + 0.6 * g, 0.1 * g, 0.1 * g], -1)[low]
            panels.append(rgb)
            for o in (hills, stone):
                me = o.data
                me.color_attributes.remove(me.color_attributes["silva_diag"])
                idx = me.color_attributes.find(C.COLOR_ATTR)
                me.color_attributes.active_color_index = idx
                me.color_attributes.render_color_index = idx
    finally:
        for o in hidden:
            o.hide_render = False
        (scene.render.engine, scene.render.film_transparent, sh.light, sh.color_type,
         scene.render.image_settings.color_mode, scene.render.resolution_percentage,
         scene.view_settings.view_transform) = prev
    # visible gap share: moss pixels (R >= 0.6, i.e. not the undercut front face) with B < 0.5 / < 0.4
    rch, bch = panels[0][..., 1], panels[1][..., 0]
    rv = np.where(panels[0][..., 0] > panels[0][..., 1] + 0.05, 0.0, rch)     # red-marked = R < 0.6
    moss = rv >= 0.6
    gstats = dict(moss_px=int(moss.sum()),
                  B_below_0_5=round(float((moss & (bch < 0.5)).sum() / max(moss.sum(), 1)), 4),
                  B_below_0_4=round(float((moss & (bch < 0.4)).sum() / max(moss.sum(), 1)), 4),
                  B_p10_p50_p90=[round(float(np.percentile(bch[moss], p)), 3) for p in (10, 50, 90)]
                  if moss.any() else None)
    img = np.concatenate(panels, 1)
    SL.label(img, "main: color0 r (red below 0.6) / color0 b", 6, 6)
    path = os.path.join(CAPTURES, "finale_cmp_color0.png")
    SL.save_rgb(img, path)
    return path, gstats


def emblem_sheet(coll):
    """finale_cmp_emblem: the cut-out walls of stone_finale (vertices near the front plane, in seed units)
    over the outlines parsed independently from seed.svg (cyan) and our rounded construction (yellow)."""
    frame = emblem_frame()
    stone = coll.objects["stone_finale"]
    co, _, _, _ = C.mesh_arrays(stone.data)
    q, hgt = frame.local(co)
    sel = (np.abs(hgt + HALF_T) < HALF_T - 0.002) & (np.abs(q[:, 0]) < 34) & (np.abs(q[:, 1]) < 56)
    S_ = 6
    W, H = 70 * S_, 120 * S_
    img = np.full((H, W, 3), 0.08)

    def to_px(P):
        return (P[:, 0] + 35) / 70.0, (60 - P[:, 1]) / 120.0

    for P in SL.svg_seed_paths(SVG):
        u, v = to_px(P)
        SL.draw_polyline(img, np.stack([u, v], 1), (0.2, 0.95, 1.0), closed=True, width=1.5)
    for piece in SL.PIECES:
        P = SL.seed_outline(piece, 0.0, step=0.2)
        u, v = to_px(P)
        SL.draw_polyline(img, np.stack([u, v], 1), (1.0, 0.9, 0.1), closed=True, width=0.6)
    u, v = to_px(q[sel])
    for a, b in zip(u, v):
        SL.draw_dot(img, a, b, 0.8, (1.0, 0.3, 0.3))
    SL.label(img, "svg cyan  ours yellow  mesh red", 6, 6, scale=2)
    path = os.path.join(CAPTURES, "finale_cmp_emblem.png")
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
                  cameras={k: dict(loc=[round(x, 3) for x in v["loc"]], focus=round(v["focus"], 3))
                           for k, v in CAMS.items()})
    glb = os.path.abspath(a["glb"]) if a["glb"] else GLB_PATH
    if a["export"]:
        C.export_glb(scene, glb)
        result["glb"] = glb
        result["glb_bytes"] = os.path.getsize(glb)
    if a["validate"]:
        rep = C.validate_glb(glb)
        extra = validate_finale(rep, coll, info)
        rep["finale"] = extra
        result["validate"] = dict(errors=rep["errors"] + extra["errors"], warnings=rep["warnings"],
                                  totals=rep["totals"], finale=extra)
        os.makedirs(CAPTURES, exist_ok=True)
        with open(os.path.join(CAPTURES, "finale_validate.json"), "w") as f:
            json.dump(rep, f, indent=1, default=str)
    if a["render"] is not None or a["compare"]:
        result["renders"] = render_previews(scene, coll, a["render"] or None, a["percent"])
    if a["compare"]:
        result["compare"], result["compare_stats"] = compare_sheets(scene, coll, a["percent"])
        if not _PORTRAIT_IDS:
            portrait_coverage(coll)
        result["compare"].append(portrait_sheet(scene, coll))
    if a["save"]:
        save_blend()
    C.log("result", json.dumps(result, default=str))
    return result


if __name__ == "__main__":
    _argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    main(_argv)
