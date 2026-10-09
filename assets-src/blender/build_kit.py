"""build_kit.py - instanced vegetation kit for Silva: kit.glb + kit_basecolor.webp.

Run from the repo root (headless):
  "/c/Program Files/Blender Foundation/Blender 5.1/blender.exe" --background --factory-startup \
      --python assets-src/blender/build_kit.py [-- --no-render]

Outputs:
  public/nature/models/kit.glb             16 small meshes, one node each, at the origin
  public/nature/textures/kit_basecolor.webp 1024^2 sRGB atlas (veins, midribs, tone)
  docs/captures/blender/kit_preview.png     labelled preview grid (rendered from the
                                            re-imported GLB, i.e. what actually ships)

Contract (CLAUDE.md "Asset contract", kit meshes):
  * metres, real plant size; origin = attachment point; Blender +Z = growth axis
    (exported Y-up); every node sits at the world origin with identity transform
  * COLOR_0 (linear RGBA; authored as a float point attribute, the glTF exporter
    writes it as normalized unsigned short VEC4): R = wind flex weight = path length
    from the root along stem/midrib, normalised so the farthest tip = 1 (root = 0; it
    is linear, shape it in the shader, e.g. R*R); G = per-vertex shade (darker towards
    the base and in the cup of a blade, ~0.66..1); B = 1 on leaf blades, petals and
    the flower centres, 0 on stems, twigs and seed beads; A = 1
  * TEXCOORD_0 into the shared atlas (glTF convention, load with flipY = false).
    Atlas = 4 x 4 cells of 256 px, see CELLS (row 0 = bottom of the picture)
  * blades are single-sided sheets (front = upper side): render with
    side = DoubleSide (the exporter already marks mat_kit doubleSided); silhouettes
    are geometry, the atlas has no alpha
  * one placeholder material "mat_kit" on every node; no cameras, lights, animation
Deterministic: fixed seeds only.
"""

import os
import sys
import math
import json
import struct
import time

import numpy as np

sys.dont_write_bytecode = True          # keep __pycache__ out of assets-src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proclib as pl  # noqa: E402

import bpy  # noqa: E402
from mathutils import Vector, Matrix  # noqa: E402
from mathutils import geometry as mgeo  # noqa: E402

CM = 0.01
MM = 0.001
ATLAS_N = 1024
GRID = 4                      # 4 x 4 cells of 256 px
GLB_PATH = os.path.join(pl.MODEL_DIR, "kit.glb")
ATLAS_PATH = os.path.join(pl.TEX_DIR, "kit_basecolor.webp")


def log(*a):
    print("[kit]", *a, flush=True)


# ============================================================================
# atlas layout
# ============================================================================
# (col, row); row 0 is the BOTTOM of the image (UV v = 0), as in Blender.
CELLS = {
    "round_a": (0, 3), "round_b": (1, 3), "round_c": (2, 3), "serrated": (3, 3),
    "clover": (0, 2), "canopy_a": (1, 2), "canopy_b": (2, 2), "canopy_c": (3, 2),
    "sprig": (0, 1), "petal_white": (1, 1), "petal_star": (2, 1), "centre": (3, 1),
    "stem": (0, 0), "twig": (1, 0), "dry": (2, 0), "serrated_b": (3, 0),
}
# leaf frame per cell: u_rel = 0.5 + ax * x/L ; v_rel = v0 + ay * y/L
# (x across, y along the midrib from the attachment point, L = blade length)
FRAMES = {
    "round_a": (0.70, 0.70, 0.13), "round_b": (0.70, 0.70, 0.13), "round_c": (0.70, 0.70, 0.13),
    "serrated": (1.15, 0.88, 0.06), "serrated_b": (1.15, 0.88, 0.06),
    "clover": (0.88, 0.80, 0.06),
    "canopy_a": (1.7, 0.9, 0.05), "canopy_b": (1.7, 0.9, 0.05), "canopy_c": (1.7, 0.9, 0.05),
    "sprig": (1.35, 0.9, 0.05),
    "petal_white": (1.0, 0.9, 0.05), "petal_star": (2.2, 0.9, 0.05),
}


def cell_uv(cell, u_rel, v_rel):
    c, r = CELLS[cell]
    u_rel = min(max(u_rel, 0.01), 0.99)
    v_rel = min(max(v_rel, 0.01), 0.99)
    return ((c + u_rel) / GRID, (r + v_rel) / GRID)


def leaf_uv(cell, xn, yn):
    ax, ay, v0 = FRAMES[cell]
    return cell_uv(cell, 0.5 + ax * xn, v0 + ay * yn)


# ============================================================================
# atlas painting (numpy, per cell, in normalised leaf coordinates)
# ============================================================================

def _seg_dist(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    t = np.clip(((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy + 1e-12), 0, 1)
    return np.hypot(px - (ax + t * vx), py - (ay + t * vy)), t


def _polyline_mask(px, py, pts, w0, w1):
    """Soft line mask for a polyline whose half-width tapers from w0 to w1."""
    m = np.zeros_like(px)
    n = len(pts) - 1
    for i in range(n):
        d, t = _seg_dist(px, py, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
        w = w0 + (w1 - w0) * ((i + t) / n)
        m = np.maximum(m, pl.smoothstep(w, 0.35 * w, d))
    return m


def _curve(x0, y0, ang_deg, length, bend_deg, side, steps=6):
    """Vein starting at (x0, y0), leaving at ang (from +y) towards `side`,
    bending by bend_deg towards the tip over its length."""
    pts = [(x0, y0)]
    x, y = x0, y0
    for i in range(steps):
        a = math.radians(ang_deg - bend_deg * (i + 0.5) / steps)
        x += side * math.sin(a) * length / steps
        y += math.cos(a) * length / steps
        pts.append((x, y))
    return pts


def _venation(kind, rng):
    """Vein polylines in normalised leaf coordinates: list of (pts, w0, w1, strength).
    The first entry is the midrib."""
    mid = [(0.0, 0.0), (0.004, 0.35), (-0.003, 0.7), (0.0, 1.0)]
    veins = []
    if kind == "cordate":       # palmate: main veins fan out from the petiole
        veins.append((mid, 0.019, 0.006, 0.55))
        for side in (-1, 1):
            for ang, ln, bend in ((38, 0.62, 26), (74, 0.52, 34), (114, 0.36, 40), (150, 0.2, 40)):
                main = _curve(0.0, 0.0, ang + rng.uniform(-5, 5), ln, bend, side)
                veins.append((main, 0.012, 0.004, 0.42))
                for f in (0.4, 0.7):        # second-order veins off the outer side of each main vein
                    i = int(f * (len(main) - 1))
                    x0, y0 = main[i]
                    veins.append((_curve(x0, y0, ang - 10 + 45, 0.16 * (1.2 - f), 20, side), 0.006, 0.002, 0.3))
            for yk in (0.5, 0.68, 0.83):
                veins.append((_curve(0.0, yk, 50, 0.42 * (1.1 - yk) + 0.06, 24, side), 0.008, 0.003, 0.35))
        return veins
    if kind in ("pinnate", "sprig"):
        veins.append((mid, 0.020, 0.006, 0.52))
        n = 8 if kind == "pinnate" else 6
        for side in (-1, 1):
            for yk in np.linspace(0.08, 0.86, n):
                veins.append((_curve(0.0, yk + rng.uniform(-0.015, 0.015), 52, 0.42 * (1.1 - yk) + 0.06, 22, side),
                              0.010, 0.004, 0.32))
        return veins
    if kind == "canopy":
        veins.append((mid, 0.018, 0.005, 0.52))
        for side in (-1, 1):
            for yk in np.linspace(0.08, 0.84, 7):
                veins.append((_curve(0.0, yk + rng.uniform(-0.015, 0.015), 40, 0.28 * (1.1 - yk) + 0.04, 18, side),
                              0.007, 0.003, 0.32))
        return veins
    if kind == "clover":
        veins.append((mid, 0.014, 0.005, 0.5))
        for side in (-1, 1):
            for yk in np.linspace(0.15, 0.8, 5):
                veins.append((_curve(0.0, yk, 62, 0.42, 25, side), 0.006, 0.003, 0.3))
        return veins
    if kind == "petal":
        veins.append((mid, 0.012, 0.004, 0.5))
        for a in (-24, -12, 12, 24):
            veins.append((_curve(0.0, 0.0, a, 0.95, 0.0, 1), 0.007, 0.003, 0.4))
        return veins
    raise ValueError(kind)


def _cell_coords(cell, n):
    ax, ay, v0 = FRAMES.get(cell, (1.0, 1.0, 0.0))
    s = (np.arange(n, dtype=np.float32) + 0.5) / n
    U, V = np.meshgrid(s, s)                 # row 0 = bottom (v = 0)
    return U, V, (U - 0.5) / ax, (V - v0) / ay


def _paint_leaf(cell, n, rng, base_hex, vein_hex, kind, tint_hex=None, chevron=False, tone_amp=1.0):
    U, V, xn, yn = _cell_coords(cell, n)
    base = pl.hexrgb(base_hex)
    vein = pl.hexrgb(vein_hex)
    veins = _venation(kind, rng)
    # tertiary network: faint cell borders between the veins
    f1, f2, _ = pl.worley(n, rng, 22, 1.0)
    m_ter = pl.smoothstep(0.09, 0.02, f2 - f1)
    # broad, gentle tone variation (no fine speckle)
    tone = pl.spectral_noise(n, rng, pl.fbm_filter(2.6, kmin=1.0, kmax=12.0))
    col = base * (1.0 + 0.035 * tone_amp * tone[..., None])
    if tint_hex is not None:
        col = pl.lerp(col, pl.hexrgb(tint_hex), 0.22 * tone_amp * pl.smoothstep(-0.5, 1.5, tone))
    col = col * (0.94 + 0.08 * np.clip(yn, 0, 1))[..., None]          # tip a touch lighter
    col = pl.lerp(col, vein, 0.08 * m_ter)
    for pts, w0, w1, strength in veins[::-1]:                        # midrib painted last
        col = pl.lerp(col, vein, strength * _polyline_mask(xn, yn, pts, w0, w1))
    if chevron:   # pale V mark of a clover leaflet
        d = np.abs(yn - (0.42 + 0.85 * np.abs(xn)))
        col = pl.lerp(col, pl.hexrgb("#b4c79d"), 0.38 * pl.smoothstep(0.07, 0.025, d) * pl.smoothstep(0.75, 0.6, yn))
    return np.clip(col, 0, 1)


def _paint_petal(cell, n, rng, base_hex, base_tint_hex, vein_hex, mid_tint=None):
    U, V, xn, yn = _cell_coords(cell, n)
    col = _paint_leaf(cell, n, rng, base_hex, vein_hex, "petal", tone_amp=0.35)
    col = pl.lerp(col, pl.hexrgb(base_tint_hex), 0.6 * pl.smoothstep(0.35, 0.0, yn))
    if mid_tint is not None:
        col = pl.lerp(col, pl.hexrgb(mid_tint), 0.25 * pl.smoothstep(0.05, 0.0, np.abs(xn)) * pl.smoothstep(0.8, 0.2, yn))
    return col


def _paint_strip(n, rng, c0_hex, c1_hex, streak=0.06, dots=None):
    s = (np.arange(n, dtype=np.float32) + 0.5) / n
    U, V = np.meshgrid(s, s)
    col = pl.lerp(pl.hexrgb(c0_hex), pl.hexrgb(c1_hex), V)
    st = pl.spectral_noise(n, rng, pl.fibre_filter(18.0, 1.5))
    col = col * (1.0 + streak * st[..., None])
    if dots is not None:
        f1, _, _ = pl.worley(n, rng, 14, 1.0)
        col = pl.lerp(col, pl.hexrgb(dots), 0.5 * pl.smoothstep(0.16, 0.08, f1))
    return np.clip(col, 0, 1)


def build_atlas(seed=2207):
    rng = np.random.default_rng(seed)
    n = ATLAS_N // GRID
    atlas = np.zeros((ATLAS_N, ATLAS_N, 3), np.float32)

    def put(cell, img):
        c, r = CELLS[cell]
        atlas[r * n:(r + 1) * n, c * n:(c + 1) * n] = img

    # broad matte pale-green hero leaves (details/01): three tones
    put("round_a", _paint_leaf("round_a", n, rng, "#729955", "#a5c084", "cordate", "#86a35c"))
    put("round_b", _paint_leaf("round_b", n, rng, "#67904b", "#9dbb7c", "cordate", "#5a8746"))
    put("round_c", _paint_leaf("round_c", n, rng, "#7f9f57", "#afc78a", "cordate", "#95a657"))
    put("serrated", _paint_leaf("serrated", n, rng, "#668c4f", "#98b67e", "pinnate", "#5b8048"))
    put("serrated_b", _paint_leaf("serrated_b", n, rng, "#729352", "#a2bb82", "pinnate", "#7f9a52"))
    put("clover", _paint_leaf("clover", n, rng, "#4f7a3c", "#86a874", "clover", "#466f37", chevron=True))
    # canopy leaves (details/08): fresh, darker and yellow-green tones
    put("canopy_a", _paint_leaf("canopy_a", n, rng, "#5a9a3a", "#8fc06a", "canopy", "#4f8c34"))
    put("canopy_b", _paint_leaf("canopy_b", n, rng, "#4a7f34", "#7ba660", "canopy", "#3f7330"))
    put("canopy_c", _paint_leaf("canopy_c", n, rng, "#79a63f", "#a8c873", "canopy", "#8aab3e"))
    put("sprig", _paint_leaf("sprig", n, rng, "#5f8e45", "#93b879", "sprig", "#6c9346"))
    put("petal_white", _paint_petal("petal_white", n, rng, "#eeeee6", "#e2e6c4", "#dcdcd2"))
    put("petal_star", _paint_petal("petal_star", n, rng, "#e9ece2", "#b9cc9a", "#d6dccb", "#c7d6b4"))
    # centre cell: left half flower centre (yellow), right half dry seed beads
    s = (np.arange(n, dtype=np.float32) + 0.5) / n
    U, V = np.meshgrid(s, s)
    f1, _, _ = pl.worley(n, rng, 16, 1.0)
    ctr = pl.lerp(pl.hexrgb("#d9c45a"), pl.hexrgb("#b7a13e"), pl.smoothstep(0.2, 0.05, f1))
    bead = pl.lerp(pl.hexrgb("#a68d62"), pl.hexrgb("#c9b48a"), V) * (1 + 0.05 * pl.spectral_noise(n, rng, pl.band_filter(8, 0.7)))[..., None]
    put("centre", np.where((U < 0.5)[..., None], ctr, bead))
    put("stem", _paint_strip(n, rng, "#56723a", "#7c9a52", 0.05))
    put("twig", _paint_strip(n, rng, "#3d2e22", "#4f3d2d", 0.10, dots="#7a6a55"))
    put("dry", _paint_strip(n, rng, "#a99a72", "#d0c39c", 0.07))
    return np.clip(atlas, 0, 1)


# ============================================================================
# mesh building
# ============================================================================

class Kit:
    """Accumulates one mesh: positions, per-vertex uv and (flex, shade, blade)."""

    def __init__(self, name):
        self.name = name
        self.co, self.uv, self.flex, self.shade, self.blade = [], [], [], [], []
        self.tris = []

    def v(self, co, uv, flex, shade, blade):
        self.co.append(Vector(co))
        self.uv.append(uv)
        self.flex.append(flex)
        self.shade.append(shade)
        self.blade.append(blade)
        return len(self.co) - 1

    def t(self, a, b, c):
        self.tris.append((a, b, c))


def tube(k, pts, radii, cell, s0=0.0, sides=3, cap=True, shade=(0.72, 0.95), twist=0.0, n0=None):
    """Closed-sided tube (prism) along pts with per-point radius; parallel-
    transport frames; base left open (it sits in the host surface)."""
    pts = [Vector(p) for p in pts]
    seg = [(pts[i + 1] - pts[i]).length for i in range(len(pts) - 1)]
    s = [0.0]
    for d in seg:
        s.append(s[-1] + d)
    total = s[-1]
    T = []
    for i in range(len(pts)):
        a = pts[max(i - 1, 0)]
        b = pts[min(i + 1, len(pts) - 1)]
        T.append((b - a).normalized())
    N = n0.copy() if n0 is not None else T[0].orthogonal().normalized()
    N = (N - T[0] * N.dot(T[0])).normalized()
    rings = []
    for i, p in enumerate(pts):
        if i > 0:
            axis = T[i - 1].cross(T[i])
            if axis.length > 1e-8:
                ang = T[i - 1].angle(T[i])
                N = Matrix.Rotation(ang, 3, axis.normalized()) @ N
            N = (N - T[i] * N.dot(T[i])).normalized()
        B = T[i].cross(N)
        ring = []
        tt = s[i] / total
        for j in range(sides):
            a = 2 * math.pi * j / sides + twist
            off = (N * math.cos(a) + B * math.sin(a)) * radii[i]
            uv = cell_uv(cell, 0.3 + 0.4 * j / max(sides - 1, 1), 0.03 + 0.94 * tt)
            ring.append(k.v(p + off, uv, s0 + s[i], shade[0] + (shade[1] - shade[0]) * tt, 0.0))
        rings.append(ring)
    for i in range(len(rings) - 1):
        r0, r1 = rings[i], rings[i + 1]
        for j in range(sides):
            a, b = r0[j], r0[(j + 1) % sides]
            c, d = r1[(j + 1) % sides], r1[j]
            k.t(a, b, c)
            k.t(a, c, d)
    if cap:
        tip = pts[-1] + T[-1] * radii[-1] * 1.5
        ti = k.v(tip, cell_uv(cell, 0.5, 0.98), s0 + total + radii[-1] * 1.5, shade[1], 0.0)
        r = rings[-1]
        for j in range(sides):
            k.t(r[j], r[(j + 1) % sides], ti)
    return total


def bezier(p0, p1, p2, n):
    p0, p1, p2 = Vector(p0), Vector(p1), Vector(p2)
    return [(1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t * t * p2 for t in np.linspace(0, 1, n + 1)]


# ---------------------------------------------------------------- outlines

def _resample(pts, m, anchors):
    """Resample a closed dense polyline to m points by arc length, keeping
    the dense indices listed in `anchors` exactly (in order)."""
    pts = np.asarray(pts, np.float64)
    out = []
    a_idx = list(anchors) + [anchors[0] + len(pts)]
    counts = []
    seglens = []
    for i in range(len(anchors)):
        i0, i1 = a_idx[i], a_idx[i + 1]
        idx = np.arange(i0, i1 + 1) % len(pts)
        seg = pts[idx]
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))]
        seglens.append((seg, d))
        counts.append(d[-1])
    tot = sum(counts)
    for (seg, d), ln in zip(seglens, counts):
        k = max(2, int(round(m * ln / tot)))
        for t in np.linspace(0, d[-1], k + 1)[:-1]:
            j = min(np.searchsorted(d, t, side="right") - 1, len(seg) - 2)
            f = (t - d[j]) / max(d[j + 1] - d[j], 1e-12)
            out.append(tuple(seg[j] * (1 - f) + seg[j + 1] * f))
    return out


def outline_cordate(w_ratio, p=0.75, tip=0.1, m=18):
    """Heart / kidney blade, sinus at (0,0), tip at (0,1); CCW; m points.
    Returns points and the index of the sinus (tip is index 0)."""
    dense = []
    K = 720
    for i in range(K):
        psi = 2 * math.pi * i / K
        phi = psi if psi <= math.pi else psi - 2 * math.pi
        g = ((1 + math.cos(phi)) / 2) ** p * (1 + tip * math.exp(-(phi / 0.35) ** 2))
        dense.append((-w_ratio * g * math.sin(phi), g * math.cos(phi)))
    out = _resample(dense, m, [0, K // 2])
    return out, out.index(tuple(np.asarray(dense[K // 2], np.float64)))


def outline_ovate(w_ratio, widest=0.4, tip_sharp=1.0, base_round=0.6, m=16):
    """Ovate/lanceolate blade, base at (0,0), tip at (0,1); CCW; m points, tip first.
    Returns points and the index of the base point."""
    K = 400
    ts = np.linspace(0, 1, K // 2 + 1)
    a = math.log(0.5) / math.log(widest)
    half = []
    for t in ts:
        w = 0.5 * w_ratio * math.sin(math.pi * t ** a) ** (base_round if t < widest else tip_sharp)
        half.append((w, t))
    left = [(-w, t) for (w, t) in reversed(half)]                 # tip -> base on the left
    right = [(w, t) for (w, t) in half[1:-1]]                     # base -> tip on the right
    out = _resample(left + right, m, [0, len(half) - 1])
    bi = min(range(len(out)), key=lambda i: out[i][0] ** 2 + out[i][1] ** 2)
    return out, bi


def outline_teeth(w_ratio, teeth, depth, widest=0.38):
    """Serrated ovate outline: explicit tooth tip + notch vertices per tooth."""
    a = math.log(0.5) / math.log(widest)

    def w(t):
        return 0.5 * w_ratio * math.sin(math.pi * t ** a) ** (0.65 if t < widest else 1.0)
    right = [(0.0, 0.0)]
    t_lo, t_hi = 0.14, 0.93
    right.append((w(0.07), 0.07))
    for i in range(teeth):
        t0 = t_lo + (t_hi - t_lo) * i / teeth
        t1 = t_lo + (t_hi - t_lo) * (i + 1) / teeth
        right.append((w(t0) * (1 - depth), t0 + 0.15 * (t1 - t0)))     # notch
        right.append((w(t1 - 0.25 * (t1 - t0)) * 1.0, t1 - 0.05 * (t1 - t0)))  # tooth tip
    right.append((0.0, 1.0))
    left = [(-x, y) for (x, y) in reversed(right[1:-1])]
    pts = [(0.0, 1.0)] + left + [(0.0, 0.0)] + right[1:-1]
    return pts, len(left) + 1


# ---------------------------------------------------------------- blades

def blade(k, outline, base_i, interior, L, cell, frame, origin, s0, *,
          fold=10.0, cup=0.2, curl=0.0, wave=0.0, wave_n=3, twist=0.0, flip_y=False,
          shade_base=0.78, chain=True):
    """Triangulated leaf blade.
    outline: CCW list of (xn, yn) normalised by L, tip at index 0, base/sinus at base_i
    interior: extra (xn, yn) points (midrib points first, they become constraint edges)
    frame: 3x3 matrix (columns: blade x, blade y = base direction, blade normal)
    fold: V-fold half angle (deg) along the midrib, cup: edge rise ~ cup * x^2 / L,
    curl: midrib curvature * L (rad; + = tip goes down), wave: edge ripple (in L),
    twist: deg about the midrib at the tip."""
    m = len(outline)
    verts2 = [Vector(p) for p in outline] + [Vector(p) for p in interior]
    mids = [i for i, p in enumerate(interior) if abs(p[0]) < 1e-9]
    edges = []
    if chain and mids:
        ch = [base_i] + [m + i for i in sorted(mids, key=lambda i: interior[i][1])] + [0]
        edges = [(ch[i], ch[i + 1]) for i in range(len(ch) - 1)]
    res = mgeo.delaunay_2d_cdt(verts2, edges, [list(range(m))], 1, 1e-7)
    out_v, _, out_f, orig_v = res[0], res[1], res[2], res[3]
    on_outline = [any(o < m for o in ov) for ov in orig_v]
    outl_idx = [min(ov) if any(o < m for o in ov) else -1 for ov in orig_v]
    kappa = curl / L if abs(curl) > 1e-6 else 0.0
    tw = math.radians(twist)
    tf = math.tan(math.radians(fold))
    xs = [p.x for p in out_v]
    half_w = max(1e-6, max(abs(x) for x in xs))
    idx = []
    for i, p in enumerate(out_v):
        xn, yn = p.x, p.y
        x, y = xn * L, yn * L
        z = tf * abs(x) + cup * x * x / L
        if on_outline[i] and wave:
            # wave_n ripples around the whole margin, fading out at tip and base
            ph = outl_idx[i] / m * 2 * math.pi * wave_n
            z += wave * L * math.sin(ph) * min(1.0, abs(xn) / (0.3 * half_w + 1e-6))
        a = tw * max(yn, 0.0)
        x, z = x * math.cos(a) - z * math.sin(a), x * math.sin(a) + z * math.cos(a)
        if kappa:
            R = 1.0 / kappa
            ph = y / R
            y, z = R * math.sin(ph) + z * math.sin(ph), -R * (1 - math.cos(ph)) + z * math.cos(ph)
        co = Vector(origin) + frame @ Vector((x, y, z))
        uvx, uvy = (xn, 1.0 - yn) if flip_y else (xn, yn)
        uv = leaf_uv(cell, uvx, uvy)
        dist = math.hypot(xn, yn) * L
        rho = min(1.0, math.hypot(xn / (half_w + 1e-6), yn) / 1.0)
        cupness = max(0.0, 1.0 - abs(xn) / (0.6 * half_w + 1e-6)) * max(0.0, 1.0 - abs(yn - 0.3) / 0.5)
        shade = shade_base + (1.0 - shade_base) * pl.smoothstep(0.0, 0.9, rho) - 0.07 * cupness
        idx.append(k.v(co, uv, s0 + dist, float(np.clip(shade, 0.62, 1.0)), 1.0))
    for f in out_f:
        a, b, c = (out_v[f[0]], out_v[f[1]], out_v[f[2]])
        area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
        if abs(area) < 1e-12:
            continue
        if area > 0:
            k.t(idx[f[0]], idx[f[1]], idx[f[2]])
        else:
            k.t(idx[f[0]], idx[f[2]], idx[f[1]])
    return len(out_f)


def frame_from(direction_az, elev_deg, roll_deg=0.0, up=(0, 0, 1)):
    """Blade frame: y = base direction at azimuth/elevation, z = blade normal."""
    a = math.radians(direction_az)
    e = math.radians(elev_deg)
    D = Vector((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))
    X = Vector((math.sin(a), -math.cos(a), 0.0))
    Z = X.cross(D).normalized()
    if roll_deg:
        rot = Matrix.Rotation(math.radians(roll_deg), 3, D)
        X, Z = rot @ X, rot @ Z
    return Matrix((X, D, Z)).transposed()


def frame_axis(D, ref=(0, 0, 1), roll_deg=0.0):
    """Blade frame whose y axis is D; blade normal as close to `ref` as possible."""
    D = Vector(D).normalized()
    ref = Vector(ref)
    X = D.cross(ref)
    if X.length < 1e-6:
        X = D.orthogonal()
    X.normalize()
    Z = X.cross(D).normalized()
    if Z.dot(ref) < 0:
        X, Z = -X, -Z
    if roll_deg:
        rot = Matrix.Rotation(math.radians(roll_deg), 3, D)
        X, Z = rot @ X, rot @ Z
    return Matrix((X, D, Z)).transposed()


# ============================================================================
# kit items
# ============================================================================

def petiole_and_blade(name, cell, L, w_ratio, petiole_h, elev, *, kind="cordate", p=0.75, tip=0.1,
                      fold=10, cup=0.25, curl=0.3, wave=0.02, wave_n=3, twist=0.0, lean=0.25,
                      teeth=0, depth=0.0, petiole_r=(0.75 * MM, 0.5 * MM), seed=1):
    k = Kit(name)
    az = 0.0
    frame = frame_from(az, elev)
    D = frame.col[1]
    # petiole rises from the origin and arcs forward into the blade direction
    P = Vector((lean * L * D.x, lean * L * D.y, petiole_h))
    path = bezier((0, 0, 0), (0.0, 0.0, 0.72 * petiole_h), P, 2)
    radii = [petiole_r[0] + (petiole_r[1] - petiole_r[0]) * i / (len(path) - 1) for i in range(len(path))]
    s_top = tube(k, path, radii, "stem", cap=False, shade=(0.66, 0.9))
    if kind == "cordate":
        outline, base_i = outline_cordate(w_ratio, p=p, tip=tip, m=20)
        interior = [(0.0, 0.32), (0.0, 0.64), (-0.3 * w_ratio, 0.42), (0.3 * w_ratio, 0.42)]
    else:
        outline, base_i = outline_teeth(w_ratio, teeth, depth)
        interior = [(0.0, 0.3), (0.0, 0.62), (-0.18 * w_ratio, 0.36), (0.18 * w_ratio, 0.36)]
    blade(k, outline, base_i, interior, L, cell, frame, P, s_top,
          fold=fold, cup=cup, curl=curl, wave=wave, wave_n=wave_n, twist=twist)
    return k


def make_clover(seed=3):
    k = Kit("leaf_clover")
    rng = np.random.default_rng(seed)
    H = 1.5 * CM
    P = Vector((0.6 * MM, -0.3 * MM, H))
    path = bezier((0, 0, 0), (0.0, 0.0, 0.6 * H), P, 2)
    s_top = tube(k, path, [0.5 * MM, 0.42 * MM, 0.35 * MM], "stem", cap=False, shade=(0.66, 0.9))
    # obcordate leaflet = heart outline flipped: base at (0,0), notch at the top
    o, base_i = outline_cordate(0.62, p=0.7, tip=0.0, m=12)
    ymax = max(y for _, y in o)
    ymin = min(y for _, y in o)
    span = 1.0 - ymin
    flipped = [(x / span, (1.0 - y) / span) for (x, y) in o]
    # after the flip the old tip (index 0) is the leaflet base and the sinus is the notch
    flipped = [flipped[0]] + flipped[1:][::-1]          # restore CCW after mirroring y
    base_idx = 0
    pts = [(x, y) for (x, y) in flipped]
    # rotate list so the notch (top, max y near x=0) is index 0 as blade() expects
    notch = min(range(len(pts)), key=lambda i: abs(pts[i][0]) + abs(pts[i][1] - (1.0 / span)) * 0.2
                if pts[i][1] > 0.5 else 9)
    pts = pts[notch:] + pts[:notch]
    base_i = pts.index(flipped[base_idx])
    Ll = 0.95 * CM
    for i, az in enumerate((90.0, 210.0, 330.0)):
        az += rng.uniform(-8, 8)
        fr = frame_from(az, 14 + rng.uniform(-4, 4), roll_deg=rng.uniform(-6, 6))
        origin = P + fr.col[1] * 0.4 * MM
        blade(k, pts, base_i, [(0.0, 0.45)], Ll, "clover", fr, origin, s_top,
              fold=12, cup=0.25, curl=0.15, wave=0.0, chain=True)
    return k


def make_flower_small(seed=5):
    k = Kit("flower_white_small")
    rng = np.random.default_rng(seed)
    H = 2.3 * CM
    P = Vector((1.8 * MM, 0.6 * MM, H))
    path = bezier((0, 0, 0), (0.0, 0.0, 0.55 * H), P, 2)
    s_top = tube(k, path, [0.26 * MM, 0.22 * MM, 0.2 * MM], "stem", cap=False, shade=(0.7, 0.92))
    tilt = Matrix.Rotation(math.radians(16), 3, Vector((0.3, -1, 0)).normalized())
    up = tilt @ Vector((0, 0, 1))
    # centre disc (pentagon), slightly raised
    c_r = 0.85 * MM
    ci = k.v(P + up * 0.35 * MM, cell_uv("centre", 0.25, 0.5), s_top + 0.3 * MM, 0.9, 1.0)
    ring = []
    for j in range(5):
        a = 2 * math.pi * j / 5
        d = tilt @ Vector((math.cos(a), math.sin(a), 0))
        ring.append(k.v(P + d * c_r, cell_uv("centre", 0.25 + 0.2 * math.cos(a), 0.5 + 0.4 * math.sin(a)),
                        s_top + c_r, 0.88, 1.0))
    for j in range(5):
        k.t(ci, ring[j], ring[(j + 1) % 5])
    petal, base_i = outline_ovate(0.85, widest=0.62, tip_sharp=0.35, base_round=0.9, m=6)
    for j in range(5):
        a = 360.0 * j / 5 + 36.0 + rng.uniform(-6, 6)
        dl = tilt @ Vector((math.cos(math.radians(a)), math.sin(math.radians(a)), 0))
        D = (dl + up * math.tan(math.radians(14))).normalized()
        fr = frame_axis(D, ref=up)
        blade(k, petal, base_i, [(0.0, 0.5)], 3.4 * MM, "petal_white", fr, P + dl * 0.6 * c_r, s_top + 0.5 * MM,
              fold=4, cup=0.35, curl=-0.25, shade_base=0.86)
    return k


def make_flower_star(seed=7):
    k = Kit("flower_star")
    rng = np.random.default_rng(seed)
    H = 1.0 * CM
    P = Vector((0.5 * MM, 0.0, H))
    path = bezier((0, 0, 0), (0.0, 0.0, 0.5 * H), P, 1)
    s_top = tube(k, path, [0.9 * MM, 0.7 * MM], "stem", cap=False, shade=(0.66, 0.85))
    up = Vector((0, 0, 1))
    c_r = 2.2 * MM
    ci = k.v(P + up * 0.8 * MM, cell_uv("centre", 0.25, 0.5), s_top + 0.5 * MM, 0.86, 1.0)
    ring = []
    for j in range(6):
        a = 2 * math.pi * j / 6
        ring.append(k.v(P + Vector((math.cos(a), math.sin(a), 0)) * c_r,
                        cell_uv("centre", 0.25 + 0.2 * math.cos(a), 0.5 + 0.4 * math.sin(a)), s_top + c_r, 0.82, 1.0))
    for j in range(6):
        k.t(ci, ring[j], ring[(j + 1) % 6])
    petal, base_i = outline_ovate(0.32, widest=0.34, tip_sharp=1.2, base_round=0.7, m=8)
    for j in range(6):
        a = 60.0 * j + rng.uniform(-7, 7)
        ln = (2.6 if j % 2 == 0 else 2.2) * CM * rng.uniform(0.95, 1.05)
        fr = frame_from(a, 22 + rng.uniform(-4, 4), roll_deg=rng.uniform(-5, 5))
        origin = P + Vector((math.cos(math.radians(a)), math.sin(math.radians(a)), 0)) * 0.7 * c_r
        blade(k, petal, base_i, [(0.0, 0.35), (0.0, 0.7)], ln, "petal_star", fr, origin, s_top + c_r,
              fold=12, cup=0.2, curl=-0.35, shade_base=0.82)
    return k


def make_sprig(name, length, n_leaves, bend, seed):
    k = Kit(name)
    rng = np.random.default_rng(seed)
    n = 5
    pts = []
    for i in range(n + 1):
        t = i / n
        pts.append(Vector((bend * length * t * t, 0.15 * bend * length * math.sin(3 * t), length * t)))
    radii = [0.95 * MM * (1 - 0.5 * i / n) for i in range(n + 1)]
    total = tube(k, pts, radii, "twig", cap=True, shade=(0.7, 0.95))
    seg = np.r_[0, np.cumsum([(pts[i + 1] - pts[i]).length for i in range(n)])]
    leaf, base_i = outline_ovate(0.55, widest=0.42, tip_sharp=1.1, base_round=0.7, m=10)
    ss = list(np.linspace(0.32, 0.86, n_leaves - 1)) + [1.0]
    for i, s in enumerate(ss):
        d = s * total
        j = min(int(np.searchsorted(seg, d, side="right") - 1), n - 1)
        f = (d - seg[j]) / max(seg[j + 1] - seg[j], 1e-9)
        p = pts[j].lerp(pts[j + 1], f)
        tdir = (pts[j + 1] - pts[j]).normalized()
        if s >= 1.0:
            D = tdir
            ln = 1.7 * CM
        else:
            az = math.radians(137.5 * i + rng.uniform(-15, 15))
            radial = Vector((math.cos(az), math.sin(az), 0))
            radial = (radial - tdir * radial.dot(tdir)).normalized()
            D = (radial * math.cos(math.radians(40 + rng.uniform(-10, 12))) + tdir * 0.75).normalized()
            ln = rng.uniform(1.3, 1.85) * CM
        fr = frame_axis(D, ref=(0, 0, 1), roll_deg=rng.uniform(-20, 20))
        blade(k, leaf, base_i, [(0.0, 0.45)], ln, "sprig", fr, p, d,
              fold=14, cup=0.25, curl=rng.uniform(0.2, 0.6), shade_base=0.8)
    return k


def make_seedhead(seed=11):
    k = Kit("seedhead_a")
    rng = np.random.default_rng(seed)
    Lh = 10.5 * CM
    n = 6
    pts = [Vector((0.06 * Lh * (i / n) ** 2, 0.012 * Lh * math.sin(2.5 * i / n), Lh * i / n)) for i in range(n + 1)]
    radii = [0.62 * MM * (1 - 0.45 * i / n) for i in range(n + 1)]
    total = tube(k, pts, radii, "dry", cap=True, shade=(0.75, 1.0))
    seg = np.r_[0, np.cumsum([(pts[i + 1] - pts[i]).length for i in range(n)])]

    def at(s):
        d = s * total
        j = min(int(np.searchsorted(seg, d, side="right") - 1), n - 1)
        f = (d - seg[j]) / max(seg[j + 1] - seg[j], 1e-9)
        return pts[j].lerp(pts[j + 1], f), (pts[j + 1] - pts[j]).normalized(), d

    def bead(c, axis, s0, length=3.0 * MM, width=1.45 * MM):
        axis = axis.normalized()
        u = axis.orthogonal().normalized()
        w = axis.cross(u)
        top = k.v(c + axis * length * 0.5, cell_uv("centre", 0.75, 0.9), s0 + length, 0.95, 0.0)
        bot = k.v(c - axis * length * 0.5, cell_uv("centre", 0.75, 0.1), s0, 0.85, 0.0)
        ring = []
        for j in range(4):
            a = 2 * math.pi * j / 4 + 0.4
            ring.append(k.v(c + (u * math.cos(a) + w * math.sin(a)) * width * 0.5,
                            cell_uv("centre", 0.6 + 0.3 * j / 3, 0.5), s0 + 0.5 * length, 0.92, 0.0))
        for j in range(4):
            k.t(ring[j], ring[(j + 1) % 4], top)
            k.t(ring[(j + 1) % 4], ring[j], bot)

    for i, s in enumerate((0.74, 0.81, 0.87, 0.93)):
        p, tdir, d = at(s)
        az = math.radians(137.5 * i + 30 + rng.uniform(-15, 15))
        side = Vector((math.cos(az), math.sin(az), 0))
        side = (side - tdir * side.dot(tdir)).normalized()
        bdir = (tdir * math.cos(math.radians(34)) + side * math.sin(math.radians(34))).normalized()
        blen = rng.uniform(0.6, 0.95) * CM * (1.15 - 0.4 * (s - 0.74) / 0.2)
        bpts = [p, p + bdir * blen * 0.5, p + bdir * blen]
        tube(k, bpts, [0.28 * MM, 0.24 * MM, 0.2 * MM], "dry", s0=d, cap=False, shade=(0.85, 1.0))
        bead(p + bdir * (blen + 1.4 * MM), bdir, d + blen)
        bead(p + bdir * blen * 0.55 + side * 1.1 * MM, (bdir + side * 0.5), d + blen * 0.55,
             length=2.4 * MM, width=1.2 * MM)
    p, tdir, d = at(1.0)
    bead(p + tdir * 1.9 * MM, tdir, d)
    return k


def make_canopy_leaf(name, cell, L, w_ratio, fold, curl, twist, seed):
    k = Kit(name)
    rng = np.random.default_rng(seed)
    out, base_i = outline_ovate(w_ratio, widest=0.45, tip_sharp=1.25, base_round=0.75, m=8)
    fr = frame_axis((0, 0, 1), ref=(0, -1, 0))           # midrib along +Z (growth axis)
    blade(k, out, base_i, [(0.0, 0.33), (0.0, 0.66)], L, cell, fr, (0, 0, 0), 0.0,
          fold=fold, cup=0.12, curl=curl, twist=twist, shade_base=0.8)
    return k


def make_twig(name, length, forks, seed):
    k = Kit(name)
    rng = np.random.default_rng(seed)
    n = 6
    pts = [Vector((0, 0, 0))]
    d = Vector((0, 0, 1))
    for i in range(n):
        d = (d + Vector((rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25), 0.0))).normalized()
        pts.append(pts[-1] + d * (length / n))
    radii = [2.0 * MM * (1 - 0.68 * i / n) for i in range(n + 1)]
    total = tube(k, pts, radii, "twig", cap=True, shade=(0.7, 0.95))
    seg = np.r_[0, np.cumsum([(pts[i + 1] - pts[i]).length for i in range(n)])]
    for fi, (s, ln, az, sub) in enumerate(forks):
        dd = s * total
        j = min(int(np.searchsorted(seg, dd, side="right") - 1), n - 1)
        f = (dd - seg[j]) / max(seg[j + 1] - seg[j], 1e-9)
        p = pts[j].lerp(pts[j + 1], f)
        tdir = (pts[j + 1] - pts[j]).normalized()
        side = Vector((math.cos(math.radians(az)), math.sin(math.radians(az)), 0))
        side = (side - tdir * side.dot(tdir)).normalized()
        bdir = (tdir * math.cos(math.radians(38)) + side * math.sin(math.radians(38))).normalized()
        m = 3
        bp = [p]
        bd = bdir
        for i in range(m):
            bd = (bd + tdir * 0.12 + Vector((rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), 0))).normalized()
            bp.append(bp[-1] + bd * ln / m)
        r0 = radii[j] * 0.62
        brad = [r0 * (1 - 0.55 * i / m) for i in range(m + 1)]
        tube(k, bp, brad, "twig", s0=dd, cap=True, shade=(0.8, 0.96))
        if sub:
            q = bp[2]
            sd = (bd.cross(Vector((0, 0, 1))).normalized() * 0.7 + bd).normalized()
            sp = [q, q + sd * 0.5 * sub, q + sd * sub]
            tube(k, sp, [brad[2] * 0.7, brad[2] * 0.5, brad[2] * 0.35], "twig", s0=dd + 2 * ln / m,
                 cap=True, shade=(0.85, 0.97))
    return k


def build_items():
    items = []
    items.append(petiole_and_blade("leaf_round_a", "round_a", 4.3 * CM, 0.80, 4.3 * CM, 18,
                                   p=0.72, tip=0.10, fold=12, cup=0.28, curl=0.35, wave=0.016, wave_n=4, seed=1))
    items.append(petiole_and_blade("leaf_round_b", "round_b", 3.5 * CM, 0.92, 3.3 * CM, 30,
                                   p=0.62, tip=0.0, fold=8, cup=0.36, curl=0.18, wave=0.02, wave_n=5, seed=2))
    items.append(petiole_and_blade("leaf_round_c", "round_c", 4.7 * CM, 0.74, 5.6 * CM, 10,
                                   p=0.8, tip=0.18, fold=16, cup=0.2, curl=0.6, wave=0.014, wave_n=4,
                                   twist=12, seed=3))
    items.append(petiole_and_blade("leaf_serrated_a", "serrated", 4.6 * CM, 0.70, 4.0 * CM, 24,
                                   kind="serrate", teeth=6, depth=0.16, fold=14, cup=0.22, curl=0.35,
                                   wave=0.0, seed=4))
    items.append(petiole_and_blade("leaf_serrated_b", "serrated_b", 4.9 * CM, 0.62, 3.6 * CM, 38,
                                   kind="serrate", teeth=7, depth=0.15, fold=18, cup=0.18, curl=2.0,
                                   wave=0.0, twist=-10, seed=5))
    items.append(make_clover())
    items.append(make_flower_small())
    items.append(make_flower_star())
    items.append(make_sprig("sprig_a", 5.2 * CM, 4, 0.12, seed=21))
    items.append(make_sprig("sprig_b", 6.6 * CM, 5, 0.22, seed=22))
    items.append(make_seedhead())
    items.append(make_canopy_leaf("leaf_canopy_a", "canopy_a", 2.6 * CM, 0.46, 22, 0.35, 0, seed=31))
    items.append(make_canopy_leaf("leaf_canopy_b", "canopy_b", 2.2 * CM, 0.52, 14, 0.2, 10, seed=32))
    items.append(make_canopy_leaf("leaf_canopy_c", "canopy_c", 2.9 * CM, 0.40, 28, 0.6, -8, seed=33))
    items.append(make_twig("twig_canopy_a", 15.5 * CM,
                           [(0.35, 5.5 * CM, 20, 0), (0.55, 4.5 * CM, 160, 1.8 * CM), (0.75, 3.5 * CM, 280, 0)],
                           seed=41))
    items.append(make_twig("twig_canopy_b", 18.5 * CM,
                           [(0.4, 6.0 * CM, 200, 2.2 * CM), (0.68, 4.2 * CM, 40, 1.5 * CM)], seed=42))
    return items


# ============================================================================
# Blender objects, export, validation
# ============================================================================

def make_material():
    mat = bpy.data.materials.get("mat_kit") or bpy.data.materials.new("mat_kit")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.45, 0.6, 0.35, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.7
    return mat


def to_object(k, mat):
    me = bpy.data.meshes.new(k.name)
    co = [tuple(c) for c in k.co]
    me.from_pydata(co, [], k.tris)
    me.validate(clean_customdata=False)
    nl = len(me.loops)
    lv = np.zeros(nl, np.int64)
    me.loops.foreach_get("vertex_index", lv)
    uv = me.uv_layers.new(name="UVMap")
    uvs = np.asarray(k.uv, np.float32)[lv]
    uv.data.foreach_set("uv", uvs.ravel())
    flex = np.asarray(k.flex, np.float32)
    flex = flex / max(flex.max(), 1e-9)
    col = np.stack([flex, np.asarray(k.shade, np.float32), np.asarray(k.blade, np.float32),
                    np.ones_like(flex)], -1)
    ca = me.color_attributes.new(name="Color", type="FLOAT_COLOR", domain="POINT")
    ca.data.foreach_set("color", col.ravel())
    me.color_attributes.active_color = ca
    try:
        me.color_attributes.render_color_index = me.color_attributes.active_color_index
    except Exception:
        pass
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
    me.materials.append(mat)
    me.update()
    ob = bpy.data.objects.new(k.name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def export_glb(objs, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    for ob in objs:
        ob.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.export_scene.gltf(
        filepath=path, check_existing=False, export_format="GLB", use_selection=True,
        export_yup=True, export_apply=True, export_normals=True, export_texcoords=True,
        export_tangents=False, export_vertex_color="ACTIVE", export_all_vertex_colors=False,
        export_materials="EXPORT", export_image_format="NONE", export_cameras=False,
        export_lights=False, export_animations=False, export_extras=False, export_attributes=False)


def read_glb(path):
    data = open(path, "rb").read()
    magic, version, length = struct.unpack("<III", data[:12])
    assert magic == 0x46546C67 and version == 2, "not a glTF 2 binary"
    jlen, jtype = struct.unpack("<II", data[12:20])
    js = json.loads(data[20:20 + jlen])
    off = 20 + jlen
    blen, btype = struct.unpack("<II", data[off:off + 8])
    return js, data[off + 8:off + 8 + blen]


def accessor_array(js, binc, idx):
    acc = js["accessors"][idx]
    bv = js["bufferViews"][acc["bufferView"]]
    comp = {5126: np.float32, 5123: np.uint16, 5125: np.uint32, 5121: np.uint8}[acc["componentType"]]
    ncomp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[acc["type"]]
    start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = bv.get("byteStride", 0)
    itemsize = np.dtype(comp).itemsize * ncomp
    if stride and stride != itemsize:
        raw = np.frombuffer(binc, np.uint8, count=stride * acc["count"], offset=start)
        arr = raw.reshape(acc["count"], stride)[:, :itemsize].copy().view(comp)
    else:
        arr = np.frombuffer(binc, comp, count=acc["count"] * ncomp, offset=start)
    arr = arr.reshape(acc["count"], ncomp).astype(np.float64)
    if acc.get("normalized"):
        arr /= {np.uint16: 65535.0, np.uint8: 255.0}[comp]
    return arr


def validate_glb(path, expected):
    js, binc = read_glb(path)
    report = []
    ok = True
    names = [n.get("name") for n in js.get("nodes", [])]
    if sorted(names) != sorted(expected):
        ok = False
        report.append("NODE NAMES MISMATCH: %s" % names)
    for key in ("cameras", "animations", "skins"):
        if js.get(key):
            ok = False
            report.append("unexpected %s in GLB" % key)
    if "KHR_lights_punctual" in json.dumps(js.get("extensions", {})):
        ok = False
        report.append("unexpected lights")
    mats = [m.get("name") for m in js.get("materials", [])]
    if mats != ["mat_kit"]:
        ok = False
        report.append("materials: %s" % mats)
    total = 0
    rows = []
    for node in js["nodes"]:
        nm = node["name"]
        for key in ("translation", "rotation", "scale", "matrix"):
            if key in node:
                vals = node[key]
                ident = {"translation": [0, 0, 0], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]}.get(key)
                if ident is None or any(abs(a - b) > 1e-6 for a, b in zip(vals, ident)):
                    ok = False
                    report.append("%s has non-identity %s %s" % (nm, key, vals))
        mesh = js["meshes"][node["mesh"]]
        tris = 0
        lo = np.array([1e9] * 3)
        hi = -lo
        cmin = np.array([1e9] * 4)
        cmax = -cmin
        root_flex = []
        for prim in mesh["primitives"]:
            attrs = prim["attributes"]
            for need in ("POSITION", "NORMAL", "TEXCOORD_0", "COLOR_0"):
                if need not in attrs:
                    ok = False
                    report.append("%s missing %s" % (nm, need))
            pos = accessor_array(js, binc, attrs["POSITION"])
            col = accessor_array(js, binc, attrs["COLOR_0"])
            lo = np.minimum(lo, pos.min(0))
            hi = np.maximum(hi, pos.max(0))
            cmin = np.minimum(cmin, col.min(0) if col.shape[1] == 4 else np.r_[col.min(0), 1])
            cmax = np.maximum(cmax, col.max(0) if col.shape[1] == 4 else np.r_[col.max(0), 1])
            d0 = np.linalg.norm(pos, axis=1)
            root_flex.append(col[np.argmin(d0), 0])
            tris += js["accessors"][prim["indices"]]["count"] // 3
        total += tris
        size = (hi - lo) / CM        # glTF: x, y(up), z
        rows.append((nm, tris, size, lo / CM, hi / CM, cmin, cmax, min(root_flex),
                     js["accessors"][mesh["primitives"][0]["attributes"]["COLOR_0"]]))
    return ok, report, rows, total


def reimport_check(path, expected):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=path)
    objs = {o.name: o for o in bpy.context.scene.objects}
    problems = []
    for nm in expected:
        o = objs.get(nm)
        if o is None:
            problems.append("missing object %s after re-import" % nm)
            continue
        if o.location.length > 1e-6 or o.parent is not None:
            problems.append("%s not at origin / parented" % nm)
        if not o.data.color_attributes:
            problems.append("%s has no colour attribute after re-import" % nm)
        zmin = min(v.co.z for v in o.data.vertices)
        if zmin < -0.02:
            problems.append("%s extends %.1f cm below its origin" % (nm, zmin / CM))
        if max(o.dimensions) > 0.25 or max(o.dimensions) < 0.005:
            problems.append("%s implausible size %s m" % (nm, tuple(o.dimensions)))
        log("  re-imported %-20s dims %.3f x %.3f x %.3f m  colour attrs %s" % (
            nm, *o.dimensions, [(a.name, a.data_type, a.domain) for a in o.data.color_attributes]))
    extra = [n for n in objs if n not in expected]
    if extra:
        problems.append("extra objects: %s" % extra)
    return problems, objs


# ============================================================================
# preview grid (rendered from the re-imported GLB)
# ============================================================================

def preview(objs, atlas_path, out_path, rows_info):
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE"
    sc.eevee.taa_render_samples = 32
    sc.render.resolution_x = 360
    sc.render.resolution_y = 360
    sc.render.image_settings.file_format = "PNG"
    sc.view_settings.view_transform = "AgX"
    world = bpy.data.worlds.new("w")
    sc.world = world
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs[0].default_value = (0.36, 0.40, 0.33, 1)
    bg.inputs[1].default_value = 0.7
    # preview material: atlas x vertex shade (COLOR_0.G), slight translucency on blades
    mat = bpy.data.materials.new("kit_preview")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(atlas_path)
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "Color"
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(attr.outputs["Color"], sep.inputs["Color"])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    ins = {s.identifier: s for s in mix.inputs}
    outs = {s.identifier: s for s in mix.outputs}
    ins["Factor_Float"].default_value = 1.0
    nt.links.new(tex.outputs["Color"], ins["A_Color"])
    nt.links.new(sep.outputs["Green"], ins["B_Color"])
    nt.links.new(outs["Result_Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.6
    for o in objs.values():
        if o.data.color_attributes:
            o.data.color_attributes[0].name = "Color"
        o.data.materials.clear()
        o.data.materials.append(mat)
    for name, (rx, ry, ang, en) in {"key": (50, 10, -35, 3.2), "fill": (65, 0, 150, 0.7),
                                    "rim": (25, 0, 110, 1.4)}.items():
        ld = bpy.data.lights.new(name, "SUN")
        ld.energy = en
        ld.angle = math.radians(4)
        lo = bpy.data.objects.new(name, ld)
        sc.collection.objects.link(lo)
        lo.rotation_euler = (math.radians(rx), math.radians(ry), math.radians(ang))
    cd = bpy.data.cameras.new("cam")
    cd.sensor_fit = "VERTICAL"
    cd.angle_y = math.radians(30)
    cam = bpy.data.objects.new("cam", cd)
    sc.collection.objects.link(cam)
    sc.camera = cam
    def shoot(vs, az_deg, el_deg, res):
        c = (vs.min(0) + vs.max(0)) / 2
        r = np.linalg.norm(vs - c, axis=1).max()
        dist = r / math.sin(math.radians(15)) * 1.05
        az, el = math.radians(az_deg), math.radians(el_deg)
        cam.location = Vector(c) + Vector((math.cos(az) * math.cos(el), math.sin(az) * math.cos(el),
                                          math.sin(el))) * dist
        cam.rotation_euler = (Vector(c) - cam.location).to_track_quat("-Z", "Y").to_euler()
        cd.clip_start = dist * 0.01
        cd.clip_end = dist * 10
        sc.render.resolution_x = sc.render.resolution_y = res
        p = os.path.join(os.path.dirname(out_path), "_kit_tile.png")
        sc.render.filepath = p
        bpy.ops.render.render(write_still=True)
        img = pl.load_image(p, "sRGB")[::-1].copy()
        os.remove(p)
        return img

    # main view (azimuth, elevation) per item; inset = (lowest fraction of the
    # height to include, azimuth, elevation) - a top view for blades, a close-up
    # for the tiny heads
    views = {"leaf_canopy_a": (-75, 18), "leaf_canopy_b": (-75, 18), "leaf_canopy_c": (-75, 18),
             "flower_star": (-58, 50)}
    for nm in ("leaf_round_a", "leaf_round_b", "leaf_round_c", "leaf_serrated_a", "leaf_serrated_b",
               "leaf_clover"):
        views[nm] = (-150, 42)
    insets = {"flower_white_small": (0.72, -58, 50), "seedhead_a": (0.68, -58, 30)}
    for nm in ("leaf_round_a", "leaf_round_b", "leaf_round_c", "leaf_serrated_a", "leaf_serrated_b",
               "leaf_clover", "flower_star"):
        insets[nm] = (0.0, -90, 89)
    tiles = []
    for nm, tris, size, lo, hi, cmin, cmax, rf, _ in rows_info:
        for o in objs.values():
            o.hide_render = o.name != nm
        o = objs[nm]
        vs = np.array([v.co[:] for v in o.data.vertices])
        img = shoot(vs, *views.get(nm, (-58, 38)), 360)
        if nm in insets:
            frac, iaz, iel = insets[nm]
            zc = vs[:, 2].min() + frac * (vs[:, 2].max() - vs[:, 2].min())
            ins = shoot(vs[vs[:, 2] >= zc], iaz, iel, 150)
            img[360 - 156:360 - 4, 360 - 156:360 - 4] = 0.9
            img[360 - 155:360 - 5, 360 - 155:360 - 5] = ins
        bar = np.full((40, 360, 3), 0.12, np.float32)
        pl.draw_text(bar, nm, 6, 4, 2)
        pl.draw_text(bar, "%d TRIS  %.1fX%.1fX%.1f CM" % (tris, size[0], size[2], size[1]), 6, 22, 2,
                     (0.85, 0.95, 0.75))
        tiles.append(np.concatenate([bar, img], 0))
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 0.2))
    rows = [np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)]
    sheet = np.concatenate(rows, 0)
    pl.save_image(out_path, sheet[::-1], "PNG")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    render = "--no-render" not in argv
    t0 = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    atlas = build_atlas()
    pl.save_image(ATLAS_PATH, atlas, "WEBP", 92, "sRGB")
    log("atlas %s %.1f KB" % (os.path.basename(ATLAS_PATH), os.path.getsize(ATLAS_PATH) / 1024))
    mat = make_material()
    items = build_items()
    objs = [to_object(k, mat) for k in items]
    names = [k.name for k in items]
    export_glb(objs, GLB_PATH)
    log("exported %s %.1f KB in %.1fs" % (GLB_PATH, os.path.getsize(GLB_PATH) / 1024, time.time() - t0))
    ok, report, rows, total = validate_glb(GLB_PATH, names)
    log("GLB check:", "OK" if ok else "PROBLEMS", report)
    log("%-20s %5s  %-22s %-14s %-14s" % ("node", "tris", "size cm (x, z, y-up)", "flex R min..max", "shade G"))
    for nm, tris, size, lo, hi, cmin, cmax, rf, cacc in rows:
        log("%-20s %5d  %5.1f x %5.1f x %5.1f   R %.2f..%.2f (root %.2f) G %.2f..%.2f B %.0f..%.0f  ymin %.2f" % (
            nm, tris, size[0], size[2], size[1], cmin[0], cmax[0], rf, cmin[1], cmax[1], cmin[2], cmax[2], lo[1]))
    log("total triangles", total)
    problems, objs_in = reimport_check(GLB_PATH, names)
    log("re-import check:", "OK" if not problems else problems)
    if render:
        preview(objs_in, ATLAS_PATH, os.path.join(pl.CAPTURE_DIR, "kit_preview.png"), rows)
        log("preview written")
    log("done in %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
