"""
stonelib.py - helpers for the 'stone' (S07) and 'finale' (S10) scene sets of Silva.

Import-only (build_stone.py / build_finale.py import it next to common.py, which stays untouched).
Everything is deterministic; randomness only from explicit seeds.

seed emblem     the exact construction of assets-src/emblem/seed.svg in seed units (height 100, y up):
                almond = intersection of two discs r 62.5 centred at (-37.5, 0) / (37.5, 0); channel 6 wide,
                tilted 18 deg clockwise from vertical through (4, 0); left piece keeps the top tip, right piece
                the bottom tip; corners rounded r 1 (morphological opening: the outline itself is unchanged).
                seed_outline(piece, offset)   CCW boundary of the piece offset by `offset` units (+ = outward;
                                              exact offset curves: erode the convex core, dilate by a disc)
                seed_sdf2(Q)                  2D signed distance (units) to the union of both pieces
                svg_seed_paths(path)          independent parse of the two <path>s in seed.svg (for checks)
emblem meshes   EmblemFrame                   local frame of an emblem on a (slightly curved) face
                emblem_cutter(...)            closed cutter mesh per piece: chamfered walls + flat caps or a
                                              domed floor (material slot 1) -> boolean DIFFERENCE
                boolean_difference, split_by_material, delete_faces
rock tools      plane_field, chip, carve_tube, surface_samples, project_to_surface (on common.SDF grids)
field tools     resample_sdf (finer grid, trilinear), displace_band (custom displacement in a band),
                top_heightfield / ball_closing / fill_heightfield (rolling-ball fill of narrow pits),
                CellField + poisson_surface (Voronoi cushion cells: F1 / F2 / nearest cell per point),
                decimate_vgroup (collapse with locked vertices), dilate_mask (vertex rings)
previews        lit previews (EEVEE, sun along a key empty, AO from COLOR_0.b), Workbench silhouettes,
                numpy image helpers for comparison sheets (load/save/outline/grid/polyline)
"""

import math
import os

import bpy
import numpy as np
from mathutils import Vector, Matrix
from mathutils import geometry as mgeo

import common as C

# --------------------------------------------------------------------------------------------------
# seed emblem: exact construction (seed units, y up)
# --------------------------------------------------------------------------------------------------

SEED_R = 62.5
SEED_CX = 37.5
SEED_CH_W = 6.0
SEED_CH_DEG = 18.0
SEED_CH_P = np.array([4.0, 0.0])
SEED_CORNER_R = 1.0
SEED_HEIGHT = 100.0

_A = np.array([-SEED_CX, 0.0])      # disc whose arc is the almond's RIGHT boundary
_B = np.array([SEED_CX, 0.0])       # disc whose arc is the LEFT boundary
_S, _Cs = math.sin(math.radians(SEED_CH_DEG)), math.cos(math.radians(SEED_CH_DEG))
CH_DIR = np.array([_S, _Cs])        # channel centre line direction (up, top leaning right)
CH_N = np.array([_Cs, -_S])         # unit normal of the channel line, pointing right
PIECES = ("left", "right")


def _components(piece, e):
    """CCW boundary components of the core K(e) (piece eroded by e units).
    ('disc', centre, radius) or ('line', point, outward normal)."""
    rad = SEED_R - e
    half = 0.5 * SEED_CH_W + e
    if piece == "left":                      # A-arc (P1 -> top tip), B-arc (tip -> P2), channel edge
        return [("disc", _A, rad), ("disc", _B, rad), ("line", SEED_CH_P - half * CH_N, CH_N)]
    if piece == "right":                     # A-arc (bottom tip -> Q1), channel edge, B-arc (Q2 -> tip)
        return [("disc", _A, rad), ("line", SEED_CH_P + half * CH_N, -CH_N), ("disc", _B, rad)]
    raise ValueError(piece)


def _inside(comp, p, tol=1e-7):
    if comp[0] == "disc":
        return np.linalg.norm(p - comp[1]) <= comp[2] + tol
    return float((p - comp[1]) @ comp[2]) <= tol


def _boundary_hits(c1, c2):
    """Intersections of the boundaries of two components."""
    if c1[0] == "line" and c2[0] == "line":
        return []
    if c1[0] == "line":
        c1, c2 = c2, c1
    if c2[0] == "disc":                      # circle-circle
        p0, r0, p1, r1 = c1[1], c1[2], c2[1], c2[2]
        d = float(np.linalg.norm(p1 - p0))
        a = (r0 * r0 - r1 * r1 + d * d) / (2 * d)
        h = math.sqrt(max(r0 * r0 - a * a, 0.0))
        m = p0 + a * (p1 - p0) / d
        perp = np.array([-(p1 - p0)[1], (p1 - p0)[0]]) / d
        return [m + h * perp, m - h * perp]
    c, r = c1[1], c1[2]                      # circle-line
    q, n = c2[1], c2[2]
    t_dir = np.array([-n[1], n[0]])
    w = q - c
    b = float(w @ t_dir)
    cc = float(w @ w) - r * r
    disc = b * b - cc
    if disc < 0:
        return []
    s = math.sqrt(disc)
    return [q + (-b + s) * t_dir, q + (-b - s) * t_dir]


def _corners(comps):
    """corner i = end of component i = start of component i+1."""
    n = len(comps)
    out = []
    for i in range(n):
        a, b, other = comps[i], comps[(i + 1) % n], comps[(i + 2) % n]
        cand = [p for p in _boundary_hits(a, b) if _inside(other, p, 1e-6)]
        if not cand:
            raise RuntimeError("seed corner not found")
        out.append(cand[0])
    return out


def _normal_at(comp, p):
    if comp[0] == "disc":
        v = p - comp[1]
        return v / np.linalg.norm(v)
    return comp[2]


def _ang(v):
    return math.atan2(v[1], v[0])


def seed_outline(piece, offset=0.0, step=0.25, r=SEED_CORNER_R, min_corner_pts=8, return_tags=False):
    """CCW boundary polyline (N, 2) of a seed piece (seed units), offset by `offset` (+ outward).
    Rounded with corner radius r: shape = K(r) (+) disc(r); offset o -> K(max(r, -o)) (+) disc(max(0, r + o)).
    return_tags: also an int array (0 = straight / arc component, 1 = corner fillet)."""
    e = max(r, -offset)
    rho = max(0.0, r + offset)
    comps = _components(piece, e)
    corners = _corners(comps)
    n = len(comps)
    pts, tags = [], []
    for i in range(n):
        start, end, comp = corners[i - 1], corners[i], comps[i]
        if comp[0] == "disc":
            c, rad = comp[1], comp[2]
            a0, a1 = _ang(start - c), _ang(end - c)
            da = (a1 - a0) % (2 * math.pi)
            R2 = rad + rho
            k = max(2, int(math.ceil(da * R2 / step)))
            a = a0 + da * np.arange(k) / k
            pts.append(c + R2 * np.stack([np.cos(a), np.sin(a)], 1))
        else:
            nn = comp[2]
            L = float(np.linalg.norm(end - start))
            k = max(1, int(math.ceil(L / step)))
            t = np.arange(k) / k
            pts.append(start + rho * nn + t[:, None] * (end - start))
        tags.append(np.zeros(len(pts[-1]), int))
        if rho > 0:
            n0, n1 = _normal_at(comp, end), _normal_at(comps[(i + 1) % n], end)
            a0, a1 = _ang(n0), _ang(n1)
            da = (a1 - a0) % (2 * math.pi)
            k = max(min_corner_pts, int(math.ceil(da * rho / step)))
            a = a0 + da * np.arange(k) / k
            pts.append(end + rho * np.stack([np.cos(a), np.sin(a)], 1))
            tags.append(np.ones(k, int))
    P = np.vstack(pts)
    return (P, np.concatenate(tags)) if return_tags else P


def seed_corner_points(piece, e=SEED_CORNER_R):
    return _corners(_components(piece, e))


def seed_sdf2(Q, r=SEED_CORNER_R):
    """Approximate signed distance (seed units) from 2D points Q (..., 2) to the union of the two rounded
    pieces (exact inside and near edges, a lower bound outside near corners)."""
    Q = np.asarray(Q, float)
    out = None
    for piece in PIECES:
        d = None
        for comp in _components(piece, r):
            if comp[0] == "disc":
                di = np.linalg.norm(Q - comp[1], axis=-1) - comp[2]
            else:
                di = (Q - comp[1]) @ comp[2]
            d = di if d is None else np.maximum(d, di)
        d = d - r
        out = d if out is None else np.minimum(out, d)
    return out


def polyline_dist(Q, poly):
    """Unsigned distance from points Q (N, 2) to a closed polyline (M, 2)."""
    Q = np.asarray(Q, float)
    A = np.asarray(poly, float)
    B = np.roll(A, -1, 0)
    best = np.full(len(Q), np.inf)
    E = B - A
    L2 = np.maximum((E * E).sum(1), 1e-30)
    for i0 in range(0, len(A), 256):
        a, e, l2 = A[i0:i0 + 256], E[i0:i0 + 256], L2[i0:i0 + 256]
        w = Q[:, None, :] - a[None, :, :]
        t = np.clip((w * e[None]).sum(-1) / l2[None], 0.0, 1.0)
        d = np.linalg.norm(w - t[..., None] * e[None], axis=-1)
        best = np.minimum(best, d.min(1))
    return best


def svg_seed_paths(svg_path):
    """Independent check: parse the two <path d="..."> of seed.svg (M / A / Z only) and sample them.
    Returns [(N, 2) arrays] in seed units with y UP (the SVG has y down)."""
    import re
    txt = open(svg_path, encoding="utf-8").read()
    out = []
    for d in re.findall(r'<path\s+d="([^"]+)"', txt):
        toks = re.findall(r"[MAZmaz]|-?\d*\.?\d+(?:e-?\d+)?", d)
        i, pts, cur, start = 0, [], None, None
        while i < len(toks):
            t = toks[i]
            if t == "M":
                cur = np.array([float(toks[i + 1]), float(toks[i + 2])])
                start = cur.copy()
                pts.append(cur.copy())
                i += 3
            elif t == "A":
                rx, ry, rot, large, sweep, x, y = [float(v) for v in toks[i + 1:i + 8]]
                end = np.array([x, y])
                pts.extend(_svg_arc(cur, end, rx, int(large), int(sweep))[1:])
                cur = end
                i += 8
            elif t in "Zz":
                pts.append(start.copy())
                i += 1
            else:
                i += 1
        P = np.array(pts)
        P[:, 1] *= -1.0
        out.append(P)
    return out


def _svg_arc(p0, p1, r, large, sweep, n=96):
    """Sample an SVG circular arc (endpoint parametrisation, y down)."""
    mid = 0.5 * (p0 + p1)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    h = math.sqrt(max(r * r - (0.5 * L) ** 2, 0.0))
    perp = np.array([-d[1], d[0]]) / L
    cands = [mid + h * perp, mid - h * perp]
    for c in cands:
        a0 = math.atan2(p0[1] - c[1], p0[0] - c[0])
        a1 = math.atan2(p1[1] - c[1], p1[0] - c[0])
        da = (a1 - a0) % (2 * math.pi) if sweep else -((a0 - a1) % (2 * math.pi))
        if (abs(da) > math.pi) == bool(large):
            a = a0 + da * np.linspace(0, 1, n)
            return c + r * np.stack([np.cos(a), np.sin(a)], 1)
    raise RuntimeError("svg arc")


# --------------------------------------------------------------------------------------------------
# polar ring sampling (each piece is convex: rays from an interior point cross every offset curve once)
# --------------------------------------------------------------------------------------------------


def piece_centre(piece):
    P = seed_outline(piece, 0.0, step=0.5)
    x, y = P[:, 0], P[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cr = x * yn - xn * y
    A = cr.sum() / 2.0
    return np.array([((x + xn) * cr).sum() / (6 * A), ((y + yn) * cr).sum() / (6 * A)])


def ring_angles(piece, centre, ds=1.0, corner_pts=6):
    """Ray angles (sorted, radians) from `centre`: arc-length spaced on the offset-0 outline (spacing ds
    units) plus `corner_pts` per fillet."""
    P, tags = seed_outline(piece, 0.0, step=0.02, return_tags=True)
    seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    keep = np.zeros(len(P), bool)
    # arc-length spacing on the non-fillet parts
    last = -1e9
    for i in range(len(P)):
        if tags[i] == 0 and s[i] - last >= ds:
            keep[i] = True
            last = s[i]
    # fillets: corner_pts evenly in each run of tag 1
    runs = np.split(np.arange(len(P)), np.nonzero(np.diff(tags))[0] + 1)
    for run in runs:
        if tags[run[0]] == 1:
            idx = run[np.linspace(0, len(run) - 1, corner_pts + 2).round().astype(int)[1:-1]]
            keep[idx] = True
            keep[max(run[0] - 1, 0)] = True
            keep[min(run[-1] + 1, len(P) - 1)] = True
    Q = P[keep] - centre
    ang = np.sort(np.arctan2(Q[:, 1], Q[:, 0]))
    # drop near-duplicates
    d = np.diff(np.concatenate([ang, [ang[0] + 2 * math.pi]]))
    return ang[d > 1e-5]


def ring_at(piece, offset, centre, angles):
    """Points (K, 2) of the offset outline along rays from `centre` at `angles`."""
    P = seed_outline(piece, offset, step=0.03)
    Q = P - centre
    a = np.arctan2(Q[:, 1], Q[:, 0])
    r = np.linalg.norm(Q, axis=1)
    o = np.argsort(a)
    a, r = a[o], r[o]
    a_ext = np.concatenate([a[-1:] - 2 * math.pi, a, a[:1] + 2 * math.pi])
    r_ext = np.concatenate([r[-1:], r, r[:1]])
    rr = np.interp(angles, a_ext, r_ext)
    return centre + rr[:, None] * np.stack([np.cos(angles), np.sin(angles)], 1)


# --------------------------------------------------------------------------------------------------
# emblem frame and cutter meshes
# --------------------------------------------------------------------------------------------------


class EmblemFrame:
    """Local frame of an emblem on a face.

    origin   world point: emblem centre (seed (0, 0)) on the reference surface
    normal   out of the face (towards the viewer)
    up_hint  world direction that becomes the emblem's up before `rot_deg`
    height_m emblem height (tip to tip) in metres -> scale s = height_m / 100 (m per seed unit)
    rot_deg  clockwise rotation as seen by the viewer (top leans right)
    surf     optional fn(x_m, y_m) -> height (m, along normal) of the reference surface above the plane
    """

    def __init__(self, origin, normal, up_hint, height_m, rot_deg=0.0, surf=None):
        n = C.normalize(np.asarray(normal, float))
        up = np.asarray(up_hint, float)
        ey0 = C.normalize(up - (up @ n) * n)
        ex0 = np.cross(ey0, n)
        a = math.radians(rot_deg)
        self.ex = math.cos(a) * ex0 - math.sin(a) * ey0
        self.ey = math.cos(a) * ey0 + math.sin(a) * ex0
        self.n = n
        self.o = np.asarray(origin, float)
        self.s = height_m / SEED_HEIGHT
        self.height = height_m
        self.surf = surf

    def world(self, Q, w):
        """Seed-unit points Q (..., 2) and heights w (m along n above the reference surface) -> world."""
        Q = np.asarray(Q, float)
        x, y = Q[..., 0] * self.s, Q[..., 1] * self.s
        h = np.asarray(w, float) + (self.surf(x, y) if self.surf is not None else 0.0)
        return (self.o + x[..., None] * self.ex + y[..., None] * self.ey + h[..., None] * self.n)

    def local(self, P):
        """World points (N, 3) -> (seed-unit (N, 2), height above the plane (m))."""
        d = np.asarray(P, float) - self.o
        return np.stack([d @ self.ex, d @ self.ey], -1) / self.s, d @ self.n

    def matrix(self):
        M = np.eye(4)
        M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = self.ex, self.ey, self.n, self.o
        return M


def floor_profile(delta_m, round_h=0.0025, round_w=0.007, dome_h=0.003, dome_w=0.05):
    """Height (m) of a domed floor above its edge at inset distance delta_m: a rounded tile edge
    (quarter-ellipse round_h x round_w) plus a gentle dome."""
    t = np.clip(np.asarray(delta_m, float) / round_w, 0.0, 1.0)
    edge = round_h * np.sqrt(np.maximum(1.0 - (1.0 - t) ** 2, 0.0))
    return edge + dome_h * C.smoothstep(round_w, dome_w, delta_m)


def _cdt_fill(ring2d, interior2d):
    """Triangulate the polygon `ring2d` (K, 2, CCW) with extra interior points. Returns (verts (M, 2),
    tris (T, 3)); the first K verts are the ring in order."""
    K = len(ring2d)
    pts = [Vector((float(x), float(y))) for x, y in ring2d] + [Vector((float(x), float(y))) for x, y in interior2d]
    edges = [(i, (i + 1) % K) for i in range(K)]
    faces = [list(range(K))]
    res = mgeo.delaunay_2d_cdt(pts, edges, faces, 1, 1e-9)
    vco, _e, f, orig_v = res[0], res[1], res[2], res[3]
    # map output verts back to input indices where possible
    out_v = np.array([[v.x, v.y] for v in vco])
    order = np.full(len(vco), -1, int)
    for oi, ov in enumerate(orig_v):
        if len(ov):
            order[oi] = min(ov)
    # build verts list: input order first, then any new verts
    n_in = len(pts)
    new_ids = [oi for oi in range(len(vco)) if order[oi] < 0]
    remap = np.empty(len(vco), int)
    for oi in range(len(vco)):
        remap[oi] = order[oi] if order[oi] >= 0 else n_in + new_ids.index(oi)
    verts = np.vstack([np.array([[p.x, p.y] for p in pts])] + ([out_v[new_ids]] if new_ids else []))
    tris = []
    for face in f:
        fl = [remap[i] for i in face]
        for k in range(1, len(fl) - 1):
            tris.append((fl[0], fl[k], fl[k + 1]))
    tris = np.array(tris, int)
    # orient CCW
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    cr = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    tris[cr < 0] = tris[cr < 0][:, ::-1]
    return verts, tris


def _interior_points(ring2d, spacing, margin):
    """Hex-grid points inside the polygon, at least `margin` from its boundary."""
    lo, hi = ring2d.min(0), ring2d.max(0)
    pts = []
    row = 0
    y = lo[1] + spacing * 0.5
    while y < hi[1]:
        x0 = lo[0] + (0.5 * spacing if row % 2 else 0.0)
        xs = np.arange(x0, hi[0], spacing)
        pts.append(np.stack([xs, np.full_like(xs, y)], 1))
        y += spacing * 0.866
        row += 1
    P = np.vstack(pts) if pts else np.zeros((0, 2))
    if not len(P):
        return P
    inside = C.poly_sdf(P, ring2d) < 0
    P = P[inside]
    if len(P):
        P = P[polyline_dist(P, ring2d) > margin]
    return P


def emblem_cutter(name, coll, frame, mode="recess", depth=0.025, bevel=0.003, lip=0.004, above=0.06,
                  thickness=None, floor=None, ds=0.8, mats=(None, None), floor_rings_m=None):
    """Closed cutter mesh for both seed pieces (boolean DIFFERENCE from a rock).

    mode 'recess': walls from above the face down to `depth` (m below the reference surface) with a 45 deg
                   chamfer of `bevel` m at the lip, closed by a domed floor (material slot 1).
    mode 'through': walls through a stone of `thickness` (m, back face at -thickness) with chamfers on both
                   faces; flat caps outside the stone (slot 0).
    Profiles are in (offset m outward, height m along the frame normal); every ring of a piece is sampled
    along the same rays from the piece centre (quads between rings).
    Returns (obj, info) with info['floor_faces'] (count) and per-piece ring data.
    """
    s = frame.s
    floor = dict(round_h=0.0025, round_w=0.007, dome_h=0.003, dome_w=0.05, **(floor or {}))
    if floor_rings_m is None:
        floor_rings_m = [0.0004, 0.001, 0.0019, 0.0032, 0.005, 0.0075, 0.011]
    o_lip = bevel + lip
    if mode == "recess":
        prof = [(o_lip, above), (o_lip, lip), (0.0, -bevel), (0.0, -depth)]
    elif mode == "through":
        t = float(thickness)
        prof = [(o_lip, above), (o_lip, lip), (0.0, -bevel), (0.0, -t + bevel), (o_lip, -t - lip),
                (o_lip, -t - above)]
    else:
        raise ValueError(mode)
    V, F, M = [], [], []
    nv = 0
    info = dict(pieces={})
    for piece in PIECES:
        c = piece_centre(piece)
        ang = ring_angles(piece, c, ds=ds)
        K = len(ang)
        rings = []
        for (o_m, w) in prof:
            q = ring_at(piece, o_m / s, c, ang)
            rings.append(frame.world(q, np.full(K, w)))
        base = nv
        for P in rings:
            V.append(P)
            nv += K
        jj = np.arange(K)
        for r in range(len(rings) - 1):
            a0, a1 = base + r * K, base + (r + 1) * K
            F.append(np.stack([a0 + jj, a0 + (jj + 1) % K, a1 + (jj + 1) % K, a1 + jj], 1))
            M.append(np.zeros(K, int))
        # top cap (flat, outside the rock)
        q_top = ring_at(piece, prof[0][0] / s, c, ang)
        verts2, tris = _cdt_fill(q_top, _interior_points(q_top, 6.0, 2.0))
        extra = verts2[K:]
        if len(extra):
            V.append(frame.world(extra, np.full(len(extra), prof[0][1])))
        ids = np.concatenate([base + np.arange(K), nv + np.arange(len(extra))])
        nv += len(extra)
        F.append(ids[tris][:, ::-1])           # cap faces the +normal side; orientation fixed globally
        M.append(np.zeros(len(tris), int))
        last = base + (len(rings) - 1) * K
        if mode == "recess":
            # domed floor: inset rings (same rays) + CDT interior
            prev = last
            for dm in floor_rings_m:
                q = ring_at(piece, -dm / s, c, ang)
                h = -depth + floor_profile(dm, **floor)
                V.append(frame.world(q, np.full(K, h)))
                cur = nv
                nv += K
                F.append(np.stack([prev + jj, prev + (jj + 1) % K, cur + (jj + 1) % K, cur + jj], 1))
                M.append(np.ones(K, int))
                prev = cur
            q_in = ring_at(piece, -floor_rings_m[-1] / s, c, ang)
            interior = _interior_points(q_in, 2.2, 1.0)
            verts2, tris = _cdt_fill(q_in, interior)
            extra = verts2[K:]
            if len(extra):
                outline0 = seed_outline(piece, 0.0, step=0.1)
                dm = polyline_dist(extra, outline0) * s
                V.append(frame.world(extra, -depth + floor_profile(dm, **floor)))
            ids = np.concatenate([prev + np.arange(K), nv + np.arange(len(extra))])
            nv += len(extra)
            F.append(ids[tris])
            M.append(np.ones(len(tris), int))
            info["pieces"][piece] = dict(rays=K, interior=len(extra))
        else:
            q_bot = ring_at(piece, prof[-1][0] / s, c, ang)
            verts2, tris = _cdt_fill(q_bot, _interior_points(q_bot, 6.0, 2.0))
            extra = verts2[K:]
            if len(extra):
                V.append(frame.world(extra, np.full(len(extra), prof[-1][1])))
            ids = np.concatenate([last + np.arange(K), nv + np.arange(len(extra))])
            nv += len(extra)
            F.append(ids[tris])
            M.append(np.zeros(len(tris), int))
            info["pieces"][piece] = dict(rays=K)
    Vall = np.vstack(V)
    me = C.mesh_from_numpy(name, Vall, F, smooth=False)
    mats_flat = np.concatenate(M).astype(np.int32)
    obj = C.object_from_mesh(name, me, coll)
    for m in mats:
        me.materials.append(m)
    # consistent outward orientation: recalc normals through bmesh
    _recalc_normals_outside(me)
    me.polygons.foreach_set("material_index", mats_flat)
    me.update()
    info["floor_faces"] = int((mats_flat == 1).sum())
    info["verts"] = len(Vall)
    return obj, info


def _recalc_normals_outside(me):
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(me)
    bm.free()
    me.update()


def boolean_difference(target, cutter, solver="EXACT", material_mode="TRANSFER"):
    mod = target.modifiers.new("silva_bool", "BOOLEAN")
    mod.operation = "DIFFERENCE"
    mod.operand_type = "OBJECT"
    mod.object = cutter
    mod.solver = solver
    try:
        mod.material_mode = material_mode
    except Exception:
        pass
    C.apply_modifiers(target)
    return target


def split_by_material(obj, mat_name, new_name, coll, new_material=None):
    """Move the faces using material `mat_name` into a new object (world-identical copy), remove them
    from obj. Returns the new object (or None)."""
    me = obj.data
    idx = [i for i, m in enumerate(me.materials) if m is not None and m.name == mat_name]
    if not idx:
        return None
    mi = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get("material_index", mi)
    sel = np.isin(mi, idx)
    if not sel.any():
        return None
    co, _, _, _ = C.mesh_arrays(me)
    lt = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get("loop_total", lt)
    ls = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get("loop_start", ls)
    lv = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", lv)
    faces = []
    for k in np.unique(lt[sel]):
        f = np.nonzero(sel & (lt == k))[0]
        faces.append(lv[ls[f][:, None] + np.arange(k)[None, :]])
    used = np.unique(np.concatenate([x.ravel() for x in faces]))
    remap = -np.ones(len(co), np.int64)
    remap[used] = np.arange(len(used))
    me2 = C.mesh_from_numpy(new_name, co[used], [remap[x] for x in faces], smooth=True)
    new = C.object_from_mesh(new_name, me2, coll, new_material)
    delete_faces(obj, sel)
    # drop the now unused slot(s)
    for i in sorted(idx, reverse=True):
        if i < len(me.materials):
            me.materials.pop(index=i)
    return new


def delete_faces(obj, mask):
    import bmesh
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    kill = [bm.faces[i] for i in np.nonzero(mask)[0]]
    bmesh.ops.delete(bm, geom=kill, context="FACES_ONLY")
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
    bm.to_mesh(me)
    bm.free()
    me.update()


def weld(obj, dist=1e-5):
    import bmesh
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=dist)
    bm.to_mesh(me)
    bm.free()
    me.update()


def keep_largest_part(obj, min_share=0.0):
    """Delete loose parts smaller than the largest one (boolean / VDB slivers)."""
    me = obj.data
    nv = len(me.vertices)
    ed = np.empty(len(me.edges) * 2, np.int32)
    me.edges.foreach_get("vertices", ed)
    ed = ed.reshape(-1, 2)
    parent = np.arange(nv)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in ed:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(nv)])
    ids, counts = np.unique(roots, return_counts=True)
    big = ids[np.argmax(counts)]
    keep_roots = set(ids[counts >= max(counts.max() * min_share, 1)]) if min_share > 0 else {big}
    vk = np.isin(roots, list(keep_roots))
    if vk.all():
        return 0
    lt = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get("loop_total", lt)
    ls = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get("loop_start", ls)
    lv = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", lv)
    face_v = lv[ls]
    kill = ~vk[face_v]
    delete_faces(obj, kill)
    return int(kill.sum())


# --------------------------------------------------------------------------------------------------
# rock tools on common.SDF grids
# --------------------------------------------------------------------------------------------------


def plane_dist(P, point, normal, concave=0.0, centre=None):
    """Signed distance-ish to a plane (positive on the normal side); concave > 0 bends the surface
    outwards quadratically away from `centre` (a slightly concave face seen from outside)."""
    n = C.normalize(np.asarray(normal, float))
    d = (P - np.asarray(point, float)) @ n
    if concave:
        c = np.asarray(centre if centre is not None else point, float)
        q = P - c
        t = q - (q @ n)[:, None] * n
        d = d - concave * np.einsum("ij,ij->i", t, t)
    return d


def convex_planes(P, planes, k=0.0):
    """Smooth intersection of half-spaces [(point, normal, concave, centre)] (negative inside)."""
    d = None
    for pl in planes:
        point, normal = pl[0], pl[1]
        conc = pl[2] if len(pl) > 2 else 0.0
        cen = pl[3] if len(pl) > 3 else None
        di = plane_dist(P, point, normal, conc, cen)
        d = di if d is None else C.smax(d, di, k)
    return d


def sdf_normals(sdf, P, h=None):
    h = h or sdf.voxel
    P = np.asarray(P, float)
    g = np.zeros_like(P)
    for a in range(3):
        e = np.zeros(3)
        e[a] = h
        g[:, a] = sdf.sample(P + e) - sdf.sample(P - e)
    return C.normalize(g)


def project_to_surface(sdf, P, iters=4):
    P = np.array(P, float)
    for _ in range(iters):
        d = sdf.sample(P)
        n = sdf_normals(sdf, P)
        P = P - d[:, None] * n
    return P


def surface_samples(sdf, n, rng, weight=None, band=None):
    """`n` random points on the zero level set (+ normals), optionally weighted by fn(P) >= 0."""
    band = band or sdf.voxel * 0.75
    idx, P = sdf.surface_points(band)
    if not len(P):
        return np.zeros((0, 3)), np.zeros((0, 3))
    w = np.ones(len(P)) if weight is None else np.clip(np.asarray(weight(P), float), 0.0, None)
    if w.sum() <= 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    pick = rng.choice(len(P), size=min(n, int((w > 0).sum())), replace=False, p=w / w.sum())
    Q = project_to_surface(sdf, P[pick] + rng.uniform(-0.5, 0.5, (len(pick), 3)) * sdf.voxel)
    return Q, sdf_normals(sdf, Q)


def chip(sdf, c, n, depth, radius, k=0.004):
    """Flake scar: remove the cap of a ball (centre c, radius) above the plane through c - n*depth.
    Leaves a planar facet with a sharp-ish scar edge (k = edge softness, m)."""
    c = np.asarray(c, float)
    n = C.normalize(np.asarray(n, float))
    c0 = c - n * depth

    def fn(P):
        return np.maximum(-((P - c0) @ n), np.linalg.norm(P - c, axis=1) - radius)

    m = radius + 2 * k + 3 * sdf.voxel
    sdf._combine(c - m, c + m, fn, k, mode="subtract")


def carve_tube(sdf, pts, radii, k=0.003):
    sdf.tube(pts, radii, k=k, mode="subtract")


def surface_walk(sdf, start, direction, steps, step_len, rng, jitter=0.35, bias=None):
    """Polyline walking on the surface from `start` (crack paths)."""
    p = project_to_surface(sdf, np.asarray(start, float)[None])[0]
    d = C.normalize(np.asarray(direction, float))
    out = [p]
    for _ in range(steps):
        nrm = sdf_normals(sdf, p[None])[0]
        d = d - (d @ nrm) * nrm
        d = C.normalize(d + jitter * C.normalize(rng.normal(size=3)) + (0 if bias is None else bias))
        d = d - (d @ nrm) * nrm
        d = C.normalize(d)
        p = project_to_surface(sdf, (p + d * step_len)[None])[0]
        out.append(p)
    return np.array(out)


# --------------------------------------------------------------------------------------------------
# field tools on common.SDF grids (finer resampling, band displacement, pit fill, cushion cells)
# --------------------------------------------------------------------------------------------------


def resample_sdf(src, voxel):
    """A new common.SDF over the same box with voxel size `voxel`, trilinear samples of `src`."""
    hi = src.origin + (np.array(src.shape) - 1) * src.voxel
    dst = C.SDF(src.origin, hi, voxel)
    nx, ny, nz = dst.shape
    ys = dst.origin[1] + np.arange(ny) * voxel
    zs = dst.origin[2] + np.arange(nz) * voxel
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    Y, Z = Y.ravel(), Z.ravel()
    for i in range(nx):
        P = np.stack([np.full(Y.shape, dst.origin[0] + i * voxel), Y, Z], 1)
        dst.d[i] = src.sample(P).reshape(ny, nz).astype(np.float32)
    return dst


def displace_band(sdf, fn, lo, hi):
    """d -= fn(P) for the voxels with lo < d < hi (the surface moves out by fn). The band must contain the
    whole displacement range (lo < min(-fn), hi > max(fn)) so no zero crossing appears at its border."""
    idx = np.nonzero((sdf.d > lo) & (sdf.d < hi))
    if not len(idx[0]):
        return 0
    n = len(idx[0])
    for c0 in range(0, n, 400000):
        sub = tuple(a[c0:c0 + 400000] for a in idx)
        P = sdf.origin + np.stack(sub, -1) * sdf.voxel
        sdf.d[sub] -= np.asarray(fn(P), np.float32)
    return n


def top_heightfield(sdf):
    """Highest zero crossing of the field per (x, y) column -> (z (nx, ny), has (nx, ny) bool)."""
    d = sdf.d
    nz = d.shape[2]
    inside = d < 0
    has = inside.any(2)
    k = nz - 1 - np.argmax(inside[:, :, ::-1], axis=2)
    k1 = np.minimum(k + 1, nz - 1)
    d0 = np.take_along_axis(d, k[..., None], 2)[..., 0].astype(np.float64)
    d1 = np.take_along_axis(d, k1[..., None], 2)[..., 0].astype(np.float64)
    ok = (d0 < 0) & (d1 > 0) & (k1 > k)
    t = np.zeros_like(d0)
    t[ok] = d0[ok] / (d0[ok] - d1[ok])
    z = sdf.origin[2] + (k + t) * sdf.voxel
    return np.where(has, z, -1e3), has


def _shift2(a, di, dj, fill):
    out = np.full_like(a, fill)
    n0, n1 = a.shape
    out[max(di, 0):n0 + min(di, 0), max(dj, 0):n1 + min(dj, 0)] = a[max(-di, 0):n0 + min(-di, 0),
                                                                    max(-dj, 0):n1 + min(-dj, 0)]
    return out


def ball_closing(h, radius, spacing, flat=False):
    """Morphological closing of a height map (grid spacing `spacing` m) with a ball of `radius` m: the
    surface a ball rolling on top of it can reach (pits and gaps narrower than the ball are filled with
    concave arcs, peaks stay); flat=True uses a flat disc instead. Empty columns hold a very low value."""
    rv = radius / spacing
    n = int(math.ceil(rv))
    offs = []
    for di in range(-n, n + 1):
        for dj in range(-n, n + 1):
            r2 = (di * di + dj * dj) * spacing * spacing
            if r2 <= radius * radius:
                offs.append((di, dj, 0.0 if flat else math.sqrt(radius * radius - r2) - radius))
    dil = np.full_like(h, -1e9)
    for di, dj, b in offs:
        dil = np.maximum(dil, _shift2(h, di, dj, -1e9) + b)
    clo = np.full_like(h, 1e9)
    for di, dj, b in offs:
        clo = np.minimum(clo, _shift2(dil, -di, -dj, 1e9) - b)
    return np.maximum(clo, h)


def depression_depth(h, valid):
    """Depth of the water that would pool on the height map h (priority flood, 4-neighbourhood) over the
    `valid` columns; valid columns next to invalid ones or the grid border drain freely. 0 elsewhere."""
    import heapq
    n0, n1 = h.shape
    valid = np.asarray(valid, bool)
    level = np.where(valid, np.inf, h).astype(np.float64)
    done = ~valid
    edge = np.zeros_like(valid)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        edge |= _shift2(~valid, di, dj, True)
    heap = []
    for i, j in zip(*np.nonzero(valid & edge)):
        level[i, j] = h[i, j]
        done[i, j] = True
        heap.append((float(h[i, j]), int(i), int(j)))
    heapq.heapify(heap)
    hl = h.tolist()
    dl = done.tolist()
    lv = level.tolist()
    while heap:
        z, i, j = heapq.heappop(heap)
        for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if 0 <= a < n0 and 0 <= b < n1 and not dl[a][b]:
                dl[a][b] = True
                w = max(hl[a][b], z)
                lv[a][b] = w
                heapq.heappush(heap, (w, a, b))
    level = np.array(lv)
    return np.where(valid, level - h, 0.0)


def fill_heightfield(sdf, z_fill, z_bottom, mask, k=0.02):
    """Smooth union of vertical slabs z_bottom..z_fill on the masked (x, y) columns (vertical distance:
    meant for shallow fills of a solid body, z_bottom below its old top)."""
    nz = sdf.shape[2]
    zs = sdf.origin[2] + np.arange(nz) * sdf.voxel
    ii, jj = np.nonzero(mask)
    if not len(ii):
        return 0
    top = z_fill[ii, jj][:, None]
    bot = z_bottom[ii, jj][:, None]
    f = np.maximum(zs[None, :] - top, bot - zs[None, :]).astype(np.float32)
    sdf.d[ii, jj, :] = C.smin(sdf.d[ii, jj, :], f, k).astype(np.float32)
    return len(ii)


def poisson_surface(P, radii, spacing, cell=0.4, seeded=0):
    """Dart throwing over candidate points P (N, 3) in the given order: keep a point when no kept point
    lies closer than spacing * (r_i + r_j). The first `seeded` points are kept unconditionally. Returns the
    kept indices (deterministic)."""
    buckets = {}
    kept = []
    for i in range(len(P)):
        p = P[i]
        r = radii[i]
        key = (int(math.floor(p[0] / cell)), int(math.floor(p[1] / cell)), int(math.floor(p[2] / cell)))
        ok = True
        if i < seeded:
            kept.append(i)
            buckets.setdefault(key, []).append(i)
            continue
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in buckets.get((key[0] + dx, key[1] + dy, key[2] + dz), ()):
                        q = P[j]
                        lim = spacing * (r + radii[j])
                        if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2 < lim * lim:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            kept.append(i)
            buckets.setdefault(key, []).append(i)
    return np.array(kept, np.int64)


class CellField:
    """Voronoi cells around centres (n, 3): distance to the nearest (F1) and second-nearest (F2) centre and
    the nearest index, searched in a hash grid of `cell` m (must exceed the typical F2)."""

    def __init__(self, centres, cell=0.4):
        self.c = np.asarray(centres, float)
        self.cell = float(cell)
        keys = np.floor(self.c / self.cell).astype(np.int64)
        self.buckets = {}
        for i, k in enumerate(keys):
            self.buckets.setdefault((int(k[0]), int(k[1]), int(k[2])), []).append(i)

    def _groups(self, P):
        """Yield (point indices, candidate centre indices) per hash bucket of the points."""
        n = len(P)
        if not n or not len(self.c):
            return
        keys = np.floor(P / self.cell).astype(np.int64)
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        ks = keys[order]
        brk = np.nonzero(np.any(np.diff(ks, axis=0) != 0, axis=1))[0] + 1
        starts = np.concatenate([[0], brk])
        ends = np.concatenate([brk, [n]])
        for s, e in zip(starts, ends):
            k = ks[s]
            cand = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        cand.extend(self.buckets.get((int(k[0]) + dx, int(k[1]) + dy, int(k[2]) + dz), ()))
            if cand:
                yield order[s:e], np.array(sorted(cand), np.int64)

    def query(self, P):
        P = np.asarray(P, float)
        n = len(P)
        F1 = np.full(n, 9.0)
        F2 = np.full(n, 9.0)
        I1 = np.full(n, -1, np.int64)
        for sel, cand in self._groups(P):
            D = np.sqrt(((P[sel, None, :] - self.c[None, cand, :]) ** 2).sum(-1))
            a = np.argmin(D, axis=1)
            F1[sel] = D[np.arange(len(sel)), a]
            I1[sel] = cand[a]
            if len(cand) > 1:
                D[np.arange(len(sel)), a] = np.inf
                F2[sel] = D.min(axis=1)
        return F1, F2, I1

    def top2(self, P, fn, empty=-1.0):
        """The two largest values of fn(D (n, m) distances, cand (m,) centre indices) per point over the
        centres of the neighbouring buckets, and the centre of the largest: (v1, v2, i1); `empty` / -1
        where there is no candidate."""
        P = np.asarray(P, float)
        n = len(P)
        v1 = np.full(n, float(empty))
        v2 = np.full(n, float(empty))
        i1 = np.full(n, -1, np.int64)
        for sel, cand in self._groups(P):
            D = np.sqrt(((P[sel, None, :] - self.c[None, cand, :]) ** 2).sum(-1))
            V = np.asarray(fn(D, cand), float)
            a = np.argmax(V, axis=1)
            rows = np.arange(len(sel))
            v1[sel] = V[rows, a]
            i1[sel] = cand[a]
            if len(cand) > 1:
                V[rows, a] = -np.inf
                v2[sel] = V.max(axis=1)
        return v1, v2, i1


def dilate_mask(mask, edges, iterations=1):
    """Grow a per-vertex bool mask by `iterations` edge rings."""
    m = np.asarray(mask, bool).copy()
    e0, e1 = edges[:, 0], edges[:, 1]
    for _ in range(int(iterations)):
        g = m.copy()
        g[e0[m[e1]]] = True
        g[e1[m[e0]]] = True
        m = g
    return m


def decimate_vgroup(obj, ratio, free_weights):
    """Collapse-decimate with a vertex group: weight 1 = free, 0 = locked (Blender: vertices with weight 0
    are never collapsed). `ratio` applies to the whole mesh. The group is removed afterwards."""
    w = np.clip(np.asarray(free_weights, float), 0.0, 1.0)
    vg = obj.vertex_groups.new(name="silva_decimate_free")
    q = np.round(w, 2)
    for value in np.unique(q):
        if value <= 0.0:
            continue
        vg.add(np.nonzero(q == value)[0].tolist(), float(value), "REPLACE")
    mod = obj.modifiers.new("silva_decimate_vg", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = float(ratio)
    mod.use_collapse_triangulate = True
    mod.vertex_group = vg.name
    mod.vertex_group_factor = 1.0
    C.apply_modifiers(obj)
    obj.vertex_groups.clear()
    return C.tri_count(obj)


# --------------------------------------------------------------------------------------------------
# cameras
# --------------------------------------------------------------------------------------------------


def look_quat(loc, target, roll_deg=0.0):
    """Camera quaternion looking from loc at target with world +Z up."""
    d = Vector(np.asarray(target, float) - np.asarray(loc, float)).normalized()
    q = d.to_track_quat("-Z", "Y")
    if roll_deg:
        from mathutils import Quaternion
        q = q @ Quaternion((0.0, 0.0, 1.0), math.radians(roll_deg))
    return q


# --------------------------------------------------------------------------------------------------
# previews (Blender only; the engine has its own look)
# --------------------------------------------------------------------------------------------------


def preview_ao_material(name, hex_color, roughness=0.85, ao_strength=1.0, emission=0.0):
    """Preview-only material: base colour x COLOR_0.b (baked AO), principled BSDF."""
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    try:
        mat.use_nodes = True
    except Exception:
        pass
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    attr = nt.nodes.new("ShaderNodeVertexColor")
    attr.layer_name = C.COLOR_ATTR
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    lin = C.hex_to_linear(hex_color)
    mix.inputs["Factor"].default_value = ao_strength
    mix.inputs["A"].default_value = (lin[0], lin[1], lin[2], 1.0)
    nt.links.new(attr.outputs["Color"], sep.inputs["Color"])
    comb = nt.nodes.new("ShaderNodeCombineColor")
    nt.links.new(sep.outputs["Blue"], comb.inputs["Red"])
    nt.links.new(sep.outputs["Blue"], comb.inputs["Green"])
    nt.links.new(sep.outputs["Blue"], comb.inputs["Blue"])
    nt.links.new(comb.outputs["Color"], mix.inputs["B"])
    nt.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = roughness
    if emission:
        bsdf.inputs["Emission Color"].default_value = (lin[0], lin[1], lin[2], 1.0)
        bsdf.inputs["Emission Strength"].default_value = emission
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    mat.diffuse_color = (lin[0], lin[1], lin[2], 1.0)
    return mat


class PreviewMaterials:
    """Swap the named placeholder materials for preview materials while rendering (restore after)."""

    def __init__(self, coll, mapping):
        self.coll = coll
        self.mapping = mapping          # prefix -> material
        self.saved = {}

    def __enter__(self):
        for o in self.coll.objects:
            if o.type != "MESH":
                continue
            self.saved[o.name] = [m for m in o.data.materials]
            for prefix, mat in self.mapping.items():
                if o.name.startswith(prefix):
                    o.data.materials.clear()
                    o.data.materials.append(mat)
                    break
        return self

    def __exit__(self, *exc):
        for name, mats in self.saved.items():
            o = self.coll.objects.get(name)
            if o is None:
                continue
            o.data.materials.clear()
            for m in mats:
                o.data.materials.append(m)
        return False


def setup_lit_preview(scene, key_dir, bg_hex, percent=50, sun_strength=4.0, sun_color=(1.0, 0.93, 0.82),
                      sun_angle_deg=3.0, fill_dir=None, fill_strength=0.5, fill_color=(0.75, 0.85, 1.0),
                      world_strength=1.0, samples=24, exposure=0.0):
    """EEVEE preview: background colour, one sun along key_dir (direction the light travels), optional
    fill sun; Standard view transform. Lights live in a hidden helper collection."""
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x, scene.render.resolution_y = C.FRAME_W, C.FRAME_H
    scene.render.resolution_percentage = int(percent)
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    try:
        scene.eevee.taa_render_samples = samples
        scene.eevee.use_shadows = True
        scene.eevee.shadow_ray_count = 2
        scene.eevee.shadow_step_count = 8
        scene.eevee.use_raytracing = False
    except Exception:
        pass
    vs = scene.view_settings
    vs.view_transform = "Standard"
    vs.look = "None"
    vs.exposure = exposure
    world = scene.world or bpy.data.worlds.new(scene.name + "_preview_world")
    scene.world = world
    lin = C.hex_to_linear(bg_hex)
    world.use_nodes = True
    bg = next((n for n in world.node_tree.nodes if n.type == "BACKGROUND"), None)
    if bg is not None:
        bg.inputs["Color"].default_value = (lin[0], lin[1], lin[2], 1.0)
        bg.inputs["Strength"].default_value = world_strength
    hc = bpy.data.collections.get("silva_preview_lights")
    if hc is None:
        hc = bpy.data.collections.new("silva_preview_lights")
    if scene.collection.children.get(hc.name) is None:
        scene.collection.children.link(hc)
    for o in list(hc.objects):
        bpy.data.objects.remove(o, do_unlink=True)

    def sun(name, d, strength, color, angle):
        ld = bpy.data.lights.get(name) or bpy.data.lights.new(name, "SUN")
        ld.energy = strength
        ld.color = color
        ld.angle = math.radians(angle)
        ob = bpy.data.objects.new(name, ld)
        ob.rotation_mode = "QUATERNION"
        ob.rotation_quaternion = Vector(d).normalized().to_track_quat("-Z", "Y")
        hc.objects.link(ob)
        return ob

    sun("silva_prev_key", key_dir, sun_strength, sun_color, sun_angle_deg)
    if fill_dir is not None and fill_strength > 0:
        sun("silva_prev_fill", fill_dir, fill_strength, fill_color, 20.0)
    return hc


def remove_preview_lights(scene):
    hc = bpy.data.collections.get("silva_preview_lights")
    if hc is None:
        return
    for o in list(hc.objects):
        data = o.data
        bpy.data.objects.remove(o, do_unlink=True)
        if data is not None and data.users == 0:
            bpy.data.lights.remove(data)
    if scene.collection.children.get(hc.name) is not None:
        scene.collection.children.unlink(hc)
    bpy.data.collections.remove(hc)


def render_mask(scene, cam_obj, path, objects, percent=50, occluders=()):
    """Silhouette of `objects` (white) on a transparent film, `occluders` drawn black (they hide parts of
    the objects); every other mesh hidden. Workbench, flat object colours. Mask = red channel > 0.5."""
    sh = scene.display.shading
    vs = scene.view_settings
    prev = (scene.render.engine, scene.render.film_transparent, sh.light, sh.color_type, sh.show_cavity,
            scene.render.image_settings.color_mode, scene.render.resolution_percentage, vs.view_transform)
    hidden, colors = [], {}
    keep = set(objects) | set(occluders)
    for o in scene.objects:
        if o.type == "MESH" and o not in keep and not o.hide_render:
            o.hide_render = True
            hidden.append(o)
    for o in keep:
        colors[o.name] = tuple(o.color)
        o.color = (1.0, 1.0, 1.0, 1.0) if o in objects else (0.0, 0.0, 0.0, 1.0)
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.film_transparent = True
    scene.render.resolution_percentage = int(percent)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    vs.view_transform = "Standard"
    sh.light = "FLAT"
    sh.color_type = "OBJECT"
    sh.show_cavity = False
    try:
        C.render_camera(scene, cam_obj, path)
    finally:
        for o in hidden:
            o.hide_render = False
        for o in keep:
            o.color = colors[o.name]
        (scene.render.engine, scene.render.film_transparent, sh.light, sh.color_type, sh.show_cavity,
         scene.render.image_settings.color_mode, scene.render.resolution_percentage, vs.view_transform) = prev
    return path


def load_rgba(path):
    img = bpy.data.images.load(path, check_existing=False)
    w, h = img.size
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    return px.reshape(h, w, 4)[::-1].copy()


def save_rgb(arr, path):
    h, w = arr.shape[:2]
    img = bpy.data.images.new("silva_sheet_tmp", width=w, height=h, alpha=False)
    rgba = np.ones((h, w, 4), np.float32)
    rgba[..., :3] = np.clip(arr[..., :3], 0, 1)
    img.pixels.foreach_set(rgba[::-1].ravel())
    img.filepath_raw = path
    img.file_format = "PNG"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save()
    bpy.data.images.remove(img)
    return path


def resize_to(a, h, w):
    """Box/nearest resize of an image array to (h, w)."""
    H, W = a.shape[:2]
    if (H, W) == (h, w):
        return a
    if H % h == 0 and W % w == 0:
        fy, fx = H // h, W // w
        return a[:h * fy, :w * fx].reshape(h, fy, w, fx, -1).mean((1, 3))
    yi = (np.arange(h) * H / h).astype(int)
    xi = (np.arange(w) * W / w).astype(int)
    return a[yi][:, xi]


def outline(mask, width=1):
    m = mask.astype(bool)
    e = np.zeros_like(m)
    for dy in range(-width, width + 1):
        for dx in range(-width, width + 1):
            if dx or dy:
                e |= np.roll(np.roll(m, dy, 0), dx, 1) != m
    return e & m


def draw_grid(img, step=0.1, col=(0.3, 0.45, 1.0), major=(1.0, 0.25, 0.25)):
    h, w = img.shape[:2]
    for i in range(1, int(round(1 / step))):
        c = major if abs(i * step - 0.5) < 1e-6 else col
        x = int(round(i * step * w))
        y = int(round(i * step * h))
        img[:, x, :3] = img[:, x, :3] * 0.4 + np.array(c) * 0.6
        img[y, :, :3] = img[y, :, :3] * 0.4 + np.array(c) * 0.6


def draw_polyline(img, uv, col, closed=False, width=1.0):
    """Draw a polyline given in frame fractions (u, v)."""
    h, w = img.shape[:2]
    P = np.asarray(uv, float) * np.array([w, h])
    if closed:
        P = np.vstack([P, P[:1]])
    for a, b in zip(P[:-1], P[1:]):
        n = int(max(abs(b - a).max(), 1) * 2)
        t = np.linspace(0, 1, n + 1)[:, None]
        Q = a + t * (b - a)
        for q in Q:
            x0, y0 = int(q[0] - width), int(q[1] - width)
            x1, y1 = int(q[0] + width) + 1, int(q[1] + width) + 1
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
            if x0 < x1 and y0 < y1:
                img[y0:y1, x0:x1, :3] = col


def draw_dot(img, u, v, r, col):
    h, w = img.shape[:2]
    cx, cy = u * w, v * h
    y0, y1 = int(max(0, cy - r - 1)), int(min(h, cy + r + 2))
    x0, x1 = int(max(0, cx - r - 1)), int(min(w, cx + r + 2))
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    sel = np.hypot(xx - cx, yy - cy) <= r
    img[y0:y1, x0:x1][sel, :3] = col


def label(img, text, x, y, col=(1.0, 1.0, 0.2), scale=2):
    """Tiny 3x5 bitmap font (digits, a few letters) for sheet labels."""
    font = {
        "0": "111101101101111", "1": "010110010010111", "2": "111001111100111", "3": "111001111001111",
        "4": "101101111001001", "5": "111100111001111", "6": "111100111101111", "7": "111001001001001",
        "8": "111101111101111", "9": "111101111001111", ".": "000000000000010", "-": "000000111000000",
        "s": "011100010001110", "t": "010111010010011", "m": "000111111101101", "a": "000011101101011",
        "i": "010000010010010", "n": "000110101101101", "o": "000010101101010", "u": "000101101101011",
        "r": "000110101100100", "e": "010101111100011", "f": "011100110100100", "c": "000011100100011",
        "p": "000110101110100", "l": "010010010010011", "g": "011101011001110", "h": "100100110101101",
        "d": "001001011101011", "b": "100100110101110", "k": "100101110101101", "w": "000101101111101",
        "x": "000101010010101", "y": "000101011001110", "v": "000101101101010", "z": "000111001100111",
        " ": "000000000000000", "_": "000000000000111", "/": "001001010100100", ":": "000010000010000",
    }
    h, w = img.shape[:2]
    cx = x
    for ch in text.lower():
        g = font.get(ch, font[" "])
        for i, bit in enumerate(g):
            if bit == "1":
                r, c = divmod(i, 3)
                y0, x0 = y + r * scale, cx + c * scale
                if 0 <= y0 < h - scale and 0 <= x0 < w - scale:
                    img[y0:y0 + scale, x0:x0 + scale, :3] = col
        cx += 4 * scale
