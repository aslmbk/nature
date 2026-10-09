"""
canopylib.py - helpers for the 'canyon' (S03) and 'canopy' (S08-S09) scene sets of Silva.

Import-only (build_canyon.py / build_canopy.py import it next to common.py, which stays untouched).
Everything is deterministic: randomness only from explicit integer seeds (numpy default_rng / hashes).

cameras         Cam (pose in screen terms: S(u, v, depth) -> world, project(P) -> (u, v, depth)),
                cam_loc_for(target, u, v, depth, pitch, yaw, roll): location that frames a world point
hashing         hash01(seed, *int_arrays) -> [0, 1) per integer tuple (block / plate ids)
rock tools      joint_insets(sdf, sets, seed, band)  layered / jointed rock: every block between two joint
                sets is set back by its own amount (+ grooves along the joints), so the surface breaks into
                slabs and ledges instead of a smooth blob; sdf_normals, surface_points
moss blobs      blob_field(...)  flattened lumps hugging chosen surface points (moss islands on ledges)
twigs           Twig, grow_twig(...), tube_mesh(...): thin branching tubes with U around / V along the grain
volumes         DensityGrid: splat weighted points, blur, march transmittance (volumetric AO / leaf shadow)
previews        flat colour previews (Workbench FLAT + per-vertex colour on preview copies; the exported
                objects are hidden from render meanwhile and never modified), alpha masks
images          load_rgba, save_rgb, down2, box_blur, fill_holes, outline, grid, dot, polyline
"""

import math
import os

import bpy
import numpy as np
from mathutils import Vector

import common as C

TV = math.tan(math.radians(C.FOV_V_DEG) / 2.0)
TH = TV * C.ASPECT

# --------------------------------------------------------------------------------------------------
# cameras in screen terms
# --------------------------------------------------------------------------------------------------


class Cam:
    """A camera pose: location + (pitch, yaw, roll) as in common.cam_quat (zero = look along +Y)."""

    def __init__(self, name, loc, pitch=0.0, yaw=0.0, roll=0.0, focus=3.0, t=None, extras=None):
        self.name = name
        self.loc = np.asarray(loc, float)
        self.pitch, self.yaw, self.roll = float(pitch), float(yaw), float(roll)
        self.focus = float(focus)
        self.t = t
        self.extras = dict(extras or {})

    @property
    def q(self):
        return C.cam_quat(self.pitch, self.yaw, self.roll)

    def basis(self):
        return C.cam_basis(self.q)

    def S(self, u, v, d):
        return C.screen_to_world(self.loc, self.q, u, v, d)

    def project(self, P):
        P = np.asarray(P, float)
        u, v, z = C.project_to_screen(self.loc, self.q, P.reshape(-1, 3))
        shp = P.shape[:-1]
        return u.reshape(shp), v.reshape(shp), z.reshape(shp)

    def frame_height(self, depth=None):
        return 2.0 * TV * (self.focus if depth is None else depth)


def cam_loc_for(target, u, v, depth, pitch=0.0, yaw=0.0, roll=0.0):
    """Camera location that shows world point `target` at frame (u, v) at view depth `depth`."""
    off = C.screen_to_world((0.0, 0.0, 0.0), C.cam_quat(pitch, yaw, roll), u, v, depth)
    return np.asarray(target, float) - off


# --------------------------------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------------------------------

_M32 = np.uint64(0xFFFFFFFF)


def hash01(seed, *ints):
    """Deterministic pseudo-random value in [0, 1) for each tuple of integer arrays (broadcast)."""
    h = np.uint64((int(seed) * 0x9E3779B1 + 0x632BE5AB) & 0xFFFFFFFF)
    out = None
    with np.errstate(over="ignore"):
        acc = np.zeros(np.broadcast(*[np.asarray(i) for i in ints]).shape if ints else (), np.uint64) + h
        for k, i in enumerate(ints):
            v = (np.asarray(i).astype(np.int64) + 0x40000).astype(np.uint64) & _M32
            acc = (acc ^ (v * np.uint64(0x85EBCA6B + 0x1000193 * k))) & _M32
            acc = (acc * np.uint64(0xC2B2AE35)) & _M32
            acc = acc ^ (acc >> np.uint64(15))
            acc = (acc * np.uint64(0x27D4EB2F)) & _M32
            acc = acc ^ (acc >> np.uint64(13))
        out = acc.astype(np.float64) / 4294967296.0
    return out


# --------------------------------------------------------------------------------------------------
# rock tools (on common.SDF grids)
# --------------------------------------------------------------------------------------------------


def surface_points(sdf, band):
    idx = np.nonzero(np.abs(sdf.d) < band)
    return idx, sdf.origin + np.stack(idx, -1) * sdf.voxel


def joint_insets(sdf, sets, seed, band=0.25, amp_fn=None):
    """Layered / jointed rock. Each joint set k: dict(normal, spacing, plate=m, block=m, groove=m,
    groove_w=m, warp=(amp in cells, freq 1/m), power) cuts space into slabs along `normal`. Every
    slab is set back by plate * h(slab) and every block (the cell of all sets together) by
    block * h(block); grooves sink the surface along the joint planes. The field moves inward only
    (insets >= 0), within `band` of the current surface. amp_fn(P) -> scale per point (0 = untouched).
    Returns the voxel indices touched and the per-voxel slab ids (for colouring if wanted)."""
    noise = C.Noise3(seed + 911)
    idx, P = surface_points(sdf, band)
    if not len(P):
        return idx, []
    inset = np.zeros(len(P))
    ids = []
    for k, js in enumerate(sets):
        n = C.normalize(np.asarray(js["normal"], float))
        sp = float(js["spacing"])
        w = (P @ n) / sp
        wa = js.get("warp")
        if wa:
            w = w + float(wa[0]) * noise.fbm(P * float(wa[1]) + 13.7 * (k + 1), 2)
        i = np.floor(w).astype(np.int64)
        f = w - i
        ids.append(i)
        pw = float(js.get("power", 1.0))
        if js.get("plate"):
            inset += float(js["plate"]) * hash01(seed + 31 * k, i) ** pw
        if js.get("wedge"):
            # every slab is a little wedge-shaped: its face slopes across the slab (catches light
            # differently from its neighbours); slope sign / size per slab
            sl = 2.0 * hash01(seed + 57 * k + 3, i) - 1.0
            inset += float(js["wedge"]) * (0.5 + sl * (f - 0.5))
        if js.get("bulge"):
            # rounded rib: the slab face falls off towards both joints (weathered blade cross-section)
            inset += float(js["bulge"]) * (2.0 * np.abs(f - 0.5)) ** 2
        if js.get("groove"):
            e = np.minimum(f, 1.0 - f) * sp
            inset += float(js["groove"]) * (1.0 - C.smoothstep(0.0, float(js.get("groove_w", 0.02)), e))
    blk = [js.get("block", 0.0) for js in sets]
    if any(blk) and ids:
        hb = hash01(seed + 7, *ids)
        inset += max(blk) * hb ** float(sets[0].get("power", 1.0))
    if amp_fn is not None:
        inset *= np.clip(amp_fn(P), 0.0, None)
    sdf.d[idx] += inset.astype(np.float32)
    return idx, ids


def components(me):
    """Connected components of a mesh (vertex labels, sizes)."""
    co, _, ed, _ = C.mesh_arrays(me)
    lab = np.arange(len(co))
    a, b = ed[:, 0], ed[:, 1]
    for _ in range(100000):            # min-label propagation with pointer jumping
        old = lab.copy()
        m = np.minimum(lab[a], lab[b])
        np.minimum.at(lab, a, m)
        np.minimum.at(lab, b, m)
        lab = lab[lab]
        lab = lab[lab]
        if np.array_equal(lab, old):
            break
    labels = np.unique(lab, return_inverse=True)[1]
    return labels, np.bincount(labels)


def drop_small_islands(obj, min_share=0.01):
    """Delete loose parts with fewer than min_share of the vertices (clamp / noise crumbs)."""
    import bmesh
    me = obj.data
    labels, sizes = components(me)
    small = sizes[labels] < min_share * len(labels)
    if not small.any():
        return 0, len(sizes)
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[bm.verts[i] for i in np.nonzero(small)[0]], context="VERTS")
    bm.to_mesh(me)
    bm.free()
    me.update()
    return int(small.sum()), len(sizes)


def sdf_normals(sdf, P, h=None):
    h = sdf.voxel if h is None else h
    g = np.zeros((len(P), 3))
    for a in range(3):
        e = np.zeros(3)
        e[a] = h
        g[:, a] = sdf.sample(P + e) - sdf.sample(P - e)
    return C.normalize(g)


def blob_field(points, normals, radii, thick, voxel, noise=None, k=0.02, pad=0.06, lumps=None):
    """Flattened ellipsoids (radius r in the tangent plane, half-thickness t along the normal) at the
    given surface points, smooth-unioned into one SDF; lumps = [(amp, freq, octaves), ...]."""
    P = np.asarray(points, float)
    lo = P.min(0) - radii.max() - pad
    hi = P.max(0) + radii.max() + pad
    sdf = C.SDF(lo, hi, voxel)
    for p, n, r, t in zip(P, normals, radii, thick):
        R = C.rot_from_axes(z_axis=n)
        sdf.ellipsoid(p, (r, r * 0.85, t), R, k=k)
    if noise is not None:
        for (amp, freq, octv) in (lumps or [(0.25 * float(np.mean(thick)), 9.0, 3)]):
            sdf.noise(noise, amp, freq, octaves=octv)
    return sdf


# --------------------------------------------------------------------------------------------------
# twigs
# --------------------------------------------------------------------------------------------------


class Twig:
    """One tube of a branching cluster: centreline X (n, 3), radii r (n,), arc length s (n,)."""

    def __init__(self, X, r, level, parent=-1, s_parent=0.0, base_dist=0.0):
        self.X = np.asarray(X, float)
        self.r = np.asarray(r, float)
        seg = np.linalg.norm(np.diff(self.X, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        self.L = float(self.s[-1])
        self.level = int(level)
        self.parent = int(parent)
        self.s_parent = float(s_parent)
        self.base_dist = float(base_dist)     # path length from the cluster base to this twig's start
        T = np.zeros_like(self.X)
        T[1:-1] = self.X[2:] - self.X[:-2]
        T[0] = self.X[1] - self.X[0]
        T[-1] = self.X[-1] - self.X[-2]
        self.T = C.normalize(T)

    def at(self, s):
        """Point, tangent and radius at arc length s."""
        s = float(np.clip(s, 0.0, self.L))
        p = np.array([np.interp(s, self.s, self.X[:, k]) for k in range(3)])
        t = C.normalize(np.array([np.interp(s, self.s, self.T[:, k]) for k in range(3)]))
        return p, t, float(np.interp(s, self.s, self.r))


def resample_polyline(P, spacing, min_pts=3):
    P = np.asarray(P, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(min_pts, int(math.ceil(s[-1] / max(spacing, 1e-6))) + 1)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, P[:, k]) for k in range(3)], 1)


def grow_path(rng, start, direction, length, n_seg, wobble, bias_fn=None, smooth=True):
    """Polyline from `start` along a direction that drifts: d <- normalize(d + wobble * N(0, 1) +
    bias_fn(p, d, frac)). Returns a Catmull-Rom smoothed dense polyline."""
    d = C.normalize(np.asarray(direction, float))
    p = np.asarray(start, float)
    pts = [p.copy()]
    step = length / n_seg
    for k in range(n_seg):
        jit = rng.normal(size=3) * wobble
        b = bias_fn(p, d, k / n_seg) if bias_fn is not None else 0.0
        d = C.normalize(d + jit + b)
        p = p + d * step
        pts.append(p.copy())
    pts = np.array(pts)
    if smooth and len(pts) >= 3:
        pts = C.catmull_rom_chain(pts, n_per_seg=6)
    return pts


def perp_basis(t):
    t = C.normalize(np.asarray(t, float))
    a = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    n = C.normalize(np.cross(a, t))
    b = np.cross(t, n)
    return n, b


def tube_mesh(twigs, sides_by_level, ring_spacing_by_level, origin, uv_m_per_unit=C.UV_M_PER_UNIT,
              base_cap_levels=(0,)):
    """Tubes along twigs (one ring per resampled centreline point, a cone tip at the end, a flat cap
    at the base for levels in base_cap_levels; child tubes start inside their parent so they need no
    cap). Coordinates relative to `origin`. Returns (V, quads, tris, uv_loops, per-vertex info dict:
    twig index, s, L, level, radius, centre point)."""
    origin = np.asarray(origin, float)
    Vs, Q, Tr, UVq, UVt = [], [], [], [], []
    info = dict(twig=[], s=[], L=[], level=[], radius=[], centre=[])
    off = 0
    for ti, tw in enumerate(twigs):
        K = int(sides_by_level[min(tw.level, len(sides_by_level) - 1)])
        sp = float(ring_spacing_by_level[min(tw.level, len(ring_spacing_by_level) - 1)])
        X = resample_polyline(tw.X, sp)
        seg = np.linalg.norm(np.diff(X, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        r = np.interp(s, tw.s * (s[-1] / max(tw.L, 1e-9)), tw.r)
        T = np.zeros_like(X)
        T[1:-1] = X[2:] - X[:-2]
        T[0] = X[1] - X[0]
        T[-1] = X[-1] - X[-2]
        T = C.normalize(T)
        n0, _ = perp_basis(T[0])
        Rf, Bf = C.rotation_minimizing_frames(X, T, up_hint=n0)
        n = len(X)
        th = np.arange(K) * (2.0 * math.pi / K) + (ti * 0.618 % 1.0) * 2.0 * math.pi / K
        ring = (X[:, None, :] + r[:, None, None] * (np.cos(th)[None, :, None] * Rf[:, None, :] +
                                                     np.sin(th)[None, :, None] * Bf[:, None, :]))
        V = ring.reshape(-1, 3)
        vid = np.arange(n * K).reshape(n, K) + off
        i0, j0 = np.meshgrid(np.arange(n - 1), np.arange(K), indexing="ij")
        j1 = (j0 + 1) % K
        q = np.stack([vid[i0, j0], vid[i0, j1], vid[i0 + 1, j1], vid[i0 + 1, j0]], -1).reshape(-1, 4)
        U = np.arange(K + 1) / K
        Vv = s / uv_m_per_unit
        uvq = np.stack([np.stack([U[j0], Vv[i0]], -1), np.stack([U[j0 + 1], Vv[i0]], -1),
                        np.stack([U[j0 + 1], Vv[i0 + 1]], -1), np.stack([U[j0], Vv[i0 + 1]], -1)],
                       2).reshape(-1, 2)
        Q.append(q)
        UVq.append(uvq)
        # tip cone
        tip = X[-1] + T[-1] * max(r[-1] * 1.5, 0.0008)
        tip_id = off + n * K
        jj = np.arange(K)
        Tr.append(np.stack([vid[-1, jj], vid[-1, (jj + 1) % K], np.full(K, tip_id)], -1))
        UVt.append(np.stack([np.stack([U[jj], np.full(K, Vv[-1])], -1),
                             np.stack([U[jj + 1], np.full(K, Vv[-1])], -1),
                             np.stack([U[jj] + 0.5 / K, np.full(K, Vv[-1] + 0.002)], -1)], 1).reshape(-1, 2))
        extra = [tip[None, :]]
        ex_s = [s[-1] + max(r[-1] * 1.5, 0.0008)]
        ex_r = [0.0]
        ex_c = [tip[None, :]]
        if tw.level in base_cap_levels:
            cid = tip_id + 1
            Tr.append(np.stack([vid[0, (jj + 1) % K], vid[0, jj], np.full(K, cid)], -1))
            UVt.append(np.stack([np.stack([U[jj + 1], np.zeros(K)], -1), np.stack([U[jj], np.zeros(K)], -1),
                                 np.stack([U[jj] + 0.5 / K, np.full(K, -0.002)], -1)], 1).reshape(-1, 2))
            extra.append((X[0] - T[0] * r[0] * 0.3)[None, :])
            ex_s.append(0.0)
            ex_r.append(r[0])
            ex_c.append(X[0][None, :])
        Vs.append(V)
        Vs.extend(extra)
        nv = n * K + len(extra)
        info["twig"].append(np.full(nv, ti))
        info["s"].append(np.concatenate([np.repeat(s, K), ex_s]))
        info["L"].append(np.full(nv, s[-1]))
        info["level"].append(np.full(nv, tw.level))
        info["radius"].append(np.concatenate([np.repeat(r, K), ex_r]))
        info["centre"].append(np.vstack([np.repeat(X, K, axis=0)] + ex_c))
        off += nv
    V = np.vstack(Vs) - origin
    quads = np.vstack(Q)
    tris = np.vstack(Tr)
    uv = np.vstack(UVq + UVt)
    for k in info:
        info[k] = np.concatenate(info[k])
    info["centre"] = info["centre"].reshape(-1, 3) - origin
    return V, quads, tris, uv, info


def mesh_object(name, coll, V, quads, tris, uv, material=None, location=(0.0, 0.0, 0.0), smooth=True):
    me = C.mesh_from_numpy(name, V, [quads, tris], smooth=smooth)
    C.set_uvs(me, uv)
    obj = C.object_from_mesh(name, me, coll, material)
    obj.location = Vector(tuple(float(x) for x in location))
    return obj


# --------------------------------------------------------------------------------------------------
# volumes: density grid for volumetric occlusion / leaf shadow
# --------------------------------------------------------------------------------------------------


class DensityGrid:
    def __init__(self, lo, hi, voxel):
        self.voxel = float(voxel)
        self.lo = np.asarray(lo, float)
        self.shape = tuple(int(math.ceil((np.asarray(hi, float)[k] - self.lo[k]) / self.voxel)) + 1 for k in range(3))
        self.g = np.zeros(self.shape, np.float64)

    def splat(self, P, w):
        """Trilinear splat of weights w at points P (per cubic metre after division by voxel^3)."""
        f = (np.asarray(P, float) - self.lo) / self.voxel
        i = np.floor(f).astype(np.int64)
        f = f - i
        shp = np.array(self.shape)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    ii = i + np.array([dx, dy, dz])
                    ok = np.all((ii >= 0) & (ii < shp), 1)
                    wt = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                    np.add.at(self.g, (ii[ok, 0], ii[ok, 1], ii[ok, 2]), (w * wt)[ok])
        return self

    def blur(self, passes=2):
        g = self.g
        for _ in range(passes):
            for ax in range(3):
                g = 0.25 * (np.roll(g, 1, ax) + 2.0 * g + np.roll(g, -1, ax))
        self.g = g
        return self

    def sample(self, P):
        f = (np.asarray(P, float) - self.lo) / self.voxel
        shp = np.array(self.shape) - 1
        inside = np.all((f >= 0) & (f <= shp), 1)
        f = np.clip(f, 0, shp - 1e-6)
        i = np.floor(f).astype(np.int64)
        f = f - i
        out = np.zeros(len(f))
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    wt = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                    out += wt * self.g[i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz]
        return np.where(inside, out, 0.0)

    def transmittance(self, P, dirs, sigma, length, steps=16, start=0.0):
        """exp(-sigma * integral of density) from each point along its direction(s).
        dirs: (3,) one direction for all, or (N, 3)."""
        P = np.asarray(P, float)
        D = np.broadcast_to(np.asarray(dirs, float), P.shape)
        ds = (length - start) / steps
        acc = np.zeros(len(P))
        for k in range(steps):
            t = start + (k + 0.5) * ds
            acc += self.sample(P + D * t) * ds
        return np.exp(-sigma * acc)

    def occlusion(self, P, nrm, sigma, length, n_dirs=12, steps=12, start=0.02):
        """Mean transmittance over a cosine-weighted hemisphere around nrm (volumetric AO, 1 = open)."""
        H = C.hemisphere_dirs(n_dirs)
        nrm = C.normalize(np.asarray(nrm, float))
        a = np.where(np.abs(nrm[:, 2:3]) < 0.9, np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]))
        t = C.normalize(np.cross(a, nrm))
        b = np.cross(nrm, t)
        acc = np.zeros(len(P))
        for d in H:
            D = d[0] * t + d[1] * b + d[2] * nrm
            acc += self.transmittance(P, D, sigma, length, steps, start)
        return acc / len(H)


# --------------------------------------------------------------------------------------------------
# flat colour previews (Workbench FLAT, colour attribute on preview copies)
# --------------------------------------------------------------------------------------------------


def preview_copy(src, coll, rgb, name=None):
    """Linked-free copy of a mesh object with a 'Col' colour attribute = rgb (N, 3) linear."""
    me = src.data.copy()
    me.name = (name or src.name) + "__pv"
    for a in list(me.color_attributes):
        me.color_attributes.remove(a)
    rgba = np.ones((len(me.vertices), 4), np.float32)
    rgba[:, :3] = np.clip(rgb, 0.0, 1.0)
    at = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    at.data.foreach_set("color", rgba.ravel())
    me.color_attributes.active_color_index = 0
    me.color_attributes.render_color_index = 0
    obj = bpy.data.objects.new((name or src.name) + "__pv", me)
    obj.matrix_world = src.matrix_world.copy()
    coll.objects.link(obj)
    return obj


def preview_mesh(name, coll, V, faces, rgb):
    me = C.mesh_from_numpy(name, V, faces, smooth=False)
    rgba = np.ones((len(V), 4), np.float32)
    rgba[:, :3] = np.clip(rgb, 0.0, 1.0)
    at = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    at.data.foreach_set("color", rgba.ravel())
    obj = bpy.data.objects.new(name, me)
    coll.objects.link(obj)
    return obj


class FlatPreview:
    """Context: hide the exported meshes from render, show preview copies in a temporary collection.
    Restores everything (and deletes the copies) on exit."""

    def __init__(self, scene, coll_name):
        self.scene = scene
        self.coll_name = coll_name

    def __enter__(self):
        self.coll = bpy.data.collections.new(self.coll_name)
        self.scene.collection.children.link(self.coll)
        self.hidden = []
        for o in self.scene.objects:
            if o.type == "MESH" and not o.hide_render and o.users_collection and o.users_collection[0] != self.coll:
                o.hide_render = True
                self.hidden.append(o.name)
        return self

    def __exit__(self, *exc):
        for o in list(self.coll.objects):
            me = o.data
            bpy.data.objects.remove(o, do_unlink=True)
            if me is not None and me.users == 0:
                bpy.data.meshes.remove(me)
        bpy.data.collections.remove(self.coll)
        for n in self.hidden:
            o = bpy.data.objects.get(n)
            if o is not None:
                o.hide_render = False
        return False


def setup_flat(scene, percent=50, bg=(0.0, 0.0, 0.0), aa="16"):
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x, scene.render.resolution_y = C.FRAME_W, C.FRAME_H
    scene.render.resolution_percentage = int(percent)
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    vs = scene.view_settings
    vs.view_transform = "Standard"
    vs.look = "None"
    vs.exposure = 0.0
    vs.gamma = 1.0
    world = scene.world
    if world is None:
        world = bpy.data.worlds.get(scene.name + "_preview_world") or bpy.data.worlds.new(scene.name + "_preview_world")
        scene.world = world
    world.color = bg
    sh = scene.display.shading
    sh.light = "FLAT"
    sh.color_type = "VERTEX"
    sh.show_cavity = False
    sh.show_shadows = False
    sh.show_specular_highlight = False
    sh.background_type = "WORLD"
    scene.display.render_aa = aa
    return scene


def render_masks(scene, cams, outdir, objects, percent=50, prefix="mask"):
    """White silhouettes of `objects` (others hidden) on a transparent film; returns {cam name: path}."""
    sh = scene.display.shading
    prev = (scene.render.engine, scene.render.film_transparent, sh.light, sh.color_type,
            scene.render.image_settings.color_mode, scene.render.resolution_percentage)
    keep = {o.name for o in objects}
    hidden = []
    for o in scene.objects:
        if o.type == "MESH" and o.name not in keep and not o.hide_render:
            o.hide_render = True
            hidden.append(o.name)
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.film_transparent = True
    scene.render.resolution_percentage = int(percent)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    sh.light = "FLAT"
    sh.color_type = "SINGLE"
    sh.single_color = (1.0, 1.0, 1.0)
    out = {}
    try:
        for cam in cams:
            path = os.path.join(outdir, f"{prefix}_{cam.name}.png")
            C.render_camera(scene, cam, path)
            out[cam.name] = path
    finally:
        for n in hidden:
            bpy.data.objects[n].hide_render = False
        (scene.render.engine, scene.render.film_transparent, sh.light, sh.color_type,
         scene.render.image_settings.color_mode, scene.render.resolution_percentage) = prev
    return out


def lambert(nrm, light_dir, wrap=0.0):
    """Lit fraction for normals vs the direction the light travels (wrap > 0 softens the terminator)."""
    l = -C.normalize(np.asarray(light_dir, float))
    return np.clip((np.asarray(nrm, float) @ l + wrap) / (1.0 + wrap), 0.0, 1.0)


def shadow_rays(bvh, P, nrm, light_dir, bias=0.004, max_dist=20.0):
    """1 where the point sees the light (no hit towards -light_dir), else 0."""
    l = -C.normalize(np.asarray(light_dir, float))
    org = (np.asarray(P, float) + np.asarray(nrm, float) * bias).tolist()
    lv = Vector(tuple(l))
    cast = bvh.ray_cast
    return np.array([0.0 if cast(Vector(o), lv, max_dist)[0] is not None else 1.0 for o in org])


# --------------------------------------------------------------------------------------------------
# images (numpy; Blender image IO)
# --------------------------------------------------------------------------------------------------


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
    img.save()
    bpy.data.images.remove(img)
    return path


def down2(a):
    h, w = a.shape[:2]
    return a[:h // 2 * 2, :w // 2 * 2].reshape(h // 2, 2, w // 2, 2, -1).mean((1, 3))


def box_blur(a, r):
    k = 2 * r + 1
    p = np.pad(a, r, mode="edge")
    c = np.cumsum(np.cumsum(p, 0), 1)
    c = np.pad(c, ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def fill_holes(m):
    """Fill every background region that does not touch the frame border."""
    bg = ~m
    reach = np.zeros_like(m)
    reach[0, :] = bg[0, :]
    reach[-1, :] = bg[-1, :]
    reach[:, 0] = bg[:, 0]
    reach[:, -1] = bg[:, -1]
    for _ in range(4000):
        r = reach.copy()
        r[1:] |= reach[:-1]
        r[:-1] |= reach[1:]
        r[:, 1:] |= reach[:, :-1]
        r[:, :-1] |= reach[:, 1:]
        r &= bg
        if (r == reach).all():
            break
        reach = r
    return ~reach


def outline(mask, width=1):
    m = mask.astype(bool)
    e = np.zeros_like(m)
    for dy in range(-width, width + 1):
        for dx in range(-width, width + 1):
            if dx or dy:
                e |= np.roll(np.roll(m, dy, 0), dx, 1) != m
    return e & m


def grid(img, step=0.1, col=(0.3, 0.45, 1.0), major=(1.0, 0.25, 0.25)):
    h, w = img.shape[:2]
    for i in range(1, int(round(1 / step))):
        c = major if abs(i * step - 0.5) < 1e-6 else col
        x = int(round(i * step * w))
        y = int(round(i * step * h))
        img[:, x, :3] = img[:, x, :3] * 0.4 + np.array(c) * 0.6
        img[y, :, :3] = img[y, :, :3] * 0.4 + np.array(c) * 0.6


def dot(img, u, v, r, col):
    h, w = img.shape[:2]
    cx, cy = u * w, v * h
    y0, y1 = int(max(0, cy - r - 1)), int(min(h, cy + r + 2))
    x0, x1 = int(max(0, cx - r - 1)), int(min(w, cx + r + 2))
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    sel = np.hypot(xx - cx, yy - cy) <= r
    img[y0:y1, x0:x1][sel, :3] = col


def polyline(img, uv, col, r=1.0):
    uv = np.asarray(uv, float)
    h, w = img.shape[:2]
    for a, b in zip(uv[:-1], uv[1:]):
        n = int(max(2, np.hypot((b[0] - a[0]) * w, (b[1] - a[1]) * h)))
        for t in np.linspace(0, 1, n):
            p = a + (b - a) * t
            dot(img, p[0], p[1], r, col)


def iou(a, b, valid=None):
    if valid is None:
        valid = np.ones_like(a, bool)
    inter = (a & b & valid).sum()
    union = ((a | b) & valid).sum()
    return float(inter) / max(float(union), 1.0)
