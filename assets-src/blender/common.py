"""
common.py - shared, reusable Blender helpers for the Silva nature assets.

Target: Blender 5.1 (bpy, mathutils, numpy, openvdb are all bundled).
Every helper is deterministic: randomness only comes from explicit integer seeds.

Coordinate convention (see CLAUDE.md "Asset contract"): metres, Blender Z-up,
cameras look roughly along +Y, export with +Y-up glTF.

Main entry points
-----------------
scene / data      ensure_scene, activate_scene, ensure_collection, purge_collection,
                  mesh_from_numpy, object_from_mesh, mesh_arrays, decimate, tri_count
noise             Noise3(seed)(P) / .fbm(P, octaves) / .ridged(P, octaves)   (vectorised numpy)
lumpy masses      SDF(bmin, bmax, voxel): .ellipsoid .round_cone .tube .carve_ellipsoid
                  .clip_plane .noise .to_object   (smooth-union field meshed with OpenVDB)
                  .clip_view(cam_loc, cam_q, poly_uv, invert=False, weight=fn)  visual-hull clamp:
                  the mass never leaves (or, inverted, never enters) a screen polygon of a camera,
                  so reference silhouettes are hit exactly; poly_sdf(Q, V) is the 2D helper
swept trunks      sweep_trunk(...) -> Sweep  (variable radius, varying cross-section,
                  twisted bark ridges / folds / knots, UV: U around (integer wraps),
                  V along the grain, 1 UV unit = 0.5 m; caps incl. dict(kind="splinter") for
                  splintered broken ends)
join              join_objects(name, coll, objs)  merge meshes (keeps UVMap; COLOR_0 written after)
moss cushions     moss_cushions(sweep, mask, ...)  lumpy cushions hugging a swept trunk;
                  blur_grid(mask) softens a trunk-grid mask first (no grid-aligned borders)
vertex colours    surface_convexity, build_bvh, ray_ao, rim_factor, write_color0
                  (COLOR_0: R density, G length, B AO, A 1 - linear float)
uv                box_uvs(mesh, metres_per_unit)
cameras / keys    cam_quat, screen_to_world, project_to_screen, make_camera, make_key_empty
materials         preview_material (named placeholders mat_wood / mat_moss / ...)
previews          setup_preview_render, render_camera
export / check    GLB_EXPORT_OPTIONS, export_glb, read_glb, validate_glb

Usage from another build script (inside Blender, live or headless):

    import sys, importlib; sys.path.insert(0, "<repo>/assets-src/blender")
    import common as C; importlib.reload(C)
    scene = C.ensure_scene("branch"); coll = C.ensure_collection(scene, "branch")
    sw = C.sweep_trunk("wood_jbranch", coll, ctrl_pts, radii, twist=0.8, ridges=dict(count=22))
    ...
    C.export_glb(scene, ".../public/nature/models/branch.glb")
    print(C.validate_glb(".../branch.glb"))
"""

import json
import math
import os
import re
import struct

import bpy
import numpy as np
from mathutils import Euler, Matrix, Quaternion, Vector
from mathutils.bvhtree import BVHTree

try:  # bundled with Blender >= 4.x
    import openvdb as _vdb
except Exception:  # pragma: no cover
    _vdb = None

# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------

FRAME_W, FRAME_H = 1440, 1020
ASPECT = FRAME_W / FRAME_H
FOV_V_DEG = 35.0
UV_M_PER_UNIT = 0.5          # 1 UV unit = 0.5 m (contract)
COLOR_ATTR = "Col"           # exported as COLOR_0
UV_NAME = "UVMap"            # exported as TEXCOORD_0

PALETTE = {
    "sage": "#5B654F",
    "sage_low": "#757F69",
    "moss": "#5D6B22",
    "leaf": "#6F8034",
    "wood": "#796449",
    "stone": "#8A816F",
}

# --------------------------------------------------------------------------------------
# small utils
# --------------------------------------------------------------------------------------


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def hex_to_linear(hexstr):
    h = hexstr.lstrip("#")
    rgb = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    return tuple(float(x) for x in srgb_to_linear(rgb))


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def normalize(v, axis=-1):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, 1e-12)


def log(*args):
    print("[silva]", *args, flush=True)


# --------------------------------------------------------------------------------------
# noise (vectorised, deterministic)
# --------------------------------------------------------------------------------------


def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


class Noise3:
    """Gradient (Perlin) noise in 3D, vectorised. Output roughly in [-1, 1]."""

    def __init__(self, seed=0):
        r = np.random.default_rng(int(seed))
        p = r.permutation(256).astype(np.int64)
        self.perm = np.concatenate([p, p])
        g = r.normal(size=(256, 3))
        self.grad = g / np.linalg.norm(g, axis=1, keepdims=True)

    def __call__(self, P):
        P = np.asarray(P, dtype=np.float64)
        shp = P.shape[:-1]
        P = P.reshape(-1, 3)
        fl = np.floor(P)
        f = P - fl
        i = fl.astype(np.int64) & 255
        u = _fade(f)
        perm, grad = self.perm, self.grad
        xi, yi, zi = i[:, 0], i[:, 1], i[:, 2]
        xf, yf, zf = f[:, 0], f[:, 1], f[:, 2]
        A = perm[xi] + yi
        B = perm[xi + 1] + yi
        AA = perm[A] + zi
        AB = perm[A + 1] + zi
        BA = perm[B] + zi
        BB = perm[B + 1] + zi

        def gd(h, x, y, z):
            g = grad[h]
            return g[:, 0] * x + g[:, 1] * y + g[:, 2] * z

        n000 = gd(perm[AA], xf, yf, zf)
        n100 = gd(perm[BA], xf - 1, yf, zf)
        n010 = gd(perm[AB], xf, yf - 1, zf)
        n110 = gd(perm[BB], xf - 1, yf - 1, zf)
        n001 = gd(perm[AA + 1], xf, yf, zf - 1)
        n101 = gd(perm[BA + 1], xf - 1, yf, zf - 1)
        n011 = gd(perm[AB + 1], xf, yf - 1, zf - 1)
        n111 = gd(perm[BB + 1], xf - 1, yf - 1, zf - 1)
        ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]
        x00 = n000 + ux * (n100 - n000)
        x10 = n010 + ux * (n110 - n010)
        x01 = n001 + ux * (n101 - n001)
        x11 = n011 + ux * (n111 - n011)
        y0 = x00 + uy * (x10 - x00)
        y1 = x01 + uy * (x11 - x01)
        return ((y0 + uz * (y1 - y0)) * 1.8).reshape(shp)

    def fbm(self, P, octaves=4, lacunarity=2.03, gain=0.5):
        P = np.asarray(P, dtype=np.float64)
        tot = 0.0
        amp = 1.0
        norm = 0.0
        for o in range(octaves):
            tot = tot + amp * self(P * (lacunarity ** o) + o * 17.317)
            norm += amp
            amp *= gain
        return tot / norm

    def ridged(self, P, octaves=3, lacunarity=2.1, gain=0.5):
        """Ridged multifractal-ish in [0, 1]; 1 on ridges."""
        P = np.asarray(P, dtype=np.float64)
        tot = 0.0
        amp = 1.0
        norm = 0.0
        for o in range(octaves):
            n = 1.0 - np.abs(self(P * (lacunarity ** o) + o * 31.7))
            tot = tot + amp * n * n
            norm += amp
            amp *= gain
        return tot / norm

    def line(self, s, freq=1.0, offset=0.0, octaves=3):
        """1D fbm along a parameter (used for variation along trunks)."""
        s = np.asarray(s, dtype=np.float64)
        P = np.stack([s * freq + offset, np.full_like(s, 0.37 + offset * 0.1), np.full_like(s, 0.71)], -1)
        return self.fbm(P, octaves)


# --------------------------------------------------------------------------------------
# scene / collections / datablocks
# --------------------------------------------------------------------------------------


def activate_scene(scene):
    win = bpy.context.window
    if win is not None and win.scene != scene:
        win.scene = scene
    return scene


def ensure_scene(name, activate=True):
    scene = bpy.data.scenes.get(name)
    if scene is None:
        scene = bpy.data.scenes.new(name)
    r = scene.render
    r.resolution_x, r.resolution_y = FRAME_W, FRAME_H
    r.resolution_percentage = 100
    r.pixel_aspect_x = r.pixel_aspect_y = 1.0
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    if activate:
        activate_scene(scene)
    return scene


def ensure_collection(scene, name, parent=None):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
    parent_coll = parent if parent is not None else scene.collection
    if parent_coll.children.get(coll.name) is None:
        parent_coll.children.link(coll)
    return coll


def _remove_datablock(data):
    if data is None or data.users > 0:
        return
    if isinstance(data, bpy.types.Mesh):
        bpy.data.meshes.remove(data)
    elif isinstance(data, bpy.types.Camera):
        bpy.data.cameras.remove(data)
    elif isinstance(data, bpy.types.Light):
        bpy.data.lights.remove(data)
    elif isinstance(data, bpy.types.Curve):
        bpy.data.curves.remove(data)
    elif isinstance(data, bpy.types.MetaBall):
        bpy.data.metaballs.remove(data)


def remove_object(name):
    obj = bpy.data.objects.get(name)
    if obj is None:
        return
    data = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    _remove_datablock(data)


def purge_collection(coll, names=None):
    """Delete objects in `coll` (all, or only `names`) and their orphaned data."""
    for obj in list(coll.objects):
        if names is not None and obj.name not in names:
            continue
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        _remove_datablock(data)


def _free_name(collection, name):
    """Make `name` available in a bpy.data collection (removes an unused block, renames a used one)."""
    old = collection.get(name)
    if old is None:
        return
    if old.users == 0:
        collection.remove(old)
    else:
        old.name = name + "__stale"


def mesh_from_numpy(name, verts, faces, smooth=True):
    """Create a mesh datablock from numpy arrays.

    verts: (N, 3) float; faces: (F, k) int array or a list of such arrays (mixed arity ok).
    """
    if isinstance(faces, np.ndarray):
        face_list = [faces]
    else:
        face_list = [np.asarray(f) for f in faces if f is not None and len(f)]
    sizes = np.concatenate([np.full(len(f), f.shape[1], np.int32) for f in face_list])
    flat = np.concatenate([f.reshape(-1) for f in face_list]).astype(np.int32)
    starts = np.zeros(len(sizes), np.int32)
    if len(sizes) > 1:
        starts[1:] = np.cumsum(sizes)[:-1]
    _free_name(bpy.data.meshes, name)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.loops.add(len(flat))
    me.polygons.add(len(sizes))
    me.vertices.foreach_set("co", np.asarray(verts, np.float32).ravel())
    me.polygons.foreach_set("loop_start", starts)
    me.polygons.foreach_set("vertices", flat)
    me.update(calc_edges=True)
    me.validate(clean_customdata=False)
    if smooth:
        me.shade_smooth()
    else:
        me.shade_flat()
    return me


def object_from_mesh(name, me, coll, material=None):
    remove_object(name)
    obj = bpy.data.objects.new(name, me)
    coll.objects.link(obj)
    if material is not None:
        me.materials.clear()
        me.materials.append(material)
    return obj


def mesh_arrays(me):
    """Return (co (N,3), vertex normals (N,3), edges (E,2), triangles (T,3)) as numpy arrays."""
    nv = len(me.vertices)
    co = np.empty(nv * 3, np.float32)
    me.vertices.foreach_get("co", co)
    nrm = np.empty(nv * 3, np.float32)
    me.vertex_normals.foreach_get("vector", nrm)
    ed = np.empty(len(me.edges) * 2, np.int32)
    me.edges.foreach_get("vertices", ed)
    me.calc_loop_triangles()
    tr = np.empty(len(me.loop_triangles) * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", tr)
    return (co.reshape(-1, 3).astype(np.float64), nrm.reshape(-1, 3).astype(np.float64),
            ed.reshape(-1, 2), tr.reshape(-1, 3))


def set_vertex_positions(me, co):
    me.vertices.foreach_set("co", np.asarray(co, np.float32).ravel())
    me.update()


def tri_count(obj_or_mesh):
    me = obj_or_mesh.data if isinstance(obj_or_mesh, bpy.types.Object) else obj_or_mesh
    me.calc_loop_triangles()
    return len(me.loop_triangles)


def apply_modifiers(obj):
    """Bake the evaluated mesh (all modifiers) into obj.data (keeps the datablock name)."""
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    eo = obj.evaluated_get(dg)
    new = bpy.data.meshes.new_from_object(eo, preserve_all_data_layers=True, depsgraph=dg)
    old = obj.data
    name = old.name
    mats = list(old.materials)
    obj.modifiers.clear()
    obj.data = new
    if old.users == 0:
        bpy.data.meshes.remove(old)
    new.name = name
    if not len(new.materials):
        for m in mats:
            new.materials.append(m)
    return obj


def decimate(obj, target_tris):
    """Collapse-decimate obj to about `target_tris` triangles (no-op if already below)."""
    cur = tri_count(obj)
    if cur <= target_tris:
        return cur
    mod = obj.modifiers.new("silva_decimate", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = float(target_tris) / float(cur)
    mod.use_collapse_triangulate = True
    apply_modifiers(obj)
    return tri_count(obj)


def neighbor_mean(values, edges, n):
    values = np.asarray(values, dtype=np.float64)
    e0, e1 = edges[:, 0], edges[:, 1]
    cnt = np.bincount(e0, minlength=n) + np.bincount(e1, minlength=n)
    cols = values.shape[1] if values.ndim == 2 else 1
    v2 = values.reshape(n, cols)
    out = np.zeros((n, cols))
    for c in range(cols):
        out[:, c] = (np.bincount(e0, weights=v2[e1, c], minlength=n) +
                     np.bincount(e1, weights=v2[e0, c], minlength=n))
    out /= np.maximum(cnt, 1)[:, None]
    return out if values.ndim == 2 else out[:, 0]


def taubin_smooth(co, edges, iterations=2, lam=0.5, mu=-0.53, fixed=None):
    """Shrink-free Laplacian smoothing (Taubin lambda/mu)."""
    n = len(co)
    co = np.array(co, dtype=np.float64)
    for _ in range(iterations):
        for f in (lam, mu):
            d = f * (neighbor_mean(co, edges, n) - co)
            if fixed is not None:
                d[fixed] = 0.0
            co += d
    return co


# --------------------------------------------------------------------------------------
# SDF modelling for lumpy organic masses
# --------------------------------------------------------------------------------------


def smin(a, b, k):
    """Polynomial smooth minimum (union). k = blend radius in metres."""
    if k <= 0.0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1.0 - h) + a * h - k * h * (1.0 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def sd_ellipsoid(q, r):
    k0 = np.linalg.norm(q / r, axis=-1)
    k1 = np.linalg.norm(q / (r * r), axis=-1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def sd_round_cone(p, a, b, r1, r2):
    """Exact SDF of a capsule with radius r1 at a and r2 at b (Inigo Quilez)."""
    ba = b - a
    l2 = float(ba @ ba)
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    pa = p - a
    y = pa @ ba
    z = y - l2
    xv = pa * l2 - y[:, None] * ba
    x2 = np.einsum("ij,ij->i", xv, xv)
    y2 = y * y * l2
    z2 = z * z * l2
    k = math.copysign(1.0, rr) * rr * rr * x2
    d3 = (np.sqrt(np.maximum(x2 * a2 * il2, 0.0)) + y * rr) * il2 - r1
    d1 = np.sqrt(x2 + z2) * il2 - r2
    d2 = np.sqrt(x2 + y2) * il2 - r1
    return np.where(np.sign(z) * a2 * z2 > k, d1, np.where(np.sign(y) * a2 * y2 < k, d2, d3))


def poly_sdf(Q, V):
    """Signed distance from 2D points Q (N, 2) to the closed polygon V (M, 2); negative inside."""
    Q = np.asarray(Q, float)
    V = np.asarray(V, float)
    d = np.full(len(Q), np.inf)
    inside = np.zeros(len(Q), bool)
    for i in range(len(V)):
        a = V[i]
        b = V[(i + 1) % len(V)]
        e = b - a
        w = Q - a
        t = np.clip((w @ e) / max(float(e @ e), 1e-30), 0.0, 1.0)
        dist = np.hypot(w[:, 0] - t * e[0], w[:, 1] - t * e[1])
        d = np.minimum(d, dist)
        cond = ((a[1] <= Q[:, 1]) & (Q[:, 1] < b[1])) | ((b[1] <= Q[:, 1]) & (Q[:, 1] < a[1]))
        dy = b[1] - a[1]
        xint = a[0] + (Q[:, 1] - a[1]) * e[0] / (dy if abs(dy) > 1e-30 else 1e-30)
        inside ^= cond & (Q[:, 0] < xint)
    return np.where(inside, -d, d)


def rot_from_axes(x_axis=None, z_axis=None):
    """3x3 matrix whose columns are the local axes (world from local)."""
    if z_axis is None:
        z_axis = (0, 0, 1)
    z = normalize(np.asarray(z_axis, float))
    x = np.asarray(x_axis if x_axis is not None else (1, 0, 0), float)
    x = x - (x @ z) * z
    if np.linalg.norm(x) < 1e-6:
        x = np.array([0.0, 1.0, 0.0]) - z[1] * z
    x = normalize(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], 1)


class SDF:
    """Signed distance field on a regular grid (negative inside), meshed with OpenVDB.

    Build lumpy organic masses as smooth unions of ellipsoids / round cones / tubes, carve
    folds with smooth subtraction, add fbm lumps in a narrow band, then `to_object`.
    """

    def __init__(self, bmin, bmax, voxel):
        self.voxel = float(voxel)
        self.origin = np.asarray(bmin, dtype=np.float64)
        bmax = np.asarray(bmax, dtype=np.float64)
        self.shape = tuple(int(math.ceil((bmax[i] - self.origin[i]) / self.voxel)) + 1 for i in range(3))
        self.d = np.full(self.shape, 1.0e3, dtype=np.float32)

    # ---- grid access
    def _box(self, lo, hi):
        lo = np.asarray(lo, float)
        hi = np.asarray(hi, float)
        shp = np.array(self.shape)
        i0 = np.clip(np.floor((lo - self.origin) / self.voxel).astype(int), 0, shp)
        i1 = np.clip(np.ceil((hi - self.origin) / self.voxel).astype(int) + 1, 0, shp)
        if np.any(i1 <= i0):
            return None, None
        sl = tuple(slice(int(i0[k]), int(i1[k])) for k in range(3))
        ax = [self.origin[k] + np.arange(i0[k], i1[k]) * self.voxel for k in range(3)]
        X, Y, Z = np.meshgrid(*ax, indexing="ij")
        return sl, np.stack([X, Y, Z], -1)

    def _combine(self, lo, hi, fn, k, mode="union"):
        sl, P = self._box(lo, hi)
        if sl is None:
            return
        d = fn(P.reshape(-1, 3)).reshape(P.shape[:-1]).astype(np.float32)
        cur = self.d[sl]
        if mode == "union":
            self.d[sl] = smin(cur, d, k)
        elif mode == "subtract":
            self.d[sl] = smax(cur, -d, k)
        elif mode == "intersect":
            self.d[sl] = smax(cur, d, k)

    # ---- primitives
    def ellipsoid(self, c, r, rot=None, k=0.0, mode="union"):
        c = np.asarray(c, float)
        r = np.asarray(r, float) if np.ndim(r) else np.array([r, r, r], float)
        R = np.eye(3) if rot is None else np.asarray(rot, float)
        ext = np.abs(R) @ r
        m = 2.0 * k + 3.0 * self.voxel
        self._combine(c - ext - m, c + ext + m, lambda P: sd_ellipsoid((P - c) @ R, r), k, mode)

    def sphere(self, c, r, k=0.0, mode="union"):
        self.ellipsoid(c, (r, r, r), None, k, mode)

    def round_cone(self, a, b, ra, rb, k=0.0, mode="union"):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        rmax = max(ra, rb)
        m = 2.0 * k + 3.0 * self.voxel + rmax
        lo = np.minimum(a, b) - m
        hi = np.maximum(a, b) + m
        self._combine(lo, hi, lambda P: sd_round_cone(P, a, b, ra, rb), k, mode)

    def tube(self, pts, radii, k=0.0, mode="union"):
        """Chain of round cones (hard union inside the chain, smooth `k` with the field)."""
        pts = np.asarray(pts, float)
        radii = np.asarray(radii, float)
        m = 2.0 * k + 3.0 * self.voxel + radii.max()
        lo = pts.min(0) - m
        hi = pts.max(0) + m

        def fn(P):
            d = np.full(len(P), 1e3)
            for i in range(len(pts) - 1):
                d = np.minimum(d, sd_round_cone(P, pts[i], pts[i + 1], radii[i], radii[i + 1]))
            return d

        self._combine(lo, hi, fn, k, mode)

    def carve_ellipsoid(self, c, r, rot=None, k=0.0):
        self.ellipsoid(c, r, rot, k, mode="subtract")

    def clip_plane(self, point, normal):
        """Keep the half-space dot(p - point, normal) <= 0 (hard intersection)."""
        point = np.asarray(point, float)
        n = normalize(np.asarray(normal, float))
        sl, P = self._box(self.origin, self.origin + np.array(self.shape) * self.voxel)
        d = ((P - point) @ n).astype(np.float32)
        self.d[sl] = np.maximum(self.d[sl], d)

    def offset(self, amount):
        """Grow (+) or shrink (-) the whole field."""
        self.d -= np.float32(amount)

    def clip_view(self, loc, q, poly_uv, margin=0.0, k=0.02, fov_v_deg=FOV_V_DEG, aspect=ASPECT,
                  near=0.1, weight=None, invert=False):
        """Visual-hull clamp: keep the field inside the screen polygon `poly_uv` [(u, v), ...] as seen
        from camera (loc, q), i.e. the mass never covers frame areas outside the polygon.
        margin (m, at the voxel's depth) grows the allowed region; k softens the cut.
        weight: optional fn(P (n, 3) world points) -> [0, 1] fading the clamp in/out by region.
        invert: keep the field *outside* the polygon instead (keeps a window of the frame clear)."""
        tv = math.tan(math.radians(fov_v_deg) / 2.0)
        th = tv * aspect
        V = np.array([((2.0 * u - 1.0) * th, (2.0 * v - 1.0) * tv) for (u, v) in poly_uv], float)
        idx = np.nonzero(self.d < max(k, 0.0) + 3.0 * self.voxel)
        if not len(idx[0]):
            return
        P = self.origin + np.stack(idx, -1) * self.voxel
        right, up, fwd = cam_basis(q)
        rel = P - np.asarray(loc, float)
        z = rel @ fwd
        zc = np.maximum(z, 1e-3)
        Q = np.stack([(rel @ right) / zc, -(rel @ up) / zc], 1)
        cone = (-1.0 if invert else 1.0) * poly_sdf(Q, V) * zc - margin
        cone = np.where(z < near, -1.0, cone)
        cur = self.d[idx]
        new = smax(cur, cone.astype(np.float32), k)
        if weight is not None:
            w = np.clip(np.asarray(weight(P), float), 0.0, 1.0).astype(np.float32)
            new = cur + w * (new - cur)
        self.d[idx] = new

    # ---- noise
    def surface_points(self, band):
        idx = np.nonzero(np.abs(self.d) < band)
        P = self.origin + np.stack(idx, -1) * self.voxel
        return idx, P

    def noise(self, noise3, amp, freq, octaves=3, offset=(0.0, 0.0, 0.0), aniso=(1.0, 1.0, 1.0),
              ridged=False, amp_fn=None):
        """Displace the surface by amp * fbm(freq * p). amp_fn(P)->(N,) scales amp per point."""
        band = abs(amp) * 1.3 + 3.0 * self.voxel
        idx, P = self.surface_points(band)
        if len(P) == 0:
            return
        Q = (P + np.asarray(offset, float)) * freq * np.asarray(aniso, float)
        n = noise3.ridged(Q, octaves) * 2.0 - 1.0 if ridged else noise3.fbm(Q, octaves)
        a = amp if amp_fn is None else amp * amp_fn(P)
        self.d[idx] -= (a * n).astype(np.float32)

    # ---- sampling
    def sample(self, P):
        """Trilinear sample of the field at world points P (N, 3)."""
        g = (np.asarray(P, float) - self.origin) / self.voxel
        shp = np.array(self.shape) - 1
        g = np.clip(g, 0, shp - 1e-6)
        i = np.floor(g).astype(np.int64)
        f = g - i
        i = np.minimum(i, shp - 1)
        d = self.d
        out = 0.0
        for dx in (0, 1):
            wx = f[:, 0] if dx else 1.0 - f[:, 0]
            for dy in (0, 1):
                wy = f[:, 1] if dy else 1.0 - f[:, 1]
                for dz in (0, 1):
                    wz = f[:, 2] if dz else 1.0 - f[:, 2]
                    out = out + wx * wy * wz * d[i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz]
        return out

    def convexity(self, P, radius, n_dirs=26):
        """Mean field value on a sphere of `radius` around surface points: + crests, - crevices (m)."""
        D = normalize(np.array([(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)
                                if (x, y, z) != (0, 0, 0)], float))[:n_dirs]
        acc = np.zeros(len(P))
        for d in D:
            acc += self.sample(np.asarray(P) + radius * d)
        return acc / len(D)

    # ---- meshing
    def mesh_arrays(self, adaptivity=0.0):
        if _vdb is None:
            raise RuntimeError("openvdb python module not available in this Blender build")
        g = _vdb.FloatGrid(background=1.0e3)
        g.copyFromArray(np.ascontiguousarray(self.d), ijk=(0, 0, 0), tolerance=0.0)
        g.transform = _vdb.createLinearTransform(voxelSize=self.voxel)
        pts, tris, quads = g.convertToPolygons(isovalue=0.0, adaptivity=float(adaptivity))
        pts = np.asarray(pts, np.float64) + self.origin
        return pts, np.asarray(tris, np.int64).reshape(-1, 3), np.asarray(quads, np.int64).reshape(-1, 4)

    def to_object(self, name, coll, material=None, target_tris=None, smooth_iter=2, adaptivity=0.0,
                  drop_cut_planes=()):
        """Mesh the field into an object. drop_cut_planes: [(point, normal)] faces lying on these
        clip planes are deleted (they are hidden inside a neighbouring mesh)."""
        pts, tris, quads = self.mesh_arrays(adaptivity)
        faces = [quads, tris]
        # orientation: make normals point outwards (signed volume > 0)
        allt = np.vstack([tris, quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]]) if len(quads) else tris
        v0, v1, v2 = pts[allt[:, 0]], pts[allt[:, 1]], pts[allt[:, 2]]
        vol = np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0
        if vol < 0:
            faces = [quads[:, ::-1], tris[:, ::-1]]
        if drop_cut_planes:
            keep_q = np.ones(len(faces[0]), bool)
            keep_t = np.ones(len(faces[1]), bool)
            for point, normal in drop_cut_planes:
                n = normalize(np.asarray(normal, float))
                dist = (pts - np.asarray(point, float)) @ n
                near = np.abs(dist) < 1.01 * self.voxel
                if len(faces[0]):
                    keep_q &= ~near[faces[0]].all(1)
                if len(faces[1]):
                    keep_t &= ~near[faces[1]].all(1)
            faces = [faces[0][keep_q], faces[1][keep_t]]
            used = np.unique(np.concatenate([faces[0].ravel(), faces[1].ravel()]))
            remap = -np.ones(len(pts), np.int64)
            remap[used] = np.arange(len(used))
            pts = pts[used]
            faces = [remap[faces[0]], remap[faces[1]]]
        me = mesh_from_numpy(name, pts, faces, smooth=True)
        obj = object_from_mesh(name, me, coll, material)
        if smooth_iter:
            co, _, ed, _ = mesh_arrays(me)
            set_vertex_positions(me, taubin_smooth(co, ed, iterations=smooth_iter))
        if target_tris:
            decimate(obj, target_tris)
        return obj


# --------------------------------------------------------------------------------------
# swept trunks
# --------------------------------------------------------------------------------------


def catmull_rom_chain(P, n_per_seg=24, alpha=0.5):
    """Centripetal Catmull-Rom through all points. Control point i is sample i*n_per_seg."""
    P = np.asarray(P, dtype=np.float64)
    ext = np.vstack([P[0] + (P[0] - P[1]), P, P[-1] + (P[-1] - P[-2])])
    out = []
    for i in range(len(P) - 1):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]

        def tj(ti, a, b):
            return ti + max(float(np.linalg.norm(b - a)), 1e-6) ** alpha

        t0 = 0.0
        t1 = tj(t0, p0, p1)
        t2 = tj(t1, p1, p2)
        t3 = tj(t2, p2, p3)
        t = np.linspace(t1, t2, n_per_seg, endpoint=False)[:, None]
        A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
        B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
        out.append((t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2)
    out.append(P[-1:])
    return np.vstack(out)


def interp_smooth(xc, yc, x):
    """1D Catmull-Rom style interpolation through (xc, yc) (xc increasing)."""
    xc = np.asarray(xc, float)
    yc = np.asarray(yc, float)
    x = np.asarray(x, float)
    n = len(xc)
    if n == 1:
        return np.full_like(x, yc[0])
    m = np.zeros(n)
    m[1:-1] = (yc[2:] - yc[:-2]) / (xc[2:] - xc[:-2])
    m[0] = (yc[1] - yc[0]) / (xc[1] - xc[0])
    m[-1] = (yc[-1] - yc[-2]) / (xc[-1] - xc[-2])
    i = np.clip(np.searchsorted(xc, x) - 1, 0, n - 2)
    h = xc[i + 1] - xc[i]
    t = np.clip((x - xc[i]) / h, 0.0, 1.0)
    t2, t3 = t * t, t * t * t
    return ((2 * t3 - 3 * t2 + 1) * yc[i] + (t3 - 2 * t2 + t) * h * m[i] +
            (-2 * t3 + 3 * t2) * yc[i + 1] + (t3 - t2) * h * m[i + 1])


def rotation_minimizing_frames(X, T, up_hint=(0.0, 0.0, 1.0)):
    """Double-reflection RMF (Wang et al. 2008). Returns normals R (N,3) and binormals B."""
    n = len(X)
    up = np.asarray(up_hint, float)
    r0 = up - (up @ T[0]) * T[0]
    if np.linalg.norm(r0) < 1e-6:
        r0 = np.array([1.0, 0.0, 0.0]) - T[0][0] * T[0]
    R = np.zeros((n, 3))
    R[0] = normalize(r0)
    for i in range(n - 1):
        v1 = X[i + 1] - X[i]
        c1 = v1 @ v1
        if c1 < 1e-14:
            R[i + 1] = R[i]
            continue
        rL = R[i] - (2.0 / c1) * (v1 @ R[i]) * v1
        tL = T[i] - (2.0 / c1) * (v1 @ T[i]) * v1
        v2 = T[i + 1] - tL
        c2 = v2 @ v2
        R[i + 1] = rL if c2 < 1e-14 else rL - (2.0 / c2) * (v2 @ rL) * v2
        R[i + 1] = normalize(R[i + 1] - (R[i + 1] @ T[i + 1]) * T[i + 1])
    B = np.cross(T, R)
    return R, B


def _wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class Sweep:
    """Result of sweep_trunk. Grid arrays are (rings N, segments M, ...)."""

    def __init__(self):
        self.obj = None
        self.N = self.M = 0
        self.centres = self.T = self.Rf = self.Bf = None
        self.s = None              # arc length per ring (m)
        self.length = 0.0
        self.radius = None         # base radius per ring (m)
        self.theta = None          # (M,) geometric angle
        self.grain = None          # (N, M) grain angle theta' = theta - twist(s)
        self.smooth = None         # (N, M, 3) surface without bark relief
        self.nsmooth = None        # (N, M, 3) normals of the smooth surface
        self.pos = None            # (N, M, 3) final surface (with bark relief)
        self.relief = None         # (N, M) bark relief height (m) along nsmooth
        self.relief_max = 0.0
        self.wraps = 1
        self.uv = None             # (N, M+1, 2) grid UVs (column M = seam duplicate)
        self.cap_vertex_count = 0


def _grid_normals(G):
    """Normals of a closed-in-theta grid surface G (N, M, 3) by central differences."""
    dth = np.roll(G, -1, axis=1) - np.roll(G, 1, axis=1)
    ds = np.zeros_like(G)
    ds[1:-1] = G[2:] - G[:-2]
    ds[0] = G[1] - G[0]
    ds[-1] = G[-1] - G[-2]
    return normalize(np.cross(dth, ds))


def sweep_trunk(name, coll, ctrl_pts, radii, *, material=None, seed=0, ring_spacing=0.04, segments=96,
                up_hint=(0.0, 0.0, 1.0), twist=0.0, twist_noise=0.0,
                profile=None, ridges=None, folds=(), knots=(), radius_noise=0.06, lump=0.03,
                caps=("flat", "flat"), uv_m_per_unit=UV_M_PER_UNIT, wraps=None, build_object=True):
    """Sweep a trunk/branch along a Catmull-Rom curve through ctrl_pts.

    radii       per control point base radius (m), smoothly interpolated.
    twist       total grain twist over the length (radians); ridges and UVs follow it.
    profile     dict(lobes=[(k, amp, phase), ...], var=0.5, lobe_twist=rad, drift=rad)
                radial multiplier 1 + sum amp_k(s) cos(k (theta - phi_k(s))).
    ridges      dict(count=int, depth=m, groove=0.18, meander=0.35, layers=2, var=0.5)
                longitudinal bark plates & grooves following the twisted grain.
    folds       [(theta_c, width_rad, depth_rel, s0, s1)] creases (s0/s1 in 0..1 of length).
    knots       [(s_rel, theta_c, radius_m, height_rel, swirl)] bumps; grain swirls around them.
    caps        per end: 'open' | 'flat' | 'round' | 'broken' | dict(kind='splinter', ...) (see
                _splinter_cap: tall slivers + sunken fibrous fracture; per-vertex info in sw.cap_meta).
    UV          U = wraps * theta'/(2 pi) (integer wraps, follows the grain twist), V = s / 0.5 m.
    """
    rng = np.random.default_rng(seed)
    noise = Noise3(seed + 101)
    ctrl = np.asarray(ctrl_pts, float)
    nps = 32
    dense = catmull_rom_chain(ctrl, n_per_seg=nps)
    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    Ld = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(Ld[-1])
    N = max(3, int(round(total / ring_spacing)) + 1)
    s = np.linspace(0.0, total, N)
    X = np.stack([np.interp(s, Ld, dense[:, k]) for k in range(3)], 1)
    T = np.zeros_like(X)
    T[1:-1] = X[2:] - X[:-2]
    T[0] = X[1] - X[0]
    T[-1] = X[-1] - X[-2]
    T = normalize(T)
    Rf, Bf = rotation_minimizing_frames(X, T, up_hint)

    ctrl_s = Ld[np.arange(len(ctrl)) * nps]
    R = interp_smooth(ctrl_s, np.asarray(radii, float), s)
    if radius_noise:
        R = R * (1.0 + radius_noise * noise.line(s, freq=0.9, offset=3.1))
    M = int(segments)
    theta = np.arange(M) * (2.0 * math.pi / M)
    sn = s / max(total, 1e-6)

    tw = twist * sn
    if twist_noise:
        tw = tw + twist_noise * noise.line(s, freq=0.5, offset=7.7)
    grain = theta[None, :] - tw[:, None]           # (N, M)

    # ---------------- cross-section profile
    prof = dict(lobes=[(2, 0.10, 0.0), (3, 0.04, 1.0)], var=0.6, lobe_twist=0.0, drift=0.6)
    if profile:
        prof.update(profile)
    P = np.ones((N, M))
    for li, (k, amp, ph) in enumerate(prof["lobes"]):
        a = amp * (1.0 + prof["var"] * noise.line(s, freq=0.7, offset=11.0 + li * 5.3))
        phi = ph + prof["lobe_twist"] * sn + prof["drift"] * noise.line(s, freq=0.4, offset=21.0 + li * 3.1)
        P += a[:, None] * np.cos(k * (theta[None, :] - phi[:, None]))
    for (thc, w, depth, s0, s1) in folds:
        env = smoothstep(s0 - 0.08, s0 + 0.05, sn) * (1.0 - smoothstep(s1 - 0.05, s1 + 0.08, sn))
        d = _wrap_angle(grain - thc)
        P -= depth * env[:, None] * np.exp(-(d / w) ** 2)
    # low-frequency lumpiness, periodic in theta
    if lump:
        cth, sth = np.cos(theta), np.sin(theta)
        Q = np.stack([np.broadcast_to(cth * 1.3, (N, M)), np.broadcast_to(sth * 1.3, (N, M)),
                      np.broadcast_to((s * 1.6)[:, None], (N, M))], -1)
        P += lump * noise.fbm(Q, 3)
    rad = R[:, None] * P

    # knots: radial bumps + grain swirl
    swirl = np.zeros((N, M))
    for (ks, kth, kr, kh, ksw) in knots:
        ds_m = s[:, None] - ks * total
        dth = _wrap_angle(theta[None, :] - tw[:, None] - kth)
        dist2 = ds_m ** 2 + (R[:, None] * dth) ** 2
        g = np.exp(-dist2 / (kr * kr))
        rad = rad + kh * R[:, None] * g
        swirl += ksw * np.sign(dth) * np.exp(-dist2 / (2.0 * kr * kr)) * (1.0 - g)

    dirs = (np.cos(theta)[None, :, None] * Rf[:, None, :] + np.sin(theta)[None, :, None] * Bf[:, None, :])
    smooth = X[:, None, :] + rad[..., None] * dirs
    nsmooth = _grid_normals(smooth)

    # ---------------- bark relief (ridges along the grain)
    relief = np.zeros((N, M))
    if ridges:
        rp = dict(count=18, depth=0.012, groove=0.2, meander=0.35, layers=2, var=0.5, scale_with_radius=True)
        rp.update(ridges)
        cg, sg = np.cos(grain), np.sin(grain)
        Q = np.stack([cg * 0.9, sg * 0.9, np.broadcast_to((s * 1.2)[:, None], (N, M))], -1)
        mea = rp["meander"] * noise.fbm(Q, 3)
        h = np.zeros((N, M))
        wsum = 0.0
        for layer in range(int(rp["layers"])):
            cnt = int(rp["count"]) * (1 + layer) + layer * 3
            off = rng.uniform(0, 1)
            x = (grain + swirl) * cnt / (2.0 * math.pi) + mea * (1.0 + 0.6 * layer) + off
            f = x - np.floor(x)
            gw = rp["groove"]
            plate = smoothstep(0.0, gw, f) * smoothstep(0.0, gw, 1.0 - f)
            wgt = 1.0 / (1.0 + layer * 1.6)
            h += wgt * plate
            wsum += wgt
        h /= wsum
        Qv = np.stack([cg * 2.0, sg * 2.0, np.broadcast_to((s * 3.0)[:, None], (N, M))], -1)
        var = 1.0 + rp["var"] * noise.fbm(Qv + 50.0, 2)
        depth = rp["depth"] * (R[:, None] / max(float(np.mean(R)), 1e-6) if rp["scale_with_radius"] else 1.0)
        relief = depth * (h - 0.5) * np.clip(var, 0.2, 2.0)
    pos = smooth + nsmooth * relief[..., None]

    # ---------------- UVs
    if wraps is None:
        wraps = max(1, int(round(2.0 * math.pi * float(np.mean(R)) / uv_m_per_unit)))
    th_ext = np.append(theta, 2.0 * math.pi)
    U = wraps * (th_ext[None, :] - tw[:, None]) / (2.0 * math.pi)
    V = np.broadcast_to((s / uv_m_per_unit)[:, None], (N, M + 1))
    uv = np.stack([U, V], -1)

    sw = Sweep()
    sw.N, sw.M = N, M
    sw.centres, sw.T, sw.Rf, sw.Bf = X, T, Rf, Bf
    sw.s, sw.length, sw.radius = s, total, R
    sw.theta, sw.grain = theta, grain
    sw.smooth, sw.nsmooth, sw.pos = smooth, nsmooth, pos
    sw.relief, sw.relief_max = relief, float(relief.max()) if relief.size else 0.0
    sw.wraps, sw.uv = wraps, uv
    sw.uv_m_per_unit = uv_m_per_unit
    sw.twist_s = tw
    if build_object:
        sw.obj = _sweep_object(name, coll, sw, caps, material, noise)
    return sw


def _sweep_object(name, coll, sw, caps, material, noise):
    N, M = sw.N, sw.M
    verts = [sw.pos.reshape(-1, 3)]
    vid = np.arange(N * M).reshape(N, M)
    i0, j0 = np.meshgrid(np.arange(N - 1), np.arange(M), indexing="ij")
    j1 = (j0 + 1) % M
    quads = np.stack([vid[i0, j0], vid[i0, j1], vid[i0 + 1, j1], vid[i0 + 1, j0]], -1).reshape(-1, 4)
    # per-loop uvs for the quads
    uvq = np.stack([sw.uv[i0, j0], sw.uv[i0, j0 + 1], sw.uv[i0 + 1, j0 + 1], sw.uv[i0 + 1, j0]], 2)
    uv_loops = [uvq.reshape(-1, 2)]
    faces = [quads]
    nxt = N * M
    sw.cap_meta = {}
    for end, kind in (("start", caps[0]), ("end", caps[1])):
        if isinstance(kind, dict):          # parametric cap, e.g. dict(kind="splinter", ...)
            spec = dict(kind)
            kind = spec.pop("kind", "splinter")
        else:
            spec = {}
        if kind == "open":
            continue
        ring = 0 if end == "start" else N - 1
        rv = vid[ring]
        if kind == "splinter":
            cv, cf, cuv, meta = _splinter_cap(sw, end, ring, rv, nxt, spec, noise)
            verts.append(cv)
            faces.extend(cf)
            uv_loops.extend(cuv)
            meta["first"] = nxt
            sw.cap_meta[end] = meta
            nxt += len(cv)
            continue
        c = sw.centres[ring]
        t = sw.T[ring] * (-1.0 if end == "start" else 1.0)
        r = float(sw.radius[ring])
        if kind == "round":
            k = 4
            prev = rv
            for q in range(1, k + 1):
                a = q / (k + 1) * (math.pi / 2)
                ring_pts = c + (sw.pos[ring] - c) * math.cos(a) + t * r * math.sin(a)
                verts.append(ring_pts)
                ids = nxt + np.arange(M)
                nxt += M
                jj = np.arange(M)
                if end == "end":
                    faces.append(np.stack([prev[jj], prev[(jj + 1) % M], ids[(jj + 1) % M], ids[jj]], -1))
                else:
                    faces.append(np.stack([prev[jj], ids[jj], ids[(jj + 1) % M], prev[(jj + 1) % M]], -1))
                uv_loops.append(np.zeros((M * 4, 2)))
                prev = ids
            verts.append((c + t * r)[None, :])
            tip = nxt
            nxt += 1
            jj = np.arange(M)
            tri = (np.stack([prev[jj], prev[(jj + 1) % M], np.full(M, tip)], -1) if end == "end"
                   else np.stack([prev[(jj + 1) % M], prev[jj], np.full(M, tip)], -1))
            faces.append(tri)
            uv_loops.append(np.zeros((M * 3, 2)))
            continue
        centre = c.copy()
        if kind == "broken":
            # jagged splintered rim + sunken rough centre
            jag = 0.25 * r * (0.5 + 0.5 * noise.fbm(np.stack([np.cos(sw.theta) * 3, np.sin(sw.theta) * 3,
                                                               np.full(M, 4.2)], -1), 3))
            spikes = (np.sin(sw.theta * 7 + 1.3) > 0.55) * 0.2 * r
            pts = sw.pos[ring] + t[None, :] * (jag + spikes)[:, None]
            verts[0][rv] = pts
            inner = c + (sw.pos[ring] - c) * 0.72 + t * 0.05 * r
            verts.append(inner)
            ids = nxt + np.arange(M)
            nxt += M
            jj = np.arange(M)
            if end == "end":
                faces.append(np.stack([rv[jj], rv[(jj + 1) % M], ids[(jj + 1) % M], ids[jj]], -1))
            else:
                faces.append(np.stack([rv[jj], ids[jj], ids[(jj + 1) % M], rv[(jj + 1) % M]], -1))
            uv_loops.append(np.zeros((M * 4, 2)))
            rv = ids
            centre = c - t * 0.12 * r
        verts.append(centre[None, :])
        cid = nxt
        nxt += 1
        jj = np.arange(M)
        tri = (np.stack([rv[jj], rv[(jj + 1) % M], np.full(M, cid)], -1) if end == "end"
               else np.stack([rv[(jj + 1) % M], rv[jj], np.full(M, cid)], -1))
        faces.append(tri)
        uv_loops.append(np.zeros((M * 3, 2)))
    V = np.vstack(verts)
    sw.cap_vertex_count = len(V) - N * M
    me = mesh_from_numpy(name, V, faces, smooth=True)
    # UVs (loop order == face order == concatenated faces)
    set_uvs(me, np.vstack(uv_loops))
    return object_from_mesh(name, me, coll, material)


def _splinter_cap(sw, end, ring, rim_ids, first, spec, noise):
    """Splintered fracture cap for one sweep end (caps=(..., dict(kind="splinter", ...))).

    The tube's last ring is the rim. Outer rings continue the bark surface straight out to a
    rim height H(theta): low and jagged everywhere, with tall slivers where `splinters` stand;
    inner rings form a sunken, fibrous fracture face down to a centre vertex.

    spec keys (lengths relative to the ring radius r unless noted):
      jag=0.14          base rim jaggedness            sink=0.16    centre depression
      fibre=0.05        fibrous noise on the fracture  seed=0.0     noise offset
      outer_rings=4     subdivisions of the splinter outer faces
      rho=(...)         radii (0..1) of the inner rings, decreasing
      splinters=[dict(dir=(x, y, z) world direction | theta=rad, height=1.2, width=0.35 rad,
                      thick=0.25 (sliver thickness, 0..1 of r), lean=0.0 (outward per unit height),
                      tip=0.8 (peak sharpness), rag=0.3 (ragged top),
                      shape='peak' | 'slab' (flat top, sides fall off from shoulder*width to width))]
    Returns (verts (K, 3), [faces...], [uv loops...], meta) where meta holds per-vertex rho, h (m),
    `outer` (bark-like splinter faces) and `fracture` (torn inner face) masks for colouring.
    """
    M = sw.M
    sign = 1.0 if end == "end" else -1.0
    c = sw.centres[ring]
    t = sw.T[ring] * sign
    r = float(sw.radius[ring])
    rim = sw.pos[ring]
    radial = rim - c
    radial = radial - (radial @ t)[:, None] * t
    rlen = np.linalg.norm(radial, axis=1)
    rdir = radial / np.maximum(rlen, 1e-9)[:, None]
    th = sw.theta
    so = float(spec.get("seed", 0.0))
    jag = float(spec.get("jag", 0.14))
    sink = float(spec.get("sink", 0.16))
    fibre = float(spec.get("fibre", 0.05))
    n_out = int(spec.get("outer_rings", 4))
    rhos = [float(x) for x in spec.get("rho", (0.93, 0.84, 0.72, 0.58, 0.44, 0.3, 0.16))]
    P = np.stack([np.cos(th) * 2.3, np.sin(th) * 2.3, np.full(M, 3.7 + so)], -1)
    H = r * jag * (0.45 + 0.55 * np.clip(0.5 + noise.fbm(P, 3), 0.0, 1.0))
    H += r * 0.6 * jag * np.clip(noise.fbm(P * 3.3 + 11.0 + so, 2), 0.0, 1.0)      # small spikes
    H_base = H.copy()
    thick = np.zeros(M)
    lean = np.zeros(M)
    splint = np.zeros(M)
    for si, sp in enumerate(spec.get("splinters", ())):
        if "dir" in sp:
            d = np.asarray(sp["dir"], float)
            d = d - (d @ t) * t
            th_c = th[int(np.argmax(rdir @ normalize(d)))]
        else:
            th_c = float(sp["theta"])
        w = float(sp.get("width", 0.35))
        dth = np.abs(_wrap_angle(th - th_c))
        if sp.get("shape", "peak") == "slab":           # flat-topped sliver with steep sides
            prof = 1.0 - smoothstep(float(sp.get("shoulder", 0.55)) * w, w, dth)
        else:                                            # tapering peak
            prof = np.clip(1.0 - dth / w, 0.0, 1.0) ** float(sp.get("tip", 0.8))
        rag = 1.0 + float(sp.get("rag", 0.3)) * noise.fbm(P * 4.1 + 5.0 + 7.3 * si + so, 2)
        h = r * float(sp.get("height", 1.0)) * prof * np.clip(rag, 0.4, 1.6)
        on = h > H
        H = np.where(on, h, H)
        thick = np.where(on, float(sp.get("thick", 0.25)), thick)
        lean = np.where(on, float(sp.get("lean", 0.0)), lean)
        splint = np.maximum(splint, prof)

    def place(rho, h):
        return c + rdir * (rlen * rho)[:, None] + t * h[:, None] + rdir * (lean * h)[:, None]

    rings_p, rings_rho, rings_h, rings_outer = [], [], [], []
    for i in range(1, n_out + 1):                       # splinter outer faces (bark continues)
        f = i / n_out
        rho = np.full(M, 1.0 - 0.004 * i)
        h = H * f
        rings_p.append(place(rho, h))
        rings_rho.append(rho)
        rings_h.append(h)
        rings_outer.append(np.ones(M, bool))
    for k, rho_k in enumerate(rhos):                    # fracture face
        rho = np.full(M, rho_k)
        base = H_base * (0.35 + 0.65 * rho_k) - sink * r * (1.0 - rho_k * rho_k)
        top = smoothstep(1.0 - thick - 0.1, 1.0 - thick + 0.02, rho) * (thick > 0)
        h = base + (H - base) * top
        Q = np.stack([np.cos(th) * rho_k * 6.0, np.sin(th) * rho_k * 6.0, np.full(M, 1.3 + so + 0.37 * k)], -1)
        h = h + fibre * r * (0.35 + 0.65 * rho_k) * noise.fbm(Q, 2)
        rings_p.append(place(rho, h))
        rings_rho.append(rho)
        rings_h.append(h)
        rings_outer.append(np.zeros(M, bool))
    ch = -sink * r + fibre * r * 0.3 * float(noise(np.array([[0.3 + so, 1.7, 2.9]]))[0])
    centre = c + t * ch

    K = len(rings_p)
    cv = np.vstack(rings_p + [centre[None, :]])
    ids = [rim_ids] + [first + k * M + np.arange(M) for k in range(K)]
    cid = first + K * M
    jj = np.arange(M)
    faces, uvs = [], []
    U = sw.uv[ring, :, 0]                                # (M+1,) incl. seam column
    V0 = float(sw.uv[ring, 0, 1])
    um = float(getattr(sw, "uv_m_per_unit", UV_M_PER_UNIT))
    hv = [np.zeros(M)] + rings_h

    def vv(k, j):
        return V0 + sign * hv[k][j % M] / um

    for k in range(K):
        a, b = ids[k], ids[k + 1]
        if end == "end":
            faces.append(np.stack([a[jj], a[(jj + 1) % M], b[(jj + 1) % M], b[jj]], -1))
            uvs.append(np.stack([np.stack([U[jj], vv(k, jj)], -1), np.stack([U[jj + 1], vv(k, jj + 1)], -1),
                                 np.stack([U[jj + 1], vv(k + 1, jj + 1)], -1), np.stack([U[jj], vv(k + 1, jj)], -1)],
                                1).reshape(-1, 2))
        else:
            faces.append(np.stack([a[jj], b[jj], b[(jj + 1) % M], a[(jj + 1) % M]], -1))
            uvs.append(np.stack([np.stack([U[jj], vv(k, jj)], -1), np.stack([U[jj], vv(k + 1, jj)], -1),
                                 np.stack([U[jj + 1], vv(k + 1, jj + 1)], -1), np.stack([U[jj + 1], vv(k, jj + 1)], -1)],
                                1).reshape(-1, 2))
    last = ids[K]
    vc = V0 + sign * ch / um
    if end == "end":
        faces.append(np.stack([last[jj], last[(jj + 1) % M], np.full(M, cid)], -1))
        uvs.append(np.stack([np.stack([U[jj], vv(K, jj)], -1), np.stack([U[jj + 1], vv(K, jj + 1)], -1),
                             np.stack([0.5 * (U[jj] + U[jj + 1]), np.full(M, vc)], -1)], 1).reshape(-1, 2))
    else:
        faces.append(np.stack([last[(jj + 1) % M], last[jj], np.full(M, cid)], -1))
        uvs.append(np.stack([np.stack([U[jj + 1], vv(K, jj + 1)], -1), np.stack([U[jj], vv(K, jj)], -1),
                             np.stack([0.5 * (U[jj] + U[jj + 1]), np.full(M, vc)], -1)], 1).reshape(-1, 2))
    meta = dict(rho=np.concatenate(rings_rho + [np.zeros(1)]), h=np.concatenate(rings_h + [np.array([ch])]),
                outer=np.concatenate(rings_outer + [np.zeros(1, bool)]),
                splinter=np.concatenate([splint] * K + [np.zeros(1)]), radius=r, rim_height=H, count=len(cv))
    meta["fracture"] = ~meta["outer"]
    return cv, faces, uvs, meta


def join_objects(name, coll, objs, material=None, smooth=True):
    """Merge mesh objects (world space) into one new object `name`; keeps the 'UVMap' layer
    (zeros where a part has none). Vertex colours are not carried (write COLOR_0 afterwards).
    Returns (obj, parts) with parts = [(source name, first vertex, vertex count), ...]."""
    Vs, F, UV, parts = [], [], [], []
    off = 0
    for o in objs:
        me = o.data
        nv, nl, nf = len(me.vertices), len(me.loops), len(me.polygons)
        co = np.empty(nv * 3, np.float32)
        me.vertices.foreach_get("co", co)
        Mw = np.array(o.matrix_world)
        co = co.reshape(-1, 3).astype(np.float64) @ Mw[:3, :3].T + Mw[:3, 3]
        lv = np.empty(nl, np.int32)
        me.loops.foreach_get("vertex_index", lv)
        lt = np.empty(nf, np.int32)
        me.polygons.foreach_get("loop_total", lt)
        ls = np.empty(nf, np.int32)
        me.polygons.foreach_get("loop_start", ls)
        uvl = me.uv_layers.get(UV_NAME) or (me.uv_layers[0] if len(me.uv_layers) else None)
        uv = np.zeros(nl * 2, np.float32)
        if uvl is not None:
            uvl.data.foreach_get("uv", uv)
        uv = uv.reshape(-1, 2)
        for k in np.unique(lt):
            sel = np.nonzero(lt == k)[0]
            idx = ls[sel][:, None] + np.arange(k)[None, :]
            F.append(lv[idx] + off)
            UV.append(uv[idx].reshape(-1, 2))
        Vs.append(co)
        parts.append((o.name, off, nv))
        off += nv
    # loop order of mesh_from_numpy == order of the face arrays
    me = mesh_from_numpy(name, np.vstack(Vs), F, smooth=smooth)
    set_uvs(me, np.vstack(UV))
    for o in objs:
        remove_object(o.name)
    return object_from_mesh(name, me, coll, material), parts


def set_uvs(me, uv_loops, name=UV_NAME):
    for layer in list(me.uv_layers):
        me.uv_layers.remove(layer)
    layer = me.uv_layers.new(name=name)
    layer.data.foreach_set("uv", np.asarray(uv_loops, np.float32).ravel())
    return layer


# --------------------------------------------------------------------------------------
# moss cushions on a swept trunk
# --------------------------------------------------------------------------------------


def blur_grid(a, iterations=3):
    """[1 2 1] blur of a trunk-grid array (N, M): clamped along the rings axis, wrapped around."""
    a = np.asarray(a, float).copy()
    for _ in range(int(iterations)):
        a = 0.25 * (np.roll(a, 1, axis=1) + 2.0 * a + np.roll(a, -1, axis=1))
        up = np.vstack([a[:1], a[:-1]])
        dn = np.vstack([a[1:], a[-1:]])
        a = 0.25 * (up + 2.0 * a + dn)
    return a


def moss_cushions(name, coll, sw, mask, *, thickness=0.04, material=None, seed=0, lump_amp=0.45,
                  lump_freq=5.0, border=(0.32, 0.72), tuck=0.015, keep_above=0.22, step=1, sag=0.0):
    """Lumpy moss cushions hugging the trunk where mask (N, M) is high.

    thickness: m (scalar or (N, M)); ~2-6 % of the trunk diameter.
    Towards the mask border the sheet settles onto the bark relief and ends `tuck` m below it,
    so the visible edge is a soft, irregular contour of the mask (not grid-aligned).
    Returns (obj, info) where info has per-vertex 'offset', 'shape', 'rows', 'cols' for colours.
    """
    noise = Noise3(seed + 303)
    rows = np.arange(0, sw.N, step)
    if rows[-1] != sw.N - 1:
        rows = np.append(rows, sw.N - 1)
    cols = np.arange(0, sw.M, step)
    m = np.asarray(mask, float)[np.ix_(rows, cols)]
    base = sw.smooth[np.ix_(rows, cols)]
    nrm = sw.nsmooth[np.ix_(rows, cols)]
    T = np.broadcast_to(np.asarray(thickness, float), (sw.N, sw.M))[np.ix_(rows, cols)]
    sh = smoothstep(border[0], border[1], m)
    lum = 1.0 + lump_amp * noise.fbm(base * lump_freq, 3)
    clear = max(sw.relief_max, 0.0)
    # inside: clear of the highest bark ridge by T*lum; towards the border the sheet blends down onto
    # the bark relief itself and ends `tuck` m below it, so the visible edge is a soft contour of
    # the mask (no comb pattern where a flat sheet would cut through the ridges)
    rel = sw.relief[np.ix_(rows, cols)] if sw.relief is not None else np.zeros_like(m)
    off = (rel - tuck) * (1.0 - sh) + (clear + T * lum) * sh
    pos = base + nrm * off[..., None]
    if sag:
        pos[..., 2] -= sag * T * sh * np.clip(-nrm[..., 2], 0, 1)
    n, mm = len(rows), len(cols)
    vid = np.arange(n * mm).reshape(n, mm)
    i0, j0 = np.meshgrid(np.arange(n - 1), np.arange(mm), indexing="ij")
    j1 = (j0 + 1) % mm
    quads = np.stack([vid[i0, j0], vid[i0, j1], vid[i0 + 1, j1], vid[i0 + 1, j0]], -1).reshape(-1, 4)
    flat_m = m.reshape(-1)
    keep = (flat_m[quads] > keep_above).all(1)
    quads = quads[keep]
    if not len(quads):
        return None, None
    # uvs from the trunk grid (column index mm of uvg is the seam duplicate at theta = 2 pi)
    uvg = sw.uv[np.ix_(rows, np.append(cols, sw.M))]
    a_i, a_j = np.divmod(quads[:, 0], mm)
    c_i = np.divmod(quads[:, 2], mm)[0]
    j_next = a_j + 1
    uv_l = np.stack([uvg[a_i, a_j], uvg[a_i, j_next], uvg[c_i, j_next], uvg[c_i, a_j]], 1).reshape(-1, 2)
    used = np.unique(quads)
    remap = -np.ones(n * mm, np.int64)
    remap[used] = np.arange(len(used))
    V = pos.reshape(-1, 3)[used]
    me = mesh_from_numpy(name, V, remap[quads], smooth=True)
    set_uvs(me, uv_l)
    obj = object_from_mesh(name, me, coll, material)
    info = dict(offset=off.reshape(-1)[used], shape=sh.reshape(-1)[used], thickness=T.reshape(-1)[used],
                relief=rel.reshape(-1)[used], clear=clear, rows=rows, cols=cols, used=used)
    return obj, info


# --------------------------------------------------------------------------------------
# vertex colour helpers (COLOR_0)
# --------------------------------------------------------------------------------------


def surface_convexity(co, nrm, edges, iterations=8):
    """Signed height of each vertex above its smoothed neighbourhood (m).

    + on crests / bumps, - in crevices. `iterations` sets the scale (more = larger features).
    """
    n = len(co)
    sm = np.array(co, dtype=np.float64)
    for _ in range(int(iterations)):
        sm = 0.5 * sm + 0.5 * neighbor_mean(sm, edges, n)
    return np.einsum("ij,ij->i", co - sm, nrm)


def build_bvh(objects):
    """BVH over the evaluated world-space triangles of `objects` (for AO)."""
    Vs, Fs = [], []
    off = 0
    for obj in objects:
        if obj.type != "MESH":
            continue
        co, _, _, tri = mesh_arrays(obj.data)
        M = np.array(obj.matrix_world)
        co = co @ M[:3, :3].T + M[:3, 3]
        Vs.append(co)
        Fs.append(tri + off)
        off += len(co)
    V = np.vstack(Vs)
    F = np.vstack(Fs)
    return BVHTree.FromPolygons(V.tolist(), F.tolist(), all_triangles=True)


def hemisphere_dirs(n):
    """Cosine-weighted hemisphere directions (z up), Fibonacci spiral (deterministic)."""
    i = np.arange(n) + 0.5
    r = np.sqrt(i / n)
    phi = i * math.pi * (3.0 - math.sqrt(5.0))
    return np.stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(np.maximum(0.0, 1.0 - r * r))], 1)


def ray_ao(co, nrm, bvh, n_rays=12, max_dist=0.5, bias=0.004, chunk=4000):
    """Ambient occlusion per vertex (1 = open) by casting cosine-weighted rays into `bvh`."""
    D = hemisphere_dirs(n_rays)
    co = np.asarray(co, float)
    nrm = normalize(np.asarray(nrm, float))
    a = np.where(np.abs(nrm[:, 2:3]) < 0.9, np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]))
    t = normalize(np.cross(a, nrm))
    b = np.cross(nrm, t)
    out = np.ones(len(co))
    cast = bvh.ray_cast
    for c0 in range(0, len(co), chunk):
        c1 = min(len(co), c0 + chunk)
        dirs = (D[None, :, 0:1] * t[c0:c1, None, :] + D[None, :, 1:2] * b[c0:c1, None, :] +
                D[None, :, 2:3] * nrm[c0:c1, None, :])
        org = (co[c0:c1] + nrm[c0:c1] * bias).tolist()
        dl = dirs.tolist()
        res = []
        for i in range(c1 - c0):
            o = org[i]
            occ = 0.0
            for d in dl[i]:
                hit = cast(o, d, max_dist)
                if hit[0] is not None:
                    occ += 1.0 - 0.6 * (hit[3] / max_dist)
            res.append(occ)
        out[c0:c1] = 1.0 - np.asarray(res) / n_rays
    return np.clip(out, 0.0, 1.0)


def rim_factor(co, nrm, cam_positions):
    """1 where the surface is seen edge-on from any of the cameras (silhouette rims)."""
    best = np.zeros(len(co))
    for C in cam_positions:
        v = normalize(np.asarray(co) - np.asarray(C, float))
        facing = -np.einsum("ij,ij->i", nrm, v)
        rim = np.clip(1.0 - np.abs(facing), 0.0, 1.0) * (facing > -0.35)
        best = np.maximum(best, rim)
    return best


def write_color0(me, r, g, b, a=None):
    """Write COLOR_0 (linear float RGBA, point domain) and make it the render/active colour."""
    n = len(me.vertices)
    rgba = np.ones((n, 4), np.float32)
    rgba[:, 0] = np.clip(r, 0, 1)
    rgba[:, 1] = np.clip(g, 0, 1)
    rgba[:, 2] = np.clip(b, 0, 1)
    if a is not None:
        rgba[:, 3] = np.clip(a, 0, 1)
    for attr in [x for x in me.color_attributes]:
        me.color_attributes.remove(attr)
    attr = me.color_attributes.new(COLOR_ATTR, "FLOAT_COLOR", "POINT")
    attr.data.foreach_set("color", rgba.ravel())
    idx = me.color_attributes.find(COLOR_ATTR)
    me.color_attributes.active_color_index = idx
    me.color_attributes.render_color_index = idx
    return attr


def read_color0(me):
    attr = me.color_attributes.get(COLOR_ATTR)
    if attr is None:
        return None
    buf = np.empty(len(attr.data) * 4, np.float32)
    attr.data.foreach_get("color", buf)
    return buf.reshape(-1, 4)


# --------------------------------------------------------------------------------------
# UVs
# --------------------------------------------------------------------------------------


def box_uvs(me, metres_per_unit=UV_M_PER_UNIT):
    """Per-face box projection (dominant normal axis), 1 UV unit = metres_per_unit."""
    nv, nl, nf = len(me.vertices), len(me.loops), len(me.polygons)
    co = np.empty(nv * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3) / metres_per_unit
    lv = np.empty(nl, np.int32)
    me.loops.foreach_get("vertex_index", lv)
    pn = np.empty(nf * 3, np.float32)
    me.polygon_normals.foreach_get("vector", pn)
    pn = pn.reshape(-1, 3)
    lt = np.empty(nf, np.int32)
    me.polygons.foreach_get("loop_total", lt)
    pol = np.repeat(np.arange(nf), lt)
    n = pn[pol]
    ax = np.argmax(np.abs(n), 1)
    sg = np.sign(n[np.arange(nl), ax])
    p = co[lv]
    u = np.where(ax == 0, p[:, 1] * sg, np.where(ax == 1, -p[:, 0] * sg, p[:, 0] * sg))
    v = np.where(ax == 2, p[:, 1], p[:, 2])
    set_uvs(me, np.stack([u, v], 1))


# --------------------------------------------------------------------------------------
# materials
# --------------------------------------------------------------------------------------


def preview_material(name, hex_color, roughness=0.9):
    """Named placeholder material (engine replaces it by node-name prefix)."""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    lin = hex_to_linear(hex_color)
    mat.diffuse_color = (lin[0], lin[1], lin[2], 1.0)
    mat.roughness = roughness
    mat.metallic = 0.0
    try:
        if mat.node_tree is None:
            mat.use_nodes = True
    except Exception:
        pass
    nt = mat.node_tree
    if nt is not None:
        bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = (lin[0], lin[1], lin[2], 1.0)
            bsdf.inputs["Roughness"].default_value = roughness
            bsdf.inputs["Metallic"].default_value = 0.0
    return mat


# --------------------------------------------------------------------------------------
# cameras, screen anchoring, key-light empties
# --------------------------------------------------------------------------------------


def cam_quat(pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0):
    """Camera orientation: zero angles look along +Y with +Z up.
    pitch > 0 looks up, yaw > 0 turns left (towards -X), roll rotates about the view axis."""
    q = Euler((math.radians(90.0 + pitch_deg), 0.0, math.radians(yaw_deg)), "XYZ").to_quaternion()
    if roll_deg:
        q = q @ Quaternion((0.0, 0.0, 1.0), math.radians(roll_deg))
    return q


def cam_basis(q):
    Rm = q.to_matrix()
    return (np.array(Rm @ Vector((1, 0, 0))), np.array(Rm @ Vector((0, 1, 0))),
            np.array(Rm @ Vector((0, 0, -1))))


def screen_to_world(loc, q, u, v, depth, fov_v_deg=FOV_V_DEG, aspect=ASPECT):
    """World point at frame position (u, v) (0..1, origin top-left) and `depth` metres along the view axis."""
    right, up, fwd = cam_basis(q)
    tv = math.tan(math.radians(fov_v_deg) / 2.0)
    th = tv * aspect
    return (np.asarray(loc, float) + fwd * depth + right * ((2.0 * u - 1.0) * th * depth) +
            up * ((1.0 - 2.0 * v) * tv * depth))


def project_to_screen(loc, q, P, fov_v_deg=FOV_V_DEG, aspect=ASPECT):
    """World points (N,3) -> (u, v, depth) arrays."""
    right, up, fwd = cam_basis(q)
    d = np.asarray(P, float) - np.asarray(loc, float)
    z = d @ fwd
    tv = math.tan(math.radians(fov_v_deg) / 2.0)
    th = tv * aspect
    u = 0.5 + 0.5 * (d @ right) / (np.maximum(z, 1e-6) * th)
    v = 0.5 - 0.5 * (d @ up) / (np.maximum(z, 1e-6) * tv)
    return u, v, z


def make_camera(name, coll, loc, quat, fov_v_deg=FOV_V_DEG, clip=(0.05, 200.0), focus=None, extras=None):
    remove_object(name)
    _free_name(bpy.data.cameras, name)
    cam = bpy.data.cameras.new(name)
    cam.type = "PERSP"
    cam.sensor_fit = "VERTICAL"
    cam.sensor_height = 24.0
    cam.sensor_width = 24.0 * ASPECT
    cam.angle = math.radians(fov_v_deg)
    cam.clip_start, cam.clip_end = clip
    if focus:
        cam.dof.focus_distance = float(focus)
        cam.dof.use_dof = False
    obj = bpy.data.objects.new(name, cam)
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = quat
    obj.location = Vector(loc)
    coll.objects.link(obj)
    for k, v in (extras or {}).items():
        obj[k] = v
    return obj


def blender_to_gltf_dir(d):
    """Direction in Blender (Z-up) world -> glTF (Y-up) world."""
    return (float(d[0]), float(d[2]), float(-d[1]))


def make_key_empty(name, coll, loc, light_dir, size=0.35, extras=None):
    """Key-light hint. After the +Y-up glTF export the node's local -Z equals `light_dir`.

    Blender empties get no axis correction on export, so Blender local +Y becomes glTF local -Z:
    the empty is oriented with its local +Y along light_dir (direction the light travels).
    """
    remove_object(name)
    d = Vector(light_dir).normalized()
    q = d.to_track_quat("Y", "Z")
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = "ARROWS"
    obj.empty_display_size = size
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = q
    obj.location = Vector(loc)
    coll.objects.link(obj)
    obj["light_dir_gltf"] = list(blender_to_gltf_dir(d))
    for k, v in (extras or {}).items():
        obj[k] = v
    return obj


# --------------------------------------------------------------------------------------
# preview renders (Blender only; the engine has its own look)
# --------------------------------------------------------------------------------------


def setup_preview_render(scene, engine="BLENDER_WORKBENCH", percent=50, bg_hex=PALETTE["sage"],
                         color_type="MATERIAL"):
    scene.render.engine = engine
    scene.render.resolution_x, scene.render.resolution_y = FRAME_W, FRAME_H
    scene.render.resolution_percentage = int(percent)
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    vs = scene.view_settings
    try:
        vs.view_transform = "Standard"
        vs.look = "None"
    except Exception:
        pass
    vs.exposure = 0.0
    vs.gamma = 1.0
    world = scene.world
    if world is None:
        world = bpy.data.worlds.get(scene.name + "_preview_world") or bpy.data.worlds.new(scene.name + "_preview_world")
        scene.world = world
    lin = hex_to_linear(bg_hex)
    world.color = lin
    try:
        if world.node_tree is None:
            world.use_nodes = True
        bg = next((n for n in world.node_tree.nodes if n.type == "BACKGROUND"), None)
        if bg is not None:
            bg.inputs["Color"].default_value = (lin[0], lin[1], lin[2], 1.0)
            bg.inputs["Strength"].default_value = 1.0
    except Exception:
        pass
    sh = scene.display.shading
    sh.light = "STUDIO"
    sh.color_type = color_type
    sh.show_cavity = True
    sh.cavity_type = "WORLD"
    sh.show_shadows = False
    sh.show_specular_highlight = False
    try:
        sh.studio_light = "outdoor.sl" if "outdoor.sl" in bpy.context.preferences.studio_lights else sh.studio_light
    except Exception:
        pass
    scene.display.render_aa = "8"
    return scene


def render_camera(scene, cam_obj, path):
    scene.camera = cam_obj
    scene.render.filepath = path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.ops.render.render(write_still=True, scene=scene.name)
    return path


# --------------------------------------------------------------------------------------
# GLB export and validation
# --------------------------------------------------------------------------------------

GLB_EXPORT_OPTIONS = dict(
    export_format="GLB",
    check_existing=False,
    use_active_scene=True,          # only the active scene (the build activates it)
    use_selection=False,
    use_visible=False,
    use_renderable=False,
    use_active_collection=False,
    export_yup=True,                # glTF +Y up
    export_apply=True,              # modifiers applied
    export_texcoords=True,          # TEXCOORD_0
    export_normals=True,            # NORMAL
    export_tangents=False,
    export_materials="EXPORT",      # named placeholder materials only (no textures)
    export_image_format="NONE",
    export_vertex_color="ACTIVE",   # active/render colour attribute -> COLOR_0 (linear, RGBA)
    export_all_vertex_colors=False,
    export_active_vertex_color_when_no_material=True,
    export_attributes=False,
    use_mesh_edges=False,
    use_mesh_vertices=False,
    export_cameras=True,
    export_lights=False,
    export_extras=True,             # camera / key custom properties -> node extras
    export_animations=False,
    export_skins=False,
    export_morph=False,
    export_draco_mesh_compression_enable=False,   # (5.1.2's exporter has no meshopt option)
    export_use_gltfpack=False,
    export_gpu_instances=False,
    export_hierarchy_flatten_objs=False,
    export_shared_accessors=False,
    will_save_settings=False,
)


def export_glb(scene, path, **overrides):
    activate_scene(scene)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    opts = dict(GLB_EXPORT_OPTIONS)
    opts.update(overrides)
    bpy.ops.export_scene.gltf(filepath=path, **opts)
    return path


_COMP = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
_NP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}


def read_glb(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2:
        raise ValueError("not a glTF 2 binary")
    off = 12
    js, binb = None, None
    while off < length:
        clen, ctype = struct.unpack_from("<II", data, off)
        off += 8
        chunk = data[off:off + clen]
        off += clen
        if ctype == 0x4E4F534A:
            js = json.loads(chunk.decode("utf-8"))
        elif ctype == 0x004E4942:
            binb = chunk
    return js, binb


def glb_accessor(js, binb, idx):
    acc = js["accessors"][idx]
    bv = js["bufferViews"][acc["bufferView"]]
    ct = acc["componentType"]
    nc = _NCOMP[acc["type"]]
    start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = bv.get("byteStride", 0)
    itemsize = _COMP[ct][1] * nc
    if stride and stride != itemsize:
        raw = np.frombuffer(binb, np.uint8, count=stride * acc["count"], offset=start).reshape(-1, stride)[:, :itemsize]
        arr = np.frombuffer(raw.tobytes(), _NP[ct]).reshape(-1, nc)
    else:
        arr = np.frombuffer(binb, _NP[ct], count=acc["count"] * nc, offset=start).reshape(-1, nc)
    if acc.get("normalized"):
        arr = arr.astype(np.float64) / float(np.iinfo(_NP[ct]).max)
    return arr


_SUFFIX = re.compile(r"\.\d{3}$")


def validate_glb(path, max_total_tris=250_000, max_mesh_tris=120_000, max_bytes=8 * 1024 * 1024,
                 fov_v_deg=FOV_V_DEG, aspect=ASPECT, reimport=True):
    """Check a GLB against the asset contract. Returns a dict report (also printed)."""
    js, binb = read_glb(path)
    rep = dict(path=path, bytes=os.path.getsize(path), errors=[], warnings=[], nodes=[], meshes={},
               cameras={}, keys={}, totals={})
    err, warn = rep["errors"], rep["warnings"]
    nodes = js.get("nodes", [])
    names = [n.get("name", "") for n in nodes]
    if len(set(names)) != len(names):
        err.append("duplicate node names")
    total = 0
    for ni, n in enumerate(nodes):
        name = n.get("name", "")
        kind = "mesh" if "mesh" in n else "camera" if "camera" in n else "empty"
        rep["nodes"].append(dict(name=name, kind=kind))
        if _SUFFIX.search(name):
            err.append(f"node name with Blender suffix: {name}")
        if kind == "mesh":
            mesh = js["meshes"][n["mesh"]]
            tris = 0
            attrs = {}
            for prim in mesh["primitives"]:
                mode = prim.get("mode", 4)
                if mode != 4:
                    err.append(f"{name}: primitive mode {mode} (not triangles)")
                if "indices" in prim:
                    tris += js["accessors"][prim["indices"]]["count"] // 3
                else:
                    tris += js["accessors"][prim["attributes"]["POSITION"]]["count"] // 3
                for a, ai in prim["attributes"].items():
                    acc = js["accessors"][ai]
                    attrs[a] = dict(type=acc["type"], componentType=acc["componentType"],
                                    normalized=bool(acc.get("normalized", False)), count=acc["count"])
                if "COLOR_0" in prim["attributes"]:
                    c = glb_accessor(js, binb, prim["attributes"]["COLOR_0"])
                    attrs["COLOR_0"]["min"] = [round(float(x), 3) for x in c.min(0)]
                    attrs["COLOR_0"]["max"] = [round(float(x), 3) for x in c.max(0)]
                    attrs["COLOR_0"]["mean"] = [round(float(x), 3) for x in c.mean(0)]
            total += tris
            prefix = name.split("_")[0]
            rep["meshes"][name] = dict(tris=tris, attributes=attrs, material=[js["materials"][p["material"]]["name"]
                                                                              for p in mesh["primitives"] if "material" in p],
                                       fg=name.endswith("__fg"), transform=[k for k in ("translation", "rotation", "scale",
                                                                                       "matrix") if k in n])
            if tris > max_mesh_tris:
                err.append(f"{name}: {tris} tris > {max_mesh_tris}")
            for req in ("POSITION", "NORMAL", "COLOR_0"):
                if req not in attrs:
                    err.append(f"{name}: missing {req}")
            if prefix == "wood" and "TEXCOORD_0" not in attrs:
                err.append(f"{name}: wood mesh without TEXCOORD_0")
            if "COLOR_0" in attrs:
                a = attrs["COLOR_0"]
                if a["type"] != "VEC4":
                    err.append(f"{name}: COLOR_0 is {a['type']} (want VEC4)")
                if a["max"][3] < 0.999 or a["min"][3] < 0.999:
                    err.append(f"{name}: COLOR_0 alpha not 1")
            if prefix not in ("wood", "moss", "rock", "stone", "emblem", "far"):
                warn.append(f"{name}: unknown prefix")
            if rep["meshes"][name]["transform"]:
                warn.append(f"{name}: node has a transform {rep['meshes'][name]['transform']}")
        elif kind == "camera":
            cam = js["cameras"][n["camera"]]
            p = cam.get("perspective", {})
            yfov = math.degrees(p.get("yfov", 0.0))
            ar = p.get("aspectRatio", 0.0)
            rep["cameras"][name] = dict(yfov_deg=round(yfov, 4), aspect=round(ar, 5), znear=p.get("znear"),
                                        zfar=p.get("zfar"), translation=n.get("translation"),
                                        rotation=n.get("rotation"), extras=n.get("extras"))
            if abs(yfov - fov_v_deg) > 0.01:
                err.append(f"{name}: yfov {yfov:.3f} deg (want {fov_v_deg})")
            if abs(ar - aspect) > 1e-3:
                err.append(f"{name}: aspect {ar} (want {aspect:.5f})")
            if not re.match(r"^cam_[a-z0-9]+_(main|in|out|p\d+)$", name):
                err.append(f"camera name not cam_<episode>_<pose>: {name}")
        else:
            if name.startswith("key_"):
                q = n.get("rotation", [0, 0, 0, 1])
                qq = Quaternion((q[3], q[0], q[1], q[2]))
                d = qq @ Vector((0.0, 0.0, -1.0))
                rep["keys"][name] = dict(local_minus_z_gltf=[round(x, 4) for x in d],
                                         extras=n.get("extras"), translation=n.get("translation"))
    rep["totals"] = dict(tris=total, meshes=len(rep["meshes"]), cameras=len(rep["cameras"]),
                         materials=[m.get("name") for m in js.get("materials", [])])
    if total > max_total_tris:
        err.append(f"total tris {total} > {max_total_tris}")
    if rep["bytes"] > max_bytes:
        err.append(f"file {rep['bytes']} bytes > {max_bytes}")
    if js.get("animations"):
        err.append("file has animations")
    if "KHR_lights_punctual" in js.get("extensionsUsed", []):
        err.append("file has lights")
    if js.get("extensionsUsed"):
        warn.append(f"extensionsUsed: {js.get('extensionsUsed')}")
    if reimport:
        rep["reimport"] = _reimport_check(path)
    log("validate_glb:", json.dumps(dict(errors=err, warnings=warn, totals=rep["totals"]), indent=None))
    return rep


def _reimport_check(path):
    """Import the GLB into a temporary scene and report what Blender sees, then clean up."""
    win = bpy.context.window
    prev = win.scene if win is not None else bpy.context.scene
    kinds = ("objects", "meshes", "materials", "cameras", "images", "collections", "actions", "lights")
    before = {k: set(getattr(bpy.data, k).keys()) for k in kinds}
    tmp = bpy.data.scenes.new("__glb_validate__")
    out = dict(objects=[], ok=False)
    try:
        if win is not None:
            win.scene = tmp
        bpy.ops.import_scene.gltf(filepath=path)
        for obj in tmp.objects:
            item = dict(name=obj.name, type=obj.type)
            if obj.type == "MESH":
                me = obj.data
                me.calc_loop_triangles()
                item["tris"] = len(me.loop_triangles)
                item["color_attributes"] = [(a.name, a.data_type, a.domain) for a in me.color_attributes]
                item["uv_layers"] = [u.name for u in me.uv_layers]
                item["max_abs_coord"] = round(float(np.abs(np.array([v.co[:] for v in me.vertices[:1]])).max()), 3) if len(me.vertices) else 0
            elif obj.type == "CAMERA":
                item["angle_y_deg"] = round(math.degrees(obj.data.angle_y), 4)
                item["sensor_fit"] = obj.data.sensor_fit
                item["location"] = [round(x, 4) for x in obj.matrix_world.translation]
            out["objects"].append(item)
        out["ok"] = True
    except Exception as e:  # report, never crash the build
        out["error"] = repr(e)
    finally:
        if win is not None:
            win.scene = prev
        for obj in list(tmp.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.scenes.remove(tmp)
        for k in kinds:
            coll = getattr(bpy.data, k)
            for nm in set(coll.keys()) - before[k]:
                blk = coll.get(nm)
                if blk is not None:
                    try:
                        coll.remove(blk)
                    except Exception:
                        pass
    return out
