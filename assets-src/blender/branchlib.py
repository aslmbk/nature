"""
branchlib.py - geometry helpers owned by build_branch.py (the 'branch' scene set, S06).

Built on common.py (sweep_trunk with build_object=False, _sweep_object for the mesh + caps,
mesh / uv / colour helpers); nothing here is shared with the other builds. Deterministic: every
random number comes from an explicit numpy Generator or a seeded common.Noise3.

    retwist        variable grain twist along a sweep (rad/m); keeps U = wraps * grain / 2 pi
    grain_noise    fbm on a sweep grid in grain space (periodic around, stretched along the grain)
    worley_edges   Voronoi cells in grain space (F2 - F1 in metres): moss clumps
    strand_relief  long rope-like strands with V creases that wander, merge and split
    make_cracks    random grain-following cracks (optionally branching)
    crack_relief   deep V cracks with flaring lips -> relief (<= 0) and a crack mask
    fibre_relief   faint fibre striation + bark plates along the twisted grain
    cross_fissures short fissures across the strands of strand_relief (strands -> long plates)
    reshape_section  oval / kidney / bumpy cross-sections (scales the radial offset per ring)
    dir_band       angular band around a world direction on a sweep (moss sheaths on the stumps)
    finish_sweep   relief -> final surface + mesh object (caps through common._sweep_object)
    moss_coat      lumpy moss sheet on a sweep: Voronoi clumps 3-8 cm, billows, mounds, thin
                   ragged edge, sagging underside; same info keys as common.moss_cushions
    end_mound      moss dome over the end face of a sweep, continuous with its moss coat
    end_crescent   the same, kept only where a mask over (rho, direction) is high (an annulus
                   over a broken rim), tucked below the end plane elsewhere
    bake_lit ...   vertex-lit Workbench previews (key light + sky fill + baked AO + cast shadows)
"""

import math

import numpy as np

import common as C

TAU = 2.0 * math.pi


def wrap(a):
    return (np.asarray(a, float) + math.pi) % TAU - math.pi


def perp(v, T):
    """Per ring: v without its T component, normalised. v (3,) or (N, 3)."""
    v = np.asarray(v, float)
    if v.ndim == 1:
        v = np.broadcast_to(v, T.shape)
    return C.normalize(v - np.sum(v * T, -1)[:, None] * T)


def ring_angle(sw, d):
    """Sweep angle theta of world directions d (N, 3) on each ring (vertex dir = cos Rf + sin Bf)."""
    d = perp(d, sw.T)
    return np.arctan2(np.sum(d * sw.Bf, -1), np.sum(d * sw.Rf, -1))


def rotate(vec, axis, a):
    """Rodrigues rotation of vec (N, 3) about axis (N, 3) by a (N,) (or scalars)."""
    axis = C.normalize(np.asarray(axis, float))
    vec = np.asarray(vec, float)
    a = np.asarray(a, float)
    if vec.ndim == 2:
        a = a[:, None] if a.ndim == 1 else a
        dot = np.sum(axis * vec, -1)[:, None]
    else:
        dot = axis @ vec
    return vec * np.cos(a) + np.cross(axis, vec) * np.sin(a) + axis * dot * (1.0 - np.cos(a))


# ------------------------------------------------------------------------------------------
# grain twist and grain-space noise
# ------------------------------------------------------------------------------------------

def retwist(sw, rate, noise=None, amp=0.0, freq=0.6, offset=0.0):
    """Replace the sweep's linear twist by tw(s) = integral of rate (N,) rad/m (+ slow noise).
    Updates sw.twist_s, sw.grain and the U of sw.uv (integer wraps kept, V untouched)."""
    s = np.asarray(sw.s, float)
    rate = np.broadcast_to(np.asarray(rate, float), s.shape)
    tw = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(s))])
    if noise is not None and amp:
        tw = tw + amp * noise.line(s, freq=freq, offset=offset)
    sw.twist_s = tw
    sw.grain = np.asarray(sw.theta, float)[None, :] - tw[:, None]
    th_ext = np.append(sw.theta, TAU)
    uv = np.array(sw.uv, float)
    uv[..., 0] = sw.wraps * (th_ext[None, :] - tw[:, None]) / TAU
    sw.uv = uv
    return tw


def grain_noise(sw, noise, freq, stretch=1.0, octaves=2, offset=0.0, radius=None):
    """fbm on the sweep grid in grain space: features ~1/freq m around the trunk, stretch x longer
    along the grain; periodic around (sampled on circles)."""
    R = np.asarray(sw.radius if radius is None else radius, float)[:, None]
    g = sw.grain
    Q = np.stack([np.cos(g) * R * freq, np.sin(g) * R * freq,
                  np.broadcast_to((np.asarray(sw.s, float) * freq / stretch)[:, None], g.shape)], -1)
    return noise.fbm(Q + offset, octaves)


# ------------------------------------------------------------------------------------------
# bark relief: cracks, fibres, plates
# ------------------------------------------------------------------------------------------

def make_cracks(sw, rng, n, *, depth, width, length, s_lim=(0.0, 1.0), meander=0.06, branch=0.0,
                g_centre=None, g_spread=None):
    """n random cracks following the twisted grain. depth / width (half-width at the lips) /
    length: (min, max) in metres. g_centre / g_spread: restrict the start grain angle (rad)."""
    L = float(sw.length)
    out = []
    if g_centre is None:
        g0s = (np.arange(n) + rng.uniform(0.15, 0.85, n)) / n * TAU
        rng.shuffle(g0s)
    else:
        g0s = g_centre + g_spread * (2.0 * rng.uniform(0.0, 1.0, n) - 1.0)
    for k in range(n):
        ln = rng.uniform(*length)
        lo, hi = s_lim[0] * L, s_lim[1] * L
        a = rng.uniform(lo - 0.3 * ln, max(lo - 0.3 * ln, hi - 0.7 * ln))
        c = dict(g0=float(g0s[k]), s0=float(a), s1=float(a + ln), depth=float(rng.uniform(*depth)),
                 width=float(rng.uniform(*width)), meander=float(meander * rng.uniform(0.5, 1.5)),
                 mfreq=float(rng.uniform(1.5, 3.5)), moff=float(rng.uniform(0.0, 60.0)),
                 sharp=float(rng.uniform(1.3, 2.1)), taper=float(rng.uniform(0.08, 0.2)))
        out.append(c)
        if branch and rng.uniform() < branch:
            sb = float(rng.uniform(c["s0"] + 0.25 * ln, c["s0"] + 0.6 * ln))
            out.append(dict(c, s0=sb, s1=float(sb + rng.uniform(0.3, 0.7) * ln), depth=c["depth"] * 0.7,
                            width=c["width"] * 0.8, parent_s=sb,
                            diverge=float(rng.choice([-1.0, 1.0]) * rng.uniform(0.25, 0.5)),
                            taper=0.06))
    return out


def crack_relief(sw, cracks, noise, r_ref=None):
    """Deep V cracks with flaring lips. Crack centre: dict 'theta' (N,) explicit sweep angle, or
    grain-following g0 + twist(s) + meander (+ divergence after 'parent_s'). Depth scales with
    the radius (r_ref). Returns relief (N, M) <= 0 and mask (N, M) in [0, 1]."""
    N, M = sw.N, sw.M
    relief = np.zeros((N, M))
    mask = np.zeros((N, M))
    R = np.asarray(sw.radius, float)
    r_ref = float(np.mean(R)) if r_ref is None else float(r_ref)
    th = np.asarray(sw.theta, float)[None, :]
    S = np.asarray(sw.s, float)
    for c in cracks:
        i0 = int(np.searchsorted(S, c["s0"] - 0.02))
        i1 = int(np.searchsorted(S, c["s1"] + 0.02))
        if i1 - i0 < 2:
            continue
        rows = slice(i0, i1)
        s = S[rows]
        if "theta" in c:
            centre = np.asarray(c["theta"], float)[rows]
        else:
            centre = c["g0"] + sw.twist_s[rows] + c["meander"] * noise.line(s, freq=c["mfreq"], offset=c["moff"])
            if "parent_s" in c:
                centre = centre + c["diverge"] * np.clip((s - c["parent_s"]) / 0.35, 0.0, 1.0) ** 1.4
        tp = min(c.get("taper", 0.12), 0.45 * (c["s1"] - c["s0"]))
        env = C.smoothstep(c["s0"], c["s0"] + tp, s) * (1.0 - C.smoothstep(c["s1"] - tp, c["s1"], s))
        var = 0.6 + 0.4 * np.clip(0.5 + noise.line(s, freq=5.0, offset=c["moff"] + 7.0), 0.0, 1.0)
        w = c["width"] * (0.55 + 0.45 * env) * (1.0 + 0.3 * noise.line(s, freq=7.0, offset=c["moff"] + 3.0))
        w = np.maximum(w, 1e-4)
        x = np.abs(wrap(th - centre[:, None])) * R[rows, None]
        prof = np.clip(1.0 - x / w[:, None], 0.0, 1.0) ** c["sharp"]
        d = c["depth"] * (R[rows] / r_ref) * env * var
        relief[rows] = np.minimum(relief[rows], -d[:, None] * prof)
        mask[rows] = np.maximum(mask[rows], (prof ** 0.7) * env[:, None])
    return relief, mask


def worley_edges(sw, rng, n_around, cell_len, jitter=0.85):
    """Voronoi cells in grain space: n_around cells per circumference (they widen with the radius),
    cell_len m along the grain (elongated cells = strands / plates). Returns (F2 - F1, F1) in
    metres per grid vertex; F2 - F1 ~ 0 on the cell borders."""
    g = np.mod(sw.grain, TAU)
    x = g * n_around / TAU
    y = np.broadcast_to((np.asarray(sw.s, float) / cell_len)[:, None], g.shape)
    nj = int(np.ceil(float(sw.length) / cell_len)) + 4
    pts = rng.uniform(0.5 - 0.5 * jitter, 0.5 + 0.5 * jitter, size=(n_around, nj, 2))
    ix = np.floor(x).astype(np.int64)
    iy = np.floor(y).astype(np.int64)
    fx, fy = x - ix, y - iy
    ax = TAU * np.asarray(sw.radius, float)[:, None] / n_around
    F1 = np.full(g.shape, np.inf)
    F2 = np.full(g.shape, np.inf)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ci = (ix + di) % n_around
            cj = np.clip(iy + dj + 1, 0, nj - 1)
            p = pts[ci, cj]
            dx = (di + p[..., 0] - fx) * ax
            dy = (dj + p[..., 1] - fy) * cell_len
            d = np.sqrt(dx * dx + dy * dy)
            F2 = np.where(d < F1, F1, np.minimum(F2, d))
            F1 = np.minimum(F1, d)
    return F2 - F1, F1


def strand_relief(sw, noise, *, n_around, depth, r_ref, meander=0.45, merge=1.0, crown=0.85, offset=0.0):
    """Long rounded strands (rope-like domes) separated by V creases, following the twisted grain:
    n_around strands per circumference, crease lines warped by low-frequency grain-space noise
    (meander, in strand widths) so strands swell, narrow and wander; crease depth varies along the
    grain down to ~0 (merge) so strands join and split. No transverse creases.
    depth: m at radius r_ref (scales with the radius). crown: share of the strand width that is
    rounded (1 = the dome spans the whole strand). Returns (relief <= 0, crease mask 0..1)."""
    warp = (meander * grain_noise(sw, noise, 2.4, stretch=3.0, octaves=2, offset=offset) +
            0.35 * meander * grain_noise(sw, noise, 7.0, stretch=2.5, octaves=2, offset=offset + 9.0))
    x = sw.grain * n_around / TAU + warp
    f = x - np.floor(x)
    u = 2.0 * np.minimum(f, 1.0 - f)                                   # 0 at a crease, 1 mid-strand
    shoulder = 1.0 - (1.0 - np.clip(u / crown, 0.0, 1.0)) ** 2
    var = np.clip(0.4 + merge * 1.5 * grain_noise(sw, noise, 3.2, stretch=4.0, octaves=2, offset=offset + 17.0),
                  0.05, 1.3)
    d = depth * (np.asarray(sw.radius, float)[:, None] / r_ref) * var
    crease = (1.0 - C.smoothstep(0.0, 0.35 * crown, u)) * C.smoothstep(0.35, 0.9, var)
    return d * (shoulder - 1.0), crease


def fibre_relief(sw, noise, *, spacing=0.022, amp=0.0016, plate_amp=0.003, plate_freq=14.0, offset=0.0):
    """Fine fibre striation (ridges ~spacing apart around, meandering along the grain) and long
    bark plates (anisotropic noise stretched 5x along the grain). Small: +-amp / +-plate_amp m."""
    R = float(np.mean(sw.radius))
    n = max(8, int(round(TAU * R / spacing)))
    mea = grain_noise(sw, noise, 3.0, stretch=4.0, octaves=2, offset=offset + 5.0)
    x = sw.grain * n / TAU + 0.8 * mea
    f = x - np.floor(x)
    ridge = C.smoothstep(0.0, 0.3, f) * C.smoothstep(0.0, 0.3, 1.0 - f)
    gate = C.smoothstep(-0.2, 0.4, grain_noise(sw, noise, 9.0, stretch=3.0, octaves=2, offset=offset + 19.0))
    fib = amp * (ridge - 0.65) * (0.35 + 0.65 * gate)
    plates = plate_amp * grain_noise(sw, noise, plate_freq, stretch=5.0, octaves=2, offset=offset + 31.0)
    return fib + plates


def _hash01(a, b, salt):
    """Deterministic per-cell random numbers in [0, 1) for integer arrays a, b (any shape)."""
    a = np.asarray(a).astype(np.uint64)
    b = np.asarray(b).astype(np.uint64)
    with np.errstate(over="ignore"):
        h = (a * np.uint64(73856093)) ^ (b * np.uint64(19349663)) ^ np.uint64((int(salt) * 83492791) & 0xFFFFFFFF)
        h = h ^ (h >> np.uint64(13))
        h = h * np.uint64(1274126177)
        h = h ^ (h >> np.uint64(16))
    return (h & np.uint64(0xFFFFFF)).astype(np.float64) / float(0x1000000)


def cross_fissures(sw, rng, noise, *, n_around, spacing, depth, width, r_ref, meander=0.45, offset=0.0,
                   extent=(0.18, 0.45), p=0.7, skew=0.8, crown=(0.08, 0.3), steep=0.35):
    """Short fissures across the grain, on the strands of strand_relief (same n_around, meander
    and offset -> the same strands): per strand, a fissure every spacing (min, max) m along the
    grain (random phase per strand), each one spanning part of the strand only (extent = half-length
    in strand widths, random centre), a little diagonal (skew, in strand-width units of along-shift)
    and only on the crown (not in the creases); asymmetric V (one steep lip: `steep` share of the
    width), width = half-width (m), depth m at r_ref. Strands so read as stacks of scales / plates
    instead of continuous ropes. Returns (relief <= 0, mask 0..1)."""
    warp = (meander * grain_noise(sw, noise, 2.4, stretch=3.0, octaves=2, offset=offset) +
            0.35 * meander * grain_noise(sw, noise, 7.0, stretch=2.5, octaves=2, offset=offset + 9.0))
    x = sw.grain * n_around / TAU + warp
    kf = np.floor(x)
    f = x - kf
    k = np.mod(kf.astype(np.int64), n_around)
    sp = rng.uniform(spacing[0], spacing[1], n_around)
    ph = rng.uniform(0.0, 1.0, n_around)
    sk = rng.uniform(-1.0, 1.0, n_around) * skew
    s = np.broadcast_to(np.asarray(sw.s, float)[:, None], f.shape)
    wob = 0.18 * grain_noise(sw, noise, 30.0, stretch=1.0, octaves=2, offset=offset + 23.0)
    y = s / sp[k] + ph[k] + sk[k] * (f - 0.5) + wob
    jf = np.floor(y)
    fy = y - jf
    j = jf.astype(np.int64)
    h1, h2, h3, h4 = (_hash01(k, j, q) for q in (1, 2, 3, 4))
    on = (h1 < p).astype(float)
    cu = 0.22 + 0.56 * h2
    hl = extent[0] + (extent[1] - extent[0]) * h3
    across = 1.0 - C.smoothstep(0.55 * hl, hl, np.abs(f - cu))
    crw = C.smoothstep(crown[0], crown[1], f) * C.smoothstep(crown[0], crown[1], 1.0 - f)
    # distance (m) along the grain from the fissure line, one lip steep, the other gentle
    up = fy * sp[k]
    dn = (1.0 - fy) * sp[k]
    w = np.maximum(float(width), 1e-4)
    prof = np.maximum(np.clip(1.0 - up / (steep * 2.0 * w), 0.0, 1.0), np.clip(1.0 - dn / ((2.0 - steep * 2.0) * w), 0.0, 1.0))
    prof = prof ** 1.4
    d = depth * (np.asarray(sw.radius, float)[:, None] / r_ref) * (0.55 + 0.9 * h4)
    mask = prof * across * crw * on
    return -d * mask, mask


def reshape_section(sw, lobes, noise=None, bumps=0.0, bump_freq=40.0, bump_stretch=1.5, offset=0.0):
    """Shape the cross-section of a sweep (before the relief): the radial offset from the centre
    line is scaled by 1 + sum amp cos(k (theta - theta0)) + bumps * fbm. lobes: [(k, amp,
    theta0 (N,) per ring or scalar)], e.g. an oval whose long axis follows a world direction
    (theta0 = ring_angle(sw, d)) and turns along the length. Bumps: grain-space fbm (freq 1/m
    around, stretched along) -> bulges and dents. Recomputes the smooth normals."""
    th = np.asarray(sw.theta, float)[None, :]
    k_ = np.ones((sw.N, sw.M))
    for (k, amp, th0) in lobes:
        th0 = np.broadcast_to(np.asarray(th0, float), (sw.N,))[:, None]
        amp = np.broadcast_to(np.asarray(amp, float), (sw.N,))[:, None]
        k_ = k_ + amp * np.cos(k * (th - th0))
    if bumps and noise is not None:
        k_ = k_ + bumps * grain_noise(sw, noise, bump_freq, stretch=bump_stretch, octaves=3, offset=offset)
    k_ = np.maximum(k_, 0.25)
    X = sw.centres[:, None, :]
    sw.smooth = X + (sw.smooth - X) * k_[..., None]
    sw.nsmooth = C._grid_normals(sw.smooth)
    sw.pos = sw.smooth.copy()
    return k_


def dir_band(sw, d, W, jitter=None):
    """Angular distance (rad) of every grid vertex from the radial direction closest to world
    direction d (per ring, d projected on the ring plane), minus the half-width W (N,) or scalar:
    < 0 inside the band."""
    dd = perp(np.asarray(d, float), sw.T)
    dirs = C.normalize(sw.smooth - sw.centres[:, None, :])
    ang = np.arccos(np.clip(np.einsum("nmk,nk->nm", dirs, dd), -1.0, 1.0))
    if jitter is not None:
        ang = ang + jitter
    return ang - np.broadcast_to(np.asarray(W, float), (sw.N,))[:, None]


def finish_sweep(sw, relief, name, coll, caps, material, seed):
    """Apply relief (N, M) along the smooth normals and build the mesh object (caps through
    common._sweep_object, exactly as common.sweep_trunk would)."""
    sw.relief = np.asarray(relief, float)
    sw.relief_max = float(sw.relief.max())
    sw.pos = sw.smooth + sw.nsmooth * sw.relief[..., None]
    sw.obj = C._sweep_object(name, coll, sw, caps, material, C.Noise3(seed + 101))
    return sw.obj


# ------------------------------------------------------------------------------------------
# moss coat
# ------------------------------------------------------------------------------------------

def moss_coat(name, coll, sw, mask, *, base, lump, noise, material=None, border=(0.3, 0.72), tuck=0.012,
              keep_above=0.2, sag=0.35, lump_freq=(15.0, 30.0), mound_freq=4.5, thin_edge=(0.4, 0.95),
              seed_off=0.0, max_off=None, ramp=None, rng=None, clump=0.065):
    """Lumpy moss sheet hugging a sweep where mask (N, M) is high.

    base (N, M) / scalar: thickness (m) of the carpet; lump: height (m) of the cushions on top:
    Voronoi clumps ~`clump` m across (rounded domes with sharp creases between them, when rng is
    given), billow bumps (|fbm| at lump_freq) and broad mounds (mound_freq). The carpet thins out
    towards the mask border (thin_edge, or an explicit ramp (N, M) 0..1) and ends `tuck` m below
    the bark relief, so the edge is a low, ragged contour of the mask - not a lip. Faces turned
    down sag (droop by sag x thickness). Returns (obj, info) with the keys of
    common.moss_cushions (offset, relief, thickness per vertex) for the colours."""
    N, M = sw.N, sw.M
    m = np.asarray(mask, float)
    P = sw.smooth
    nrm = sw.nsmooth
    sh = C.smoothstep(border[0], border[1], m)
    so = float(seed_off)
    bil1 = np.abs(noise.fbm(P * lump_freq[0] + 3.1 + so, 2))
    bil2 = np.abs(noise.fbm(P * lump_freq[1] + 17.3 + so, 2))
    mound = noise.fbm(P * mound_freq + 41.7 + so, 2)
    if rng is not None and clump:
        n_ar = max(4, int(round(TAU * float(np.mean(sw.radius)) / clump)))
        e, _ = worley_edges(sw, rng, n_ar, clump, jitter=0.9)
        vor = C.smoothstep(0.0, 0.55 * clump, e) ** 0.6
        dome = np.clip(0.7 * vor + 0.3 * bil1 + 0.2 * bil2, 0.0, 1.0)
    else:
        dome = np.clip(0.75 * bil1 + 0.4 * bil2, 0.0, 1.0) ** 0.8
    T = np.broadcast_to(np.asarray(base, float), (N, M)) * (0.75 + 0.5 * C.smoothstep(-0.4, 0.5, mound))
    T = T + np.broadcast_to(np.asarray(lump, float), (N, M)) * (0.8 * dome + 0.35 * np.clip(mound, 0.0, 1.0))
    T = T * (0.3 + 0.7 * (C.smoothstep(thin_edge[0], thin_edge[1], m) if ramp is None else np.asarray(ramp, float)))
    if max_off is not None:
        T = np.minimum(T, max_off)
    rel = sw.relief if sw.relief is not None else np.zeros((N, M))
    clear = max(float(np.max(rel)), 0.0)
    off = (rel - tuck) * (1.0 - sh) + (clear + T) * sh
    pos = P + nrm * off[..., None]
    if sag:
        down = np.clip(-nrm[..., 2], 0.0, 1.0)
        pos[..., 2] -= sag * T * sh * down * (0.6 + 0.8 * dome)
    vid = np.arange(N * M).reshape(N, M)
    i0, j0 = np.meshgrid(np.arange(N - 1), np.arange(M), indexing="ij")
    j1 = (j0 + 1) % M
    quads = np.stack([vid[i0, j0], vid[i0, j1], vid[i0 + 1, j1], vid[i0 + 1, j0]], -1).reshape(-1, 4)
    keep = (m.reshape(-1)[quads] > keep_above).all(1)
    quads = quads[keep]
    if not len(quads):
        return None, None
    uvg = sw.uv
    a_i, a_j = np.divmod(quads[:, 0], M)
    c_i = np.divmod(quads[:, 2], M)[0]
    j_next = a_j + 1
    uv_l = np.stack([uvg[a_i, a_j], uvg[a_i, j_next], uvg[c_i, j_next], uvg[c_i, a_j]], 1).reshape(-1, 2)
    used = np.unique(quads)
    remap = -np.ones(N * M, np.int64)
    remap[used] = np.arange(len(used))
    me = C.mesh_from_numpy(name, pos.reshape(-1, 3)[used], remap[quads], smooth=True)
    C.set_uvs(me, uv_l)
    obj = C.object_from_mesh(name, me, coll, material)
    info = dict(offset=off.reshape(-1)[used], shape=sh.reshape(-1)[used], thickness=T.reshape(-1)[used],
                relief=rel.reshape(-1)[used], dome=dome.reshape(-1)[used], used=used, n=len(used),
                pos_grid=pos, off_grid=off, thick_grid=T, shape_grid=sh)
    return obj, info


def end_mound(name, coll, sw, coat_info, *, height, lump, noise, rng=None, clump=0.04, rings=14, material=None,
              seed_off=0.0):
    """Moss mound over the end face of a sweep (the stumps / fibres rise through it): a polar sheet
    from the end-ring centre (raised by `height` m along the end tangent) out to the moss coat's
    last ring (pos_grid of moss_coat), elliptic in section so coat and mound meet without a step or
    a corner. Lumps from Voronoi clumps + billows. Returns (obj, info) with the moss_coat info keys."""
    N, M = sw.N, sw.M
    c = sw.centres[N - 1]
    t = sw.T[N - 1]
    rim = np.asarray(coat_info["pos_grid"], float)[N - 1]                 # (M, 3)
    rim_T = np.asarray(coat_info["thick_grid"], float)[N - 1]
    rho = np.linspace(0.0, 1.0, rings + 1)[1:]
    P = c[None, None, :] + rho[:, None, None] * (rim - c)[None, :, :]
    bil = np.abs(noise.fbm(P * 26.0 + 5.0 + seed_off, 2))
    if rng is not None and clump:
        th = np.arctan2(np.sum((rim - c) * sw.Bf[N - 1], -1), np.sum((rim - c) * sw.Rf[N - 1], -1))
        ang = np.broadcast_to(th[None, :], (rings, M))
        r_m = rho[:, None] * np.linalg.norm(rim - c, axis=-1)[None, :]
        xy = np.stack([np.cos(ang) * r_m, np.sin(ang) * r_m], -1) / clump
        pts = rng.uniform(0.0, 1.0, size=(12, 12, 2))
        ix = np.floor(xy).astype(np.int64)
        f = xy - ix
        F1 = np.full(ang.shape, np.inf)
        F2 = np.full(ang.shape, np.inf)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                p = pts[(ix[..., 0] + di) % 12, (ix[..., 1] + dj) % 12]
                d = np.sqrt((di + p[..., 0] - f[..., 0]) ** 2 + (dj + p[..., 1] - f[..., 1]) ** 2)
                F2 = np.where(d < F1, F1, np.minimum(F2, d))
                F1 = np.minimum(F1, d)
        vor = C.smoothstep(0.0, 0.55, F2 - F1) ** 0.6
        dome = np.clip(0.7 * vor + 0.3 * bil, 0.0, 1.0)
    else:
        dome = np.clip(bil, 0.0, 1.0)
    # elliptic dome: vertical tangent at the rim, so the coat runs on over the end without a corner
    prof = np.sqrt(np.clip(1.0 - rho ** 2, 0.0, 1.0))[:, None]
    h = prof * (height + lump * dome)
    P = P + t[None, None, :] * h[..., None]
    d0 = float(dome[0].mean())
    verts = np.concatenate([(c + t * (height + lump * d0))[None, :], P.reshape(-1, 3)], 0)
    jj = np.arange(M)
    tris = np.stack([np.zeros(M, np.int64), 1 + jj, 1 + (jj + 1) % M], -1)
    ii, jg = np.meshgrid(np.arange(rings - 1), jj, indexing="ij")
    a = 1 + ii * M + jg
    b = 1 + ii * M + (jg + 1) % M
    quads = np.stack([a, a + M, b + M, b], -1).reshape(-1, 4)
    v0, v1, v2 = verts[tris[0, 0]], verts[tris[0, 1]], verts[tris[0, 2]]
    if float(np.cross(v1 - v0, v2 - v0) @ t) < 0.0:                    # faces must look out along +t
        tris, quads = tris[:, ::-1].copy(), quads[:, ::-1].copy()
    me = C.mesh_from_numpy(name, verts, [tris, quads], smooth=True)
    # polar UVs, 1 unit = 0.5 m like the coat (radius in metres / 0.5 around the centre)
    rr = np.concatenate([[0.0], (rho[:, None] * np.linalg.norm(rim - c, axis=-1)[None, :]).reshape(-1)])
    aa = np.concatenate([[0.0], np.broadcast_to(TAU * jj / M, (rings, M)).reshape(-1)])
    uv_v = np.stack([rr * np.cos(aa), rr * np.sin(aa)], -1) / 0.5
    C.set_uvs(me, np.concatenate([uv_v[tris.reshape(-1)], uv_v[quads.reshape(-1)]], 0))
    obj = C.object_from_mesh(name, me, coll, material)
    # colour info: a full-density cushion (vis = offset - relief = thickness), thinning to the
    # coat's own thickness at the rim
    thick = np.concatenate([[height + lump * d0], (h + rho[:, None] * np.maximum(rim_T, 0.0)[None, :]).reshape(-1)])
    thick = np.maximum(thick, 1e-3)
    n = len(verts)
    info = dict(offset=thick.copy(), shape=np.ones(n), thickness=thick, relief=np.zeros(n),
                dome=np.concatenate([[d0], dome.reshape(-1)]), used=np.arange(n), n=n)
    return obj, info


def end_crescent(name, coll, sw, coat_info, *, keep, height, lump, noise, rng=None, clump=0.04, rings=14,
                 material=None, seed_off=0.0, tuck=0.035):
    """Partial moss mound over the end face of a sweep: like end_mound (polar sheet from the end
    ring's centre to the coat's last ring, elliptic in section, Voronoi / billow lumps) but only
    where keep(rho (R,), dirs (M, 3)) -> (R, M) mask in 0..1 is high; towards the open side the
    sheet sinks `tuck` m below the ring plane (into the sunken fracture) and faces whose corners
    are all open are dropped. The fracture of the log end stays visible there. Returns (obj, info)
    with the moss_coat info keys (None, None when nothing is kept)."""
    N, M = sw.N, sw.M
    c = sw.centres[N - 1]
    t = sw.T[N - 1]
    rim = np.asarray(coat_info["pos_grid"], float)[N - 1]
    rim_T = np.asarray(coat_info["thick_grid"], float)[N - 1]
    rho = np.linspace(0.0, 1.0, rings + 1)[1:]
    radial = rim - c
    radial = radial - (radial @ t)[:, None] * t
    dirs = C.normalize(radial)
    P = c[None, None, :] + rho[:, None, None] * (rim - c)[None, :, :]
    bil = np.abs(noise.fbm(P * 26.0 + 5.0 + seed_off, 2))
    if rng is not None and clump:
        th = np.arctan2(np.sum((rim - c) * sw.Bf[N - 1], -1), np.sum((rim - c) * sw.Rf[N - 1], -1))
        ang = np.broadcast_to(th[None, :], (rings, M))
        r_m = rho[:, None] * np.linalg.norm(rim - c, axis=-1)[None, :]
        xy = np.stack([np.cos(ang) * r_m, np.sin(ang) * r_m], -1) / clump
        pts = rng.uniform(0.0, 1.0, size=(12, 12, 2))
        ix = np.floor(xy).astype(np.int64)
        f = xy - ix
        F1 = np.full(ang.shape, np.inf)
        F2 = np.full(ang.shape, np.inf)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                p = pts[(ix[..., 0] + di) % 12, (ix[..., 1] + dj) % 12]
                d = np.sqrt((di + p[..., 0] - f[..., 0]) ** 2 + (dj + p[..., 1] - f[..., 1]) ** 2)
                F2 = np.where(d < F1, F1, np.minimum(F2, d))
                F1 = np.minimum(F1, d)
        dome = np.clip(0.7 * C.smoothstep(0.0, 0.55, F2 - F1) ** 0.6 + 0.3 * bil, 0.0, 1.0)
    else:
        dome = np.clip(bil, 0.0, 1.0)
    km = np.clip(np.asarray(keep(rho, dirs), float), 0.0, 1.0)
    kc = float(np.clip(np.asarray(keep(np.zeros(1), dirs), float), 0.0, 1.0).mean())
    on = C.smoothstep(0.25, 0.75, km)
    prof = np.sqrt(np.clip(1.0 - rho ** 2, 0.0, 1.0))[:, None]
    h = prof * (height + lump * dome) * on - tuck * (1.0 - on)
    P = P + t[None, None, :] * h[..., None]
    d0 = float(dome[0].mean())
    hc = (height + lump * d0) * float(C.smoothstep(0.25, 0.75, kc)) - tuck * (1.0 - float(C.smoothstep(0.25, 0.75, kc)))
    verts = np.concatenate([(c + t * hc)[None, :], P.reshape(-1, 3)], 0)
    jj = np.arange(M)
    tris = np.stack([np.zeros(M, np.int64), 1 + jj, 1 + (jj + 1) % M], -1)
    ii, jg = np.meshgrid(np.arange(rings - 1), jj, indexing="ij")
    a = 1 + ii * M + jg
    b = 1 + ii * M + (jg + 1) % M
    quads = np.stack([a, a + M, b + M, b], -1).reshape(-1, 4)
    v0, v1, v2 = verts[tris[0, 0]], verts[tris[0, 1]], verts[tris[0, 2]]
    if float(np.cross(v1 - v0, v2 - v0) @ t) < 0.0:                    # faces must look out along +t
        tris, quads = tris[:, ::-1].copy(), quads[:, ::-1].copy()
    vk = np.concatenate([[kc], km.reshape(-1)])
    tris = tris[(vk[tris] > 0.12).any(1)]
    quads = quads[(vk[quads] > 0.12).any(1)]
    if not len(tris) and not len(quads):
        return None, None
    used = np.unique(np.concatenate([tris.reshape(-1), quads.reshape(-1)]))
    remap = -np.ones(len(verts), np.int64)
    remap[used] = np.arange(len(used))
    rr = np.concatenate([[0.0], (rho[:, None] * np.linalg.norm(rim - c, axis=-1)[None, :]).reshape(-1)])
    aa = np.concatenate([[0.0], np.broadcast_to(TAU * jj / M, (rings, M)).reshape(-1)])
    uv_v = np.stack([rr * np.cos(aa), rr * np.sin(aa)], -1) / 0.5
    me = C.mesh_from_numpy(name, verts[used], [remap[tris], remap[quads]], smooth=True)
    C.set_uvs(me, np.concatenate([uv_v[tris.reshape(-1)], uv_v[quads.reshape(-1)]], 0))
    obj = C.object_from_mesh(name, me, coll, material)
    thick = np.concatenate([[max(hc, 0.0)], (np.maximum(h, 0.0) + rho[:, None] * np.maximum(rim_T, 0.0)[None, :]
                                             * on).reshape(-1)])
    thick = np.maximum(thick, 1e-3)[used]
    n = len(used)
    vis = thick * np.concatenate([[float(C.smoothstep(0.25, 0.75, kc))], on.reshape(-1)])[used]
    info = dict(offset=vis.copy(), shape=np.ones(n), thickness=thick, relief=np.zeros(n),
                dome=np.concatenate([[d0], dome.reshape(-1)])[used], used=used, n=n)
    return obj, info


def concat_info(infos):
    """Concatenate per-vertex info dicts (same keys) in join order."""
    keys = ("offset", "shape", "thickness", "relief", "dome")
    return {k: np.concatenate([np.asarray(i[k], float) for i in infos]) for k in keys}


# ------------------------------------------------------------------------------------------
# vertex-lit preview (Workbench, colour attribute; never exported - removed after rendering)
# ------------------------------------------------------------------------------------------

LIT_ATTR = "silva_lit"


def bake_lit(objs, key_dir, bvh, albedo_fn, sky=(0.20, 0.22, 0.25), key=(1.0, 0.95, 0.86), key_gain=1.3,
             sky_up=(0.0, 0.0, 1.0)):
    """Per-vertex Lambert colour: albedo x (sky fill (hemisphere, up-weighted) x AO + key x
    max(0, n.-L) x shadow). AO from COLOR_0.B. Written as a separate colour attribute that is made
    active for the preview render only (restore with unbake_lit)."""
    L = -C.normalize(np.asarray(key_dir, float))
    up = C.normalize(np.asarray(sky_up, float))
    for o in objs:
        me = o.data
        co, nrm, _, _ = C.mesh_arrays(me)
        Mw = np.array(o.matrix_world)
        co = co @ Mw[:3, :3].T + Mw[:3, 3]
        nrm = C.normalize(nrm @ Mw[:3, :3].T)
        col = C.read_color0(me)
        ao = col[:, 2].astype(float) if col is not None else np.ones(len(co))
        ndl = np.clip(nrm @ L, 0.0, 1.0)
        sh = np.ones(len(co))
        if bvh is not None:
            org = (co + nrm * 0.004).tolist()
            dl = list(map(float, L))
            for i in np.nonzero(ndl > 0.0)[0]:
                if bvh.ray_cast(org[i], dl, 6.0)[0] is not None:
                    sh[i] = 0.0
        hemi = 0.55 + 0.45 * (nrm @ up)
        alb = np.asarray(albedo_fn(o, co, nrm), float)
        lit = alb * (np.asarray(sky)[None, :] * (hemi * ao)[:, None] +
                     key_gain * np.asarray(key)[None, :] * (ndl * sh * (0.4 + 0.6 * ao))[:, None])
        for a in [x for x in me.color_attributes if x.name == LIT_ATTR]:
            me.color_attributes.remove(a)
        attr = me.color_attributes.new(LIT_ATTR, "FLOAT_COLOR", "POINT")
        rgba = np.ones((len(co), 4), np.float32)
        rgba[:, :3] = np.clip(lit, 0.0, 1.0)
        attr.data.foreach_set("color", rgba.ravel())
        idx = me.color_attributes.find(LIT_ATTR)
        me.color_attributes.active_color_index = idx
        me.color_attributes.render_color_index = idx


def unbake_lit(objs):
    for o in objs:
        me = o.data
        for a in [x for x in me.color_attributes if x.name == LIT_ATTR]:
            me.color_attributes.remove(a)
        idx = me.color_attributes.find(C.COLOR_ATTR)
        if idx >= 0:
            me.color_attributes.active_color_index = idx
            me.color_attributes.render_color_index = idx


def setup_lit_workbench(scene, percent=50, bg_hex="#5E655F"):
    C.setup_preview_render(scene, percent=percent, bg_hex=bg_hex, color_type="VERTEX")
    sh = scene.display.shading
    sh.light = "FLAT"
    sh.show_cavity = False
    return scene
