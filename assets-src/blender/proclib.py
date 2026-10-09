"""proclib.py - numpy helpers for the procedural texture and kit builders.

Owned by build_textures.py / build_kit.py (not shared with common.py).
Everything here is deterministic: all randomness comes from numpy Generators
seeded by the caller, and every field is periodic on the unit torus, so any
texture built from these helpers tiles exactly on both axes.

Array convention: 2-D arrays are indexed [row, col] = [v, u] with row 0 at
the BOTTOM of the image (Blender pixel order). So "+rows" is "+V" (image up),
which is what the OpenGL normal-map convention calls +Y.
"""

import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
TEX_DIR = os.path.join(ROOT, "public", "nature", "textures")
MODEL_DIR = os.path.join(ROOT, "public", "nature", "models")
CAPTURE_DIR = os.path.join(ROOT, "docs", "captures", "blender")


# ----------------------------------------------------------------------------
# colour helpers
# ----------------------------------------------------------------------------

def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float32), 0.0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def hexrgb(h):
    """'#796449' -> float sRGB triple (0..1)."""
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lerp(a, b, t):
    """Linear blend; a 2-D weight is broadcast over colour channels."""
    t = np.asarray(t, dtype=np.float32)
    if t.ndim == 2 and (np.ndim(a) in (1, 3) or np.ndim(b) in (1, 3)):
        t = t[..., None]
    return a + (b - a) * t


def normalize01(a, lo_pct=0.5, hi_pct=99.5):
    lo, hi = np.percentile(a, [lo_pct, hi_pct])
    return np.clip((a - lo) / (hi - lo + 1e-12), 0.0, 1.0).astype(np.float32)


def standardize(a):
    a = a - a.mean()
    return (a / (a.std() + 1e-12)).astype(np.float32)


# ----------------------------------------------------------------------------
# periodic spectral noise
# ----------------------------------------------------------------------------

def freq_grid(n):
    """Integer frequencies (cycles per tile) for an rfft2 of an n x n field.
    Returns KU (along columns / u, >= 0) and KV (along rows / v, signed)."""
    kv = np.fft.fftfreq(n) * n
    ku = np.fft.rfftfreq(n) * n
    KU, KV = np.meshgrid(ku, kv)
    return KU.astype(np.float32), KV.astype(np.float32)


def spectral_noise(n, rng, filt):
    """Gaussian noise shaped by an amplitude filter filt(KU, KV) in frequency
    space. Periodic on both axes by construction. Returns zero-mean, unit-std."""
    white = rng.standard_normal((n, n))
    spec = np.fft.rfft2(white)
    KU, KV = freq_grid(n)
    amp = filt(KU, KV)
    amp[0, 0] = 0.0
    out = np.fft.irfft2(spec * amp, s=(n, n))
    return standardize(out)


def fbm_filter(beta, kmin=1.0, kmax=None, aniso=(1.0, 1.0)):
    """Power-law (1/f^beta/2 amplitude) isotropic or anisotropic spectrum."""
    su, sv = aniso

    def f(KU, KV):
        k = np.sqrt((KU * su) ** 2 + (KV * sv) ** 2)
        k = np.maximum(k, 1e-6)
        a = k ** (-beta / 2.0)
        a *= 1.0 - np.exp(-(k / kmin) ** 4)
        if kmax is not None:
            a *= np.exp(-(k / kmax) ** 2)
        return a
    return f


def band_filter(k0, width=0.6):
    """Isotropic log-normal band around k0 cycles/tile."""
    def f(KU, KV):
        k = np.maximum(np.sqrt(KU ** 2 + KV ** 2), 1e-6)
        return np.exp(-(np.log(k / k0) / width) ** 2)
    return f


def fibre_filter(ku, kv, sharp=1.0):
    """Anisotropic spectrum for features elongated along V (rows):
    a broad band peaking at |k_u| = ku across the grain, Gaussian of width kv
    along the grain. ku/kv is roughly the length/width ratio of a feature."""
    def f(KU, KV):
        x = KU / ku
        band = (x ** sharp) * np.exp(-0.5 * sharp * (x ** 2 - 1.0))
        return band * np.exp(-(KV / kv) ** 2)
    return f


def lowpass_filter(ku, kv, kmin=0.7):
    def f(KU, KV):
        k = np.sqrt(KU ** 2 + KV ** 2)
        return np.exp(-(KU / ku) ** 2 - (KV / kv) ** 2) * (1.0 - np.exp(-(k / kmin) ** 4))
    return f


def blur(a, sigma_px):
    """Periodic Gaussian blur through the FFT (exactly tileable)."""
    n_r, n_c = a.shape
    A = np.fft.rfft2(a)
    fv = np.fft.fftfreq(n_r)[:, None]
    fu = np.fft.rfftfreq(n_c)[None, :]
    g = np.exp(-2.0 * (np.pi ** 2) * (sigma_px ** 2) * (fu ** 2 + fv ** 2))
    return np.fft.irfft2(A * g, s=a.shape).astype(np.float32)


# ----------------------------------------------------------------------------
# periodic resampling (domain warping)
# ----------------------------------------------------------------------------

def warp_u(a, du):
    """Sample a(v, u + du) with Catmull-Rom along u, wrapping. du in pixels."""
    n_r, n_c = a.shape
    xs = np.arange(n_c, dtype=np.float32)[None, :] + du
    x0 = np.floor(xs).astype(np.int64)
    t = (xs - x0).astype(np.float32)
    rows = np.arange(n_r)[:, None]
    p0 = a[rows, (x0 - 1) % n_c]
    p1 = a[rows, x0 % n_c]
    p2 = a[rows, (x0 + 1) % n_c]
    p3 = a[rows, (x0 + 2) % n_c]
    t2 = t * t
    t3 = t2 * t
    return (0.5 * ((2.0 * p1) + (-p0 + p2) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                   + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)).astype(np.float32)


def warp_uv(a, du, dv):
    """Bilinear sample a at (u + du, v + dv) (pixels), wrapping on both axes."""
    n_r, n_c = a.shape
    xs = np.arange(n_c, dtype=np.float32)[None, :] + du
    ys = np.arange(n_r, dtype=np.float32)[:, None] + dv
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    fx = (xs - x0).astype(np.float32)
    fy = (ys - y0).astype(np.float32)
    x0m = x0 % n_c
    x1m = (x0 + 1) % n_c
    y0m = y0 % n_r
    y1m = (y0 + 1) % n_r
    return (a[y0m, x0m] * (1 - fx) * (1 - fy) + a[y0m, x1m] * fx * (1 - fy)
            + a[y1m, x0m] * (1 - fx) * fy + a[y1m, x1m] * fx * fy).astype(np.float32)


# ----------------------------------------------------------------------------
# periodic cellular (Worley) noise
# ----------------------------------------------------------------------------

def worley(n, rng, cells, jitter=0.9, du=None, dv=None):
    """Periodic Worley noise with one feature point per cell of a cells x cells
    grid on the unit torus. Optional warp (du, dv in pixels).
    Returns F1, F2 (in cell units) and the id of the nearest point."""
    jit = (rng.random((cells, cells, 2)).astype(np.float32) - 0.5) * jitter + 0.5
    u = (np.arange(n, dtype=np.float32) + 0.5) / n
    U = np.broadcast_to(u[None, :], (n, n)).copy()
    V = np.broadcast_to(u[:, None], (n, n)).copy()
    if du is not None:
        U = U + du / n
    if dv is not None:
        V = V + dv / n
    U = np.mod(U, 1.0) * cells
    V = np.mod(V, 1.0) * cells
    cu = np.floor(U).astype(np.int64)
    cv = np.floor(V).astype(np.int64)
    f1 = np.full((n, n), 1e9, np.float32)
    f2 = np.full((n, n), 1e9, np.float32)
    idn = np.zeros((n, n), np.int64)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            gu = cu + ox
            gv = cv + oy
            gum = gu % cells
            gvm = gv % cells
            pu = gu + jit[gvm, gum, 0]
            pv = gv + jit[gvm, gum, 1]
            d = np.sqrt((U - pu) ** 2 + (V - pv) ** 2).astype(np.float32)
            pid = gvm * cells + gum
            closer = d < f1
            f2 = np.where(closer, f1, np.minimum(f2, d))
            idn = np.where(closer, pid, idn)
            f1 = np.where(closer, d, f1)
    return f1, f2, idn


def worley_strands(n, rng, gu, gv, stretch, jitter=0.85, du=None, dv=None):
    """Periodic Worley noise for strand/plate networks elongated along V.

    gu columns x gv rows of feature points; every column gets its own random
    phase along V so cell ends never line up horizontally. Distances are in
    pixels with V compressed by `stretch` (cells ~stretch x longer along V).
    Returns F1, F2 (px), id1 (nearest point id)."""
    cell_u = n / gu
    cell_v = n / gv
    phase = rng.random(gu).astype(np.float32)
    jit = ((rng.random((gu, gv, 2)) - 0.5) * jitter + 0.5).astype(np.float32)
    X = np.broadcast_to(np.arange(n, dtype=np.float32)[None, :] + 0.5, (n, n)).copy()
    Y = np.broadcast_to(np.arange(n, dtype=np.float32)[:, None] + 0.5, (n, n)).copy()
    if du is not None:
        X += du
    if dv is not None:
        Y += dv
    X = np.mod(X, n)
    Y = np.mod(Y, n)
    ci = np.floor(X / cell_u).astype(np.int64)
    f1 = np.full((n, n), 1e9, np.float32)
    f2 = np.full((n, n), 1e9, np.float32)
    id1 = np.zeros((n, n), np.int64)
    for ox in (-1, 0, 1):
        col = ci + ox
        colm = col % gu
        ph = phase[colm]
        rj = np.floor(Y / cell_v - ph).astype(np.int64)
        for oy in (-1, 0, 1):
            row = rj + oy
            rowm = row % gv
            pu = (col + jit[colm, rowm, 0]) * cell_u
            pv = (row + ph + jit[colm, rowm, 1]) * cell_v
            d = np.sqrt((X - pu) ** 2 + ((Y - pv) / stretch) ** 2).astype(np.float32)
            pid = colm * gv + rowm
            closer = d < f1
            f2 = np.where(closer, f1, np.minimum(f2, d))
            id1 = np.where(closer, pid, id1)
            f1 = np.where(closer, d, f1)
    return f1, f2, id1


def cell_union(n, rng, cells, profile, mode="max", reach=2, jitter=1.0, du=None, dv=None,
               with_offset=False):
    """Continuous union of per-point profiles on a periodic jittered grid.

    profile(d, pid) -> value for distance d (cell units) to point pid
    (profile(d, pid, dx, dy) when with_offset=True). The
    result is the max (or min) over all points within `reach` cells, so it
    is continuous even when each point has its own radius/height (unlike a
    nearest-point lookup, which steps at the Voronoi borders).
    Returns (value, id of the winning point)."""
    jit = ((rng.random((cells, cells, 2)) - 0.5) * jitter + 0.5).astype(np.float32)
    u = (np.arange(n, dtype=np.float32) + 0.5) / n
    U = np.broadcast_to(u[None, :], (n, n)).copy()
    V = np.broadcast_to(u[:, None], (n, n)).copy()
    if du is not None:
        U = U + du / n
    if dv is not None:
        V = V + dv / n
    U = np.mod(U, 1.0) * cells
    V = np.mod(V, 1.0) * cells
    cu = np.floor(U).astype(np.int64)
    cv = np.floor(V).astype(np.int64)
    best = None
    best_id = None
    for oy in range(-reach, reach + 1):
        for ox in range(-reach, reach + 1):
            gu = cu + ox
            gv = cv + oy
            gum = gu % cells
            gvm = gv % cells
            pid = gvm * cells + gum
            dx = (U - (gu + jit[gvm, gum, 0])).astype(np.float32)
            dy = (V - (gv + jit[gvm, gum, 1])).astype(np.float32)
            d = np.sqrt(dx * dx + dy * dy)
            if with_offset:
                val = profile(d, pid, dx, dy).astype(np.float32)
            else:
                val = profile(d, pid).astype(np.float32)
            if best is None:
                best, best_id = val, pid
            else:
                upd = val > best if mode == "max" else val < best
                best = np.where(upd, val, best)
                best_id = np.where(upd, pid, best_id)
    return best, best_id


def abs_grad(a):
    """Max of forward/backward absolute differences along u and v (periodic);
    robust on V-shaped kinks where a central difference would vanish."""
    fu = np.abs(np.roll(a, -1, 1) - a)
    bu = np.abs(a - np.roll(a, 1, 1))
    fv = np.abs(np.roll(a, -1, 0) - a)
    bv = np.abs(a - np.roll(a, 1, 0))
    return np.maximum(fu, bu), np.maximum(fv, bv)


# ----------------------------------------------------------------------------
# height-derived maps
# ----------------------------------------------------------------------------

def normal_from_height(h, px_size, strength=1.0):
    """Tangent-space normal (OpenGL: +X = +U right, +Y = +V up) from a
    periodic height field h (same units as px_size). Returns (n, n, 3) in -1..1."""
    dhdu = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) / (2.0 * px_size)
    dhdv = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) / (2.0 * px_size)
    nx = -dhdu * strength
    ny = -dhdv * strength
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + 1.0)
    return np.stack([nx * inv, ny * inv, inv], axis=-1).astype(np.float32)


def encode_normal(nrm):
    return np.clip(nrm * 0.5 + 0.5, 0.0, 1.0)


def horizon_ao(h, px_size, dirs=8, steps=(1, 2, 3, 5, 8, 12, 17, 24, 34, 48, 64)):
    """Cosine-weighted ambient occlusion of a periodic height field:
    mean over azimuths of cos^2(horizon elevation)."""
    acc = np.zeros_like(h, dtype=np.float32)
    for k in range(dirs):
        ang = 2.0 * np.pi * (k + 0.5) / dirs
        cu, cv = np.cos(ang), np.sin(ang)
        m = np.zeros_like(h, dtype=np.float32)
        seen = set()
        for s in steps:
            ou = int(round(cu * s))
            ov = int(round(cv * s))
            if (ou, ov) in seen or (ou == 0 and ov == 0):
                continue
            seen.add((ou, ov))
            dist = np.hypot(ou, ov) * px_size
            nb = np.roll(h, shift=(-ov, -ou), axis=(0, 1))
            np.maximum(m, (nb - h) / dist, out=m)
        acc += 1.0 / (1.0 + m * m)
    return (acc / dirs).astype(np.float32)


def curvature(h, sigma_px):
    """Positive on convex bumps, negative in concave creases (periodic)."""
    b = blur(h, sigma_px)
    lap = (np.roll(b, 1, 0) + np.roll(b, -1, 0) + np.roll(b, 1, 1) + np.roll(b, -1, 1) - 4 * b)
    return (-lap).astype(np.float32)


# ----------------------------------------------------------------------------
# image IO through Blender (raw byte buffers, no colour transform)
# ----------------------------------------------------------------------------

def save_image(path, rgb, fmt="WEBP", quality=90, colorspace="sRGB"):
    """Write an (h, w, 3) float array (0..1, already in the file's encoding:
    sRGB values for colour maps, raw data for normal/ORM) as an 8-bit image.
    Byte images are written verbatim by Blender: no view transform is applied."""
    import bpy
    rgb = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    rgb = np.round(rgb * 255.0) / 255.0
    hgt, wid = rgb.shape[:2]
    name = os.path.basename(path)
    img = bpy.data.images.new(name, wid, hgt, alpha=False, float_buffer=False)
    img.colorspace_settings.name = colorspace
    px = np.ones((hgt, wid, 4), np.float32)
    px[..., :3] = rgb
    img.pixels.foreach_set(px.ravel())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.filepath_raw = path
    img.file_format = fmt
    if fmt == "WEBP":
        img.save(quality=int(quality))
    else:
        img.save()
    bpy.data.images.remove(img)
    return path


def load_image(path, colorspace="Non-Color"):
    """Read an image back as an (h, w, 3) float array of raw byte values / 255."""
    import bpy
    img = bpy.data.images.load(path, check_existing=False)
    img.colorspace_settings.name = colorspace
    w, h = img.size
    a = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(a)
    bpy.data.images.remove(img)
    return a.reshape(h, w, 4)[..., :3]


def downsample(a, f):
    if f == 1:
        return a
    h, w = a.shape[:2]
    c = a.shape[2] if a.ndim == 3 else 1
    b = a.reshape(h // f, f, w // f, f, c).mean(axis=(1, 3))
    return b if a.ndim == 3 else b[..., 0]


def tile2(a):
    return np.tile(a, (2, 2, 1) if a.ndim == 3 else (2, 2))


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 1e-12 else 10.0 * np.log10(1.0 / mse)


# ----------------------------------------------------------------------------
# tiny bitmap font for labelling preview sheets (no PIL available)
# ----------------------------------------------------------------------------

_FONT = {
    "A": "01110100011000111111100011000110001", "B": "11110100011000111110100011000111110",
    "C": "01110100011000010000100001000101110", "D": "11110100011000110001100011000111110",
    "E": "11111100001000011110100001000011111", "F": "11111100001000011110100001000010000",
    "G": "01110100011000010111100011000101111", "H": "10001100011000111111100011000110001",
    "I": "01110001000010000100001000010001110", "J": "00111000100001000010000101001001100",
    "K": "10001100101010011000101001001010001", "L": "10000100001000010000100001000011111",
    "M": "10001110111010110101100011000110001", "N": "10001100011100110101100111000110001",
    "O": "01110100011000110001100011000101110", "P": "11110100011000111110100001000010000",
    "Q": "01110100011000110001101011001001101", "R": "11110100011000111110101001001010001",
    "S": "01111100001000001110000010000111110", "T": "11111001000010000100001000010000100",
    "U": "10001100011000110001100011000101110", "V": "10001100011000110001100010101000100",
    "W": "10001100011000110101101011010101010", "X": "10001100010101000100010101000110001",
    "Y": "10001100010101000100001000010000100", "Z": "11111000010001000100010001000011111",
    "0": "01110100011001110101110011000101110", "1": "00100011000010000100001000010001110",
    "2": "01110100010000100010001000100011111", "3": "11111000100010000010000011000101110",
    "4": "00010001100101010010111110001000010", "5": "11111100001111000001000011000101110",
    "6": "00110010001000011110100011000101110", "7": "11111000010001000100010000100001000",
    "8": "01110100011000101110100011000101110", "9": "01110100011000101111000010001001100",
    " ": "00000000000000000000000000000000000", "_": "00000000000000000000000000000011111",
    "-": "00000000000000011111000000000000000", ".": "00000000000000000000000000110001100",
    ":": "00000011000110000000011000110000000", "/": "00001000010001000100010001000010000",
    "(": "00010001000100001000010000010000010", ")": "01000001000001000010000100010001000",
    "X": "10001100010101000100010101000110001", "=": "00000000001111100000111110000000000",
    "+": "00000001000010011111001000010000000", ",": "00000000000000000000001100010001000",
}


def draw_text(img, text, x, y, scale=2, color=(1.0, 1.0, 1.0)):
    """Draw text into a top-down (row 0 = top) float RGB image in place."""
    col = np.asarray(color, np.float32)
    cx = x
    for ch in text.upper():
        bits = _FONT.get(ch, _FONT[" "])
        for r in range(7):
            for c in range(5):
                if bits[r * 5 + c] == "1":
                    y0 = y + r * scale
                    x0 = cx + c * scale
                    if 0 <= y0 < img.shape[0] - scale and 0 <= x0 < img.shape[1] - scale:
                        img[y0:y0 + scale, x0:x0 + scale, :3] = col
        cx += 6 * scale
    return img
