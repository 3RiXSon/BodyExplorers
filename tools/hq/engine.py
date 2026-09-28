"""Python front end for the hq.c path tracer: material lookup, cinematic
lighting rig, high-resolution geometry and the film post-processing chain
(bloom -> ACES tonemap -> vignette -> grain).
"""
import ctypes
import math
import os
import subprocess

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "hq.c")
_LIB = os.path.join(_HERE, "libhq.so")


def _load():
    if (not os.path.exists(_LIB)) or os.path.getmtime(_SRC) > os.path.getmtime(_LIB):
        subprocess.run(
            ["gcc", "-O3", "-march=native", "-ffast-math", "-fPIC", "-shared",
             "-o", _LIB, _SRC, "-lm", "-lpthread"],
            check=True,
        )
    lib = ctypes.CDLL(_LIB)
    F = ctypes.POINTER(ctypes.c_float)
    I = ctypes.POINTER(ctypes.c_int)
    lib.hq_set_mesh.argtypes = [F, F, ctypes.c_int, I, ctypes.c_int]
    lib.hq_set_instances.argtypes = [F, I, ctypes.c_int]
    lib.hq_set_lights.argtypes = [F, ctypes.c_int]
    lib.hq_set_sky.argtypes = [F, F, F, ctypes.c_float]
    lib.hq_render.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, F,
        ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_uint64,
        F, F, F, F,
    ]
    lib.hq_denoise.argtypes = [
        F, F, F, F, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ]
    lib.hq_denoise_mt.argtypes = [
        F, F, F, F, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int,
    ]
    return lib


LIB = _load()


def _p(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _pi(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


# --------------------------------------------------------------- materials
# (roughness, metallic, emissive, subsurface) keyed by the palette render.py uses.
MATERIALS = {
    (0.07, 0.76, 0.76): (0.20, 0.00, 0.00, 0.10),   # TEAL   Pip's shell
    (1.00, 0.66, 0.16): (0.18, 0.85, 0.00, 0.00),   # GOLD   helmet ring, trim
    (0.97, 0.16, 0.30): (0.33, 0.00, 0.00, 0.38),   # RED    Ruby
    (0.98, 0.40, 0.49): (0.42, 0.00, 0.00, 0.55),   # PINK   lungs
    (0.96, 0.99, 1.00): (0.12, 0.00, 0.00, 0.15),   # WHITE  sclera
    (0.035, 0.095, 0.17): (0.08, 0.00, 0.00, 0.00),  # NAVY  visor, pupils
    (0.54, 0.32, 0.79): (0.35, 0.00, 0.00, 0.35),   # PURPLE stomach
    (0.45, 1.00, 0.94): (0.25, 0.00, 0.60, 0.00),   # Pip's ear lights
    (0.10, 0.29, 0.36): (0.40, 0.00, 0.30, 0.00),   # background motes
    (0.04, 0.16, 0.23): (0.34, 0.00, 0.00, 0.00),   # stage floor
    (0.66, 0.39, 0.24): (0.52, 0.00, 0.00, 0.55),   # Maya's skin
    (0.15, 0.09, 0.075): (0.40, 0.00, 0.00, 0.05),  # hair
    (0.90, 0.71, 0.59): (0.50, 0.00, 0.00, 0.55),   # windpipe
    (1.00, 0.85, 0.61): (0.50, 0.00, 0.00, 0.60),   # airways
    (0.93, 0.66, 0.56): (0.50, 0.00, 0.00, 0.60),   # airway stem
    (0.97, 0.59, 0.47): (0.46, 0.00, 0.00, 0.75),   # alveoli
    (0.36, 0.12, 0.22): (0.45, 0.00, 0.00, 0.30),   # vessel rings
    (0.48, 0.14, 0.24): (0.45, 0.00, 0.00, 0.35),   # capillary walls
    (0.58, 0.16, 0.30): (0.40, 0.00, 0.00, 0.40),   # vena cava
    (0.66, 0.13, 0.26): (0.35, 0.00, 0.00, 0.55),   # other red cells
    (0.26, 0.49, 0.74): (0.30, 0.00, 0.00, 0.20),   # mitochondrion
    (0.20, 0.43, 0.47): (0.45, 0.00, 0.00, 0.00),   # ground dashes
    (0.52, 0.72, 0.88): (0.25, 0.00, 0.55, 0.00),   # CO2 dots
    (0.53, 0.73, 0.92): (0.25, 0.00, 0.55, 0.00),   # exhaled dots
    (0.85, 0.19, 0.30): (0.42, 0.00, 0.00, 0.60),   # muscle fibre
    (1.00, 0.59, 0.55): (0.45, 0.00, 0.00, 0.60),   # muscle highlight
}
_MAT_KEYS = np.array(list(MATERIALS.keys()), dtype="f4")
_MAT_VALS = np.array(list(MATERIALS.values()), dtype="f4")


def material_for(color, max_scale):
    """Nearest palette match, with size-aware tweaks for the glowing molecules."""
    c = np.array(color, dtype="f4")
    d = np.linalg.norm(_MAT_KEYS - c, axis=1)
    i = int(np.argmin(d))
    rough, metal, emis, sss = _MAT_VALS[i]
    if d[i] > 0.14:                       # unknown colour: sensible default
        lum = float(c.mean())
        rough, metal, emis, sss = 0.38, 0.0, 0.0, 0.35 if lum > 0.3 else 0.0
    # Small gold spheres are the oxygen molecules: make them glow instead of
    # reading as metal balls. Large gold pieces stay as polished trim.
    if d[i] < 0.06 and tuple(np.round(_MAT_KEYS[i], 3)) == (1.0, 0.66, 0.16) and max_scale < 0.26:
        rough, metal, emis, sss = 0.30, 0.0, 0.70, 0.0
    return float(rough), float(metal), float(emis), float(sss)


# ---------------------------------------------------------------- geometry
def rbc_mesh(n=120, m=180):
    """High-resolution biconcave red blood cell — same surface of revolution as
    render.mesh_rbc, but finely tessellated since the ray tracer only pays for
    it once (shared, instanced, BVH-accelerated)."""
    verts, normals, idx = [], [], []
    for i in range(n + 1):
        a = math.pi * i / n
        r = math.sin(a)
        z = math.cos(a) * (0.18 + 0.58 * r * r)
        da = 1e-4
        rr = math.sin(a + da)
        zz = math.cos(a + da) * (0.18 + 0.58 * rr * rr)
        dr, dz = (rr - r) / da, (zz - z) / da
        for j in range(m + 1):
            b = j * 2 * math.pi / m
            nv = np.array([-dz * math.cos(b), -dz * math.sin(b), dr])
            nl = np.linalg.norm(nv)
            nv = nv / nl if nl > 1e-9 else np.array([0.0, 0.0, 1.0])
            verts.append([r * math.cos(b), r * math.sin(b), z])
            normals.append(nv)
    for i in range(n):
        for j in range(m):
            k = i * (m + 1) + j
            idx += [k, k + m + 1, k + 1, k + 1, k + m + 1, k + m + 2]
    return (np.array(verts, "f4"), np.array(normals, "f4"),
            np.array(idx, "i4").reshape(-1, 3))


_MESH_SET = False


def ensure_mesh():
    global _MESH_SET, _MESH_KEEP
    if _MESH_SET:
        return
    v, n, f = rbc_mesh()
    _MESH_KEEP = (np.ascontiguousarray(v), np.ascontiguousarray(n), np.ascontiguousarray(f))
    LIB.hq_set_mesh(_p(_MESH_KEEP[0]), _p(_MESH_KEEP[1]), len(v),
                    _pi(_MESH_KEEP[2]), len(f))
    _MESH_SET = True


# ---------------------------------------------------------------- lighting
def lighting_rig():
    """Warm key, cool fill, cyan rim and a soft bounce — a classic animation
    three-point rig, all as sphere area lights so shadows have real penumbrae."""
    lights = [
        # x, y, z, radius, r, g, b, intensity
        (-5.0, 8.0, 9.5, 3.4, 1.00, 0.94, 0.86, 7.2),    # key
        (7.5, 0.5, 7.0, 4.5, 0.58, 0.76, 1.00, 2.7),     # fill
        (0.0, 4.5, -9.0, 3.6, 0.45, 0.95, 0.95, 3.6),    # rim / backlight
        (0.0, -7.0, 4.0, 5.0, 0.24, 0.50, 0.62, 1.4),    # bounce
    ]
    return np.array(lights, dtype="f4").reshape(-1)


def _lin(c):
    """Authoring colours are display-referred; the tracer works in linear light."""
    return np.power(np.array(c, dtype="f8"), 2.2).astype("f4")


# Deep navy studio environment matching the series' background plate.
SKY_TOP = _lin([0.030, 0.070, 0.120])
SKY_HOR = _lin([0.048, 0.105, 0.155])
SKY_BOT = _lin([0.010, 0.026, 0.045])


def setup_environment(sky_gain=1.0):
    ensure_mesh()
    lights = lighting_rig()
    LIB.hq_set_lights(_p(lights), len(lights) // 8)
    LIB.hq_set_sky(_p(SKY_TOP), _p(SKY_HOR), _p(SKY_BOT), ctypes.c_float(sky_gain))


# ------------------------------------------------------------------ camera
def camera_from_vp(vp, eye):
    """Recover eye/forward/right/up/fov from the projection*view matrix that
    render.camera() produced, so the ray tracer frames shots identically."""
    inv = np.linalg.inv(vp.astype("f8"))

    def unproject(x, y):
        p = inv @ np.array([x, y, 0.0, 1.0])
        return p[:3] / p[3]

    eye = np.array(eye, dtype="f8")
    c = unproject(0.0, 0.0)
    rx = unproject(1.0, 0.0)
    ry = unproject(0.0, 1.0)
    fwd_vec = c - eye
    dist = np.linalg.norm(fwd_vec)
    fwd = fwd_vec / dist
    right_vec = rx - c
    up_vec = ry - c
    tan_half = np.linalg.norm(up_vec) / dist
    aspect = np.linalg.norm(right_vec) / np.linalg.norm(up_vec)
    right = right_vec / np.linalg.norm(right_vec)
    up = up_vec / np.linalg.norm(up_vec)
    cam = np.zeros(14, dtype="f4")
    cam[0:3] = eye
    cam[3:6] = fwd
    cam[6:9] = right
    cam[9:12] = up
    cam[12] = tan_half
    cam[13] = aspect
    return cam


# ------------------------------------------------------------------- scene
def upload_draws(draws):
    n = len(draws)
    data = np.zeros((n, 39), dtype="f4")
    mesh_ids = np.zeros(n, dtype="i4")
    for i, (model, color, kind) in enumerate(draws):
        m = np.array(model, dtype="f8")
        data[i, 0:16] = m.reshape(-1)
        data[i, 16:32] = np.linalg.inv(m).reshape(-1)
        data[i, 32:35] = np.power(np.clip(np.array(color, 'f8'), 0.0, 1.0), 2.2)
        scale = float(np.abs(np.linalg.eigvals(m[:3, :3])).max().real) if n else 1.0
        scale = max(scale, float(np.linalg.norm(m[:3, :3], axis=0).max()))
        rough, metal, emis, sss = material_for(color, scale)
        data[i, 35:39] = (rough, metal, emis, sss)
        mesh_ids[i] = kind
    data = np.ascontiguousarray(data)
    mesh_ids = np.ascontiguousarray(mesh_ids)
    LIB.hq_set_instances(_p(data), _pi(mesh_ids), n)
    return data, mesh_ids


def render_hdr(draws, vp, eye, W, H, spp=9, bounces=1, aperture=0.055,
               focus=None, threads=None, denoise=True, seed=1):
    keep = upload_draws(draws)                 # keep alive during the call
    cam = camera_from_vp(np.array(vp), np.array(eye))
    if focus is None:
        focus = float(np.linalg.norm(np.array(eye, dtype="f8")))
    if threads is None:
        threads = max(1, os.cpu_count() or 1)

    color = np.zeros((H, W, 3), dtype="f4")
    albedo = np.zeros((H, W, 3), dtype="f4")
    normal = np.zeros((H, W, 3), dtype="f4")
    depth = np.zeros((H, W), dtype="f4")

    LIB.hq_render(W, H, spp, bounces, _p(cam), ctypes.c_float(aperture),
                  ctypes.c_float(focus), threads, ctypes.c_uint64(seed),
                  _p(color), _p(albedo), _p(normal), _p(depth))
    if denoise:
        LIB.hq_denoise_mt(_p(color), _p(albedo), _p(normal), _p(depth), W, H,
                          4, ctypes.c_float(64.0), ctypes.c_float(0.30),
                          ctypes.c_float(1.10), threads)
    del keep
    return color


# --------------------------------------------------------------------- post
def _box_blur(img, r):
    """Separable box blur via summed-area rows/cols (float, edge clamped)."""
    if r < 1:
        return img
    out = img
    for axis in (0, 1):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (r, r)
        p = np.pad(out, pad, mode="edge")
        c = np.cumsum(p, axis=axis, dtype="f4")
        zero = np.zeros_like(np.take(c, [0], axis=axis))
        c = np.concatenate([zero, c], axis=axis)
        n = out.shape[axis]
        hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
        lo = np.take(c, np.arange(0, n), axis=axis)
        out = (hi - lo) / float(2 * r + 1)
    return out.astype("f4")


def _blur(img, radius, downsample=4):
    """Approximate a wide gaussian cheaply: shrink, three box blurs, regrow."""
    from PIL import Image
    h, w = img.shape[:2]
    ds = max(1, int(downsample))
    small = img[::ds, ::ds].copy()
    r = max(1, int(round(radius / ds)))
    for _ in range(3):
        small = _box_blur(small, r)
    if ds == 1:
        return small
    out = np.empty_like(img)
    for k in range(3):
        im = Image.fromarray(small[:, :, k], mode="F").resize((w, h), Image.BILINEAR)
        out[:, :, k] = np.asarray(im, dtype="f4")
    return out


def aces(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def post_process(hdr, exposure=1.14, bloom=0.48, vignette=0.32, grain=0.0022,
                 saturation=1.24, contrast=1.07, seed=7):
    """Filmic finish tuned for a punchy, toy-bright animation look: highlight
    bloom, ACES tonemap, saturation lift, an S-curve for contrast, vignette
    and a touch of grain so flat areas do not band."""
    img = hdr * exposure
    if bloom > 0:
        bright = np.maximum(img - 0.85, 0.0)
        glow = _blur(bright, max(2.0, img.shape[1] / 220.0))
        glow += _blur(bright, max(6.0, img.shape[1] / 70.0)) * 0.7
        img = img + glow * bloom

    ldr = aces(img)
    ldr = np.clip(ldr, 0.0, 1.0) ** (1.0 / 2.2)

    if saturation != 1.0:
        lum = (ldr * np.array([0.2126, 0.7152, 0.0722], dtype="f4")).sum(axis=2, keepdims=True)
        ldr = np.clip(lum + (ldr - lum) * saturation, 0.0, 1.0)

    if contrast != 1.0:
        # pivot around mid grey, then a gentle smoothstep for filmic shoulders
        ldr = np.clip((ldr - 0.5) * contrast + 0.5, 0.0, 1.0)
        ldr = ldr * 0.75 + (ldr * ldr * (3.0 - 2.0 * ldr)) * 0.25

    if vignette > 0:

        h, w = ldr.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        nx = (xx / (w - 1) - 0.5) * 2.0
        ny = (yy / (h - 1) - 0.5) * 2.0
        r = np.sqrt(nx * nx + ny * ny) / math.sqrt(2.0)
        ldr *= (1.0 - vignette * (r ** 2.2))[:, :, None]

    if grain > 0:
        rng = np.random.default_rng(seed)
        ldr += rng.normal(0.0, grain, ldr.shape).astype("f4")

    return np.clip(ldr * 255.0 + 0.5, 0, 255).astype(np.uint8)
