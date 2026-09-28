"""Drop-in software replacement for the small slice of the `moderngl` API that
Body-Explorers-Source-Package/render.py uses.

The sandbox that builds the episode has no GPU and no EGL/OpenGL driver, so the
standalone GL context that render.py expects cannot be created. This module
emulates that context on the CPU: vertex transforms are done with numpy and the
triangles are rasterized by the companion C rasterizer (softraster.c), which
implements the very same lighting math as the GLSL program in render.py.

Put this directory first on PYTHONPATH and render.py runs unmodified.
"""
import ctypes
import os
import re
import subprocess
import sys

import numpy as np

DEPTH_TEST = 1
BLEND = 2
CULL_FACE = 3

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_HERE, "libsoftraster.so")
_SRC_PATH = os.path.join(_HERE, "softraster.c")


def _load_lib():
    if (not os.path.exists(_LIB_PATH)) or (
        os.path.getmtime(_SRC_PATH) > os.path.getmtime(_LIB_PATH)
    ):
        subprocess.run(
            ["gcc", "-O3", "-ffast-math", "-fPIC", "-shared",
             "-o", _LIB_PATH, _SRC_PATH, "-lm"],
            check=True,
        )
    lib = ctypes.CDLL(_LIB_PATH)
    f32 = ctypes.POINTER(ctypes.c_float)
    lib.draw_indexed.argtypes = [
        f32, f32, f32, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        f32, f32, ctypes.POINTER(ctypes.c_ubyte), f32,
        ctypes.c_int, ctypes.c_int,
    ]
    lib.draw_indexed.restype = None
    lib.clear_buffers.argtypes = [
        ctypes.POINTER(ctypes.c_ubyte), f32, ctypes.c_int, ctypes.c_int,
        ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_ubyte,
    ]
    lib.clear_buffers.restype = None
    return lib


_LIB = _load_lib()


def _ptr(arr, ctype):
    return arr.ctypes.data_as(ctypes.POINTER(ctype))


class Error(Exception):
    pass


class Buffer:
    def __init__(self, data):
        self.data = bytes(data)


class Uniform:
    def __init__(self, name, kind):
        self.name = name
        self.kind = kind          # 'mat4' or 'vec3'
        self._value = None

    def write(self, data):
        raw = np.frombuffer(bytes(data), dtype="f4")
        if self.kind == "mat4":
            # GL consumes column-major data; store the row-major math matrix.
            self._value = raw.reshape(4, 4).T.copy()
        else:
            self._value = raw.copy()

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = np.array(v, dtype="f4")


class Program:
    """Recognises the single shader pair used by render.py."""

    def __init__(self, vertex_shader, fragment_shader):
        self.vertex_shader = vertex_shader
        self.fragment_shader = fragment_shader
        names = set(re.findall(r"uniform\s+\w+\s+(\w+)", vertex_shader + fragment_shader))
        kinds = dict(re.findall(r"uniform\s+(\w+)\s+(\w+)", vertex_shader + fragment_shader)[::1])
        kind_by_name = {}
        for k, n in re.findall(r"uniform\s+(\w+)\s+(\w+)", vertex_shader + fragment_shader):
            kind_by_name[n] = "mat4" if k == "mat4" else "vec3"
        self._uniforms = {n: Uniform(n, kind_by_name.get(n, "vec3")) for n in names}
        for required in ("model", "vp", "color", "eye"):
            if required not in self._uniforms:
                raise Error("software renderer expects uniform %r" % required)

    def __getitem__(self, name):
        return self._uniforms[name]

    def get(self, name, default=None):
        return self._uniforms.get(name, default)


class VertexArray:
    def __init__(self, ctx, program, content, index_buffer):
        self.ctx = ctx
        self.program = program
        buf, fmt, *_attrs = content[0]
        stride = sum(int(tok[0]) for tok in fmt.split())
        verts = np.frombuffer(buf.data, dtype="f4").reshape(-1, stride)
        self.pos = np.ascontiguousarray(verts[:, 0:3], dtype="f4")
        self.nrm = np.ascontiguousarray(verts[:, 3:6], dtype="f4")
        self.idx = np.ascontiguousarray(
            np.frombuffer(index_buffer.data, dtype="i4"), dtype="i4"
        )
        # Scratch buffers reused across draws to avoid per-frame allocation.
        n = len(self.pos)
        self._world = np.empty((n, 3), dtype="f4")
        self._clip = np.empty((n, 4), dtype="f4")
        self._nout = np.empty((n, 3), dtype="f4")

    def render(self, *args, **kwargs):
        prog = self.program
        model = prog["model"].value
        vp = prog["vp"].value
        color = np.ascontiguousarray(prog["color"].value, dtype="f4")
        eye = np.ascontiguousarray(prog["eye"].value, dtype="f4")
        if model is None or vp is None:
            raise Error("model/vp uniforms not set")

        m3 = model[:3, :3]
        np.dot(self.pos, m3.T, out=self._world)
        self._world += model[:3, 3]

        # mat3(transpose(inverse(model))) — same as the vertex shader.
        try:
            nmat = np.linalg.inv(m3.astype("f8")).T.astype("f4")
        except np.linalg.LinAlgError:
            nmat = m3
        np.dot(self.nrm, nmat.T, out=self._nout)

        # clip = vp * vec4(world, 1)
        clip = self._clip
        np.dot(self._world, np.ascontiguousarray(vp[:, :3].T), out=clip)
        clip += vp[:, 3]

        fbo = self.ctx.fbo
        _LIB.draw_indexed(
            _ptr(clip, ctypes.c_float),
            _ptr(self._world, ctypes.c_float),
            _ptr(self._nout, ctypes.c_float),
            _ptr(self.idx, ctypes.c_int),
            ctypes.c_int(len(self.idx) // 3),
            _ptr(color, ctypes.c_float),
            _ptr(eye, ctypes.c_float),
            _ptr(fbo.color, ctypes.c_ubyte),
            _ptr(fbo.depth, ctypes.c_float),
            ctypes.c_int(fbo.width),
            ctypes.c_int(fbo.height),
        )


class Framebuffer:
    def __init__(self, ctx, size, components=3):
        self.ctx = ctx
        self.width, self.height = size
        self.components = components
        self.color = np.zeros((self.height, self.width, components), dtype=np.uint8)
        self.depth = np.ones((self.height, self.width), dtype="f4")

    def use(self):
        self.ctx.fbo = self

    def clear(self, r=0.0, g=0.0, b=0.0, a=1.0):
        _LIB.clear_buffers(
            _ptr(self.color, ctypes.c_ubyte),
            _ptr(self.depth, ctypes.c_float),
            ctypes.c_int(self.width),
            ctypes.c_int(self.height),
            ctypes.c_ubyte(int(min(max(r, 0.0), 1.0) * 255 + 0.5)),
            ctypes.c_ubyte(int(min(max(g, 0.0), 1.0) * 255 + 0.5)),
            ctypes.c_ubyte(int(min(max(b, 0.0), 1.0) * 255 + 0.5)),
        )

    def read(self, viewport=None, components=3, attachment=0, alignment=1, dtype="f1"):
        # glReadPixels order: bottom-up rows, which render.py flips afterwards.
        return self.color[:, :, :components].tobytes()


class Context:
    def __init__(self):
        self.fbo = None
        self.enabled = set()
        self.version_code = 330

    def simple_framebuffer(self, size, components=3, **kwargs):
        return Framebuffer(self, size, components)

    def framebuffer(self, *args, **kwargs):
        raise Error("not supported by the software renderer")

    def enable(self, flag):
        self.enabled.add(flag)

    def disable(self, flag):
        self.enabled.discard(flag)

    def program(self, vertex_shader, fragment_shader, **kwargs):
        return Program(vertex_shader, fragment_shader)

    def buffer(self, data=None, reserve=0, **kwargs):
        return Buffer(data if data is not None else b"\x00" * reserve)

    def vertex_array(self, program, content, index_buffer=None, **kwargs):
        return VertexArray(self, program, content, index_buffer)

    def clear(self, r=0.0, g=0.0, b=0.0, a=1.0, *args, **kwargs):
        (self.fbo or self.simple_framebuffer((1, 1))).clear(r, g, b, a)


def create_standalone_context(*args, **kwargs):
    return Context()


def create_context(*args, **kwargs):
    return Context()


__version__ = "softgl-1.0"
