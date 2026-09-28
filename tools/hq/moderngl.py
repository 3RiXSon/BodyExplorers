"""Capture backend: pretends to be `moderngl` so render.py runs unmodified,
but instead of rasterizing it records the scene as a list of instances
(transform + colour + primitive) for the ray tracer in engine.py.

render.py's whole visual language is affine-transformed unit spheres plus the
biconcave red-blood-cell mesh, so a captured draw call is all the information
the path tracer needs — and it can substitute analytic, infinitely smooth
geometry for the tessellated meshes the GL path used.
"""
import numpy as np

DEPTH_TEST = 1
BLEND = 2
CULL_FACE = 3

SPHERE = 0
MESH = 1


class Error(Exception):
    pass


class Buffer:
    def __init__(self, data):
        self.data = bytes(data)


class Uniform:
    def __init__(self, name, kind):
        self.name, self.kind = name, kind
        self._value = None

    def write(self, data):
        raw = np.frombuffer(bytes(data), dtype="f4")
        self._value = raw.reshape(4, 4).T.copy() if self.kind == "mat4" else raw.copy()

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = np.array(v, dtype="f4")


class Program:
    def __init__(self, vertex_shader="", fragment_shader=""):
        import re
        kinds = {}
        for k, n in re.findall(r"uniform\s+(\w+)\s+(\w+)", vertex_shader + fragment_shader):
            kinds[n] = "mat4" if k == "mat4" else "vec3"
        for required in ("model", "vp", "color", "eye"):
            kinds.setdefault(required, "mat4" if required in ("model", "vp") else "vec3")
        self._u = {n: Uniform(n, k) for n, k in kinds.items()}

    def __getitem__(self, name):
        return self._u[name]


class VertexArray:
    def __init__(self, ctx, program, content, index_buffer):
        self.ctx, self.program = ctx, program
        buf, fmt, *_ = content[0]
        stride = sum(int(tok[0]) for tok in fmt.split())
        verts = np.frombuffer(buf.data, dtype="f4").reshape(-1, stride)
        self.pos = np.ascontiguousarray(verts[:, 0:3], dtype="f4")
        self.nrm = np.ascontiguousarray(verts[:, 3:6], dtype="f4")
        self.idx = np.frombuffer(index_buffer.data, dtype="i4").copy()
        # A unit sphere becomes an analytic primitive; anything else stays a mesh.
        radii = np.linalg.norm(self.pos.astype("f8"), axis=1)
        self.kind = SPHERE if np.allclose(radii, 1.0, atol=1e-3) else MESH

    def render(self, *a, **kw):
        p = self.program
        self.ctx.draws.append((p["model"].value.copy(), tuple(p["color"].value), self.kind))
        self.ctx.vp = p["vp"].value
        self.ctx.eye = p["eye"].value


class Framebuffer:
    def __init__(self, ctx, size, components=3):
        self.ctx = ctx
        self.width, self.height = size
        self.components = components

    def use(self):
        self.ctx.fbo = self

    def clear(self, r=0.0, g=0.0, b=0.0, a=1.0):
        self.ctx.clear(r, g, b, a)

    def read(self, *a, **kw):
        # The HQ pipeline never reads through this path; render.frame() is bypassed.
        return b"\x00" * (self.width * self.height * self.components)


class Context:
    def __init__(self):
        self.fbo = None
        self.draws = []
        self.vp = None
        self.eye = None
        self.clear_color = (0.0, 0.0, 0.0)

    def simple_framebuffer(self, size, components=3, **kw):
        return Framebuffer(self, size, components)

    def enable(self, flag):
        pass

    def disable(self, flag):
        pass

    def program(self, vertex_shader="", fragment_shader="", **kw):
        return Program(vertex_shader, fragment_shader)

    def buffer(self, data=None, reserve=0, **kw):
        return Buffer(data if data is not None else b"")

    def vertex_array(self, program, content, index_buffer=None, **kw):
        return VertexArray(self, program, content, index_buffer)

    def clear(self, r=0.0, g=0.0, b=0.0, a=1.0, *args, **kw):
        self.clear_color = (r, g, b)
        self.draws = []          # a clear starts a new frame


CONTEXT = None


def create_standalone_context(*a, **kw):
    global CONTEXT
    CONTEXT = Context()
    return CONTEXT


create_context = create_standalone_context
__version__ = "hq-capture-1.0"
