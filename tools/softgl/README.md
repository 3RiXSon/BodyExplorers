# softgl — CPU fallback for the episode renderer

`render.py` expects a standalone OpenGL 3.3 context via EGL. On machines with
no GPU/EGL driver (CI containers, sandboxes) that context cannot be created.

This directory provides a drop-in `moderngl` module that emulates the small
slice of the API `render.py` uses. Vertex transforms run in numpy; triangles are
rasterized by `softraster.c`, which reimplements the exact GLSL lighting model
from `render.py` (lambert + rim + specular, gamma 0.9), with a depth buffer,
near-plane clipping, perspective-correct interpolation and no face culling.
The shared library is compiled automatically with gcc on first import.

Render the episode without a GPU:

    PYTHONPATH=tools/softgl python render.py     # silent.mp4 + captions
    PYTHONPATH=tools/softgl python finish.py     # narration + music -> final MP4

Output is identical in format to the GPU path: 1280x720, 24 fps, 5:00, H.264/AAC.
If a real GL driver is present, simply omit PYTHONPATH and the stock moderngl
package is used instead.
