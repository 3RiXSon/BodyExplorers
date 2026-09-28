# HQ renderer — path-traced "Pixar style" pass

A CPU path tracer that re-renders the existing episodes with production-style
lighting. It reuses the scene choreography in
`Body-Explorers-Source-Package/render.py` **unchanged**: that module is imported
against the capture backend in `moderngl.py`, so each draw call becomes a
ray-traced instance instead of a rasterized triangle batch.

Why a bespoke tracer: the sandbox has no GPU, no `/dev/dri`, blocked distro
mirrors and no reachable cloud API, so Blender/Cycles, EEVEE and OpenGL are all
off the table. Everything here builds with plain `gcc`.

## What it adds over the raster path
| | raster (`tools/softgl`) | path tracer (`tools/hq`) |
|---|---|---|
| Geometry | tessellated meshes | analytic ellipsoids — perfectly smooth at any resolution, plus a 43k-triangle red blood cell |
| Lighting | one hard directional light | 4 sphere **area** lights (key / fill / rim / bounce) + gradient sky IBL |
| Shadows | none | ray-traced soft shadows with real penumbrae |
| Indirect | none | one diffuse GI bounce: ambient occlusion and colour bleed |
| Materials | lambert + rim + phong | GGX specular, fresnel, metallic, roughness, wrap-scatter subsurface, emissive |
| Camera | pinhole | thin lens with depth of field, stochastic AA |
| Grade | gamma 0.9 | bloom, ACES tonemap, saturation, S-curve, vignette, grain |
| Denoise | — | multithreaded a-trous filter guided by albedo/normal/depth |

Sampling is adaptive: pixels stop early once their estimate converges, which
roughly halves the cost of the mostly empty backdrops.

## Use
    bash tools/setup_env.sh                      # numpy, Pillow, ffmpeg (no root)
    python tools/hq/render_hq.py stills          # preview frames
    python tools/hq/render_hq.py all --preset good   # every scene -> mp4
    python tools/hq/finish_hq.py                 # mux narration + score

Presets: `preview` 960x540, `good` 720p, `high` 1080p, `ultra`, `showcase` 1440p.
`tools/hq/run_full.sh` renders the whole episode resumably (scene by scene,
skipping finished ones) and concatenates the picture cut.

Rough cost on 2 CPU cores: ~2.2 s/frame at 720p/14spp, ~6 s/frame at 1080p.
