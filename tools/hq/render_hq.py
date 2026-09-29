#!/usr/bin/env python3
"""High-quality renderer for Body Explorers episodes.

Reuses the scene choreography in Body-Explorers-Source-Package/render.py
unchanged: that module is imported against the capture backend in this
directory, so every draw call becomes a ray-traced instance instead of a
rasterized triangle batch.

Usage
  python tools/hq/render_hq.py stills                 # preview frames
  python tools/hq/render_hq.py scene 3                # one 15 s shot -> mp4
  python tools/hq/render_hq.py all                    # whole episode -> mp4
Options: --width --height --fps --spp --preset --out --workers
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SRC = REPO / "Body-Explorers-Source-Package"

sys.path.insert(0, str(HERE))       # capture backend shadows the real moderngl
sys.path.insert(0, str(SRC))

import engine                        # noqa: E402
import compose                       # noqa: E402

PRESETS = {
    # name:      (width, height, spp, bounces, aperture, denoise)
    "preview":  (960, 540, 4, 1, 0.045, True),
    "good":     (1280, 720, 6, 1, 0.050, True),
    "high":     (1920, 1080, 8, 1, 0.055, True),
    "ultra":    (1920, 1080, 24, 2, 0.055, True),
    "showcase": (2560, 1440, 32, 2, 0.060, True),
}


def load_render():
    import render
    return render


def ffmpeg():
    return os.environ.get("FFMPEG", "ffmpeg")


def render_frame(render_mod, i, t, W, H, spp, bounces, aperture, denoise, exposure):
    """Ray trace one frame and composite the UI layer on top."""
    render_mod.scene(i, t)                       # fills the capture context
    ctx = render_mod.ctx
    hdr = engine.render_hdr(ctx.draws, ctx.vp, ctx.eye, W, H, spp=spp,
                            bounces=bounces, aperture=aperture, denoise=denoise)
    rgb = engine.post_process(hdr, exposure=exposure)
    im = Image.fromarray(rgb, "RGB").convert("RGBA")
    im.alpha_composite(compose.overlay(render_mod.S, render_mod.CAP, i, t, W, H))
    return im.convert("RGB")


def encode_scene(render_mod, i, args, out_path):
    W, H = args.width, args.height
    tmp_path = Path(str(out_path) + ".part")
    cmd = [ffmpeg(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-", "-an",
           "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf),
           "-pix_fmt", "yuv420p", str(tmp_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    nframes = int(args.fps * 15)
    t0 = time.time()
    for f in range(nframes):
        t = f / args.fps
        im = render_frame(render_mod, i, t, W, H, args.spp, args.bounces,
                          args.aperture, args.denoise, args.exposure)
        proc.stdin.write(im.tobytes())
        if f % 12 == 0:
            done = f + 1
            rate = (time.time() - t0) / done
            print(f"  scene {i + 1:02}: frame {done}/{nframes} "
                  f"({rate:.2f}s/frame, eta {(nframes - done) * rate / 60:.1f} min)",
                  flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit(f"ffmpeg failed on scene {i}")
    tmp_path.replace(out_path)
    print(f"scene {i + 1:02} done in {(time.time() - t0) / 60:.1f} min -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["stills", "scene", "all"])
    ap.add_argument("index", nargs="?", type=int, default=0)
    ap.add_argument("--preset", default="high", choices=sorted(PRESETS))
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--spp", type=int)
    ap.add_argument("--bounces", type=int)
    ap.add_argument("--aperture", type=float)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--crf", type=int, default=17)
    ap.add_argument("--exposure", type=float, default=1.25)
    ap.add_argument("--no-denoise", dest="denoise", action="store_false")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--scenes", default=None, help="comma separated scene indices")
    args = ap.parse_args()

    W, H, spp, bounces, aperture, denoise = PRESETS[args.preset]
    args.width = args.width or W
    args.height = args.height or H
    args.spp = args.spp or spp
    args.bounces = args.bounces if args.bounces is not None else bounces
    args.aperture = args.aperture if args.aperture is not None else aperture
    args.denoise = args.denoise and denoise

    engine.setup_environment()
    render_mod = load_render()

    outdir = Path(args.out) if args.out else (SRC / "hq")
    outdir.mkdir(parents=True, exist_ok=True)

    if args.mode == "stills":
        picks = [int(x) for x in (args.scenes or "0,4,8,11,15,18").split(",")]
        for i in picks:
            t0 = time.time()
            im = render_frame(render_mod, i, 7.0, args.width, args.height, args.spp,
                              args.bounces, args.aperture, args.denoise, args.exposure)
            p = outdir / f"hq-still-{i:02}.jpg"
            im.save(p, quality=95)
            print(f"{p}  ({time.time() - t0:.1f}s)", flush=True)
        return

    if args.mode == "scene":
        encode_scene(render_mod, args.index, args, outdir / f"scene-{args.index:02}.mp4")
        return

    # mode == all
    scenes = [int(x) for x in args.scenes.split(",")] if args.scenes else list(range(20))
    for i in scenes:
        target = outdir / f"scene-{i:02}.mp4"
        if target.exists() and target.stat().st_size > 1000:
            print(f"scene {i + 1:02} already rendered, skipping", flush=True)
            continue
        encode_scene(render_mod, i, args, target)


if __name__ == "__main__":
    main()
