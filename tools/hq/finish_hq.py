#!/usr/bin/env python3
"""Lay the episode audio onto the HQ picture cut.

Identical audio treatment to Body-Explorers-Source-Package/finish.py — the same
narration fitting and the same original pentatonic score — but muxed against
the path-traced video and encoded at a higher audio bitrate.
"""
import argparse
import math
import json
import os
import subprocess
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
SRC = REPO / "Body-Explorers-Source-Package"
SR = 44100
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")


def build_voice(script, out_path):
    """Fit each narration clip inside its 15 second shot without clipping."""
    voice = np.zeros(SR * 300, dtype=np.float32)
    for i, shot in enumerate(script):
        speed = max(1.0, shot["audio_duration"] / 14.0)
        pcm = subprocess.check_output([
            FFMPEG, "-v", "error", "-i", str(SRC / f"narration-{i:02}.mp3"),
            "-af", f"atempo={speed},asetpts=PTS-STARTPTS",
            "-ar", str(SR), "-ac", "1", "-f", "f32le", "-",
        ])
        samples = np.frombuffer(pcm, dtype="<f4")
        assert len(samples) / SR < 14.4, f"scene {i} narration too long"
        start = int((i * 15 + 0.5) * SR)
        voice[start:start + len(samples)] = samples
    rms = np.sqrt(np.mean(voice[voice != 0] ** 2))
    voice *= min(0.14 / max(rms, 1e-6), 0.86 / max(np.max(np.abs(voice)), 1e-6))
    with wave.open(str(out_path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SR)
        out.writeframes((voice * 32767).astype("<i2").tobytes())


def build_music(out_path, length=300):
    """Original, quiet pentatonic score; no sampled or third-party material."""
    music = np.zeros(SR * length, dtype=np.float32)

    def note(at, midi, duration, vol):
        start = int(at * SR)
        n = min(int(duration * SR), len(music) - start)
        if n <= 0:
            return
        t = np.arange(n) / SR
        f = 440 * 2 ** ((midi - 69) / 12)
        w = ((np.sin(2 * np.pi * f * t) + 0.23 * np.sin(2 * np.pi * f * 2 * t)
              + 0.08 * np.sin(2 * np.pi * f * 3 * t))
             * np.exp(-t * 3.5) * np.minimum(t / 0.012, 1))
        music[start:start + n] += w.astype("f4") * vol

    chords = [[48, 52, 55, 59], [45, 48, 52, 55], [41, 45, 48, 52], [43, 47, 50, 55]]
    beat = 60 / 96
    for bar in range(120):
        at = bar * 4 * beat
        if at >= length:
            break
        chord = chords[bar % 4]
        for k, m in enumerate(chord):
            note(at + k * 0.12, m + 12, 2.5, 0.018)
        for k, m in enumerate([72, 76, 79, 76] if bar % 2 == 0 else [74, 79, 81, 79]):
            note(at + k * beat, m, 1, 0.009)
        note(at, chord[0] - 12, 2, 0.016)
    for i in (8, 16):
        for j in np.arange(i * 15, (i + 1) * 15, 0.8):
            note(j, 32, 0.18, 0.019)
            note(j + 0.19, 29, 0.18, 0.012)
    fade = (np.minimum(np.arange(len(music)) / SR / 2, 1)
            * np.minimum((len(music) - np.arange(len(music))) / SR / 3, 1))
    music *= fade
    with wave.open(str(out_path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SR)
        out.writeframes((np.clip(music, -1, 1) * 32767).astype("<i2").tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(SRC / "hq" / "silent-hq.mp4"))
    ap.add_argument("--out", default=str(SRC / "Body-Explorers-E01-The-Great-Oxygen-Delivery-HQ.mp4"))
    args = ap.parse_args()

    script = json.loads((SRC / "script.json").read_text())
    voice_wav = SRC / "hq" / "voice.wav"
    music_wav = SRC / "hq" / "music.wav"
    voice_wav.parent.mkdir(parents=True, exist_ok=True)
    build_voice(script, voice_wav)
    build_music(music_wav)

    subprocess.run([
        FFMPEG, "-y", "-v", "error",
        "-i", args.video, "-i", str(voice_wav), "-i", str(music_wav),
        "-filter_complex",
        "[1:a]asetpts=PTS-STARTPTS[voc];[2:a]asetpts=PTS-STARTPTS[mus];"
        "[voc][mus]amix=inputs=2:duration=longest:normalize=0,"
        "alimiter=limit=0.94:level=false,apad,atrim=duration=300[a]",
        "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-t", "300", "-movflags", "+faststart", args.out,
    ], check=True)
    print(f"Finished HQ film -> {args.out}")


if __name__ == "__main__":
    main()
