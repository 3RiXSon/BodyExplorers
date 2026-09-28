#!/usr/bin/env bash
# Recreate the render environment (no system packages required).
# ffmpeg comes from the imageio-ffmpeg wheel, so no apt/root access is needed.
set -euo pipefail
VENV="${VENV:-$HOME/.venv}"
BIN="${BIN:-$HOME/bin}"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet numpy Pillow imageio-ffmpeg
mkdir -p "$BIN"
FF=$("$VENV/bin/python" -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
ln -sf "$FF" "$BIN/ffmpeg"
echo "env ready: $VENV  |  ffmpeg: $("$BIN/ffmpeg" -version | head -1)"
