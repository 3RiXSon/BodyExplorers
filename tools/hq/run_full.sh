#!/usr/bin/env bash
# Render the whole episode with the HQ path tracer, resiliently.
#
# Scenes are encoded one at a time and skipped if already present, so the job
# can be interrupted and resumed. The virtualenv is rebuilt automatically if it
# disappears (the sandbox prunes ~/.venv between sessions).
set -uo pipefail
cd "$(dirname "$0")/../.."

REPO="$PWD"
VENV="${VENV:-$HOME/.venv}"
OUT="${OUT:-$REPO/Body-Explorers-Source-Package/hq}"
PRESET="${PRESET:-good}"
SPP="${SPP:-14}"
CRF="${CRF:-16}"
SCENES="${SCENES:-20}"

export PATH="$HOME/bin:$PATH"
mkdir -p "$OUT"

while true; do
    done_count=$(ls "$OUT"/scene-*.mp4 2>/dev/null | wc -l)
    if [ "$done_count" -ge "$SCENES" ]; then
        echo "ALL $SCENES SCENES RENDERED"
        break
    fi
    if [ ! -x "$VENV/bin/python" ]; then
        echo "rebuilding render environment..."
        bash "$REPO/tools/setup_env.sh" || { sleep 10; continue; }
    fi
    echo "--- launching renderer ($done_count/$SCENES scenes complete) ---"
    "$VENV/bin/python" "$REPO/tools/hq/render_hq.py" all \
        --preset "$PRESET" --spp "$SPP" --crf "$CRF" --out "$OUT"
    sleep 5
done

# Stitch the scenes together in order and lay the original audio back on top.
: > "$OUT/concat.txt"
for f in $(ls "$OUT"/scene-*.mp4 | sort); do echo "file '$f'" >> "$OUT/concat.txt"; done
ffmpeg -y -v error -f concat -safe 0 -i "$OUT/concat.txt" -c copy "$OUT/silent-hq.mp4"
echo "SILENT CUT READY: $OUT/silent-hq.mp4"
