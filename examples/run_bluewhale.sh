#!/bin/bash
# ---------------------------------------------------------------------------
# BlueWhale (LSF) job body.  This is the script the compute node runs; submit it
# with examples/submit_bluewhale.sh (or the equivalent bsub line).
#
# Runs INSIDE the cluster, from the directory that holds code/ and data/.
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="${1:-$ROOT/data}"
OUTPUT="${2:-$ROOT/work}"
PATTERN="${3:-*_HA.fits}"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1          # keep the shared tree clean

mkdir -p "$OUTPUT"
echo "host      : $(hostname)"
echo "python    : $(command -v python)  $(python -V 2>&1)"
echo "nproc     : $(nproc 2>/dev/null || echo '?')"
echo "started   : $(date -Is)"

# One process over the whole directory; the analysis is single-threaded by
# design, so request the cores you want to use for *parallel files* here.
time python -m solar_chromosphere_velocity.cli \
    --input  "$INPUT" \
    --output "$OUTPUT" \
    --pattern "$PATTERN" \
    --spatial-strip-rows 32 \
    --visualize \
    --log-level INFO

echo "finished  : $(date -Is)"
echo "index     : $OUTPUT/index.csv"
