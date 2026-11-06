#!/bin/bash
# ---------------------------------------------------------------------------
# Job body for a computing cluster.
#
#   bash examples/run_cluster.sh [data_dir] [output_dir] [pattern] [nproc]
#
# Every cube is independent, so the work is spread by running one process per
# cube, `nproc` of them at a time.  One process holds about 1.5 GB, so choose
# `nproc` from the memory the node gives you rather than from its core count.
#
# Submit this file with whatever scheduler the cluster provides; the script does
# not depend on one.
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="${1:-$ROOT/data}"
OUTPUT="${2:-$ROOT/work}"
PATTERN="${3:-*_HA.fits}"
NPROC="${4:-4}"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1          # keep a shared source tree clean

mkdir -p "$OUTPUT"
echo "node      : $(hostname)"
echo "python    : $(command -v python)  $(python -V 2>&1)"
echo "processes : $NPROC"
echo "started   : $(date -Is)"

# --no-index: the observation index is a single file, and concurrent writers
# would interleave it.  Every product carries its own metadata in PRIMARY, so the
# index is only a convenience; for a handful of cubes, run this serially instead.
find "$INPUT" -maxdepth 1 -name "$PATTERN" -print0 \
  | xargs -0 -r -I {} -P "$NPROC" \
      python -m solar_chromosphere_velocity.cli \
          --file {} --output "$OUTPUT" --no-index --log-level INFO

echo "finished  : $(date -Is)"
