#!/bin/bash
# ---------------------------------------------------------------------------
# Single-machine example: process a directory of CHASE/HIS level-1 cubes.
#
#   bash examples/run_local.sh <input_dir> <output_dir> [pattern]
#
# Everything is written under <output_dir>/<YYYYMMDD>/; the input directory is
# never written to.  Figures are optional and cost roughly as much time as the
# analysis itself, so they are behind a flag.
# ---------------------------------------------------------------------------
set -euo pipefail

INPUT="${1:?usage: run_local.sh <input_dir> <output_dir> [pattern]}"
OUTPUT="${2:?usage: run_local.sh <input_dir> <output_dir> [pattern]}"
PATTERN="${3:-*_HA.fits}"

python -m solar_chromosphere_velocity.cli \
    --input  "$INPUT" \
    --output "$OUTPUT" \
    --pattern "$PATTERN" \
    --spatial-strip-rows 32 \
    --visualize \
    --plot-per-component \
    --log-level INFO

# The products:
#   <output_dir>/index.csv                     observation log, one row per input
#   <output_dir>/<YYYYMMDD>/<time>_<LINE>_scv.fits   10 layers + FITPARAMS
#   <output_dir>/<YYYYMMDD>/<time>_<LINE>_scv_*.png   9 figures per time step

# Notes
#   * add --legacy to reproduce v1.0.1 (regression mode)
#   * add --overwrite to reuse an existing output name instead of adding a suffix
#   * drop --visualize / --plot-per-component for the fastest batch runs
#   * --limb-exponent 2.0 fixes N instead of scanning the grid
