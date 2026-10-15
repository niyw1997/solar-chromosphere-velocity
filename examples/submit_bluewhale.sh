#!/bin/bash
# ---------------------------------------------------------------------------
# Submit examples/run_bluewhale.sh to the BlueWhale mpi queue.
#
#   bash examples/submit_bluewhale.sh [job_name] [walltime]
#
# Cluster conventions: only the `mpi` queue is used, -n must be a multiple of
# 64, and the job is kept on one host unless you deliberately span nodes.
# ---------------------------------------------------------------------------
set -euo pipefail

JOB="${1:-scv-run}"
WALL="${2:-01:00}"
CORES=64
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

echo "would submit:"
cat <<CMD
bsub -q mpi -n $CORES -W $WALL -R 'span[hosts=1]' \\
     -J $JOB \\
     -o $ROOT/logs/$JOB-%J.out -e $ROOT/logs/$JOB-%J.err \\
     bash $HERE/run_bluewhale.sh $ROOT/data $ROOT/work '*_HA.fits'
CMD

read -r -p "submit now? [y/N] " ans
if [ "${ans:-N}" = "y" ] || [ "${ans:-N}" = "Y" ]; then
    mkdir -p "$ROOT/logs"
    bsub -q mpi -n "$CORES" -W "$WALL" -R 'span[hosts=1]' \
         -J "$JOB" \
         -o "$ROOT/logs/$JOB-%J.out" -e "$ROOT/logs/$JOB-%J.err" \
         bash "$HERE/run_bluewhale.sh" "$ROOT/data" "$ROOT/work" '*_HA.fits'
else
    echo "not submitted"
fi
