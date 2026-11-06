#!/usr/bin/env python3
"""Joint H-alpha + Fe I velocity-field grid, one figure per observation time.

This is the standard figure for a run that processed both spectral lines:

    row 1  H-alpha   (a) V_LOS  (b) V_diffrot  (c) V_limb+V_merid+V_offset  (d) V_resid
    row 2  Fe I      (e) V_LOS  (f) V_diffrot  (g) V_limb+V_merid+V_offset  (h) V_resid

It scans an output tree for ``<stamp>_HA_scv.fits`` and ``<stamp>_FEI_scv.fits``
and pairs them on the observation stamp, so every observing time gets its own
figure.

Examples
--------
    # one figure per observing time found under an output directory
    python scripts/plot_joint_ha_fe.py /data/scv -o /data/scv/joint

    # a single pair
    python scripts/plot_joint_ha_fe.py --ha A_HA_scv.fits --fe A_FEI_scv.fits \
        -o figures/joint.png

    # compact version for a talk
    python scripts/plot_joint_ha_fe.py /data/scv --font-scale 1.5 --dpi 150
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from solar_chromosphere_velocity.plotting_grid import (  # noqa: E402
    find_joint_pairs, plot_joint_grid, plot_joint_series)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", help="output directory to scan")
    ap.add_argument("--ha", help="a single H-alpha product")
    ap.add_argument("--fe", help="the matching Fe I product")
    ap.add_argument("-o", "--out", help="output file (single pair) or "
                                        "directory (scan mode)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--font-scale", type=float, default=2.5)
    ap.add_argument("--panel-height", type=float, default=4.75)
    ap.add_argument("--gap", type=float, default=0.048)
    args = ap.parse_args(argv)

    common = dict(dpi=args.dpi, font_scale=args.font_scale,
                  panel_height_in=args.panel_height, gap_in=args.gap)

    if args.ha or args.fe:
        if not (args.ha and args.fe):
            ap.error("--ha and --fe must be given together")
        out = args.out or "joint_grid.png"
        print(plot_joint_grid(args.ha, args.fe, out, **common))
        return 0

    if not args.root:
        ap.error("give either a root directory, or --ha and --fe")
    pairs = find_joint_pairs(args.root)
    if not pairs:
        print("no observation time has both lines under %s" % args.root,
              file=sys.stderr)
        return 1
    out_dir = args.out or os.path.join(args.root, "joint")
    made = plot_joint_series(pairs, out_dir, **common)
    for stamp in sorted(made):
        print("%s -> %s" % (stamp, made[stamp]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
