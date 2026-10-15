#!/usr/bin/env python3
"""Plot the velocity components of a v2 multi-extension product.

Thin command-line wrapper around ``solar_chromosphere_velocity.plotting``; the
same figures can be produced from Python with ``plot_layers`` /
``plot_components`` / ``plot_summary`` / ``write_component_figures``, or
automatically during processing with ``--visualize``.

Examples
--------
    # all components present, 3 per row
    python scripts/visualize_layers.py out/20260424/*_scv.fits -o figs

    # only the model terms, one file per component
    python scripts/visualize_layers.py f.fits -o figs \
        --panels V_DIFFROT V_LIMB V_MERID V_OFFSET --per-component

    # the preset 2x3 component layout + the summary figure
    python scripts/visualize_layers.py f.fits -o figs --preset components summary
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from solar_chromosphere_velocity.plotting import (  # noqa: E402
    COMPONENT_PANELS, PANEL_ORDER, plot_from_file, plot_summary,
    write_component_figures)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("-o", "--output", default=".", help="output directory")
    ap.add_argument("--panels", nargs="*", default=None,
                    help="layers to draw, in order (default: %s)"
                         % " ".join(PANEL_ORDER))
    ap.add_argument("--preset", nargs="*", default=["layers"],
                    choices=["layers", "components", "summary"],
                    help="which preset figures to write")
    ap.add_argument("--per-component", action="store_true",
                    help="additionally write one figure per component")
    ap.add_argument("--ncols", type=int, default=3)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--cmap", default="bwr")
    ap.add_argument("--percentile", type=float, default=99.5)
    ap.add_argument("--shared-scale", action="store_true",
                    help="use one colour scale for every panel (default: per panel)")
    ap.add_argument("--overlay-mask", action="store_true",
                    help="contour the active regions onto the panels")
    args = ap.parse_args(argv)

    os.makedirs(args.output, exist_ok=True)
    common = dict(ncols=args.ncols, dpi=args.dpi, cmap=args.cmap,
                  percentile=args.percentile,
                  per_panel_scale=not args.shared_scale,
                  overlay_mask=args.overlay_mask)
    rc = 0
    for path in args.files:
        base = os.path.join(args.output, os.path.splitext(os.path.basename(path))[0])
        try:
            if "layers" in args.preset:
                out = plot_from_file(path, base + "_layers.png",
                                     panels=args.panels, **common)
                print("wrote %s" % out)
            if "components" in args.preset:
                out = plot_from_file(path, base + "_components.png",
                                     panels=list(COMPONENT_PANELS), **common)
                print("wrote %s" % out)
            if "summary" in args.preset:
                out = plot_summary(path, base + "_summary.png", dpi=args.dpi)
                print("wrote %s" % out)
            if args.per_component:
                for name, out in write_component_figures(
                        path, base + "_component", panels=args.panels,
                        dpi=args.dpi, cmap=args.cmap).items():
                    print("wrote %s" % out)
        except Exception as exc:
            print("FAILED %s: %s: %s" % (path, type(exc).__name__, exc),
                  file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
