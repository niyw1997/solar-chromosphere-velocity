#!/usr/bin/env python3
"""Plot the velocity layers of a v2 multi-extension product.

Examples
--------
    python scripts/visualize_layers.py out/20260424/*_scv.fits -o figs
    python scripts/visualize_layers.py file.fits --layers VLOS_OBS V_RESID
"""

import argparse
import os
import sys

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    p.add_argument("-o", "--output", default=".", help="output directory")
    p.add_argument("--layers", nargs="*", default=None)
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args(argv)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from astropy.io import fits
    except ImportError as exc:
        print("matplotlib and astropy are required: %s" % exc, file=sys.stderr)
        return 2

    os.makedirs(args.output, exist_ok=True)
    default_layers = ["VLOS_OBS", "V_DIFFROT", "V_LIMB", "V_MERID",
                      "V_LARGE", "V_RESID"]
    for path in args.files:
        with fits.open(path) as h:
            names = [hd.header.get("EXTNAME", "") for hd in h]
            wanted = args.layers or [n for n in default_layers if n in names]
            wanted = [w for w in wanted if w in names]
            if not wanted:
                print("no plottable layers in %s (has %s)" % (path, names))
                continue
            ncol = min(3, len(wanted))
            nrow = int(np.ceil(len(wanted) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.5 * nrow),
                                     dpi=args.dpi, squeeze=False)
            for ax, name in zip(axes.ravel(), wanted):
                data = h[name].data.astype(float)
                vmax = args.vmax if args.vmax is not None else \
                    np.nanpercentile(np.abs(data), 99)
                vmin = args.vmin if args.vmin is not None else -vmax
                im = ax.imshow(data, origin="lower", cmap="bwr",
                               vmin=vmin, vmax=vmax)
                unit = h[name].header.get("BUNIT", "")
                ax.set_title("%s%s" % (name, " [%s]" % unit if unit else ""))
                plt.colorbar(im, ax=ax, fraction=0.046)
            for ax in axes.ravel()[len(wanted):]:
                ax.axis("off")
            ph = h[0].header
            fig.suptitle("%s   %s   model=%s" % (
                os.path.basename(path), ph.get("OBSTIME", "?"),
                ph.get("MODELVER", "?")), fontsize=12)
            plt.tight_layout()
            base = os.path.splitext(os.path.basename(path))[0]
            out = os.path.join(args.output, base + "_layers.png")
            plt.savefig(out, bbox_inches="tight")
            plt.close(fig)
            print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
