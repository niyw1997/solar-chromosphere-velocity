#!/usr/bin/env python3
"""Use the Python API directly instead of the command line.

    python examples/api_example.py <level1.fits> <output_dir>
"""

import os
import sys

from solar_chromosphere_velocity import (
    ChromosphereProcessor, Config, plot_from_file,
)


def main(src, out_dir):
    cfg = Config(
        input_dir=os.path.dirname(os.path.abspath(src)),
        output_dir=out_dir,
        visualize=False,              # the figure is drawn explicitly below
        spatial_strip_rows=32,
        log_level="INFO",
    )
    result = ChromosphereProcessor(cfg).process_file(src)

    print("status        :", result.status)
    print("product       :", result.outfile)
    print("fit points    :", result.n_fit_points)
    print("RMS           : %.4f km/s" % result.fit.rms_km_s)
    print("parameters    :", {k: round(v, 5)
                              for k, v in zip(("A", "B", "C", "D", "G", "e",
                                               "N", "MC1", "MC2"),
                                              result.fit.params)})
    print("not estimated :", result.fit.flags or "{}")
    print("layers        :", sorted(result.layers))
    print("timings (s)   :", {k: round(v, 2) for k, v in result.timings.items()})

    # Draw the six model components from the product that was just written.
    fig = plot_from_file(result.outfile, os.path.join(out_dir, "components.png"),
                         panels=["V_DIFFROT", "V_LIMB", "V_MERID", "V_OFFSET",
                                 "V_LARGE", "V_RESID"])
    print("figure        :", fig)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
