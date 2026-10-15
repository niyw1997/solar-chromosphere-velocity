"""Command line interface.

Two entry points share one implementation:

``scv``            v2 defaults (complete physics, multi-HDU output)
``process-chase``  v1-compatible entry: implies ``--legacy`` and accepts the
                   v1 flag names, so existing scripts keep running.

v1 options that never did anything are accepted and explicitly reported as
ignored rather than silently accepted (see :data:`V1_DEAD_OPTIONS`).
"""

import argparse
import glob
import json
import logging
import os
import sys

from .config import Config
from .processor import ChromosphereProcessor

__all__ = ["main", "build_parser"]

# v1 flags that were dead (never read) - kept so old command lines still parse.
V1_DEAD_OPTIONS = {
    "spectral_processing": "v2 always inverts spectra when the input is spectral",
    "spectral_line": "the line is taken from the header/filename",
    "reference_profile": "the template is always the disk-centre average profile",
    "doppler_method": "v2 uses the cross-correlation estimator",
    "no_validation": "validation is always performed",
    "debug": "removed",
    "save_intermediate": "v2 always writes the multi-HDU product",
}


def build_parser(prog: str = "scv", legacy_default: bool = False):
    p = argparse.ArgumentParser(
        prog=prog,
        description="Solar chromosphere velocity processor (v2, multi-HDU output).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    src = p.add_argument_group("input / output")
    src.add_argument("--input", "-i", help="input directory")
    src.add_argument("--output", "-o", required=True, help="output directory")
    src.add_argument("--file", "-f", help="process a single file instead of a directory")
    src.add_argument("--pattern", "-p", default="*.fits",
                     help="glob for --input (default matches *_HA/_FE files)")
    src.add_argument("--exclude", action="append", default=None,
                     help="glob to exclude (repeatable)")
    src.add_argument("--flat", action="store_true",
                     help="do not create per-day subdirectories")
    src.add_argument("--overwrite", action="store_true",
                     help="allow overwriting existing outputs")
    src.add_argument("--compress", action="store_true",
                     help="write RICE_1 compressed output HDUs")
    src.add_argument("--no-index", action="store_true", help="skip index.csv")

    mod = p.add_argument_group("model / physics")
    mod.add_argument("--legacy", action="store_true", default=legacy_default,
                     help="reproduce v1.0.1 behaviour (regression mode)")
    mod.add_argument("--rotation-amplitude", choices=["kms", "rad_per_s"],
                     default="kms",
                     help="units of A,B,C,D (rad_per_s multiplies by R_sun)")
    mod.add_argument("--n-limb", type=float, default=None,
                     help="fix the limb exponent N (default: scan)")
    mod.add_argument("--allow-negative-limb", action="store_true",
                     help="do not pin the limb amplitude e to >= 0")
    mod.add_argument("--zero-point", type=float, default=0.0,
                     help="constant subtracted from VLOS_OBS (km/s)")

    inv = p.add_argument_group("inversion")
    inv.add_argument("--window-half", type=int, default=10)
    inv.add_argument("--profile-radius", type=float, default=100.0,
                     help="radius of the disk-centre averaging mask (pixels)")
    inv.add_argument("--raw-correlation", action="store_true",
                     help="do not mean-subtract the correlation windows")

    msk = p.add_argument_group("masks")
    msk.add_argument("--disk-exclusion", type=float, default=0.95)
    msk.add_argument("--edge-angle", type=float, default=80.0)
    msk.add_argument("--activity-threshold", type=float, default=3.0)
    msk.add_argument("--min-area", type=int, default=100)
    msk.add_argument("--sigma-clip", action="store_true",
                     help="use mean/std instead of the robust MAD threshold")

    perf = p.add_argument_group("performance")
    perf.add_argument("--row-block", type=int, default=32,
                      help="rows per streaming block")
    perf.add_argument("--workers", "-w", type=int, default=1,
                      help="reserved; v2 is fast enough single-process")

    out = p.add_argument_group("plots / logging")
    out.add_argument("--visualize", "-v", action="store_true")
    out.add_argument("--plot-dpi", type=int, default=150)
    out.add_argument("--plot-format", default="png")
    out.add_argument("--log-level", default="INFO",
                     choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    out.add_argument("--log-file")
    out.add_argument("--config", "-c", help="JSON configuration file")
    out.add_argument("--json", action="store_true",
                     help="print the per-file summary as JSON")

    if legacy_default:
        # v1 flag names, accepted for compatibility
        g = p.add_argument_group("v1 compatibility (accepted, mostly ignored)")
        g.add_argument("--spectral-processing", action="store_true", default=True)
        g.add_argument("--no-spectral-processing", action="store_true")
        g.add_argument("--spectral-line", choices=["auto", "HA", "FE"], default="auto")
        g.add_argument("--reference-profile")
        g.add_argument("--doppler-method",
                       choices=["cross_correlation", "gaussian_fit"],
                       default="cross_correlation")
        g.add_argument("--save-intermediate", action="store_true")
        g.add_argument("--no-validation", action="store_true")
        g.add_argument("--debug", action="store_true")
        g.add_argument("--visualize-all", action="store_true")
        g.add_argument("--no-visualize", action="store_true")
    return p


def _setup_logging(level: str, log_file=None):
    handlers = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(levelname)s %(name)s: %(message)s",
                        handlers=handlers, force=True)


def _report_ignored(args):
    ignored = []
    for name, why in V1_DEAD_OPTIONS.items():
        if not hasattr(args, name):
            continue
        val = getattr(args, name)
        if val not in (None, False, "auto", "cross_correlation"):
            ignored.append("%s (%s)" % (name, why))
    if getattr(args, "no_visualize", False):
        args.visualize = False
    if getattr(args, "visualize_all", False):
        args.visualize = True
    if ignored:
        logging.getLogger("scv").warning(
            "these v1 options have no effect in v2: %s", "; ".join(ignored))


def config_from_args(args) -> Config:
    kw = dict(
        output_dir=args.output,
        input_dir=args.input or "",
        pattern=args.pattern,
        legacy=bool(args.legacy),
        rotation_amplitude=args.rotation_amplitude,
        n_limb=args.n_limb,
        limb_nonneg=not args.allow_negative_limb,
        zero_point_km_s=args.zero_point,
        window_half=args.window_half,
        profile_mask_radius=args.profile_radius,
        demean_window=False if args.raw_correlation else None,
        disk_exclusion_radius=args.disk_exclusion,
        edge_exclusion_angle=args.edge_angle,
        activity_threshold_sigma=args.activity_threshold,
        min_activity_area=args.min_area,
        use_mad=False if args.sigma_clip else None,
        row_block=args.row_block,
        n_workers=args.workers,
        visualize=args.visualize,
        plot_dpi=args.plot_dpi,
        plot_format=args.plot_format,
        compress_output=args.compress,
        flat_output=args.flat,
        overwrite=args.overwrite,
        write_index=not args.no_index,
        log_level=args.log_level,
        log_file=args.log_file,
    )
    if args.exclude:
        kw["exclude_globs"] = tuple(args.exclude)
    if args.config:
        with open(args.config) as fh:
            kw.update(json.load(fh))
    return Config(**kw)


def main(argv=None):
    prog = os.path.basename(sys.argv[0] or "scv")
    legacy_default = prog.startswith("process-chase")
    parser = build_parser(prog=prog, legacy_default=legacy_default)
    args = parser.parse_args(argv)

    if not args.file and not args.input:
        parser.error("either --file or --input is required")
    _setup_logging(args.log_level, args.log_file)
    _report_ignored(args)

    cfg = config_from_args(args)
    proc = ChromosphereProcessor(cfg)

    if args.file:
        results = [proc.process_file(args.file)]
    else:
        results = proc.process_directory()

    ok = [r for r in results if r.status == "ok"]
    bad = [r for r in results if r.status != "ok"]
    if args.json:
        print(json.dumps([r.summary() for r in results], indent=2, default=str))
    else:
        for r in results:
            if r.fit is not None:
                print("%-44s -> %s  RMS=%.4f km/s  N=%.2f  V_OFFSET=%+.4f km/s"
                      % (r.srcfile, os.path.basename(r.outfile or "-"),
                         r.fit.rms_km_s, r.fit.n_limb, r.fit.params[4]))
            else:
                print("%-44s -> %s (%s: %s)"
                      % (r.srcfile, r.outfile or "-", r.status, r.message))
        print("summary: %d ok, %d failed" % (len(ok), len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
