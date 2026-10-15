"""v2 processing pipeline.

End-to-end per input file, streaming and block-wise:

1. locate the cube and its geometry (RICE or plain, HDUs auto-detected);
2. Doppler inversion (batched FFT, header-driven calibration);
3. zero-point correction, active-region mask;
4. streamed accumulation of the normal equations over the quiet-Sun sample;
5. one linear solve (+ profile-likelihood scan over N), covariance;
6. block-wise reconstruction of every model component and its 1-sigma;
7. one multi-extension FITS per observation time, plus index.csv.

Nothing is ever written next to the input files.
"""

import glob
import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .config import Config
from .data_loader import (DataType, SpectralCube, detect_data_type,
                          extract_solar_params, open_plane)
from .geometry import coord_arrays, solar_coords
from .inversion import DopplerInverter, DopplerResult
from .masks import (active_region_mask, apply_disk_exclusion, disk_mask,
                    effective_disk_mask, effective_radius_px)
from .models import ModelFit, VelocityModel, fit_model_streaming
from .output import LAYERS, OutputLayout, observation_key, write_multihdu

logger = logging.getLogger(__name__)

__all__ = ["ChromosphereProcessor", "ProcessResult", "ProcessingError"]


class ProcessingError(RuntimeError):
    """Raised by :meth:`process_file`; carries the partially built result."""

    def __init__(self, message, result=None):
        super().__init__(message)
        self.result = result


@dataclass
class ProcessResult:
    srcfile: str
    outfile: Optional[str] = None
    key: Optional[object] = None
    fit: Optional[ModelFit] = None
    layers: Dict[str, np.ndarray] = field(default_factory=dict)
    pixel_scale_km_s: Optional[float] = None
    line_center: Optional[object] = None
    mask_stats: Dict[str, Any] = field(default_factory=dict)
    binding_mask: str = ""
    timings: Dict[str, float] = field(default_factory=dict)
    figures: Dict[str, str] = field(default_factory=dict)
    indexed: bool = False                  # already recorded in index.csv
    status: str = "ok"
    message: str = ""
    n_fit_points: int = 0

    def summary(self) -> Dict[str, Any]:
        d = {"srcfile": self.srcfile, "outfile": self.outfile,
             "status": self.status, "n_fit_points": self.n_fit_points}
        if self.fit is not None:
            d.update(self.fit.as_dict())
            d["rms_km_s"] = self.fit.rms_km_s
        return d


class ChromosphereProcessor:
    """Streaming, vectorised processor (v2)."""

    min_fit_points = 100

    def __init__(self, config: Config):
        self.config = config
        self._setup_logging()
        self.layout = OutputLayout(config)
        self.inverter = DopplerInverter(config)

    # ------------------------------------------------------------------ #
    def _setup_logging(self):
        logger.setLevel(getattr(logging, str(self.config.log_level).upper(),
                                logging.INFO))
        if self.config.log_file:
            fh = logging.FileHandler(self.config.log_file)
            fh.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(fh)

    # ------------------------------------------------------------------ #
    def _read_input(self, path):
        """Return (vlos, header, params, kind, extra)."""
        dtype = detect_data_type(path)
        if dtype in (DataType.SPECTRAL_HA, DataType.SPECTRAL_FE):
            cube = SpectralCube(path)
            params = cube.params.require_geometry()
            t0 = time.perf_counter()
            res = self.inverter.run(cube, params)
            self._t_invert = time.perf_counter() - t0
            header = cube.header
            cube.close()
            return res.velocity, header, params, dtype, res
        if dtype is DataType.DOPPLER_VDOP:
            data, header = open_plane(
                path, require_unit=self.config.require_velocity_unit)
            params = extract_solar_params(header).require_geometry()
            data = apply_disk_exclusion(data, params,
                                        self.config.disk_mask_radius)
            return data, header, params, dtype, None
        raise ValueError("unrecognised input %s (type=%s)"
                         % (os.path.basename(path), dtype.value))

    # ------------------------------------------------------------------ #
    def _fit_sample_blocks(self, flagged, params, mask):
        """Yield (coords, velocity) blocks over the fit sample."""
        ny = flagged.shape[0]
        for y0 in range(0, ny, self.config.spatial_strip_rows):
            y1 = min(ny, y0 + self.config.spatial_strip_rows)
            sub = mask[y0:y1]
            if not sub.any():
                continue
            iy, ix = np.nonzero(sub)
            coords = solar_coords(iy + y0, ix, radius=params.radius,
                                  b0_deg=params.b0, p0_deg=params.p0,
                                  cx=params.cx, cy=params.cy)
            yield coords, flagged[y0:y1][sub]

    def _reconstruct_blocks(self, params, model, fit, radius_px, ny, nx):
        """Block-wise model components + 1-sigma uncertainty."""
        out = {name: np.full((ny, nx), np.nan, dtype=np.float32)
               for name in ("V_DIFFROT", "V_LIMB", "V_MERID", "V_OFFSET",
                            "V_LARGE", "V_MODEL_ERR")}
        for y0 in range(0, ny, self.config.spatial_strip_rows):
            y1 = min(ny, y0 + self.config.spatial_strip_rows)
            iy, ix = np.nonzero(self._block_disk_mask(y0, y1, nx, params, radius_px))
            if iy.size == 0:
                continue
            coords = solar_coords(iy + y0, ix, radius=params.radius,
                                  b0_deg=params.b0, p0_deg=params.p0,
                                  cx=params.cx, cy=params.cy)
            comp = model.components(coords, fit.params)
            err = model.uncertainty(coords, fit.cov)
            rows = slice(y0, y1)
            for name in ("V_DIFFROT", "V_LIMB", "V_MERID", "V_OFFSET", "V_LARGE"):
                out[name][y0 + iy, ix] = comp[name].astype(np.float32)
            out["V_MODEL_ERR"][y0 + iy, ix] = err.astype(np.float32)
        return out

    @staticmethod
    def _block_disk_mask(y0, y1, nx, params, radius_px):
        yy = np.arange(y0, y1, dtype=np.float64)[:, None]
        xx = np.arange(nx, dtype=np.float64)[None, :]
        return ((yy - params.cy) ** 2 + (xx - params.cx) ** 2) <= radius_px ** 2

    # ------------------------------------------------------------------ #
    def process_file(self, path: str) -> ProcessResult:
        cfg = self.config
        t_start = time.perf_counter()
        res = ProcessResult(srcfile=os.path.basename(path))
        timings: Dict[str, float] = {}

        t0 = time.perf_counter()
        vlos, header, params, dtype, dop = self._read_input(path)
        timings["read+invert"] = time.perf_counter() - t0
        if dop is not None:
            timings["invert"] = getattr(self, "_t_invert", float("nan"))
            res.pixel_scale_km_s = dop.scale_km_s
            res.line_center = dop.line_center
        else:
            res.pixel_scale_km_s = None

        # zero point + active regions
        t0 = time.perf_counter()
        vcorr = (vlos - cfg.zero_point).astype(np.float32, copy=True)
        active, stats = active_region_mask(vcorr, params, cfg, return_stats=True)
        res.mask_stats = stats
        timings["masks"] = time.perf_counter() - t0

        # fit sample = effective disk  &  no active region  &  finite
        eff_radius, binding, r_edge, r_disk = effective_radius_px(
            params, cfg.disk_mask_radius, cfg.edge_exclusion_angle)
        res.binding_mask = binding
        fit_region = disk_mask(vcorr.shape, params,
                               frac=eff_radius / float(params.radius))
        fit_mask = fit_region & (~active) & np.isfinite(vcorr)
        n_fit = int(fit_mask.sum())
        res.n_fit_points = n_fit

        key = observation_key(header, path, params=params)
        res.key = key
        base_cards = self._base_cards(key, params, dtype, off=cfg.zero_point,
                                      binding=binding, r_edge=r_edge,
                                      r_disk=r_disk, dop=dop, n_fit=n_fit,
                                      edge_angle=cfg.edge_exclusion_angle)

        if n_fit < self.min_fit_points:
            res.status = "INSUFFICIENT_DATA"
            res.message = ("only %d usable fit points (< %d)"
                           % (n_fit, self.min_fit_points))
            logger.error("%s: %s", res.srcfile, res.message)
            self._index_row(res, key, None, cfg)
            raise ProcessingError(res.message, result=res)

        # linear fit (streamed normal equations, profile scan over N)
        t0 = time.perf_counter()
        model = VelocityModel(model_version=cfg.model_version,
                              limb_exponent=cfg.limb_exponent or 2.0,
                              rotation_radius=cfg.use_rotation_radius)
        fit = fit_model_streaming(
            self._fit_sample_blocks(vcorr, params, fit_mask),
            limb_exponent_values=cfg.limb_exponent_values,
            model_version=cfg.model_version,
            rotation_radius=cfg.use_rotation_radius,
            limb_amplitude_positive=cfg.limb_amplitude_positive)
        model.limb_exponent = fit.limb_exponent
        timings["fit"] = time.perf_counter() - t0
        res.fit = fit

        # reconstruction
        t0 = time.perf_counter()
        ny, nx = vcorr.shape
        comp = self._reconstruct_blocks(params, model, fit, eff_radius, ny, nx)
        large = comp["V_LARGE"]
        resid = (vcorr - large).astype(np.float32)

        # zero-mean convention: v1 used the whole disk, v2 the quiet-Sun sample
        mean_mask = fit_region if not cfg.legacy else disk_mask(
            resid.shape, params, frac=cfg.disk_mask_radius)
        sel = mean_mask & np.isfinite(resid)
        quiet_mean = float(np.mean(resid[sel])) if sel.any() else 0.0
        resid = (resid - quiet_mean).astype(np.float32)
        resid[~np.isfinite(vcorr)] = np.nan
        timings["reconstruct"] = time.perf_counter() - t0

        # layers
        layers = {
            "VLOS_OBS": vlos.astype(np.float32),
            "VLOS_CORR": vcorr,
            "V_DIFFROT": comp["V_DIFFROT"],
            "V_LIMB": comp["V_LIMB"],
            "V_MERID": comp["V_MERID"],
            "V_OFFSET": comp["V_OFFSET"],
            "V_LARGE": large,
            "V_RESID": resid,
            "V_MODEL_ERR": comp["V_MODEL_ERR"],
            "MASK_ACTIVE": active.astype(np.uint8),
        }
        res.layers = layers

        base_cards["QUIETMN"] = (quiet_mean,
                                 "mean removed from V_RESID (km/s)")
        base_cards["MASKBIND"] = (binding, "which radial cut limited the fit")
        base_cards["EFFRAD"] = (float(eff_radius), "effective fit radius (px)")
        if dop is not None:
            for k, v in dop.as_header_cards().items():
                base_cards[k] = v

        # write
        t0 = time.perf_counter()
        out_path = self.layout.resolve(key, product="scv")
        write_multihdu(out_path, layers, fits_header(header), fit=fit,
                       base_cards=base_cards, compress=cfg.rice_compression,
                       overwrite=True)
        res.outfile = out_path
        timings["write"] = time.perf_counter() - t0

        if cfg.visualize:
            t0 = time.perf_counter()
            res.figures = self._make_figures(out_path, layers, cfg,
                                             header=self._plot_header(base_cards,
                                                                      fit))
            timings["plots"] = time.perf_counter() - t0
        timings["total"] = time.perf_counter() - t_start
        res.timings = timings
        self._index_row(res, key, fit, cfg)
        logger.info("%s -> %s (%.1f s, %d fit points, RMS %.4f km/s)",
                    res.srcfile, os.path.basename(out_path),
                    timings["total"], n_fit, fit.rms_km_s)
        return res

    @staticmethod
    def _plot_header(base_cards, fit) -> "object":
        """Metadata for the figures, so their axes match the written file."""
        from astropy.io import fits          # no module-level import here
        h = fits.Header()
        items = dict(base_cards or {})
        if fit is not None:
            items.update(fit.header_cards())
        for k, v in items.items():
            val = v[0] if isinstance(v, tuple) else v
            try:
                h[k] = val
            except Exception:
                pass
        return h

    def _make_figures(self, out_path: str, layers: Dict[str, np.ndarray],
                      cfg, header=None) -> Dict[str, str]:
        """Write the multi-panel figures.  Never fatal: the FITS product is
        already on disk, so a plotting problem must not fail the run."""
        from .plotting import COMPONENT_PANELS, plot_layers, plot_summary, \
            write_component_figures
        base = os.path.splitext(out_path)[0]
        fmt = cfg.plot_format
        figs: Dict[str, str] = {}
        common = dict(dpi=cfg.plot_dpi, cmap=cfg.colormap,
                      overlay_mask=cfg.plot_overlay_mask, header=header)
        try:
            figs["layers"] = plot_layers(
                layers, "%s_layers.%s" % (base, fmt),
                panels=cfg.plot_panels, **common)
            figs["components"] = plot_layers(
                layers, "%s_components.%s" % (base, fmt),
                panels=COMPONENT_PANELS, **common)
            figs["summary"] = plot_summary(
                layers, "%s_summary.%s" % (base, fmt), dpi=cfg.plot_dpi,
                header=header)
            if cfg.plot_per_component:
                figs.update(write_component_figures(
                    layers, "%s_component" % base, dpi=cfg.plot_dpi,
                    cmap=cfg.colormap, header=header))
            logger.info("wrote %d figure(s): %s", len(figs),
                        ", ".join(os.path.basename(v) for v in figs.values()))
        except Exception as exc:                       # pragma: no cover
            logger.warning("plotting failed (%s: %s); the FITS product is "
                           "unaffected", type(exc).__name__, exc)
            figs["error"] = "%s: %s" % (type(exc).__name__, exc)
        return figs

    # ------------------------------------------------------------------ #
    def _base_cards(self, key, params, dtype, off, binding, r_edge, r_disk,
                    dop, n_fit, edge_angle):
        cards = {
            "OBSTIME": (key.stamp, "observation time used as the output key"),
            "TIMEKEY": (key.source, "where OBSTIME came from"),
            "SPECLINE": (params.spec_line or key.line, "spectral line"),
            "SRCFILE": (key.srcfile, "source file name"),
            "DATATYPE": (dtype.value, "input data type"),
            "CX": (float(params.cx), "disk centre x (pixels)"),
            "CY": (float(params.cy), "disk centre y (pixels)"),
            "R_SUN": (float(params.radius), "solar radius (pixels)"),
            "B0": (float(params.b0), "solar B0 (deg)"),
            "INST_ROT": (float(params.p0), "instrument P0 (deg)"),
            "DISKEXCL": (float(self.config.disk_mask_radius),
                         "disk exclusion radius (fraction of R_SUN)"),
            "EDGEEXC": (float(edge_angle), "edge exclusion angle (deg)"),
            "MASKBIND": (binding, "binding radial cut: disk|edge|equal"),
            "R_DISKPX": (float(r_disk), "disk-exclusion radius (px)"),
            "R_EDGEPX": (float(r_edge), "edge-exclusion radius (px)"),
            "ZEROPT": (float(off), "zero point subtracted from VLOS_OBS (km/s)"),
            "MODELVER": (self.config.model_version, "model formula version"),
            "LEGACY": (bool(self.config.legacy), "v1 compatibility mode"),
            "ROWBLK": (int(self.config.spatial_strip_rows), "streaming row-block size"),
            "UNITSYS": ("LOS km/s, +receding",
                        "sign convention of every velocity layer"),
            "INVMETH": ("xcorr_fft_batched" if dtype in (DataType.SPECTRAL_HA,
                                                         DataType.SPECTRAL_FE)
                        else "external_product",
                        "how VLOS_OBS was obtained"),
            "SOFTWARE": "solar-chromosphere-velocity",
            "VERSION": _version(),
            "PROCDATE": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if params.cdelt1:
            cards["CDELT1"] = (float(params.cdelt1), "arcsec per pixel (x)")
        if params.cdelt2:
            cards["CDELT2"] = (float(params.cdelt2), "arcsec per pixel (y)")
        if params.cdelt3:
            cards["CDELT3"] = (float(params.cdelt3), "Angstrom per spectral pixel")
        if params.wave_len:
            cards["WAVE_LEN"] = (float(params.wave_len), "line-core wavelength (A)")
        if params.crval3 is not None:
            cards["CRVAL3"] = float(params.crval3)
        return cards

    def _index_row(self, res, key, fit, cfg):
        row = {
            "obs_time": key.stamp, "day": key.day, "line": key.line,
            "source": key.source, "hdu_file": res.outfile or "",
            "src_file": key.srcfile, "model_version": cfg.model_version,
            "legacy": cfg.legacy, "n_points": res.n_fit_points,
            "disk_excl": cfg.disk_mask_radius,
            "edge_deg": cfg.edge_exclusion_angle,
            "status": res.status,
        }
        if fit is not None:
            row.update(fit.as_dict())
            row["rms_km_s"] = fit.rms_km_s
            row["limb_exponent"] = fit.limb_exponent
        self.layout.append_index(row)
        res.indexed = True

    # ------------------------------------------------------------------ #
    def process_directory(self, pattern: Optional[str] = None) -> List[ProcessResult]:
        cfg = self.config
        if not cfg.input_dir:
            raise ValueError("input_dir is not set")
        files = sorted(glob.glob(os.path.join(cfg.input_dir,
                                              pattern or cfg.pattern)))
        excluded = []
        for g in cfg.exclude_patterns:
            hits = set(glob.glob(os.path.join(cfg.input_dir, g)))
            excluded.extend(sorted(hits))
        if excluded:
            exc = set(excluded)
            files = [f for f in files if f not in exc]
            logger.info("excluded %d file(s) matching %s",
                        len(excluded), ", ".join(cfg.exclude_patterns))
        if not files:
            logger.warning("no input files matched %s in %s",
                           pattern or cfg.pattern, cfg.input_dir)
            return []
        logger.info("processing %d file(s)", len(files))
        out: List[ProcessResult] = []
        failures = 0
        for f in files:
            try:
                out.append(self.process_file(f))
            except Exception as exc:            # keep going, but count it
                failures += 1
                logger.error("FAILED %s: %s: %s", os.path.basename(f),
                             type(exc).__name__, exc)
                # keep the partially built result (it carries the real status
                # and may already be in index.csv) instead of inventing FAILED
                r = getattr(exc, "result", None)
                if r is None:
                    r = ProcessResult(srcfile=os.path.basename(f),
                                      status="FAILED",
                                      message="%s: %s" % (type(exc).__name__, exc))
                out.append(r)
                if not r.indexed:
                    # index.csv must be an inventory, failures included - but
                    # exactly once per input
                    try:
                        k = observation_key_from_file(f)
                        self._index_row(r, k, None, cfg)
                    except Exception:
                        logger.debug("could not record the failure in index.csv")
        if failures:
            logger.error("%d of %d file(s) failed", failures, len(files))
        return out


def observation_key_from_file(path):
    """Best-effort observation key for a file that failed before loading.

    Must never raise: index.csv is an inventory, so even a completely
    unreadable file gets a row (keyed on the file name, then its mtime).
    """
    from astropy.io import fits as _fits
    from .data_loader import open_plane as _op
    header = _fits.Header()
    for getter in (lambda: _op(path, require_unit=False)[1],
                   lambda: _fits.open(path, ignore_missing_end=True,
                                      verify="silentfix")[0].header):
        try:
            header = getter()
            break
        except Exception:
            continue
    try:
        return observation_key(header, path)
    except Exception:
        return observation_key(_fits.Header(), path)


def fits_header(header):
    """Return a plain FITS header copy (input may be a live HDU header)."""
    from astropy.io import fits
    return fits.Header(header)


def _version():
    from . import __version__
    return __version__
