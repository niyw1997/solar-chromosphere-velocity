"""Backward-compatible helpers.

v1's ``utils`` mixed geometry, masks, validation and parallelism.  v2 splits
those responsibilities (``geometry``, ``masks``) and keeps this module so that
``from solar_chromosphere_velocity.utils import SolarCoordinateTransformer`` and
friends continue to work.  Everything here is a thin, vectorised wrapper - the
v1 scalar Python loops are gone.
"""

from functools import wraps
import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .geometry import coord_arrays, distance_px, pixel_to_solar, solar_coords
from .masks import (active_region_mask, apply_disk_exclusion as _apply_de,
                    create_solar_mask as _create_mask)

__all__ = [
    "SolarCoordinateTransformer",
    "ActiveRegionDetector",
    "DataValidator",
    "ParallelProcessor",
    "timeit",
    "create_solar_mask",
    "apply_disk_exclusion",
    "calculate_statistics",
]

logger = logging.getLogger(__name__)


class SolarCoordinateTransformer:
    """Vectorised replacement for v1's per-pixel transformer."""

    def __init__(self, config=None):
        self.config = config

    def pixel_to_solar(self, x, y, params) -> Dict[str, float]:
        return pixel_to_solar(x, y, params)

    def create_coordinate_grid(self, shape, params) -> Dict[str, np.ndarray]:
        """Full-image coordinate grids, computed in one vectorised pass.

        Keys match v1 (``radius_angle`` in degrees, ``normalized_radius``).
        """
        out = coord_arrays(shape, params, dtype=np.float32)
        out["radius_angle"] = out.pop("rho_deg")
        out["normalized_radius"] = (out["r_px"] / float(params["radius"])).astype(
            np.float32)
        return out


class ActiveRegionDetector:
    """Thin wrapper over :func:`masks.active_region_mask`."""

    def __init__(self, config):
        self.config = config

    def detect(self, velocity_map, params) -> np.ndarray:
        cfg = self.config
        if not hasattr(cfg, "mad_threshold"):
            raise TypeError("ActiveRegionDetector needs a v2 Config")
        return active_region_mask(velocity_map, _Params(params), cfg)

    def refine_mask(self, mask, velocity_map):
        grad_y, grad_x = np.gradient(np.nan_to_num(velocity_map))
        g = np.hypot(grad_x, grad_y)
        gmax = np.nanmax(g)
        if not np.isfinite(gmax) or gmax <= 0:
            return mask
        return mask | (g / gmax > 0.5)


class _Params:
    """Adapter: turn a v1 params dict into the attribute access v2 expects."""

    def __init__(self, d):
        self.cx = float(d["cx"])
        self.cy = float(d["cy"])
        self.radius = float(d["radius"])
        self.b0 = float(d.get("b0", 0.0))
        self.p0 = float(d.get("p0", 0.0))
        self.cdelt3 = d.get("cdelt3")
        self.crval3 = d.get("crval3")
        self.crpix3 = d.get("crpix3", 1.0)
        self.wave_len = d.get("wave_len")
        self.spec_line = d.get("spec_line", "")


class DataValidator:
    """Data sanity checks.

    v1 compared a NaN *count* against ``nan_threshold`` (default 1e6), which
    almost never triggered, and rejected any map whose range exceeded
    100 km/s.  v2 treats the threshold as a *fraction* of the map; a value > 1
    is interpreted as a pixel count for backward compatibility.
    """

    def __init__(self, config=None, nan_threshold: float = 0.5,
                 max_range_km_s: float = 100.0):
        self.config = config
        if config is not None and hasattr(config, "nan_threshold"):
            nan_threshold = config.nan_threshold
        self.nan_threshold = float(nan_threshold)
        self.max_range_km_s = float(max_range_km_s)

    def _limit(self, data) -> float:
        if self.nan_threshold > 1.0:            # v1 semantics: a pixel count
            return self.nan_threshold
        return self.nan_threshold * data.size

    def validate(self, data) -> bool:
        n_nan = int(np.isnan(data).sum())
        if n_nan > self._limit(data):
            logger.warning("too many NaNs: %d", n_nan)
            return False
        if np.isinf(data).any():
            logger.warning("data contains infinite values")
            return False
        valid = data[np.isfinite(data)]
        if valid.size and (valid.max() - valid.min()) > self.max_range_km_s:
            logger.warning("unusually large velocity range: %.2f km/s",
                           valid.max() - valid.min())
            return False
        return True

    def clean(self, data) -> np.ndarray:
        """Vectorised NaN filling (v1 looped over every NaN pixel)."""
        from scipy import ndimage
        out = np.array(data, dtype=np.float32, copy=True)
        bad = ~np.isfinite(out)
        if bad.any() and (~bad).any():
            filled = ndimage.grey_erosion(np.nan_to_num(out, nan=0.0),
                                          size=(3, 3))
            out[bad] = filled[bad]
        return out


class ParallelProcessor:
    """Kept for API compatibility; v2 no longer needs a worker pool.

    NOTE: on macOS (spawn) a driver script that builds a ``Pool`` **must** use
    an ``if __name__ == '__main__':`` guard, otherwise every worker re-executes
    the whole script and the pool crash-loops.  v2 avoids multiprocessing
    entirely because the batched implementation already meets the target.
    """

    def __init__(self, n_workers: int = 1):
        import multiprocessing as mp
        self.n_workers = max(1, min(int(n_workers), mp.cpu_count()))
        self.pool = None

    def map_function(self, func, iterable):
        return [func(x) for x in iterable]


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        out = func(*args, **kwargs)
        dt = time.time() - t0
        logger.info("%s took %.2f s", func.__name__, dt)
        if isinstance(out, dict) and "metadata" in out:
            out["metadata"]["processing_time"] = dt
        return out
    return wrapper


def create_solar_mask(shape, center, radius, exclusion_ratio: float = 1.0):
    """v1-compatible; the debug ``print`` is gone."""
    return _create_mask(shape, center, radius, exclusion_ratio)


def apply_disk_exclusion(data, center, radius, exclusion_ratio: float = 0.95):
    """v1 signature: ``center`` is ``(cx, cy)``, ``radius`` in pixels."""
    p = _Params(dict(cx=center[0], cy=center[1], radius=radius))
    return _apply_de(data, p, exclusion_ratio)


def calculate_statistics(data, mask=None) -> Dict[str, float]:
    if mask is not None:
        v = data[mask & np.isfinite(data)]
    else:
        v = data[np.isfinite(data)]
    if v.size == 0:
        return dict(mean=0.0, std=0.0, min=0.0, max=0.0, median=0.0, n=0)
    return dict(mean=float(v.mean()), std=float(v.std()), min=float(v.min()),
                max=float(v.max()), median=float(np.median(v)), n=int(v.size))
