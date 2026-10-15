"""Configuration for the v2 processor.

The single most important knob is :attr:`Config.legacy`:

* ``legacy=False`` (default) - complete physics (rotation amplitude scaled by
  R_sun and cos B0, projected meridional circulation), header-driven velocity
  calibration, WCS-based line centre, mean-subtracted cross-correlation,
  clipped sub-pixel interpolation, MAD-based active-region detection.
* ``legacy=True`` - reproduce v1.0.1 behaviour for regression.
"""

from dataclasses import dataclass, fields
from enum import Enum
from typing import Optional, Sequence, Tuple
import os

from .constants import LEGACY_ZERO_POINT_KM_S

__all__ = ["Config", "ChromosphereConfig", "ProcessingMode", "VisualizationMode",
           "config_from_v1"]


class ProcessingMode(Enum):
    SINGLE_FILE = "single"
    BATCH = "batch"


class VisualizationMode(Enum):
    NONE = "none"
    BASIC = "basic"
    ALL = "all"


@dataclass
class Config:
    """v2 processing configuration."""

    # --- input / output ---------------------------------------------------
    input_dir: str = ""
    output_dir: str = ""
    pattern: str = "*.fits"
    exclude_globs: Tuple[str, ...] = ("*_Vdop*", "*_residual*", "*_scv*")
    overwrite: bool = False
    flat_output: bool = False
    write_index: bool = True
    compress_output: bool = False

    # --- physics / model --------------------------------------------------
    legacy: bool = False
    rotation_amplitude: str = "kms"        # 'kms' | 'rad_per_s'
    n_limb: Optional[float] = None         # None -> scan
    n_limb_scan: Sequence[float] = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
    limb_nonneg: bool = True
    zero_point_km_s: float = 0.0

    # --- Doppler inversion ------------------------------------------------
    window_half: int = 10
    profile_mask_radius: float = 100.0
    demean_window: Optional[bool] = None   # None -> not legacy
    clip_subpixel: Optional[bool] = None   # None -> not legacy
    line_center_min_len: int = 90          # legacy fallback threshold

    # --- masks ------------------------------------------------------------
    disk_exclusion_radius: float = 0.95
    edge_exclusion_angle: float = 80.0
    activity_threshold_sigma: float = 3.0
    use_mad: Optional[bool] = None         # None -> not legacy
    min_activity_area: int = 100
    dilation_iterations: int = 3
    erosion_iterations: int = 1

    # --- performance ------------------------------------------------------
    # 32 rows/block measured fastest AND leanest on a 2313x2304 RICE cube
    # (8.1 s inversion, 0.94 GB peak) versus 64/128/256/512 rows.
    row_block: int = 32
    n_workers: int = 1
    use_multiprocessing: bool = False

    # --- plots / logging --------------------------------------------------
    visualize: bool = False
    plot_dpi: int = 150
    plot_format: str = "png"
    colormap: str = "bwr"
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # ------------------------------------------------------------------ #
    def __post_init__(self):
        if self.rotation_amplitude not in ("kms", "rad_per_s"):
            raise ValueError("rotation_amplitude must be 'kms' or 'rad_per_s'")
        if not (0.0 < self.disk_exclusion_radius <= 1.0):
            raise ValueError("disk_exclusion_radius must be in (0, 1]")
        if not (0.0 < self.edge_exclusion_angle < 90.0):
            raise ValueError("edge_exclusion_angle must be in (0, 90)")
        if self.window_half < 2:
            raise ValueError("window_half must be >= 2")
        if self.row_block < 1:
            raise ValueError("row_block must be >= 1")
        if self.n_workers < 1:
            raise ValueError("n_workers must be >= 1")
        if self.input_dir and not os.path.isdir(self.input_dir):
            raise ValueError("input_dir does not exist: %s" % self.input_dir)
        if self.legacy and self.zero_point_km_s != 0.0:
            raise ValueError("--legacy pins the zero point to %.3f km/s; "
                             "do not also set zero_point_km_s"
                             % LEGACY_ZERO_POINT_KM_S)

    # --- resolved (legacy-aware) values -----------------------------------
    @property
    def model_version(self) -> str:
        return "legacy-1.0" if self.legacy else "full-2.0"

    @property
    def demean(self) -> bool:
        return (not self.legacy) if self.demean_window is None else bool(self.demean_window)

    @property
    def clip_subpixel_on(self) -> bool:
        return (not self.legacy) if self.clip_subpixel is None else bool(self.clip_subpixel)

    @property
    def mad_threshold(self) -> bool:
        return (not self.legacy) if self.use_mad is None else bool(self.use_mad)

    @property
    def zero_point(self) -> float:
        return LEGACY_ZERO_POINT_KM_S if self.legacy else float(self.zero_point_km_s)

    @property
    def n_limb_values(self) -> Tuple[float, ...]:
        if self.n_limb is not None:
            return (float(self.n_limb),)
        return tuple(float(x) for x in self.n_limb_scan)

    @property
    def use_rotation_radius(self) -> bool:
        """True when A,B,C,D are reported as angular rates (rad/s)."""
        return self.rotation_amplitude == "rad_per_s"

    # v1 name kept so existing code that reads config.visualize_steps works.
    @property
    def visualize_steps(self) -> bool:
        return bool(self.visualize)

    # --- helpers ----------------------------------------------------------
    def update(self, **kwargs):
        names = {f.name for f in fields(self)}
        unknown = set(kwargs) - names
        if unknown:
            raise AttributeError("unknown configuration keys: %s" % sorted(unknown))
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.__post_init__()

    def to_dict(self):
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = list(v) if isinstance(v, (tuple, list)) else v
        return out


# v1 name kept for import compatibility (a major version changes the fields).
ChromosphereConfig = Config


# --- v1 configuration bridge ---------------------------------------------
_V1_MAP = {
    "visualize_steps": ("visualize", None),
    "gravitational_redshift": ("zero_point_km_s", lambda v: float(v) / 1000.0),
    "overwrite_existing": ("overwrite", None),
    "use_multiprocessing": ("use_multiprocessing", None),
    "n_workers": ("n_workers", None),
    "activity_threshold_sigma": ("activity_threshold_sigma", None),
    "min_activity_area": ("min_activity_area", None),
    "edge_exclusion_angle": ("edge_exclusion_angle", None),
    "disk_exclusion_radius": ("disk_exclusion_radius", None),
    "plot_dpi": ("plot_dpi", None),
    "plot_format": ("plot_format", None),
    "colormap": ("colormap", None),
    "log_level": ("log_level", None),
    "log_file": ("log_file", None),
    "input_dir": ("input_dir", None),
    "output_dir": ("output_dir", None),
    "overwrite": ("overwrite", None),
    "legacy": ("legacy", None),
}


def config_from_v1(**kwargs) -> Config:
    """Build a v2 :class:`Config` from v1 ``ChromosphereConfig`` keyword names.

    Used by the legacy ``process-chase`` entry point.  Unknown v1 keys are
    collected and attached so that silent behaviour changes cannot happen.
    """
    mapped, ignored = {}, []
    for k, v in kwargs.items():
        if k in _V1_MAP:
            name, fn = _V1_MAP[k]
            mapped[name] = fn(v) if fn else v
        else:
            ignored.append(k)
    cfg = Config(**mapped)
    cfg.ignored_v1_keys = tuple(sorted(ignored))  # type: ignore[attr-defined]
    return cfg
