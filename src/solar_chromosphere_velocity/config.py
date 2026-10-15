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
    exclude_patterns: Tuple[str, ...] = ("*_Vdop*", "*_residual*", "*_scv*")
    overwrite: bool = False
    no_day_subdirectory: bool = False
    write_observation_log: bool = True
    rice_compression: bool = False
    require_velocity_unit: bool = True     # refuse 2-D inputs without a velocity BUNIT

    # --- physics / model --------------------------------------------------
    legacy: bool = False
    rotation_amplitude: str = "kms"        # 'kms' | 'rad_per_s'
    limb_exponent: Optional[float] = None         # None -> scan
    limb_exponent_scan: Sequence[float] = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
    limb_amplitude_positive: bool = True
    zero_point_km_s: float = 0.0

    # --- Doppler inversion ------------------------------------------------
    correlation_half_width: int = 10
    reference_profile_radius: float = 100.0
    continuum_subtraction: Optional[bool] = None   # None -> not legacy
    limit_subpixel_shift: Optional[bool] = None   # None -> not legacy
    min_spectral_samples: int = 90          # legacy fallback threshold

    # --- masks ------------------------------------------------------------
    disk_mask_radius: float = 0.95
    edge_exclusion_angle: float = 80.0
    active_region_sigma_threshold: float = 3.0
    active_region_robust_sigma: Optional[bool] = None         # None -> not legacy
    min_active_region_area: int = 100
    active_region_grow_steps: int = 3
    active_region_shrink_steps: int = 1

    # --- performance ------------------------------------------------------
    # 32 rows/block measured fastest AND leanest on a 2313x2304 RICE cube
    # (8.1 s inversion, 0.94 GB peak) versus 64/128/256/512 rows.
    spatial_strip_rows: int = 32
    parallel_processes: int = 1
    parallel_processing: bool = False

    # --- plots / logging --------------------------------------------------
    visualize: bool = False
    plot_dpi: int = 150
    plot_format: str = "png"
    colormap: str = "bwr"
    plot_per_component: bool = False       # also write one figure per component
    plot_panels: Optional[Tuple[str, ...]] = None   # None -> all present layers
    plot_overlay_mask: bool = False        # contour the active regions
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # ------------------------------------------------------------------ #
    def __post_init__(self):
        if self.rotation_amplitude not in ("kms", "rad_per_s"):
            raise ValueError("rotation_amplitude must be 'kms' or 'rad_per_s'")
        if not (0.0 < self.disk_mask_radius <= 1.0):
            raise ValueError("disk_mask_radius must be in (0, 1]")
        if not (0.0 < self.edge_exclusion_angle < 90.0):
            raise ValueError("edge_exclusion_angle must be in (0, 90)")
        if self.correlation_half_width < 2:
            raise ValueError("correlation_half_width must be >= 2")
        if self.spatial_strip_rows < 1:
            raise ValueError("spatial_strip_rows must be >= 1")
        if self.parallel_processes < 1:
            raise ValueError("parallel_processes must be >= 1")
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
        return (not self.legacy) if self.continuum_subtraction is None else bool(self.continuum_subtraction)

    @property
    def limit_subpixel_shift_on(self) -> bool:
        return (not self.legacy) if self.limit_subpixel_shift is None else bool(self.limit_subpixel_shift)

    @property
    def active_region_robust_threshold(self) -> bool:
        return (not self.legacy) if self.active_region_robust_sigma is None else bool(self.active_region_robust_sigma)

    @property
    def zero_point(self) -> float:
        return LEGACY_ZERO_POINT_KM_S if self.legacy else float(self.zero_point_km_s)

    @property
    def limb_exponent_values(self) -> Tuple[float, ...]:
        if self.limb_exponent is not None:
            return (float(self.limb_exponent),)
        return tuple(float(x) for x in self.limb_exponent_scan)

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
    "use_multiprocessing": ("parallel_processing", None),
    "n_workers": ("parallel_processes", None),
    "activity_threshold_sigma": ("active_region_sigma_threshold", None),
    "min_activity_area": ("min_active_region_area", None),
    "edge_exclusion_angle": ("edge_exclusion_angle", None),
    "disk_exclusion_radius": ("disk_mask_radius", None),
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
