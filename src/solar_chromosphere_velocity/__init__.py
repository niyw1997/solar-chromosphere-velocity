"""Solar Chromosphere Velocity Processor - v2.

Major changes vs v1.0.1
-----------------------
* one multi-extension FITS per observation time, with the velocity components
  separated into named HDUs (see ``docs/OUTPUT.md``);
* output paths are derived only from ``output_dir`` - the processor never
  writes next to its input files;
* streaming and vectorised: RICE-compressed cubes are read window-by-window
  through ``.section``, geometry and the Doppler inversion are batched, and the
  velocity model is fitted by a single linear least-squares solve;
* complete physics (R_sun / cos B0 scaling and projected meridional
  circulation) with ``legacy=True`` to reproduce v1 exactly.
"""

__version__ = "2.0.0"
__author__ = "Solar Physics Team"

from .config import (
    Config,
    ChromosphereConfig,
    ProcessingMode,
    VisualizationMode,
    config_from_v1,
)
from .constants import (
    GRAVITATIONAL_REDSHIFT_KM_S,
    REST_WAVELENGTH_AIR,
    velocity_scale_km_s_per_pixel,
)
from .data_loader import (
    DataType,
    SolarParams,
    SpectralCube,
    detect_data_type,
    extract_solar_params,
    open_plane,
)
from .geometry import coord_arrays, distance_px, pixel_to_solar, solar_coords
from .linecenter import fit_line_center, wcs_line_center
from .inversion import DopplerInverter, DopplerResult
from .models import (
    MODEL_TERMS,
    ModelFit,
    VelocityModel,
    fit_model_streaming,
    fit_model,
)
from .masks import (
    active_region_mask,
    apply_disk_exclusion,
    create_solar_mask,
    effective_disk_mask,
    effective_radius_px,
)
from .output import (
    LAYERS,
    OutputLayout,
    observation_key,
    write_multihdu,
)
from .processor import ChromosphereProcessor

__all__ = [
    "__version__",
    "Config",
    "ChromosphereConfig",
    "ProcessingMode",
    "VisualizationMode",
    "config_from_v1",
    "GRAVITATIONAL_REDSHIFT_KM_S",
    "REST_WAVELENGTH_AIR",
    "velocity_scale_km_s_per_pixel",
    "DataType",
    "SolarParams",
    "SpectralCube",
    "detect_data_type",
    "extract_solar_params",
    "open_plane",
    "coord_arrays",
    "distance_px",
    "pixel_to_solar",
    "solar_coords",
    "fit_line_center",
    "wcs_line_center",
    "DopplerInverter",
    "DopplerResult",
    "MODEL_TERMS",
    "ModelFit",
    "VelocityModel",
    "fit_model_streaming",
    "fit_model",
    "active_region_mask",
    "apply_disk_exclusion",
    "create_solar_mask",
    "effective_disk_mask",
    "effective_radius_px",
    "LAYERS",
    "OutputLayout",
    "observation_key",
    "write_multihdu",
    "ChromosphereProcessor",
]
