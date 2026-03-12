"""
Solar Chromosphere Velocity Processor
A package for processing solar chromospheric velocity data from CHASE mission.
"""

__version__ = "1.0.1"  # Updated version
__author__ = "Solar Physics Team"
__email__ = "solar@example.com"

from .config import ChromosphereConfig, ProcessingConfig, ProcessingMode, VisualizationMode
from .processor import ChromosphereProcessor
from .models import VelocityModels, ModelFitter, ModelParameters
from .utils import (
    SolarCoordinateTransformer,
    ActiveRegionDetector,
    DataValidator,
    ParallelProcessor,
    timeit,
    apply_disk_exclusion,
    create_solar_mask,
    calculate_statistics,
)
from .plotting import VisualizationManager, create_step_plot
from .data_loader import SmartDataLoader, DataType
from .spectral_processor import SpectralProcessor

__all__ = [
    "ChromosphereConfig",
    "ProcessingConfig",
    "ProcessingMode",
    "VisualizationMode",
    "ChromosphereProcessor",
    "VelocityModels",
    "ModelFitter",
    "ModelParameters",
    "SolarCoordinateTransformer",
    "ActiveRegionDetector",
    "DataValidator",
    "ParallelProcessor",
    "VisualizationManager",
    "create_step_plot",
    "SmartDataLoader",
    "DataType",
    "SpectralProcessor",
    "apply_disk_exclusion",
    "create_solar_mask",
    "calculate_statistics",
    "timeit",
]