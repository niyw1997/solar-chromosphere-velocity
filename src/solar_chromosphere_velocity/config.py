"""
Configuration module for solar chromosphere velocity processing.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum


class ProcessingMode(Enum):
    """Processing modes for velocity data."""
    SINGLE_FILE = "single"
    BATCH = "batch"
    INTERACTIVE = "interactive"


class VisualizationMode(Enum):
    """Visualization modes."""
    NONE = "none"
    BASIC = "basic"
    DETAILED = "detailed"
    ALL = "all"


@dataclass
class ChromosphereConfig:
    """
    Configuration for chromosphere velocity processing.
    
    Attributes:
        input_dir: Input directory containing FITS files
        output_dir: Output directory for processed data
        visualize_steps: Whether to generate visualizations at each step
        visualization_mode: Level of visualization detail
        save_intermediate: Whether to save intermediate results
        n_workers: Number of parallel workers
        use_multiprocessing: Whether to use multiprocessing
        activity_threshold_sigma: Threshold for active region detection (in sigma)
        min_activity_area: Minimum area for active regions (in pixels)
        edge_exclusion_angle: Angle from disk center to exclude (in degrees)
        disk_exclusion_radius: Radius threshold for disk exclusion (0-1, default: 0.95)
        max_fitting_iterations: Maximum iterations for model fitting
        gravitational_redshift: Gravitational redshift value (m/s)
        solar_radius_km: Solar radius in kilometers
        instrument_rotation: Default instrument rotation angle (degrees)
        spectral_processing: Enable spectral data preprocessing
        spectral_line_type: Spectral line type (HA or FE)
        doppler_inversion_method: Method for Doppler inversion
        reference_profile_path: Path to reference spectral profile
        log_level: Logging level
    """
    
    # File paths
    input_dir: str = ""
    output_dir: str = ""
    
    # Processing options
    processing_mode: ProcessingMode = ProcessingMode.BATCH
    save_intermediate: bool = True
    overwrite_existing: bool = False
    
    # Visualization options
    visualize_steps: bool = False
    visualization_mode: VisualizationMode = VisualizationMode.BASIC
    plot_dpi: int = 150
    plot_format: str = "png"
    colormap: str = "bwr"
    
    # Performance options
    n_workers: int = 4
    use_multiprocessing: bool = True
    chunk_size: int = 1000
    
    # Active region detection
    activity_threshold_sigma: float = 3.0
    min_activity_area: int = 100
    dilation_iterations: int = 3
    erosion_iterations: int = 1
    
    # Data processing
    edge_exclusion_angle: float = 80.0
    disk_exclusion_radius: float = 0.95  # New parameter: exclude > 0.95 solar radius
    nan_threshold: float = 1e6
    normalize_data: bool = True
    
    # Model fitting
    max_fitting_iterations: int = 10000
    fitting_tolerance: float = 1e-6
    use_bounds: bool = True
    
    # Physical constants
    gravitational_redshift: float = 632.0  # m/s
    solar_radius_km: float = 6.963e5
    solar_radius_arcsec: float = 960.0
    
    # Instrument parameters
    instrument_rotation: float = 0.0
    pixel_scale_arcsec: float = 0.5218 * 2 / 60  # CHASE pixel scale
    
    # Spectral processing
    spectral_processing: bool = True
    spectral_line_type: str = "auto"  # auto, HA, FE
    doppler_inversion_method: str = "cross_correlation"  # cross_correlation, gaussian_fit
    reference_profile_path: Optional[str] = None
    wavelength_calibration: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Advanced options
    enable_validation: bool = True
    enable_debug_output: bool = False
    memory_limit_gb: float = 4.0
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_paths()
        self._validate_values()
    
    def _validate_paths(self):
        """Validate input and output paths."""
        if self.input_dir and not os.path.exists(self.input_dir):
            raise ValueError(f"Input directory does not exist: {self.input_dir}")
        
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        
        if self.reference_profile_path and not os.path.exists(self.reference_profile_path):
            raise ValueError(f"Reference profile path does not exist: {self.reference_profile_path}")
    
    def _validate_values(self):
        """Validate configuration values."""
        if self.n_workers < 1:
            raise ValueError("Number of workers must be at least 1")
        
        if self.activity_threshold_sigma <= 0:
            raise ValueError("Activity threshold must be positive")
        
        if self.edge_exclusion_angle <= 0 or self.edge_exclusion_angle >= 90:
            raise ValueError("Edge exclusion angle must be between 0 and 90 degrees")
        
        if self.disk_exclusion_radius <= 0 or self.disk_exclusion_radius > 1:
            raise ValueError("Disk exclusion radius must be between 0 and 1")
        
        if self.spectral_line_type not in ["auto", "HA", "FE"]:
            raise ValueError(f"Invalid spectral line type: {self.spectral_line_type}")
    
    def update(self, **kwargs):
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Invalid configuration key: {key}")
        
        self.__post_init__()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ChromosphereConfig":
        """Create configuration from dictionary."""
        # Handle Enum values
        for key in ["processing_mode", "visualization_mode"]:
            if key in config_dict and isinstance(config_dict[key], str):
                enum_class = ProcessingMode if key == "processing_mode" else VisualizationMode
                config_dict[key] = enum_class(config_dict[key])
        
        return cls(**config_dict)


@dataclass
class ProcessingConfig:
    """
    Runtime processing configuration.
    
    Attributes:
        current_file: Currently processed file
        step_number: Current processing step
        total_steps: Total number of steps
        progress_callback: Callback for progress updates
    """
    
    current_file: str = ""
    step_number: int = 0
    total_steps: int = 10
    progress_callback: Optional[callable] = None
    visualization_callback: Optional[callable] = None
    
    def update_progress(self, step: int, message: str = ""):
        """Update processing progress."""
        self.step_number = step
        if self.progress_callback:
            self.progress_callback(step, self.total_steps, message)
    
    def trigger_visualization(self, step_name: str, data: Any, metadata: Dict[str, Any]):
        """Trigger visualization for a processing step."""
        if self.visualization_callback:
            self.visualization_callback(step_name, data, metadata)