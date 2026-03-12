"""
Utility functions for solar chromosphere velocity processing.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Any, Callable
import multiprocessing as mp
from multiprocessing import Pool
from functools import wraps
import time
import logging
from scipy import ndimage
import warnings
warnings.filterwarnings("ignore")

from .config import ChromosphereConfig

logger = logging.getLogger(__name__)


class SolarCoordinateTransformer:
    """Transformer for solar coordinate systems."""
    
    def __init__(self, config: ChromosphereConfig):
        """
        Initialize coordinate transformer.
        
        Args:
            config: Configuration object
        """
        self.config = config
    
    def pixel_to_solar(self, x: float, y: float, 
                       params: Dict[str, Any]) -> Dict[str, float]:
        """
        Convert pixel coordinates to solar coordinates.
        
        Args:
            x, y: Pixel coordinates
            params: Dictionary containing solar parameters (cx, cy, p0, b0, radius)
            
        Returns:
            Dictionary of solar coordinates
        """
        cx = params["cx"]
        cy = params["cy"]
        p0 = params["p0"]
        b0 = params["b0"]
        radius = params["radius"]
        
        # Relative coordinates
        x_rel = x - cx
        y_rel = y - cy
        
        # Apply instrument rotation (P0)
        p0_rad = np.radians(p0)
        cos_p0, sin_p0 = np.cos(p0_rad), np.sin(p0_rad)
        
        x_rot = x_rel * cos_p0 + y_rel * sin_p0
        y_rot = -x_rel * sin_p0 + y_rel * cos_p0
        
        # Distance from center
        r = np.sqrt(x_rot**2 + y_rot**2)
        
        if r == 0:
            return {
                "latitude": 0.0,
                "longitude": 0.0,
                "radius_angle": 0.0,
                "sin_lat": 0.0,
                "cos_lat": 1.0,
                "sin_2lat": 0.0,
                "sin_lon": 0.0,
                "cos_lon": 1.0,
                "normalized_radius": 0.0  # Add normalized radius
            }
        
        # Angular distance from disk center (simplified)
        # In reality, this should use the solar angular radius
        rho = np.arcsin(r / radius)  # Simplified
        
        # Calculate solar latitude (B)
        costheta = y_rot / r
        b0_rad = np.radians(b0)
        
        sinB = np.sin(b0_rad) * np.cos(rho) + np.cos(b0_rad) * np.sin(rho) * costheta
        B = np.arcsin(sinB)
        
        # Calculate solar longitude (L)
        sintheta = -x_rot / r
        sinL = -np.sin(rho) * sintheta / np.cos(B)
        L = np.arcsin(sinL)
        
        # Calculate normalized radius (0 at center, 1 at edge)
        normalized_radius = r / radius
        
        return {
            "latitude": np.degrees(B),
            "longitude": np.degrees(L),
            "radius_angle": np.degrees(rho),
            "sin_lat": np.sin(B),
            "cos_lat": np.cos(B),
            "sin_2lat": np.sin(2 * B),
            "sin_lon": np.sin(L),
            "cos_lon": np.cos(L),
            "normalized_radius": normalized_radius  # Add normalized radius
        }
    
    def create_coordinate_grid(self, shape: Tuple[int, int],
                              params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Create coordinate grids for an entire image.
        
        Args:
            shape: Image shape (height, width)
            params: Solar parameters
            
        Returns:
            Dictionary of coordinate arrays
        """
        height, width = shape
        
        # Create coordinate arrays
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # Initialize output arrays
        sin_lat = np.zeros(shape, dtype=np.float32)
        cos_lat = np.zeros_like(sin_lat)
        sin_2lat = np.zeros_like(sin_lat)
        sin_lon = np.zeros_like(sin_lat)
        cos_lon = np.zeros_like(sin_lat)
        radius_angle = np.zeros_like(sin_lat)
        normalized_radius = np.zeros_like(sin_lat)  # Add normalized radius
        
        # Transform each pixel
        for i in range(height):
            for j in range(width):
                coords = self.pixel_to_solar(j, i, params)
                
                sin_lat[i, j] = coords["sin_lat"]
                cos_lat[i, j] = coords["cos_lat"]
                sin_2lat[i, j] = coords["sin_2lat"]
                sin_lon[i, j] = coords["sin_lon"]
                cos_lon[i, j] = coords["cos_lon"]
                radius_angle[i, j] = coords["radius_angle"]
                normalized_radius[i, j] = coords["normalized_radius"]
        
        return {
            "sin_lat": sin_lat,
            "cos_lat": cos_lat,
            "sin_2lat": sin_2lat,
            "sin_lon": sin_lon,
            "cos_lon": cos_lon,
            "radius_angle": radius_angle,
            "normalized_radius": normalized_radius
        }


class ActiveRegionDetector:
    """Detector for active regions in solar velocity maps."""
    
    def __init__(self, config: ChromosphereConfig):
        """
        Initialize active region detector.
        
        Args:
            config: Configuration object
        """
        self.config = config
    
    def detect(self, velocity_map: np.ndarray, 
               params: Dict[str, Any]) -> np.ndarray:
        """
        Detect active regions in velocity map.
        
        Args:
            velocity_map: Velocity data
            params: Solar parameters
            
        Returns:
            Boolean mask of active regions
        """
        # Create solar disk mask with exclusion radius
        height, width = velocity_map.shape
        cx, cy = params["cx"], params["cy"]
        radius = params["radius"]
        
        # Apply disk exclusion radius (exclude pixels beyond threshold)
        exclusion_radius = radius * self.config.disk_exclusion_radius
        y, x = np.ogrid[-cy:height-cy, -cx:width-cx]
        distance_squared = x**2 + y**2
        disk_mask = distance_squared <= exclusion_radius**2
        
        # Get valid data within disk (excluding NaNs and edge)
        valid_data = velocity_map[disk_mask & ~np.isnan(velocity_map)]
        
        if len(valid_data) == 0:
            logger.warning("No valid data for active region detection")
            return np.zeros_like(velocity_map, dtype=bool)
        
        # Calculate statistics
        mean_v = np.mean(valid_data)
        std_v = np.std(valid_data)
        
        # Threshold for active regions
        threshold = self.config.activity_threshold_sigma * std_v
        
        # Initial detection within disk
        activity_mask = np.abs(velocity_map - mean_v) > threshold
        activity_mask = activity_mask & disk_mask
        
        # Fill NaN areas
        activity_mask[np.isnan(velocity_map)] = False
        
        # Connected component analysis
        structure = ndimage.generate_binary_structure(2, 2)
        labeled, num_features = ndimage.label(activity_mask, structure=structure)
        
        # Remove small regions
        if num_features > 0:
            sizes = ndimage.sum(activity_mask, labeled, range(num_features + 1))
            
            for i in range(1, num_features + 1):
                if sizes[i] < self.config.min_activity_area:
                    activity_mask[labeled == i] = False
        
        # Morphological operations
        if self.config.erosion_iterations > 0:
            activity_mask = ndimage.binary_erosion(
                activity_mask,
                structure=structure,
                iterations=self.config.erosion_iterations
            )
        
        if self.config.dilation_iterations > 0:
            activity_mask = ndimage.binary_dilation(
                activity_mask,
                structure=structure,
                iterations=self.config.dilation_iterations
            )
        
        logger.info(f"Detected active regions covering {np.sum(activity_mask)} pixels")
        
        return activity_mask
    
    def refine_mask(self, mask: np.ndarray, 
                   velocity_map: np.ndarray) -> np.ndarray:
        """
        Refine active region mask using velocity gradients.
        
        Args:
            mask: Initial active region mask
            velocity_map: Velocity data
            
        Returns:
            Refined mask
        """
        # Calculate gradient magnitude
        grad_y, grad_x = np.gradient(velocity_map)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize gradient
        grad_normalized = (grad_magnitude - np.min(grad_magnitude)) / \
                         (np.max(grad_magnitude) - np.min(grad_magnitude) + 1e-10)
        
        # Enhance mask based on gradients
        enhanced_mask = mask | (grad_normalized > 0.5)
        
        return enhanced_mask


class DataValidator:
    """Validator for solar velocity data."""
    
    def __init__(self, config: ChromosphereConfig):
        """
        Initialize data validator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.nan_threshold = config.nan_threshold
    
    def validate(self, data: np.ndarray) -> bool:
        """
        Validate velocity data.
        
        Args:
            data: Velocity data array
            
        Returns:
            True if data is valid
        """
        # Check for NaN values
        nan_count = np.sum(np.isnan(data))
        if nan_count > self.nan_threshold:
            logger.warning(f"Too many NaN values: {nan_count}")
            return False
        
        # Check for infinite values
        if np.any(np.isinf(data)):
            logger.warning("Data contains infinite values")
            return False
        
        # Check data range (reasonable velocity values for chromosphere)
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            data_range = np.max(valid_data) - np.min(valid_data)
            if data_range > 100:  # km/s, arbitrary threshold
                logger.warning(f"Unusually large velocity range: {data_range:.2f} km/s")
                return False
        
        return True
    
    def clean(self, data: np.ndarray) -> np.ndarray:
        """
        Clean invalid data.
        
        Args:
            data: Input data array
            
        Returns:
            Cleaned data array
        """
        cleaned = data.copy()
        
        # Replace NaN with median of neighbors
        nan_mask = np.isnan(cleaned)
        if np.any(nan_mask):
            from scipy.ndimage import generic_filter
            cleaned = self._fill_nans(cleaned)
        
        # Replace infinite values
        inf_mask = np.isinf(cleaned)
        if np.any(inf_mask):
            median_val = np.median(cleaned[~inf_mask & ~nan_mask])
            cleaned[inf_mask] = median_val
        
        return cleaned
    
    def _fill_nans(self, data: np.ndarray) -> np.ndarray:
        """Fill NaN values using median filtering."""
        from scipy.ndimage import generic_filter
        
        def median_filter(values):
            valid = values[~np.isnan(values)]
            return np.median(valid) if len(valid) > 0 else 0.0
        
        # Apply filter only to NaN regions
        result = data.copy()
        nan_indices = np.where(np.isnan(data))
        
        for i, j in zip(nan_indices[0], nan_indices[1]):
            # Get 3x3 neighborhood
            i_min = max(0, i - 1)
            i_max = min(data.shape[0], i + 2)
            j_min = max(0, j - 1)
            j_max = min(data.shape[1], j + 2)
            
            neighborhood = data[i_min:i_max, j_min:j_max]
            result[i, j] = np.median(neighborhood[~np.isnan(neighborhood)])
        
        return result


class ParallelProcessor:
    """Parallel processing utilities."""
    
    def __init__(self, n_workers: int):
        """
        Initialize parallel processor.
        
        Args:
            n_workers: Number of worker processes
        """
        self.n_workers = min(n_workers, mp.cpu_count())
        self.pool = None
    
    def __enter__(self):
        """Enter context manager."""
        self.pool = Pool(self.n_workers)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if self.pool:
            self.pool.close()
            self.pool.join()
    
    def process_chunks(self, chunks: List[Any], 
                      process_func: Callable) -> List[Any]:
        """
        Process data chunks in parallel.
        
        Args:
            chunks: List of data chunks
            process_func: Processing function
            
        Returns:
            List of processed chunks
        """
        if self.pool is None:
            self.pool = Pool(self.n_workers)
        
        try:
            results = list(self.pool.map(process_func, chunks))
            return results
        except Exception as e:
            logger.error(f"Parallel processing failed: {e}")
            raise
        finally:
            self.pool.close()
            self.pool.join()
            self.pool = None
    
    def map_function(self, func: Callable, iterable: List[Any]) -> List[Any]:
        """
        Map function over iterable in parallel.
        
        Args:
            func: Function to apply
            iterable: Input data
            
        Returns:
            List of results
        """
        return self.process_chunks(iterable, func)


def timeit(func: Callable) -> Callable:
    """
    Decorator to measure execution time.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        
        # Store timing in result if it's a dictionary
        if isinstance(result, dict) and "metadata" in result:
            result["metadata"]["processing_time"] = end_time - start_time
        
        return result
    return wrapper


def create_solar_mask(shape: Tuple[int, int], 
                     center: Tuple[float, float],
                     radius: float,
                     exclusion_ratio: float = 1.0) -> np.ndarray:
    """
    Create solar disk mask with optional exclusion radius.
    
    Args:
        shape: Image shape (height, width)
        center: Disk center (cx, cy)
        radius: Disk radius
        exclusion_ratio: Radius exclusion ratio (0-1, default: 1.0 = no exclusion)
        
    Returns:
        Boolean mask (True for valid regions)
    """
    height, width = shape
    cx, cy = center
    
    # Calculate effective radius
    effective_radius = radius * exclusion_ratio
    print(f"radius: {radius}, exclusion_ratio: {exclusion_ratio}")
    print(f'effective_radius: {effective_radius}')
    
    y, x = np.ogrid[-cy:height-cy, -cx:width-cx]
    mask = x**2 + y**2 <= effective_radius**2
    
    return mask


def apply_disk_exclusion(data: np.ndarray,
                        center: Tuple[float, float],
                        radius: float,
                        exclusion_ratio: float = 0.95) -> np.ndarray:
    """
    Apply disk exclusion mask to data.
    
    Args:
        data: Input data array
        center: Disk center (cx, cy)
        radius: Disk radius
        exclusion_ratio: Radius exclusion ratio (0-1, default: 0.95)
        
    Returns:
        Data with excluded regions set to NaN
    """
    if exclusion_ratio >= 1.0:
        return data
    
    result = data.copy()
    mask = create_solar_mask(data.shape, center, radius, exclusion_ratio)
    
    # Set values outside mask to NaN
    result[~mask] = np.nan
    
    logger.info(f"Applied disk exclusion: {np.sum(~mask)} pixels set to NaN "
               f"(exclusion ratio: {exclusion_ratio})")
    
    return result


def calculate_statistics(data: np.ndarray, 
                        mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate statistics for velocity data.
    
    Args:
        data: Velocity data
        mask: Optional mask for valid data
        
    Returns:
        Dictionary of statistics
    """
    if mask is not None:
        valid_data = data[mask & ~np.isnan(data)]
    else:
        valid_data = data[~np.isnan(data)]
    
    if len(valid_data) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0
        }
    
    return {
        "mean": float(np.mean(valid_data)),
        "std": float(np.std(valid_data)),
        "min": float(np.min(valid_data)),
        "max": float(np.max(valid_data)),
        "median": float(np.median(valid_data))
    }