"""
Main processor module for solar chromosphere velocity data processing.
"""

from functools import partial
import os
from turtle import shape
from turtle import shape
import numpy as np
from astropy.io import fits
import sunpy.map
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from .config import ChromosphereConfig, ProcessingConfig, VisualizationMode
from .utils import (
    SolarCoordinateTransformer,
    ActiveRegionDetector,
    DataValidator,
    ParallelProcessor,
    timeit,
    apply_disk_exclusion,
    create_solar_mask
)
from .models import VelocityModels, ModelFitter
from .plotting import VisualizationManager
from .data_loader import SmartDataLoader, DataType
from .spectral_processor import SpectralProcessor

def _reconstruct_chunk_worker(chunk_indices, model_params, coord_transformer, params):
    """Worker function for parallel reconstruction of velocity field."""
    import numpy as np
    from .models import VelocityModels  # 注意相对导入
    
    chunk_result = np.zeros(len(chunk_indices), dtype=np.float32)
    for idx, (i, j) in enumerate(chunk_indices):
        coords = coord_transformer.pixel_to_solar(j, i, params)
        coord_tuple = (
            coords["sin_lat"],
            coords["sin_2lat"],
            coords["cos_lat"],
            coords["sin_lon"],
            coords["cos_lon"],
            coords["radius_angle"]
        )
        chunk_result[idx] = VelocityModels.chromosphere_full_model(
            coord_tuple, *model_params
        )
    return chunk_result


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromosphereProcessor:
    """
    Main processor for chromosphere velocity data.
    
    This class handles the complete processing pipeline including
    spectral data preprocessing, active region detection, and
    large-scale flow removal.
    """
    
    def __init__(self, config: ChromosphereConfig):
        """
        Initialize the processor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.setup_logging()
        
        # Initialize components
        self.data_loader = SmartDataLoader(config)
        self.coord_transformer = SolarCoordinateTransformer(config)
        self.active_region_detector = ActiveRegionDetector(config)
        self.data_validator = DataValidator(config)
        self.visualization_manager = VisualizationManager(config)
        self.model_fitter = ModelFitter(config)
        self.spectral_processor = SpectralProcessor(config)
        
        # Initialize parallel processor if enabled
        if config.use_multiprocessing:
            self.parallel_processor = ParallelProcessor(config.n_workers)
        else:
            self.parallel_processor = None
        
        logger.info(f"Initialized ChromosphereProcessor with configuration")
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.log_level.upper())
        logger.setLevel(log_level)
        
        if self.config.log_file:
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setLevel(log_level)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    @timeit
    def process_file(self, filepath: str, 
                     processing_config: Optional[ProcessingConfig] = None) -> Dict[str, Any]:
        """
        Process a single file with automatic preprocessing.
        
        Args:
            filepath: Path to the input file
            processing_config: Runtime processing configuration
            
        Returns:
            Dictionary containing processing results
        """
        logger.info(f"Processing file: {filepath}")
        
        if processing_config:
            processing_config.current_file = os.path.basename(filepath)
            processing_config.update_progress(0, "Detecting data type")
        
        # Step 1: Detect data type and load data
        if processing_config:
            processing_config.update_progress(1, "Loading data")
        
        data, header, data_type, params = self.data_loader.load_data(filepath)
        
        # Step 2: Apply disk exclusion if not already done in preprocessing
        if processing_config:
            processing_config.update_progress(2, "Applying disk exclusion")
        
        # Apply disk exclusion to raw data (if not spectral preprocessing)
        if self.config.disk_exclusion_radius < 1.0 and not self.data_loader.needs_preprocessing(data_type):
            logger.info(f"Applying disk exclusion (radius > {self.config.disk_exclusion_radius})")
            center = (params.get('cx', data.shape[1] // 2),
                     params.get('cy', data.shape[0] // 2))
            radius = params.get('radius', min(data.shape) // 2)
            data = apply_disk_exclusion(
                data, center, radius, self.config.disk_exclusion_radius
            )
        
        # Step 3: Check if preprocessing is needed
        if self.config.spectral_processing and self.data_loader.needs_preprocessing(data_type):
            logger.info(f"Preprocessing required for {data_type.value}")
            
            if processing_config:
                processing_config.update_progress(3, "Doppler velocity inversion")
            
            # Perform Doppler velocity inversion
            doppler_map, updated_header = self.spectral_processor.compute_doppler_velocity(
                data, header, data_type, params
            )
            
            # Generate output filename
            output_suffix = "_HA_Vdop" if data_type == DataType.SPECTRAL_HA else "_FE_Vdop"
            output_path = self.data_loader.generate_output_filename(filepath, output_suffix)
            
            # Save preprocessed data
            self.spectral_processor.save_doppler_data(
                doppler_map, updated_header, output_path
            )
            
            # Update filepath and reload data
            filepath = output_path
            
            if processing_config:
                processing_config.trigger_visualization(
                    "preprocessed_doppler", doppler_map,
                    {"title": "Preprocessed Doppler Velocity", "filepath": filepath}
                )
            
            # Reload preprocessed data
            data, header, data_type, params = self.data_loader.load_data(filepath)
        
        # Step 4: Validate data
        if processing_config:
            processing_config.update_progress(4, "Validating data")
        
        if not self.data_validator.validate(data):
            logger.warning(f"Data validation failed for {filepath}")
            data = self.data_validator.clean(data)
        
        if self.config.visualize_steps:
            self.visualization_manager.plot_step(
                "raw_velocity", data, 
                {"title": "Raw/Preprocessed Velocity Map", "filepath": filepath}
            )
        
        if processing_config:
            processing_config.trigger_visualization("validated_data", data, 
                                                   {"title": "Validated Data"})
        
        # Step 5: Remove gravitational redshift
        if processing_config:
            processing_config.update_progress(5, "Removing gravitational redshift")
        
        data_corrected = data - self.config.gravitational_redshift / 1000.0  # Convert to km/s
        
        if processing_config:
            processing_config.trigger_visualization("redshift_corrected", data_corrected,
                                                   {"title": "Gravitational Redshift Corrected"})
        
        # Step 6: Detect active regions (automatically excludes > 0.95 radius)
        if processing_config:
            processing_config.update_progress(6, "Detecting active regions")
        
        active_region_mask = self.active_region_detector.detect(data_corrected, params)
        
        if self.config.visualize_steps:
            self.visualization_manager.plot_step(
                "active_region_mask", active_region_mask,
                {"title": "Active Region Mask", "cmap": "Reds"}
            )
        
        if processing_config:
            processing_config.trigger_visualization("active_regions", active_region_mask,
                                                   {"title": "Detected Active Regions"})
        
        # Step 7: Prepare fitting data (exclude active regions and edge)
        if processing_config:
            processing_config.update_progress(7, "Preparing fitting data")
        
        fitting_data = self.prepare_fitting_data(data_corrected, active_region_mask, params)
        
        # Check if we have enough data for fitting
        if fitting_data["coordinates"] is None or len(fitting_data["velocity"]) < 100:
            logger.warning(f"Insufficient valid data points for fitting: {filepath}")
            return self._create_minimal_results(data, data_corrected, active_region_mask, 
                                               header, params, filepath)
        
        if processing_config:
            processing_config.trigger_visualization("fitting_data", fitting_data["velocity"],
                                                   {"title": "Data for Fitting (Active Regions Excluded)"})
        
        # Step 8: Fit large-scale velocity model
        if processing_config:
            processing_config.update_progress(8, "Fitting large-scale model")
        
        model_params, model_errors = self.model_fitter.fit(fitting_data)
        
        logger.info(f"Fitted model parameters: {model_params}")
        logger.info(f"Parameter errors: {model_errors}")
        
        if processing_config:
            processing_config.trigger_visualization("model_parameters", 
                                                   {"params": model_params, "errors": model_errors},
                                                   {"title": "Fitted Model Parameters"})
        
        # Step 9: Reconstruct and subtract large-scale flow
        if processing_config:
            processing_config.update_progress(9, "Reconstructing large-scale flow")
        
        large_scale_flow = self.reconstruct_velocity_field(model_params, data_corrected.shape, params)
        
        if self.config.visualize_steps:
            self.visualization_manager.plot_step(
                "large_scale_flow", large_scale_flow,
                {"title": "Fitted Large-Scale Flow", "vmin": -3, "vmax": 3}
            )
        
        if processing_config:
            processing_config.trigger_visualization("large_scale_flow", large_scale_flow,
                                                   {"title": "Large-Scale Flow"})
        
        # Step 10: Subtract large-scale flow
        if processing_config:
            processing_config.update_progress(10, "Subtracting large-scale flow")
        
        residual = data_corrected - large_scale_flow
        
        if processing_config:
            processing_config.trigger_visualization("raw_residual", residual,
                                                   {"title": "Raw Residual"})
        
        # Step 11: Normalize residual
        if processing_config:
            processing_config.update_progress(11, "Normalizing residual")
        
        residual_normalized = self.normalize_residual(residual, params)
        
        if self.config.visualize_steps:
            self.visualization_manager.plot_step(
                "final_residual", residual_normalized,
                {"title": "Final Residual Velocity", "vmin": -3, "vmax": 3}
            )
        
        if processing_config:
            processing_config.trigger_visualization("final_residual", residual_normalized,
                                                   {"title": "Final Residual Velocity"})
        
        # Step 12: Prepare results
        if processing_config:
            processing_config.update_progress(12, "Preparing results")
        
        results = {
            "filepath": filepath,
            "original_data": data,
            "corrected_data": data_corrected,
            "active_region_mask": active_region_mask,
            "fitted_params": model_params,
            "param_errors": model_errors,
            "large_scale_flow": large_scale_flow,
            "residual": residual_normalized,
            "header": header,
            "params": params,
            "metadata": {
                "processing_time": None,
                "config": self.config.to_dict(),
                "data_type": data_type.value,
                "preprocessed": self.data_loader.needs_preprocessing(data_type),
                "disk_exclusion_radius": self.config.disk_exclusion_radius
            }
        }
        
        # Step 13: Save results
        if processing_config:
            processing_config.update_progress(13, "Saving results")
        
        if self.config.save_intermediate or self.config.output_dir:
            self.save_results(results, filepath)
        
        logger.info(f"Completed processing: {filepath}")
        
        return results
    
    def _create_minimal_results(self, data: np.ndarray, data_corrected: np.ndarray,
                               active_region_mask: np.ndarray, header: fits.Header,
                               params: Dict[str, Any], filepath: str) -> Dict[str, Any]:
        """Create minimal results when fitting is not possible."""
        return {
            "filepath": filepath,
            "original_data": data,
            "corrected_data": data_corrected,
            "active_region_mask": active_region_mask,
            "residual": data_corrected,
            "header": header,
            "params": params,
            "metadata": {
                "processing_time": None,
                "config": self.config.to_dict(),
                "data_type": "unknown",
                "preprocessed": False,
                "fitting_failed": True,
                "disk_exclusion_radius": self.config.disk_exclusion_radius
            }
        }
    
    def prepare_fitting_data(self, data: np.ndarray, 
                            exclude_mask: np.ndarray,
                            params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Prepare data for model fitting, excluding active regions, edge, and outer disk.
        
        Args:
            data: Velocity data
            exclude_mask: Mask for regions to exclude (active regions)
            params: Solar parameters
            
        Returns:
            Dictionary containing coordinates and velocities for fitting
        """
        height, width = data.shape
        cx, cy = params["cx"], params["cy"]
        radius = params["radius"]
        
        # Create edge mask (based on edge_exclusion_angle)
        edge_radius = radius * np.sin(np.radians(self.config.edge_exclusion_angle))
        y, x = np.ogrid[-cy:height-cy, -cx:width-cx]
        distance_squared = x**2 + y**2
        edge_mask = distance_squared <= edge_radius**2
        
        # Create disk exclusion mask (based on disk_exclusion_radius)
        disk_mask = distance_squared <= (radius * self.config.disk_exclusion_radius)**2
        
        # Combine all masks: within edge, within disk, not active region, not NaN
        valid_mask = edge_mask & disk_mask & (~exclude_mask) & (~np.isnan(data))
        
        # Extract valid data points
        valid_indices = np.where(valid_mask)
        
        if len(valid_indices[0]) == 0:
            logger.warning("No valid data points for fitting")
            return {"coordinates": None, "velocity": None}
        
        # Prepare coordinates for valid points
        coordinates = []
        velocities = []
        
        for i, j in zip(valid_indices[0], valid_indices[1]):
            coords = self.coord_transformer.pixel_to_solar(j, i, params)
            coordinates.append([
                coords["sin_lat"],
                coords["sin_2lat"],
                coords["cos_lat"],
                coords["sin_lon"],
                coords["cos_lon"],
                coords["radius_angle"]
            ])
            velocities.append(data[i, j])
        
        logger.info(f"Prepared {len(velocities)} data points for model fitting")
        
        return {
            "coordinates": np.array(coordinates).T,  # Shape: (6, n_points)
            "velocity": np.array(velocities)
        }
    
    def reconstruct_velocity_field(self, model_params: np.ndarray,
                                  shape: Tuple[int, int],
                                  params: Dict[str, Any]) -> np.ndarray:
        """
        Reconstruct velocity field from model parameters.
        
        Args:
            model_params: Fitted model parameters
            shape: Output array shape
            params: Solar parameters
            
        Returns:
            Reconstructed velocity field
        """
        height, width = shape
        reconstructed = np.zeros(shape, dtype=np.float32)
        
        # Only compute for points within edge exclusion and disk exclusion
        cx, cy = params["cx"], params["cy"]
        radius = params["radius"]
        edge_radius = radius * np.sin(np.radians(self.config.edge_exclusion_angle))
        disk_radius = radius * self.config.disk_exclusion_radius
        
        y, x = np.ogrid[-cy:height-cy, -cx:width-cx]
        distance_squared = x**2 + y**2
        
        # Valid mask includes both edge and disk constraints
        valid_mask = (distance_squared <= edge_radius**2) & (distance_squared <= disk_radius**2)
        
        # Use parallel processing for large arrays
        if self.parallel_processor and np.sum(valid_mask) > 10000:
            return self._reconstruct_parallel(model_params, shape, params, valid_mask)
        
        # Sequential reconstruction
        valid_indices = np.where(valid_mask)
        
        for i, j in zip(valid_indices[0], valid_indices[1]):
            coords = self.coord_transformer.pixel_to_solar(j, i, params)
            
            # Prepare coordinate tuple for model
            coord_tuple = (
                coords["sin_lat"],
                coords["sin_2lat"],
                coords["cos_lat"],
                coords["sin_lon"],
                coords["cos_lon"],
                coords["radius_angle"]
            )
            
            # Evaluate model
            reconstructed[i, j] = VelocityModels.chromosphere_full_model(
                coord_tuple, *model_params
            )
        
        return reconstructed
    
    def _reconstruct_parallel(self, model_params: np.ndarray,
                             shape: Tuple[int, int],
                             params: Dict[str, Any],
                             valid_mask: np.ndarray) -> np.ndarray:
        """Parallel reconstruction of velocity field."""
        from functools import partial
        
        valid_indices = list(zip(*np.where(valid_mask)))
    
        chunk_size = self.config.chunk_size
        chunks = [valid_indices[i:i+chunk_size]
                  for i in range(0, len(valid_indices), chunk_size)]

        worker_func = partial(
            _reconstruct_chunk_worker,
            model_params=model_params,
            coord_transformer=self.coord_transformer,
            params=params
        )

        results = self.parallel_processor.process_chunks(chunks, worker_func)
    
        reconstructed = np.zeros(shape, dtype=np.float32)
        flat_results = np.concatenate(results)
    
        for idx, (i, j) in enumerate(valid_indices):
            reconstructed[i, j] = flat_results[idx]

        return reconstructed
    
    def normalize_residual(self, residual: np.ndarray, 
                          params: Dict[str, Any]) -> np.ndarray:
        """
        Normalize residual velocity field.
        
        Args:
            residual: Residual velocity field
            params: Solar parameters
            
        Returns:
            Normalized residual
        """
        if not self.config.normalize_data:
            return residual
        
        # Create solar disk mask with exclusion
        height, width = residual.shape
        cx, cy = params["cx"], params["cy"]
        radius = params["radius"]
        
        # Apply disk exclusion radius
        disk_radius = radius * self.config.disk_exclusion_radius
        y, x = np.ogrid[-cy:height-cy, -cx:width-cx]
        disk_mask = x**2 + y**2 <= disk_radius**2
        
        # Calculate mean of valid points on disk
        valid_data = residual[disk_mask & ~np.isnan(residual)]
        
        if len(valid_data) > 0:
            mean_val = np.mean(valid_data)
            residual_normalized = residual - mean_val
        else:
            residual_normalized = residual
        
        return residual_normalized

    def save_results(self, results: Dict[str, Any], original_filepath: str):
        """
        Save processing results.
        
        Args:
            results: Processing results
            original_filepath: Original file path
        """
        if not self.config.output_dir:
            return
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        base_name = Path(original_filepath).stem
        
        # Save residual velocity map (uncompressed, hdu[0])
        residual_file = os.path.join(
            self.config.output_dir,
            f"{base_name}_residual.fits"
        )
        
        # Create a clean header for output
        header = fits.Header()
        
        # Add basic required FITS keywords
        header['SIMPLE'] = True
        header['BITPIX'] = -32  # 32-bit floating point
        header['NAXIS'] = 2
        header['NAXIS1'] = results["residual"].shape[1]
        header['NAXIS2'] = results["residual"].shape[0]
        header['EXTEND'] = False
        
        # Copy important keywords from original header
        original_header = results["header"]
        important_keys = [
            'DATE-OBS', 'TELESCOP', 'INSTRUME', 'OBSERVER', 'OBJECT',
            'EXPTIME', 'FILTER', 'WAVELNTH', 'WAVELEN', 'BUNIT',
            'CTYPE1', 'CTYPE2', 'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2',
            'CDELT1', 'CDELT2', 'CUNIT1', 'CUNIT2', 'RSUN', 'R_SUN',
            'CX', 'CY', 'RADII', 'INST_ROT', 'B0', 'DATE'
        ]
        
        for key in important_keys:
            if key in original_header:
                value = original_header[key]
                if isinstance(value, str):
                    # Clean string values
                    cleaned_value = value.replace('\n', ' ').replace('\r', ' ')
                    cleaned_value = ''.join(char for char in cleaned_value if 32 <= ord(char) <= 126)
                    if len(cleaned_value) > 70:
                        cleaned_value = cleaned_value[:70]
                    header[key] = cleaned_value
                else:
                    header[key] = value
        
        # Add processing information
        header['HISTORY'] = 'Processed by Solar Chromosphere Velocity Processor'
        header['DISKEXCL'] = (self.config.disk_exclusion_radius, 
                             'Disk exclusion radius (0-1)')
        
        # Add fitted parameters to header
        param_names = ["A", "B", "C", "D", "G", "e", "N", "MC1", "MC2"]
        if "fitted_params" in results:
            for name, value in zip(param_names, results["fitted_params"]):
                header[f"FIT_{name}"] = (float(value), f"Fitted parameter {name}")
        
        # Save as uncompressed FITS
        hdu = fits.PrimaryHDU(results["residual"], header=header)
        hdu.writeto(residual_file, overwrite=True)
        logger.info(f"Saved residual to: {residual_file}")
        
        # Save active region mask (uncompressed, hdu[0])
        mask_file = os.path.join(
            self.config.output_dir,
            f"{base_name}_active_regions.fits"
        )
        
        mask_hdu = fits.PrimaryHDU(
            results["active_region_mask"].astype(np.uint8),
            header=header
        )
        mask_hdu.writeto(mask_file, overwrite=True)
        
        # Save parameters to text file
        param_file = os.path.join(
            self.config.output_dir,
            f"{base_name}_parameters.txt"
        )
        
        with open(param_file, "w") as f:
            f.write(f"Processing results for: {original_filepath}\n")
            f.write("=" * 50 + "\n\n")
            
            if "fitted_params" in results:
                f.write("Fitted Parameters:\n")
                f.write("-" * 30 + "\n")
                for name, value, error in zip(param_names, 
                                             results["fitted_params"],
                                             results.get("param_errors", [])):
                    f.write(f"{name}: {value:.6f} ± {error:.6f}\n")
            
            f.write("\nProcessing Configuration:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Disk exclusion radius: {self.config.disk_exclusion_radius}\n")
            for key, value in self.config.to_dict().items():
                if key != "disk_exclusion_radius":  # Already printed
                    f.write(f"{key}: {value}\n")
        
        # Generate summary plot if visualization is enabled
        if self.config.visualize_steps:
            plot_file = os.path.join(
                self.config.output_dir,
                f"{base_name}_summary.{self.config.plot_format}"
            )
            
            self.visualization_manager.create_summary_plot(
                results, plot_file,
                title=f"Processing Summary: {base_name}"
            )

    @timeit
    def process_directory(self, pattern: str = "*.fits",
                         processing_config: Optional[ProcessingConfig] = None) -> List[Dict[str, Any]]:
        """
        Process all matching files in the input directory.
        
        Args:
            pattern: File pattern to match
            processing_config: Runtime processing configuration
            
        Returns:
            List of processing results
        """
        import glob
        
        # Find files
        files = sorted(glob.glob(os.path.join(self.config.input_dir, pattern)))
        
        if not files:
            logger.warning(f"No files found matching pattern: {pattern}")
            return []
        
        logger.info(f"Found {len(files)} files to process")
        
        if processing_config:
            processing_config.total_steps = len(files) * 13  # 13 steps per file
        
        # Process files
        results = []
        
        for idx, filepath in enumerate(tqdm(files, desc="Processing files")):
            try:
                if processing_config:
                    processing_config.update_progress(
                        idx * 13,
                        f"Processing file {idx+1}/{len(files)}"
                    )
                
                result = self.process_file(filepath, processing_config)
                results.append(result)
                
                logger.info(f"Successfully processed: {filepath}")
                
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")
                continue
        
        # Generate batch summary
        if results and self.config.visualize_steps:
            self.visualization_manager.create_batch_summary(results)
        
        logger.info(f"Completed batch processing. Successfully processed {len(results)}/{len(files)} files")
        
        return results