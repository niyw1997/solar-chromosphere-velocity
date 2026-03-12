"""
Command Line Interface for solar chromosphere velocity processor.
"""

import argparse
import sys
import os
from pathlib import Path
import json
import logging

from .config import ChromosphereConfig, ProcessingMode, VisualizationMode
from .processor import ChromosphereProcessor
from .plotting import VisualizationManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Process solar chromosphere velocity data from CHASE mission.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input directory containing FITS files"
    )
    
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for processed data"
    )
    
    # Optional arguments
    parser.add_argument(
        "--pattern", "-p",
        default="*.fits",
        help="File pattern to match in input directory"
    )
    
    parser.add_argument(
        "--config", "-c",
        help="Path to JSON configuration file"
    )
    
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Enable step-by-step visualization"
    )
    
    parser.add_argument(
        "--visualize-all",
        action="store_true",
        help="Enable detailed visualization (all steps)"
    )
    
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable all visualization"
    )
    
    # Spectral processing options
    parser.add_argument(
        "--spectral-processing",
        action="store_true",
        default=True,
        help="Enable spectral data preprocessing"
    )
    
    parser.add_argument(
        "--no-spectral-processing",
        action="store_true",
        help="Disable spectral data preprocessing"
    )
    
    parser.add_argument(
        "--spectral-line",
        choices=["auto", "HA", "FE"],
        default="auto",
        help="Specify spectral line type"
    )
    
    parser.add_argument(
        "--reference-profile",
        help="Path to reference spectral profile file"
    )
    
    parser.add_argument(
        "--doppler-method",
        choices=["cross_correlation", "gaussian_fit"],
        default="cross_correlation",
        help="Method for Doppler velocity inversion"
    )
    
    # Processing options
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    
    parser.add_argument(
        "--activity-threshold",
        type=float,
        default=3.0,
        help="Threshold for active region detection (in sigma)"
    )
    
    parser.add_argument(
        "--min-area",
        type=int,
        default=100,
        help="Minimum area for active regions (in pixels)"
    )
    
    parser.add_argument(
        "--edge-angle",
        type=float,
        default=80.0,
        help="Angle from disk center to exclude (in degrees)"
    )
    
    # New: Disk exclusion parameter
    parser.add_argument(
        "--disk-exclusion",
        type=float,
        default=0.95,
        help="Disk exclusion radius (0-1, exclude > this fraction of solar radius)"
    )
    
    # Output options
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate processing results"
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files"
    )
    
    # Logging options
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level"
    )
    
    parser.add_argument(
        "--log-file",
        help="Path to log file"
    )
    
    # Advanced options
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Disable data validation"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    
    return parser


def load_config_from_file(config_path: str) -> dict:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config_dict
    except Exception as e:
        logger.error(f"Error loading configuration file: {e}")
        return {}


def save_config_to_file(config: ChromosphereConfig, output_dir: str):
    """Save configuration to JSON file."""
    config_path = os.path.join(output_dir, "processing_config.json")
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2, default=str)
        logger.info(f"Saved configuration to {config_path}")
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")


def setup_config(args: argparse.Namespace) -> ChromosphereConfig:
    """Setup configuration from command line arguments."""
    # Start with default configuration
    config = ChromosphereConfig()
    
    # Load configuration from file if provided
    if args.config:
        file_config = load_config_from_file(args.config)
        config.update(**file_config)
    
    # Update with command line arguments
    config.input_dir = args.input
    config.output_dir = args.output
    
    # Set visualization mode
    if args.no_visualize:
        config.visualize_steps = False
        config.visualization_mode = VisualizationMode.NONE
    elif args.visualize_all:
        config.visualize_steps = True
        config.visualization_mode = VisualizationMode.ALL
    elif args.visualize:
        config.visualize_steps = True
        config.visualization_mode = VisualizationMode.BASIC
    else:
        config.visualize_steps = False
        config.visualization_mode = VisualizationMode.NONE
    
    # Set spectral processing options
    if args.no_spectral_processing:
        config.spectral_processing = False
    else:
        config.spectral_processing = args.spectral_processing
    
    config.spectral_line_type = args.spectral_line
    config.doppler_inversion_method = args.doppler_method
    
    if args.reference_profile:
        config.reference_profile_path = args.reference_profile
    
    # Set processing options
    config.n_workers = args.workers
    config.activity_threshold_sigma = args.activity_threshold
    config.min_activity_area = args.min_area
    config.edge_exclusion_angle = args.edge_angle
    config.disk_exclusion_radius = args.disk_exclusion  # New parameter
    config.save_intermediate = args.save_intermediate
    config.overwrite_existing = args.overwrite
    config.log_level = args.log_level
    config.log_file = args.log_file
    config.enable_validation = not args.no_validation
    config.enable_debug_output = args.debug
    
    return config


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input):
        logger.error(f"Input directory does not exist: {args.input}")
        sys.exit(1)
    
    # Validate disk exclusion parameter
    if args.disk_exclusion <= 0 or args.disk_exclusion > 1:
        logger.error(f"Disk exclusion must be between 0 and 1, got: {args.disk_exclusion}")
        sys.exit(1)
    
    # Setup configuration
    config = setup_config(args)
    
    # Save configuration
    save_config_to_file(config, config.output_dir)
    
    # Create processor
    try:
        processor = ChromosphereProcessor(config)
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")
        sys.exit(1)
    
    # Process files
    try:
        logger.info(f"Starting processing of files in {args.input}")
        logger.info(f"Output will be saved to {args.output}")
        logger.info(f"Disk exclusion radius: {config.disk_exclusion_radius}")
        
        results = processor.process_directory(args.pattern)
        
        logger.info(f"Processing completed successfully!")
        logger.info(f"Processed {len(results)} files")
        
        # Print summary
        if results:
            print_summary(results, config)
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


def print_summary(results: list, config: ChromosphereConfig):
    """Print processing summary."""
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    
    # Count different types of data
    spectral_count = sum(1 for r in results 
                        if r.get("metadata", {}).get("data_type", "").startswith("spectral"))
    doppler_count = len(results) - spectral_count
    preprocessed_count = sum(1 for r in results 
                            if r.get("metadata", {}).get("preprocessed", False))
    
    print(f"Total files processed: {len(results)}")
    print(f"Spectral data files: {spectral_count}")
    print(f"Doppler velocity files: {doppler_count}")
    print(f"Preprocessed files: {preprocessed_count}")
    print(f"Spectral processing: {'Enabled' if config.spectral_processing else 'Disabled'}")
    print(f"Spectral line type: {config.spectral_line_type}")
    print(f"Doppler method: {config.doppler_inversion_method}")
    print(f"Disk exclusion radius: {config.disk_exclusion_radius}")
    print(f"Output directory: {config.output_dir}")
    print("="*60)
    
    if config.visualize_steps:
        print(f"\nVisualizations saved to: {os.path.join(config.output_dir, 'plots')}")


if __name__ == "__main__":
    main()