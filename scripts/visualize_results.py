#!/usr/bin/env python3
"""
Script for visualizing processing results.
"""

import sys
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from pathlib import Path

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solar_chromosphere_velocity.plotting import VisualizationManager
from solar_chromosphere_velocity.config import ChromosphereConfig


def visualize_single_file(filepath: str, output_dir: str):
    """Visualize a single processed file."""
    # Load data
    with fits.open(filepath) as hdul:
        data = hdul[0].data
        header = hdul[0].header
    
    # Create config
    config = ChromosphereConfig(
        output_dir=output_dir,
        visualize_steps=True,
        plot_dpi=150
    )
    
    # Create visualization manager
    viz = VisualizationManager(config)
    
    # Create visualization
    metadata = {
        "title": f"Processed: {Path(filepath).name}",
        "filepath": filepath
    }
    
    if "residual" in filepath.lower():
        viz.plot_step("residual", data, metadata)
    elif "mask" in filepath.lower():
        viz.plot_step("mask", data, metadata)
    else:
        viz.plot_step("data", data, metadata)
    
    print(f"Visualization saved to {output_dir}")


def compare_files(file1: str, file2: str, output_dir: str):
    """Compare two processed files."""
    # Load data
    with fits.open(file1) as hdul1:
        data1 = hdul1[0].data
    
    with fits.open(file2) as hdul2:
        data2 = hdul2[0].data
    
    # Calculate difference
    diff = data1 - data2
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot file 1
    im1 = axes[0].imshow(data1, cmap="bwr", origin="lower", vmin=-3, vmax=3)
    axes[0].set_title(Path(file1).name)
    plt.colorbar(im1, ax=axes[0])
    
    # Plot file 2
    im2 = axes[1].imshow(data2, cmap="bwr", origin="lower", vmin=-3, vmax=3)
    axes[1].set_title(Path(file2).name)
    plt.colorbar(im2, ax=axes[1])
    
    # Plot difference
    vmax = max(np.abs(np.nanmin(diff)), np.abs(np.nanmax(diff)))
    im3 = axes[2].imshow(diff, cmap="bwr", origin="lower", vmin=-vmax, vmax=vmax)
    axes[2].set_title("Difference")
    plt.colorbar(im3, ax=axes[2])
    
    # Save figure
    output_path = os.path.join(output_dir, "comparison.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Comparison saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize processing results")
    parser.add_argument("--input", "-i", help="Input FITS file or directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--compare", "-c", nargs=2, help="Two files to compare")
    parser.add_argument("--batch", "-b", action="store_true", help="Process batch of files")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    if args.compare:
        compare_files(args.compare[0], args.compare[1], args.output)
    elif args.batch and args.input:
        # Process all FITS files in directory
        import glob
        files = glob.glob(os.path.join(args.input, "*.fits"))
        
        for filepath in files:
            visualize_single_file(filepath, args.output)
        
        print(f"Processed {len(files)} files")
    elif args.input:
        visualize_single_file(args.input, args.output)
    else:
        print("Please specify input file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()