"""
Visualization module for solar chromosphere velocity processing.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib import ticker

from .config import ChromosphereConfig, VisualizationMode


class VisualizationManager:
    """Manager for visualization during processing."""
    
    def __init__(self, config: ChromosphereConfig):
        """
        Initialize visualization manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.figure_size = (10, 8)
        self.dpi = config.plot_dpi
        
        # Setup color map
        self.cmap = self._get_colormap(config.colormap)
        
        # Create output directory for plots if needed
        if config.visualize_steps and config.output_dir:
            self.plot_dir = os.path.join(config.output_dir, "plots")
            os.makedirs(self.plot_dir, exist_ok=True)
        else:
            self.plot_dir = None
    
    def _get_colormap(self, cmap_name: str) -> cm.ScalarMappable:
        """Get color map with proper normalization."""
        if cmap_name == "bwr":
            # Blue-White-Red diverging colormap
            return cm.get_cmap("bwr")
        else:
            return cm.get_cmap(cmap_name)
    
    def plot_step(self, step_name: str, data: np.ndarray, 
                  metadata: Dict[str, Any]):
        """
        Create plot for a processing step.
        
        Args:
            step_name: Name of the processing step
            data: Data to plot
            metadata: Additional metadata for the plot
        """
        if not self.config.visualize_steps:
            return
        filepath = metadata.get('filepath', 'unknown')
        if filepath != 'unknown':
            filename = os.path.basename(filepath)
            base_name = os.path.splitext(filename)[0]
        else:
            base_name = 'unknown'

        plot_filename = f"{base_name}_{step_name}.{self.config.plot_format}"
        plot_filename = plot_filename.replace('/', '_').replace('\\', '_')

        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # Determine plot type based on step name
        if "mask" in step_name.lower():
            self._plot_mask(ax, data, metadata)
        elif "residual" in step_name.lower():
            self._plot_residual(ax, data, metadata)
        elif "flow" in step_name.lower():
            self._plot_velocity(ax, data, metadata, vmin=-3, vmax=3)
        else:
            self._plot_generic(ax, data, metadata)
        
        # Add title
        title = metadata.get("title", step_name.replace("_", " ").title())
        ax.set_title(title, fontsize=14, fontweight="bold")
        
        # Add colorbar for non-mask plots
        if "mask" not in step_name.lower():
            cbar = plt.colorbar(ax.images[0], ax=ax, pad=0.02)
            cbar.set_label("Velocity (km/s)", fontsize=12)
        
        # Save figure
        if self.plot_dir:
            save_path = os.path.join(self.plot_dir, plot_filename)
            plt.tight_layout()
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
    
    def _plot_velocity(self, ax: plt.Axes, data: np.ndarray,
                       metadata: Dict[str, Any],
                       vmin: Optional[float] = None,
                       vmax: Optional[float] = None):
        """Plot velocity data."""
        if vmin is None:
            vmin = np.nanpercentile(data, 1)
        if vmax is None:
            vmax = np.nanpercentile(data, 99)
        
        # Use symmetric colormap range for velocity
        vmax_abs = max(abs(vmin), abs(vmax))
        vmin, vmax = -vmax_abs, vmax_abs
        
        im = ax.imshow(data, cmap=self.cmap, origin="lower",
                       vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        
        ax.set_xlabel("X (pixels)", fontsize=12)
        ax.set_ylabel("Y (pixels)", fontsize=12)
        
        return im
    
    def _plot_residual(self, ax: plt.Axes, data: np.ndarray,
                       metadata: Dict[str, Any]):
        """Plot residual velocity."""
        # Calculate symmetric range around zero
        vmax = np.nanpercentile(np.abs(data), 95)
        vmin = -vmax
        
        im = ax.imshow(data, cmap=self.cmap, origin="lower",
                       vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        
        ax.set_xlabel("X (pixels)", fontsize=12)
        ax.set_ylabel("Y (pixels)", fontsize=12)
        
        # Add statistics text
        stats = self._calculate_statistics(data)
        stats_text = f"Mean: {stats['mean']:.3f} km/s\nStd: {stats['std']:.3f} km/s"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        
        return im
    
    def _plot_mask(self, ax: plt.Axes, data: np.ndarray,
                   metadata: Dict[str, Any]):
        """Plot binary mask."""
        # Create custom colormap for mask
        from matplotlib.colors import ListedColormap
        mask_cmap = ListedColormap(["white", "red"])
        
        im = ax.imshow(data, cmap=mask_cmap, origin="lower",
                       vmin=0, vmax=1,
                       interpolation="nearest")
        
        ax.set_xlabel("X (pixels)", fontsize=12)
        ax.set_ylabel("Y (pixels)", fontsize=12)
        
        # Add mask statistics
        mask_fraction = np.sum(data) / data.size
        stats_text = f"Active regions: {mask_fraction:.2%}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        
        return im
    
    def _plot_generic(self, ax: plt.Axes, data: np.ndarray,
                      metadata: Dict[str, Any]):
        """Generic plot for any data type."""
        if data.dtype == bool or np.issubdtype(data.dtype, np.integer):
            # For boolean or integer data
            im = ax.imshow(data, cmap="viridis", origin="lower",
                           interpolation="nearest")
        else:
            # For float data
            vmin = np.nanmin(data)
            vmax = np.nanmax(data)
            im = ax.imshow(data, cmap=self.cmap, origin="lower",
                           vmin=vmin, vmax=vmax,
                           interpolation="nearest")
            
            # Add colorbar
            plt.colorbar(im, ax=ax, pad=0.02)
        
        ax.set_xlabel("X (pixels)", fontsize=12)
        ax.set_ylabel("Y (pixels)", fontsize=12)
        
        return im
    
    def _calculate_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate basic statistics for data."""
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        return {
            "mean": float(np.mean(valid_data)),
            "std": float(np.std(valid_data)),
            "min": float(np.min(valid_data)),
            "max": float(np.max(valid_data))
        }
    
    def create_summary_plot(self, results: Dict[str, Any],
                           save_path: str,
                           title: str = "Processing Summary"):
        """
        Create summary plot of processing results.
        
        Args:
            results: Processing results dictionary
            save_path: Path to save the plot
            title: Plot title
        """
        # Extract data
        original = results.get("original_data")
        corrected = results.get("corrected_data")
        mask = results.get("active_region_mask")
        large_scale = results.get("large_scale_flow")
        residual = results.get("residual")
        
        if original is None:
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=self.dpi)
        axes = axes.flatten()
        
        # Plot 1: Original data
        self._plot_velocity(axes[0], original, {"title": "Original Velocity"})
        axes[0].set_title("Original Velocity", fontsize=12, fontweight="bold")
        
        # Plot 2: Gravitational redshift corrected
        self._plot_velocity(axes[1], corrected, {"title": "Redshift Corrected"})
        axes[1].set_title("Redshift Corrected", fontsize=12, fontweight="bold")
        
        # Plot 3: Active region mask
        self._plot_mask(axes[2], mask, {"title": "Active Regions"})
        axes[2].set_title("Active Regions", fontsize=12, fontweight="bold")
        
        # Plot 4: Large-scale flow
        self._plot_velocity(axes[3], large_scale, {"title": "Large-Scale Flow"})
        axes[3].set_title("Large-Scale Flow", fontsize=12, fontweight="bold")
        
        # Plot 5: Residual velocity
        self._plot_residual(axes[4], residual, {"title": "Residual Velocity"})
        axes[4].set_title("Residual Velocity", fontsize=12, fontweight="bold")
        
        # Plot 6: Histogram of residual
        self._plot_histogram(axes[5], residual, {"title": "Residual Distribution"})
        axes[5].set_title("Residual Distribution", fontsize=12, fontweight="bold")
        
        # Add main title
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
    
    def _plot_histogram(self, ax: plt.Axes, data: np.ndarray,
                        metadata: Dict[str, Any]):
        """Plot histogram of data."""
        valid_data = data[~np.isnan(data)].flatten()
        
        if len(valid_data) == 0:
            ax.text(0.5, 0.5, "No valid data",
                    transform=ax.transAxes,
                    ha="center", va="center")
            return
        
        # Create histogram
        n, bins, patches = ax.hist(valid_data, bins=50, alpha=0.7,
                                   edgecolor="black", density=True)
        
        # Add Gaussian fit
        from scipy.stats import norm
        mu, std = norm.fit(valid_data)
        x = np.linspace(min(valid_data), max(valid_data), 100)
        p = norm.pdf(x, mu, std)
        ax.plot(x, p, 'r-', linewidth=2, label=f'Gaussian fit\nμ={mu:.3f}, σ={std:.3f}')
        
        ax.set_xlabel("Velocity (km/s)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    def create_batch_summary(self, results: List[Dict[str, Any]]):
        """
        Create summary of batch processing.
        
        Args:
            results: List of processing results
        """
        if not results or not self.plot_dir:
            return
        
        # Extract fitted parameters
        param_names = ["A", "B", "C", "D", "G", "e", "N", "MC1", "MC2"]
        param_arrays = {name: [] for name in param_names}
        
        for result in results:
            params = result.get("fitted_params")
            if params is not None and len(params) == 9:
                for name, value in zip(param_names, params):
                    param_arrays[name].append(value)
        
        # Create parameter distribution plots
        fig, axes = plt.subplots(3, 3, figsize=(15, 12), dpi=self.dpi)
        axes = axes.flatten()
        
        for idx, name in enumerate(param_names):
            if param_arrays[name]:
                axes[idx].hist(param_arrays[name], bins=20, alpha=0.7,
                               edgecolor="black")
                axes[idx].set_title(f"Parameter: {name}", fontsize=11)
                axes[idx].set_xlabel("Value", fontsize=10)
                axes[idx].set_ylabel("Count", fontsize=10)
                axes[idx].grid(True, alpha=0.3)
        
        # Remove unused axes
        for idx in range(len(param_names), len(axes)):
            fig.delaxes(axes[idx])
        
        fig.suptitle("Parameter Distributions from Batch Processing",
                    fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()
        
        # Save batch summary
        save_path = os.path.join(self.plot_dir, "batch_summary.png")
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)


def create_step_plot(step_name: str, data: np.ndarray,
                    metadata: Dict[str, Any],
                    config: ChromosphereConfig) -> Optional[str]:
    """
    Convenience function to create a single step plot.
    
    Args:
        step_name: Name of the processing step
        data: Data to plot
        metadata: Plot metadata
        config: Configuration object
        
    Returns:
        Path to saved plot if successful, None otherwise
    """
    if not config.visualize_steps:
        return None
    
    manager = VisualizationManager(config)
    manager.plot_step(step_name, data, metadata)
    
    if manager.plot_dir:
        filename = f"{step_name}_{metadata.get('filepath', 'unknown')}.{config.plot_format}"
        filename = filename.replace("/", "_").replace("\\", "_")
        return os.path.join(manager.plot_dir, filename)
    
    return None