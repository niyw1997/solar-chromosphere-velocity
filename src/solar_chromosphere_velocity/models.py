"""
Physical models for solar velocity fields.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from scipy.optimize import curve_fit
import logging
from dataclasses import dataclass

from .config import ChromosphereConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelParameters:
    """Container for model parameters."""
    
    # Differential rotation parameters
    A: float  # Equatorial rotation rate
    B: float  # Latitude dependence coefficient
    C: float  # Higher order latitude dependence
    D: float  # Edge correction term
    
    # Constant offset
    G: float  # Global offset (includes unremoved redshift, etc.)
    
    # Limb shift parameters
    e: float  # Limb shift amplitude
    N: float  # Limb shift power
    
    # Meridional circulation parameters
    MC1: float  # First meridional circulation term
    MC2: float  # Second meridional circulation term
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.A, self.B, self.C, self.D, 
                         self.G, self.e, self.N, self.MC1, self.MC2])
    
    @classmethod
    def from_array(cls, array: np.ndarray) -> "ModelParameters":
        """Create from numpy array."""
        if len(array) != 9:
            raise ValueError(f"Expected 9 parameters, got {len(array)}")
        return cls(*array)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "A": self.A,
            "B": self.B,
            "C": self.C,
            "D": self.D,
            "G": self.G,
            "e": self.e,
            "N": self.N,
            "MC1": self.MC1,
            "MC2": self.MC2
        }


class VelocityModels:
    """Collection of velocity field models."""
    
    @staticmethod
    def chromosphere_full_model(coords: Tuple[np.ndarray, ...], 
                               A: float, B: float, C: float, D: float,
                               G: float, e: float, N: float,
                               MC1: float, MC2: float) -> np.ndarray:
        """
        Full chromospheric velocity model.
        
        This model includes:
        - Differential rotation (A + B*sin²φ + C*sin⁴φ + D*sin¹⁰ρ)
        - Meridional circulation (MC1*sinφ + MC2*sin2φ)
        - Limb shift (e*(1-cosρ)^N)
        - Constant offset (G)
        
        Args:
            coords: Tuple of coordinate arrays:
                sinφ, sin2φ, cosφ, sinL, cosL, ρ
            A, B, C, D: Differential rotation parameters
            G: Constant offset
            e, N: Limb shift parameters
            MC1, MC2: Meridional circulation parameters
            
        Returns:
            Model velocity field
        """
        sinphi, sin2phi, cosphi, sinL, cosL, rho = coords
        
        # Convert rho from radians if needed
        if np.max(rho) > 2 * np.pi:  # Assume degrees
            rho_rad = np.radians(rho)
        else:
            rho_rad = rho
        
        # Differential rotation term
        diff_rot = A + B * sinphi**2 + C * sinphi**4 + D * np.sin(rho_rad)**10
        
        # Projection factor for differential rotation
        # Assuming solar radius and B0 are accounted for elsewhere
        proj_factor = cosphi * sinL
        
        # Limb shift term
        limb_shift = e * (1 - np.cos(rho_rad))**N
        
        # Meridional circulation term
        # This term needs proper projection, simplified here
        meridional = MC1 * sinphi + MC2 * sin2phi
        
        return diff_rot * proj_factor + limb_shift + meridional + G
    
    @staticmethod
    def differential_rotation_only(coords: Tuple[np.ndarray, ...],
                                  A: float, B: float, C: float) -> np.ndarray:
        """
        Differential rotation only model.
        
        Args:
            coords: Tuple of (sinφ, cosφ, sinL)
            A, B, C: Model parameters
            
        Returns:
            Differential rotation velocity
        """
        sinphi, cosphi, sinL = coords
        diff_rot = A + B * sinphi**2 + C * sinphi**4
        return diff_rot * cosphi * sinL
    
    @staticmethod
    def limb_shift_model(rho: np.ndarray, e: float, N: float) -> np.ndarray:
        """
        Limb shift model.
        
        Args:
            rho: Angular distance from disk center (radians)
            e, N: Model parameters
            
        Returns:
            Limb shift velocity
        """
        if np.max(rho) > 2 * np.pi:  # Assume degrees
            rho_rad = np.radians(rho)
        else:
            rho_rad = rho
        
        return e * (1 - np.cos(rho_rad))**N
    
    @staticmethod
    def simple_meridional_circulation(sinphi: np.ndarray,
                                     MC1: float, MC2: float) -> np.ndarray:
        """
        Simple meridional circulation model.
        
        Args:
            sinphi: sin(latitude)
            MC1, MC2: Model parameters
            
        Returns:
            Meridional circulation velocity
        """
        sin2phi = 2 * sinphi * np.sqrt(1 - sinphi**2)
        return MC1 * sinphi + MC2 * sin2phi


class ModelFitter:
    """Fitter for velocity field models."""
    
    def __init__(self, config: ChromosphereConfig):
        """
        Initialize model fitter.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.default_bounds = self._get_default_bounds()
    
    def _get_default_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get default parameter bounds."""
        # Lower bounds
        lower = np.array([
            0.5,   # A: minimum rotation rate
            -2.0,  # B: latitude dependence
            -1.0,  # C: higher order term
            0.0,   # D: edge correction
            -100.0, # G: constant offset
            0.0,   # e: limb shift amplitude
            1.0,   # N: limb shift power
            -50.0, # MC1: meridional term 1
            -50.0  # MC2: meridional term 2
        ])
        
        # Upper bounds
        upper = np.array([
            3.0,   # A: maximum rotation rate
            2.0,   # B: latitude dependence
            1.0,   # C: higher order term
            1.0,   # D: edge correction
            100.0, # G: constant offset
            2000.0, # e: limb shift amplitude
            5.0,   # N: limb shift power
            50.0,  # MC1: meridional term 1
            50.0   # MC2: meridional term 2
        ])
        
        return (lower, upper)
    
    def fit(self, fitting_data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the chromosphere velocity model.
        
        Args:
            fitting_data: Dictionary containing 'coordinates' and 'velocity'
            
        Returns:
            Tuple of (fitted_parameters, parameter_errors)
        """
        if fitting_data["coordinates"] is None or fitting_data["velocity"] is None:
            logger.warning("No valid data for fitting")
            return self._get_default_parameters()
        
        coords = fitting_data["coordinates"]
        velocities = fitting_data["velocity"]
        
        # Initial parameter guess
        initial_guess = np.array([2.0, -0.5, -0.2, 0.1, 
                                  0.0, 500.0, 2.0, 10.0, 5.0])
        
        # Prepare bounds
        if self.config.use_bounds:
            bounds = self.default_bounds
        else:
            bounds = (-np.inf, np.inf)
        
        try:
            # Perform the fit
            params, pcov = curve_fit(
                VelocityModels.chromosphere_full_model,
                coords,
                velocities,
                p0=initial_guess,
                bounds=bounds,
                maxfev=self.config.max_fitting_iterations,
                ftol=self.config.fitting_tolerance,
                xtol=self.config.fitting_tolerance
            )
            
            # Calculate parameter errors from covariance matrix
            errors = np.sqrt(np.diag(pcov))
            
            logger.info(f"Successfully fitted model with parameters: {params}")
            
            return params, errors
            
        except Exception as e:
            logger.error(f"Model fitting failed: {e}")
            
            # Return default parameters
            return self._get_default_parameters()
    
    def _get_default_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get default parameters when fitting fails."""
        default_params = np.array([2.0, -0.5, -0.2, 0.1, 
                                   0.0, 500.0, 2.0, 10.0, 5.0])
        default_errors = np.ones_like(default_params) * 0.1
        
        return default_params, default_errors
    
    def validate_parameters(self, params: np.ndarray) -> bool:
        """
        Validate fitted parameters.
        
        Args:
            params: Fitted parameters
            
        Returns:
            True if parameters are valid
        """
        if len(params) != 9:
            return False
        
        # Check for NaN or infinite values
        if np.any(np.isnan(params)) or np.any(np.isinf(params)):
            return False
        
        # Check parameter bounds (optional)
        lower, upper = self.default_bounds
        if np.any(params < lower) or np.any(params > upper):
            logger.warning("Some parameters are outside typical bounds")
        
        return True
    
    def get_parameter_descriptions(self) -> Dict[str, str]:
        """Get descriptions of model parameters."""
        return {
            "A": "Equatorial rotation rate (km/s)",
            "B": "Latitude dependence coefficient for sin²φ term",
            "C": "Higher order latitude dependence for sin⁴φ term",
            "D": "Edge correction coefficient for sin¹⁰ρ term",
            "G": "Global constant offset (km/s)",
            "e": "Limb shift amplitude (km/s)",
            "N": "Limb shift power law exponent",
            "MC1": "First meridional circulation coefficient (km/s)",
            "MC2": "Second meridional circulation coefficient (km/s)"
        }