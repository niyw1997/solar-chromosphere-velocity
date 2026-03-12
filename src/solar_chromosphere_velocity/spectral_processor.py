"""
Spectral processor module for Doppler velocity inversion from spectral data.
Based on multipro_chase_LSF.py implementation.
"""

import os
import numpy as np
from astropy.io import fits
import multiprocessing as mp
from scipy.optimize import curve_fit
from scipy import interpolate
from typing import Dict, Tuple, Optional, List, Any
import logging
from tqdm import tqdm
from numba import jit
import warnings
warnings.filterwarnings('ignore')

from .data_loader import DataType
from .utils import apply_disk_exclusion

logger = logging.getLogger(__name__)


class SpectralProcessor:
    """Spectral processor for Doppler velocity inversion."""
    
    def __init__(self, config):
        """
        Initialize spectral processor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Spectral line parameters
        self.spectral_lines = {
            'HA': {
                'wavelength': 6562.8,  # Å, Hα line center
                'window_width': 20.0,   # Å, processing window width
                'core_window': 2.0,     # Å, line core window
                'reference_profile': None,
            },
            'FE': {
                'wavelength': 6173.0,   # Å, Fe I line center
                'window_width': 15.0,
                'core_window': 1.5,
                'reference_profile': None,
            }
        }
        
        # Initialize reference profiles if provided
        if config.reference_profile_path:
            self._load_reference_profiles(config.reference_profile_path)
    
    def compute_doppler_velocity(self, spectral_data: np.ndarray,
                                header: fits.Header,
                                data_type: DataType,
                                params: Dict[str, Any]) -> Tuple[np.ndarray, fits.Header]:
        """
        Compute Doppler velocity from spectral data.
        
        Args:
            spectral_data: Spectral data cube (n_wavelength, n_y, n_x)
            header: Original FITS header
            data_type: Data type
            params: Processing parameters
            
        Returns:
            Tuple of (Doppler velocity map, updated header)
        """
        logger.info(f"Starting Doppler velocity inversion, spectral data shape: {spectral_data.shape}")
        
        # Verify data dimensions
        if spectral_data.ndim != 3:
            raise ValueError(f"Expected 3D spectral data (wavelength, y, x), got {spectral_data.ndim}D with shape {spectral_data.shape}")
        
        # Check wavelength dimension is reasonable
        n_wavelengths = spectral_data.shape[0]
        if n_wavelengths < 50:
            logger.warning(f"Wavelength dimension is very small: {n_wavelengths}")
        elif n_wavelengths > 5000:
            logger.warning(f"Wavelength dimension is very large: {n_wavelengths}")
        
        # Determine spectral line type
        if data_type == DataType.SPECTRAL_HA:
            line_type = 'HA'
        elif data_type == DataType.SPECTRAL_FE:
            line_type = 'FE'
        else:
            raise ValueError(f"Unsupported data type for Doppler inversion: {data_type}")
        
        # Get spectral line parameters
        line_params = self.spectral_lines[line_type]
        
        # Step 1: Compute average spectral profile
        logger.info("Computing average spectral profile...")
        avg_profile = self._compute_average_profile(spectral_data, params)
        logger.info(f"Average profile length: {len(avg_profile)}")
        
        # Step 2: Fit line center
        logger.info("Fitting line center...")
        ha_core = self._fit_line_center(avg_profile, line_params, header, params)
        logger.info(f"Line center index: {ha_core}")
        
        # Step 3: Compute Doppler map
        logger.info("Computing Doppler velocity map...")
        doppler_map = self._compute_doppler_map_chase(
            spectral_data, avg_profile, ha_core, header, params
        )
        
        # Step 4: Apply disk exclusion (set > 0.95 solar radius to NaN)
        if self.config.disk_exclusion_radius < 1.0:
            logger.info(f"Applying disk exclusion (radius > {self.config.disk_exclusion_radius})...")
            center = (params.get('cx', doppler_map.shape[1] // 2),
                     params.get('cy', doppler_map.shape[0] // 2))
            radius = params.get('radius', min(doppler_map.shape) // 2)
            doppler_map = apply_disk_exclusion(
                doppler_map, center, radius, self.config.disk_exclusion_radius
            )
        
        # Step 5: Update header (based on multipro_chase_LSF.py inherit_chase_fits_header)
        updated_header = self._create_output_header(header, doppler_map.shape, params)
        
        logger.info(f"Doppler velocity inversion completed, map shape: {doppler_map.shape}")
        
        return doppler_map, updated_header
    
    def _create_output_header(self, original_header: fits.Header,
                             shape: Tuple[int, int],
                             params: Dict[str, Any]) -> fits.Header:
        """
        Create output FITS header based on multipro_chase_LSF.py inherit_chase_fits_header.
        
        Args:
            original_header: Original FITS header
            shape: Output data shape
            params: Processing parameters
            
        Returns:
            Updated FITS header
        """
        # Create a new header with basic WCS information
        header = fits.Header()
        
        # Basic required FITS keywords
        header['SIMPLE'] = True
        header['BITPIX'] = -32  # 32-bit floating point
        header['NAXIS'] = 2
        header['NAXIS1'] = shape[1]
        header['NAXIS2'] = shape[0]
        header['EXTEND'] = False
        
        # Try to get WCS information from original header
        crval1 = self._safe_get_float(original_header, 'CRVAL1', 0.0)
        crval2 = self._safe_get_float(original_header, 'CRVAL2', 0.0)
        crpix1 = self._safe_get_float(original_header, 'CRPIX1', shape[1] // 2)
        crpix2 = self._safe_get_float(original_header, 'CRPIX2', shape[0] // 2)
        cdelt1 = self._safe_get_float(original_header, 'CDELT1', 0.0)
        cdelt2 = self._safe_get_float(original_header, 'CDELT2', 0.0)
        
        # Add WCS information (from multipro_chase_LSF.py create_fits_header)
        header['CTYPE1'] = 'HPLN-TAN'
        header['CTYPE2'] = 'HPLT-TAN'
        header['CUNIT1'] = 'arcsec'
        header['CUNIT2'] = 'arcsec'
        header['CRVAL1'] = crval1
        header['CRVAL2'] = crval2
        header['CRPIX1'] = crpix1
        header['CRPIX2'] = crpix2
        header['CDELT1'] = cdelt1
        header['CDELT2'] = cdelt2
        header['PC1_1'] = 1.0
        header['PC1_2'] = 0.0
        header['PC2_1'] = 0.0
        header['PC2_2'] = 1.0
        
        # Solar observation parameters
        b0_angle = params.get('b0', 0.0)
        p0_angle = params.get('p0', self.config.instrument_rotation)
        rsun_obs = params.get('rsun_obs', 960.0)
        date_obs = params.get('date_obs', '')
        
        if b0_angle is not None:
            header['SOLAR_B0'] = b0_angle
        if p0_angle is not None:
            header['SOLAR_P0'] = p0_angle
        if rsun_obs is not None:
            header['RSUN_OBS'] = rsun_obs
        if date_obs:
            header['DATE-OBS'] = date_obs
        
        # Copy additional keywords from original header (from multipro_chase_LSF.py)
        additional_keys = [
            'STEPTIME', 'STR_TIME', 'END_TIME', 'EXP_TIME',
            'SAT_POS1', 'SAT_POS2', 'SAT_POS3',
            'SAT_VEL1', 'SAT_VEL2', 'SAT_VEL3',
            'R_SUN', 'RSUN_REF'
        ]
        
        for key in additional_keys:
            if key in original_header:
                try:
                    value = original_header[key]
                    if isinstance(value, str):
                        # Clean string values
                        value = str(value).strip()
                        # Remove non-printable characters
                        import string
                        printable = set(string.printable)
                        value = ''.join(filter(lambda x: x in printable, value))
                        if len(value) > 70:
                            value = value[:70]
                    header[key] = value
                except Exception as e:
                    logger.debug(f"Failed to copy header key {key}: {e}")
        
        # Add solar parameters
        header['CX'] = params.get('cx', crpix1)
        header['CY'] = params.get('cy', crpix2)
        header['RADII'] = params.get('radius', min(shape) // 2)
        header['B0'] = params.get('b0', 0.0)
        header['INST_ROT'] = params.get('p0', self.config.instrument_rotation)
        
        # Add processing information
        from datetime import datetime
        header['PROCDATE'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        header['PROCMETH'] = 'cross_correlation'
        header['SOFTWARE'] = 'Solar Chromosphere Velocity Processor'
        header['VERSION'] = '1.0.1'
        
        # Add disk exclusion information
        header['DISKEXCL'] = (self.config.disk_exclusion_radius, 
                             'Disk exclusion radius (0-1)')
        
        # Add velocity unit
        header['BUNIT'] = 'km/s'
        header['DATATYPE'] = 'DOPPLER_VELOCITY'
        
        return header
    
    def _safe_get_float(self, header: fits.Header, key: str, default: float = 0.0) -> float:
        """
        Safely get a float value from FITS header.
        
        Args:
            header: FITS header
            key: Header keyword
            default: Default value if extraction fails
            
        Returns:
            Float value
        """
        try:
            if key in header:
                value = header[key]
                if isinstance(value, str):
                    # Try to extract number from string
                    import re
                    match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', value)
                    if match:
                        return float(match.group())
                    else:
                        return default
                else:
                    return float(value)
        except Exception:
            pass
        
        return default
    
    def _compute_average_profile(self, spectral_data: np.ndarray,
                                params: Dict[str, Any]) -> np.ndarray:
        """
        Compute average spectral profile based on multipro_chase_LSF.py logic.
        
        Args:
            spectral_data: Spectral data cube (n_wavelength, n_y, n_x)
            params: Processing parameters
            
        Returns:
            Average spectral profile
        """
        n_wavelengths, n_y, n_x = spectral_data.shape
        
        # Get solar parameters
        cx = params.get('cx', n_x // 2)
        cy = params.get('cy', n_y // 2)
        r = params.get('radius', min(n_y, n_x) // 2)
        
        # Create mask: (y - cy) ** 2 + (x - cx) ** 2 <= 100 ** 2
        # Note: multipro_chase_LSF.py uses radius 100, not the full solar radius
        mask_radius = 100  # Based on multipro_chase_LSF.py
        
        # Create coordinate grid
        y, x = np.indices((n_y, n_x))
        mask = (y - cy) ** 2 + (x - cx) ** 2 <= mask_radius ** 2
        
        # Use first 118 wavelength points (based on multipro_chase_LSF.py)
        # But ensure we don't exceed available wavelengths
        n_wavelength_to_use = min(118, n_wavelengths)
        logger.info(f"Using first {n_wavelength_to_use} wavelength points for average profile")
        
        # Extract masked data: shape (n_wavelength_to_use, n_masked_pixels)
        masked_data = spectral_data[:n_wavelength_to_use, mask]
        
        # Sum along spatial dimension and divide by count
        sum_spectra = masked_data.sum(axis=1)
        count = mask.sum()
        
        if count == 0:
            logger.warning("No pixels in mask, using entire image")
            masked_data = spectral_data[:n_wavelength_to_use].reshape(n_wavelength_to_use, -1)
            sum_spectra = masked_data.sum(axis=1)
            count = n_y * n_x
        
        avg_profile = sum_spectra / count
        
        logger.info(f"Computed average profile using {n_wavelength_to_use} wavelengths "
                   f"and {count} pixels within radius {mask_radius}")
        
        return avg_profile
    
    def _fit_line_center(self, profile: np.ndarray,
                        line_params: Dict[str, Any],
                        header: fits.Header,
                        params: Dict[str, Any]) -> int:
        """
        Fit spectral line center based on multipro_chase_LSF.py logic.
        
        Args:
            profile: Average spectral profile
            line_params: Line parameters
            header: FITS header
            params: Processing parameters
            
        Returns:
            Line center index
        """
        # Based on multipro_chase_LSF.py:
        # lx = np.arange(45)
        # paraha, _ = gaussfit(lx, ave_spectra[45:90], p0=[-1400, 22, 200, 2000])
        # ha_core = int(round(paraha[1])) + 45
        
        # Gaussian function for Ha core fitting (from multipro_chase_LSF.py)
        def gaussian(lx, *param):
            return param[0] * np.e ** (-(lx - param[1]) ** 2 / param[2]) + param[3]
        
        # Perform Gaussian fit (from multipro_chase_LSF.py)
        def gaussfit(lx, ly, p0=[1, 1, 1, 1]):
            para, pcov = curve_fit(gaussian, lx, ly, p0, method='lm')
            return para, pcov
        
        profile_length = len(profile)
        logger.info(f"Profile length for line center fitting: {profile_length}")
        
        # Check if profile is long enough
        if profile_length < 90:
            logger.warning(f"Profile length {profile_length} is less than minimum required 90")
            # Use center of available profile
            ha_core = profile_length // 2
            return ha_core
        
        # Extract profile segment (45:90 as in multipro_chase_LSF.py)
        start_idx = 45
        end_idx = 90
        
        # Ensure indices are within bounds
        if end_idx > profile_length:
            end_idx = profile_length
            start_idx = max(0, end_idx - 45)
        
        lx = np.arange(end_idx - start_idx)
        ly = profile[start_idx:end_idx]
        
        # Initial parameters from multipro_chase_LSF.py
        p0 = [-1400, 22, 200, 2000]
        
        try:
            para, _ = gaussfit(lx, ly, p0=p0)
            ha_core = int(round(para[1])) + start_idx
            
            # Ensure ha_core is within bounds and has enough margin for window
            ha_core = max(10, min(profile_length - 11, ha_core))
            
            logger.info(f"Fitted line center at index: {ha_core}")
        except Exception as e:
            logger.warning(f"Gaussian fit failed: {e}, using profile minimum as center")
            # Find minimum in the segment
            ha_core = np.argmin(profile[start_idx:end_idx]) + start_idx
        
        return ha_core
    
    def _compute_doppler_map_chase(self, spectral_data: np.ndarray,
                                  avg_profile: np.ndarray,
                                  ha_core: int,
                                  header: fits.Header,
                                  params: Dict[str, Any]) -> np.ndarray:
        """
        Compute Doppler velocity map based on multipro_chase_LSF.py logic.
        
        Args:
            spectral_data: Spectral data cube
            avg_profile: Average profile
            ha_core: Line center index
            header: FITS header
            params: Processing parameters
            
        Returns:
            Doppler velocity map
        """
        n_wavelengths, n_y, n_x = spectral_data.shape
        
        # Get solar parameters
        cx = params.get('cx', n_x // 2)
        cy = params.get('cy', n_y // 2)
        r = params.get('radius', min(n_y, n_x) // 2)
        
        # Check if ha_core is within valid range for spectral data
        if ha_core < 0 or ha_core >= n_wavelengths:
            logger.error(f"Invalid ha_core index {ha_core} for spectral data with {n_wavelengths} wavelengths")
            # Use middle of spectrum as fallback
            ha_core = n_wavelengths // 2
            logger.info(f"Using middle of spectrum as ha_core: {ha_core}")
        
        # Create output arrays
        doppler_map = np.zeros((n_y, n_x), dtype=np.float32)
        
        # Define window for cross-correlation (ha_core ± 10 as in multipro_chase_LSF.py)
        window_half = 10
        start_idx = max(0, ha_core - window_half)
        end_idx = min(n_wavelengths, ha_core + window_half + 1)
        
        # Ensure window size is valid
        if end_idx - start_idx < 5:
            logger.error(f"Window too small: {start_idx}:{end_idx}")
            return doppler_map
        
        # Reference spectrum (average profile in the window)
        ref_spec = avg_profile[start_idx:end_idx]
        
        logger.info(f"Computing Doppler map using window {start_idx}:{end_idx} "
                   f"around ha_core={ha_core}, window size={len(ref_spec)}")
        
        # Process each pixel
        for i in tqdm(range(n_y), desc="Computing Doppler velocities"):
            for j in range(n_x):
                # Check if pixel is within solar disk
                state = (i - cy) ** 2 + (j - cx) ** 2 <= r ** 2
                
                if state:
                    # Extract spectrum for this pixel
                    spec = spectral_data[start_idx:end_idx, i, j]
                    
                    if np.any(np.isnan(spec)):
                        doppler_map[i, j] = np.nan
                        continue
                    
                    try:
                        # Compute cross-correlation
                        correlation = self._cross_correlation_using_fft(spec, ref_spec)
                        
                        # Compute Doppler shift
                        dopplern = self._doppler_shift(correlation, len(ref_spec))
                        
                        doppler_map[i, j] = dopplern
                        
                    except Exception as e:
                        doppler_map[i, j] = np.nan
                else:
                    # Outside solar disk
                    doppler_map[i, j] = 0.0
        
        return doppler_map
    
    def _cross_correlation_using_fft(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute cross-correlation using FFT (from multipro_chase_LSF.py).
        
        Args:
            x: First array
            y: Second array
            
        Returns:
            Cross-correlation array
        """
        f1 = np.fft.fft(x)
        f2 = np.fft.fft(y)
        cc = np.fft.ifft(f1 * np.conjugate(f2)).real
        return np.fft.fftshift(cc)
    
    @staticmethod
    @jit(nopython=True)
    def _doppler_shift(correlation: np.ndarray, len_reference: int) -> float:
        """
        Compute Doppler shift from cross-correlation (from multipro_chase_LSF.py).
        
        Args:
            correlation: Cross-correlation array
            len_reference: Length of reference spectrum
            
        Returns:
            Doppler velocity in km/s
        """
        i_max = np.argmax(correlation)
        
        if i_max == 0 or i_max == len(correlation) - 1:
            displace = i_max - (len_reference // 2)  # Integer displacement relative to center
        else:
            R_0 = correlation[i_max]      
            R_m1 = correlation[i_max - 1] 
            R_p1 = correlation[i_max + 1]  
            x_offset = (R_m1 - R_p1) / (2 * (R_m1 - 2 * R_0 + R_p1))
            displace = (i_max + x_offset) - (len_reference // 2)
        
        # Convert displacement to velocity (from multipro_chase_LSF.py)
        # 1.1068 * 2 km/s per pixel displacement
        dopplern = displace * 1.1068 * 2  # in km/s
        
        return dopplern
    
    def save_doppler_data(self, doppler_map: np.ndarray,
                         header: fits.Header,
                         output_path: str):
        """
        Save Doppler velocity data to uncompressed FITS file.
        
        Processed Doppler velocity data is saved as uncompressed FITS in hdu[0].
        
        Args:
            doppler_map: Doppler velocity map
            header: FITS header
            output_path: Output file path
        """
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as uncompressed FITS file (hdu[0])
        hdu = fits.PrimaryHDU(doppler_map, header=header)
        
        # Remove compression-related keywords if present
        compression_keys = ['ZCMPTYPE', 'ZNAMES', 'ZVAL0', 'ZVAL1', 'ZVAL2', 
                           'XTENSION', 'PCOUNT', 'GCOUNT', 'TFIELDS', 'TTYPE1',
                           'TFORM1', 'TUNIT1']
        for key in compression_keys:
            if key in hdu.header:
                del hdu.header[key]
        
        hdu.writeto(output_path, overwrite=True)
        
        logger.info(f"Doppler velocity data saved to: {output_path}")
        logger.info(f"Data saved as uncompressed FITS (hdu[0])")
    
    def _load_reference_profiles(self, profile_path: str):
        """Load reference spectral profiles."""
        try:
            with fits.open(profile_path, fix=True, ignore_missing_end=True) as hdul:
                # Reference profiles are usually in hdu[0]
                if hdul[0].data is not None:
                    data = hdul[0].data
                    header = hdul[0].header
                elif len(hdul) > 1 and hdul[1].data is not None:
                    # Fallback to hdu[1] if hdu[0] is empty
                    data = hdul[1].data
                    header = hdul[1].header
                else:
                    raise ValueError(f"No valid data found in reference profile: {profile_path}")
                
                # Extract reference profiles
                if data.ndim == 3 and data.shape[0] >= 2:
                    # Assuming first slice is HA, second is FE
                    self.spectral_lines['HA']['reference_profile'] = data[0]
                    self.spectral_lines['FE']['reference_profile'] = data[1]
                elif 'HA' in header:
                    self.spectral_lines['HA']['reference_profile'] = data[0]
                elif 'FE' in header:
                    self.spectral_lines['FE']['reference_profile'] = data[1]
                    
                logger.info("Reference profiles loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load reference profiles: {e}")