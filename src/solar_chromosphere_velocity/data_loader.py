"""
Smart data loader module for automatic data type detection and loading.
"""

import os
import numpy as np
from astropy.io import fits
from typing import Dict, Tuple, Optional, Any
import logging
from pathlib import Path
from enum import Enum
import warnings
from astropy.io.fits.verify import VerifyWarning

logger = logging.getLogger(__name__)


class DataType(Enum):
    """Data type enumeration."""
    UNKNOWN = "unknown"
    SPECTRAL_HA = "spectral_ha"  # Hα spectral data (RICE compressed)
    SPECTRAL_FE = "spectral_fe"  # Fe spectral data (RICE compressed)
    DOPPLER_HA = "doppler_ha"    # Hα Doppler velocity
    DOPPLER_FE = "doppler_fe"    # Fe Doppler velocity
    DOPPLER_VDOP = "doppler_vdop" # Processed Doppler velocity


class SmartDataLoader:
    """Smart data loader for automatic data type detection and loading."""
    
    def __init__(self, config):
        """
        Initialize smart data loader.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
    def detect_data_type(self, filepath: str) -> DataType:
        """
        Detect data type from filename and FITS header.
        
        Args:
            filepath: Path to data file
            
        Returns:
            Detected data type
        """
        filename = Path(filepath).name.lower()
        
        # Check filename patterns
        if '_ha.fits' in filename:
            if '_vdop' in filename:
                return DataType.DOPPLER_HA
            else:
                return DataType.SPECTRAL_HA
        elif '_fe.fits' in filename:
            if '_vdop' in filename:
                return DataType.DOPPLER_FE
            else:
                return DataType.SPECTRAL_FE
        elif '_vdop.fits' in filename:
            return DataType.DOPPLER_VDOP
        elif 'doppler' in filename:
            return DataType.DOPPLER_VDOP
        else:
            # Try to determine from FITS header
            try:
                # Suppress verification warnings for detection
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', VerifyWarning)
                    with fits.open(filepath, fix=True, ignore_missing_end=True, 
                                 verify='silentfix', output_verify='ignore') as hdul:
                        # Try to get header from primary HDU
                        header = hdul[0].header
                        
                        # Check for spectral dimensions
                        if 'NAXIS' in header and header['NAXIS'] == 3:
                            # 3D data, likely spectral
                            if 'ha' in header.get('CTYPE3', '').lower():
                                return DataType.SPECTRAL_HA
                            elif 'fe' in header.get('CTYPE3', '').lower():
                                return DataType.SPECTRAL_FE
                        
                        # Check for velocity units
                        if 'BUNIT' in header and 'm/s' in str(header['BUNIT']).lower():
                            return DataType.DOPPLER_VDOP
                            
            except Exception as e:
                logger.warning(f"Cannot determine data type from FITS header: {e}")
            
            return DataType.UNKNOWN
    
    def load_data(self, filepath: str) -> Tuple[np.ndarray, fits.Header, DataType, Dict[str, Any]]:
        """
        Load data based on detected type.
        
        Args:
            filepath: Path to data file
            
        Returns:
            Tuple of (data, header, data_type, parameters)
        """
        data_type = self.detect_data_type(filepath)
        logger.info(f"Detected data type: {data_type.value}")
        
        if data_type in [DataType.SPECTRAL_HA, DataType.SPECTRAL_FE]:
            return self._load_spectral_data(filepath, data_type)
        else:
            return self._load_doppler_data(filepath, data_type)
    
    def _load_spectral_data(self, filepath: str, data_type: DataType) -> Tuple[np.ndarray, fits.Header, DataType, Dict[str, Any]]:
        """
        Load spectral data cube from RICE compressed FITS files.
        
        For CHASE spectral data (_HA.fits, _FE.fits), the data is RICE compressed
        and stored in hdu[1]. For other spectral data, it may be in hdu[0].
        
        Args:
            filepath: Path to spectral data file
            data_type: Detected data type
            
        Returns:
            Spectral data cube and metadata
        """
        try:
            # Open file with minimal verification (like multipro_chase_LSF.py)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', VerifyWarning)
                
                hdul = fits.open(filepath, fix=True, ignore_missing_end=True, 
                               verify='silentfix', output_verify='ignore')
                
                # Determine which HDU contains the data
                # Based on multipro_chase_LSF.py, CHASE data is in hdu[1]
                if len(hdul) > 1 and hdul[1].data is not None:
                    logger.info("Loading data from hdu[1] (RICE compressed)")
                    raw_data = hdul[1].data
                    # Primary header is in hdu[0]
                    header = hdul[1].header
                    data_hdu = 1
                elif hdul[0].data is not None:
                    logger.info("Loading data from hdu[0]")
                    raw_data = hdul[0].data
                    header = hdul[0].header
                    data_hdu = 0
                else:
                    hdul.close()
                    raise ValueError(f"No valid data found in FITS file: {filepath}")
                
                original_shape = raw_data.shape
                logger.info(f"Original data shape: {original_shape}")
                
                # Check data dimensions and transpose if needed
                if len(original_shape) == 3:
                    # Determine correct dimension order
                    if original_shape[0] == 118:
                        # Shape is (118, 2313, 2304) - correct order
                        data = raw_data.astype(np.float32)
                        logger.info("Data shape is correct: (wavelength, y, x)")
                    elif original_shape[2] == 118:
                        # Shape is (2313, 2304, 118) - need to transpose
                        data = np.transpose(raw_data.astype(np.float32), (2, 0, 1))
                        logger.info("Transposed data to: (wavelength, y, x)")
                    else:
                        # Unknown order, assume first dimension is wavelength
                        data = raw_data.astype(np.float32)
                        logger.warning(f"Unknown dimension order, using as-is: {original_shape}")
                else:
                    hdul.close()
                    raise ValueError(f"Expected 3D spectral data, got {len(original_shape)}D")
                
                # Extract parameters from header (using safe extraction)
                params = self._extract_solar_params(header)
                params.update({
                    'filename': os.path.basename(filepath),
                    'data_type': data_type.value,
                    'data_hdu': data_hdu,
                    'original_shape': original_shape,
                    'n_wavelengths': data.shape[0],
                    'n_y': data.shape[1],
                    'n_x': data.shape[2],
                    'compression': 'RICE' if data_hdu == 1 else 'NONE'
                })
                
                hdul.close()
                logger.info(f"Successfully loaded spectral data with shape: {data.shape}")
                return data, header, data_type, params
                
        except Exception as e:
            logger.error(f"Failed to load spectral data {filepath}: {e}")
            raise
    
    def _load_doppler_data(self, filepath: str, data_type: DataType) -> Tuple[np.ndarray, fits.Header, DataType, Dict[str, Any]]:
        """
        Load Doppler velocity data from uncompressed FITS files.
        
        For Doppler velocity data (_Vdop.fits, etc.), the data is uncompressed
        and stored in hdu[0].
        
        Args:
            filepath: Path to Doppler data file
            data_type: Detected data type
            
        Returns:
            Doppler velocity map and metadata
        """
        try:
            # Open file with minimal verification
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', VerifyWarning)
                
                hdul = fits.open(filepath, fix=True, ignore_missing_end=True, 
                               verify='silentfix', output_verify='ignore')
                
                # Doppler velocity data is usually in hdu[0]
                if hdul[0].data is None:
                    # If hdu[0] is empty, try hdu[1] as fallback
                    if len(hdul) > 1 and hdul[1].data is not None:
                        logger.warning(f"Doppler data found in hdu[1] instead of hdu[0] for {filepath}")
                        data = hdul[1].data.astype(np.float32)
                        header = hdul[1].header
                    else:
                        hdul.close()
                        raise ValueError(f"No valid data found in FITS file: {filepath}")
                else:
                    data = hdul[0].data.astype(np.float32)
                    header = hdul[0].header
                
                # Extract solar parameters
                params = self._extract_solar_params(header)
                params.update({
                    'filename': os.path.basename(filepath),
                    'data_type': data_type.value,
                    'data_hdu': 0,
                    'compression': 'NONE'
                })
                
                hdul.close()
                logger.info(f"Loaded Doppler data with shape: {data.shape}")
                return data, header, data_type, params
                
        except Exception as e:
            logger.error(f"Failed to load Doppler data {filepath}: {e}")
            raise
    
    def _extract_solar_params(self, header: fits.Header) -> Dict[str, Any]:
        """
        Extract solar parameters from FITS header.
        
        This function safely extracts parameters from the header, handling
        potential non-standard characters by converting to appropriate types.
        
        Args:
            header: FITS header
            
        Returns:
            Dictionary of solar parameters
        """
        params = {}
        
        # Extract parameters with safe conversion
        # Based on multipro_chase_LSF.py
        
        # Solar center coordinates
        params['cx'] = self._safe_get_float(header, 'CRPIX1', default=0.0)
        params['cy'] = self._safe_get_float(header, 'CRPIX2', default=0.0)
        
        # Additional CHASE-specific center keywords (from multipro_chase_LSF.py)
        if 'CX' in header:
            params['cx'] = self._safe_get_float(header, 'CX', default=params['cx'])
        if 'CY' in header:
            params['cy'] = self._safe_get_float(header, 'CY', default=params['cy'])
        
        # Solar radius
        params['radius'] = self._safe_get_float(header, 'R_SUN', default=0.0)
        if 'RSUN' in header:
            params['radius'] = self._safe_get_float(header, 'RSUN', default=params['radius'])
        if 'RADII' in header:
            params['radius'] = self._safe_get_float(header, 'RADII', default=params['radius'])
        
        # Instrument rotation (P0)
        params['p0'] = self._safe_get_float(header, 'INST_ROT', default=self.config.instrument_rotation)
        
        # Solar B0 angle
        params['b0'] = self._safe_get_float(header, 'B0', default=0.0)
        
        # Observation date
        params['date_obs'] = self._safe_get_string(header, 'DATE-OBS', default='')
        
        # Exposure time
        params['exposure_time'] = self._safe_get_float(header, 'EXPTIME', default=0.0)
        
        # Additional parameters from multipro_chase_LSF.py
        if 'STEPTIME' in header:
            params['step_time'] = self._safe_get_float(header, 'STEPTIME')
        if 'STR_TIME' in header:
            params['str_time'] = self._safe_get_string(header, 'STR_TIME')
        if 'END_TIME' in header:
            params['end_time'] = self._safe_get_string(header, 'END_TIME')
        if 'SAT_POS1' in header:
            params['sat_pos1'] = self._safe_get_float(header, 'SAT_POS1')
        if 'SAT_POS2' in header:
            params['sat_pos2'] = self._safe_get_float(header, 'SAT_POS2')
        if 'SAT_POS3' in header:
            params['sat_pos3'] = self._safe_get_float(header, 'SAT_POS3')
        
        # Spatial scales
        params['cdelt1'] = self._safe_get_float(header, 'CDELT1', default=0.0)
        params['cdelt2'] = self._safe_get_float(header, 'CDELT2', default=0.0)
        
        # Reference pixel values
        params['crval1'] = self._safe_get_float(header, 'CRVAL1', default=0.0)
        params['crval2'] = self._safe_get_float(header, 'CRVAL2', default=0.0)
        
        # Solar radius in arcsec
        params['rsun_obs'] = self._safe_get_float(header, 'RSUN_OBS', default=960.0)
        
        logger.debug(f"Extracted solar params: cx={params['cx']}, cy={params['cy']}, radius={params['radius']}")
        
        return params
    
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
                # Convert to float if it's a string
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
        except Exception as e:
            logger.debug(f"Failed to get float value for {key}: {e}")
        
        return default
    
    def _safe_get_string(self, header: fits.Header, key: str, default: str = '') -> str:
        """
        Safely get a string value from FITS header.
        
        Args:
            header: FITS header
            key: Header keyword
            default: Default value if extraction fails
            
        Returns:
            String value
        """
        try:
            if key in header:
                value = str(header[key])
                # Remove any non-printable characters
                import string
                printable = set(string.printable)
                return ''.join(filter(lambda x: x in printable, value))
        except Exception as e:
            logger.debug(f"Failed to get string value for {key}: {e}")
        
        return default
    
    def needs_preprocessing(self, data_type: DataType) -> bool:
        """
        Check if data needs preprocessing.
        
        Args:
            data_type: Data type to check
            
        Returns:
            True if preprocessing is needed
        """
        return data_type in [DataType.SPECTRAL_HA, DataType.SPECTRAL_FE]
    
    def generate_output_filename(self, input_filepath: str, 
                                output_suffix: str = "_Vdop") -> str:
        """
        Generate output filename for processed data.
        
        Args:
            input_filepath: Input file path
            output_suffix: Output suffix
            
        Returns:
            Output file path
        """
        input_path = Path(input_filepath)
        
        # Remove existing suffix
        stem = input_path.stem
        
        # Handle compressed files
        if stem.endswith('.fits'):
            stem = stem[:-5]
        
        # Remove existing type identifiers
        for suffix in ['_HA', '_FE', '_ha', '_fe']:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        
        # Add new suffix
        new_stem = f"{stem}{output_suffix}.fits"
        
        return str(input_path.parent / new_stem)