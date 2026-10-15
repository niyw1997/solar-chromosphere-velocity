"""Physical and instrument constants (v2).

All velocities are km/s unless the name says otherwise.
"""

import numpy as np

# --- fundamental ---------------------------------------------------------
C_KM_S = 299792.458                 # speed of light, km/s
R_SUN_KM = 6.957e5                  # nominal solar radius, km (IAU 2015)
GM_SUN_M3_S2 = 1.32712440018e20     # heliocentric gravitational constant, m^3/s^2

# --- solar gravitational redshift ---------------------------------------
# GM/(R c) with the nominal radius above -> 0.6363 km/s.
# NOTE: this is a *diagnostic* constant only.  It must never be subtracted
# from the Doppler maps: the inversion template is the disk-centre average
# profile, so any absolute (gravitational + convective) line shift is already
# contained in the template.  See docs/OUTPUT.md and the v2 audit notes.
GRAVITATIONAL_REDSHIFT_KM_S = GM_SUN_M3_S2 / ((R_SUN_KM * 1e3) * (C_KM_S * 1e3))

# v1 subtracted 632 m/s from every map.  Kept only so that --legacy can
# reproduce v1 bit-for-bit.
LEGACY_ZERO_POINT_KM_S = 0.632

# --- rest wavelengths (air) ---------------------------------------------
# CHASE/HIS works in air wavelengths.  Verified against real headers:
#   HA  : WAVE_LEN = 6562.82, SPECLINE = 'HA'
#   FEI : WAVE_LEN = 6569.22, SPECLINE = 'FEI'
REST_WAVELENGTH_AIR = {
    "HALPHA": 6562.80,
    "FEI": 6569.22,
}

# map a FITS SPECLINE value onto a rest wavelength
SPECLINE_ALIASES = {
    "HA": "HALPHA",
    "HALPHA": "HALPHA",
    "H-ALPHA": "HALPHA",
    "FE": "FEI",
    "FEI": "FEI",
    "FE I": "FEI",
}


def rest_wavelength(spec_line):
    """Rest wavelength (air, Angstrom) for a FITS SPECLINE value, or None."""
    if spec_line is None:
        return None
    key = str(spec_line).strip().upper()
    key = SPECLINE_ALIASES.get(key, key)
    return REST_WAVELENGTH_AIR.get(key)


def velocity_scale_km_s_per_pixel(cdelt3, wave_len):
    """km/s per spectral pixel for a grating sampling ``cdelt3`` (Angstrom/pix).

    ``wave_len`` is the line-centre wavelength in Angstrom (header WAVE_LEN or
    the rest wavelength).  This replaces v1's hard-coded ``1.1068*2``.
    """
    cdelt3 = float(cdelt3)
    wave_len = float(wave_len)
    if not np.isfinite(cdelt3) or cdelt3 <= 0:
        raise ValueError("CDELT3 must be a positive number, got %r" % (cdelt3,))
    if not np.isfinite(wave_len) or wave_len <= 0:
        raise ValueError("WAVE_LEN must be a positive number, got %r" % (wave_len,))
    return C_KM_S * cdelt3 / wave_len


# v1's hard-coded value, retained for --legacy parity.
LEGACY_VELOCITY_SCALE_KM_S_PER_PIXEL = 1.1068 * 2.0
