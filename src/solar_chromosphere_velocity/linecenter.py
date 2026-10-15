"""Spectral line-centre location.

v1 fitted a Gaussian to ``avg[45:90]`` and, whenever the profile had fewer than
90 samples, silently returned ``len//2``.  For CHASE Fe I cubes (46 samples)
that put the window 12.3 pixels (~27 km/s) away from the real line core.

v2 locates the core from the data with the WCS as a *hint* only:

1. ``wcs_line_center`` gives an initial guess from CRVAL3/CDELT3/WAVE_LEN;
2. the local minimum of the average profile in a window around that guess;
3. a Gaussian refinement around that minimum.

No length threshold is used; short profiles are handled by shrinking the
refinement window.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit

from .constants import rest_wavelength, velocity_scale_km_s_per_pixel

__all__ = ["LineCenter", "wcs_line_center", "fit_line_center"]


@dataclass
class LineCenter:
    index: int                 # integer sample index used for the window
    continuous: float          # sub-sample centre from the Gaussian refinement
    method: str                # 'gauss' | 'min' | 'midpoint' | 'wcs'
    wcs_guess: Optional[float] = None
    wcs_offset_px: Optional[float] = None   # data - WCS, in pixels
    wcs_offset_km_s: Optional[float] = None
    gauss_params: Optional[np.ndarray] = None
    n_samples: int = 0

    def as_dict(self):
        return {
            "index": self.index,
            "continuous": self.continuous,
            "method": self.method,
            "wcs_guess": self.wcs_guess,
            "wcs_offset_px": self.wcs_offset_px,
            "wcs_offset_km_s": self.wcs_offset_km_s,
            "n_samples": self.n_samples,
        }


def wcs_line_center(params, rest=None) -> Optional[float]:
    """Line-centre sample index implied by the wavelength solution.

    ``lambda(i) = CRVAL3 + (i - (CRPIX3 - 1)) * CDELT3``  ->  solve for i.
    Returns ``None`` when the header lacks the required keywords.

    NOTE: for the 20260424 level-1 file this gives ~69.0 while the observed
    core sits at ~70.3, i.e. the absolute wavelength solution and the data
    disagree by ~1.4 px (~3 km/s).  That is exactly why the WCS value is used
    only as a hint, and why V_OFFSET must not be read as a gravitational
    redshift.
    """
    if params is None:
        return None
    cdelt3 = getattr(params, "cdelt3", None)
    crval3 = getattr(params, "crval3", None)
    if not cdelt3 or crval3 is None:
        return None
    lam0 = getattr(params, "wave_len", None) or rest or rest_wavelength(
        getattr(params, "spec_line", ""))
    if not lam0:
        return None
    crpix3 = getattr(params, "crpix3", None)
    if crpix3 is None:                      # note: 0.0 is a legitimate value
        crpix3 = 1.0
    index = (float(lam0) - float(crval3)) / float(cdelt3) + (float(crpix3) - 1.0)
    return float(index)


def _gauss(x, *p):
    return p[0] * np.e ** (-(x - p[1]) ** 2 / p[2]) + p[3]


def _legacy_center(profile: np.ndarray, min_len: int) -> LineCenter:
    """Reproduce v1 `_fit_line_center` exactly (45:90 segment, len<90 fallback)."""
    n = len(profile)
    if n < min_len:
        hc = n // 2
        return LineCenter(index=int(hc), continuous=float(hc), method="midpoint",
                          n_samples=n)
    start, end = 45, 90
    if end > n:
        end = n
        start = max(0, end - 45)
    lx = np.arange(end - start)
    ly = profile[start:end]
    try:
        para, _ = curve_fit(_gauss, lx, ly, [-1400, 22, 200, 2000], method="lm")
        hc = int(round(para[1])) + start
        hc = max(10, min(n - 11, hc))
        return LineCenter(index=int(hc), continuous=float(para[1]) + start,
                          method="gauss", gauss_params=np.asarray(para), n_samples=n)
    except Exception:
        hc = int(np.argmin(profile[start:end])) + start
        return LineCenter(index=int(hc), continuous=float(hc), method="min", n_samples=n)


def fit_line_center(profile: np.ndarray, params=None, *, legacy: bool = False,
                    min_len: int = 90, rest=None,
                    window_half: Optional[int] = None) -> LineCenter:
    """Locate the absorption line centre of an average profile."""
    profile = np.asarray(profile, dtype=np.float64)
    n = len(profile)
    if n < 3:
        raise ValueError("profile too short to locate a line centre: %d samples" % n)

    if legacy:
        return _legacy_center(profile, min_len)

    guess = wcs_line_center(params, rest=rest)
    # search half-width: shrink automatically for short profiles
    half = window_half if window_half is not None else max(4, min(45, n // 4))
    i_glob = int(np.argmin(profile))
    i_min = i_glob
    if guess is not None and np.isfinite(guess):
        g = int(np.clip(round(guess), 0, n - 1))
        lo, hi = max(0, g - half), min(n, g + half + 1)
        # If the global minimum lies outside the WCS neighbourhood the profile
        # probably contains more than one feature; trust the header then.
        if not (lo <= i_glob < hi):
            i_min = int(np.argmin(profile[lo:hi])) + lo
    else:
        lo, hi = 0, n

    # Gaussian refinement around the local minimum
    w = max(2, min(half, n // 3))
    a, b = max(0, i_min - w), min(n, i_min + w + 1)
    x = np.arange(b - a, dtype=np.float64)
    y = profile[a:b]
    cont = float(np.max(y))
    amp = float(np.min(y)) - cont
    p0 = [amp, i_min - a, float(max(1.0, (w / 2.0) ** 2)), cont]
    center = LineCenter(index=i_min, continuous=float(i_min), method="min",
                        wcs_guess=guess, n_samples=n)
    try:
        para, _ = curve_fit(_gauss, x, y, p0=p0, maxfev=20000)
        if np.isfinite(para[1]) and -0.5 <= para[1] <= (b - a) + 0.5:
            center.continuous = float(para[1]) + a
            center.index = int(np.clip(round(center.continuous), 1, n - 2))
            center.method = "gauss"
            center.gauss_params = np.asarray(para)
    except Exception:
        pass

    if guess is not None and np.isfinite(guess):
        center.wcs_offset_px = center.continuous - float(guess)
        cdelt3 = getattr(params, "cdelt3", None)
        wave_len = (getattr(params, "wave_len", None) or rest
                    or rest_wavelength(getattr(params, "spec_line", "")))
        if cdelt3 and wave_len:
            center.wcs_offset_km_s = center.wcs_offset_px * \
                velocity_scale_km_s_per_pixel(cdelt3, wave_len)
    return center
