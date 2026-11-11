"""Removal of the scan-saddle and flat-field terms from ``V_RESID``.

The level-1 residual still contains two detector-scale systematics that are not
part of the large-scale flow model:

``V_SCAN_SADDLE``
    A full-disk quadratic surface (constant, linear and quadratic terms in the
    normalised disk coordinates).  It absorbs the scan-saddle pattern left by the
    raster scan and any residual zero-point gradient that the nine-parameter
    model cannot represent.

``V_FLAT``
    Parallel stripes (interference fringes) at a fixed wavelength and
    orientation, removed either by a Fourier band-stop (``"band"``) or by
    demodulating the stripe phase and smoothing its envelope (``"envelope"``).

Their sum is the correction

    ``V_SS = V_SCAN_SADDLE + V_FLAT``

and the local velocity written to the product is

    ``V_LOCAL = V_RESID - V_SS``

Active regions carry real signal and must not be erased by the flat-field term.
The flat field is therefore multiplied by a smooth weight that falls to zero
where the smoothed absolute velocity is large, so ``V_LOCAL`` keeps the blobs.

All the routines operate on a 2-D velocity map plus its header and return arrays
of the same shape.  Nothing here changes the spectral inversion; the correction
is applied to the residual only.

The equations, the taper, the band-stop kernel, the envelope demodulation and
the protection weight follow the method described in the manual, section
"Scan-saddle and flat-field removal".
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import (distance_transform_edt, gaussian_filter,
                           maximum_filter)

__all__ = [
    "ScanCorrection",
    "detect_stripe_peaks",
    "scan_saddle",
    "stripe_band",
    "stripe_envelope",
    "protect_weight",
    "second_pass",
    "scan_correction",
]

#: protection presets, per spectral line: (lo_pct, hi_pct, strength)
PROTECTION_PRESETS = {
    "HA": (90.0, 99.0, 1.0),     # H-alpha is structured, protect strongly
    "FEI": (98.0, 99.8, 0.6),    # Fe I keeps blobs, weaker suppression
}
DEFAULT_PROTECTION = (95.0, 99.5, 1.0)


@dataclass
class LocalSecondPass:
    """What a second fringe pass removed from an already corrected map."""

    flat: np.ndarray                       # the extra term that was removed
    local: np.ndarray                      # the map after that removal
    stripe_peaks: Tuple[Tuple[float, float], ...]
    n_detected: int
    contrast: Dict[Tuple[float, float], float]


@dataclass
class ScanCorrection:
    """The correction surfaces plus what was actually fitted."""

    scan_saddle: np.ndarray          # quadratic surface, km/s
    flat: np.ndarray                 # stripe template after protection
    flat_raw: np.ndarray             # stripe template before protection
    ss: np.ndarray                   # scan_saddle + flat
    local: np.ndarray                # V_RESID - ss
    coeffs: np.ndarray               # 6 quadratic coefficients
    stripe_peaks: Tuple[Tuple[float, float], ...]
    second_pass_ratios: Dict[Tuple[float, float], float]
    protect_weight: np.ndarray
    n_detected: int = 0
    mode: str = "band"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _geometry(header, shape):
    """Disk-centre pixel and radius, tolerating either header spelling."""
    ny, nx = shape
    def get(*keys, default=None):
        for key in keys:
            if key in header:
                try:
                    return float(header[key])
                except (TypeError, ValueError):
                    continue
        return default
    cx = get("CX")
    if cx is None:
        crpix1 = get("CRPIX1", default=(nx + 1) / 2.0)
        cx = crpix1 - 1.0
    cy = get("CY")
    if cy is None:
        crpix2 = get("CRPIX2", default=(ny + 1) / 2.0)
        cy = crpix2 - 1.0
    radius = get("R_SUN", "RSUN_OBS", default=None)
    if not radius:
        radius = min(ny, nx) / 2.0
    return cx, cy, abs(radius)


def _taper(mask, edge_px: float = 25.0, smooth: float = 5.0):
    """Cosine-like weight that goes to zero at the edge of the valid area."""
    dist = distance_transform_edt(mask)
    tap = np.clip(dist / edge_px, 0.0, 1.0)
    return gaussian_filter(tap, smooth)


def _k_grid(shape):
    ny, nx = shape
    kx = (np.arange(nx) - nx // 2) / nx
    ky = (np.arange(ny) - ny // 2) / ny
    return np.meshgrid(kx, ky)


# ---------------------------------------------------------------------------
# the two terms
# ---------------------------------------------------------------------------
def scan_saddle(velocity, header):
    """Fit and return the full-disk quadratic surface ``V_SCAN_SADDLE``.

    The basis is ``[1, Y, Y^2, X, X*Y, X^2]`` with ``X = (x - xc) / R`` and
    ``Y = (y - yc) / R``, so the coefficients are in km/s and do not depend on
    the image size.  Only finite pixels take part.
    """
    d = np.asarray(velocity, dtype=np.float64)
    ny, nx = d.shape
    finite = np.isfinite(d)
    if finite.sum() < 12:
        return np.zeros_like(d), np.zeros(6)
    cx, cy, radius = _geometry(header, (ny, nx))
    y, x = np.mgrid[0:ny, 0:nx]
    X = (x - cx) / radius
    Y = (y - cy) / radius
    terms = [np.ones_like(X), Y, Y * Y, X, X * Y, X * X]
    A = np.stack([t[finite] for t in terms], axis=1)
    coeffs, *_ = np.linalg.lstsq(A, d[finite], rcond=None)
    surface = sum(c * t for c, t in zip(coeffs, terms))
    return np.where(finite, surface, np.nan), coeffs


def _prepared(velocity):
    """Taper, FFT and the taper itself, so several notches share one transform."""
    d = np.asarray(velocity, dtype=np.float64)
    finite = np.isfinite(d)
    tap = _taper(finite)
    a = np.where(finite, np.nan_to_num(d), 0.0) * tap
    F = np.fft.fftshift(np.fft.fft2(a))
    return F, tap, finite


def _notch_spectrum(F, shape, lam: float, ang: float, width: float):
    """Multiply the spectrum by a Gaussian notch pair at +/-k0 (no inverse yet)."""
    ny, nx = shape
    KX, KY = _k_grid((ny, nx))
    k0 = 1.0 / float(lam)
    theta = np.deg2rad(float(ang))
    kx0, ky0 = k0 * np.cos(theta), k0 * np.sin(theta)
    # a notch much narrower than one k-grid step removes nothing, which matters
    # for the full-image-scale stripes (lambda of order the image size)
    sigma = max(float(width) * k0, 1.2 / max(nx, ny))
    G = (np.exp(-((KX + kx0) ** 2 + (KY + ky0) ** 2) / (2 * sigma ** 2))
         + np.exp(-((KX - kx0) ** 2 + (KY - ky0) ** 2) / (2 * sigma ** 2)))
    return F * G


def _finish(spectrum_sum, tap):
    """Inverse transform and undo the taper."""
    S = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum_sum)))
    return np.where(tap > 0.05, S / np.maximum(tap, 1e-6), 0.0)


def stripe_bands(velocity, peaks, width: float = 0.45, prepared=None):
    """Sum of the band-stop templates at several wave vectors, one FFT."""
    if not peaks:
        return np.zeros_like(np.asarray(velocity, dtype=np.float64))
    F, tap, finite = prepared if prepared is not None else _prepared(velocity)
    total = None
    for lam, ang in peaks:
        part = _notch_spectrum(F, finite.shape, lam, ang, width)
        total = part if total is None else total + part
    return _finish(total, tap)


def stripe_band(velocity, lam: float, ang: float, width: float = 0.45):
    """Fourier band-stop at wavelength ``lam`` (pixels) and ``ang`` (degrees)."""
    return stripe_bands(velocity, [(lam, ang)], width)


def stripe_envelope(velocity, lam: float, ang: float, sigma: float = 80.0):
    """Envelope template: demodulate the stripe phase, smooth, re-modulate."""
    d = np.asarray(velocity, dtype=np.float64)
    ny, nx = d.shape
    finite = np.isfinite(d)
    y, x = np.mgrid[0:ny, 0:nx]
    k0 = 1.0 / float(lam)
    theta = np.deg2rad(float(ang))
    phase = k0 * np.cos(theta) * x + k0 * np.sin(theta) * y
    C = np.where(finite, np.nan_to_num(d), 0.0) * np.exp(-1j * phase)
    num = gaussian_filter(np.real(C), sigma) + 1j * gaussian_filter(np.imag(C), sigma)
    den = gaussian_filter(finite.astype(float), sigma)
    Cs = num / np.maximum(den, 1e-6)
    return np.where(finite, 2.0 * np.real(Cs * np.exp(1j * phase)), 0.0)


def protect_weight(velocity, lo_pct: float = 95.0, hi_pct: float = 99.5,
                   strength: float = 1.0, sigma: float = 8.0,
                   smooth: float = 12.0):
    """Smooth weight in [0, 1] that suppresses the flat field in active regions.

    ``strength = 1`` removes the flat-field term completely where the smoothed
    absolute velocity is in the top ``hi_pct`` percentile; ``strength < 1``
    leaves a floor so part of the stripe is still removed there.
    """
    d = np.asarray(velocity, dtype=np.float64)
    finite = np.isfinite(d)
    if finite.sum() == 0:
        return np.zeros_like(d)
    smoothed = np.abs(gaussian_filter(np.nan_to_num(d), sigma))
    lo = float(np.percentile(smoothed[finite], lo_pct))
    hi = float(np.percentile(smoothed[finite], hi_pct))
    if hi <= lo:
        return np.where(finite, 1.0, 0.0)
    w = np.clip((hi - smoothed) / (hi - lo), 0.0, 1.0)
    w = 1.0 - strength * (1.0 - w)
    w = gaussian_filter(w, smooth)
    return np.where(finite, w, 0.0)


def second_pass(before, after, peaks: Sequence[Tuple[float, float]],
                ratio_threshold: float = 0.4, sigma: float = 40.0,
                prepared_before=None, prepared_after=None,
                width: float = 0.45):
    """Subtract residual stripe peaks, but only where they survived.

    ``ratio_threshold`` is the smallest after/before stripe RMS ratio that
    triggers a second pass; a ratio near 1 means the wave vector was not removed.
    """
    extra = np.zeros_like(np.asarray(before, dtype=np.float64))
    ratios: Dict[Tuple[float, float], float] = {}
    if peaks:
        # one transform per map, reused for every wave vector
        F_before, tap_b, finite_b = (prepared_before if prepared_before is not None
                                     else _prepared(before))
        F_after, tap_a, finite_a = (prepared_after if prepared_after is not None
                                    else _prepared(after))
        for lam, ang in peaks:
            b1 = _finish(_notch_spectrum(F_before, finite_b.shape, lam, ang, width), tap_b)
            b2 = _finish(_notch_spectrum(F_after, finite_a.shape, lam, ang, width), tap_a)
            s1 = float(np.nanstd(b1))
            ratio = float(np.nanstd(b2)) / s1 if s1 > 0 else 0.0
            ratios[(float(lam), float(ang))] = ratio
            if ratio <= ratio_threshold:
                continue
            amp = np.sqrt(gaussian_filter(b2 * b2, sigma))
            p99 = float(np.nanpercentile(amp, 99))
            if p99 <= 0:
                continue
            w = np.clip(amp / (0.5 * p99), 0.0, 1.0)
            w = gaussian_filter(w, 10)
            extra += w * b2
    return extra, ratios


# ---------------------------------------------------------------------------
# automatic stripe detection
# ---------------------------------------------------------------------------
def detect_stripe_peaks(velocity, *, lambda_min: float = 30.0,
                        lambda_max: float = 1500.0, topn: int = 2,
                        min_contrast: float = 8.0,
                        max_peaks: int = 12, prepared=None,
                        angle_tolerance: float = 10.0,
                        angle_center: Optional[float] = None,
                        contrast_out: Optional[Dict] = None
                        ) -> Tuple[Tuple[float, float], ...]:
    """Find the strongest parallel-stripe wave vectors in a velocity map.

    The 2-D power spectrum is computed on a tapered map, local maxima are
    collected, and only peaks whose wavelength falls inside
    ``[lambda_min, lambda_max]`` and whose power exceeds ``min_contrast`` times
    the median power in that annulus are kept.  The ``topn`` strongest are
    returned as ``(lambda_px, angle_deg)``.

    The 30-1500 px interval covers the fringes this instrument produces, which
    are periodic, parallel and image-wide.

    ``angle_center`` additionally restricts the search to wave vectors within
    ``angle_tolerance`` of that orientation, and measures the contrast against
    the median power inside that restricted band.  A second pass uses it to look
    for what a first pass left along the known fringe direction, where a free
    search would also fire on the peaks of a pure noise field.
    """
    d = np.asarray(velocity, dtype=np.float64)
    ny, nx = d.shape
    finite = np.isfinite(d)
    if finite.sum() < 4 * 64:
        return ()
    if prepared is not None:
        F = prepared[0]
    else:
        tap = _taper(finite)
        a = np.where(finite, np.nan_to_num(d), 0.0) * tap
        F = np.fft.fftshift(np.fft.fft2(a))
    P = np.abs(F) ** 2
    KX, KY = _k_grid((ny, nx))
    K = np.hypot(KX, KY)
    with np.errstate(divide="ignore"):
        lam = np.where(K > 0, 1.0 / np.maximum(K, 1e-12), np.inf)
    band = (lam >= lambda_min) & (lam <= lambda_max)
    if angle_center is not None:
        ang = np.degrees(np.arctan2(KY, KX))
        ang = (ang + 90.0) % 180.0 - 90.0
        off = np.abs(((ang - float(angle_center) + 90.0) % 180.0) - 90.0)
        band = band & (off <= float(angle_tolerance))
    if not band.any():
        return ()
    local_max = (P == maximum_filter(P, size=9)) & band
    ys, xs = np.nonzero(local_max)
    if ys.size == 0:
        return ()
    powers = P[ys, xs]
    median = float(np.median(P[band]))
    keep = powers > min_contrast * max(median, 1e-30)
    ys, xs, powers = ys[keep], xs[keep], powers[keep]
    if ys.size == 0:
        return ()
    order = np.argsort(powers)[::-1]
    peaks = []
    seen = set()
    dominant = None
    for i in order:
        if len(peaks) >= int(topn):
            break
        j, k = int(ys[i]), int(xs[i])
        lam0 = float(lam[j, k])
        # a stripe at angle and at angle + 180 degrees is the same stripe
        angle = float(np.degrees(np.arctan2(KY[j, k], KX[j, k])))
        angle = (angle + 90.0) % 180.0 - 90.0
        # the fringes are parallel, so every vector must share one orientation
        if dominant is None:
            dominant = angle
        elif abs(((angle - dominant + 90.0) % 180.0) - 90.0) > float(angle_tolerance):
            continue
        key = (round(lam0, 1), round(angle, 1))
        if key in seen:
            continue
        seen.add(key)
        peaks.append((lam0, angle))
        if contrast_out is not None:
            contrast_out[(lam0, angle)] = float(powers[i] / max(median, 1e-30))
    return tuple(peaks)


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------
def scan_correction(velocity, header, *, enabled: bool = True,
                    use_scan_saddle: bool = True,
                    stripe_removal: bool = True,
                    stripe_peaks: Optional[Iterable[Sequence[float]]] = None,
                    stripe_mode: str = "band",
                    lambda_min: float = 25.0, lambda_max: float = 150.0,
                    topn: int = 3, band_width: float = 0.45,
                    envelope_sigma: float = 80.0,
                    second_peaks: Optional[Iterable[Sequence[float]]] = None,
                    second_ratio: float = 0.4,
                    second_auto: bool = True,
                    flat_strength: float = 1.0,
                    protect: Optional[Sequence[float]] = None,
                    protect_sigma: float = 8.0,
                    protect_smooth: float = 12.0,
                    spec_line: str = "") -> ScanCorrection:
    """Compute ``V_SS`` and ``V_LOCAL`` for one residual map.

    ``stripe_peaks`` given explicitly is used as is; otherwise the wave vectors
    are detected from the map.  ``protect`` is ``(lo_pct, hi_pct, strength)`` and
    defaults to the preset for the spectral line.

    The strength of the flat-field removal is controlled by ``band_width`` (how
    wide the notch is in units of the central wavenumber), ``topn`` (how many
    stripe wave vectors are removed), ``second_auto`` (re-detect and remove what
    survived the first pass) and ``flat_strength`` (a final multiplier on the
    template).  ``protect_sigma`` smooths the velocity before the protection
    threshold is applied and ``protect_smooth`` dilates the protected area, so
    the active regions are spared over a larger footprint than the raw blob.
    """
    d = np.asarray(velocity, dtype=np.float64)
    finite = np.isfinite(d)
    zeros = np.zeros_like(d)

    if not enabled or finite.sum() == 0:
        return ScanCorrection(
            scan_saddle=zeros, flat=zeros, flat_raw=zeros, ss=zeros,
            local=np.where(finite, d, np.nan), coeffs=np.zeros(6),
            stripe_peaks=(), second_pass_ratios={},
            protect_weight=np.where(finite, 1.0, 0.0), n_detected=0,
            mode=stripe_mode)

    # 1) the quadratic scan saddle
    if use_scan_saddle:
        saddle, coeffs = scan_saddle(d, header)
    else:
        saddle, coeffs = zeros, np.zeros(6)

    # 2) the stripe template, after removing the saddle
    residual = d - saddle
    peaks: Tuple[Tuple[float, float], ...] = ()
    flat_raw = zeros
    ratios: Dict[Tuple[float, float], float] = {}
    if stripe_removal:
        # One spectrum of the residual serves detection, the first band-stop and
        # the second pass, so the heavy 2-D FFT is done once per map.
        prepared = _prepared(residual)
        if stripe_peaks:
            peaks = tuple((float(a), float(b)) for a, b in stripe_peaks)
        else:
            peaks = detect_stripe_peaks(residual, lambda_min=lambda_min,
                                        lambda_max=lambda_max, topn=topn,
                                        prepared=prepared)
        if peaks:
            if stripe_mode == "envelope":
                for lam, ang in peaks:
                    flat_raw = flat_raw + stripe_envelope(
                        residual, lam, ang, envelope_sigma)
            else:
                flat_raw = stripe_bands(residual, peaks, band_width,
                                        prepared=prepared)
            after = residual - flat_raw
            prepared_after = _prepared(after)

            # explicit peaks: per-vector ratio test
            if second_peaks:
                extra, ratios = second_pass(
                    residual, after,
                    [(float(a), float(b)) for a, b in second_peaks],
                    ratio_threshold=second_ratio, width=band_width,
                    prepared_before=prepared, prepared_after=prepared_after)
                flat_raw = flat_raw + extra

            # automatic second pass: whatever is still visible in the spectrum
            # of the corrected map is removed with a local amplitude weight
            if second_auto and stripe_mode != "envelope":
                residual_peaks = detect_stripe_peaks(
                    after, lambda_min=lambda_min, lambda_max=lambda_max,
                    topn=topn, prepared=prepared_after)
                residual_peaks = tuple(p for p in residual_peaks
                                       if p not in peaks)
                if residual_peaks:
                    bands = stripe_bands(after, residual_peaks, band_width,
                                         prepared=prepared_after)
                    amp = np.sqrt(gaussian_filter(bands * bands, 40.0))
                    p99 = float(np.nanpercentile(amp, 99)) if np.isfinite(amp).any() else 0.0
                    if p99 > 0:
                        w2 = np.clip(amp / (0.5 * p99), 0.0, 1.0)
                        w2 = gaussian_filter(w2, 10)
                        flat_raw = flat_raw + w2 * bands
                        ratios.update({p: 1.0 for p in residual_peaks})

    # 3) protect active regions and assemble
    lo, hi, strength = (tuple(protect) if protect is not None
                        else PROTECTION_PRESETS.get(str(spec_line).upper(),
                                                    DEFAULT_PROTECTION))
    weight = protect_weight(d, lo, hi, strength, protect_sigma, protect_smooth)
    flat = float(flat_strength) * weight * flat_raw
    ss = saddle + flat
    local = np.where(finite, d - ss, np.nan)
    return ScanCorrection(
        scan_saddle=saddle, flat=flat, flat_raw=flat_raw, ss=ss, local=local,
        coeffs=coeffs, stripe_peaks=peaks, second_pass_ratios=ratios,
        protect_weight=weight, n_detected=len(peaks), mode=stripe_mode)


# ---------------------------------------------------------------------------
# two-arm combination: a strong and a weak estimate of the same flat field
# ---------------------------------------------------------------------------
def blend_stripe_fields(strong, weak, *, mode: str = "agreement",
                        weight: float = 0.5, smooth: float = 12.0,
                        prepared_from=None):
    """Combine a strong and a weak flat-field estimate into one stripe field.

    ``mode="mean"`` averages the two arms with a fixed ``weight`` on the strong
    one.  ``mode="agreement"`` keeps the strong arm only where it does not add
    much to the weak one, that is where both agree that a fringe is present, and
    falls back to the weak arm where the strong one reaches into structure.  The
    weight is then smoothed with ``smooth`` pixels so the two arms join
    gradually instead of switching pixel by pixel.

    The weight is built from the *unprotected* templates (``strong``/``weak``
    should be the raw fields) and applied to the protected pair, so active
    regions, where both arms vanish, do not bias it.
    """
    fs = np.asarray(strong, dtype=np.float64)
    fw = np.asarray(weak, dtype=np.float64)
    if mode == "mean":
        w = np.full(fs.shape, float(weight))
    elif mode == "agreement":
        a = np.abs(fw)
        d = np.abs(fs - fw)
        w = a ** 2 / (a ** 2 + d ** 2 + 1e-12)
    else:
        raise ValueError("blend mode must be 'mean' or 'agreement', got %r" % (mode,))
    finite = np.isfinite(w)
    w = gaussian_filter(np.where(finite, w, 0.0), float(smooth))
    w = np.clip(w, 0.0, 1.0)
    out = (1.0 - w) * fw + w * fs
    return np.where(np.isfinite(out), out, np.nan)


def scan_correction_two_arm(velocity, header, *, strong=None, weak=None,
                            mode: str = "agreement", weight: float = 0.5,
                            smooth: float = 12.0, **kwargs) -> ScanCorrection:
    """Run the removal twice and blend the two flat-field estimates.

    ``strong`` and ``weak`` are dicts of keyword overrides on top of ``kwargs``;
    the same wave vectors, protection preset and saddle fit are shared, so the
    two arms differ only in how hard they remove.
    """
    strong = dict(strong or {})
    weak = dict(weak or {})
    kw_strong = {**kwargs, **strong}
    kw_weak = {**kwargs, **weak}
    hi = scan_correction(velocity, header, enabled=True, **kw_strong)
    lo = scan_correction(velocity, header, enabled=True, **kw_weak)

    flat_raw = blend_stripe_fields(hi.flat_raw, lo.flat_raw, mode=mode,
                                   weight=weight, smooth=smooth)
    wprot = hi.protect_weight
    flat = wprot * flat_raw
    saddle = hi.scan_saddle
    d = np.asarray(velocity, dtype=np.float64)
    finite = np.isfinite(d)
    ss = saddle + flat
    local = np.where(finite, d - ss, np.nan)
    return ScanCorrection(
        scan_saddle=saddle, flat=flat, flat_raw=flat_raw, ss=ss, local=local,
        coeffs=hi.coeffs, stripe_peaks=hi.stripe_peaks,
        second_pass_ratios=hi.second_pass_ratios, protect_weight=wprot,
        n_detected=hi.n_detected, mode="%s+%s" % (hi.mode, mode))


def second_pass_local(local, header, *, protect=None, spec_line: str = "",
                      lambda_min: float = 30.0, lambda_max: float = 1500.0,
                      topn: int = 2, angle_center=None,
                      angle_tolerance: float = 8.0,
                      min_contrast: float = 40.0, band_width: float = 0.20,
                      protect_sigma: float = 8.0, protect_smooth: float = 12.0,
                      exclude: Iterable[Sequence[float]] = ()
                      ) -> LocalSecondPass:
    """Remove what a first fringe pass left, working on the corrected map.

    A first pass removes the fringes it can characterise; what remains along the
    same orientation is sometimes still visible.  This second pass looks for it
    on the already corrected map, with two safeguards that a free search cannot
    offer.

    ``angle_center`` locks the search to the fringe direction found by the first
    pass, and ``min_contrast`` is raised well above the first-pass value.  On a
    map that has already been corrected the power is close to noise, and a free
    search fires on noise peaks at nine to sixteen times the annulus median; a
    genuine residual fringe in the 2026-11-04 data sits above one hundred.
    ``exclude`` drops wave vectors already removed, so nothing is subtracted
    twice.
    """
    d = np.asarray(local, dtype=np.float64)
    finite = np.isfinite(d)
    empty = LocalSecondPass(np.zeros_like(d), d, (), 0, {})
    if finite.sum() < 4 * 64:
        return empty

    prepared = _prepared(d)
    measured: Dict = {}
    peaks = detect_stripe_peaks(d, lambda_min=lambda_min, lambda_max=lambda_max,
                                topn=max(int(topn), 1) + len(tuple(exclude)),
                                min_contrast=min_contrast, prepared=prepared,
                                angle_tolerance=angle_tolerance,
                                angle_center=angle_center,
                                contrast_out=measured)
    if not peaks:
        return empty

    kept, contrasts = [], {}
    for lam, ang in peaks:
        clash = any(abs(lam / float(el) - 1.0) < 0.05
                    and abs(((ang - float(ea) + 90.0) % 180.0) - 90.0) < 3.0
                    for el, ea in exclude)
        if clash:
            continue
        kept.append((float(lam), float(ang)))
        if len(kept) >= int(topn):
            break
    if not kept:
        return empty

    for lam, ang in kept:
        contrasts[(lam, ang)] = float(measured.get((lam, ang), np.nan))

    bands = stripe_bands(d, kept, band_width, prepared=prepared)
    lo, hi, strength = (tuple(protect) if protect is not None
                        else PROTECTION_PRESETS.get(str(spec_line).upper(),
                                                    DEFAULT_PROTECTION))
    wprot = protect_weight(d, lo, hi, strength, protect_sigma, protect_smooth)
    flat = wprot * bands
    out = np.where(finite, d - flat, np.nan)
    return LocalSecondPass(flat=flat, local=out, stripe_peaks=tuple(kept),
                           n_detected=len(kept), contrast=contrasts)
