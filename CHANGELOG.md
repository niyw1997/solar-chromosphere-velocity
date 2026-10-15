# Changelog

## 2.0.0

Complete rewrite of the processing core.  `--legacy` reproduces v1.0.1 results.

### Added

- **One multi-extension FITS per observation time**: `PRIMARY` +
  `VLOS_OBS`, `VLOS_CORR`, `V_DIFFROT`, `V_LIMB`, `V_MERID`, `V_OFFSET`,
  `V_LARGE`, `V_RESID`, `V_MODEL_ERR`, `MASK_ACTIVE` + a `FITPARAMS` BINTABLE.
  Every layer header carries `BUNIT`, `QUANTITY`, `FORMULA`, `PARAMS`,
  `MODELVER`, `UNITSYS`, `SRCFILE`, `OBSTIME`, `INVMETH`, `CALSPEC` and a
  `COMMENT` describing it.
- **Observation-time output keys**: `<output_dir>/<YYYYMMDD>/<YYYYMMDD>T<HHMMSS>_<LINE>_scv.fits`,
  plus `index.csv`.  Same key from a different source file is disambiguated
  rather than overwritten.
- **Streaming RICE-aware reader**: partial reads via `hdu.section`, HDU located
  by shape, no assumption that the cube lives in `hdu[1]`, no hard-coded 118.
- **Vectorised geometry** (`geometry.py`), **batched FFT inversion**
  (`inversion.py`), **linear least-squares model fit with a profile-likelihood
  scan over N** (`models.py`).
- **Complete physics**: rotation term scaled by R_sun and projected with
  `cos B0`; meridional circulation projected on the line of sight.
  `--rotation-amplitude rad_per_s` reports A,B,C,D as angular rates.
- `--legacy` regression mode, `process-chase` compatibility entry point,
  `V_MODEL_ERR` (per-pixel model uncertainty), per-layer masks, `index.csv`.

### Changed

- The velocity scale is computed from `CDELT3` / `WAVE_LEN`
  (`c * dlambda / lambda`) instead of the hard-coded `1.1068 * 2` km/s per
  pixel.  The old constant is still used under `--legacy`.
- The line centre is located from the data with the WCS as a hint; the
  `len(profile) < 90` fallback to the profile midpoint is gone.  This fixes a
  ~27 km/s bias on CHASE Fe I cubes (46 samples).
- Correlation windows are mean-subtracted, and the parabolic sub-pixel step is
  guarded and clipped to +/-0.5 px.
- Active-region detection uses median + 1.4826*MAD with sigma clipping instead
  of a `3*sigma` threshold computed on a distribution that already contains the
  active regions.
- `V_RESID` is de-meaned over the quiet-Sun fit sample rather than the whole
  disk.
- Packaging is now a single `pyproject.toml`; `setup.py`, `setup.cfg` and the
  broken `branch` / `pyproject.toml.backup` files are gone.
- Dependencies reduced to numpy/scipy/astropy (matplotlib is an extra).
  `sunpy` was imported by v1 but never used; `fitsio` and `scikit-image` were
  never imported; `numba` and `tqdm` are no longer needed.

### Fixed

- **Outputs are never written next to the inputs.**  v1 derived the
  intermediate Doppler product from `input_path.parent` with `overwrite=True`,
  which silently destroyed the official `*_HA_Vdop.fits` level-1 products.
- `overwrite_existing` actually works (it was a dead option in v1), as do
  `--no-validation`-style switches, which are now reported as ignored instead
  of being silently accepted.
- The correlation is computed with a batched FFT rather than a Python loop over
  every pixel: measured ~38x on the inversion, ~59x on the geometry, and the
  curve_fit-based model fit is replaced by an exact linear solve
  (~0.02 s vs ~120 s on 2.4e6 points, with a lower residual sum of squares).
- Peak memory on a 2313x2304 RICE cube: ~0.6 GB vs 3.5-5 GB in v1.
- `create_solar_mask` no longer prints debug output; the duplicate
  `from turtle import shape` imports and the repeated logging configuration are
  gone.
- `matplotlib.cm.get_cmap` (removed in matplotlib 3.9) is no longer used.

### Notes

- `V_OFFSET` is the **systemic translational velocity** - a global zero point.
  It is exactly degenerate with the template choice, the zero-point convention
  and the residual de-meaning, and it is **not** a gravitational redshift.
  See `docs/OUTPUT.md`.
- `edge_exclusion_angle` is a no-op whenever it exceeds
  `arcsin(disk_exclusion_radius)` (e.g. 80 deg with 0.95).  The header card
  `MASKBIND` records which cut actually bound the fit.
