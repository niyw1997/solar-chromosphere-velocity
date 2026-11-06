# Changelog

## 2.4.0

### Changed

- **The stripe search now runs over the right scales.**  The fringes that matter
  are full-image-scale, parallel and spatially low-frequency, so the wavelength
  window moved from 25-150 px to **30-1500 px**, and only peaks that share the
  orientation of the strongest one are kept (`stripe_angle_tolerance`, 10
  degrees), because the fringes are parallel.  On 2025-11-11 this picks 326 px at
  -45 degrees for H-alpha and 544 px at -45 degrees for Fe I, the same scale and
  orientation that per-event tuning had identified; the earlier window was
  locking onto genuine mid-scale structure and removing far too much.
- The notch is narrower (`stripe_band_width` 0.45 to **0.20** k0) so only the
  stripe itself is removed, and it is never allowed to fall below one k-grid
  step, which otherwise made a long-wavelength fringe impossible to notch at all.
- A second pass is now **opt-in** (`--stripe-second-pass`) instead of on, because
  it over-subtracts.
- `V_SS` gained a column in the joint grid, between `V_OTHER` and `V_LOCAL`, so a
  row reads `V_LOS`, `V_diffrot`, `V_limb+V_merid+V_offset`, `V_SS`, `V_local`.
  Products carry the scan terms whenever `--joint-grid` is on, so the column is
  always available, and older products without `V_SS` fall back to four columns.
- **The `V_LOCAL` colour scale is fixed per spectral line**, `+/-0.5 km/s` for
  Fe I and `+/-1.5` for H-alpha, instead of a percentile.  The Fe I residual is
  about a third of the H-alpha one, and an automatic limit made the same quantity
  look unrelated from one figure to the next.  The rule lives in
  `constants.LOCAL_VELOCITY_LIMIT_KM_S` and is honoured by the per-file component
  plots, the joint grid and the overlays alike.

### Added

- `blend_stripe_fields()` and `scan_correction_two_arm()`: run the removal twice,
  once hard and once soft, then blend the two flat-field estimates with a
  smoothed weight.  The hard arm removes about twice as much as the soft one but
  costs 5 % (H-alpha) to 11 % (Fe I) of the 20-50 px power; an equal-weight blend
  recovers most of that while still removing 1.4 to 2.1 times more than the soft
  arm.  The `agreement` weighting turned out to collapse back onto the soft arm,
  so the mean weighting is the useful one.  Not wired into the default pipeline.

### Fixed

- `plot_map_grid()` drew one colour bar per column above the first row only, so
  as soon as the rows carried different scales the lower row silently read
  against the wrong one.  It now draws a bar row above every data row
  (`cbar_per_row`), so every row reads against its own scale.

## 2.3.0

### Added

- **Scan-saddle and flat-field removal** (`scan_correction.py`), on by default.
  The level-1 residual still carries two detector-scale systematics, and both
  are now removed before the product is written:

  ```
  V_SS    = V_SCAN_SADDLE + V_FLAT
  V_LOCAL = V_RESID - V_SS
  ```

  `V_SCAN_SADDLE` is a full-disk quadratic (`c0 + c1 Y + c2 Y^2 + c3 X + c4 XY +
  c5 X^2` in normalised disk coordinates), fitted by least squares on every
  finite pixel.  `V_FLAT` is a parallel-stripe template, either a Fourier
  band-stop at a wavelength and orientation or an envelope template
  (demodulate, smooth, re-modulate).  The wave vectors are detected from the
  map by default (local maxima of the 2-D power spectrum with 25-150 px
  wavelength), or supplied with `--stripe-peaks LAM:ANG`; an optional second
  pass re-removes peaks that survived (after/before stripe RMS ratio > 0.4).
  The flat-field term is multiplied by a smooth active-region weight, so real
  blobs are not erased.  Defaults follow the spectral line (H-alpha protects
  strongly, Fe I keeps blobs).

- `V_LOCAL` is now the science layer, and `V_RESID` is demoted to a level-0.5
  intermediate that is **not written** unless `--write-vresid` is given.  The
  other intermediates (`V_SCAN_SADDLE`, `V_FLAT`, `V_SS`) come with
  `--write-scan-terms`.  Figures fall back to `V_RESID` when reading an older
  product, so archived files still plot.

- The removal is auditable from the product: `SCANCORR`, `SCSADDLE`, `SCFLAT`,
  `SCMODE`, `SCSTRIPE`, `SCNSTRI`, `SCQUADF`, `SCWMIN`, `RESSTD`, `SCSTD`,
  `LOCSTD`.  `--no-scan-correction`, `--no-scan-saddle` and
  `--no-stripe-removal` switch it off or partly off.

The implementation was cross-checked against an independent implementation of
the same method on the 2025-11-11 H-alpha product: `V_SCAN_SADDLE`, `V_FLAT`, the
protection weight, `V_SS` and the corrected residual all agree to `max|Δ| <
1e-13`.

## 2.2.0

### Added

- **Joint H-alpha + Fe I component grid** (`plotting_grid.py`): the standard
  product of a run that processes both spectral lines.  One figure per
  observation time, one row per line and one column per physical component
  (`V_LOS`, `V_diffrot`, `V_limb + V_merid + V_offset`, `V_resid`), with a
  horizontal colour bar per column above the first row, panel letters, the
  observing time and the median `V_offset` annotated, and a layout fixed in code
  (equal absolute row/column gaps, panels sized from the data aspect, view
  limits derived from `R_SUN` and `CDELT1`).
- `scv --joint-grid` (with `--joint-grid-dir`) writes one grid per observation
  time at the end of a directory run; `scripts/plot_joint_ha_fe.py` does the
  same over an existing output tree; `plot_joint_grid`, `find_joint_pairs` and
  `make_joint_series` expose it in Python.  Pairing uses the observation stamp,
  so only times that carry both lines get a figure.
- Manual section "The joint H-alpha + Fe I product" (English and Chinese
  editions) and a README section listing the layout rules.

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
- `--legacy` regression mode (it reproduces the **identifiable** v1.0.1
  parameters; `e` and `N` are degenerate and are reported as flagged/NaN
  instead), `process-chase` compatibility entry point,
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

- A 2-D input whose `BUNIT` is not a velocity is now refused instead of being
  fitted as if it were km/s - previously a raw intensity map mis-named
  `*_Vdop.fits` produced a plausible-looking but meaningless product.  `m/s`
  is converted to km/s.
- `open_plane` located its HDU with `hdu.data`, which decompresses a compressed
  cube; it now uses `hdu.shape`.
- `PRIMARY` contained up to 38 duplicated keywords (the input header and our
  own cards overlap, and the card loop had been written twice), including a
  conflicting `INST_ROT`.  Duplicates are now collapsed, keeping ours.
- `e` and `N` were published with `ERROR = 0`, i.e. as perfectly measured, even
  when `e` was pinned to its bound and `N` had no effect on the model.  They are
  now flagged and carry `NaN` errors.
- `NSCAN` truncated the scan mid-number at 50 characters; it is now split
  across `NSCAN`, `NSCAN2`, ... without cutting a value.
- `index.csv` did not record files that failed before the fit stage; it is now
  a complete inventory.
- `UNITSYS` and `INVMETH` are written (the docs had promised them all along),
  and the `FITPARAMS` unit column follows `--rotation-amplitude rad_per_s`
  instead of always claiming km/s.
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
- `matplotlib.cm.get_cmap` is no longer used.  It is present in matplotlib
  3.10.8 and **absent in 3.11.2** (verified with both versions installed
  locally), and v1 calls it from `ChromosphereProcessor.__init__`, so v1
  cannot even be *instantiated* under matplotlib 3.11 regardless of
  whether plotting is enabled - which is what stopped the v1 baseline on
  BlueWhale (matplotlib 3.11.2).  v2 uses the `matplotlib.colormaps`
  registry with a fallback.

### Renamed (2.1)

The configuration and CLI now use solar-physics terms for the quantities that
denote physical things; the generic I/O names are unchanged.

| 2.0 | 2.1 |
|---|---|
| `n_limb` | `limb_exponent` |
| `n_limb_scan` | `limb_exponent_scan` |
| `limb_nonneg` | `limb_amplitude_positive` |
| `window_half` | `correlation_half_width` |
| `profile_mask_radius` | `reference_profile_radius` |
| `demean_window` | `continuum_subtraction` |
| `clip_subpixel` | `limit_subpixel_shift` |
| `line_center_min_len` | `min_spectral_samples` |
| `activity_threshold_sigma` | `active_region_sigma_threshold` |
| `use_mad` | `active_region_robust_sigma` |
| `min_activity_area` | `min_active_region_area` |
| `dilation_iterations` | `active_region_grow_steps` |
| `erosion_iterations` | `active_region_shrink_steps` |
| `disk_exclusion_radius` | `disk_mask_radius` |
| `row_block` | `spatial_strip_rows` |
| `n_workers` | `parallel_processes` |
| `use_multiprocessing` | `parallel_processing` |
| `flat_output` | `no_day_subdirectory` |
| `write_index` | `write_observation_log` |
| `compress_output` | `rice_compression` |
| `exclude_globs` | `exclude_patterns` |

CLI flags follow the same names (e.g. `--n-limb` -> `--limb-exponent`,
`--row-block` -> `--spatial-strip-rows`).  Old v1 configuration keys are still
accepted through `config_from_v1`.

### Notes

- `V_OFFSET` is the **systemic translational velocity** - a global zero point.
  It is exactly degenerate with the template choice, the zero-point convention
  and the residual de-meaning, and it is **not** a gravitational redshift.
  See `docs/OUTPUT.md`.
- `edge_exclusion_angle` is a no-op whenever it exceeds
  `arcsin(disk_exclusion_radius)` (e.g. 80 deg with 0.95).  The header card
  `MASKBIND` records which cut actually bound the fit.
