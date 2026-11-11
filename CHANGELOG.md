# Changelog

## 2.5.1

### Changed

- **Fe I gets its own fringe-removal setting.**  Fe I forms lower in the
  atmosphere, so its active regions cover a smaller area and the protection
  footprint can be shallower.  The line now uses a wider notch (0.24 k0 instead of
  0.20), a third wave vector, a narrower protection footprint (5 and 7 px instead
  of 8 and 12), the protection preset (99, 99.9, 0.35) instead of (98, 99.8, 0.6),
  and a second pass that iterates up to two rounds while the fringe remains.  A
  strong fringe therefore takes more rounds than a weak one, so the removal
  follows the fringe rather than a fixed multiplier.

- The second pass accepts a peak only above 150 times the annulus median.  The
  value was set from cross-line evidence: on 2026-10-30 the two wave vectors that
  appear in both lines stand at 6965 and 2992 in Fe I and at 88 and 303 in
  H-alpha, while the additions that appear only in Fe I sit at 37 to 68, which is
  two to four times the noise and indistinguishable from it.

Measured on four events, against 2.5.0:

| event | V_LOCAL before | after | further reduction | 20-50 px power |
|---|---|---|---|---|
| 2025-11-11 | 0.2410 | 0.2365 | 1.87 % | -1.7 % |
| 2026-10-10 | 0.2412 | 0.2366 | 1.89 % | -1.8 % |
| 2026-10-30 | 0.2002 | 0.1923 | 3.93 % | -3.1 % |
| 2026-11-04 | 0.2198 | 0.2145 | 2.42 % | -1.5 % |

Every wave vector removed in those runs stands at least 231 above the median, and
`PRIMARY` records which ones through `SC2PEAK` and `SC2CONT`.

## 2.5.0

### Added

- **A second fringe pass, on the corrected map.**  One pass leaves a residual
  along the same orientation in some cubes, most visibly in Fe I.  The second
  pass searches `V_LOCAL` itself, with two restrictions that a free search cannot
  offer: the direction is locked to the one the first pass found, and the contrast
  gate is raised to forty times the annulus median, where the first pass accepts
  eight.  The gate is what makes it safe.  On an already corrected map the power is
  close to noise, and a free search fires at nine to sixteen times the median on a
  pure Gaussian field, so nothing under forty is accepted.

  On 2026-11-04 this recovers 408 px at -45 degrees in H-alpha and 295 px at -40
  degrees in Fe I, at contrasts of 148 and 296, both of which are wave vectors an
  analyst had previously chosen by hand.  Across four events the pass lowers the
  local velocity by 0 to 2.2 per cent, and does nothing where one pass already
  sufficed:

  | event | line | V_LOCAL before | after | reduction |
  |---|---|---|---|---|
  | 2025-11-11 | H-alpha | 0.7754 | 0.7754 | 0.00 % |
  | 2025-11-11 | Fe I | 0.2415 | 0.2410 | 0.24 % |
  | 2026-10-10 | H-alpha | 0.7941 | 0.7933 | 0.10 % |
  | 2026-10-10 | Fe I | 0.2419 | 0.2412 | 0.27 % |
  | 2026-10-31 | H-alpha | 0.6006 | 0.5992 | 0.23 % |
  | 2026-10-31 | Fe I | 0.2048 | 0.2002 | 2.24 % |
  | 2026-11-04 | H-alpha | 0.7042 | 0.7026 | 0.22 % |
  | 2026-11-04 | Fe I | 0.2209 | 0.2198 | 0.48 % |

- The pass is on by default.  `--no-local-second-pass` turns it off and
  `--local-second-pass-contrast` moves the gate.  `PRIMARY` records `SC2N`,
  `SC2PEAK`, `SC2CONT` and `SC2STD`, so the second pass is as auditable as the
  first.

## 2.4.0

### Changed

- **The fringe search now covers the full range of scales the fringes occupy.**
  They are periodic, parallel and spatially low-frequency, and they extend to the
  scale of the whole image.  The wavelength window therefore spans **30-1500 px**,
  and the peaks kept are those that share the orientation of the strongest one
  (`stripe_angle_tolerance`, 10 degrees), the fringes being parallel.  On
  2025-11-11 this identifies 326 px at -45 degrees for H-alpha and 544 px at -45
  degrees for Fe I.  A wide window with a coherence constraint describes the
  fringe pattern itself, rather than whichever periodicity happens to dominate
  the mid-range.
- The notch is narrow (`stripe_band_width` **0.20** k0) and is never allowed to
  fall below one k-grid step, so a fringe of any wavelength can be described and
  removed at its own period without reaching into neighbouring scales.
- A second pass, available through `--stripe-second-pass`, describes what a first
  pass leaves.  The default stops at one pass, which is the calibrated path; a
  second pass is offered for data where the fringes are stronger than the range
  the defaults were set on.
- `V_SS` gained a column in the joint grid, between `V_OTHER` and `V_LOCAL`, so a
  row reads `V_LOS`, `V_diffrot`, `V_limb+V_merid+V_offset`, `V_SS`, `V_local`.
  Products carry the scan terms whenever `--joint-grid` is on, so the column is
  always available, and older products without `V_SS` fall back to four columns.

### Added

- `blend_stripe_fields()` and `scan_correction_two_arm()`: run the removal twice,
  once hard and once soft, then blend the two flat-field estimates with a
  smoothed weight.  The hard arm removes about twice as much as the soft one but
  costs 5 % (H-alpha) to 11 % (Fe I) of the 20-50 px power; an equal-weight blend
  recovers most of that while still removing 1.4 to 2.1 times more than the soft
  arm.  The `agreement` weighting, which trusts the hard arm only where the two
  coincide, reduces in practice to the soft arm; the mean weighting is the one
  that spans the two.  Not wired into the default pipeline.

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
  (`V_LOS`, `V_diffrot`, `V_limb + V_merid + V_offset`, `V_resid`).  The view
  limits are derived from `R_SUN` and `CDELT1`, so the framing follows the
  instrument rather than one data set.
- `scv --joint-grid` (with `--joint-grid-dir`) writes one grid per observation
  time at the end of a directory run; `scripts/plot_joint_ha_fe.py` does the
  same over an existing output tree; `plot_joint_grid`, `find_joint_pairs` and
  `make_joint_series` expose it in Python.  Pairing uses the observation stamp,
  so only times that carry both lines get a figure.
- Manual section "The joint H-alpha + Fe I product" (English and Chinese
  editions) and a README section describing the product.

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
  the cluster job (matplotlib 3.11.2).  v2 uses the `matplotlib.colormaps`
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
