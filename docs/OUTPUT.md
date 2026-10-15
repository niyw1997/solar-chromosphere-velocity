# Output contract (v2)

## Naming

```
<output_dir>/<YYYYMMDD>/<YYYYMMDD>T<HHMMSS>_<LINE>_scv.fits
<output_dir>/index.csv
```

* The key is the **observation time from the header**: `STR_TIME`, then
  `DATE-OBS`, then a timestamp regex on the file name, then the file mtime.
  The card `TIMEKEY` records which source was used.
* `<LINE>` is `HA` / `FEI` / `UNK`, from `SPECLINE` or the file name.
* `--flat` writes everything into `output_dir` without the per-day subdirectory.
* If the same key is produced by a **different source file**, the name gets a
  `_b`, `_c`, … suffix and both sources stay in `index.csv`.  Files are only
  replaced with `--overwrite`.
* `--compress` writes the image extensions as `RICE_1` `CompImageHDU`.

## HDU layout

| HDU | EXTNAME | content | unit |
|---|---|---|---|
| 0 | `PRIMARY` | no data: observation metadata + fitted parameters | — |
| 1 | `VLOS_OBS` | observed line-of-sight velocity | km/s |
| 2 | `VLOS_CORR` | `VLOS_OBS - zero_point` | km/s |
| 3 | `V_DIFFROT` | differential-rotation component | km/s |
| 4 | `V_LIMB` | limb-shift component | km/s |
| 5 | `V_MERID` | meridional-circulation component (projected) | km/s |
| 6 | `V_OFFSET` | systemic translational velocity (global zero point) | km/s |
| 7 | `V_LARGE` | `V_DIFFROT + V_LIMB + V_MERID + V_OFFSET` | km/s |
| 8 | `V_RESID` | `VLOS_CORR - V_LARGE`, quiet-Sun mean removed | km/s |
| 9 | `V_MODEL_ERR` | 1-sigma uncertainty of the model surface | km/s |
| 10 | `MASK_ACTIVE` | 0 = quiet Sun, 1 = active region | — |
| 11 | `FITPARAMS` | 9 parameters + errors + covariance-derived sigma | mixed |

`V_DIFFROT` … `V_MODEL_ERR` are `NaN` outside the fitted region (the tighter of
the disk and edge cuts).  `VLOS_*` are `NaN` outside `disk_exclusion_radius`.

## Per-layer header

```
EXTNAME   layer name
BUNIT     unit (absent for MASK_ACTIVE)
QUANTITY  one-sentence physical definition
FORMULA   the term this layer is responsible for
PARAMS    parameters entering it
MODELVER  'full-2.0' or 'legacy-1.0'
UNITSYS   sign convention
SRCFILE   source file name
OBSTIME   observation time (the output key)
DISKEXCL  disk-exclusion radius used
EDGEEXC   edge-exclusion angle used
MASKBIND  which radial cut actually bound the fit: disk | edge | equal
EFFRAD    effective fit radius in pixels
ZEROPT    zero point subtracted from VLOS_OBS
INVMETH   inversion method
CALSPEC   km/s per spectral pixel actually used
PROCDATE, SOFTWARE, VERSION
COMMENT   caveats specific to the layer
```

`PRIMARY` additionally carries the fitted parameters (`FIT_*`), their 1-sigma
errors (`ERR_*`), `NLIMB`, `RMSFIT`, `RSSFIT`, `NPOINT`, `FITCOND`, `V_OFFSET`,
`LEGACY`, `MODELVER`, `ROTSCALE`, `OMEGA_A` (A as an angular rate, deg/day) and
`QUIETMN` (the mean removed from `V_RESID`).

## `V_OFFSET` is not a gravitational redshift

`V_OFFSET` (the fitted constant `G`) is the **systemic translational velocity**:
the global zero point of the map relative to the inversion template.

It is *exactly* degenerate with three other choices:

1. the template - the inversion measures each profile against the disk-centre
   average profile, so the map's zero point is the disk centre by construction;
2. the zero-point convention (`ZEROPT`);
3. the de-meaning of the residual.

Measured: fitting `y` and `y - 0.632` shifts `G` by exactly `-0.632` while every
other parameter changes by < 2e-10.  Subtracting 632 m/s therefore has **no**
effect on any output.

The solar gravitational redshift (`GM/(Rc)` = 636 m/s) cannot be recovered from
this quantity.  It is also not independently measurable from the absolute
wavelength solution: for `RSM20260424T080829_0016_HA.fits` the observed
disk-centre line core sits ~2-3 km/s from the laboratory Hα air wavelength
(6562.80 A), of which the absolute calibration uncertainty of HIS (~1-2 km/s)
is a comparable contribution.

If an absolute gravitational-redshift diagnostic is ever wanted it must be
reported separately (disk-centre core wavelength vs a laboratory rest
wavelength, with an explicit systematic error), never as `V_OFFSET`.
