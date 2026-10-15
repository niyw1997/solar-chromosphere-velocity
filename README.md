# Solar Chromosphere Velocity Processor — v2

Processing of CHASE/HIS chromospheric spectra into line-of-sight velocity maps,
with the large-scale flow modelled and removed.

* **v2.0.0** — streaming, vectorised, one multi-extension FITS per observation
  time.  `--legacy` reproduces v1.0.1 results bit-for-bit on the identifiable
  parameters (see `CHANGELOG.md` for the full list of fixes).

## Install

```bash
pip install --no-deps -e .          # numpy / scipy / astropy must be present
pip install -e '.[all]'             # + matplotlib (plots) and pytest
```

On a cluster without internet, install the dependencies with conda first
(`conda create -n scv_env -c conda-forge python=3.11 pip numpy scipy astropy`)
and then use `pip install --no-deps -e .` for the package itself.

## Usage

```bash
# one observation time, complete physics
scv --file RSM20260424T080829_0016_HA.fits --output /data/scv

# a whole directory
scv --input /data/level1 --pattern '*_HA.fits' --output /data/scv

# reproduce v1 exactly (also available as `process-chase`)
scv --input /data/level1 --output /data/scv --legacy
process-chase -i /data/level1 -o /data/scv          # implies --legacy

# angular rotation rates instead of km/s amplitudes
scv --input ... --output ... --rotation-amplitude rad_per_s
```

```python
from solar_chromosphere_velocity import Config, ChromosphereProcessor

cfg = Config(input_dir="/data/level1", output_dir="/data/scv",
             pattern="*_HA.fits", spatial_strip_rows=32)
results = ChromosphereProcessor(cfg).process_directory()
for r in results:
    print(r.srcfile, r.fit.as_dict(), r.fit.rms_km_s)
```

## Output

One file per observation time, keyed on the header observation time:

```
<output_dir>/<YYYYMMDD>/<YYYYMMDD>T<HHMMSS>_<LINE>_scv.fits
<output_dir>/index.csv
```

The FITS file holds the velocity components in separate extensions
(`VLOS_OBS`, `VLOS_CORR`, `V_DIFFROT`, `V_LIMB`, `V_MERID`, `V_OFFSET`,
`V_LARGE`, `V_RESID`, `V_MODEL_ERR`, `MASK_ACTIVE`) plus a `FITPARAMS`
binary table.  See `docs/OUTPUT.md` for the exact contract.

**Outputs are never written next to the inputs.**

## Inputs

* CHASE/HIS level-1 spectral cubes: **RICE_1 compressed** (`CompImageHDU`) or
  plain; the data HDU is located by shape, not by index; `NAXIS3` may be 118
  (Hα), 46 (Fe I) or any other value.
* pre-computed Doppler products (2-D, km/s) are accepted too.

Cubes are read window-by-window with `hdu.section`; the reader never touches
`hdu.data` on a compressed HDU (that would decompress the whole cube: 5.5 s and
1.6 GB for a 118x2313x2304 file).

## The velocity model

```
V = (A + B sin^2 phi + C sin^4 phi + D sin^10 rho) cos phi sin L cos B0  [x R_sun]
  + e (1 - cos rho)^N
  + (MC1 sin phi + MC2 sin 2phi) (cos L sin phi cos B0 - cos phi sin B0)
  + G
```

Linear in every parameter except `N`, so v2 solves the normal equations
directly and scans `N` by profile likelihood; `e` is optionally pinned to
`>= 0`.  `--legacy` drops the R_sun / `cos B0` scaling and the meridional
projection, reproducing v1.

## Performance

Measured on a 118x2313x2304 RICE level-1 cube, single process:

| stage | v1.0.1 | v2.0.0 |
|---|---|---|
| geometry (whole image) | ~33 s | 0.6 s |
| Doppler inversion | 5.5–7.5 min | ~10 s |
| model fit (2.4e6 points) | ~120 s | ~1.3 s |
| peak memory | 3.5–5 GB | ~0.6 GB |
| **end to end** | **~10–12 min** | **~14 s** |

## Tests

```bash
SCV_TEST_DATA=/path/to/level1 pytest tests -q
```
