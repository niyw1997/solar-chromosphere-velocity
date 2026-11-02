# Optional CHASE pointing-jitter preprocessing

## Why

CHASE/HIS Level-1 `_HA.fits` cubes keep the detector geometry: `CRPIX1`/`CRPIX2` move with the
image from frame to frame. Measured on 2025-06-10 (349 frames, 72 s cadence):

| quantity | median | p95 | max |
|---|---|---|---|
| per-frame disk shift | 5.3" | 10.1" | 13.6" |
| step between consecutive frames | 8.3" | 11.2" | 19.4" |

In 72 s the chromosphere itself moves by about 0.5". A fixed pixel window therefore sees a
sawtooth of about 8" between frames, which is an instrumental offset, not a velocity signal.

## How to enable

```bash
pip install -e ".[prep]"     # installs pyprepchase>=0.1
process-chase data out --use-pyprepchase --pyprepchase-cache ~/chase/prep_cache
```

```python
from solar_chromosphere_velocity import ChromosphereConfig
cfg = ChromosphereConfig(input_dir="data", output_dir="out")
cfg.use_pyprepchase = True
cfg.pyprepchase_cache_dir = "~/chase/prep_cache"   # default
cfg.pyprepchase_overwrite = False                  # recompute the cache
```

## What happens

1. `ChromosphereProcessor._maybe_pyprep()` asks `pyprepchase.scv.prepare()` for a corrected cube.
2. `prepare()` reads the headers of the day, builds the per-frame shift to the daily median
   `CRPIX`, resamples the cube with a cubic spline, and writes it to the cache.
3. scv continues with the cache path exactly as if the user had passed it in.

The cache key covers the file size, mtime and the applied shift, so changing the reference grid or
the file does not reuse a stale product. Source files are opened read-only.

## What is not changed

Spectral WCS (`CRVAL3`, `CDELT3`), units, data type and array shape. Only `CRPIX1/CRPIX2` and the
spatial pixels move. The wavelength zero-point jitter (about 5.3 km/s) is not corrected here.

## Failure behaviour

If `pyprepchase` is missing, or the correction raises, scv logs a warning and processes the raw
file. The optional step never hard-fails a production run.
