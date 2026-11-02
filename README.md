# Solar Chromosphere Velocity Processor

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://github.com/yourusername/solar-chromosphere-velocity)

A Python package for processing solar chromospheric velocity data from CHASE mission, featuring:

- **Automatic spectral data preprocessing** for Hα and Fe lines
- **Disk exclusion** (configurable radius threshold, default: 0.95 solar radius)
- **Automatic active region detection**
- **Large-scale flow removal** (differential rotation, meridional circulation, limb shift)
- **Step-by-step visualization** at each processing stage
- **Parallel processing** for batch operations
- **Comprehensive model fitting** with parameter estimation

## Optional: CHASE pointing-jitter preprocessing

CHASE Level-1 cubes are delivered in detector coordinates: the disk moves by up to ~10 arcsec
from frame to frame (median 5.3" on 2025-06-10), and the header records that motion in
`CRPIX1`/`CRPIX2`. A fixed pixel window therefore sees a sawtooth that is not solar motion.
The optional [pyprepchase](https://github.com/niyw1997/pyprepchase) step removes it before the
Doppler inversion:

```bash
pip install -e ".[prep]"
process-chase data out --use-pyprepchase --pyprepchase-cache ~/chase/prep_cache
```

or in Python:

```python
cfg = ChromosphereConfig(input_dir="data", output_dir="out")
cfg.use_pyprepchase = True          # default: False
cfg.pyprepchase_cache_dir = "~/chase/prep_cache"
```

The step only changes the spatial `CRPIX` and the pixel resampling; the spectral axis, units and
data type are preserved. Corrected cubes are cached by file hash, and if pyprepchase is not
installed scv logs a warning and processes the raw file. See [docs/pyprepchase.md](docs/pyprepchase.md).

## New Feature: Disk Exclusion

The package now includes disk exclusion functionality that:
- Sets velocities beyond a configurable radius threshold (default: 0.95) to NaN
- Excludes these regions from active region detection
- Excludes these regions from model fitting
- Ensures cleaner analysis by removing noisy edge regions

## Installation

### From PyPI (Coming Soon)
```bash
pip install solar-chromosphere-velocity