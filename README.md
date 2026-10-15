# Solar Chromosphere Velocity Processor

Processing of CHASE/HIS chromospheric spectra into line-of-sight (LOS) velocity
maps, with the large-scale flow modelled, decomposed into its physical
components, and removed.

**Version 2.1** — one multi-extension FITS per observation time, streaming reads,
vectorised geometry, batched-FFT Doppler inversion and a streaming linear
least-squares velocity model. `--legacy` reproduces v1.0.1 on the identifiable
parameters.

* Full method write-up (program flow, parallel framework, spectral fitting,
  velocity model, physical meaning and expression of every term):
  **[`docs/theory.pdf`](docs/theory.pdf)** (source: [`docs/theory.tex`](docs/theory.tex),
  in Chinese; rebuild with `latexmk -xelatex docs/theory.tex`).
* Output contract: [`docs/OUTPUT.md`](docs/OUTPUT.md)
* Change history: [`CHANGELOG.md`](CHANGELOG.md)

---

## Contents

- [What is new](#what-is-new)
- [Performance](#performance)
- [Install](#install)
- [Quick start — single machine](#quick-start--single-machine)
- [Running on a cluster (BlueWhale / LSF)](#running-on-a-cluster-bluewhale--lsf)
- [Python API](#python-api)
- [Method](#method)
  - [Program flow](#program-flow)
  - [Streaming and parallel framework](#streaming-and-parallel-framework)
  - [Spectral line-centre fitting](#spectral-line-centre-fitting)
  - [Doppler inversion](#doppler-inversion)
  - [Velocity model and the physical meaning of each term](#velocity-model-and-the-physical-meaning-of-each-term)
  - [Parameter estimation and uncertainty](#parameter-estimation-and-uncertainty)
- [Output](#output)
- [Inputs](#inputs)
- [Tests](#tests)
- [License](#license)

---

## What is new

### 2.1 — solar-physics naming

The configuration and CLI now name physical quantities in the terms a solar
physicist expects, instead of generic computing terms. The generic I/O names
(`input_dir`, `output_dir`, `pattern`, `overwrite`) are unchanged.

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

CLI flags follow the same names (`--n-limb` → `--limb-exponent`,
`--row-block` → `--spatial-strip-rows`, …). Old **v1** configuration keys are
still accepted through `config_from_v1` / the `process-chase` entry point.

### 2.0 — the rewrite

* **Complete physics.** The rotation term carries the `cos B0` factor, the
  meridional circulation is projected onto the LOS, and the velocity scale is
  taken from the header (`CDELT3`, `WAVE_LEN`) instead of a hard-coded constant.
* **Streaming I/O.** Compressed cubes are read window-by-window with
  `hdu.section`; the whole cube is never decompressed.
* **Speed.** Vectorised geometry, batched-FFT cross-correlation, streaming
  normal equations (see below).
* **A real output contract.** One multi-extension FITS per observation time,
  keyed on the header observation time, plus `index.csv`. Outputs are **never**
  written next to the inputs.
* **Honest uncertainties.** Parameters that are not identifiable (`e` pinned to
  its `>= 0` bound, and `N`, which the model cannot see once `e = 0`) are flagged
  and given `NaN` errors instead of `±0`, which would read as a perfect
  measurement.

## Performance

Measured on BlueWhale (64-core `mpi` queue, Python 3.11 / astropy 8.0.1 /
numpy 2.4.6) on a 2026-04-24 Hα level-1 cube, `2313 x 2304 x 118`, one process:

| implementation | wall time | peak memory |
|---|---|---|
| v1.0.1 | 169.2 s | 4.266 GB |
| v2 `--legacy` | **8.9 s** | 0.575 GB |
| v2 complete physics | **9.6 s** | 0.583 GB |

**Speed-up 17.6×** (complete physics) / **19.0×** (`--legacy`);
**7.3×** less memory. The two versions agree numerically: `VLOS_OBS` differs by
`max|Δ| = 9.5e-7 km/s` (rms `7.5e-8`) and the identifiable parameters by
`max|Δ| = 4.4e-7`.

Where the time went:

| stage | v1.0.1 | v2 |
|---|---|---|
| vectorised solar geometry | ~33 s | ~0.6 s |
| Doppler inversion | 5.5–7.5 min | ~6 s |
| model fit (2.4e6 points) | ~120 s | ~1.3 s |

Note that `--visualize` is *not* free: the nine figures roughly double the wall
time and raise the peak from 0.58 GB to about 1.5 GB (3 figures) or 3.0 GB
(9 figures). The FITS product is unaffected if plotting fails.

## Install

```bash
pip install -e .                    # numpy / scipy / astropy must be present
pip install -e '.[all]'             # + matplotlib (figures) and pytest
```

On a cluster without internet, install the compiled dependencies with conda
first and then the package without dependencies:

```bash
conda create -n scv_env -c conda-forge python=3.11 pip numpy scipy astropy \
      matplotlib sunpy
conda activate scv_env
pip install --no-deps -e .
```

## Quick start — single machine

```bash
# one observation time, complete physics
scv --file RSM20260424T080829_0016_HA.fits --output /data/scv

# a whole directory of level-1 cubes
scv --input /data/level1 --pattern '*_HA.fits' --output /data/scv

# reproduce v1.0.1 exactly (also available as the `process-chase` entry point)
scv --input /data/level1 --output /data/scv --legacy
process-chase -i /data/level1 -o /data/scv            # implies --legacy

# angular rotation rates (rad/s) instead of km/s amplitudes
scv --input /data/level1 --output /data/scv --rotation-amplitude rad_per_s

# fix the limb exponent instead of scanning the grid
scv --input /data/level1 --output /data/scv --limb-exponent 2.0

# with figures: 3 overview panels plus one file per component
scv --input /data/level1 --output /data/scv --visualize --plot-per-component
```

A ready-to-edit wrapper is provided:

```bash
bash examples/run_local.sh /data/level1 /data/scv '*_HA.fits'
```

## Running on a cluster (BlueWhale / LSF)

Two example scripts are provided: `examples/run_bluewhale.sh` is the job body
(runs on the compute node) and `examples/submit_bluewhale.sh` submits it to the
`mpi` queue with the cluster conventions — `-n` a multiple of 64 and
`-R 'span[hosts=1]'` so the job stays on one host.

```bash
# review the exact bsub line, then answer y
bash examples/submit_bluewhale.sh scv-run 01:00
```

The equivalent by hand, from the directory that holds `src/` and `data/`:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
mkdir -p work logs

bsub -q mpi -n 64 -W 01:00 -R 'span[hosts=1]' \
     -J scv-run \
     -o logs/scv-run-%J.out -e logs/scv-run-%J.err \
     bash -lc 'cd '"$PWD"' && python -m solar_chromosphere_velocity.cli \
               --input data --pattern "*_HA.fits" --output work \
               --spatial-strip-rows 32 --visualize --log-level INFO'

bjobs                 # watch it
tail -f work/run.log  # the job's own log
```

The analysis is single-threaded by design (see
[Streaming and parallel framework](#streaming-and-parallel-framework)), so use
the cores for **parallel files** rather than inside one file: submit several
jobs, or have the job body loop over disjoint file lists.

## Python API

```python
from solar_chromosphere_velocity import Config, ChromosphereProcessor

cfg = Config(input_dir="/data/level1", output_dir="/data/scv",
             pattern="*_HA.fits", spatial_strip_rows=32)
for r in ChromosphereProcessor(cfg).process_directory():
    print(r.srcfile, r.status, round(r.fit.rms_km_s, 4), r.fit.flags)
    # r.layers holds VLOS_OBS, V_DIFFROT, V_LIMB, V_MERID, V_OFFSET,
    # V_LARGE, V_RESID, V_MODEL_ERR, MASK_ACTIVE as numpy arrays
```

Plotting is a thin wrapper around matplotlib and is also importable:

```python
from solar_chromosphere_velocity import plot_from_file, plot_layers

plot_from_file("20260424T080829_HA_scv.fits", "components.png",
               panels=["V_DIFFROT", "V_LIMB", "V_MERID",
                       "V_OFFSET", "V_LARGE", "V_RESID"])
```

See [`examples/api_example.py`](examples/api_example.py) for a complete script,
and [`scripts/visualize_layers.py`](scripts/visualize_layers.py) for the
stand-alone figure tool (panel selection, per-component files, shared scales).

## Method

This section summarises [`docs/theory.pdf`](docs/theory.pdf), which gives the
full derivation and the correspondence with the source.

### Program flow

```
level-1 cube ─► read header, build SolarParams
             ─► line-centre fit ─► cross-correlation Doppler inversion
             ─► disk + active-region masks ─► fit sample
             ─► streaming normal equations ─► model parameters
             ─► block-wise reconstruction of every component
             ─► multi-extension FITS + index.csv
             ─► optional multi-panel figures
```

Each stage is timed and reported in `ProcessResult.timings`
(`read+invert`, `invert`, `masks`, `fit`, `reconstruct`, `write`, `plots`,
`total`), which is where the performance table above comes from.

### Streaming and parallel framework

Three layers, in order of what actually buys the speed:

1. **Vectorised solar geometry.** Every pixel of a strip gets its
   `(B, L, rho)` at once, replacing v1's per-pixel Python loop (~59×).
2. **Batched-FFT Doppler inversion.** The cross-correlation of *all* pixels in a
   strip against the reference profile is one FFT call, with the signal axis of
   shape `(window, rows x columns)` (~38×; 24.4 µs → 0.65 µs per pixel on
   synthetic data).
3. **Streaming normal equations.** The fit accumulates an `8 x 8` matrix strip
   by strip, so its memory does not grow with the number of pixels; the
   reconstruction is block-wise too, and only the ten output layers stay
   resident.

The cube is read in strips of `spatial_strip_rows` (default 32) rows using
`hdu.section`, which decodes only the requested tiles. Accessing `hdu.data`
instead would decompress the whole cube (~0.9–1.6 GB, 5–8 s for a 118-layer
file). 32 rows measured fastest *and* leanest against 64/128/256/512.

`parallel_processing` and `parallel_processes` are **reserved**: one file is
already fast, and parallelising across *files* with the scheduler is more
efficient. The in-process path stays single-threaded for reproducibility.

Complexity, with `P` pixels, window length `L = 2w+1` and strip rows `s`:
inversion `O(P L log L)`, fit `O(P · 8²)`, peak memory
`O(s · n_x · L + 10 · n_y · n_x)`.

### Spectral line-centre fitting

The reference profile is the mean profile inside
`reference_profile_radius` pixels of disk centre. The line centre is located by
a Gaussian fit around the local minimum; the wavelength solution

$$\lambda(i) = \mathrm{CRVAL3} + \bigl(i - (\mathrm{CRPIX3}-1)\bigr)\,\mathrm{CDELT3}$$

is used only as a *hint*, because the header solution and the data can disagree
by ~1.4 pixels (~3 km/s) in level-1 files.

### Doppler inversion

With the window centred on the line core, half-width
`w = correlation_half_width`, the reference is `T(j) = Ibar(i0 - w + j)` and each
pixel's window `S_p(j)` is mean-subtracted (`continuum_subtraction`) to remove
the continuum slope. The circular cross-correlation

$$C_p(k) = \sum_{j=0}^{L-1} \tilde S_p(j)\, \tilde T(j-k)$$

is evaluated for the whole strip by FFT. The peak `k*` is refined to sub-pixel
accuracy by the three-point parabola

$$x = \frac{C_p(k^*-1) - C_p(k^*+1)}{2\bigl[C_p(k^*-1) - 2C_p(k^*) + C_p(k^*+1)\bigr]},
\qquad \delta_p = (k^* + x) - \lfloor L/2 \rfloor,$$

with `x` clipped to ±0.5 pixels (`limit_subpixel_shift`). The velocity follows
from the dispersion,

$$\kappa = \frac{c\,\mathrm{CDELT3}}{\lambda_0},
\qquad V_{\mathrm{LOS}}(p) = \delta_p\,\kappa .$$

For Hα (`CDELT3 = 0.04844 Å`, `λ0 = 6562.82 Å`) this gives
`κ ≈ 2.2139 km/s/pixel`. `--legacy` uses v1's hard-coded `1.1068 × 2`
instead. The sign convention is **redshift (receding) positive**, recorded in the
`UNITSYS` card.

### Velocity model and the physical meaning of each term

With `B` the heliographic latitude, `L` the longitude, `rho` the heliocentric
angle and `B0` the disk-centre latitude, define the LOS projections

$$\mathcal{P} = \cos B \sin L \cos B_0, \qquad
\mathcal{M} = \cos L \sin B \cos B_0 - \cos B \sin B_0,$$

then the model LOS velocity is

$$
\begin{aligned}
V_{\mathrm{model}} =\;
& s\bigl(A + B\sin^2 B + C\sin^4 B + D\sin^{10}\rho\bigr)\,\mathcal{P}
   && \text{differential rotation}\\
+\;& e\,(1-\cos\rho)^{N} && \text{limb shift}\\
+\;& \bigl(\mathrm{MC1}\sin B + \mathrm{MC2}\sin 2B\bigr)\,\mathcal{M}
   && \text{meridional circulation}\\
+\;& G && \text{systemic translation}
\end{aligned}
$$

where `s = R_sun` when the rotation parameters are angular rates
(`--rotation-amplitude rad_per_s`), otherwise `s = 1`.

| parameter | physical meaning | expression |
|---|---|---|
| `A` | equatorial rotation amplitude | `s A P` |
| `B` | rotation, `sin²B` term (differential rotation) | `s B sin²B P` |
| `C` | rotation, `sin⁴B` term | `s C sin⁴B P` |
| `D` | rotation, `sin¹⁰rho` limb term | `s D sin¹⁰rho P` |
| `e` | limb-shift amplitude (constrained `>= 0`) | `e (1-cos rho)^N` |
| `N` | limb-shift power-law exponent (grid scan) | `(1-cos rho)^N` |
| `MC1` | meridional circulation, `sin B` component | `MC1 sin B M` |
| `MC2` | meridional circulation, `sin 2B` component | `MC2 sin 2B M` |
| `G` | systemic translational velocity | `G` |

**On `G`.** The inversion template is the disk-centre average profile, so any
absolute (gravitational + convective) line shift is already inside the template.
`G` therefore mixes a true systemic velocity, the template-versus-rest-wavelength
offset and the instrument's absolute wavelength zero point. It is **not** a
gravitational-redshift measurement — for reference, the nominal
`GM/(R_sun c) ≈ 0.636 km/s`, an order of magnitude above the measured `|G|`.

`--legacy` drops the `cos B0` factor, the meridional projection and the `R_sun`
scaling, and takes the velocity scale and line centre the v1 way, so it
reproduces v1.0.1 on the identifiable parameters.

### Parameter estimation and uncertainty

Every parameter except `N` enters linearly, so with the design matrix `X` the
solution satisfies `X'X b = X'y`. The fit accumulates `X'X`, `X'y` and `y'y`
strip by strip and solves a `9 x 9` system; `N` is scanned over
`limb_exponent_scan` re-using the same accumulators, and the minimum-residual
value wins. If `e` comes out negative it is pinned to the boundary by dropping
its column (an active-set step), which necessarily raises the residual sum of
squares — that is the cost of the constraint.

The covariance is `Sigma = s² (X'X)^-1` with `s² = RSS/(n-p)`, and the per-pixel
model uncertainty written to `V_MODEL_ERR` is

$$\sigma_{\mathrm{model}}(x) = \sqrt{x^\top \Sigma x}.$$

A parameter that is not identifiable gets `ERROR = NaN` in the `FITPARAMS`
table, a `FLAG`, and **no** `ERR_*` card — `PINNED_AT_ZERO`,
`DEGENERATE_WITH_E`, `FLAT_SCAN`, `SCAN_GRID`. The covariance itself is left
intact, so `V_MODEL_ERR` stays finite.

## Output

One file per observation time, keyed on the **header** observation time:

```
<output_dir>/<YYYYMMDD>/<YYYYMMDD>T<HHMMSS>_<LINE>_scv.fits
<output_dir>/index.csv
```

The FITS file holds, in order: `PRIMARY`, then one extension per velocity layer
(`VLOS_OBS`, `VLOS_CORR`, `V_DIFFROT`, `V_LIMB`, `V_MERID`, `V_OFFSET`,
`V_LARGE`, `V_RESID`, `V_MODEL_ERR`, `MASK_ACTIVE`), then the `FITPARAMS`
binary table. See [`docs/OUTPUT.md`](docs/OUTPUT.md) for the exact card-by-card
contract.

**Outputs are never written next to the inputs.**

## Inputs

* CHASE/HIS level-1 spectral cubes — **RICE_1 compressed** (`CompImageHDU`) or
  plain. The data HDU is found by shape, not by index, and `NAXIS3` may be 118
  (Hα), 46 (Fe I) or anything else.
* Pre-computed 2-D Doppler products in km/s. `BUNIT` is honoured (`m/s` is
  converted); a 2-D map whose `BUNIT` is not a velocity is refused rather than
  silently fitted (`require_velocity_unit`).

## Tests

```bash
SCV_TEST_DATA=/path/to/level1 \
SCV_TEST_UNCOMPRESSED=/path/to/plain_cube.fits \
SCV_TEST_FE=/path/to/fe_cube.fits \
pytest tests -q
```

## License

See [`LICENSE.md`](LICENSE.md).
