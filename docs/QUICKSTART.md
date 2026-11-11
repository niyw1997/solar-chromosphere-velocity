# Quick start

## What this code measures

A CHASE/HIS observation is a spectral cube: one spectrum for every position on the
solar disk. A spectral line forms over a range of heights in the solar atmosphere,
and its displacement from the laboratory wavelength records the line-of-sight
velocity of the plasma at that height, at the moment of observation.

This code converts one such cube into a velocity map, and then separates that map
into the contributions whose origin is known, so that the remainder can be
attributed to the solar atmosphere rather than to the Sun's geometry or to the
instrument.

## The decomposition

The measured displacement gives the observed line-of-sight velocity `VLOS_OBS`.
Over the disk it is the sum of four terms imposed by the Sun's geometry and
large-scale kinematics, plus a local remainder,

```
VLOS_OBS = V_DIFFROT + V_LIMB + V_MERID + V_OFFSET + V_RESID
```

`V_DIFFROT` is the east-west antisymmetric rotation profile, `V_LIMB` the
centre-to-limb term, `V_MERID` the north-south meridional circulation projected
along the line of sight, and `V_OFFSET` a global zero point that also absorbs any
systemic velocity of the observer. The four amplitudes are fitted together with
two exponents over the whole disk.

The remainder still carries two patterns of instrumental origin, a raster
scan-saddle and a set of parallel fringes. Both are removed,

```
V_SS    = V_SCAN_SADDLE + V_FLAT
V_LOCAL = V_RESID - V_SS
```

and `V_LOCAL` is what this code reports as the local line-of-sight velocity.

## Locating the line centre

This code determines the line centre from the data instead of assuming it. A
template is built from the average spectrum within 100 pixels of the disk centre,
each spectrum is cross-correlated with that template, and the displacement of the
correlation peak is converted to velocity through the wavelength scale of the cube.
The result is the same physical quantity whether or not the line sits at the
nominal wavelength of the file.

## Install

```
pip install -e .
```

Python 3.9 or newer, with numpy, scipy and astropy. matplotlib is needed only for
figures.

## One command

```
scv --file CUBE.fits --output RESULTS
```

The product is written to `RESULTS/<date>/<stamp>_<line>_scv.fits`.

Both spectral lines of one observing sequence, with the joint figure:

```
scv --input DIR --pattern '*.fits' --output RESULTS --joint-grid -v
```

`-v` additionally writes the per-line figures; without it only the FITS products
appear. `--joint-grid` pairs the two lines by their observation stamp and draws one
figure per time.

## What comes out

Each product carries one extension per physical quantity, in this order.

| extension | quantity | unit |
|---|---|---|
| `VLOS_OBS` | observed line-of-sight velocity | km/s |
| `VLOS_CORR` | the same after the zero point is applied | km/s |
| `V_DIFFROT` | differential rotation | km/s |
| `V_LIMB` | centre-to-limb term | km/s |
| `V_MERID` | meridional circulation | km/s |
| `V_OFFSET` | global zero point | km/s |
| `V_LARGE` | the sum of the four | km/s |
| `V_LOCAL` | local velocity, fringes and scan-saddle removed | km/s |
| `V_MODEL_ERR` | uncertainty of the fitted surface | km/s |
| `MASK_ACTIVE` | active-region marker | none |
| `FITPARAMS` | the fitted parameters with their uncertainties | mixed |

`PRIMARY` records the observing time, the source file, the fitted parameters and
everything used in the fringe removal, so a product can be audited afterwards.

## Tuning the fringe removal

This code detects the fringe wave vectors from the map itself. When the detection
needs correcting, state them explicitly as wavelength in pixels and orientation in
degrees,

```
scv --file CUBE.fits --output RESULTS --stripe-peaks 326:-45
```

To keep the removal switchable for comparison,

```
--no-scan-correction     keep the residual as it is
--no-stripe-removal      remove only the scan-saddle
--write-vresid           also write the residual before removal
```

## Speed

One cube, one process, measured on a full-disk CHASE/HIS product of 2313 x 2304
pixels.

| spectral line | wall time | line-centre inversion | flow fit | reconstruction and fringe removal | writing | peak memory |
|---|---|---|---|---|---|---|
| H-alpha (118 planes) | 12.4 s | 6.8 s | 1.0 s | 4.6 s | 0.5 s | 1.40 GB |
| Fe I (46 planes) | 11.0 s | 5.3 s | 1.0 s | 4.7 s | 0.5 s | 1.32 GB |

A sequence of both lines costs about 25 s per observing time. The work inside one
cube is carried out as array operations over whole row blocks, so a single process
keeps one core busy; to process many cubes at once, run several processes, one per
cube, and each needs its own 1.4 GB.

## Where to read more

`docs/manual.pdf` derives the velocity model, the inversion, the fringe removal and
the uncertainty propagation, and lists every keyword written to `PRIMARY`.
