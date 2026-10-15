"""FITS reading for v2: RICE-compressed *and* plain cubes, read streamingly.

Why this module exists
----------------------
CHASE/HIS level-1 spectral files store the cube as a RICE_1 tiled-compression
image (``ZIMAGE=T, ZCMPTYPE='RICE_1', ZTILE1=2304, ZTILE2=1, ZTILE3=1``), which
astropy exposes as a ``CompImageHDU``.  Touching ``hdu.data`` on such an HDU
decompresses **the whole cube** (measured: 5.53 s, 1.60 GB RSS), which is what
v1 did before ``.astype(float32)`` doubled it to 2.5 GB.

``hdu.section[...]`` decodes only the requested tiles (measured: 0.2 s, a few MB
for a small window; the full array costs ~0.9-1.6 GB and 5-8 s depending on how
it is measured)
and was verified to return bit-identical values to a full-array slice for
RICE 118-layer, RICE 46-layer *and* uncompressed 21-layer files.

Rules honoured here:
  * partial reads always go through ``.section`` when the HDU is compressed;
  * the data HDU is located by shape, never assumed to be ``hdu[1]``;
  * ``NAXIS3`` may be 118 / 46 / 21 - nothing is hard-coded to 118.
"""

import logging
import os
import re
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning

logger = logging.getLogger(__name__)

__all__ = [
    "DataType",
    "detect_data_type",
    "SolarParams",
    "extract_solar_params",
    "SpectralCube",
    "open_plane",
    "is_compressed",
]

_OPEN_KW = dict(fix=True, ignore_missing_end=True, verify="silentfix",
                output_verify="ignore")


class DataType(Enum):
    UNKNOWN = "unknown"
    SPECTRAL_HA = "spectral_ha"
    SPECTRAL_FE = "spectral_fe"
    DOPPLER_VDOP = "doppler_vdop"


def detect_data_type(filepath: str) -> DataType:
    """Filename-first data-type detection (same policy as v1)."""
    name = os.path.basename(filepath).lower()
    if "_ha.fits" in name:
        return DataType.SPECTRAL_HA
    if "_fe.fits" in name:
        return DataType.SPECTRAL_FE
    if "_vdop.fits" in name or "doppler" in name:
        return DataType.DOPPLER_VDOP
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", VerifyWarning)
            with fits.open(filepath, **_OPEN_KW) as hdul:
                # never touch .data here: on a CompImageHDU that would
                # decompress the whole cube just to look at its shape
                for hdu in hdul:
                    shp = getattr(hdu, "shape", None)
                    if shp is None or len(shp) != 3:
                        continue
                    ctype3 = str(hdu.header.get("CTYPE3", "")).lower()
                    if "ha" in ctype3:
                        return DataType.SPECTRAL_HA
                    if "fe" in ctype3:
                        return DataType.SPECTRAL_FE
                    spec = str(hdu.header.get("SPECLINE", "")).upper()
                    if spec.startswith("HA"):
                        return DataType.SPECTRAL_HA
                    if spec.startswith("FE"):
                        return DataType.SPECTRAL_FE
                for hdu in hdul:
                    shp = getattr(hdu, "shape", None)
                    if shp is None or len(shp) != 2:
                        continue
                    if "m/s" in str(hdu.header.get("BUNIT", "")).lower():
                        return DataType.DOPPLER_VDOP
    except Exception:
        pass
    return DataType.UNKNOWN


def is_compressed(hdu) -> bool:
    return type(hdu).__name__ in ("CompImageHDU", "CompImageHeader")


def _safe_float(header, key, default=None):
    try:
        if key not in header:
            return default
        v = header[key]
        if isinstance(v, str):
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", v)
            return float(m.group()) if m else default
        return float(v)
    except Exception:
        return default


def _safe_str(header, key, default=""):
    try:
        if key not in header:
            return default
        v = str(header[key])
        return "".join(c for c in v if 32 <= ord(c) <= 126).strip()
    except Exception:
        return default


@dataclass
class SolarParams:
    """Solar / instrument geometry extracted from a FITS header."""

    cx: float = 0.0
    cy: float = 0.0
    radius: float = 0.0
    b0: float = 0.0
    p0: float = 0.0
    rsun_obs: float = 0.0
    cdelt1: float = 0.0
    cdelt2: float = 0.0
    crval1: float = 0.0
    crval2: float = 0.0
    cdelt3: Optional[float] = None
    crval3: Optional[float] = None
    crpix3: float = 1.0
    wave_len: Optional[float] = None
    spec_line: str = ""
    date_obs: str = ""
    str_time: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    def require_geometry(self):
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError(
                "solar radius in pixels is missing/invalid (looked for "
                "R_SUN, RSUN, RADII in the header); refusing to continue "
                "because v1 silently produced an all-NaN map in that case")
        return self

    def as_dict(self):
        d = dict(cx=self.cx, cy=self.cy, radius=self.radius, b0=self.b0, p0=self.p0)
        return d


def extract_solar_params(header) -> SolarParams:
    p = SolarParams()
    p.cx = _safe_float(header, "CRPIX1", 0.0)
    p.cy = _safe_float(header, "CRPIX2", 0.0)
    if "CX" in header:
        p.cx = _safe_float(header, "CX", p.cx)
    if "CY" in header:
        p.cy = _safe_float(header, "CY", p.cy)

    for key in ("R_SUN", "RSUN", "RADII"):
        v = _safe_float(header, key, None)
        if v:
            p.radius = float(v)
            break

    p.b0 = _safe_float(header, "B0", 0.0)
    p.p0 = _safe_float(header, "INST_ROT", 0.0)
    p.rsun_obs = _safe_float(header, "RSUN_OBS", 0.0)
    p.cdelt1 = _safe_float(header, "CDELT1", 0.0)
    p.cdelt2 = _safe_float(header, "CDELT2", 0.0)
    p.crval1 = _safe_float(header, "CRVAL1", 0.0)
    p.crval2 = _safe_float(header, "CRVAL2", 0.0)
    p.cdelt3 = _safe_float(header, "CDELT3", None)
    p.crval3 = _safe_float(header, "CRVAL3", None)
    p.crpix3 = _safe_float(header, "CRPIX3", 1.0)
    p.wave_len = _safe_float(header, "WAVE_LEN", None)
    p.spec_line = _safe_str(header, "SPECLINE", "")
    p.date_obs = _safe_str(header, "DATE-OBS", "") or _safe_str(header, "DATE_OBS", "")
    p.str_time = _safe_str(header, "STR_TIME", "")

    for key in ("STEPTIME", "END_TIME", "EXP_TIME", "EXPTIME",
                "SAT_POS1", "SAT_POS2", "SAT_POS3",
                "SAT_VEL1", "SAT_VEL2", "SAT_VEL3", "RSUN_REF",
                "TELESCOP", "INSTRUME", "BIN", "ORID"):
        if key in header:
            p.raw[key] = header[key]
    return p


class SpectralCube:
    """Streaming, context-managed access to a 3-D spectral cube.

    Typical use::

        with SpectralCube(path) as cube:
            patch = cube.read(z0, z1, y0, y1)       # -> (z1-z0, y1-y0, nx)
            prof  = cube.read_profile_patch(params)  # disk-centre box
    """

    def __init__(self, path: str):
        self.path = path
        self._hdul = None
        self.hdu_index = None
        self.header = None
        self.shape: Tuple[int, int, int] = (0, 0, 0)
        self.compressed = False

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", VerifyWarning)
            self._hdul = fits.open(path, **_OPEN_KW)
        self._locate_cube()

    # -- locating ---------------------------------------------------------
    def _locate_cube(self):
        """Find the 3-D HDU **without touching ``.data``**.

        ``hdu.data`` on a ``CompImageHDU`` decompresses the entire cube
        (measured: 5.5 s, 1.6 GB for the CHASE level-1 files).  ``.shape`` and
        ``.header`` come from the Z* keywords and are free.
        """
        candidates = []
        for i, hdu in enumerate(self._hdul):
            shp = getattr(hdu, "shape", None)
            if shp is None:
                continue
            try:
                if len(shp) == 3 and int(np.prod(shp)) > 0:
                    candidates.append(i)
            except TypeError:
                continue
        if not candidates:
            dims = [(i, getattr(h, "shape", None)) for i, h in enumerate(self._hdul)]
            raise ValueError("no 3-D spectral cube found in %s (HDUs: %s)"
                             % (os.path.basename(self.path), dims))
        # prefer hdu[1] when it is 3-D (CHASE convention), else the first 3-D HDU
        idx = 1 if 1 in candidates else candidates[0]
        hdu = self._hdul[idx]
        self.hdu_index = idx
        self.shape = tuple(int(x) for x in hdu.shape)
        self.compressed = is_compressed(hdu)
        # the science header is the cube's own header
        self.header = hdu.header.copy()

    # -- properties -------------------------------------------------------
    @property
    def n_lambda(self) -> int:
        return self.shape[0]

    @property
    def ny(self) -> int:
        return self.shape[1]

    @property
    def nx(self) -> int:
        return self.shape[2]

    @property
    def params(self) -> SolarParams:
        return extract_solar_params(self.header)

    # -- reading ----------------------------------------------------------
    def read(self, z0: int, z1: int, y0: int = 0, y1: Optional[int] = None,
             x0: int = 0, x1: Optional[int] = None) -> np.ndarray:
        """Read a sub-box without decompressing the rest of the cube."""
        y1 = self.ny if y1 is None else y1
        x1 = self.nx if x1 is None else x1
        z0, z1 = max(0, int(z0)), min(self.n_lambda, int(z1))
        y0, y1 = max(0, int(y0)), min(self.ny, int(y1))
        x0, x1 = max(0, int(x0)), min(self.nx, int(x1))
        if z1 <= z0 or y1 <= y0 or x1 <= x0:
            return np.zeros((max(z1 - z0, 0), max(y1 - y0, 0), max(x1 - x0, 0)),
                            dtype=np.float32)
        hdu = self._hdul[self.hdu_index]
        sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", VerifyWarning)
            # .section works for CompImageHDU *and* plain ImageHDU/PrimaryHDU
            # (verified bit-identical to a full-array slice in both cases).
            arr = np.asarray(hdu.section[sl])
        return arr.astype(np.float32, copy=False)

    def read_profile_patch(self, params: SolarParams, radius: float = 100.0):
        """Disk-centre box big enough to hold the profile-averaging mask.

        Returns ``(cube_box, (y0, x0))`` so the caller can translate the mask
        into box coordinates.
        """
        rad = float(radius)
        y0 = int(max(0, np.floor(params.cy - rad)))
        y1 = int(min(self.ny, np.ceil(params.cy + rad) + 1))
        x0 = int(max(0, np.floor(params.cx - rad)))
        x1 = int(min(self.nx, np.ceil(params.cx + rad) + 1))
        return self.read(0, self.n_lambda, y0, y1, x0, x1), (y0, x0)

    def close(self):
        if self._hdul is not None:
            self._hdul.close()
            self._hdul = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


#: recognised BUNIT spellings -> factor to km/s
_VELOCITY_UNITS = {
    "KM/S": 1.0, "KMS-1": 1.0, "KM S-1": 1.0, "KM/SEC": 1.0, "KMS": 1.0,
    "M/S": 1e-3, "MS-1": 1e-3, "M S-1": 1e-3, "M/SEC": 1e-3,
}


def velocity_unit_factor(bunit: str):
    """Factor converting ``bunit`` to km/s, or None if it is not a velocity."""
    if not bunit:
        return None
    key = str(bunit).strip().upper().replace(" ", "")
    if key in _VELOCITY_UNITS:
        return _VELOCITY_UNITS[key]
    for k, v in _VELOCITY_UNITS.items():
        if k.replace(" ", "") == key:
            return v
    return None


def open_plane(path: str, ext: Optional[int] = None, require_unit: bool = True):
    """Open a 2-D velocity map.  Returns ``(data_km_s, header)``.

    Two deliberate guards:

    * the HDU is located via ``.shape`` - touching ``.data`` on every HDU would
      decompress a compressed cube just to look at it;
    * ``BUNIT`` is honoured (m/s is converted) and an unrecognised or missing
      unit is refused, because a raw intensity map mis-named ``*_Vdop.fits``
      would otherwise be fitted as if it were km/s and produce a plausible but
      meaningless product.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", VerifyWarning)
        with fits.open(path, **_OPEN_KW) as hdul:
            idx = ext
            if idx is None:
                for i, hdu in enumerate(hdul):
                    shp = getattr(hdu, "shape", None)
                    if shp is not None and len(shp) == 2 and int(np.prod(shp)) > 0:
                        idx = i
                        break
            if idx is None:
                raise ValueError("no 2-D image found in %s" % path)
            hdu = hdul[idx]
            header = hdu.header.copy()
            bunit = str(header.get("BUNIT", ""))
            data = np.asarray(hdu.data, dtype=np.float32)
    factor = velocity_unit_factor(bunit)
    if factor is None:
        msg = ("%s: BUNIT=%r is not a recognised velocity unit; refusing to "
               "treat this 2-D map as km/s" % (os.path.basename(path), bunit))
        if require_unit:
            raise ValueError(msg)
        logger.warning(msg)
    elif factor != 1.0:
        data = (data * factor).astype(np.float32, copy=False)
        logger.info("%s: converted BUNIT=%s to km/s (x%g)",
                    os.path.basename(path), bunit, factor)
    return data, header
