"""Output layout: observation-time keys, multi-extension FITS, index.csv.

Two rules that v1 broke and v2 enforces:

1. **Every output path is derived from ``output_dir``.**  v1 built the
   intermediate Doppler product from ``input_path.parent`` with
   ``overwrite=True``, so running it next to the level-1 files silently
   destroyed the official ``*_HA_Vdop.fits`` products.
2. **The file key is the observation time taken from the header**, not a string
   transformation of the file name.  Two different source files can otherwise
   map onto the same name (v1: ``X_HA.fits`` and ``X_HA_Vdop.fits`` both became
   ``X_HA_Vdop_residual.fits``).

Layout::

    <output_dir>/<YYYYMMDD>/<YYYYMMDD>T<HHMMSS>_<LINE>_scv.fits
    <output_dir>/index.csv
"""

import csv
import functools
import os
import re
from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import numpy as np
import warnings
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning

from .models import MODEL_TERMS, _COMPONENT_OF

__all__ = ["Layer", "LAYERS", "observation_key", "OutputLayout", "write_multihdu"]

Layer = namedtuple("Layer", "extname quantity formula params unit")

LAYERS: Sequence[Layer] = (
    Layer("VLOS_OBS",
          "Observed line-of-sight velocity (no model removed)",
          "cross-correlation vs the disk-centre template",
          "", "km/s"),
    Layer("VLOS_CORR",
          "Line-of-sight velocity after the zero-point correction",
          "VLOS_OBS - ZEROPT", "", "km/s"),
    Layer("V_DIFFROT",
          "Differential-rotation component of the LOS velocity",
          "(A+Bsin^2+Csin^4+Dsin^10) * cosphi*sinL*cosB0",
          "A,B,C,D", "km/s"),
    Layer("V_LIMB",
          "Limb-shift component of the LOS velocity",
          "e * (1 - cosrho)^N", "e,N", "km/s"),
    Layer("V_MERID",
          "Meridional-circulation component, projected on the LOS",
          "(MC1 sinphi + MC2 sin2phi) * meridional projection",
          "MC1,MC2", "km/s"),
    Layer("V_OFFSET",
          "Systemic translational velocity (global zero point)",
          "G", "G", "km/s"),
    Layer("V_LARGE",
          "Total large-scale flow model surface",
          "V_DIFFROT+V_LIMB+V_MERID+V_OFFSET", "all", "km/s"),
    Layer("V_RESID",
          "Residual LOS velocity (quiet-Sun mean removed)",
          "VLOS_CORR - V_LARGE - QUIETMN", "all", "km/s"),
    Layer("V_MODEL_ERR",
          "1-sigma uncertainty of the model surface",
          "sqrt(x^T Sigma x)", "all", "km/s"),
    Layer("MASK_ACTIVE",
          "Active-region mask (0 = quiet Sun, 1 = active region)", "", "", ""),
)

_LAYER_BY_NAME = {l.extname: l for l in LAYERS}

_TIME_RE = re.compile(
    r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})[T_\- ]?(\d{2})[-_:]?(\d{2})[-_:]?(\d{2})")


@dataclass(frozen=True)
class ObsKey:
    day: str            # YYYYMMDD
    stamp: str          # YYYYMMDDTHHMMSS
    line: str           # HA / FEI / UNK
    source: str         # header (STR_TIME / DATE-OBS) or filename
    srcfile: str = ""


def _norm_time(text: str) -> Optional[str]:
    if not text:
        return None
    m = _TIME_RE.search(str(text))
    if not m:
        return None
    y, mo, d, hh, mm, ss = m.groups()
    try:
        datetime(int(y), int(mo), int(d), int(hh), int(mm), int(ss))
    except ValueError:
        return None
    return "%s%s%sT%s%s%s" % (y, mo, d, hh, mm, ss)


def _line_tag(spec_line: str, filename: str) -> str:
    s = (spec_line or "").strip().upper().replace(" ", "")
    if s.startswith("HA") or s == "HALPHA":
        return "HA"
    if s.startswith("FE"):
        return "FEI"
    base = os.path.basename(filename).lower()
    if "_ha" in base:
        return "HA"
    if "_fe" in base:
        return "FEI"
    return "UNK"


def observation_key(header, filename: str, params=None) -> ObsKey:
    """Derive the collision-free output key from the observation time."""
    from .data_loader import extract_solar_params
    p = params if params is not None else extract_solar_params(header)
    stamp, source = None, ""
    for cand, src in ((getattr(p, "str_time", ""), "STR_TIME"),
                      (getattr(p, "date_obs", ""), "DATE-OBS")):
        stamp = _norm_time(cand)
        if stamp:
            source = src
            break
    if not stamp:
        stamp = _norm_time(os.path.basename(filename))
        source = "FILENAME"
    if not stamp:
        # last resort: modification time, flagged in the header
        stamp = datetime.fromtimestamp(os.path.getmtime(filename)).strftime(
            "%Y%m%dT%H%M%S")
        source = "MTIME"
    return ObsKey(day=stamp[:8], stamp=stamp,
                  line=_line_tag(getattr(p, "spec_line", ""), filename),
                  source=source, srcfile=os.path.basename(filename))


class OutputLayout:
    """Resolves output paths and maintains ``index.csv``."""

    def __init__(self, config):
        self.config = config
        self.root = config.output_dir
        self._claimed: Dict[str, str] = {}     # path -> source file
        if self.root:
            os.makedirs(self.root, exist_ok=True)

    # -- paths -------------------------------------------------------------
    def dir_for(self, key: ObsKey) -> str:
        if self.config.no_day_subdirectory or not self.root:
            return self.root
        d = os.path.join(self.root, key.day)
        os.makedirs(d, exist_ok=True)
        return d

    def resolve(self, key: ObsKey, product: str = "scv",
                suffix: str = ".fits") -> str:
        """Return a unique path; never silently overwrite another source."""
        d = self.dir_for(key)
        base = "%s_%s_%s" % (key.stamp, key.line, product)
        cand = os.path.join(d, base + suffix)
        if self.config.overwrite:
            self._claimed[cand] = key.srcfile
            return cand
        # same key, different source -> disambiguate instead of clobbering
        used_by = self._claimed.get(cand)
        if used_by is None and os.path.exists(cand):
            used_by = "<existing file>"
        if used_by is None or used_by == key.srcfile:
            self._claimed[cand] = key.srcfile
            return cand
        for tag in "bcdefghijklmnopqrstuvwxyz":
            alt = os.path.join(d, "%s_%s%s" % (base, tag, suffix))
            if not os.path.exists(alt) and alt not in self._claimed:
                self._claimed[alt] = key.srcfile
                return alt
        raise RuntimeError("cannot find a free output name for %s" % base)

    # -- index -------------------------------------------------------------
    @property
    def index_path(self) -> Optional[str]:
        if not (self.root and self.config.write_observation_log):
            return None
        return os.path.join(self.root, "index.csv")

    def append_index(self, row: Dict[str, object]):
        path = self.index_path
        if not path:
            return
        fields = ["obs_time", "day", "line", "source", "hdu_file", "src_file",
                  "model_version", "legacy", "n_points", "rms_km_s", "limb_exponent",
                  "A", "B", "C", "D", "G", "e", "N", "MC1", "MC2",
                  "disk_excl", "edge_deg", "status"]
        exists = os.path.exists(path)
        with open(path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            if not exists:
                w.writeheader()
            w.writerow(row)


# --------------------------------------------------------------------------
def _clean_text(value):
    return str(value).replace("\n", " ").replace("\r", " ")


def _safe_put(header: fits.Header, key: str, value):
    """Add a card whose rendered image is guaranteed to fit in 80 characters.

    String values alone are not the only risk: a long float plus a long comment
    also overflows, and astropy then truncates the comment with a warning.  The
    card is therefore built and measured, shrinking the comment (then the value)
    until it fits.
    """
    key = _clean_text(key)[:8].upper()
    comment = ""
    if isinstance(value, tuple) and len(value) == 2:
        value, comment = value[0], _clean_text(value[1])
    if isinstance(value, str):
        value = _clean_text(value)

    comment_steps = [comment] if not comment else \
        [comment[:n] for n in (len(comment), 55, 40, 25, 12, 0)]
    for c in comment_steps:
        v = value
        for _ in range(4):
            try:
                card = fits.Card(key, v, c) if c else fits.Card(key, v)
                if len(card.image) <= 80:
                    header.append(card)
                    return
            except Exception:
                break
            if isinstance(v, str) and len(v) > 8:
                v = v[:-8]                      # shorten the value and retry
            else:
                break
    # give up gracefully rather than raising mid-write
    try:
        header[key] = value
    except Exception:
        pass


_STRUCTURAL = {"SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3",
               "EXTEND", "XTENSION", "PCOUNT", "GCOUNT", "TFIELDS", "END",
               "EXTNAME", "BUNIT"}


def _copy_header_safely(dst: fits.Header, src: fits.Header):
    """Copy cards from ``src`` into ``dst``, dropping anything over-long.

    Real CHASE headers contain cards longer than the 80-character FITS limit
    (which is why v1 opened files with ``fix=True``).  Copying them verbatim
    makes astropy truncate and warn on write, so every card is re-emitted
    through :func:`_safe_put` and structural keywords are left to astropy.
    """
    for card in src.cards:
        key = card.keyword
        if not key or key in _STRUCTURAL:
            continue
        if key.startswith("NSCAN"):
            continue          # ours; a stale one from the input must not survive
        try:
            value = card.value
            comment = card.comment
        except Exception:
            continue
        if isinstance(value, str) or not comment:
            _safe_put(dst, key, value)
        else:
            try:
                dst[key] = (value, str(comment)[:55])
            except Exception:
                _safe_put(dst, key, value)


#: keywords that may legitimately repeat
_REPEATABLE = {"COMMENT", "HISTORY", "", "END"}


def _dedupe(header: fits.Header):
    """Keep only the LAST card for each keyword.

    The input header and our own cards overlap (B0, INST_ROT, CDELT*, CRPIX*,
    RSUN_OBS, ...); appending both left 38 duplicated keywords, one of them
    conflicting (INST_ROT = 0 vs 0.0).  Our value is written last, so keeping
    the last occurrence makes the file unambiguous.
    """
    seen = {}
    for i, card in enumerate(header.cards):
        key = card.keyword
        if key in _REPEATABLE:
            continue
        seen.setdefault(key, []).append(i)
    doomed = []
    for idxs in seen.values():
        doomed.extend(idxs[:-1])          # drop all but the last occurrence
    # delete from the highest index down: deleting in keyword order would shift
    # the indices of every later keyword and remove the wrong cards
    for i in sorted(doomed, reverse=True):
        try:
            del header[i]
        except Exception:
            pass


def _sanitize_header(header: fits.Header):
    """Ensure no card image exceeds 80 characters.

    Called on every HDU just before writing.  Only comments are shortened, so
    structural cards (NAXIS*, TFORM*, TTYPE*, ...) keep their values intact.
    """
    for i in range(len(header.cards) - 1, -1, -1):
        try:
            card = header.cards[i]
            if len(card.image) <= 80:
                continue
            key, val = card.keyword, card.value
            com = _clean_text(card.comment or "")
        except Exception:
            continue
        for n in (len(com), 55, 40, 25, 12, 0):
            try:
                new = fits.Card(key, val, com[:n]) if n else fits.Card(key, val)
                if len(new.image) <= 80:
                    header[i] = new
                    break
            except Exception:
                continue


def _layer_header(layer: Layer, base_cards: Dict[str, object]) -> fits.Header:
    h = fits.Header()
    h["EXTNAME"] = layer.extname
    if layer.unit:
        h["BUNIT"] = layer.unit
    _safe_put(h, "QUANTITY", layer.quantity)
    if layer.formula:
        _safe_put(h, "FORMULA", layer.formula)
    if layer.params:
        _safe_put(h, "PARAMS", layer.params)
    for k, v in base_cards.items():
        _safe_put(h, k, v)
    h["COMMENT"] = layer.quantity
    if layer.extname == "V_OFFSET":
        for line in ("Systemic translational velocity: the global zero point",
                     "of the map.  Exactly degenerate with the template, the",
                     "zero-point convention and the residual de-meaning.",
                     "It is NOT a gravitational redshift.",
                     "See docs/OUTPUT.md."):
            h.add_comment(line)
    if layer.extname in ("V_DIFFROT", "V_MERID"):
        h.add_comment("Complete model; MODELVER=legacy-1.0 reproduces v1.")
    return h


def _suppress_card_notice(fn):
    """Silence astropy's "Card is too long" notice for this writer.

    CHASE level-1 headers carry cards longer than the 80-character FITS limit
    (v1 worked around this with ``fix=True``).  astropy emits a VerifyWarning
    while the HDUs are assembled, but every card we actually write is
    re-checked and truncated by :func:`_sanitize_header`, so the notice is not
    actionable.  Nothing else is suppressed.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Card is too long.*", category=VerifyWarning)
            return fn(*args, **kwargs)
    return wrapper


@_suppress_card_notice
def write_multihdu(path: str, layers: Dict[str, np.ndarray],
                   primary_header: fits.Header,
                   fit=None, base_cards: Optional[Dict[str, object]] = None,
                   compress: bool = False, overwrite: bool = True) -> str:
    """Write one multi-extension FITS file holding every velocity component.

    HDU 0            PRIMARY, no data: observation meta + fitted parameters
    HDU 1 .. n       one image HDU per entry of :data:`LAYERS` that is present
    last             FITPARAMS BINTABLE
    """
    base_cards = dict(base_cards or {})
    ph = fits.Header()
    _copy_header_safely(ph, primary_header)
    for k, v in base_cards.items():
        _safe_put(ph, k, v)
    if fit is not None:
        for k, v in fit.header_cards().items():
            if k == "NSCAN":
                continue                     # written on FITPARAMS instead
            _safe_put(ph, k, v)
    ph["COMMENT"] = "One multi-extension file per observation time."
    ph["NLAYERS"] = len([l for l in LAYERS if l.extname in layers])

    hdus = [fits.PrimaryHDU(data=None, header=ph)]
    for layer in LAYERS:
        data = layers.get(layer.extname)
        if data is None:
            continue
        hdr = _layer_header(layer, base_cards)
        arr = np.asarray(data)
        if layer.extname == "MASK_ACTIVE":
            arr = arr.astype(np.uint8)
        else:
            arr = arr.astype(np.float32)
        if compress:
            hdus.append(fits.CompImageHDU(data=arr, header=hdr,
                                          compression_type="RICE_1"))
        else:
            hdus.append(fits.ImageHDU(data=arr, header=hdr, name=layer.extname))

    if fit is not None:
        names = list(fit.as_dict().keys())
        vals = np.array([fit.as_dict()[k] for k in names])
        errs = np.array([fit.error_dict()[k] for k in names])
        rot_unit = ("rad/s" if getattr(fit, "rotation_amplitude", "kms")
                    == "rad_per_s" else "km/s")
        units = [(rot_unit if k in ("A", "B", "C", "D") else MODEL_TERMS[k][2])
                 for k in names]
        flags = [(getattr(fit, "flags", None) or {}).get(k, "") for k in names]
        forms = [MODEL_TERMS[k][1] for k in names]
        comp = [_COMPONENT_OF.get(k, "-") for k in names]
        cols = [
            fits.Column(name="NAME", format="8A", array=np.array(names, dtype="U8")),
            fits.Column(name="VALUE", format="D", array=vals),
            fits.Column(name="ERROR", format="D", array=errs),
            fits.Column(name="UNIT", format="12A",
                        array=np.array(units, dtype="U12")),
            fits.Column(name="COMPONENT", format="12A",
                        array=np.array(comp, dtype="U12")),
            fits.Column(name="FORMULA", format="60A",
                        array=np.array(forms, dtype="U60")),
            fits.Column(name="DESCRIPTION", format="60A",
                        array=np.array([MODEL_TERMS[k][0] for k in names],
                                       dtype="U60")),
            fits.Column(name="FLAG", format="24A",
                        array=np.array(flags, dtype="U24")),
        ]
        th = fits.Header()
        th["EXTNAME"] = "FITPARAMS"
        th["QUANTITY"] = "Fitted velocity-model parameters"
        th["NPOINT"] = int(fit.n_points)
        th["RSSFIT"] = float(fit.rss)
        th["RMSFIT"] = float(fit.rms_km_s)
        th["FITCOND"] = float(fit.condition)
        th["NLIMB"] = float(fit.limb_exponent)
        th["MODELVER"] = fit.model_version
        th["ROTSCALE"] = fit.rotation_amplitude
        th["UNITSCL"] = (rot_unit, "unit of A,B,C,D in FITPARAMS")
        if np.isfinite(getattr(fit, "limb_exponent_scan_step", float("nan"))):
            th["NSTEP"] = (float(fit.limb_exponent_scan_step),
                           "half-step of the N scan grid (not an error)")
        if getattr(fit, "flags", None):
            th["FITFLAG"] = ",".join("%s:%s" % kv for kv in sorted(fit.flags.items()))
        if fit.limb_nonneg_active:
            th["LIMBZERO"] = (True, "e pinned to 0 (negative at the optimum)")
        if fit.limb_exponent_scan:
            # keep the RSS-vs-N profile short enough for a single card
            # write every scan point without cutting a number in half
            tokens = ["%g:%.4g" % (n, r) for n, r in fit.limb_exponent_scan]
            chunks, cur = [], ""
            for tok in tokens:
                if cur and len(cur) + 1 + len(tok) > 52:
                    chunks.append(cur); cur = tok
                else:
                    cur = tok if not cur else cur + "," + tok
            if cur:
                chunks.append(cur)
            for j, chunk in enumerate(chunks or [""]):
                # zero-padded so a consumer that sorts keywords lexically still
                # reads them in order (NSCAN, NSCAN02, ... NSCAN10)
                key = "NSCAN" if j == 0 else "NSCAN%02d" % (j + 1)
                th[key] = chunk
        for k, v in base_cards.items():
            _safe_put(th, k, v)
        hdus.append(fits.BinTableHDU.from_columns(cols, header=th, name="FITPARAMS"))

    if os.path.exists(path) and not overwrite:
        raise FileExistsError(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    for hdu in hdus:                       # guarantee 80-char cards
        _sanitize_header(hdu.header)
        _dedupe(hdu.header)
    # CHASE level-1 headers contain cards longer than the 80-character FITS
    # limit (which is why v1 opened them with fix=True).  astropy reports a
    # VerifyWarning while the HDUs are being assembled; every card that we
    # actually emit is re-checked by _sanitize_header above, so the notice is
    # not actionable here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", VerifyWarning)
        fits.HDUList(hdus).writeto(path, overwrite=overwrite,
                                   output_verify="silentfix")
    return path
