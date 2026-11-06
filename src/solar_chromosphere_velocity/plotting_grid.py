"""Joint H-alpha + Fe I velocity-field grid.

This is the standard figure produced when both spectral lines are processed for
the same observation time: one row per line, one column per physical component.

    (a) V_LOS   (b) V_diffrot   (c) V_limb+V_merid+V_offset   (d) V_resid
    (e) V_LOS   (f) V_diffrot   (g) V_limb+V_merid+V_offset   (h) V_resid

Each panel is one physical quantity for one spectral line, and a row is one
observing time; rows therefore compare the two lines against each other.  The
third column sums three small terms and is an order of magnitude smaller than the
others.  The view limits are derived from ``R_SUN`` and ``CDELT1``, so the framing
follows the instrument rather than one data set.

matplotlib is imported lazily so the package still imports without it.
"""

from __future__ import annotations

import os
import string
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from .constants import local_velocity_limit_km_s

__all__ = [
    "Panel",
    "Row",
    "JOINT_COLUMNS",
    "plot_map_grid",
    "read_product",
    "product_geometry",
    "format_obstime",
    "nice_velocity_limit",
    "joint_grid_rows",
    "plot_joint_grid",
    "plot_joint_series",
    "find_joint_pairs",
    "make_joint_series",
]

#: (layer, panel label, fixed half-range or None, label font size or None)
JOINT_COLUMNS = (
    ("VLOS_OBS", r"$V_{\rm LOS}$", 3.0, None),
    ("V_DIFFROT", r"$V_{\rm diffrot}$", 2.0, None),
    ("V_OTHER", r"$V_{\rm limb}+V_{\rm meridian}$" + "\n"
     + r"$+\,V_{\rm offset}$", None, 18.5),
    ("V_SS", r"$V_{\rm SS}$", None, None),        # scan-saddle + flat field
    ("V_LOCAL", r"$V_{\rm local}$", 1.5, None),   # V_RESID fallback below
)

_LINE_LABEL = {"HA": r"H$\alpha$", "FEI": "Fe I"}


@dataclass
class Panel:
    """One map in the grid."""

    data: np.ndarray
    label: str
    extent: Tuple[float, float, float, float]
    vmin: float
    vmax: float
    cmap: str = "RdBu_r"
    annotation: Optional[str] = None
    label_fontsize: Optional[float] = None
    label_alpha: float = 0.65
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None


@dataclass
class Row:
    """One row of panels, that is, one spectral line at one observation time."""

    panels: Sequence[Panel]
    time: Optional[str] = None
    label: Optional[str] = None


def _require_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return matplotlib, plt
    except ImportError as exc:                     # pragma: no cover
        raise RuntimeError(
            "plotting requires matplotlib; install it with "
            "`pip install 'solar-chromosphere-velocity[plots]'`") from exc


# ---------------------------------------------------------------------------
# reading products
# ---------------------------------------------------------------------------
#: layers the joint grid actually needs
JOINT_LAYERS = ("VLOS_OBS", "V_DIFFROT", "V_LIMB", "V_MERID", "V_OFFSET",
                "V_SS", "V_LOCAL", "V_RESID")


def read_product(path: str, names: Optional[Sequence[str]] = None
                 ) -> Tuple[object, Dict[str, np.ndarray]]:
    """Read a v2 multi-extension product into ``(primary header, layers)``.

    ``names`` restricts the read to those extensions, which matters for a full
    disk product: ten float64 layers of ``2313 x 2304`` would be ~430 MB per
    file.  float32 is kept because plotting does not need float64.
    """
    from astropy.io import fits

    wanted = None if names is None else {n.upper() for n in names}
    with fits.open(path) as hdul:
        header = hdul[0].header.copy()
        layers: Dict[str, np.ndarray] = {}
        for hdu in hdul[1:]:
            if hdu.data is None or getattr(hdu.data, "ndim", 0) != 2:
                continue
            name = hdu.header.get("EXTNAME") or hdu.name
            if wanted is not None and str(name).upper() not in wanted:
                continue
            layers[name] = np.array(hdu.data, dtype=np.float32)
    return header, layers


def product_geometry(header, shape, *,
                     half_width_rsun: float = 1.155,
                     top_pad_rsun: float = 0.495):
    """Return ``(extent, xlim, ylim)`` in arcsec for a full-disk product.

    ``extent`` is the pixel-to-arcsec mapping of the data array; ``xlim`` and
    ``ylim`` are the view limits.  Both are derived from ``R_SUN`` and
    ``CDELT1``/``CDELT2`` so the framing adapts to the instrument rather than
    being tied to one data set, and the top of ``ylim`` is extended by
    ``top_pad_rsun`` solar radii so the component name does not cover the disk.

    The defaults reproduce the reviewed layout: at ``R_sun = 970 arcsec``
    they give ``x = +-1120`` and ``y = (-1120, +1450)`` arcsec, which is the
    framing used for the publication figures.
    """
    ny, nx = int(shape[0]), int(shape[1])
    cdelt1 = float(header.get("CDELT1", 1.0) or 1.0)
    cdelt2 = float(header.get("CDELT2", abs(cdelt1)) or abs(cdelt1))
    crpix1 = float(header.get("CRPIX1", (nx + 1) / 2) or (nx + 1) / 2)
    crpix2 = float(header.get("CRPIX2", (ny + 1) / 2) or (ny + 1) / 2)
    cx = float(header.get("CX", crpix1) or crpix1)
    cy = float(header.get("CY", crpix2) or crpix2)

    extent = ((1.0 - cx) * cdelt1, (nx + 1.0 - cx) * cdelt1,
              (1.0 - cy) * cdelt2, (ny + 1.0 - cy) * cdelt2)

    r_px = header.get("R_SUN")
    if r_px:
        rsun = abs(float(r_px)) * abs(cdelt1)
        half = half_width_rsun * rsun
        top = (1.0 + top_pad_rsun) * rsun
        return extent, (-half, half), (-half, top)
    # no solar radius: fall back to the data extent
    return extent, (extent[0], extent[1]), (extent[2], extent[3])


def format_obstime(header) -> Tuple[str, str]:
    """``(date, HH:MM)`` from the observing time card."""
    t = str(header.get("OBSTIME") or header.get("DATE-OBS")
            or header.get("DATE_OBS") or "").strip()
    if "T" not in t:
        return t, ""
    date, hhmm = t.split("T", 1)
    hhmm = hhmm[:5]
    if len(date) == 8 and date.isdigit():
        date = "%s-%s-%s" % (date[:4], date[4:6], date[6:])
    if len(hhmm) >= 4 and hhmm[:4].isdigit() and ":" not in hhmm:
        hhmm = "%s:%s" % (hhmm[:2], hhmm[2:4])
    return date, hhmm


def nice_velocity_limit(arrays: Iterable[np.ndarray], floor: float = 0.02,
                        factor: float = 1.05) -> float:
    """A round, symmetric velocity limit covering the 99.5th percentile."""
    vals = np.concatenate([np.asarray(a)[np.isfinite(a)].ravel()
                           for a in arrays])
    if vals.size == 0:
        return float(floor)
    v = max(float(floor), factor * float(np.percentile(np.abs(vals), 99.5)))
    exponent = np.floor(np.log10(v))
    return float(round(v, -int(exponent) + 1))


# ---------------------------------------------------------------------------
# the joint figure
# ---------------------------------------------------------------------------
def joint_grid_rows(ha_path: str, fe_path: str, *, font_scale: float = 2.5,
                    half_width_rsun: float = 1.155,
                    top_pad_rsun: float = 0.495) -> Sequence[Row]:
    """Build the two rows (H-alpha over Fe I) for one observation time."""
    loaded = []
    for path, key in ((ha_path, "HA"), (fe_path, "FEI")):
        header, layers = read_product(path, names=JOINT_LAYERS)
        resid_key = "V_LOCAL" if "V_LOCAL" in layers else "V_RESID"
        missing = [resid_key if name == "V_LOCAL" else name
                   for name, *_ in JOINT_COLUMNS
                   if name != "V_OTHER"
                   and (resid_key if name == "V_LOCAL" else name) not in layers]
        # V_SS is a level-0.5 intermediate: drop the column on older products
        missing = [m for m in missing if m != "V_SS"]
        if missing:
            raise ValueError("%s is missing layers %s"
                             % (os.path.basename(path), missing))
        layers["V_OTHER"] = (layers["V_LIMB"] + layers["V_MERID"]
                             + layers["V_OFFSET"])
        date, hhmm = format_obstime(header)
        label = _LINE_LABEL.get(key, key)
        loaded.append((header, layers, label, "%s  %s\n%s UT" % (label, date, hhmm)))

    vmax_other = nice_velocity_limit([x[1]["V_OTHER"] for x in loaded])
    rows = []
    for header, layers, label, time_label in loaded:
        extent, xlim, ylim = product_geometry(
            header, layers["VLOS_OBS"].shape,
            half_width_rsun=half_width_rsun, top_pad_rsun=top_pad_rsun)
        panels = []
        resid_label = (r"$V_{\rm local}$" if resid_key == "V_LOCAL"
                       else r"$V_{\rm resid}$")
        columns = [c for c in JOINT_COLUMNS
                   if (resid_key if c[0] == "V_LOCAL" else c[0]) in layers]
        for name, panel_label, fixed, lsize in columns:
            key = resid_key if name == "V_LOCAL" else name
            if name == "V_LOCAL":
                panel_label = resid_label
            # a line with a fixed V_LOCAL limit always uses it, so the same
            # quantity keeps one scale across every figure
            fixed_local = (local_velocity_limit_km_s(header.get("SPECLINE"))
                           if name == "V_LOCAL" else None)
            if fixed_local is not None:
                vmax = fixed_local
            elif fixed is not None:
                vmax = fixed
            elif name == "V_OTHER":
                vmax = vmax_other
            else:                       # V_SS has its own, much smaller scale
                vmax = nice_velocity_limit([layers[key]])
            annotation = None
            if name == "V_OTHER":
                annotation = "offset %+.4f" % float(
                    np.nanmedian(layers["V_OFFSET"]))
            panels.append(Panel(
                data=layers[key],
                label=panel_label,
                extent=extent,
                vmin=-vmax,
                vmax=vmax,
                annotation=annotation,
                label_fontsize=(lsize * font_scale / 2.5) if lsize else None,
                label_alpha=0.75 if name == "V_OTHER" else 0.65,
                xlim=xlim,
                ylim=ylim,
            ))
        rows.append(Row(panels=panels, time=time_label, label=None))
    return rows


def plot_map_grid(rows: Sequence[Row], out: str, *, dpi: int = 300,
                  font_scale: float = 2.5,
                  panel_height_in: Optional[float] = None,
                  gap_in: float = 0.048,
                  left_margin_in: Optional[float] = None,
                  bottom_margin_in: Optional[float] = None,
                  top_margin_in: Optional[float] = None,
                  right_margin_in: float = 0.05,
                  cbar_height_in: float = 0.10,
                  cbar_gap_in: float = 0.25,
                  x_label: str = "Solar X (arcsec)",
                  y_label: str = "Solar Y (arcsec)",
                  xticks: Optional[Sequence[float]] = None,
                  yticks: Optional[Sequence[float]] = None,
                  x_tick_labels: str = "first",
                  y_tick_labels: str = "first",
                  panel_letters: bool = True,
                  letter_start: int = 0,
                  component_labels: bool = True,
                  show_time: bool = True,
                  row_labels: bool = False,
                  unit: Optional[str] = "v (km/s)",
                cbar_per_row: bool = True,
                  verbose: bool = False) -> str:
    """Draw a ``rows x columns`` grid of maps and save it to ``out``.

    ``x_tick_labels`` / ``y_tick_labels`` accept ``"first"``, ``"all"`` or
    ``"none"``.  All panels must share the same view limits.

    ``cbar_per_row`` draws one scale row per data row, which is needed whenever
    the rows carry their own limits.
    """
    _, plt = _require_mpl()
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MaxNLocator

    if not rows:
        raise ValueError("rows is empty")
    nrow = len(rows)
    ncol = len(rows[0].panels)
    if ncol == 0 or any(len(r.panels) != ncol for r in rows):
        raise ValueError("every row must have the same non-zero number of panels")

    def limits(panel):
        x0, x1 = panel.xlim if panel.xlim is not None else panel.extent[:2]
        y0, y1 = panel.ylim if panel.ylim is not None else panel.extent[2:]
        return float(x0), float(x1), float(y0), float(y1)

    x0, x1, y0, y1 = limits(rows[0].panels[0])
    aspect = (x1 - x0) / (y1 - y0)
    for row in rows:
        for panel in row.panels:
            a, b, c, d = limits(panel)
            if not (np.isclose(a, x0) and np.isclose(b, x1)
                    and np.isclose(c, y0) and np.isclose(d, y1)):
                raise ValueError("all panels must share the same view limits")

    # ---- design sizes -----------------------------------------------------
    tick_pt = 8.0 * font_scale
    axis_pt = 8.0 * font_scale
    panel_pt = 11.0 * font_scale
    comp_pt = 9.2 * font_scale
    time_pt = 8.2 * font_scale
    cbar_pt = 6.8 * font_scale
    unit_pt = 8.0 * font_scale
    row_pt = 14.0 * font_scale

    ph = float(panel_height_in) if panel_height_in else 1.9 * font_scale
    pw = aspect * ph

    left = (float(left_margin_in) if left_margin_in is not None
            else 0.62 * font_scale + 0.10)
    bottom = (float(bottom_margin_in) if bottom_margin_in is not None
              else 0.70 * font_scale + 0.15)
    top = (float(top_margin_in) if top_margin_in is not None
           else 0.58 * font_scale + 0.10)
    if row_labels and any(r.label for r in rows):
        left += 0.50 * font_scale

    # every row that carries its own scale needs that strip of space above it
    bar_rows = list(range(nrow)) if cbar_per_row else [0]
    inter = (top + gap_in) if (cbar_per_row and nrow > 1) else 0.0

    fig_w = left + ncol * pw + (ncol - 1) * gap_in + right_margin_in
    fig_h = bottom + nrow * ph + (nrow - 1) * (gap_in + inter) + top
    fig = plt.figure(figsize=(fig_w, fig_h))

    # ---- panels -----------------------------------------------------------
    for i, row in enumerate(rows):
        for j, panel in enumerate(row.panels):
            x_in = left + j * (pw + gap_in)
            y_in = bottom + (nrow - 1 - i) * (ph + gap_in + inter)
            ax = fig.add_axes([x_in / fig_w, y_in / fig_h,
                               pw / fig_w, ph / fig_h])
            ax.imshow(np.ma.masked_invalid(panel.data), origin="lower",
                      extent=panel.extent, cmap=panel.cmap,
                      vmin=panel.vmin, vmax=panel.vmax,
                      interpolation="nearest", rasterized=True)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_aspect("equal", adjustable="box")

            if xticks is not None:
                ax.set_xticks(list(xticks))
            else:
                ax.xaxis.set_major_locator(MaxNLocator(3))
            if yticks is not None:
                ax.set_yticks(list(yticks))
            else:
                ax.yaxis.set_major_locator(MaxNLocator(3))

            show_x = (i == nrow - 1) and (
                x_tick_labels == "all"
                or (x_tick_labels == "first" and j == 0))
            show_y = (j == 0) and (
                y_tick_labels == "all"
                or (y_tick_labels == "first" and j == 0))
            ax.tick_params(axis="x", labelsize=tick_pt, width=1.5, length=4.0,
                           pad=2.5, bottom=(i == nrow - 1), labelbottom=show_x)
            ax.tick_params(axis="y", labelsize=tick_pt, width=1.5, length=4.0,
                           pad=2.5, left=(j == 0), labelleft=show_y)
            for spine in ax.spines.values():
                spine.set_linewidth(3.2)

            if i == nrow - 1:
                ax.set_xlabel(x_label, fontsize=axis_pt,
                              labelpad=4.0 * font_scale / 2.5)
            if j == 0:
                ax.set_ylabel(y_label, fontsize=axis_pt,
                              labelpad=5.0 * font_scale / 2.5)

            if panel_letters:
                letter = "(%s)" % string.ascii_lowercase[
                    (letter_start + i * ncol + j) % 26]
                ax.text(0.015, 0.985, letter, transform=ax.transAxes,
                        ha="left", va="top", fontsize=panel_pt, zorder=30,
                        bbox=dict(boxstyle="round,pad=0.12",
                                  facecolor="white", edgecolor="none",
                                  alpha=0.60))

            if component_labels and panel.label:
                ax.text(0.975, 0.985, panel.label, transform=ax.transAxes,
                        ha="right", va="top",
                        fontsize=panel.label_fontsize or comp_pt,
                        linespacing=1.10, zorder=25,
                        bbox=dict(boxstyle="round,pad=0.14",
                                  facecolor="white", edgecolor="none",
                                  alpha=panel.label_alpha))

            if panel.annotation:
                ax.text(0.025, 0.025, panel.annotation, transform=ax.transAxes,
                        ha="left", va="bottom", fontsize=8.0 * font_scale,
                        zorder=20,
                        bbox=dict(boxstyle="round,pad=0.14", facecolor="white",
                                  edgecolor="0.6", linewidth=0.5, alpha=0.90))

            if show_time and j == 0 and row.time:
                ax.text(0.975, 0.025, row.time, transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=time_pt,
                        color="0.05", zorder=20,
                        bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                                  edgecolor="0.55", linewidth=0.5, alpha=0.92))

    # ---- one scale row per column, above each row that needs its own --------
    for i in bar_rows:
        y_row = bottom + (nrow - 1 - i) * (ph + gap_in + inter)
        y_cbar = y_row + ph + cbar_gap_in
        for j in range(ncol):
            x_in = left + j * (pw + gap_in)
            cax = fig.add_axes([(x_in + 0.07 * pw) / fig_w, y_cbar / fig_h,
                                0.86 * pw / fig_w, cbar_height_in / fig_h])
            panel = rows[i].panels[j]
            lo, hi = float(panel.vmin), float(panel.vmax)
            ticks = [lo, 0.0, hi] if lo < 0 < hi else [lo, 0.5 * (lo + hi), hi]
            mappable = ScalarMappable(norm=Normalize(vmin=lo, vmax=hi),
                                      cmap=plt.get_cmap(panel.cmap))
            mappable.set_array([])
            cb = fig.colorbar(mappable, cax=cax, orientation="horizontal",
                              ticks=ticks)
            cb.ax.xaxis.set_ticks_position("top")
            cb.ax.xaxis.set_label_position("top")
            cb.ax.tick_params(labelsize=cbar_pt, length=3.0, width=1.2, pad=2.0)
            cb.outline.set_linewidth(0.7 * font_scale)
            if unit:
                cb.set_label(unit, fontsize=unit_pt, labelpad=4.0)

    if row_labels and any(r.label for r in rows):
        for i, row in enumerate(rows):
            if not row.label:
                continue
            y_in = bottom + (nrow - 1 - i) * (ph + gap_in + inter) + ph / 2.0
            fig.text(0.25 * font_scale / fig_w, y_in / fig_h, row.label,
                     rotation=90, ha="center", va="center",
                     fontsize=row_pt, fontweight="semibold")

    if verbose:
        print("[plotting_grid] figure %.2f x %.2f in; panel %.3f x %.3f in; "
              "gap %.4f in; cbar gap %.4f in; rows=%d cols=%d"
              % (fig_w, fig_h, pw, ph, gap_in, cbar_gap_in, nrow, ncol))

    out = _ensure_parent(out)
    fig.savefig(out, dpi=dpi, facecolor="white", bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)
    return out


def plot_joint_grid(ha_path: str, fe_path: str, out: str, *,
                    dpi: int = 350, font_scale: float = 2.5,
                    panel_height_in: float = 4.75, gap_in: float = 0.048,
                    xticks: Optional[Sequence[float]] = (-1000, 0, 1000),
                    yticks: Optional[Sequence[float]] = (-1000, 0, 1000),
                    half_width_rsun: float = 1.155,
                    top_pad_rsun: float = 0.495,
                    verbose: bool = False) -> str:
    """Draw the joint H-alpha + Fe I grid for one observation time."""
    rows = joint_grid_rows(ha_path, fe_path, font_scale=font_scale,
                          half_width_rsun=half_width_rsun,
                          top_pad_rsun=top_pad_rsun)
    return plot_map_grid(rows, out, dpi=dpi, font_scale=font_scale,
                         panel_height_in=panel_height_in, gap_in=gap_in,
                         xticks=xticks, yticks=yticks,
                         x_tick_labels="first", y_tick_labels="first",
                         unit="v (km/s)", verbose=verbose)


def plot_joint_series(pairs: Iterable[Tuple[str, str, str]], out_dir: str, *,
                      name_pattern: str = "%s_HA_FEI_scv_grid.png",
                      **kwargs) -> Dict[str, str]:
    """One joint figure per observation time.

    ``pairs`` yields ``(stamp, ha_path, fe_path)``.  Returns
    ``{stamp: figure path}``.
    """
    os.makedirs(out_dir, exist_ok=True)
    made: Dict[str, str] = {}
    for stamp, ha_path, fe_path in pairs:
        out = os.path.join(out_dir, name_pattern % stamp)
        made[stamp] = plot_joint_grid(ha_path, fe_path, out, **kwargs)
    return made



# ---------------------------------------------------------------------------
# discovering pairs in an output tree
# ---------------------------------------------------------------------------
def find_joint_pairs(root: str, *, lines: Sequence[str] = ("HA", "FEI")
                     ) -> list:
    """Find ``(stamp, ha_path, fe_path)`` triples under ``root``.

    A v2 product is named ``<YYYYMMDD>T<HHMMSS>_<LINE>_scv.fits``, so the
    observation stamp is shared by the two lines and is the pairing key.
    Only stamps that have *all* requested lines are returned, sorted by time.
    """
    import glob
    import re

    want = tuple(x.upper() for x in lines)
    found: Dict[str, Dict[str, str]] = {}
    for path in glob.glob(os.path.join(root, "**", "*_scv.fits"),
                          recursive=True):
        base = os.path.basename(path)
        m = re.match(r"^(\d{8}T\d{6})_([A-Za-z]+)_scv\.fits$", base)
        if not m:
            continue
        stamp, line = m.group(1), m.group(2).upper()
        if line not in want:
            continue
        found.setdefault(stamp, {})[line] = path
    pairs = []
    for stamp in sorted(found):
        entry = found[stamp]
        if all(line in entry for line in want):
            pairs.append((stamp, entry["HA"], entry["FEI"]))
    return pairs


def make_joint_series(root: str, out_dir: Optional[str] = None, *,
                      name_pattern: str = "%s_HA_FEI_scv_grid.png",
                      **kwargs) -> Dict[str, str]:
    """One joint H-alpha + Fe I grid per observation time found under ``root``.

    This is the standard two-line output: call it once the directory run has
    written both products.  Figures go to ``out_dir`` (default
    ``<root>/joint``).
    """
    pairs = find_joint_pairs(root)
    if not pairs:
        raise ValueError("no observation time has both H-alpha and Fe I "
                         "products under %s" % root)
    if out_dir is None:
        out_dir = os.path.join(root, "joint")
    return plot_joint_series(pairs, out_dir, name_pattern=name_pattern,
                             **kwargs)

def _ensure_parent(path: str) -> str:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path
