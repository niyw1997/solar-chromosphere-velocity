"""Multi-panel visualisation of the v2 velocity components.

A v2 product splits the velocity field into physical components; this module
draws them as **separate subplots in one figure** (and can also write one file
per component).  Per-panel colour scaling is the default because the components
differ by orders of magnitude (the differential-rotation term is several km/s,
the limb term is ~0, the residual is ~1 km/s).

matplotlib is imported lazily, so the package still imports without it.

Typical use::

    from solar_chromosphere_velocity import plot_layers, plot_from_file

    plot_from_file("20260424T080829_HA_scv.fits", "layers.png")
    plot_from_file(path, "components.png",
                   panels=["V_DIFFROT", "V_LIMB", "V_MERID",
                           "V_OFFSET", "V_LARGE", "V_RESID"])
"""

import os
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "PANEL_ORDER",
    "PANEL_TITLES",
    "COMPONENT_PANELS",
    "plot_layers",
    "plot_from_file",
    "plot_components",
    "plot_summary",
    "write_component_figures",
]

#: default panel order (physics first, then the derived fields)
PANEL_ORDER = (
    "VLOS_OBS", "VLOS_CORR",
    "V_DIFFROT", "V_LIMB", "V_MERID", "V_OFFSET",
    "V_LARGE", "V_RESID", "V_MODEL_ERR", "MASK_ACTIVE",
)

PANEL_TITLES = {
    "VLOS_OBS": "Observed LOS velocity",
    "VLOS_CORR": "Zero-point corrected LOS velocity",
    "V_DIFFROT": "Differential rotation",
    "V_LIMB": "Limb shift",
    "V_MERID": "Meridional circulation",
    "V_OFFSET": "Systemic translation (offset)",
    "V_LARGE": "Large-scale flow (sum)",
    "V_RESID": "Residual velocity",
    "V_MODEL_ERR": "Model 1-sigma",
    "MASK_ACTIVE": "Active-region mask",
}

#: the four model terms plus the total and the residual
COMPONENT_PANELS = ("V_DIFFROT", "V_LIMB", "V_MERID", "V_OFFSET",
                    "V_LARGE", "V_RESID")

#: layers that are not signed velocities
_MASK_LAYERS = ("MASK_ACTIVE",)
_POSITIVE_LAYERS = ("V_MODEL_ERR",)


def _require_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return matplotlib, plt
    except ImportError as exc:              # pragma: no cover
        raise RuntimeError(
            "plotting requires matplotlib; install it with "
            "`pip install 'solar-chromosphere-velocity[plots]'`") from exc


def colormap(name):
    """Colormap lookup that also works on matplotlib < 3.5."""
    import matplotlib
    try:
        return matplotlib.colormaps[name]
    except Exception:                       # pragma: no cover - old matplotlib
        import matplotlib.cm as cm
        return cm.get_cmap(name)


def _resolve(layers, header=None) -> Tuple[Dict[str, np.ndarray], Optional[object]]:
    """Accept a dict of arrays or the path of a v2 multi-extension FITS.

    When a dict is given, ``header`` supplies the metadata used for the axes
    (CDELT1/CDELT2/CX/CY).  The pipeline passes the header it just wrote, so the
    in-memory and from-file paths produce identical axes.
    """
    if isinstance(layers, (str, bytes, os.PathLike)):
        from astropy.io import fits
        with fits.open(layers) as h:
            header = h[0].header.copy()
            out = {}
            for hd in h:
                name = hd.header.get("EXTNAME")
                if name and hd.data is not None and hd.data.ndim == 2:
                    out[name] = np.asarray(hd.data)
        return out, header
    return dict(layers), header


def _extent_arcsec(header, shape):
    """Pixel extent in arcsec relative to the disk centre (or None)."""
    if header is None:
        return None
    try:
        cd1 = float(header["CDELT1"])
        cd2 = float(header.get("CDELT2", cd1))
        cx = float(header.get("CX", header.get("CRPIX1")))
        cy = float(header.get("CY", header.get("CRPIX2")))
    except Exception:
        return None
    if not np.all(np.isfinite([cd1, cd2, cx, cy])) or cd1 == 0:
        return None
    ny, nx = shape
    return [(0 - cx) * cd1, (nx - 1 - cx) * cd1,
            (0 - cy) * cd2, (ny - 1 - cy) * cd2]


def _draw_panel(ax, name, data, header, *, cmap="bwr", percentile=99.5,
                per_panel_scale=True, vmin=None, vmax=None, overlay=None,
                show_colorbar=True):
    data = np.asarray(data, dtype=np.float64)
    ny, nx = data.shape
    im = None

    if name in _MASK_LAYERS:
        from matplotlib.colors import ListedColormap
        im = ax.imshow(data, origin="lower", cmap=ListedColormap(["white", "#c0392b"]),
                       vmin=0, vmax=1, interpolation="nearest")
        frac = float(np.nansum(data > 0)) / data.size
        ax.text(0.02, 0.98, "active: %.2f%%" % (100 * frac), transform=ax.transAxes,
                va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    else:
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            ax.text(0.5, 0.5, "no finite data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            return None
        if name in _POSITIVE_LAYERS:
            lo, hi = 0.0, float(np.percentile(finite, percentile))
        elif per_panel_scale and vmin is None and vmax is None:
            lim = float(np.percentile(np.abs(finite), percentile))
            lim = lim if lim > 0 else max(float(np.abs(finite).max()), 1e-9)
            lo, hi = -lim, lim
        else:
            lo = vmin if vmin is not None else float(np.nanmin(finite))
            hi = vmax if vmax is not None else float(np.nanmax(finite))
        cm = colormap(cmap)
        try:
            # set_bad is pending-deprecated in matplotlib 3.11
            cm = cm.with_extremes(bad="white")
        except Exception:                       # pragma: no cover - old mpl
            cm = cm.copy()
            cm.set_bad("white")
        im = ax.imshow(data, origin="lower", cmap=cm, vmin=lo, vmax=hi,
                       interpolation="nearest")
        if overlay is not None:
            try:
                ax.contour(np.asarray(overlay) > 0, levels=[0.5], colors="k",
                           linewidths=0.5, alpha=0.6, origin="lower")
            except Exception:
                pass
        ax.text(0.02, 0.98,
                "mean %+.3f  std %.3f km/s" % (float(np.nanmean(finite)),
                                               float(np.nanstd(finite))),
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ext = _extent_arcsec(header, (ny, nx))
    if ext is not None and im is not None:
        im.set_extent(ext)
        ax.set_xlabel("x [arcsec]")
        ax.set_ylabel("y [arcsec]")
    else:
        ax.set_xlabel("x [pixel]")
        ax.set_ylabel("y [pixel]")
    if show_colorbar and im is not None and name not in _MASK_LAYERS:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=8)
    return im


def plot_layers(layers: Union[Dict[str, np.ndarray], str], outfile: str, *,
                panels: Optional[Sequence[str]] = None, ncols: int = 3,
                dpi: int = 150, cmap: str = "bwr", percentile: float = 99.5,
                per_panel_scale: bool = True, vmin: Optional[float] = None,
                vmax: Optional[float] = None, overlay_mask: bool = False,
                title: Optional[str] = None,
                panel_size: Tuple[float, float] = (4.6, 4.2),
                show_colorbar: bool = True,
                header=None) -> str:
    """Draw the requested velocity components as subplots of one figure.

    Parameters
    ----------
    layers : dict or path
        Layer name -> 2-D array, or the path of a v2 multi-extension product.
    panels : sequence of str, optional
        Which layers to draw, in order.  Defaults to every panel present.
    per_panel_scale : bool
        Give each panel its own symmetric colour scale (recommended: the
        components have very different amplitudes).

    Returns the path of the written figure.
    """
    _, plt = _require_mpl()
    data, header = _resolve(layers, header)
    if panels is None:
        panels = [p for p in PANEL_ORDER if p in data]
    panels = [p for p in panels if p in data]
    if not panels:
        raise ValueError("none of the requested panels are present; available: %s"
                         % sorted(data))

    n = len(panels)
    ncols = max(1, min(int(ncols), n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, dpi=dpi,
                             figsize=(panel_size[0] * ncols,
                                      panel_size[1] * nrows), squeeze=False)
    overlay = data.get("MASK_ACTIVE") if overlay_mask else None
    for ax, name in zip(axes.ravel(), panels):
        _draw_panel(ax, name, data[name], header, cmap=cmap,
                    percentile=percentile, per_panel_scale=per_panel_scale,
                    vmin=vmin, vmax=vmax, overlay=overlay,
                    show_colorbar=show_colorbar)
        ax.set_title("%s\n%s" % (name, PANEL_TITLES.get(name, "")), fontsize=10)
    for ax in axes.ravel()[n:]:
        ax.axis("off")

    head = ""
    if header is not None:
        head = "%s  %s  model=%s" % (header.get("SRCFILE", ""),
                                     header.get("OBSTIME", ""),
                                     header.get("MODELVER", ""))
    fig.suptitle(title or ("Velocity components  " + head).strip(), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, plt, outfile, "png")


def plot_from_file(fits_path: str, outfile: Optional[str] = None, **kwargs) -> str:
    """Read a v2 product and plot it."""
    if outfile is None:
        outfile = os.path.splitext(fits_path)[0] + "_layers.png"
    return plot_layers(fits_path, outfile, **kwargs)


def plot_components(layers, outfile: str, **kwargs) -> str:
    """Preset layout: the four model terms, their sum, and the residual."""
    kwargs.setdefault("ncols", 3)
    return plot_layers(layers, outfile, panels=COMPONENT_PANELS, **kwargs)


def plot_summary(layers, outfile: str, dpi: int = 150, header=None, **kwargs) -> str:
    """2x3 overview: data, model total, residual, histogram, mask, 1-sigma."""
    _, plt = _require_mpl()
    data, header = _resolve(layers, header)
    fig, axes = plt.subplots(2, 3, dpi=dpi, figsize=(14.4, 8.6))
    axes = axes.ravel()

    for ax, name in ((axes[0], "VLOS_OBS"), (axes[1], "V_LARGE"),
                     (axes[2], "V_RESID"), (axes[4], "V_MODEL_ERR")):
        if name in data:
            _draw_panel(ax, name, data[name], header)
            ax.set_title("%s\n%s" % (name, PANEL_TITLES.get(name, "")), fontsize=10)
        else:
            ax.axis("off")

    ax = axes[3]
    if "V_RESID" in data:
        r = np.asarray(data["V_RESID"], dtype=np.float64)
        r = r[np.isfinite(r)]
        if r.size:
            ax.hist(r, bins=120, range=(-4, 4), density=True, alpha=0.75,
                    color="#4c72b0")
            mu, sd = float(r.mean()), float(r.std())
            if sd > 0:
                xs = np.linspace(-4, 4, 200)
                ax.plot(xs, np.exp(-(xs - mu) ** 2 / (2 * sd ** 2)) /
                        (sd * np.sqrt(2 * np.pi)), "r-", lw=1.5,
                        label="Gaussian  mu=%+.3f  sigma=%.3f km/s" % (mu, sd))
                ax.legend(fontsize=8)
    ax.set_xlabel("residual [km/s]")
    ax.set_ylabel("density")
    ax.set_title("V_RESID distribution", fontsize=10)

    ax = axes[5]
    if "MASK_ACTIVE" in data:
        _draw_panel(ax, "MASK_ACTIVE", data["MASK_ACTIVE"], header,
                    show_colorbar=False)
        ax.set_title("MASK_ACTIVE", fontsize=10)
    else:
        ax.axis("off")

    head = ""
    if header is not None:
        head = "%s  %s  model=%s  N=%s" % (header.get("SRCFILE", ""),
                                           header.get("OBSTIME", ""),
                                           header.get("MODELVER", ""),
                                           header.get("NLIMB", ""))
    fig.suptitle("Processing summary  " + head, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, plt, outfile, "png")


def write_component_figures(layers, base_path: str,
                            panels: Optional[Iterable[str]] = None,
                            **kwargs) -> Dict[str, str]:
    """Write one figure **per component** (``<base>_<LAYER>.png``).

    Returns a mapping layer -> path.
    """
    # caller may pass header=... through **kwargs; must not be duplicated below
    header = kwargs.pop("header", None)
    data, header = _resolve(layers, header)
    if panels is None:
        panels = [p for p in COMPONENT_PANELS if p in data]
    out = {}
    base = os.path.splitext(base_path)[0]
    for name in panels:
        if name not in data:
            continue
        out[name] = plot_layers(data, "%s_%s.png" % (base, name), panels=[name],
                                ncols=1, header=header, **kwargs)
    return out


def _save(fig, plt, outfile: str, fmt: str) -> str:
    root, ext = os.path.splitext(str(outfile))
    if not ext:
        outfile = root + "." + fmt
    os.makedirs(os.path.dirname(os.path.abspath(outfile)), exist_ok=True)
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    return outfile
