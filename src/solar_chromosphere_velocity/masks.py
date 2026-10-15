"""Disk masks and active-region detection.

v1 issues fixed here:
  * the disk/edge masks are now resolved explicitly, so the caller learns which
    constraint actually binds (with the v1 defaults ``edge_exclusion_angle=80``
    is completely absorbed by ``disk_mask_radius=0.95``);
  * active-region detection uses a robust (MAD) threshold instead of a
    ``3*std`` computed on a distribution that already contains the active
    regions - a test that flags ~0.3% of any noise field by construction;
  * the debug ``print`` in ``create_solar_mask`` is gone.

NOTE: the disk test is a circle in detector coordinates.  The P0 rotation
preserves the distance to the disk centre, so rotated and unrotated distances
are identical.
"""

from typing import Optional

import numpy as np
from scipy import ndimage

__all__ = [
    "disk_mask",
    "effective_radius_px",
    "effective_disk_mask",
    "create_solar_mask",
    "apply_disk_exclusion",
    "active_region_mask",
]


def disk_mask(shape, params, frac: float = 1.0) -> np.ndarray:
    """Boolean mask of ``r <= frac * R_sun`` (no rotation needed)."""
    ny, nx = shape
    yy = np.arange(ny, dtype=np.float64)[:, None]
    xx = np.arange(nx, dtype=np.float64)[None, :]
    r2 = (yy - params.cy) ** 2 + (xx - params.cx) ** 2
    rad = float(params.radius) * float(frac)
    return r2 <= rad ** 2


def effective_radius_px(params, disk_mask_radius: float,
                        edge_exclusion_angle: float):
    """Which radial cut actually applies for the *fit sample*.

    Returns ``(radius_px, binding, edge_radius_px, disk_radius_px)`` where
    ``binding`` is ``'disk'``, ``'edge'`` or ``'equal'``.
    """
    r_disk = float(params.radius) * float(disk_mask_radius)
    r_edge = float(params.radius) * np.sin(np.radians(float(edge_exclusion_angle)))
    if r_edge < r_disk - 1e-9:
        binding = "edge"
    elif r_disk < r_edge - 1e-9:
        binding = "disk"
    else:
        binding = "equal"
    return min(r_disk, r_edge), binding, r_edge, r_disk


def effective_disk_mask(shape, params, disk_mask_radius: float,
                        edge_exclusion_angle: float) -> np.ndarray:
    """Intersection of the disk and edge cuts (equivalent to the tighter one)."""
    rad, _, _, _ = effective_radius_px(params, disk_mask_radius,
                                       edge_exclusion_angle)
    return disk_mask(shape, params, frac=rad / float(params.radius))


def create_solar_mask(shape, center, radius, exclusion_ratio: float = 1.0):
    """v1-compatible signature: mask of ``r <= radius * exclusion_ratio``."""
    cy, cx = float(center[1]), float(center[0])
    ny, nx = shape
    yy = np.arange(ny, dtype=np.float64)[:, None]
    xx = np.arange(nx, dtype=np.float64)[None, :]
    rad = float(radius) * float(exclusion_ratio)
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= rad ** 2


def apply_disk_exclusion(data: np.ndarray, params, exclusion_ratio: float = 0.95):
    """Set everything beyond ``exclusion_ratio * R_sun`` to NaN (copy)."""
    if exclusion_ratio >= 1.0 and params is None:
        return np.array(data, copy=True)
    m = disk_mask(data.shape, params, frac=exclusion_ratio)
    out = np.array(data, dtype=np.float32, copy=True)
    out[~m] = np.nan
    return out


def active_region_mask(velocity: np.ndarray, params, config,
                       return_stats: bool = False):
    """Boolean mask of active regions inside the (clipped) solar disk.

    Legacy mode reproduces v1 exactly: ``|v - mean| > k*std`` over the disk.
    v2 mode uses median + 1.4826*MAD with iterative sigma clipping.
    """
    m_disk = disk_mask(velocity.shape, params, frac=config.disk_mask_radius)
    valid_sel = m_disk & np.isfinite(velocity)
    valid = velocity[valid_sel]
    if valid.size == 0:
        mask = np.zeros(velocity.shape, dtype=bool)
        stats = {"threshold": 0.0, "location": 0.0, "scale": 0.0, "method": "none"}
        return (mask, stats) if return_stats else mask

    if config.active_region_robust_threshold:
        loc = float(np.median(valid))
        scale = 1.4826 * float(np.median(np.abs(valid - loc)))
        for _ in range(3):
            if scale <= 0:
                break
            sel = np.abs(valid - loc) < 3.0 * scale
            if not sel.any():
                break
            new_loc = float(np.median(valid[sel]))
            new_scale = 1.4826 * float(np.median(np.abs(valid[sel] - new_loc)))
            if not np.isfinite(new_scale) or new_scale <= 0:
                break
            loc, scale = new_loc, new_scale
        method = "mad"
    else:
        loc = float(np.mean(valid))
        scale = float(np.std(valid))
        method = "mean_std"

    if not np.isfinite(scale) or scale <= 0:
        mask = np.zeros(velocity.shape, dtype=bool)
        stats = {"threshold": 0.0, "location": loc, "scale": 0.0, "method": method}
        return (mask, stats) if return_stats else mask

    threshold = float(config.active_region_sigma_threshold) * scale
    mask = (np.abs(velocity - loc) > threshold) & m_disk
    mask &= np.isfinite(velocity)

    structure = ndimage.generate_binary_structure(2, 2)
    labeled, num = ndimage.label(mask, structure=structure)
    if num > 0:
        sizes = ndimage.sum(mask, labeled, range(num + 1))
        small = np.zeros(num + 1, dtype=bool)
        small[1:] = sizes[1:] < config.min_active_region_area
        if small.any():
            mask[small[labeled]] = False
    if config.active_region_shrink_steps > 0:
        mask = ndimage.binary_erosion(mask, structure=structure,
                                      iterations=config.active_region_shrink_steps)
    if config.active_region_grow_steps > 0:
        mask = ndimage.binary_dilation(mask, structure=structure,
                                       iterations=config.active_region_grow_steps)
    stats = {"threshold": threshold, "location": loc, "scale": scale,
             "method": method, "n_pixels": int(mask.sum())}
    return (mask, stats) if return_stats else mask
