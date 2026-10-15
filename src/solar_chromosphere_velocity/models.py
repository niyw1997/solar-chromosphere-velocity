"""Velocity model: linear least squares with a profile-likelihood over N.

The v1 model

    V = (A + B sin^2 phi + C sin^4 phi + D sin^10 rho) cos phi sin L
        + e (1 - cos rho)^N + MC1 sin phi + MC2 sin 2 phi + G

was fitted with ``scipy.optimize.curve_fit`` on ~2.4e6 points (measured ~120 s,
numerical Jacobian, and it did **not** reach the optimum: RSS 1 745 020 vs
1 744 887 for the exact solution).

Because the model is linear in every parameter except ``N``, v2 solves the
normal equations directly: one pass over the sample accumulates ``X^T X`` and
``X^T y`` for every candidate ``N`` at once, so the design matrix is never
materialised (constant memory, measured 0.02 s for the full 2.4e6-point sample).
``N`` is then chosen by scanning the residual sum of squares.

The complete physics (v2 default) adds what v1 dropped:
  * the rotation amplitude is scaled by R_sun (and optionally reported as an
    angular rate, matching the Snodgrass-type Omega expansion);
  * the ``cos B0`` factor on the rotation term;
  * the line-of-sight projection of the meridional circulation,
    ``cos L sin phi cos B0 - cos phi sin B0`` (v1 passed ``cos_lon`` into the
    model and then never used it).
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from .constants import R_SUN_KM

__all__ = [
    "MODEL_TERMS",
    "VelocityModel",
    "ModelFit",
    "NormalEquations",
    "fit_model_streaming",
    "fit_model",
]

# v1 parameter order, kept for continuity with existing parameter files.
TERM_ORDER = ("A", "B", "C", "D", "G", "e", "N", "MC1", "MC2")

MODEL_TERMS = {
    "A": ("Equatorial rotation amplitude", "A * cos(phi) sin(L) cos(B0)", "km/s"),
    "B": ("Rotation, sin^2(phi) term", "B sin^2(phi) cos(phi) sin(L) cos(B0)",
          "km/s"),
    "C": ("Rotation, sin^4(phi) term", "C sin^4(phi) cos(phi) sin(L) cos(B0)",
          "km/s"),
    "D": ("Rotation, sin^10(rho) limb term",
          "D sin^10(rho) cos(phi) sin(L) cos(B0)", "km/s"),
    "G": ("Systemic translational velocity (global zero point)",
          "G", "km/s"),
    "e": ("Limb-shift amplitude", "e (1 - cos rho)^N", "km/s"),
    "N": ("Limb-shift power-law exponent", "(1 - cos rho)^N", "-"),
    "MC1": ("Meridional circulation, sin(phi) term",
            "MC1 sin(phi) (cos L sin phi cos B0 - cos phi sin B0)", "km/s"),
    "MC2": ("Meridional circulation, sin(2 phi) term",
            "MC2 sin(2 phi) (cos L sin phi cos B0 - cos phi sin B0)", "km/s"),
}

_COMPONENT_OF = {"A": "rotation", "B": "rotation", "C": "rotation", "D": "rotation",
                 "e": "limb", "MC1": "meridional", "MC2": "meridional",
                 "G": "offset"}


# --------------------------------------------------------------------------
def _base_columns(coords: Dict[str, np.ndarray], model_version: str,
                  rotation_radius: bool):
    """Columns that do not depend on N: A, B, C, D, MC1, MC2, G (7 columns)."""
    sp = coords["sin_lat"]
    s2p = coords["sin_2lat"]
    cp = coords["cos_lat"]
    sL = coords["sin_lon"]
    cL = coords["cos_lon"]
    rho = coords["rho_rad"]
    b0 = coords["b0_rad"]

    if model_version == "legacy-1.0":
        proj = cp * sL
        merid = None                       # unprojected in v1
        scale = 1.0
    else:
        proj = cp * sL * np.cos(b0)
        merid = cL * sp * np.cos(b0) - cp * np.sin(b0)
        scale = R_SUN_KM if rotation_radius else 1.0

    a = scale * proj
    b = scale * sp ** 2 * proj
    c = scale * sp ** 4 * proj
    d = scale * np.sin(rho) ** 10 * proj
    if merid is None:
        mc1 = sp
        mc2 = s2p
    else:
        mc1 = sp * merid
        mc2 = s2p * merid
    g = np.ones_like(sp)
    return np.stack([a, b, c, d, mc1, mc2, g], axis=1)


def _limb_column(coords: Dict[str, np.ndarray], limb_exponent: float):
    return (1.0 - np.cos(coords["rho_rad"])) ** float(limb_exponent)


def _assemble(base: np.ndarray, limb: np.ndarray) -> np.ndarray:
    """Base (n,7) + limb (n,) -> design (n,8) in order [A,B,C,D,e,MC1,MC2,G]."""
    n = base.shape[0]
    X = np.empty((n, 8), dtype=np.float64)
    X[:, 0:4] = base[:, 0:4]
    X[:, 4] = limb
    X[:, 5] = base[:, 4]
    X[:, 6] = base[:, 5]
    X[:, 7] = base[:, 6]
    return X


# --------------------------------------------------------------------------
class NormalEquations:
    """Streaming accumulators for an ordinary least-squares problem."""

    def __init__(self, ncol: int = 8):
        self.ncol = int(ncol)
        self.XtX = np.zeros((ncol, ncol), dtype=np.float64)
        self.Xty = np.zeros(ncol, dtype=np.float64)
        self.yy = 0.0
        self.n = 0

    def add(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.XtX += X.T @ X
        self.Xty += X.T @ y
        self.yy += float(y @ y)
        self.n += int(y.size)

    # -- solving -----------------------------------------------------------
    def solve(self) -> Tuple[np.ndarray, float]:
        p = np.linalg.solve(self.XtX, self.Xty)
        rss = float(self.yy - 2.0 * p @ self.Xty + p @ self.XtX @ p)
        return p, max(rss, 0.0)

    def solve_drop(self, drop: int) -> Tuple[np.ndarray, float, np.ndarray]:
        """Solve with column ``drop`` pinned to zero (active-set step for e>=0)."""
        keep = [i for i in range(self.ncol) if i != drop]
        XtX = self.XtX[np.ix_(keep, keep)]
        Xty = self.Xty[keep]
        pk = np.linalg.solve(XtX, Xty)
        rss = float(self.yy - 2.0 * pk @ Xty + pk @ XtX @ pk)
        p = np.zeros(self.ncol)
        p[keep] = pk
        return p, max(rss, 0.0), np.asarray(keep)

    def covariance(self, p: np.ndarray, rss: float,
                   keep: Optional[np.ndarray] = None) -> np.ndarray:
        if keep is None:
            XtX = self.XtX
        else:
            XtX = self.XtX[np.ix_(keep, keep)]
        dof = max(self.n - XtX.shape[0], 1)
        sig2 = rss / dof
        try:
            cov_red = sig2 * np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            cov_red = sig2 * np.linalg.pinv(XtX)
        if keep is None:
            return cov_red
        cov = np.zeros((self.ncol, self.ncol))
        cov[np.ix_(keep, keep)] = cov_red
        return cov


# --------------------------------------------------------------------------
@dataclass
class ModelFit:
    """Fitted model, exposed in the v1 parameter order."""

    params: np.ndarray                 # (9,) in TERM_ORDER
    errors: np.ndarray                 # (9,) 1-sigma
    cov: np.ndarray                    # (9,9) in TERM_ORDER
    limb_exponent: float
    rss: float
    n_points: int
    model_version: str
    rotation_amplitude: str
    condition: float = float("nan")
    limb_exponent_scan: Optional[Sequence[Tuple[float, float]]] = None
    limb_nonneg_active: bool = False
    limb_exponent_scan_step: float = float("nan")   # half the scan grid spacing
    dropped: Tuple[str, ...] = ()
    flags: Dict[str, str] = field(default_factory=dict)

    def is_identifiable(self, name: str) -> bool:
        return name not in self.flags

    def flagged(self) -> Tuple[str, ...]:
        return tuple(sorted(self.flags))

    # -- lookups -----------------------------------------------------------
    def as_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in zip(TERM_ORDER, self.params)}

    def error_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in zip(TERM_ORDER, self.errors)}

    @property
    def rms_km_s(self) -> float:
        return float(np.sqrt(self.rss / max(self.n_points, 1)))

    def omega_deg_per_day(self) -> Optional[float]:
        """A expressed as an angular rate (only meaningful for km/s amplitude)."""
        if self.rotation_amplitude != "kms":
            return None
        return float(self.params[0] / R_SUN_KM * np.degrees(1.0) * 86400.0)

    def header_cards(self) -> Dict[str, object]:
        cards = {"NPOINT": int(self.n_points), "RSSFIT": float(self.rss),
                 "RMSFIT": self.rms_km_s, "FITCOND": float(self.condition),
                 "NLIMB": float(self.limb_exponent), "MODELVER": self.model_version,
                 "ROTSCALE": self.rotation_amplitude,
                 "V_OFFSET": (float(self.params[4]),
                              "systemic translational velocity (km/s)")}
        for name, val, err in zip(TERM_ORDER, self.params, self.errors):
            cards["FIT_" + name] = float(val)
            if self.is_identifiable(name):
                cards["ERR_" + name] = float(err)
            # flagged parameters deliberately get no ERR_* card: writing 0.0
            # would read as a perfectly measured value.  See FITFLAG.
        if self.flags:
            cards["FITFLAG"] = ",".join("%s:%s" % kv
                                        for kv in sorted(self.flags.items()))
        om = self.omega_deg_per_day()
        if om is not None:
            cards["OMEGA_A"] = (float(om), "A as an angular rate (deg/day)")
        return cards


# --------------------------------------------------------------------------
class VelocityModel:
    """Evaluates the model and its individual components."""

    def __init__(self, model_version: str = "full-2.0", limb_exponent: float = 2.0,
                 rotation_radius: bool = False):
        self.model_version = model_version
        self.limb_exponent = float(limb_exponent)
        self.rotation_radius = bool(rotation_radius)

    # -- design ------------------------------------------------------------
    def design(self, coords: Dict[str, np.ndarray]) -> np.ndarray:
        base = _base_columns(coords, self.model_version, self.rotation_radius)
        limb = _limb_column(coords, self.limb_exponent)
        return _assemble(base, limb)

    def design_without_limb(self, coords: Dict[str, np.ndarray]) -> np.ndarray:
        base = _base_columns(coords, self.model_version, self.rotation_radius)
        n = base.shape[0]
        X = np.empty((n, 7), dtype=np.float64)
        X[:, 0:4] = base[:, 0:4]
        X[:, 4] = base[:, 4]
        X[:, 5] = base[:, 5]
        X[:, 6] = base[:, 6]
        return X

    # -- evaluation --------------------------------------------------------
    def params8(self, params) -> np.ndarray:
        """TERM_ORDER (9,) -> design order (8,) [A,B,C,D,e,MC1,MC2,G]."""
        d = dict(zip(TERM_ORDER, np.asarray(params, dtype=np.float64)))
        return np.array([d["A"], d["B"], d["C"], d["D"], d["e"], d["MC1"],
                         d["MC2"], d["G"]], dtype=np.float64)

    def evaluate(self, coords, params):
        return self.design(coords) @ self.params8(params)

    def components(self, coords, params):
        """Per-term surfaces, in km/s."""
        d = dict(zip(TERM_ORDER, np.asarray(params, dtype=np.float64)))
        sp = coords["sin_lat"]
        s2p = coords["sin_2lat"]
        cp = coords["cos_lat"]
        sL = coords["sin_lon"]
        cL = coords["cos_lon"]
        rho = coords["rho_rad"]
        b0 = coords["b0_rad"]
        if self.model_version == "legacy-1.0":
            proj = cp * sL
            merid = np.ones_like(sp)
            scale = 1.0
        else:
            proj = cp * sL * np.cos(b0)
            merid = cL * sp * np.cos(b0) - cp * np.sin(b0)
            scale = R_SUN_KM if self.rotation_radius else 1.0
        rotation = scale * (d["A"] + d["B"] * sp ** 2 + d["C"] * sp ** 4
                            + d["D"] * np.sin(rho) ** 10) * proj
        limb = d["e"] * (1.0 - np.cos(rho)) ** self.limb_exponent
        meridional = (d["MC1"] * sp + d["MC2"] * s2p) * merid
        offset = np.full_like(sp, d["G"])
        total = rotation + limb + meridional + offset
        return {"V_DIFFROT": rotation, "V_LIMB": limb, "V_MERID": meridional,
                "V_OFFSET": offset, "V_LARGE": total}

    def uncertainty(self, coords, cov9) -> np.ndarray:
        """Per-pixel 1-sigma of the model surface, km/s: sqrt(x^T Sigma x).

        ``cov9`` uses TERM_ORDER; the design uses [A,B,C,D,e,MC1,MC2,G], so the
        covariance is re-indexed [0,1,2,3,5,7,8,4] before contracting.
        """
        X = self.design(coords)                       # (n,8) design order
        cov8 = np.asarray(cov9, dtype=np.float64)[np.ix_([0, 1, 2, 3, 5, 7, 8, 4],
                                                         [0, 1, 2, 3, 5, 7, 8, 4])]
        var = np.einsum("ij,jk,ik->i", X, cov8, X, optimize=True)
        return np.sqrt(np.maximum(var, 0.0))


# --------------------------------------------------------------------------
def fit_model_streaming(blocks: Iterable[Tuple[Dict[str, np.ndarray], np.ndarray]],
                        limb_exponent_values: Sequence[float],
                        model_version: str = "full-2.0",
                        rotation_radius: bool = False,
                        limb_amplitude_positive: bool = True) -> ModelFit:
    """Fit by streaming blocks of ``(coords, velocity)``.

    Every candidate ``N`` gets its own normal-equation accumulator, so the
    sample is read exactly once and no design matrix is ever stored whole.
    """
    ns = [float(x) for x in limb_exponent_values]
    acc = {n: NormalEquations(8) for n in ns}
    n_points = 0
    for coords, y in blocks:
        y = np.asarray(y, dtype=np.float64)
        if y.size == 0:
            continue
        base = _base_columns(coords, model_version, rotation_radius)
        for n in ns:
            X = _assemble(base, _limb_column(coords, n))
            acc[n].add(X, y)
        n_points += int(y.size)
    if n_points == 0:
        raise ValueError("no data points for fitting")

    scan = []
    best = None
    for n in ns:
        p, rss = acc[n].solve()
        scan.append((n, rss))
        if best is None or rss < best[1]:
            best = (n, rss, p)
    n_best, rss_best, p_best = best

    keep = None
    dropped = ()
    limb_active = False
    if limb_amplitude_positive and p_best[4] < 0:
        # single active-set step: the unconstrained optimum violates e >= 0, so
        # the constrained optimum sits on the boundary e = 0.  Dropping the
        # column necessarily raises the RSS - that is the cost of the
        # constraint, not a reason to reject it.
        p_best, rss_best, keep = acc[n_best].solve_drop(4)
        limb_active = True
        dropped = ("e",)

    cov8 = acc[n_best].covariance(p_best, rss_best, keep=keep)
    cond = float(np.linalg.cond(acc[n_best].XtX)) if keep is None else \
        float(np.linalg.cond(acc[n_best].XtX[np.ix_(keep, keep)]))

    # design order [A,B,C,D,e,MC1,MC2,G] -> TERM_ORDER [A,B,C,D,G,e,N,MC1,MC2]
    perm = [0, 1, 2, 3, 7, 4, -1, 5, 6]
    params9 = np.empty(9)
    for k, src in enumerate(perm):
        params9[k] = n_best if src == -1 else p_best[src]
    cov9 = np.zeros((9, 9))
    real = [i for i, s in enumerate(perm) if s >= 0]
    idx = [perm[i] for i in real]
    cov9[np.ix_(real, real)] = cov8[np.ix_(idx, idx)]
    errors9 = np.sqrt(np.maximum(np.diag(cov9), 0.0))

    # ---- identifiability bookkeeping -------------------------------------
    # A parameter pinned to a bound or lying in a degenerate direction has no
    # meaningful uncertainty.  Reporting 0 would look like a perfect
    # measurement, so the error is set to NaN and the parameter is flagged.
    flags: Dict[str, str] = {}
    scan_step = float("nan")
    arr = np.asarray(scan, dtype=np.float64) if scan else None
    if arr is not None and len(arr) > 2:
        j = int(np.argmin(arr[:, 1]))
        if 0 < j < len(arr) - 1:
            scan_step = float(abs(arr[j + 1, 0] - arr[j - 1, 0]) / 2.0)
        spread = float(arr[:, 1].max() - arr[:, 1].min())
        if spread <= 1e-9 * max(abs(float(arr[:, 1].min())), 1.0):
            flags["N"] = "FLAT_SCAN"          # the objective does not see N
    if limb_active:
        flags["e"] = "PINNED_AT_ZERO"
        if "N" not in flags:
            flags["N"] = "DEGENERATE_WITH_E"  # limb term is identically zero
    if "N" not in flags:
        # N is chosen from a discrete scan and is never a design column, so its
        # covariance-derived error is structurally 0.0 - which reads as a
        # perfectly measured value.  Report it as unestimated instead.
        flags["N"] = "SCAN_GRID"
    for name in flags:
        i = TERM_ORDER.index(name)
        # only the *reported* 1-sigma becomes NaN.  The covariance matrix is
        # left alone: for a dropped column it is already zero in that row/column
        # (from solve_drop), which is exactly the right contribution to
        # sqrt(x^T Sigma x).  Blanking it made V_MODEL_ERR 100% NaN - a real
        # regression, caught by review.
        errors9[i] = np.nan

    return ModelFit(params=params9, errors=errors9, cov=cov9, limb_exponent=n_best,
                    rss=float(rss_best), n_points=int(n_points),
                    model_version=model_version,
                    rotation_amplitude="rad_per_s" if rotation_radius else "kms",
                    condition=cond, limb_exponent_scan=tuple(scan),
                    limb_nonneg_active=limb_active, limb_exponent_scan_step=scan_step,
                    dropped=dropped, flags=flags)


def fit_model(coords: Dict[str, np.ndarray], velocity: np.ndarray,
              limb_exponent_values: Sequence[float], **kwargs) -> ModelFit:
    """Convenience wrapper for in-memory samples."""
    return fit_model_streaming([(coords, velocity)], limb_exponent_values, **kwargs)
