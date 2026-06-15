"""Threshold extraction: locate the d_lo/d_hi crossing with a bootstrap CI.

Implements the crossing-bracket method of docs/benchmarking.md section 5. The
logical-error-rate curves p_L(p_0) for two code distances cross at the
threshold p_th: below it the larger distance suppresses errors (rate_hi
rate_lo), above it the larger distance is worse (rate_hi > rate_lo). The
crossing is located by linear interpolation of the first low->high sign change
of g(p_0) = rate_hi - rate_lo; its uncertainty is a parametric bootstrap,
resampling each rate within its (two-layer Monte-Carlo) stderr.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ThresholdEstimate:
    """Threshold point estimate with a percentile-bootstrap interval.

    Attributes
    ----------
    p_th
        Crossing point estimate; NaN if the curves do not cross within the
        supplied p_0 grid.
    ci_lo, ci_hi
        Bootstrap interval bounds (NaN if no bootstrap resample crossed).
    n_boot_valid
        Number of bootstrap resamples that produced an in-grid crossing.
    """

    p_th: float
    ci_lo: float
    ci_hi: float
    n_boot_valid: int


def _first_crossing(
    p0: NDArray[np.float64],
    rate_lo: NDArray[np.float64],
    rate_hi: NDArray[np.float64],
) -> float:
    """Linear-interpolated first low->high crossing of rate_hi - rate_lo, or NaN."""
    g = rate_hi - rate_lo
    idx = np.where(np.diff(np.sign(g)) > 0)[0]
    if idx.size == 0:
        return float("nan")
    i = int(idx[0])
    g0, g1 = float(g[i]), float(g[i + 1])
    return float(p0[i] + (p0[i + 1] - p0[i]) * (-g0) / (g1 - g0))


def crossing_threshold(
    p0: NDArray[np.float64],
    rate_lo: NDArray[np.float64],
    stderr_lo: NDArray[np.float64],
    rate_hi: NDArray[np.float64],
    stderr_hi: NDArray[np.float64],
    *,
    rng: np.random.Generator,
    n_boot: int = 4000,
    ci: float = 0.95,
) -> ThresholdEstimate:
    """Locate the d_lo/d_hi crossing threshold with a parametric-bootstrap CI.

    Parameters
    ----------
    p0
        Ascending physical error-rate grid, shape (n,).
    rate_lo, stderr_lo
        Logical error rate and stderr for the SMALLER distance, shape (n,).
    rate_hi, stderr_hi
        Same for the LARGER distance.
    rng
        Generator for the bootstrap resampling.
    n_boot
        Number of bootstrap resamples.
    ci
        Central interval mass (default 0.95).
    """
    p0 = np.asarray(p0, dtype=np.float64)
    rate_lo = np.asarray(rate_lo, dtype=np.float64)
    rate_hi = np.asarray(rate_hi, dtype=np.float64)
    stderr_lo = np.asarray(stderr_lo, dtype=np.float64)
    stderr_hi = np.asarray(stderr_hi, dtype=np.float64)

    pth = _first_crossing(p0, rate_lo, rate_hi)

    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        boots[b] = _first_crossing(
            p0,
            np.clip(rng.normal(rate_lo, stderr_lo), 0.0, None),
            np.clip(rng.normal(rate_hi, stderr_hi), 0.0, None),
        )
    valid = boots[~np.isnan(boots)]
    if valid.size == 0:
        return ThresholdEstimate(pth, float("nan"), float("nan"), 0)
    lo_q = 100.0 * (1.0 - ci) / 2.0
    hi_q = 100.0 * (1.0 + ci) / 2.0
    lo, hi = np.percentile(valid, [lo_q, hi_q])
    return ThresholdEstimate(pth, float(lo), float(hi), int(valid.size))
