"""Threshold crossing extraction (docs/benchmarking.md section 5)."""

from __future__ import annotations

import math

import numpy as np

from nonmarkov_qec.benchmarks.threshold import crossing_threshold


def test_recovers_known_linear_crossing() -> None:
    # Two lines intersecting exactly at p_th = 0.0035; rate_hi steeper, so it
    # is below rate_lo for p0 < p_th and above for p0 > p_th.
    p0 = np.array([0.002, 0.003, 0.004, 0.005])
    p_th_true = 0.0035
    rate_lo = 0.01 + 1.0 * (p0 - p_th_true)
    rate_hi = 0.01 + 3.0 * (p0 - p_th_true)
    se = np.full_like(p0, 1e-5)
    est = crossing_threshold(
        p0, rate_lo, se, rate_hi, se, rng=np.random.default_rng(0), n_boot=2000
    )
    assert abs(est.p_th - p_th_true) < 1e-9
    assert est.ci_lo <= p_th_true <= est.ci_hi
    assert est.n_boot_valid > 1900


def test_no_crossing_returns_nan() -> None:
    p0 = np.array([0.002, 0.003, 0.004, 0.005])
    rate_lo = np.array([0.02, 0.03, 0.04, 0.05])
    rate_hi = rate_lo - 0.005  # always below: no low->high crossing
    se = np.full_like(p0, 1e-4)
    est = crossing_threshold(
        p0, rate_lo, se, rate_hi, se, rng=np.random.default_rng(1), n_boot=500
    )
    assert math.isnan(est.p_th)


def test_ci_widens_with_noise() -> None:
    p0 = np.array([0.002, 0.003, 0.004, 0.005])
    p_th_true = 0.0035
    rate_lo = 0.01 + 1.0 * (p0 - p_th_true)
    rate_hi = 0.01 + 3.0 * (p0 - p_th_true)
    rng = np.random.default_rng(2)
    tight = crossing_threshold(p0, rate_lo, np.full_like(p0, 1e-5), rate_hi,
                               np.full_like(p0, 1e-5), rng=rng, n_boot=3000)
    loose = crossing_threshold(p0, rate_lo, np.full_like(p0, 1e-3), rate_hi,
                               np.full_like(p0, 1e-3), rng=rng, n_boot=3000)
    assert (loose.ci_hi - loose.ci_lo) > (tight.ci_hi - tight.ci_lo)
