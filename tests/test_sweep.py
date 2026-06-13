"""Smoke tests for the two-layer Monte Carlo sweep harness.

These are fast sanity checks, not statistical validation: tiny n_traj/shots,
d=3 only. They assert the harness runs end-to-end, returns well-formed results,
and is reproducible under a fixed seed.
"""

from __future__ import annotations

import numpy as np

from nonmarkov_qec.benchmarks.sweep import SweepPoint, run_point, run_sweep
from nonmarkov_qec.noise.white import WhiteNoiseProcess

# Shared tiny-budget injection params. p_0 deliberately small (sub-threshold).
_KW = dict(m=0.3, sigma=1.0, p_meas=1e-3, shots=64, n_traj=8)


def test_run_point_returns_valid_sweeppoint() -> None:
    """run_point produces a well-formed SweepPoint with a rate in [0, 1]."""
    proc = WhiteNoiseProcess(sigma_total=1.0)
    pt = run_point(
        proc, model="markovian", distance=3, p_0=0.01, seed=1234, **_KW
    )
    assert isinstance(pt, SweepPoint)
    assert pt.model == "markovian"
    assert pt.distance == 3
    assert pt.p_0 == 0.01
    assert pt.shots == 64
    assert pt.n_traj == 8
    assert 0.0 <= pt.rate <= 1.0
    assert pt.stderr >= 0.0


def test_run_point_reproducible_under_seed() -> None:
    """Identical seed -> identical rate and stderr (full determinism)."""
    proc = WhiteNoiseProcess(sigma_total=1.0)
    a = run_point(
        proc, model="markovian", distance=3, p_0=0.01, seed=7, **_KW
    )
    b = run_point(
        proc, model="markovian", distance=3, p_0=0.01, seed=7, **_KW
    )
    assert a.rate == b.rate
    assert a.stderr == b.stderr


def test_run_point_seeds_differ() -> None:
    """Different seeds generally give different trajectories (not frozen)."""
    proc = WhiteNoiseProcess(sigma_total=1.0)
    a = run_point(
        proc, model="markovian", distance=3, p_0=0.05, seed=1, **_KW
    )
    b = run_point(
        proc, model="markovian", distance=3, p_0=0.05, seed=2, **_KW
    )
    # Not a hard guarantee, but with shots*n_traj samples a collision is
    # vanishingly unlikely; this catches an accidentally-frozen RNG.
    assert a.rate != b.rate


def test_run_sweep_table_shape_and_seed_independence() -> None:
    """run_sweep emits one row per (distance, p_0) cell, all well-formed."""
    proc = WhiteNoiseProcess(sigma_total=1.0)
    distances = [3]
    p_0_grid = np.array([0.01, 0.05, 0.10])
    rows = run_sweep(
        proc,
        model="markovian",
        distances=distances,
        p_0_grid=p_0_grid,
        m=0.3,
        sigma=1.0,
        p_meas=1e-3,
        shots=64,
        n_traj=8,
        base_seed=2024,
    )
    assert len(rows) == len(distances) * len(p_0_grid)
    assert {r.p_0 for r in rows} == {0.01, 0.05, 0.10}
    assert all(0.0 <= r.rate <= 1.0 for r in rows)
    # Distinct per-point seeds (derived from base_seed).
    seeds = [r.seed for r in rows]
    assert len(set(seeds)) == len(seeds)


def test_run_sweep_reproducible() -> None:
    """Same base_seed -> identical table."""
    proc = WhiteNoiseProcess(sigma_total=1.0)
    kw = dict(
        model="markovian",
        distances=[3],
        p_0_grid=np.array([0.02, 0.08]),
        m=0.3,
        sigma=1.0,
        p_meas=1e-3,
        shots=64,
        n_traj=8,
        base_seed=99,
    )
    r1 = run_sweep(proc, **kw)
    r2 = run_sweep(proc, **kw)
    assert [p.rate for p in r1] == [p.rate for p in r2]
