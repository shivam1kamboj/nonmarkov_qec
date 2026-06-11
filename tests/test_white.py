"""Tests for WhiteNoiseProcess and the matched-marginal property of the two arms.

The matched-marginal claim (docs/benchmarking.md, section 1): pushed through the
shared injection normalization alpha = m*p_0/sigma, both the white (Markovian)
and sum-of-OU (1/f) processes yield per-cycle Z-error probabilities with the
same marginal mean (p_0) and variance ((m*p_0)^2). Only the autocorrelation
differs. These tests verify that property through the real inject_dephasing_noise
path, in the clip-free regime where the linear-Gaussian moments hold exactly.
"""

from __future__ import annotations

import numpy as np
import stim

from nonmarkov_qec.codes.surface_code import surface_code
from nonmarkov_qec.noise.injection import inject_dephasing_noise
from nonmarkov_qec.noise.sum_of_ou import SumOfOUProcess
from nonmarkov_qec.noise.white import WhiteNoiseProcess


def test_white_shape_and_marginal() -> None:
    """sample() returns (n_trajectories, n_steps+1) with N(0, sigma^2) marginal."""
    rng = np.random.default_rng(0)
    w = WhiteNoiseProcess(sigma_total=2.0)
    x = w.sample(n_steps=4999, dt=1.0, n_trajectories=16, rng=rng)
    assert x.shape == (16, 5000)
    assert abs(float(x.mean())) < 0.05
    assert abs(float(x.std()) - 2.0) < 0.05


def test_white_is_uncorrelated() -> None:
    """Lag-1 autocorrelation is ~0 (delta-correlated, no memory)."""
    rng = np.random.default_rng(1)
    w = WhiteNoiseProcess(sigma_total=1.0)
    x = w.sample(n_steps=9999, dt=1.0, n_trajectories=8, rng=rng)
    xc = x - x.mean(axis=1, keepdims=True)
    lag1 = float((xc[:, :-1] * xc[:, 1:]).mean() / (xc * xc).mean())
    assert abs(lag1) < 0.05


def test_white_invalid_params() -> None:
    """sigma_total <= 0, negative n_steps, non-positive dt, and x0 all rejected."""
    import pytest

    with pytest.raises(ValueError):
        WhiteNoiseProcess(sigma_total=0.0)
    w = WhiteNoiseProcess(sigma_total=1.0)
    with pytest.raises(ValueError):
        w.sample(n_steps=-1, dt=1.0)
    with pytest.raises(ValueError):
        w.sample(n_steps=10, dt=0.0)
    with pytest.raises(NotImplementedError):
        w.sample(n_steps=10, dt=1.0, x0=0.0)


def _injected_z_error_probs(circuit: stim.Circuit) -> list[float]:
    """Read back every Z_ERROR probability from an injected circuit."""
    probs: list[float] = []
    for item in circuit.flattened():
        if isinstance(item, stim.CircuitInstruction) and item.name == "Z_ERROR":
            probs.extend(item.gate_args_copy())
    return probs


def test_matched_marginals_across_arms() -> None:
    """White and sum-of-OU yield matched per-cycle p marginals through injection.

    Clip-free regime (p_0=0.01, m=0.3): trajectories almost never push p outside
    [0,1], so the realized Z_ERROR probabilities have mean p_0 and std m*p_0 for
    BOTH processes. sigma passed to injection equals each process's sigma_total,
    making alpha = m*p_0/sigma_total identical across arms by construction.
    """
    p_0, m, sigma_total = 0.01, 0.3, 1.0
    code = surface_code(distance=3, rounds=3)
    n_q, n_c = code.n_qubits, code.n_cycles

    white = WhiteNoiseProcess(sigma_total=sigma_total)
    sou = SumOfOUProcess.from_frequency_band(
        f_min=1e-2, f_max=1e0, n_components=8, sigma_total=sigma_total
    )

    def realized_probs(proc: WhiteNoiseProcess | SumOfOUProcess, seed: int) -> np.ndarray:
        # Sample n_c + 1 columns, then strip index 0 (the initial condition),
        # leaving exactly n_c cycle columns as injection requires.
        raw = proc.sample(
            n_steps=n_c, dt=1.0, n_trajectories=n_q,
            rng=np.random.default_rng(seed),
        )
        traj = raw[:, 1:]  # strip initial condition -> shape (n_q, n_c)
        noisy = inject_dephasing_noise(
            code.circuit, traj, p_0=p_0, m=m, sigma=sigma_total, p_meas=0.0
        )
        return np.asarray(_injected_z_error_probs(noisy), dtype=np.float64)
    pw = realized_probs(white, seed=10)
    ps = realized_probs(sou, seed=20)

    # Both arms: mean ~ p_0, std ~ m*p_0 (the matched marginals).
    assert abs(pw.mean() - p_0) < 0.0015
    assert abs(ps.mean() - p_0) < 0.0015
    assert abs(pw.std() - m * p_0) < 0.0015
    assert abs(ps.std() - m * p_0) < 0.0015

    # And the two arms match each other.
    assert abs(pw.mean() - ps.mean()) < 0.0015
    assert abs(pw.std() - ps.std()) < 0.0015
