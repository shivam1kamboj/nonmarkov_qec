"""Statistical validation tests for the Ornstein-Uhlenbeck sampler.

These tests check the sampled trajectories against analytic expectations:
1. Stationary distribution is N(0, sigma^2)
2. Empirical autocorrelation matches sigma^2 * exp(-|tau|/tau_c)
3. Welch PSD matches the Lorentzian S(f) = 2 sigma^2 tau_c / (1 + (2 pi f tau_c)^2)
4. Markovian (white-noise) limit as tau_c -> 0 at fixed sigma^2 * tau_c

All tests use fixed seeds and tolerances tight enough that real bugs fail.
"""

import numpy as np
import pytest
from scipy.signal import welch

from nonmarkov_qec.noise.ornstein_uhlenbeck import OUProcess


def test_stationary_distribution_mean_and_variance() -> None:
    """After burn-in, samples should be ~ N(0, sigma^2)."""
    rng = np.random.default_rng(42)
    tau_c = 1.0
    sigma = 0.5
    n_trajectories = 10_000
    n_steps = 2_000
    dt = tau_c / 10.0

    ou = OUProcess(tau_c=tau_c, sigma=sigma)
    trajs = ou.sample(n_steps=n_steps, dt=dt, n_trajectories=n_trajectories, rng=rng)

    # Discard first half as burn-in; shape is (n_trajectories, n_steps+1)
    half = (n_steps + 1) // 2
    second_half = trajs[:, half:]

    sample_mean = float(second_half.mean())
    tol_mean = 3.0 * sigma / np.sqrt(n_trajectories)
    assert abs(sample_mean) < tol_mean, (
        f"|mean| = {abs(sample_mean):.6f} >= tolerance {tol_mean:.6f}"
    )

    sample_var = float(second_half.var())
    rel_var_err = abs(sample_var - sigma**2) / sigma**2
    assert rel_var_err < 0.02, (
        f"Relative variance error = {rel_var_err:.4f} >= 0.02"
    )


def test_autocorrelation_matches_exponential() -> None:
    """Empirical C(tau) should match sigma^2 * exp(-|tau|/tau_c) within 5%."""
    rng = np.random.default_rng(123)
    tau_c = 1.0
    sigma = 0.5
    n_trajectories = 10_000
    n_steps = 2_000
    dt = tau_c / 10.0

    ou = OUProcess(tau_c=tau_c, sigma=sigma)
    trajs = ou.sample(n_steps=n_steps, dt=dt, n_trajectories=n_trajectories, rng=rng)

    # Discard first half as burn-in; shape is (n_trajectories, n_steps+1)
    half = (n_steps + 1) // 2
    x = trajs[:, half:]  # shape (n_trajectories, T)
    T = x.shape[1]

    # Empirical autocorrelation at lags 0 .. 3*tau_c (= 30 steps)
    max_lag_steps = int(3.0 * tau_c / dt)
    empirical = np.empty(max_lag_steps + 1)
    for k in range(max_lag_steps + 1):
        empirical[k] = float(np.mean(x[:, : T - k] * x[:, k:]))

    lags_time = np.arange(max_lag_steps + 1) * dt
    analytic = sigma**2 * np.exp(-lags_time / tau_c)

    rel_err = np.abs(empirical - analytic) / analytic
    assert np.all(rel_err < 0.05), (
        f"Max relative autocorrelation error = {rel_err.max():.4f} at "
        f"tau = {lags_time[rel_err.argmax()]:.2f} (tolerance 0.05)"
    )


@pytest.mark.statistical
def test_power_spectrum_is_lorentzian() -> None:
    """Welch PSD should match the exact discrete-time AR(1) spectrum."""
    rng = np.random.default_rng(456)
    tau_c = 1.0
    sigma = 0.5
    n_trajectories = 2_000
    n_steps = 2_000
    dt = tau_c / 10.0
    fs = 1.0 / dt

    ou = OUProcess(tau_c=tau_c, sigma=sigma)
    trajs = ou.sample(n_steps=n_steps, dt=dt, n_trajectories=n_trajectories, rng=rng)

    nperseg = n_steps // 8  # = 250

    # Vectorised Welch across all trajectories simultaneously (axis=-1)
    freqs, psd_per_traj = welch(trajs, fs=fs, nperseg=nperseg, axis=-1)
    psd_avg = psd_per_traj.mean(axis=0)

    # b²·dt/(…) is the two-sided discrete AR(1) PSD; multiply by 2 for the
    # one-sided spectrum that welch returns for 0 < f < Nyquist.
    a = np.exp(-dt / tau_c)
    b2 = sigma**2 * (1.0 - a**2)
    ref_psd = 2.0 * b2 * dt / (1.0 - 2.0 * a * np.cos(2.0 * np.pi * freqs * dt) + a**2)

    # Exclude DC (freqs > 0) and Nyquist (freqs < f_high): welch does not double
    # the Nyquist bin, so it must be kept outside the masked comparison region.
    f_low = 1.0 / (10.0 * tau_c)
    f_high = 1.0 / (2.0 * dt)
    mask = (freqs >= f_low) & (freqs < f_high) & (freqs > 0.0)

    rel_err = np.abs(psd_avg[mask] - ref_psd[mask]) / ref_psd[mask]
    assert np.all(rel_err < 0.05), (
        f"Max relative PSD error = {rel_err.max():.4f} at "
        f"f = {freqs[mask][rel_err.argmax()]:.3f} Hz (tolerance 0.05)"
    )


def test_markovian_limit_recovers_white_noise() -> None:
    """var(int_0^t X ds) -> 2Dt as tau_c -> 0; analytic: 2Dt - 2D*tau_c*(1-exp(-t/tau_c))."""
    D = 0.1
    n_trajectories = 5_000
    t_window = 5.0
    tau_c_values = [1.0, 0.1, 0.01]
    seeds = [789, 790, 791]

    for tau_c, seed in zip(tau_c_values, seeds, strict=True):
        sigma = float(np.sqrt(D / tau_c))
        dt = tau_c / 10.0
        n_window = round(t_window / dt)  # steps covering exactly t_window

        rng = np.random.default_rng(seed)
        ou = OUProcess(tau_c=tau_c, sigma=sigma)
        # x0=None draws from stationary N(0, sigma^2); no burn-in needed
        trajs = ou.sample(n_steps=n_window, dt=dt, n_trajectories=n_trajectories, rng=rng)

        # S_i = int_0^{t_window} X_i(s) ds via left-endpoint Riemann sum
        S = trajs[:, :n_window].sum(axis=1) * dt  # shape (n_trajectories,)

        var_empirical = float(np.var(S, ddof=1))
        var_analytic = (
            2.0 * D * t_window
            - 2.0 * D * tau_c * (1.0 - np.exp(-t_window / tau_c))
        )

        rel_err = abs(var_empirical - var_analytic) / var_analytic
        assert rel_err < 0.05, (
            f"tau_c={tau_c}: var(S) = {var_empirical:.5f}, "
            f"analytic = {var_analytic:.5f}, "
            f"rel_err = {rel_err:.4f} (tolerance 0.05)"
        )
