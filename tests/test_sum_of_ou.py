"""Statistical validation tests for SumOfOUProcess.

Four tests mirror the single-OU validation suite in test_ornstein_uhlenbeck.py,
with analytic references updated for the multi-component construction:

1. Stationary distribution is N(0, sigma_total^2)
2. Empirical C(tau) matches sum_j sigma_j^2 * exp(-|tau|/tau_j)
3. Welch PSD matches the sum of discrete-time AR(1) spectra
4. Analytic summed spectrum has log-log slope ~ -1 over the 1/f band

Tests 1-3 use fixed seeds for reproducibility. Test 4 is purely analytic
(no random sampling).
"""

import numpy as np
import pytest
from scipy.signal import welch

from nonmarkov_qec.noise.sum_of_ou import SumOfOUProcess

# Shared parameters applied consistently across all tests
TAU_MIN = 0.1
TAU_MAX = 100.0
N_COMPONENTS = 8
SIGMA_TOTAL = 0.5
DT = 0.05


def test_stationary_distribution_mean_and_variance() -> None:
    """After burn-in, X_sum should be ~ N(0, sigma_total^2)."""
    rng = np.random.default_rng(42)
    n_trajectories = 10_000
    n_steps = 2_000

    proc = SumOfOUProcess(
        tau_min=TAU_MIN,
        tau_max=TAU_MAX,
        n_components=N_COMPONENTS,
        sigma_total=SIGMA_TOTAL,
    )
    trajs = proc.sample(n_steps=n_steps, dt=DT, n_trajectories=n_trajectories, rng=rng)

    # Discard first half as burn-in; shape is (n_trajectories, n_steps + 1)
    half = (n_steps + 1) // 2
    second_half = trajs[:, half:]

    sample_mean = float(second_half.mean())
    tol_mean = 3.0 * SIGMA_TOTAL / np.sqrt(n_trajectories)
    assert abs(sample_mean) < tol_mean, (
        f"|mean| = {abs(sample_mean):.6f} >= tolerance {tol_mean:.6f}"
    )

    sample_var = float(second_half.var())
    rel_var_err = abs(sample_var - SIGMA_TOTAL**2) / SIGMA_TOTAL**2
    assert rel_var_err < 0.02, (
        f"Relative variance error = {rel_var_err:.4f} >= 0.02"
    )


def test_autocorrelation_matches_sum_of_exponentials() -> None:
    """Empirical C(tau) should match sum_j sigma_j^2 * exp(-|tau|/tau_j) within 5%."""
    rng = np.random.default_rng(123)
    n_trajectories = 5_000
    # n_steps must exceed max_lag_steps = int(TAU_MAX / DT) = 2000.
    n_steps = 3_000

    proc = SumOfOUProcess(
        tau_min=TAU_MIN,
        tau_max=TAU_MAX,
        n_components=N_COMPONENTS,
        sigma_total=SIGMA_TOTAL,
    )
    trajs = proc.sample(n_steps=n_steps, dt=DT, n_trajectories=n_trajectories, rng=rng)

    # No burn-in: each component draws x0 from its own stationary distribution
    # (N(0, sigma_j^2)), so X_sum is stationary from t = 0.
    x = trajs  # shape (n_trajectories, n_steps + 1)
    T = x.shape[1]  # 3001

    # Two-part lag scheme: unit-spaced at short lags to cover fast component
    # decay (tau_min=0.1 decays in ~2 steps), then sparse for slow components.
    max_lag_steps = int(TAU_MAX / DT)  # 2000 steps = TAU_MAX time units
    short_lags = np.arange(0, 41)                              # steps 0..40
    long_lags = np.linspace(41, max_lag_steps, 60, dtype=int)  # ~60 evenly-spaced
    lag_indices = np.unique(np.concatenate([short_lags, long_lags]))

    empirical = np.empty(len(lag_indices))
    for i, k in enumerate(lag_indices):
        empirical[i] = float(np.mean(x[:, : T - k] * x[:, k:]))

    lags_time = lag_indices * DT
    # Analytic reference: sum over components of sigma_j^2 * exp(-|tau|/tau_j)
    analytic = np.sum(
        proc.sigmas[:, None] ** 2 * np.exp(-lags_time[None, :] / proc.taus[:, None]),
        axis=0,
    )  # shape (n_lags,)

    abs_err = np.abs(empirical - analytic)
    print(
        f"  Max |empirical - analytic| / sigma_total^2 = "
        f"{abs_err.max() / SIGMA_TOTAL**2:.4f} at tau = {lags_time[abs_err.argmax()]:.2f}"
    )
    assert abs_err.max() < 0.01 * SIGMA_TOTAL**2, (
        f"Max |empirical - analytic| / sigma_total^2 = "
        f"{abs_err.max() / SIGMA_TOTAL**2:.4f} at tau = {lags_time[abs_err.argmax()]:.2f} "
        f"(tolerance 0.01)"
    )


@pytest.mark.statistical
def test_power_spectrum_matches_sum_of_discrete_ar1() -> None:
    """Welch PSD should match the sum of discrete-time AR(1) spectra within 10%."""
    rng = np.random.default_rng(456)
    n_trajectories = 2_000
    n_steps = 2_000
    fs = 1.0 / DT

    proc = SumOfOUProcess(
        tau_min=TAU_MIN,
        tau_max=TAU_MAX,
        n_components=N_COMPONENTS,
        sigma_total=SIGMA_TOTAL,
    )
    trajs = proc.sample(n_steps=n_steps, dt=DT, n_trajectories=n_trajectories, rng=rng)

    nperseg = n_steps // 8  # = 250

    # Vectorised Welch across all trajectories simultaneously (axis=-1)
    freqs, psd_per_traj = welch(trajs, fs=fs, nperseg=nperseg, axis=-1)
    psd_avg = psd_per_traj.mean(axis=0)

    # Reference: sum of discrete-time AR(1) one-sided spectra.
    # Two-sided spectrum for component j: b_j^2*dt / (1 - 2*a_j*cos(...) + a_j^2).
    # Multiply by 2 for the one-sided spectrum that welch returns for 0 < f < Nyquist.
    a = np.exp(-DT / proc.taus)          # (n_components,)
    b2 = proc.sigmas**2 * (1.0 - a**2)  # (n_components,)

    ref_psd = np.sum(
        2.0 * b2[:, None] * DT
        / (
            1.0
            - 2.0 * a[:, None] * np.cos(2.0 * np.pi * freqs[None, :] * DT)
            + a[:, None] ** 2
        ),
        axis=0,
    )  # (n_freqs,)

    # Exclude DC, Nyquist, and the lowest few Welch bins (highest variance).
    # fs/nperseg is the Welch frequency resolution; the factor-of-4 cutoff
    # drops bins where Welch variance is largest without removing the 1/f rolloff.
    f_nyquist = 0.5 / DT
    mask = (freqs >= 4.0 * (fs / nperseg)) & (freqs < f_nyquist)

    rel_err = np.abs(psd_avg[mask] - ref_psd[mask]) / ref_psd[mask]
    print(
        f"  Max relative PSD error = {rel_err.max():.4f} at "
        f"f = {freqs[mask][rel_err.argmax()]:.3f} Hz"
    )
    assert np.all(rel_err < 0.10), (
        f"Max relative PSD error = {rel_err.max():.4f} at "
        f"f = {freqs[mask][rel_err.argmax()]:.3f} Hz (tolerance 0.10)"
    )


def test_spectrum_has_one_over_f_slope() -> None:
    """Analytic summed discrete spectrum should have log-log slope ~ -1 over the 1/f band."""
    proc = SumOfOUProcess(
        tau_min=TAU_MIN,
        tau_max=TAU_MAX,
        n_components=N_COMPONENTS,
        sigma_total=SIGMA_TOTAL,
    )

    # 1/f band corners set by the shortest and longest correlation times
    f_low = 1.0 / (2.0 * np.pi * TAU_MAX)   # ~ 0.00159 Hz
    f_high = 1.0 / (2.0 * np.pi * TAU_MIN)  # ~ 1.592 Hz

    freqs = np.logspace(np.log10(f_low), np.log10(f_high), 500)

    a = np.exp(-DT / proc.taus)          # (n_components,)
    b2 = proc.sigmas**2 * (1.0 - a**2)  # (n_components,)

    s_analytic = np.sum(
        2.0 * b2[:, None] * DT
        / (
            1.0
            - 2.0 * a[:, None] * np.cos(2.0 * np.pi * freqs[None, :] * DT)
            + a[:, None] ** 2
        ),
        axis=0,
    )  # (n_freqs,)

    slope, _ = np.polyfit(np.log10(freqs), np.log10(s_analytic), deg=1)
    print(f"  Fitted log-log slope over 1/f band: {slope:.4f}")

    assert abs(slope - (-1.0)) < 0.15, (
        f"Log-log slope = {slope:.4f}, expected ~ -1.0 (tolerance 0.15)"
    )
