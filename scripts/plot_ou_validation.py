"""Generate four OU-process validation plots and save to docs/figures/.

Usage
-----
    python scripts/plot_ou_validation.py

Outputs
-------
docs/figures/ou_stationary_distribution.png
docs/figures/ou_autocorrelation.png
docs/figures/ou_psd.png
docs/figures/ou_markovian_limit.png
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from nonmarkov_qec.noise.ornstein_uhlenbeck import OUProcess

FIGURES_DIR = pathlib.Path(__file__).parent.parent / "docs" / "figures"
DPI = 150


def _ensure_figures_dir() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Plot 1: stationary distribution
# ---------------------------------------------------------------------------

def plot_stationary_distribution() -> None:
    rng = np.random.default_rng(42)
    tau_c = 1.0
    sigma = 0.5
    n_trajectories = 10_000
    n_steps = 2_000
    dt = tau_c / 10.0

    ou = OUProcess(tau_c=tau_c, sigma=sigma)
    trajs = ou.sample(n_steps=n_steps, dt=dt, n_trajectories=n_trajectories, rng=rng)

    half = (n_steps + 1) // 2
    samples_flat = trajs[:, half:].ravel()

    # Subsample to ~100,000 values
    subsample_size = min(100_000, samples_flat.size)
    rng_sub = np.random.default_rng(42)
    idx = rng_sub.choice(samples_flat.size, size=subsample_size, replace=False)
    samples = samples_flat[idx]

    x_range = np.linspace(-4 * sigma, 4 * sigma, 500)
    analytic = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-(x_range**2) / (2.0 * sigma**2))

    fig, ax = plt.subplots()
    ax.hist(samples, bins=60, density=True, alpha=0.6, color="steelblue", label="Empirical")
    ax.plot(x_range, analytic, color="firebrick", linewidth=2, label=r"$\mathcal{N}(0,\,\sigma^2)$")
    ax.set_xlabel("$X$")
    ax.set_ylabel("Density")
    ax.set_title("Stationary Distribution")
    ax.legend()

    out = FIGURES_DIR / "ou_stationary_distribution.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Plot 2: autocorrelation
# ---------------------------------------------------------------------------

def plot_autocorrelation() -> None:
    rng = np.random.default_rng(123)
    tau_c = 1.0
    sigma = 0.5
    n_trajectories = 10_000
    n_steps = 2_000
    dt = tau_c / 10.0

    ou = OUProcess(tau_c=tau_c, sigma=sigma)
    trajs = ou.sample(n_steps=n_steps, dt=dt, n_trajectories=n_trajectories, rng=rng)

    half = (n_steps + 1) // 2
    x = trajs[:, half:]  # shape (n_trajectories, T)
    T = x.shape[1]

    max_lag_steps = int(3.0 * tau_c / dt)
    empirical = np.empty(max_lag_steps + 1)
    for k in range(max_lag_steps + 1):
        empirical[k] = float(np.mean(x[:, : T - k] * x[:, k:]))

    lags_time = np.arange(max_lag_steps + 1) * dt
    analytic = sigma**2 * np.exp(-lags_time / tau_c)

    fig, ax = plt.subplots()
    ax.plot(lags_time, analytic, color="firebrick", linewidth=2,
            label=r"$\sigma^2 \exp(-|\tau|/\tau_c)$")
    ax.scatter(lags_time, empirical, s=30, color="steelblue", zorder=3, label="Empirical")
    ax.set_xlabel(r"Lag $\tau$")
    ax.set_ylabel(r"$C(\tau)$")
    ax.set_title("Autocorrelation")
    ax.legend()

    out = FIGURES_DIR / "ou_autocorrelation.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Plot 3: power spectral density
# ---------------------------------------------------------------------------

def plot_psd() -> None:
    rng = np.random.default_rng(456)
    tau_c = 1.0
    sigma = 0.5
    n_trajectories = 2_000
    n_steps = 2_000
    dt = tau_c / 10.0
    fs = 1.0 / dt

    ou = OUProcess(tau_c=tau_c, sigma=sigma)
    trajs = ou.sample(n_steps=n_steps, dt=dt, n_trajectories=n_trajectories, rng=rng)

    nperseg = n_steps // 8  # 250
    freqs, psd_per_traj = welch(trajs, fs=fs, nperseg=nperseg, axis=-1)
    psd_avg = psd_per_traj.mean(axis=0)

    a = np.exp(-dt / tau_c)
    b2 = sigma**2 * (1.0 - a**2)
    ref_psd = 2.0 * b2 * dt / (1.0 - 2.0 * a * np.cos(2.0 * np.pi * freqs * dt) + a**2)

    f_low = 1.0 / (10.0 * tau_c)
    f_nyquist = fs / 2.0
    plot_mask = (freqs > 0.0) & (freqs <= f_nyquist)

    fig, ax = plt.subplots()
    ax.loglog(freqs[plot_mask], ref_psd[plot_mask], color="firebrick", linewidth=2,
              label="AR(1) discrete-time PSD")
    ax.scatter(freqs[plot_mask], psd_avg[plot_mask], s=15, color="steelblue",
               zorder=3, label="Empirical (Welch)")
    ax.axvline(f_low, color="gray", linestyle="--", linewidth=1, label=f"$f_{{low}}$ = {f_low:.2f}")
    ax.set_xlabel("Frequency [1/time]")
    ax.set_ylabel("PSD")
    ax.set_title("Power Spectral Density")
    ax.legend()

    out = FIGURES_DIR / "ou_psd.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Plot 4: Markovian limit
# ---------------------------------------------------------------------------

def plot_markovian_limit() -> None:
    D = 0.1
    n_trajectories = 5_000
    t_window = 5.0
    tau_c_values = [1.0, 0.1, 0.01]
    seeds = [789, 790, 791]
    colors = ["steelblue", "seagreen", "darkorange"]

    fig, ax = plt.subplots()

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=0.1)

    # Draw the 2Dt asymptote once
    t_asym = np.linspace(0.1, t_window, 300)
    ax.plot(t_asym, 2.0 * D * t_asym, color="black", linestyle="--",
            linewidth=1.5, label="$2Dt$ asymptote")

    for tau_c, seed, color in zip(tau_c_values, seeds, colors):
        sigma = float(np.sqrt(D / tau_c))
        dt = tau_c / 10.0
        n_window = round(t_window / dt)

        rng = np.random.default_rng(seed)
        ou = OUProcess(tau_c=tau_c, sigma=sigma)
        trajs = ou.sample(n_steps=n_window, dt=dt, n_trajectories=n_trajectories, rng=rng)

        # Cumulative integral S_i(t_k) = sum_{j=0}^{k-1} X_i[j] * dt
        cumsum = np.cumsum(trajs[:, :n_window], axis=1) * dt  # shape (n_traj, n_window)
        var_empirical = np.var(cumsum, axis=0, ddof=1)  # shape (n_window,)

        # Time axis: t_k = k*dt for k = 1 .. n_window
        t_grid = np.arange(1, n_window + 1) * dt

        # Analytic variance of S(t)
        var_analytic = 2.0 * D * t_grid - 2.0 * D * tau_c * (1.0 - np.exp(-t_grid / tau_c))

        # Plot every ~200th point as markers to keep the plot readable
        stride = max(1, n_window // 200)
        ax.scatter(t_grid[::stride], var_empirical[::stride], s=10, color=color, alpha=0.7)
        ax.plot(t_grid, var_analytic, color=color, linewidth=1.8,
                label=rf"$\tau_c={tau_c}$")

    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\mathrm{var}(S(t))$")
    ax.set_title("Markovian Limit")
    ax.legend()

    out = FIGURES_DIR / "ou_markovian_limit.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _ensure_figures_dir()
    plot_stationary_distribution()
    plot_autocorrelation()
    plot_psd()
    plot_markovian_limit()
    print("All four plots saved.")


if __name__ == "__main__":
    main()
