"""Generate four sum-of-OU validation plots and save to docs/figures/.

Usage
-----
    python scripts/plot_sum_of_ou_validation.py

Outputs
-------
docs/figures/sum_of_ou_stationary_distribution.png
docs/figures/sum_of_ou_autocorrelation.png
docs/figures/sum_of_ou_psd.png
docs/figures/sum_of_ou_spectrum_tiling.png
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.signal import welch

from nonmarkov_qec.noise.sum_of_ou import SumOfOUProcess

FIGURES_DIR = pathlib.Path(__file__).parent.parent / "docs" / "figures"
DPI = 150

TAU_MIN = 0.1
TAU_MAX = 100.0
N_COMPONENTS = 8
SIGMA_TOTAL = 0.5
DT = 0.05


def _ensure_figures_dir() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Plot 1: stationary distribution
# ---------------------------------------------------------------------------

def plot_stationary_distribution() -> None:
    rng = np.random.default_rng(42)
    n_trajectories = 10_000
    n_steps = 2_000

    proc = SumOfOUProcess(
        tau_min=TAU_MIN, tau_max=TAU_MAX,
        n_components=N_COMPONENTS, sigma_total=SIGMA_TOTAL,
    )
    trajs = proc.sample(n_steps=n_steps, dt=DT, n_trajectories=n_trajectories, rng=rng)

    half = (n_steps + 1) // 2
    samples_flat = trajs[:, half:].ravel()

    subsample_size = min(100_000, samples_flat.size)
    rng_sub = np.random.default_rng(42)
    idx = rng_sub.choice(samples_flat.size, size=subsample_size, replace=False)
    samples = samples_flat[idx]

    x_range = np.linspace(-4 * SIGMA_TOTAL, 4 * SIGMA_TOTAL, 500)
    analytic = (
        1.0 / (SIGMA_TOTAL * np.sqrt(2.0 * np.pi))
        * np.exp(-(x_range**2) / (2.0 * SIGMA_TOTAL**2))
    )

    fig, ax = plt.subplots()
    ax.hist(samples, bins=60, density=True, alpha=0.6, color="steelblue", label="Empirical")
    ax.plot(x_range, analytic, color="firebrick", linewidth=2,
            label=r"$\mathcal{N}(0,\,\sigma_{\mathrm{total}}^2)$")
    ax.set_xlabel(r"$X_{\mathrm{sum}}$")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Stationary Distribution  "
        f"($\\sigma_{{\\mathrm{{total}}}}={SIGMA_TOTAL}$, $N={N_COMPONENTS}$)"
    )
    ax.legend()

    out = FIGURES_DIR / "sum_of_ou_stationary_distribution.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Plot 2: autocorrelation
# ---------------------------------------------------------------------------

def plot_autocorrelation() -> None:
    rng = np.random.default_rng(123)
    n_trajectories = 5_000
    n_steps = 3_000  # must exceed max_lag_steps = int(TAU_MAX / DT) = 2000

    proc = SumOfOUProcess(
        tau_min=TAU_MIN, tau_max=TAU_MAX,
        n_components=N_COMPONENTS, sigma_total=SIGMA_TOTAL,
    )
    trajs = proc.sample(n_steps=n_steps, dt=DT, n_trajectories=n_trajectories, rng=rng)

    # No burn-in: each component draws x0 from its own stationary distribution,
    # so X_sum is stationary from t=0.
    x = trajs
    T = x.shape[1]

    # Two-part lag scheme matching the validation test
    max_lag_steps = int(TAU_MAX / DT)  # 2000
    short_lags = np.arange(0, 41)
    long_lags = np.linspace(41, max_lag_steps, 60, dtype=int)
    lag_indices = np.unique(np.concatenate([short_lags, long_lags]))

    empirical = np.empty(len(lag_indices))
    for i, k in enumerate(lag_indices):
        empirical[i] = float(np.mean(x[:, : T - k] * x[:, k:]))

    lags_time = lag_indices * DT
    analytic = np.sum(
        proc.sigmas[:, None] ** 2 * np.exp(-lags_time[None, :] / proc.taus[:, None]),
        axis=0,
    )

    fig, ax = plt.subplots()
    ax.plot(lags_time, analytic, color="firebrick", linewidth=2,
            label=r"$\sum_j \sigma_j^2\,\exp(-|\tau|/\tau_j)$")
    ax.scatter(lags_time, empirical, s=20, color="steelblue", zorder=3, label="Empirical")
    ax.set_xlabel(r"Lag $\tau$")
    ax.set_ylabel(r"$C(\tau)$")
    ax.set_title(f"Autocorrelation  ($N={N_COMPONENTS}$ components)")
    ax.legend()

    out = FIGURES_DIR / "sum_of_ou_autocorrelation.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Plot 3: power spectral density
# ---------------------------------------------------------------------------

def plot_psd() -> None:
    rng = np.random.default_rng(456)
    n_trajectories = 2_000
    n_steps = 2_000
    fs = 1.0 / DT

    proc = SumOfOUProcess(
        tau_min=TAU_MIN, tau_max=TAU_MAX,
        n_components=N_COMPONENTS, sigma_total=SIGMA_TOTAL,
    )
    trajs = proc.sample(n_steps=n_steps, dt=DT, n_trajectories=n_trajectories, rng=rng)

    nperseg = n_steps // 8  # 250
    freqs, psd_per_traj = welch(trajs, fs=fs, nperseg=nperseg, axis=-1)
    psd_avg = psd_per_traj.mean(axis=0)

    a = np.exp(-DT / proc.taus)           # (n_components,)
    b2 = proc.sigmas**2 * (1.0 - a**2)   # (n_components,)
    ref_psd = np.sum(
        2.0 * b2[:, None] * DT
        / (
            1.0
            - 2.0 * a[:, None] * np.cos(2.0 * np.pi * freqs[None, :] * DT)
            + a[:, None] ** 2
        ),
        axis=0,
    )

    f_nyquist = 0.5 / DT
    plot_mask = (freqs > 0.0) & (freqs <= f_nyquist)

    f_low_corner = 1.0 / (2.0 * np.pi * TAU_MAX)
    f_high_corner = 1.0 / (2.0 * np.pi * TAU_MIN)

    fig, ax = plt.subplots()
    ax.loglog(freqs[plot_mask], ref_psd[plot_mask], color="firebrick", linewidth=2,
              label="Summed AR(1) discrete-time PSD")
    ax.scatter(freqs[plot_mask], psd_avg[plot_mask], s=15, color="steelblue",
               zorder=3, label="Empirical (Welch)")
    ax.axvline(f_low_corner, color="gray", linestyle="--", linewidth=1,
               label=f"$f_{{\\mathrm{{low}}}}={f_low_corner:.4f}$")
    ax.axvline(f_high_corner, color="gray", linestyle=":", linewidth=1,
               label=f"$f_{{\\mathrm{{high}}}}={f_high_corner:.3f}$")
    ax.set_xlabel("Frequency [1/time]")
    ax.set_ylabel("PSD")
    ax.set_title(f"Power Spectral Density  ($N={N_COMPONENTS}$ components)")
    ax.legend(fontsize=8)

    out = FIGURES_DIR / "sum_of_ou_psd.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Plot 4: spectrum tiling (new)
# ---------------------------------------------------------------------------

def plot_spectrum_tiling() -> None:
    proc = SumOfOUProcess(
        tau_min=TAU_MIN, tau_max=TAU_MAX,
        n_components=N_COMPONENTS, sigma_total=SIGMA_TOTAL,
    )

    f_low_corner = 1.0 / (2.0 * np.pi * TAU_MAX)
    f_high_corner = 1.0 / (2.0 * np.pi * TAU_MIN)
    f_nyquist = 0.5 / DT

    freqs = np.logspace(
        np.log10(f_low_corner / 5.0), np.log10(f_nyquist * 0.99), 800
    )

    a = np.exp(-DT / proc.taus)           # (n_components,)
    b2 = proc.sigmas**2 * (1.0 - a**2)   # (n_components,)
    component_psds = (
        2.0 * b2[:, None] * DT
        / (
            1.0
            - 2.0 * a[:, None] * np.cos(2.0 * np.pi * freqs[None, :] * DT)
            + a[:, None] ** 2
        )
    )  # (n_components, n_freqs)
    sum_psd = component_psds.sum(axis=0)

    # 1/f guide anchored at the summed spectrum at the geometric centre of the band
    f_center = np.sqrt(f_low_corner * f_high_corner)
    s_center = sum_psd[np.argmin(np.abs(freqs - f_center))]
    band_mask = (freqs >= f_low_corner) & (freqs <= f_high_corner)
    guide_psd = s_center * (f_center / freqs[band_mask])

    fig, ax = plt.subplots()

    cmap_colors = plt.cm.viridis(np.linspace(0.15, 0.85, N_COMPONENTS))
    for comp_psd, color in zip(component_psds, cmap_colors):
        ax.loglog(freqs, comp_psd, color=color, linewidth=0.9, alpha=0.8)

    (sum_line,) = ax.loglog(freqs, sum_psd, color="black", linewidth=2.5)
    (guide_line,) = ax.loglog(
        freqs[band_mask], guide_psd, color="firebrick", linestyle="--", linewidth=1.5
    )
    ax.axvline(f_low_corner, color="gray", linestyle=":", linewidth=1)
    ax.axvline(f_high_corner, color="gray", linestyle=":", linewidth=1)

    proxy_comp = Line2D(
        [0], [0], color=cmap_colors[N_COMPONENTS // 2], linewidth=0.9, alpha=0.8
    )
    proxy_corner = Line2D([0], [0], color="gray", linestyle=":", linewidth=1)
    ax.legend(
        [proxy_comp, sum_line, guide_line, proxy_corner],
        [
            f"Individual components ($N={N_COMPONENTS}$, log-spaced $\\tau_j$)",
            "Sum (total PSD)",
            r"$1/f$ guide (slope $-1$)",
            f"Band corners ({f_low_corner:.4f}, {f_high_corner:.3f})",
        ],
        fontsize=8,
    )

    ax.set_xlabel("Frequency [1/time]")
    ax.set_ylabel("PSD")
    ax.set_title(
        f"Spectrum Tiling: {N_COMPONENTS} Lorentzians $\\rightarrow$ 1/f\n"
        f"$\\tau_{{\\mathrm{{min}}}}={TAU_MIN}$, "
        f"$\\tau_{{\\mathrm{{max}}}}={TAU_MAX}$, "
        f"$\\Delta t={DT}$"
    )

    out = FIGURES_DIR / "sum_of_ou_spectrum_tiling.png"
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
    plot_spectrum_tiling()
    print("All four plots saved.")


if __name__ == "__main__":
    main()
