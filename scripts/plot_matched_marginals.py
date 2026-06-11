"""Generate the matched-marginal trajectory overlay and save to docs/figures/.

Two stacked panels share both axes (cycle index k, injected Z-error
probability p_q(k)). Top: white (Markovian) arm. Bottom: sum-of-OU (1/f) arm.
Both arms are sampled at identical marginals -- same mean p_0 and same spread
m*p_0 -- and pushed through the SAME injection normalization
(alpha = m*p_0/sigma), so the only visible difference is temporal texture:
delta-correlated hash (white) versus slow drift / bursts (1/f). This is the
visual form of the matched-instantaneous-power comparison in
docs/benchmarking.md, section 1.

Usage
-----
    python scripts/plot_matched_marginals.py

Outputs
-------
docs/figures/matched_marginals_trajectories.png
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import stim
from numpy.typing import NDArray

from nonmarkov_qec.codes.surface_code import surface_code
from nonmarkov_qec.noise.injection import inject_dephasing_noise
from nonmarkov_qec.noise.sum_of_ou import SumOfOUProcess
from nonmarkov_qec.noise.white import WhiteNoiseProcess

FIGURES_DIR = pathlib.Path(__file__).parent.parent / "docs" / "figures"
DPI = 150

# Physics / injection parameters (clip-free regime: matched moments hold).
P_0 = 0.01
M = 0.3
SIGMA_TOTAL = 1.0
P_MEAS = 0.0

# Geometry: enough cycles to see the OU correlation structure.
DISTANCE = 3
ROUNDS = 60

# Frequency band for the 1/f arm. tau range sits inside the cycle window so the
# slow drift is visible (corner periods ~ a few to a few tens of cycles).
F_MIN = 1.0 / 40.0
F_MAX = 1.0 / 2.0
N_COMPONENTS = 8

# Which data qubit's trajectory to display (a single row, for clarity).
SHOW_QUBIT = 0
SEED_WHITE = 10
SEED_SOU = 20


def _ensure_figures_dir() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _injected_probs_for_qubit(
    circuit: stim.Circuit, qubit: int, n_cycles: int
) -> NDArray[np.float64]:
    """Recover p_q(k) for one data qubit by reading Z_ERROR args in cycle order.

    The injection layer emits one Z_ERROR(p) per data-gate target, in circuit
    order, advancing the cycle counter at each TICK. We replay that traversal:
    track the cycle index via TICKs and record the probability each time this
    qubit is the target of a Z_ERROR.
    """
    probs: dict[int, float] = {}
    k = 0
    for item in circuit.flattened():
        if not isinstance(item, stim.CircuitInstruction):
            continue
        if item.name == "TICK":
            k += 1
            continue
        if item.name == "Z_ERROR":
            args = item.gate_args_copy()
            for t in item.targets_copy():
                if t.is_qubit_target and t.value == qubit:
                    probs[k] = float(args[0])
    # Return a dense array over the cycles where this qubit was hit, in order.
    ks = sorted(probs)
    return np.array([probs[k] for k in ks], dtype=np.float64)


def plot_matched_marginals() -> None:
    code = surface_code(distance=DISTANCE, rounds=ROUNDS)
    n_q, n_c = code.n_qubits, code.n_cycles

    white = WhiteNoiseProcess(sigma_total=SIGMA_TOTAL)
    sou = SumOfOUProcess.from_frequency_band(
        f_min=F_MIN, f_max=F_MAX, n_components=N_COMPONENTS, sigma_total=SIGMA_TOTAL
    )

    def realized(proc: WhiteNoiseProcess | SumOfOUProcess, seed: int) -> NDArray[np.float64]:
        raw = proc.sample(
            n_steps=n_c, dt=1.0, n_trajectories=n_q, rng=np.random.default_rng(seed)
        )
        traj = raw[:, 1:]  # strip initial condition -> (n_q, n_c)
        noisy = inject_dephasing_noise(
            code.circuit, traj, p_0=P_0, m=M, sigma=SIGMA_TOTAL, p_meas=P_MEAS
        )
        return _injected_probs_for_qubit(noisy, SHOW_QUBIT, n_c)

    p_white = realized(white, SEED_WHITE)
    p_sou = realized(sou, SEED_SOU)

    k_white = np.arange(p_white.size)
    k_sou = np.arange(p_sou.size)

    # Shared y-limits centered on p_0 so both panels read the same spread.
    spread = M * P_0
    y_lo = max(0.0, P_0 - 5.0 * spread)
    y_hi = P_0 + 5.0 * spread

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, sharey=True, figsize=(8, 5)
    )

    ax_top.plot(k_white, p_white, color="steelblue", linewidth=0.9)
    ax_top.axhline(P_0, color="firebrick", linestyle="--", linewidth=1.2,
                   label=rf"$p_0={P_0}$")
    ax_top.set_ylabel(r"$p_q(k)$")
    ax_top.set_title("White (Markovian) arm")
    ax_top.legend(loc="upper right", fontsize=8)

    ax_bot.plot(k_sou, p_sou, color="steelblue", linewidth=0.9)
    ax_bot.axhline(P_0, color="firebrick", linestyle="--", linewidth=1.2)
    ax_bot.set_ylabel(r"$p_q(k)$")
    ax_bot.set_xlabel("Gate cycle $k$")
    ax_bot.set_title(r"Sum-of-OU ($1/f$) arm")

    ax_top.set_ylim(y_lo, y_hi)

    fig.suptitle(
        rf"Matched marginals (mean $p_0={P_0}$, std $m\,p_0={spread:.3g}$), "
        r"differing only in autocorrelation",
        fontsize=10,
    )

    out = FIGURES_DIR / "matched_marginals_trajectories.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

# Correlation times (in gate cycles) to visualize. Each is a distinct physical
# regime: tau_c is the memory length of the qubit-frequency noise, a measurable
# device property. The 1/f band is centered on tau_c spanning one decade.
TAU_C_CYCLES = (5.0, 50.0)
ROUNDS_BY_TAU = {5.0: 80, 50.0: 400}  # enough cycles for several excursions


def _band_from_tau_c(tau_c: float) -> tuple[float, float]:
    """Map a target correlation time (cycles) to a 1/f band [f_min, f_max].

    The band spans one decade centered on tau_c: tau in [tau_c/sqrt(10),
    tau_c*sqrt(10)], converted to frequency via f = 1/(2*pi*tau). f_max comes
    from the shortest tau, f_min from the longest.
    """
    tau_lo = tau_c / np.sqrt(10.0)
    tau_hi = tau_c * np.sqrt(10.0)
    f_max = 1.0 / (2.0 * np.pi * tau_lo)
    f_min = 1.0 / (2.0 * np.pi * tau_hi)
    return f_min, f_max


def plot_matched_marginals_overlay(tau_c: float, rounds: int) -> None:
    """Single-panel overlay of both arms at a given correlation time tau_c.

    tau_c (gate cycles) sets the 1/f band; larger tau_c = longer-memory noise,
    so the sum-of-OU arm holds excursions above/below p_0 for more cycles.
    """
    f_min, f_max = _band_from_tau_c(tau_c)
    code = surface_code(distance=DISTANCE, rounds=rounds)
    n_q, n_c = code.n_qubits, code.n_cycles

    white = WhiteNoiseProcess(sigma_total=SIGMA_TOTAL)
    sou = SumOfOUProcess.from_frequency_band(
        f_min=f_min, f_max=f_max, n_components=N_COMPONENTS, sigma_total=SIGMA_TOTAL
    )

    def realized(proc: WhiteNoiseProcess | SumOfOUProcess, seed: int) -> NDArray[np.float64]:
        raw = proc.sample(
            n_steps=n_c, dt=1.0, n_trajectories=n_q, rng=np.random.default_rng(seed)
        )
        traj = raw[:, 1:]
        noisy = inject_dephasing_noise(
            code.circuit, traj, p_0=P_0, m=M, sigma=SIGMA_TOTAL, p_meas=P_MEAS
        )
        return _injected_probs_for_qubit(noisy, SHOW_QUBIT, n_c)

    p_white = realized(white, SEED_WHITE)
    p_sou = realized(sou, SEED_SOU)
    k_white = np.arange(p_white.size)
    k_sou = np.arange(p_sou.size)

    spread = M * P_0
    y_lo = max(0.0, P_0 - 5.0 * spread)
    y_hi = P_0 + 5.0 * spread

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(k_white, p_white, color="steelblue", linewidth=0.9, alpha=0.75,
            label="White (Markovian)")
    ax.plot(k_sou, p_sou, color="darkorange", linewidth=1.1, alpha=0.85,
            label=r"Sum-of-OU ($1/f$)")
    ax.axhline(P_0, color="firebrick", linestyle="--", linewidth=1.2,
               label=rf"$p_0={P_0}$")
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Gate cycle $k$")
    ax.set_ylabel(r"Injected Z-error probability $p_q(k)$")
    ax.set_title(
        rf"Matched marginals, $\tau_c={tau_c:g}$ cycles "
        rf"(mean $p_0={P_0}$, std $m\,p_0={spread:.3g}$)"
    )
    ax.legend(loc="upper right", fontsize=8)

    out = FIGURES_DIR / f"matched_marginals_overlay_tau{tau_c:g}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

def main() -> None:
    _ensure_figures_dir()
    plot_matched_marginals()  # the stacked figure (fast default band)
    for tau_c in TAU_C_CYCLES:
        plot_matched_marginals_overlay(tau_c, ROUNDS_BY_TAU[tau_c])
    print("Plots saved.")

if __name__ == "__main__":
    main()
