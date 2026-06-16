"""Threshold vs correlation time: p_th^{OU}(tau_c) against the matched white baseline.

Each tau_c point is a single Ornstein-Uhlenbeck process (one Lorentzian, so an
unambiguous correlation time), matched in marginal mean/variance to a white
baseline; only the temporal autocorrelation differs.  For each tau_c the
d=3 / d=5 logical-error curves are crossed to locate the threshold, with a
parametric-bootstrap 95% CI.  The white arm is tau_c-independent (run once).

Run from repo root with the venv active:
    python scripts/tau_c_sweep.py
Writes docs/figures/tau_c_sweep_raw.csv, tau_c_sweep.csv, tau_c_sweep.png.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nonmarkov_qec.benchmarks.sweep import SweepPoint, run_sweep
from nonmarkov_qec.benchmarks.threshold import crossing_threshold
from nonmarkov_qec.noise.ornstein_uhlenbeck import OUProcess
from nonmarkov_qec.noise.white import WhiteNoiseProcess

SIGMA_TOTAL = 1.0
M = 0.5
P0_GRID = np.array([0.0020, 0.0026, 0.0032, 0.0038, 0.0044, 0.0052, 0.0062])
DISTANCES = [3, 5]
SHOTS = 4000
N_TRAJ_OU = 150
N_TRAJ_WHITE = 60
TAU_C_GRID = np.logspace(np.log10(0.2), np.log10(200.0), 8)
BASE_SEED = 20260615
N_BOOT = 4000


def _curves(rows: list[SweepPoint], model: str):
    """Return (p0, r3, se3, r5, se5) sorted by p0 for one model."""
    def col(d: int):
        pts = sorted(
            (r for r in rows if r.model == model and r.distance == d),
            key=lambda r: r.p_0,
        )
        return (
            np.array([r.p_0 for r in pts]),
            np.array([r.rate for r in pts]),
            np.array([r.stderr for r in pts]),
        )
    p0, r3, se3 = col(3)
    _, r5, se5 = col(5)
    return p0, r3, se3, r5, se5


def main() -> None:
    out = Path("docs/figures")
    out.mkdir(parents=True, exist_ok=True)
    boot_rng = np.random.default_rng(BASE_SEED)
    t0 = time.time()
    raw: list[SweepPoint] = []

    # --- White baseline (tau_c-independent): run once. -------------------
    white = WhiteNoiseProcess(sigma_total=SIGMA_TOTAL)
    wrows = run_sweep(
        white, model="markovian", distances=DISTANCES, p_0_grid=P0_GRID,
        m=M, sigma=SIGMA_TOTAL, p_meas=None, shots=SHOTS,
        n_traj=N_TRAJ_WHITE, base_seed=BASE_SEED,
    )
    raw += wrows
    w_est = crossing_threshold(*_curves(wrows, "markovian"), rng=boot_rng, n_boot=N_BOOT)
    w_pth, w_lo, w_hi = w_est.p_th, w_est.ci_lo, w_est.ci_hi
    print(f"[white] p_th = {w_pth*1e3:.4f}e-3  CI[{w_lo*1e3:.4f},{w_hi*1e3:.4f}]e-3"
          f"  ({time.time()-t0:.0f}s)", flush=True)

    # --- OU arm: one threshold per tau_c. --------------------------------
    summary: list[tuple[float, float, float, float]] = []
    for j, tau_c in enumerate(TAU_C_GRID):
        ou = OUProcess(tau_c=float(tau_c), sigma=SIGMA_TOTAL)
        orows = run_sweep(
            ou, model="sum_of_ou", distances=DISTANCES, p_0_grid=P0_GRID,
            m=M, sigma=SIGMA_TOTAL, p_meas=None, shots=SHOTS,
            n_traj=N_TRAJ_OU, base_seed=BASE_SEED + 1000 * (j + 1),
        )
        for r in orows:
            raw.append(r)
        est = crossing_threshold(
            *_curves(orows, "sum_of_ou"), rng=boot_rng, n_boot=N_BOOT
        )
        pth, lo, hi = est.p_th, est.ci_lo, est.ci_hi
        summary.append((float(tau_c), pth, lo, hi))
        print(f"[tau_c={tau_c:7.2f}] p_th = {pth*1e3:.4f}e-3  "
              f"CI[{lo*1e3:.4f},{hi*1e3:.4f}]e-3  ({time.time()-t0:.0f}s)", flush=True)

    # --- Persist. --------------------------------------------------------
    with (out / "tau_c_sweep_raw.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "tau_c", "distance", "p_0", "rate", "stderr"])
        for r in wrows:
            w.writerow([r.model, "inf", r.distance, r.p_0, r.rate, r.stderr])
        per = len(DISTANCES) * len(P0_GRID)
        ou_rows = [r for r in raw if r.model == "sum_of_ou"]
        for idx, r in enumerate(ou_rows):
            tau_c = TAU_C_GRID[idx // per]
            w.writerow([r.model, f"{tau_c:.6g}", r.distance, r.p_0, r.rate, r.stderr])

    with (out / "tau_c_sweep.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau_c", "p_th", "ci_lo", "ci_hi"])
        w.writerow(["white", w_pth, w_lo, w_hi])
        for tau_c, pth, lo, hi in summary:
            w.writerow([f"{tau_c:.6g}", pth, lo, hi])

    _plot(summary, (w_pth, w_lo, w_hi), out / "tau_c_sweep.png")
    print(f"\nTotal {time.time()-t0:.0f}s. Wrote tau_c_sweep_raw.csv, "
          f"tau_c_sweep.csv, tau_c_sweep.png", flush=True)


def _plot(summary, white, path: Path) -> None:
    w_pth, w_lo, w_hi = white
    tau = np.array([s[0] for s in summary])
    pth = np.array([s[1] for s in summary]) * 1e3
    lo = np.array([s[2] for s in summary]) * 1e3
    hi = np.array([s[3] for s in summary]) * 1e3
    style = {"font.family": "serif", "mathtext.fontset": "cm",
             "axes.labelsize": 13, "axes.titlesize": 14}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.axhspan(w_lo * 1e3, w_hi * 1e3, color="steelblue", alpha=0.16,
                   lw=0, label="white baseline (95% CI)")
        ax.axhline(w_pth * 1e3, color="steelblue", lw=1.8,
                   label=r"white $p_{\mathrm{th}}$")
        ax.errorbar(tau, pth, yerr=[pth - lo, hi - pth], fmt="o",
                    color="darkorange", markersize=6.5, markeredgecolor="white",
                    markeredgewidth=0.8, elinewidth=1.3, capsize=3.5, capthick=1.3,
                    label=r"$1/f$ (sum-of-OU) $p_{\mathrm{th}}(\tau_c)$")
        ax.set_xscale("log")
        ax.set_xlabel(r"Noise correlation time $\tau_c$ (cycles)")
        ax.set_ylabel(r"Threshold $p_{\mathrm{th}}\,(\times 10^{-3})$")
        ax.set_title("Surface-code threshold vs. noise correlation time")
        ax.grid(True, which="both", alpha=0.25, lw=0.6)
        ax.legend(loc="lower right", framealpha=0.95, edgecolor="0.8")
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
