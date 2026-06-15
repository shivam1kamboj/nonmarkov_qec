"""Coarse threshold-bracketing scan with live progress + per-point timing.

Run from repo root with the venv active:
    python scripts/coarse_scan.py
Writes docs/figures/coarse_scan.csv incrementally and coarse_scan.png at the end.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nonmarkov_qec.benchmarks.sweep import SweepPoint, run_point
from nonmarkov_qec.noise.sum_of_ou import SumOfOUProcess
from nonmarkov_qec.noise.white import WhiteNoiseProcess

SIGMA_TOTAL = 1.0
M = 0.5
TAU_MAX_CYCLES = 20.0
TAU_MIN_CYCLES = 0.5
N_COMPONENTS = 6

P0_GRID = np.array([0.003, 0.005, 0.008, 0.013, 0.02, 0.03, 0.05])
DISTANCES = [3, 5]
SHOTS = 200
N_TRAJ_OU = 40
N_TRAJ_WHITE = 20
BASE_SEED = 12345

F_MIN = 1.0 / (2.0 * np.pi * TAU_MAX_CYCLES)
F_MAX = 1.0 / (2.0 * np.pi * TAU_MIN_CYCLES)


def main() -> None:
    ou = SumOfOUProcess.from_frequency_band(
        f_min=F_MIN, f_max=F_MAX, n_components=N_COMPONENTS, sigma_total=SIGMA_TOTAL
    )
    white = WhiteNoiseProcess(sigma_total=SIGMA_TOTAL)
    arms = [("markovian", white, N_TRAJ_WHITE), ("sum_of_ou", ou, N_TRAJ_OU)]

    out = Path("docs/figures")
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "coarse_scan.csv"

    n_total = len(arms) * len(DISTANCES) * len(P0_GRID)
    done = 0
    t_start = time.time()
    rows: list[SweepPoint] = []

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "distance", "p_0", "shots", "n_traj", "rate", "stderr", "seed"])
        f.flush()

        for model, proc, n_traj in arms:
            # mirror run_sweep's per-cell seed scheme (distance outer, p_0 inner)
            n_cells = len(DISTANCES) * len(P0_GRID)
            seeds = np.random.SeedSequence(BASE_SEED).generate_state(n_cells)
            idx = 0
            for d in DISTANCES:
                for p_0 in P0_GRID:
                    t0 = time.time()
                    r = run_point(
                        proc, model=model, distance=d, p_0=float(p_0),
                        m=M, sigma=SIGMA_TOTAL, p_meas=None,
                        shots=SHOTS, n_traj=n_traj, seed=int(seeds[idx]),
                    )
                    dt = time.time() - t0
                    done += 1
                    idx += 1
                    rows.append(r)
                    w.writerow([r.model, r.distance, r.p_0, r.shots, r.n_traj,
                                r.rate, r.stderr, r.seed])
                    f.flush()
                    eta = (time.time() - t_start) / done * (n_total - done)
                    print(
                        f"[{done:2d}/{n_total}] {model:10s} d={d} "
                        f"p0={p_0:6.4f}  rate={r.rate:6.4f} +/- {r.stderr:6.4f}"
                        f"   ({dt:5.1f}s, eta {eta:5.0f}s)",
                        flush=True,
                    )

    print(f"\nTotal {time.time() - t_start:.0f}s. Wrote {csv_path}", flush=True)
    _plot(rows, out / "coarse_scan.png")
    print(f"Wrote {out / 'coarse_scan.png'}", flush=True)


def _plot(rows: list[SweepPoint], path: Path) -> None:
    palette = {3: "steelblue", 5: "darkorange"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, model, title in zip(
        axes, ("markovian", "sum_of_ou"),
        ("Markovian (white)", r"Non-Markovian ($1/f$)"),
    ):
        for d in DISTANCES:
            pts = sorted((r for r in rows if r.model == model and r.distance == d),
                         key=lambda r: r.p_0)
            ax.errorbar([r.p_0 for r in pts], [r.rate for r in pts],
                        yerr=[r.stderr for r in pts], marker="o", capsize=3,
                        color=palette[d], label=rf"$d={d}$")
        ax.set_xscale("log")
        ax.set_xlabel(r"$p_0$")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(r"$p_L$")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
