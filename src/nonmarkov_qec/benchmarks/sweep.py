"""Benchmarking harness: two-layer Monte Carlo sweep over the QEC pipeline.

Design note: docs/benchmarking.md

Produces logical-error-rate estimates p_L(p_0, d, model) via two nested Monte
Carlo layers (benchmarking.md section 2):

  Layer 2 (trajectories): draw N_traj independent noise realizations X^(i).
  Layer 1 (shots): per trajectory, sample `shots` records from the noisy circuit.

The reported rate is p_L_hat = mean_i(r_i), r_i = errors_i / shots, with
stderr = std_i(r_i) / sqrt(N_traj), which folds both layers automatically.

The MWPM decoder is built ONCE per (d, p_0) point from a constant-p_0 circuit
and reused across all trajectories (benchmarking.md section 2, "Decoder
calibration"): a real decoder is calibrated to the average noise model and
cannot adapt to an instantaneous correlated burst.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pymatching
import stim
from numpy.typing import NDArray

from nonmarkov_qec.codes.surface_code import surface_code
from nonmarkov_qec.decoders.matching import matching_from_circuit
from nonmarkov_qec.noise.injection import inject_dephasing_noise


class NoiseProcess(Protocol):
    """Structural type for a noise source usable by the harness.

    Both WhiteNoiseProcess and SumOfOUProcess satisfy this by exposing a
    sample() returning shape (n_trajectories, n_steps + 1).
    """

    def sample(
        self,
        n_steps: int,
        dt: float,
        n_trajectories: int = ...,
        rng: np.random.Generator | None = ...,
    ) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class SweepPoint:
    """One row of the sweep results table.

    Attributes
    ----------
    model
        Noise-model label ("markovian" or "sum_of_ou").
    distance
        Code distance d.
    p_0
        Baseline per-data-gate dephasing probability.
    shots
        Layer-1 samples per trajectory.
    n_traj
        Layer-2 independent trajectory draws.
    rate
        p_L_hat = mean over trajectories of (errors_i / shots).
    stderr
        Two-layer standard error: std_i(r_i) / sqrt(n_traj).
    seed
        Per-point base seed used to derive trajectory and shot RNG streams.
    """

    model: str
    distance: int
    p_0: float
    shots: int
    n_traj: int
    rate: float
    stderr: float
    seed: int


def _decode_errors(
    noisy_circuit: stim.Circuit,
    matching: pymatching.Matching,
    shots: int,
    rng: np.random.Generator,
) -> int:
    """Sample `shots` records from a noisy circuit and decode against a fixed
    matcher; return the count of logical-observable mispredictions.

    The matcher is supplied externally (built once per point) and is NOT
    rebuilt here -- this is the fixed-decoder contract of benchmarking.md
    section 2.
    """
    sampler = noisy_circuit.compile_detector_sampler(seed=int(rng.integers(2**63)))
    detectors, observables = sampler.sample(
        shots, separate_observables=True
    )
    predictions = matching.decode_batch(detectors)
    # Single observable (asserted at matcher build): compare column 0.
    mispredict = predictions[:, 0] != observables[:, 0]
    return int(np.count_nonzero(mispredict))


def run_point(
    process: NoiseProcess,
    model: str,
    distance: int,
    p_0: float,
    *,
    m: float,
    sigma: float,
    p_meas: float,
    rounds: int | None = None,
    shots: int,
    n_traj: int,
    seed: int,
) -> SweepPoint:
    """Evaluate one sweep point with the two-layer Monte Carlo of section 2.

    Parameters
    ----------
    process
        A noise process exposing
        ``.sample(n_steps, dt, n_trajectories, rng) -> (n_trajectories,
        n_steps + 1)``. Its rows are interpreted as data qubits, its columns
        (after dropping the index-0 stationary point) as cycles.
    model
        Label stored in the result row ("markovian" / "sum_of_ou").
    distance
        Code distance d (odd).
    p_0
        Baseline dephasing probability.
    m, sigma, p_meas
        Injection parameters forwarded to inject_dephasing_noise. `sigma` is the
        process stationary std used in alpha = m * p_0 / sigma; passing the
        process's nominal std keeps alpha identical across arms (matched
        marginals, section 1).
    rounds
        Number of QEC rounds; defaults to `distance` (rounds = d, section 2).
    shots
        Layer-1 samples per trajectory.
    n_traj
        Layer-2 trajectory draws.
    seed
        Per-point base seed; trajectory and shot streams are derived from it.

    Returns
    -------
    SweepPoint
    """
    if rounds is None:
        rounds = distance

    n_data = distance * distance

    # --- Bare circuit (once) ---------------------------------------------
    bare = surface_code(distance, rounds)

    # --- Fixed decoder: build ONCE on a constant-p_0 circuit -------------
    # A zero trajectory makes p_{q,k} = clip(p_0 + 0, 0, 1) = p_0 everywhere.
    const_traj = np.zeros((n_data, rounds), dtype=np.float64)
    const_circuit = inject_dephasing_noise(
        bare, const_traj, p_0=p_0, m=m, sigma=sigma, p_meas=p_meas
    )
    matching = matching_from_circuit(const_circuit)

    # --- Two-layer Monte Carlo -------------------------------------------
    # Distinct, deterministic RNG streams for trajectory draws vs. shot
    # sampling, both derived from the per-point seed.
    ss = np.random.SeedSequence(seed)
    traj_seed, shot_seed = ss.spawn(2)
    traj_rng = np.random.default_rng(traj_seed)
    shot_rng = np.random.default_rng(shot_seed)

    rates = np.empty(n_traj, dtype=np.float64)
    for i in range(n_traj):
        # process.sample returns (n_trajectories, n_steps + 1); take n_data
        # rows as qubits and drop the index-0 stationary point so the cycle
        # axis has exactly `rounds` columns.
        raw = process.sample(
            n_steps=rounds, dt=1.0, n_trajectories=n_data, rng=traj_rng
        )
        traj = raw[:, 1:]  # (n_data, rounds)
        noisy = inject_dephasing_noise(
            bare, traj, p_0=p_0, m=m, sigma=sigma, p_meas=p_meas
        )
        errors = _decode_errors(noisy, matching, shots, shot_rng)
        rates[i] = errors / shots

    rate = float(rates.mean())
    stderr = float(rates.std(ddof=1) / np.sqrt(n_traj)) if n_traj > 1 else 0.0

    return SweepPoint(
        model=model,
        distance=distance,
        p_0=p_0,
        shots=shots,
        n_traj=n_traj,
        rate=rate,
        stderr=stderr,
        seed=seed,
    )


def run_sweep(
    process: NoiseProcess,
    model: str,
    distances: list[int],
    p_0_grid: NDArray[np.float64],
    *,
    m: float,
    sigma: float,
    p_meas: float,
    shots: int,
    n_traj: int,
    base_seed: int,
) -> list[SweepPoint]:
    """Sweep run_point over (distance x p_0_grid) for one noise model.

    Per-point seeds are derived deterministically from base_seed via
    SeedSequence, so the whole sweep is reproducible and each point is
    independent.

    Returns a flat list of SweepPoint rows (model, d, p_0, ...), the tidy table
    consumed by the threshold/plotting code (section 6).
    """
    points: list[SweepPoint] = []
    # One child seed per (distance, p_0) cell, in row-major order.
    n_cells = len(distances) * len(p_0_grid)
    child_seeds = np.random.SeedSequence(base_seed).generate_state(n_cells)

    idx = 0
    for d in distances:
        for p_0 in p_0_grid:
            points.append(
                run_point(
                    process,
                    model=model,
                    distance=d,
                    p_0=float(p_0),
                    m=m,
                    sigma=sigma,
                    p_meas=p_meas,
                    shots=shots,
                    n_traj=n_traj,
                    seed=int(child_seeds[idx]),
                )
            )
            idx += 1
    return points
