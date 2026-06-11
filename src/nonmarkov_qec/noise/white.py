"""White-noise process: the Markovian baseline arm.

Provides WhiteNoiseProcess, a temporally uncorrelated (delta-correlated)
Gaussian process exposing the same sampling interface as SumOfOUProcess so
the benchmarking harness is polymorphic over the noise model. Each sample is
i.i.d. N(0, sigma_total^2) with no memory across time steps -- the white
limit against which the 1/f sum-of-OU arm is compared at matched marginals.

The matched-marginal comparison (docs/benchmarking.md, section 1) holds because
the injection layer normalizes by the trajectory standard deviation
(alpha = m * p_0 / sigma_X); both this process and SumOfOUProcess present the
same marginal std, so the per-cycle error probability has identical mean p_0
and variance (m * p_0)^2 in both arms. Only the autocorrelation differs:
delta-correlated here, a sum of decaying exponentials for sum-of-OU.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class WhiteNoiseProcess:
    """Temporally uncorrelated Gaussian process (the Markovian arm).

    Each entry is an independent draw from N(0, sigma_total^2); there is no
    correlation between time steps. The sampling interface mirrors
    SumOfOUProcess.sample so the two are interchangeable in the harness.

    Parameters
    ----------
    sigma_total
        Stationary standard deviation of the process. Must be > 0. Because the
        injection layer re-normalizes by the trajectory std, this value does
        not affect matched-marginal benchmarks; it exists for interface
        symmetry with SumOfOUProcess.

    Attributes
    ----------
    sigma_total
        As given.
    """

    def __init__(self, sigma_total: float) -> None:
        if sigma_total <= 0:
            raise ValueError(f"sigma_total must be > 0, got {sigma_total}")
        self.sigma_total = float(sigma_total)

    def sample(
        self,
        n_steps: int,
        dt: float,
        n_trajectories: int = 1,
        rng: np.random.Generator | None = None,
        x0: float | NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Sample trajectories of the white-noise process.

        Signature matches SumOfOUProcess.sample for drop-in interchangeability.

        Parameters
        ----------
        n_steps
            Number of time steps (excluding the initial point).
        dt
            Time step size. Accepted for interface symmetry; a white process is
            scale-free in time, so dt does not affect the (i.i.d.) statistics.
            Must be > 0.
        n_trajectories
            Number of independent trajectories.
        rng
            NumPy random generator. If None, a fresh default_rng() is used.
            Pass a seeded generator for reproducible results.
        x0
            Not supported (kept for API symmetry with SumOfOUProcess.sample).
            A NotImplementedError is raised if x0 is not None; index 0 is drawn
            from the stationary distribution N(0, sigma_total^2) like every
            other column.

        Returns
        -------
        ndarray of shape (n_trajectories, n_steps + 1)
            Independent N(0, sigma_total^2) samples, including the initial
            point at index 0. No temporal correlation.
        """
        if n_steps < 0:
            raise ValueError(f"n_steps must be >= 0, got {n_steps}")
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        if x0 is not None:
            raise NotImplementedError(
                "x0 is not supported. Omit x0 or pass None; index 0 is drawn "
                "from the stationary distribution like every other sample."
            )

        if rng is None:
            rng = np.random.default_rng()

        return rng.standard_normal(
            (n_trajectories, n_steps + 1)
        ) * self.sigma_total
