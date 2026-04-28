"""Ornstein-Uhlenbeck process sampler.

Implements an exact-update OU sampler:

    X_{t+dt} = X_t * exp(-dt/tau_c) + sigma * sqrt(1 - exp(-2*dt/tau_c)) * xi

where xi ~ N(0, 1). The exact update means dt does not need to be small
relative to tau_c for the sampled trajectory to have the correct stationary
distribution and autocorrelation.

References
----------
- Gillespie, "Exact numerical simulation of the Ornstein-Uhlenbeck process
  and its integral", Phys. Rev. E 54, 2084 (1996).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class OUProcess:
    """Stationary Ornstein-Uhlenbeck process X(t) with correlation time tau_c.

    The process satisfies the SDE:

        dX = -(X / tau_c) dt + sqrt(2 sigma^2 / tau_c) dW

    with stationary distribution N(0, sigma^2) and autocorrelation
    C(tau) = sigma^2 * exp(-|tau| / tau_c).

    Parameters
    ----------
    tau_c
        Correlation time of the process. Must be > 0. Units are arbitrary
        but must be consistent with `dt` passed to `sample`.
    sigma
        Stationary standard deviation. Must be > 0.

    Notes
    -----
    The sampler uses the exact update formula, so it is unconditionally
    stable in `dt`. A burn-in of ~5 * tau_c is recommended if the initial
    condition is not already drawn from the stationary distribution.
    """

    def __init__(self, tau_c: float, sigma: float) -> None:
        if tau_c <= 0:
            raise ValueError(f"tau_c must be > 0, got {tau_c}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self.tau_c = float(tau_c)
        self.sigma = float(sigma)

    def sample(
        self,
        n_steps: int,
        dt: float,
        n_trajectories: int = 1,
        rng: np.random.Generator | None = None,
        x0: float | NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Sample trajectories of the OU process.

        Parameters
        ----------
        n_steps
            Number of time steps to sample (excluding the initial point).
        dt
            Time step size.
        n_trajectories
            Number of independent trajectories.
        rng
            NumPy random generator. If None, a fresh default_rng() is used.
            For reproducible tests, always pass a seeded generator.
        x0
            Initial condition. If None, drawn from the stationary distribution.
            If scalar, all trajectories start at x0. If array, must have
            shape (n_trajectories,).

        Returns
        -------
        ndarray of shape (n_trajectories, n_steps + 1)
            Sampled trajectories, including the initial point at index 0.
        """
        raise NotImplementedError("Implement in week 1, day 3-4.")
