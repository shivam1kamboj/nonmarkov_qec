"""Sum-of-OU process sampler for approximating 1/f noise.

Implements SumOfOUProcess, which sums N independent Ornstein-Uhlenbeck
processes with logarithmically-spaced correlation times to approximate
a 1/f power spectrum over a target frequency band.

Construction follows docs/sum_of_ou.md. The summed process is:

    X_sum(t) = sum_{j=1}^{N} X_j(t),    X_j ~ OU(tau_j, sigma_j)

with log-spaced tau_j and equal per-component variance (v1 weighting):

    sigma_j = sigma_total / sqrt(N)

References
----------
- docs/sum_of_ou.md — design note for this module.
- Kogan, "Electronic Noise and Fluctuations in Solids", Cambridge (1996) —
  standard treatment of 1/f from summed Lorentzians.
- Gillespie, Phys. Rev. E 54, 2084 (1996) — exact-update OU recursion.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nonmarkov_qec.noise.ornstein_uhlenbeck import OUProcess


class SumOfOUProcess:
    """Sum of N independent OU processes approximating a 1/f power spectrum.

    The summed process is:

        X_sum(t) = sum_{j=1}^{N} X_j(t),    X_j ~ OU(tau_j, sigma_j)

    Correlation times tau_j are log-spaced between tau_min and tau_max.
    Per-component standard deviation is sigma_j = sigma_total / sqrt(N)
    (v1 equal-variance weighting), so Var(X_sum) = sigma_total^2.

    The power spectrum approximates 1/f between the corner frequencies
    f_low = 1/(2*pi*tau_max) and f_high = 1/(2*pi*tau_min). See
    docs/sum_of_ou.md for the analytic spectrum and autocorrelation.

    Parameters
    ----------
    tau_min
        Shortest correlation time (sets the high-frequency edge of the
        1/f band). Must be > 0 and < tau_max.
    tau_max
        Longest correlation time (sets the low-frequency edge of the
        1/f band). Must be > tau_min.
    n_components
        Number of OU components N. Must be >= 1. A rule of thumb of
        >= 2-3 components per decade keeps the 1/f slope within ~a few
        percent across the band.
    sigma_total
        Total stationary standard deviation of the summed process.
        Must be > 0.

    Attributes
    ----------
    tau_min, tau_max, n_components, sigma_total
        As given.
    taus : ndarray of shape (n_components,)
        Log-spaced correlation times tau_j.
    sigmas : ndarray of shape (n_components,)
        Per-component standard deviations sigma_j = sigma_total / sqrt(N).
    """

    def __init__(
        self,
        tau_min: float,
        tau_max: float,
        n_components: int,
        sigma_total: float,
    ) -> None:
        if tau_min <= 0:
            raise ValueError(f"tau_min must be > 0, got {tau_min}")
        if tau_max <= 0:
            raise ValueError(f"tau_max must be > 0, got {tau_max}")
        if tau_min >= tau_max:
            raise ValueError(f"tau_min must be < tau_max, got {tau_min} >= {tau_max}")
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        if sigma_total <= 0:
            raise ValueError(f"sigma_total must be > 0, got {sigma_total}")

        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.n_components = int(n_components)
        self.sigma_total = float(sigma_total)

        self.taus: NDArray[np.float64] = np.logspace(
            np.log10(self.tau_min),
            np.log10(self.tau_max),
            num=self.n_components,
            dtype=np.float64,
        )
        sigma_j = self.sigma_total / np.sqrt(self.n_components)
        self.sigmas: NDArray[np.float64] = np.full(
            self.n_components, sigma_j, dtype=np.float64
        )

    def sample(
        self,
        n_steps: int,
        dt: float,
        n_trajectories: int = 1,
        rng: np.random.Generator | None = None,
        x0: float | NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Sample trajectories of the sum-of-OU process.

        Parameters
        ----------
        n_steps
            Number of time steps (excluding the initial point).
        dt
            Time step size. Same units as tau_min / tau_max.
        n_trajectories
            Number of independent trajectories.
        rng
            NumPy random generator. If None, a fresh default_rng() is used.
            Pass a seeded generator for reproducible results. A single rng
            is threaded through all components so the full sum is
            reproducible from one seed.
        x0
            Not supported in v1. Must be None. The summed process has no
            unique decomposition of a scalar initial value across N
            components with different relaxation rates; each component
            instead draws from its own stationary distribution N(0, sigma_j^2),
            giving X_sum(0) ~ N(0, sigma_total^2). Pass x0=None or omit it.
            A NotImplementedError is raised if x0 is not None.
            Kept in the signature for API consistency with OUProcess.sample().

        Returns
        -------
        ndarray of shape (n_trajectories, n_steps + 1)
            Sampled trajectories of X_sum, including the initial point at index 0.

        Notes
        -----
        Components are sampled in the order they appear in self.taus
        (ascending tau). This ordering determines each component's share of
        random draws from the shared rng; results are fully reproducible
        from one seed but will differ if n_components or tau_min/tau_max change.
        """
        if n_steps < 0:
            raise ValueError(f"n_steps must be >= 0, got {n_steps}")
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        if x0 is not None:
            raise NotImplementedError(
                "x0 is not supported in v1. The summed process has no unique "
                "decomposition of an initial value across components with "
                "different relaxation rates. Omit x0 or pass None."
            )

        if rng is None:
            rng = np.random.default_rng()

        out = np.zeros((n_trajectories, n_steps + 1), dtype=np.float64)

        for tau_j, sigma_j in zip(self.taus, self.sigmas, strict=True):
            component = OUProcess(tau_c=tau_j, sigma=sigma_j)
            out += component.sample(
                n_steps=n_steps,
                dt=dt,
                n_trajectories=n_trajectories,
                rng=rng,
                x0=None,
            )

        return out
