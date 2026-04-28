"""Statistical validation tests for the Ornstein-Uhlenbeck sampler.

These tests check the sampled trajectories against analytic expectations:
1. Stationary distribution is N(0, sigma^2)
2. Empirical autocorrelation matches sigma^2 * exp(-|tau|/tau_c)
3. Welch PSD matches the Lorentzian S(w) = 2 sigma^2 tau_c / (1 + (w tau_c)^2)
4. Markovian (white-noise) limit as tau_c -> 0 at fixed sigma^2 * tau_c

All tests use fixed seeds and tolerances tight enough that real bugs fail.
"""

import pytest


@pytest.mark.skip(reason="Implementation pending: week 1, day 5-7")
def test_stationary_distribution_mean_and_variance() -> None:
    """After burn-in, samples should be ~ N(0, sigma^2)."""
    raise NotImplementedError


@pytest.mark.skip(reason="Implementation pending: week 1, day 5-7")
def test_autocorrelation_matches_exponential() -> None:
    """Empirical C(tau) should match sigma^2 * exp(-|tau|/tau_c) within ~1%."""
    raise NotImplementedError


@pytest.mark.skip(reason="Implementation pending: week 1, day 5-7")
@pytest.mark.statistical
def test_power_spectrum_is_lorentzian() -> None:
    """Welch PSD should match Lorentzian form across resolved frequencies."""
    raise NotImplementedError


@pytest.mark.skip(reason="Implementation pending: week 1, day 5-7")
def test_markovian_limit_recovers_white_noise() -> None:
    """As tau_c -> 0 with sigma^2 * tau_c fixed, the process should approach
    white noise with the corresponding intensity."""
    raise NotImplementedError
