# Noise generator: math, validation, and API

*Status: stub. To be filled in by end of Week 2 with validation plots and
final API documentation.*

## Outline

1. **The OU process.** SDE, stationary distribution, autocorrelation, PSD.
2. **Exact-update sampler.** Derivation of the update formula and why it
   beats Euler-Maruyama for our use case.
3. **Validation.** Stationary moments, autocorrelation curve, Welch PSD,
   Markovian limit. Plots of empirical vs. analytic for each.
4. **Sum-of-OU framework.** Approximating 1/f via log-spaced OU components.
   Comparison of empirical PSD slope against -1 over the target band.
5. **API reference.** `OUProcess` and `SumOfOUProcess` class signatures with
   example usage.
