# CLAUDE.md

This file orients Claude Code at the start of each session. Read it first.

## Project: nonmarkov-qec

An open-source Python library that simulates quantum error correction 
(QEC) codes under realistic, non-Markovian noise models, and benchmarks 
logical error rates against standard Markovian baselines.

This is a polished engineering artifact, not novel research. The goal is a 
clean, well-documented benchmarking tool that quantifies the gap between 
Markovian assumptions in QEC threshold theorems and the non-Markovian 
noise of real superconducting hardware.

Target: 6–10 weeks of part-time work. Ship on GitHub with README, 
examples, tests, CI, and a short technical write-up.

## Scope

**In scope.** Stochastic non-Markovian noise generator (Ornstein-Uhlenbeck 
and sum-of-OU for 1/f); small QEC codes (3-qubit bit-flip, 3-qubit 
phase-flip, Shor 9-qubit, distance-3 surface code patch); noise injection 
layer translating SDE trajectories into gate-level error channels; Monte 
Carlo benchmarking harness; minimum-weight perfect matching decoder for 
the surface code (via PyMatching) and lookup-table decoders for small 
codes; threshold plots comparing Markovian vs. non-Markovian.

**Out of scope.** Novel research claims. Hardware experiments. Qudit 
codes. Fault-tolerant gate construction (memory experiments only: encode 
→ noise → decode). Any financial / Heston framing.

## Tech stack

Python 3.11+. Stim for stabilizer simulation. PyMatching for surface code 
decoding. NumPy / SciPy for SDE integration. Matplotlib for plots. pytest 
for testing. ruff + mypy strict for lint and types. GitHub Actions for CI 
on 3.11 and 3.12. MIT license.

## Conventions

- Type hints everywhere. mypy is strict; do not weaken it.
- Docstrings on all public classes and functions. Cite sources for 
formulas (e.g. Gillespie 1996 for the OU exact-update recursion).
- RNG: use `numpy.random.Generator`, never the legacy `np.random.*` 
module-level functions. Generators should be passed in or constructed from 
a seed parameter — never implicit global state.
- Shape contracts in docstrings for all array-returning functions. 
Document axes explicitly.
- One commit per logical chunk. Don't bundle unrelated changes.
- Tests live in `tests/`, mirror the `src/nonmarkov_qec/` layout.
- Public API lives under `src/nonmarkov_qec/`. The four subpackages are 
`noise`, `codes`, `decoders`, `benchmarks`.

## Status (end of week 2)

Scaffolding and design notes are ahead of implementation.

**Done.** Package layout. `pyproject.toml` with deps and dev tooling 
pinned. CI green (lint + mypy + pytest on 3.11 / 3.12). MIT license, 
README, .gitignore. `docs/noise_model.md` is a real draft and picks the 
linear `p_k = clip(p_0 + α·X_k, 0, 1)` map for v1.

**Stubbed.** `OUProcess` class in 
`src/nonmarkov_qec/noise/ornstein_uhlenbeck.py`. `__init__` validates 
`tau_c > 0` and `sigma > 0`. The exact-update formula and API are settled 
in the docstring. `sample()` raises `NotImplementedError`.

**Tests skipped.** `tests/test_ornstein_uhlenbeck.py` has four tests 
skipped with `@pytest.mark.skip`: stationary mean/variance, 
autocorrelation, Welch PSD vs. Lorentzian, Markovian limit. Bodies raise 
`NotImplementedError`. CI is green only because everything meaningful is 
skipped.

**Not started.** Sum-of-OU class. Validation plots in `docs/figures/`. All 
QEC code implementations. Decoders. Benchmarking harness.

**Trivial cleanup outstanding.** `README.md` line 22 has a placeholder 
`YOUR_USERNAME` in the GitHub URL. `pyproject.toml` line 43 already has 
the real one.

## Current focus: finish the OU sampler

Everything downstream sits on top of `OUProcess.sample()` working and 
being statistically validated. Do this before anything else.

**Step 1 — implement `sample()`.** Use the exact-update recursion (already 
in the docstring): `X_{k+1} = X_k · exp(-Δt/τ_c) + σ · sqrt(1 - 
exp(-2·Δt/τ_c)) · Z_k` with `Z_k ~ N(0, 1)`. When `x0=None`, draw the 
initial state from the stationary distribution `N(0, σ²)`. Vectorize the 
per-step noise draw across trajectories. Output shape: `(n_trajectories, 
n_steps + 1)`. RNG is a `numpy.random.Generator` parameter; if none is 
passed, construct one from an optional `seed` parameter. Stop here and let 
the user review before continuing.

**Step 2 — unskip the four statistical tests.** Tolerances:

- *Stationary mean.* With `n_trajectories=10_000`, `n_steps=2_000`, 
`Δt=τ_c/10`, sample mean over the second half of trajectories should 
satisfy `|mean| < 3·σ / sqrt(n_trajectories)`.
- *Stationary variance.* Same setup. Relative error in sample variance < 
2%.
- *Autocorrelation.* Empirical autocorrelation should match 
`σ²·exp(-|τ|/τ_c)` with max relative error < 5% over lags in `[0, 3·τ_c]`. 
(Not 1% — empirical autocorr at large lag is noisy.)
- *Welch PSD.* `scipy.signal.welch` with `nperseg=n_steps/8`. Compare 
against the Lorentzian `2·σ²·τ_c / (1 + (2π·f·τ_c)²)` on log-log axes. 
Tolerance: < 10% relative error in the band `[1/(10·τ_c), 1/(2·Δt)]`.
- *Markovian limit.* Hold `D = σ²·τ_c` fixed and shrink `τ_c`. Check that 
the integrated autocorrelation `∫C(τ) dτ` converges to `2D` as `τ_c` 
shrinks. (Equivalent: `∫X dt` over a fixed window converges to Brownian 
with variance `2D·t`.)

Tests must be deterministic — pass an explicit seed to every `OUProcess` 
instance.

**Step 3 — generate the four validation plots.** Save to `docs/figures/` 
as PNG. Empirical vs. analytic overlays for: stationary distribution 
histogram, autocorrelation curve, PSD on log-log, and the Markovian-limit 
collapse. Reference these from `docs/noise_generator.md`.

**Step 4 — clean up.** Fix the `YOUR_USERNAME` placeholder in `README.md` 
line 22.

**Step 5 — commit.** One commit for the implementation, one for the tests, 
one for the plots and docs. Tag `v0.1.0-ou-sampler` after all three land.

## Parked

Do not work on these now. They are recorded so context isn't lost.

- **Sum-of-OU class.** Build only after single-OU is validated. Decisions 
still open: log-spacing scheme for component time constants, weights for 
1/f over a target band. Standard construction is superposition of 
Lorentzians; see Kogan, *Electronic Noise and Fluctuations in Solids*.
- **Whether `α` is user-facing or derived from a single knob.** Decide at 
week 7 when wiring noise into the QEC simulator.
- **Whether multi-qubit gates share a noise trajectory or each qubit gets 
an independent one.** Same — week 7.
- **Headline experiment for the README's first plot.** Decide before week 
7. Likely: surface code threshold under depolarizing vs. 1/f-like 
sum-of-OU noise at matched total noise power.

## Working style

- Stop at the end of each numbered step in "Current focus" and wait for 
review before proceeding. Don't chain all five steps in one session.
- If a test fails, do not edit the test to make it pass without flagging 
it. Flag the discrepancy, propose what's wrong, and let the user decide.
- If something in this file is out of date with the actual repo state, say 
so before starting work — don't silently work around it.
- When uncertain about a design choice, ask. Parked questions above are 
parked deliberately.
