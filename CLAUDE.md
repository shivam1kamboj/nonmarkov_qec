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

## Status

Updated end of week 4. OU sampler shipped at `v0.1.0-ou-sampler` (commit 7210b92).
Design note for sum-of-OU committed at 0c9f40e. Tests passing, CI green.

## Current focus: implement SumOfOUProcess

The single-OU sampler is shipped as v0.1.0-ou-sampler. The next module is the multi-component sampler that approximates 1/f noise. The full design is in `docs/sum_of_ou.md` — read that before 
starting.

**Step 1 — implement the core constructor and sample().** Create `src/nonmarkov_qec/noise/sum_of_ou.py` with class `SumOfOUProcess`. Constructor signature: `SumOfOUProcess(tau_min, tau_max, 
n_components, sigma_total)`. Validate all inputs (all positive, tau_min < tau_max, n_components >= 1). Compute and store log-spaced tau_j and per-component sigma_j = sigma_total / 
sqrt(n_components). The `sample()` method has the same signature and shape contract as `OUProcess.sample()`. Internally: sample each component independently using the exact-update recursion, sum 
across components. Reuse OUProcess internally if clean; otherwise inline the recursion. Stop after Step 1 and let me review before Step 2.

**Step 2 — implement the from_frequency_band classmethod.** Convert (f_min, f_max) to (tau_min, tau_max) via the design note's inversion. Thin wrapper, no new logic.

**Step 3 — write the four validation tests.** Three carried over from single-OU (stationary distribution, autocorrelation, PSD) with summed analytic references. One new test for the 1/f spectral 
slope.

**Step 4 — validation plots.** Mirror the single-OU plots for the sum-of-OU process, save to `docs/figures/`.

**Step 5 — tag v0.2.0-sum-of-ou.**

## Parked

Do not work on these now. They are recorded so context isn't lost.

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
