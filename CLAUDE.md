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

Updated end of week 5. Noise generation layer complete:
- `v0.1.0-ou-sampler` — single OU sampler, validated, with plots.
- `v0.2.0-sum-of-ou` — sum-of-OU for 1/f noise, validated, with plots (including spectrum-tiling figure).

All 10 tests passing, ruff + mypy clean, CI green. Next phase: the noise injection layer (bridge from noise trajectories to per-gate Pauli error rates).

## Current focus: noise injection layer (design first)

The noise generators are done. The next module is the bridge that turns a noise trajectory X(t) into per-gate Pauli error probabilities that Stim can consume. This is the first QEC-facing component.

Before implementing, the open design questions in "Parked" must be resolved — specifically whether alpha is user-facing and whether multi-qubit gates share a trajectory. So the next step is a design 
note, not code.

**Step 1 — write docs/noise_injection.md.** Specify: the mapping p_k = clip(p_0 + alpha * X_k, 0, 1) (already chosen in docs/noise_model.md), how alpha is parameterized, how a trajectory maps onto a 
Stim circuit's per-instruction error rates, and how Markovian vs. non-Markovian baselines are constructed at matched mean error rate. Resolve the two parked design questions in this note. Do not 
write code yet — review the design first.

(Steps 2+ — implementation, tests, integration — to be defined after the design note is reviewed.)

## Parked

Do not work on these now. They are recorded so context isn't lost.

- **Headline experiment for the README's first plot.** Decide before the benchmarking phase. Likely: surface code threshold under depolarizing vs. 1/f-like sum-of-OU noise at matched total noise 
power.

## Working style

- Stop at the end of each numbered step in "Current focus" and wait for 
review before proceeding. Don't chain all five steps in one session.
- If a test fails, do not edit the test to make it pass without flagging 
it. Flag the discrepancy, propose what's wrong, and let the user decide.
- If something in this file is out of date with the actual repo state, say 
so before starting work — don't silently work around it.
- When uncertain about a design choice, ask. Parked questions above are 
parked deliberately.
