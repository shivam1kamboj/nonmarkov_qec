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

Updated end of week 6. Noise generation + injection layers complete:
- `v0.1.0-ou-sampler` — single OU sampler, validated, with plots.
- `v0.2.0-sum-of-ou` — sum-of-OU for 1/f noise, validated, with plots.
- Noise injection layer (`src/nonmarkov_qec/noise/injection.py`) — bridges OU
  trajectories to per-gate Stim Z-errors. Flattens REPEAT, single-pass rebuild,
  per-qubit `Z_ERROR(clip(p_0 + alpha*X_qk))` after each data gate,
  `M`/`MR(p_meas)` for measurements, idle moments advance the clock.
  Clip-fraction warning above 5%.

All tests passing, ruff + mypy clean, CI green. Next phase: small QEC code
implementations as Stim circuits.

## Current focus: small QEC codes (design-note-first)

Design note `docs/small_codes.md` is written. Three small codes as Stim
circuits, memory experiments, exercised by the injection layer:
- bit-flip (Stim's generated `repetition_code:memory`) — blind to our Z
  dephasing by construction; serves as the injection-layer vocabulary/plumbing
  test (expected zero detection events).
- phase-flip (hand-built, X-stabilizers) — first code that actually corrects
  our Z noise.
- Shor 9-qubit (hand-built) — both error types.

Resolved in the note: `MR` is treated as `M(p_meas)` + ideal reset (added to
the injection gate sets). Hand-built circuits are restricted to the verified
Stim vocabulary (`R, TICK, H, CX, MR, M, DETECTOR, OBSERVABLE_INCLUDE`).

Implementation order: (1) MR fix + regression test; (2) bit-flip constructor +
metadata + decoder-free validations; (3) phase-flip; (4) Shor.

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
