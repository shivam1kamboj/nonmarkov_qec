 CLAUDE.md

This file orients Claude Code at the start of each session. Read it first.

## Project: nonmarkov-qec

An open-source Python library that simulates quantum error correction
(QEC) codes under realistic, non-Markovian noise models, and benchmarks
logical error rates against standard Markovian baselines.

This is a polished engineering artifact, not novel research. The goal is a
clean, well-documented benchmarking tool that quantifies the gap between
Markovian assumptions in QEC threshold theorems and the non-Markovian
noise of real superconducting hardware.

Ship on GitHub with README, examples, tests, CI, and a short technical
write-up.

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
for testing. ruff + mypy strict for lint and types. GitHub Actions for CI.
MIT license.

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

Full simulate-and-decode pipeline complete and validated end to end:
noise generator -> injection -> surface code -> MWPM decoder -> logical
error rate. 49 tests pass, ruff + mypy clean, CI green. HEAD = 16cf477.

Noise generation (committed, with plots):
- `v0.1.0-ou-sampler` — single OU sampler (Gillespie exact-update).
- `v0.2.0-sum-of-ou` — sum-of-OU for 1/f (from_frequency_band, 1/f slope
  -0.9976).

Noise injection (committed):
- `noise/injection.py` — trajectory X_q(k) -> clip(p_0 + alpha*X) Z_ERROR
  per data gate, alpha = m*p_0/sigma internal. MR/MRX/MRY/MRZ = M(p_meas)
  + ideal reset. Data-gate dephasing only in v1; measurement constant p_meas.

Small codes (committed, decoder-free validation):
- bit-flip (Stim generated, Z-blind plumbing null), phase-flip (hand-built
  X-stabilizers, first real Z correction), Shor 9-qubit (hand-built X-memory,
  exact fired-detector-set contrast test).

Surface code (committed):
- `codes/surface_code.py` — rotated d=3, 17 qubits, X-basis memory. CX
  schedule adopted from Stim's rotated_memory_x, remapped to 0-16; verified
  shortest_graphlike_error == 3. Design note docs/surface_code.md.

MWPM decoder (committed):
- `decoders/matching.py` — matching_from_circuit (DEM -> pymatching,
  decompose_errors=True, single-observable assert); estimate_logical_error_rate
  -> DecodeResult(shots, errors, .rate, .stderr). Operates on already-noisy
  circuits; knows nothing about noise generation.

Design notes in docs/: noise_model.md, noise_generator.md, sum_of_ou.md,
noise_injection.md, small_codes.md, surface_code.md.

## Current focus: benchmarking harness + headline experiment (design-note-first)

Produces the headline number: surface-code threshold under 1/f (sum-of-OU)
vs matched Markovian noise. Design note docs/benchmarking.md is being written
and reviewed BEFORE any code.

Build order (after note approved): parameterized surface_code(d) with
shortest_graphlike_error == d gate at d=3 AND d=5 + d=3 regression ->
white-noise process (shared process interface) + matched-marginal unit test
-> run_sweep harness -> coarse scan -> fine scan + threshold extraction + plot.

Matched comparison (the validity core): both arms share marginal mean p_0 and
marginal variance (m*p_0)^2 of the per-cycle error probability via the
alpha = m*p_0/sigma normalization; ONLY the temporal autocorrelation differs
(white vs 1/f). We fix sigma, never D, so there is no tau_c -> 0 blow-up.

## Parked

Do not work on these now. Recorded so context isn't lost.

- d=7 patch + full finite-size scaling collapse (v1 uses two-distance
  crossing-bracket; collapse needs >=3 distances).
- v1.1 physics refinements (deferred in design notes): quadratic/pure-dephasing
  injection map p = (1-exp(-chi))/2; reset noise; spatial correlation across
  qubits. None block the headline result.
- Sum-of-OU pedagogical notebook (separate Summer_2026 teaching track).
- Hardware calibration extension (Phase 2, after v1 ships).

## Working style

- Stop at the end of each numbered step in "Current focus" and wait for 
review before proceeding. Don't chain all five steps in one session.
- If a test fails, do not edit the test to make it pass without flagging 
it. Flag the discrepancy, propose what's wrong, and let the user decide.
- If something in this file is out of date with the actual repo state, say 
so before starting work — don't silently work around it.
- When uncertain about a design choice, ask. Parked questions above are 
parked deliberately.
