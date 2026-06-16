# CLAUDE.md

This file orients Claude Code at the start of each session. Read it first.

## Project: nonmarkov-qec

An open-source Python library that simulates quantum error correction (QEC)
under realistic non-Markovian noise (1/f-like, with temporal memory), and
benchmarks logical error rates against a statistically matched Markovian
baseline. Built on Stim (stabilizer simulation) + PyMatching (MWPM decoder).

A polished engineering / portfolio artifact, not a discovery. The v1 scientific
result — matched-marginal surface-code threshold, 1/f vs. white, under a
standard MWPM decoder — is NOT novel; it reproduces the non-detrimental regime
of Kam, Gicev, Modi, Southwell, Usman, "Detrimental non-Markovian errors for
surface code memory," Quantum Sci. Technol. 10, 035060 (2025), arXiv:2410.23779.
Cite it in README + write-up; never imply discovery. Honest framing: "a
from-scratch, validated toolkit that reproduces the known threshold-(in)variance
result end to end as its first demonstration." The artifact's value is the
SOFTWARE — no equivalent packaged library exists on PyPI: sum-of-OU generator +
injection + matched-marginal two-layer-MC harness + tested threshold extractor,
with tests / CI / docs.

Ship on GitHub with README, examples, tests, CI, a technical write-up, and a
blog post.

## Scope

**In scope.** Non-Markovian noise generator (OU and sum-of-OU for 1/f, plus a
matched white baseline); small QEC codes (3-qubit bit-flip, 3-qubit phase-flip,
Shor 9-qubit); parameterized rotated surface code `surface_code(distance,
rounds)` for arbitrary odd d; noise injection layer (trajectory -> gate-level
Z_ERROR); two-layer Monte Carlo benchmarking harness; MWPM decoder (PyMatching)
for the surface code and decoder-free validation for the small codes; tested
threshold extractor (crossing-bracket + parametric-bootstrap CI); threshold
plots comparing Markovian vs. non-Markovian.

**Out of scope (v1).** Novel research claims. Hardware experiments. Qudit codes.
Fault-tolerant gate construction (memory experiments only: encode -> noise ->
decode). Any financial / Heston framing. Correlation-aware decoding,
syndrome/two-qubit-gate correlation, spatial correlation across qubits — these
are Phase 2; do NOT pull them into v1.

## Tech stack

Python 3.13 in a .venv (3.11+ floor). Stim for stabilizer simulation. PyMatching
for surface-code decoding. NumPy / SciPy for SDE integration. Matplotlib for
plots. pytest for testing. ruff + mypy strict for lint and types. GitHub Actions
for CI. MIT license.

## Conventions

- Type hints everywhere. mypy is strict; do not weaken it.
- Docstrings on all public classes and functions. Cite sources for formulas
  (e.g. Gillespie 1996 for the OU exact-update recursion).
- RNG: use `numpy.random.Generator`, never the legacy `np.random.*` module-level
  functions. Generators should be passed in or constructed from a seed parameter
  — never implicit global state.
- Shape contracts in docstrings for all array-returning functions. Document axes
  explicitly.
- One commit per logical chunk. Don't bundle unrelated changes.
- Tests live in `tests/`, mirror the `src/nonmarkov_qec/` layout.
- Public API lives under `src/nonmarkov_qec/`. The four subpackages are `noise`,
  `codes`, `decoders`, `benchmarks`.

## Status — v1 COMPLETE; headline result shipped

HEAD = a2fbf15 (confirm with `git log --oneline -5`). All tests pass; ruff +
mypy --strict + pytest clean, CI green. The full pipeline, the two-layer MC
harness, the threshold extractor, AND the headline experiment all exist and are
committed.

**Noise generation** (validated, with plots):
- `v0.1.0-ou-sampler` — OUProcess, Gillespie exact-update recursion.
- `v0.2.0-sum-of-ou` — SumOfOUProcess for 1/f (from_frequency_band, log-spaced
  tau_j, equal per-component variance; measured 1/f slope -0.998).
- `noise/white.py` — WhiteNoiseProcess, i.i.d. N(0, sigma_total^2), no temporal
  memory; same `.sample(n_steps, dt, n_trajectories, rng)` interface
  (NoiseProcess Protocol).

**Noise injection** — `noise/injection.py`: clip(p_0 + alpha*X, 0, 1) Z_ERROR
per data gate, alpha = m*p_0/sigma. Trajectory shape (n_qubits, n_cycles),
n_cycles = circuit TICK count + 1, indexed by qubit index (ancillas included)
x cycle. MR/MRX/MRY/MRZ = M(p_meas) + ideal reset.

**Small codes** — bit-flip, phase-flip, Shor 9-qubit (decoder-free validation).

**Surface code** — `codes/surface_code.py`: `surface_code(distance, rounds)`,
rotated, X-basis memory, 2d^2-1 physical qubits, rounds = d (square spacetime
patch). Returns a CodeCircuit wrapper: circuit (raw stim.Circuit), data_qubits,
ancilla_qubits, n_qubits, n_cycles, rounds, distance. Distance gate
`shortest_graphlike_error == d` validated at d=3 AND d=5.

**MWPM decoder** — `decoders/matching.py`: matching_from_circuit (DEM ->
pymatching.Matching, decompose_errors=True, single-observable assert);
estimate_logical_error_rate -> DecodeResult(shots, errors, .rate, .stderr).
NOTE: estimate_logical_error_rate rebuilds the matcher internally, so the harness
does NOT use it on the fixed-decoder path — it samples + decodes inline against
a pre-built Matching.

**Benchmarking harness** — `benchmarks/sweep.py`: run_point(), run_sweep(),
SweepPoint dataclass row (model, distance, p_0, shots, n_traj, rate, stderr,
seed). Two-layer MC (below). The decoder is built ONCE per (d, p_0) on a
constant-p_0 (zero-trajectory) circuit and reused across all trajectories —
deliberately correlation-blind. run_point takes `sigma` explicitly; sigma
cancels analytically (alpha*X has std m*p_0 independent of sigma — a pure
normalization), so it is fixed to 1 and fed identically to both process
constructors and the injection map, guaranteeing the matched marginal exactly
(not merely to sampling tolerance).

**Threshold extractor** — `benchmarks/threshold.py`: crossing of
g(p_0) = p_L^{d=5} - p_L^{d=3} by linear interpolation of the first sign change,
with a parametric-bootstrap 95% CI (benchmarking.md section 5). Tested.

**Headline experiment — RUN.** Result: matched-marginal, per-qubit-independent
data-gate dephasing leaves the surface-code memory threshold INVARIANT under the
correlation-blind MWPM decoder, across three decades of tau_c (0.2 -> 200
cycles). Per-band (tau in [0.5, 20] cycles): white p_th = 3.35e-3, 1/f
p_th = 3.29e-3, heavily overlapping CIs (shift consistent with zero). tau_c
sweep is flat with no trend; every OU point lies inside the white 95% band, and
the tau_c -> 0 points agree with the white limit. This is an independent
reproduction of the NON-detrimental regime of Kam et al. — their detrimental
structures (streaky correlation on syndrome/ancilla qubits + two-qubit gates)
lie outside v1's noise placement and define Phase 2. Full write-up:
`docs/results.md`. README has a results section. Figures (DPI=300) in
`docs/figures/` (fine_scan, tau_c_sweep, + matched-marginal overlays).

**The two-layer Monte Carlo** (core design decision):
- Target quantity is p_L = E_X[ p_L(X) ], the logical error rate under the
  noise PROCESS, not one realization.
- Layer 1 (shots): one frozen trajectory -> one noisy circuit -> `shots` samples
  estimate the conditional rate p_L(X).
- Layer 2 (trajectories): N_traj independent draws, averaged. Estimator
  rate = mean_i(r_i); stderr = std_i(r_i)/sqrt(N_traj), folding both layers.
- WHY both: a white trajectory nearly self-averages (per-gate i.i.d.); a 1/f
  trajectory does NOT (a whole trajectory can sit in a high-p excursion), so
  between-trajectory variance is large in the OU arm — the arm carrying the
  headline. Production budgets: shots = 4000, N_traj = 150 (OU) / 60 (white).

Design notes in docs/: noise_model.md, noise_generator.md, sum_of_ou.md,
noise_injection.md, small_codes.md, surface_code.md, benchmarking.md,
results.md.

## Current focus: v1 ship is essentially done

v1 is complete: code, tests, CI, the write-up (docs/results.md), the README
results section, the figures, and the blog / LinkedIn post are all done.
Remaining v1 work is optional polish only, if requested. Do NOT begin Phase 2
without an explicit go-ahead — it is a deliberate scope expansion, not a
continuation of v1.

## Phase 2 (do NOT start without a go-ahead)

"Phase 2" and "v2" are the same thing; separable from v1. The bet: a
**correlation-aware decoder** that reweights the matching graph from the known
noise spectrum, tested against a *detrimental* noise placement (correlation on
syndrome qubits + two-qubit gates, plus spatial correlation across qubits) — i.e.
first reproduce threshold degradation, then measure how much an informed decoder
recovers. Supporting figures in the same paper: p_th(tau_c) across the crossover,
and sub-threshold p_L(d) suppression vs. tau_c. Hardware track (needs lab
access): calibrate from_frequency_band to a measured device spectrum and predict
that device's threshold — no change to the core machinery required.

## Parked

Do not work on these now. Recorded so context isn't lost.

- d=7 patch + full finite-size-scaling collapse (v1 uses the two-distance
  crossing-bracket; a collapse needs >= 3 distances).
- v1.1 physics refinements (deferred in the design notes): quadratic /
  pure-dephasing injection map p = (1-exp(-chi))/2; reset noise; spatial
  correlation across qubits. None blocked the headline result.
- Sum-of-OU pedagogical notebook (separate Summer_2026 teaching track; build
  before a professor meeting if a notebook tour is wanted).

## Working style

- Stop at the end of each step and wait for review before proceeding. Don't
  chain multiple steps in one session.
- If a test fails, do not edit the test to make it pass without flagging it.
  Flag the discrepancy, propose what's wrong, and let the user decide.
- If something in this file is out of date with the actual repo state, say so
  before starting work — don't silently work around it.
- When uncertain about a design choice, ask. Parked items above are parked
  deliberately.
