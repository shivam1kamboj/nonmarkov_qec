# nonmarkov_qec

**Quantum error correction under non-Markovian noise — a simulation and benchmarking library.**

Standard QEC threshold theorems assume Markovian (memoryless) noise. Real superconducting qubits violate that assumption: 1/f flux and charge noise, slow frequency drift, and other correlated 
processes carry memory across gate cycles. `nonmarkov_qec` quantifies what that memory does to a surface-code logical error rate — by simulating the full **encode → correlated-noise → decode** 
pipeline and comparing against a *statistically matched* Markovian baseline.

> **Status.** The full simulate-and-decode pipeline is built and validated end to end: non-Markovian noise generator → Stim noise injection → surface code → MWPM decoder → logical error rate. 49 
tests passing, `ruff` + `mypy --strict` clean, CI green. The headline threshold comparison (Markovian vs 1/f) is the remaining step.

## Why this is hard

A Markovian error channel is fully specified by a single per-gate probability. A non-Markovian one is not: the error rate at cycle *k* is correlated with the rate at cycle *k−1*, so faults arrive 
in **bursts** rather than independently. Two questions follow, and the library is built to answer the second:

1. *How do you inject correlated noise into a stabilizer simulator* that only speaks in per-gate Pauli probabilities? (Answer: sample a noise trajectory from a stochastic process, then map it to a 
time-varying `Z_ERROR` schedule.)
2. *Does temporal correlation move the threshold* relative to white noise **at identical marginal error power**? The comparison is engineered so that the per-cycle error probability has the same 
mean and variance in both arms — only its autocorrelation differs. Any measured threshold shift is therefore attributable to memory alone, not to a hidden change in noise strength.

## Pipeline

```
  OU / sum-of-OU            per-gate                 rotated              MWPM            logical
  noise generator   ──▶    Z-error injection  ──▶    surface code  ──▶    decoder  ──▶    error rate
  (1/f spectrum)           (time-varying p)          (Stim circuit)       (PyMatching)
```

Each stage is an independent, separately-tested module. The decoder knows nothing about how the noise was generated; the noise generator knows nothing about the code. This separation is what makes 
the library **calibratable to real hardware** (see Roadmap → hardware extension).

## What's built

**Non-Markovian noise generator** — Ornstein–Uhlenbeck and sum-of-OU processes. A `from_frequency_band` constructor places log-spaced OU components to synthesize a target 1/f-like spectrum; the 
reproduced power spectrum fits a slope of **−0.9976** against an ideal 1/f.

**Noise injection layer** — translates a sampled trajectory `X_q(k)` into a per-data-gate dephasing schedule, `p_q(k) = clip(p₀ + α·X_q(k), 0, 1)`, applied as `Z_ERROR` after each gate. Handles 
measure-and-reset gates and emits a clip-fraction warning when the modulation depth pushes probabilities against the [0,1] boundary.

**QEC codes** (hand-built, transparent gate schedules that map directly to hardware operations):
- 3-qubit bit-flip — Z-blind plumbing/null check
- 3-qubit phase-flip — first real correction of the dephasing noise
- Shor 9-qubit — both error types, validated against exact fired-detector sets
- Rotated **distance-3 surface code**, X-basis memory (generalizing to arbitrary odd distance; d=5 in progress)

**MWPM decoder** — builds a PyMatching matcher from the circuit's detector-error model and estimates logical error rates by Monte Carlo, with binomial error bars.

**Engineering rigor** — every module is preceded by a reviewed markdown **design note** (in `docs/`), typed under `mypy --strict`, linted with `ruff`, and covered by behavioral tests that assert 
*analytic predictions* (e.g. the surface code's code distance is verified directly via `shortest_graphlike_error`, not assumed).

## Architecture

```
src/nonmarkov_qec/
├── noise/        OU, sum-of-OU, injection layer
├── codes/        bit-flip, phase-flip, Shor, surface code
├── decoders/     MWPM via PyMatching + Monte Carlo LER estimator
└── benchmarks/   Monte Carlo sweep harness (in progress)
docs/             design notes: noise model, injection, codes, decoder, benchmarking
```

## Install

```bash
git clone https://github.com/shivam1kamboj/nonmarkov_qec.git
cd nonmarkov_qec
pip install -e ".[dev]"
```

Requires Python 3.11+. Built on [Stim](https://github.com/quantumlib/Stim) and [PyMatching](https://github.com/oscarhiggott/PyMatching).

## Quickstart

> Representative pipeline — verify against the current API before relying on it.

```python
import numpy as np
from nonmarkov_qec.noise import SumOfOUProcess
from nonmarkov_qec.noise.injection import inject_dephasing_noise
from nonmarkov_qec.codes import surface_code
from nonmarkov_qec.decoders import estimate_logical_error_rate

rng = np.random.default_rng(0)

# 1. A 1/f-like dephasing process spanning a target frequency band.
process = SumOfOUProcess.from_frequency_band(f_min=1e2, f_max=1e6, n_components=8)

# 2. A bare (noise-free) surface-code memory circuit.
code = surface_code(distance=3, rounds=3)

# 3. Sample a correlated noise trajectory and inject it as a time-varying Z-error schedule.
trajectory = process.sample(n_qubits=len(code.data_qubits), n_cycles=code.n_cycles, rng=rng)
noisy = inject_dephasing_noise(code.circuit, trajectory, p_0=1e-3, m=0.5)

# 4. Decode and estimate the logical error rate.
result = estimate_logical_error_rate(noisy, shots=10_000, seed=0)
print(f"logical error rate: {result.rate:.4g} ± {result.stderr:.2g}")
```

## Roadmap

- [x] Project scaffold, CI, MIT license
- [x] Ornstein–Uhlenbeck process sampler with statistical validation
- [x] Sum-of-OU framework for 1/f-like spectra (fitted slope −0.9976)
- [x] Small-code implementations: bit-flip, phase-flip, Shor
- [x] Noise injection layer (trajectory → per-gate Stim Z-errors)
- [x] Distance-3 surface code patch + PyMatching decoder
- [x] Parameterized surface code, d=5 patch (distance verified at d=3 and d=5)
- [x] White-noise (Markovian) baseline + matched-marginal validation
- [ ] **Benchmarking harness + threshold sweep** — Markovian-vs-1/f, threshold vs τ_c *(in progress)*
- [ ] Technical write-up + results

### Beyond v1: hardware calibration

Because the noise model is parameterized by *measurable* quantities — correlation time, noise spectrum, per-gate error rate — the same library can be **calibrated to a physical device**: measure a 
processor's noise spectrum, fit sum-of-OU parameters via `from_frequency_band`, and predict its QEC threshold. v1 is the validated, hardware-agnostic foundation for that extension.

## Design philosophy

Every module begins as a reviewed design note before any code is written — the notes in `docs/` record the physics and the modeling decisions (PSD conventions, the diffusion-constant relation, the 
matched-marginal comparison, the rotated CX schedule). The goal is a tool whose correctness can be *read*, not just trusted.

## License

MIT — see [LICENSE](LICENSE).
