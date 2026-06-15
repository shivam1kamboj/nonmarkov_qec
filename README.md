# nonmarkov_qec

**Quantum error correction under non-Markovian noise — a simulation and benchmarking library.**

Standard QEC threshold theorems assume Markovian (memoryless) noise. Real superconducting qubits violate that assumption: 1/f flux and charge noise, slow frequency drift, and other correlated 
processes carry memory across gate cycles. `nonmarkov_qec` quantifies what that memory does to a surface-code logical error rate — by simulating the full **encode → correlated-noise → decode** 
pipeline and comparing against a *statistically matched* Markovian baseline.

> **Status — v1 complete.** The full pipeline and benchmarking harness are built and validated end to end: non-Markovian noise generator → Stim noise injection → parameterized surface code → MWPM 
decoder → two-layer Monte-Carlo threshold sweep → bootstrap threshold extraction. **74 tests passing**, `ruff` + `mypy --strict` clean, CI green. The headline measurement — surface-code threshold 
vs. noise correlation time — is done; see [**Results**](#results) and [`docs/results.md`](docs/results.md).

## Why this is hard

A Markovian error channel is fully specified by a single per-gate probability. A non-Markovian one is not: the error rate at cycle *k* is correlated with the rate at cycle *k−1*, so faults arrive 
in **bursts** rather than independently. Two questions follow:

1. *How do you inject correlated noise into a stabilizer simulator* that only speaks per-gate Pauli probabilities? Sample a noise trajectory from a stochastic process, then map it to a time-varying 
`Z_ERROR` schedule.
2. *Does temporal correlation move the threshold* relative to white noise **at identical marginal error power**? The comparison is engineered so the per-cycle error probability has the same mean 
*and* variance in both arms — only its autocorrelation differs. Any measured shift is attributable to memory alone, not to a hidden change in noise strength.

## Results

The headline experiment measures the rotated surface-code memory threshold under 1/f-type dephasing against a matched white baseline, as a function of correlation time `τ_c`. Holding the marginal 
per-cycle error process identical and varying only its temporal autocorrelation:

**The threshold is invariant across three decades of correlation time** (`τ_c = 0.2 → 200` gate cycles). The 1/f and white thresholds coincide within bootstrap 95% confidence intervals at `p_th ≈ 
3.3×10⁻³`, with no trend in `τ_c`.

![Threshold vs noise correlation time](docs/figures/tau_c_sweep.png)

This is a deliberate, defensible result, not a missing one. Using the same matched-marginal methodology, [Kam et al. (2025)](#related-work--attribution) show that *not all* temporal correlations are 
detrimental — the damaging structures are specifically multi-time "streaky" correlations on **syndrome qubits** and **two-qubit gates**. v1 places correlation on **data-gate** dephasing, 
**independent per qubit**, with **uncorrelated readout** — precisely their non-detrimental class. The flat `p_th(τ_c)` is a quantitative confirmation of that benign branch; the harmful structures 
they identify are exactly what v1 does not yet model, and define the Phase 2 roadmap. Full methodology, tables, and the convexity/self-averaging argument: [`docs/results.md`](docs/results.md).

## Pipeline

```
  OU / sum-of-OU          per-gate                rotated             MWPM           two-layer MC
  noise generator   ─▶    Z-error injection  ─▶   surface code  ─▶    decoder  ─▶    threshold sweep
  (1/f spectrum)          (time-varying p)        (Stim circuit)      (PyMatching)   (+ bootstrap CI)
```

Each stage is an independent, separately-tested module. The decoder knows nothing about how the noise was generated; the noise generator knows nothing about the code. This separation is what makes 
the library **calibratable to real hardware** (see [Roadmap](#roadmap)).

## What's built

**Non-Markovian noise generator** — Ornstein–Uhlenbeck and sum-of-OU processes via an exact-update recursion (validated against the analytic stationary distribution, autocorrelation, AR(1) spectrum, 
and the `τ_c→0` white limit). A `from_frequency_band` constructor places log-spaced OU components to synthesize a target 1/f spectrum; the reproduced PSD fits a slope of **−0.998**. A matched 
`WhiteNoiseProcess` provides the Markovian arm through the identical interface.

**Noise injection layer** — translates a sampled trajectory into a per-data-gate dephasing schedule, `p_q(k) = clip(p₀ + α·X_q(k), 0, 1)` with `α = m·p₀/σ`, applied as `Z_ERROR`. A reusable 
compiled-template path makes per-trajectory injection ~35× faster (one Stim parse vs. ~10³ Python appends), verified detector-error-model- and sample-identical to the reference path.

**QEC codes** — bit-flip, phase-flip, Shor 9-qubit (decoder-free validation), and a **parameterized rotated surface code** for arbitrary odd distance, X-basis memory. Code distance is verified 
directly via `shortest_graphlike_error == d` at both `d=3` and `d=5`.

**MWPM decoder** — PyMatching matcher from the circuit's detector-error model (single logical observable), built **once per `(d, p₀)` point** and reused across trajectories (a deliberately 
correlation-blind decoder).

**Two-layer Monte-Carlo harness** — averages over `N_traj` independent noise trajectories (Layer 2), each sampled `shots` times (Layer 1), so the reported quantity is the logical error rate under 
the noise *process*, `p_L = E_X[p_L(X)]`. Polymorphic over any noise process.

**Threshold extraction** — `d=3`/`d=5` crossing-bracket with a parametric-bootstrap 95% CI (tested against synthetic crossings with known thresholds).

**Engineering rigor** — every module is preceded by a reviewed markdown **design note** (`docs/`), typed under `mypy --strict`, linted with `ruff`, and covered by tests that assert *analytic 
predictions*, not just smoke.

## Architecture

```
src/nonmarkov_qec/
├── noise/        OU, sum-of-OU, white baseline, injection layer
├── codes/        bit-flip, phase-flip, Shor, parameterized surface code
├── decoders/     MWPM via PyMatching + Monte-Carlo LER estimator
└── benchmarks/   two-layer MC sweep harness + threshold extraction
docs/             design notes + results writeup + figures
scripts/          scan drivers (coarse, fine, τ_c sweep)
```

## Install

```bash
git clone https://github.com/shivam1kamboj/nonmarkov_qec.git
cd nonmarkov_qec
pip install -e ".[dev]"
```

Requires Python 3.11+. Built on [Stim](https://github.com/quantumlib/Stim) and [PyMatching](https://github.com/oscarhiggott/PyMatching).

## Quickstart

Measure the 1/f vs. matched-white threshold (verified against the current API):

```python
import numpy as np
from nonmarkov_qec.noise.sum_of_ou import SumOfOUProcess
from nonmarkov_qec.noise.white import WhiteNoiseProcess
from nonmarkov_qec.benchmarks.sweep import run_sweep
from nonmarkov_qec.benchmarks.threshold import crossing_threshold

SIGMA = 1.0  # cancels in alpha = m*p0/sigma; one value for both arms (matched marginals)

# 1/f over tau in [0.5, 20] cycles, and the matched white baseline.
ou = SumOfOUProcess.from_frequency_band(
    f_min=1 / (2 * np.pi * 20), f_max=1 / (2 * np.pi * 0.5),
    n_components=6, sigma_total=SIGMA,
)
white = WhiteNoiseProcess(sigma_total=SIGMA)

grid = np.array([0.003, 0.0035, 0.004])
rows = run_sweep(
    ou, model="sum_of_ou", distances=[3, 5], p_0_grid=grid,
    m=0.5, sigma=SIGMA, p_meas=None, shots=2000, n_traj=80, base_seed=0,
)

def curves(rows, d):
    pts = sorted((r for r in rows if r.distance == d), key=lambda r: r.p_0)
    return (np.array([r.p_0 for r in pts]),
            np.array([r.rate for r in pts]),
            np.array([r.stderr for r in pts]))

p0, r3, s3 = curves(rows, 3)
_,  r5, s5 = curves(rows, 5)
est = crossing_threshold(p0, r3, s3, r5, s5, rng=np.random.default_rng(0))
print(f"threshold p_th = {est.p_th:.4g}  95% CI [{est.ci_lo:.4g}, {est.ci_hi:.4g}]")
```

Full sweeps that reproduce the figures: `scripts/fine_scan.py`, `scripts/tau_c_sweep.py`.

## Roadmap

- [x] Project scaffold, CI, MIT license
- [x] Ornstein–Uhlenbeck process sampler with statistical validation
- [x] Sum-of-OU framework for 1/f-like spectra (fitted slope −0.998)
- [x] Small-code implementations: bit-flip, phase-flip, Shor
- [x] Noise injection layer (trajectory → per-gate Stim Z-errors)
- [x] Parameterized surface code (distance verified at d=3 and d=5) + PyMatching decoder
- [x] White-noise (Markovian) baseline + matched-marginal validation
- [x] Two-layer Monte-Carlo harness + bootstrap threshold extraction
- [x] Headline measurement: threshold vs. correlation time (Markovian vs. 1/f)
- [x] Technical write-up ([`docs/results.md`](docs/results.md))

### Phase 2 (next)

The Phase 1 null localizes where the effect must live. Phase 2 targets the *detrimental* regime and a mitigation:

- **Correlated noise on syndrome qubits and two-qubit gates**, plus spatial correlation across qubits — the structures Kam et al. find harmful — to reproduce threshold degradation.
- **A correlation-aware decoder** that reweights the matching graph using the known noise spectrum, to test how much degradation is recoverable.
- **Hardware calibration.** Because the model is parameterized by *measurable* quantities (correlation time, noise spectrum, per-gate rate), the same pipeline can be fit to a measured device 
spectrum and used to predict that device's threshold — a hardware-facing extension requiring no change to the core machinery.

## Design philosophy

Every module begins as a reviewed design note before any code is written — the notes in `docs/` record the physics and modeling decisions (PSD conventions, the diffusion-constant relation `D = 
σ²·τ_c`, the matched-marginal comparison, the rotated CX schedule). The goal is a tool whose correctness can be *read*, not just trusted.

## License

MIT — see [LICENSE](LICENSE).

## Related work & attribution

The v1 benchmark — surface-code logical error rates under non-Markovian (1/f) versus statistically matched Markovian noise — reproduces a known methodology. The closest prior work is:

> J. F. Kam, S. Gicev, K. Modi, A. Southwell, and M. Usman,
> "Detrimental non-Markovian errors for surface code memory,"
> *Quantum Science and Technology* **10**, 035060 (2025).
> [arXiv:2410.23779](https://arxiv.org/abs/2410.23779) ·
> [DOI](https://doi.org/10.1088/2058-9565/adebab) ·
> [code](https://github.com/jkfids/corrqec)

This project is an **independent reproduction**: the noise generator, injection layer, and benchmarking harness were built from scratch before the author was aware of this paper. v1's measured 
threshold *invariance* under matched-marginal, per-qubit-independent temporal dephasing is consistent with the non-detrimental regime they identify; their detrimental structures (syndrome-qubit and 
two-qubit-gate "streaky" correlations) are the Phase 2 target. The contribution here is the **software** — a from-scratch, tested, documented toolkit (sum-of-OU 1/f generator + Stim injection + 
matched-marginal two-layer Monte-Carlo harness + bootstrap threshold extraction) — not a new scientific finding.

Related: *Phys. Rev. A* **112**, 062419 (2025); arXiv:2506.15490.
