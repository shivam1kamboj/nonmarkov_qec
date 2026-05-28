# nonmarkov_qec

Benchmarking quantum error correction codes under non-Markovian noise models.

> **Status:** Early development (Week 1–2 of 10). This README is a stub and will be rewritten when the project ships.

## What this is

Most public QEC simulators (Stim, Qiskit, Cirq) model noise as Markovian channels — depolarizing, amplitude damping, and similar memoryless processes. Real superconducting hardware exhibits non-Markovian effects: 1/f flux noise, charge noise, and slow drifts that violate the Markovian assumption underlying standard threshold theorems.

`nonmarkov_qec` is a Python library that:

1. Generates non-Markovian noise trajectories using stochastic differential equations (Ornstein–Uhlenbeck processes and sums thereof for 1/f-like spectra).
2. Injects those trajectories into Stim-based stabilizer circuit simulations as time-varying Pauli error rates.
3. Benchmarks logical error rates as a function of physical error rate, code distance, and noise model — comparing Markovian baselines against non-Markovian alternatives.

The headline experiment: how much does a colored-noise dephasing channel shift the surface code threshold relative to matched-mean Markovian dephasing?

## Install

```bash
git clone https://github.com/shivam1kamboj/nonmarkov_qec.git
cd nonmarkov_qec
pip install -e ".[dev]"
```

Requires Python 3.11+.

## Quickstart

*Coming in Week 3.*

## Roadmap

- [x] Project scaffold, CI, license
- [x] Ornstein–Uhlenbeck process sampler with statistical validation (Week 1–2)
- [ ] Sum-of-OU framework for 1/f-like spectra (Week 1–2)
- [ ] Small-code implementations: bit-flip, phase-flip, Shor (Week 3–4)
- [ ] Surface code patch + PyMatching decoder (Week 5–6)
- [ ] Noise injection layer + headline experiment (Week 7–8)
- [ ] Polish + technical write-up (Week 9–10)

## License

MIT — see [LICENSE](LICENSE).
