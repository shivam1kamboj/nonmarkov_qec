# Noise model design note

*Status: draft, to be revised at end of Week 2.*

## Motivation

Stim is a Clifford-only simulator: it tracks Pauli error channels, not continuous
Hamiltonian dynamics. To inject a non-Markovian process into a Stim circuit, we
must translate a continuous noise trajectory into a sequence of per-gate Pauli
error probabilities.

This note records the translation we use, the assumptions behind it, and the
limitations.

## The noise process

We focus on dephasing (Z) noise, motivated by 1/f flux noise in superconducting
qubits. The noise process is a stationary Ornstein-Uhlenbeck process X(t) with
correlation time tau_c and stationary standard deviation sigma:

    dX = -(X / tau_c) dt + sqrt(2 sigma^2 / tau_c) dW

For the v1.1 multi-component model, X(t) is replaced by a sum of independent
OU processes with logarithmically-spaced tau_c, weighted to approximate 1/f
over a chosen frequency band.

## Translation: trajectory to Pauli probabilities

At each gate cycle k, the noise trajectory takes value X_k. We map this to a
per-gate Z-error probability via:

    p_k = clip(p_0 + alpha * X_k, 0, 1)

where:
- `p_0` is the mean error rate (the "physical error rate" on threshold plots),
- `alpha` controls the fluctuation amplitude. We fix `alpha` such that the
  *mean* of `p_k` over a long trajectory equals `p_0` (which is true by
  construction since X has mean zero) and the *variance* of `p_k` matches a
  target value chosen to keep `p_k >= 0` with high probability.

The Markovian baseline used for comparison is `p_k = p_0` constant across
all gate cycles, with all other circuit parameters identical.

## Comparison to literature

This linear map is the simplest defensible choice. Alternative choices include:

- Quadratic / pure-dephasing model: `p_k = (1 - exp(-X_k^2 t_gate)) / 2`,
  which is what one gets by integrating a Hamiltonian H = X(t) Z over the
  gate duration and tracing out the environment in the Markovian limit.
- Time-correlated unitary errors before each gate (more physically faithful
  but harder to inject into Stim).

We use the linear model for v1 because it (a) admits exact mean matching for
clean apples-to-apples comparison, (b) is simple enough to debug, and (c)
captures the key qualitative feature — temporal correlations in the error
rate — that distinguishes non-Markovian from Markovian noise.

The quadratic model is a candidate for v1.1.

## Assumptions and limitations

1. **Pauli channel approximation.** Real flux noise produces continuous
   rotations, not discrete Z flips. The Pauli twirl approximation we
   implicitly invoke is standard for QEC threshold estimation but loses
   coherence information.
2. **No spatial correlations.** Each qubit gets an independent noise
   trajectory. A more realistic model would have correlated noise across
   nearby qubits sharing a flux line; we leave this for future work.
3. **No measurement / readout noise modeling.** v1 puts noise only on data
   gates; measurement errors use a constant Markovian rate.
4. **Gate-cycle granularity.** The trajectory is evaluated once per gate
   cycle, not continuously through the gate duration. This is exact for
   instantaneous Pauli errors but smooths over sub-cycle dynamics.

## Interface sketch

```python
def inject_dephasing_noise(
    base_circuit: stim.Circuit,
    trajectory: np.ndarray,  # shape (n_qubits, n_cycles), values of p_k
) -> stim.Circuit:
    """Insert per-cycle Z_ERROR(p_k) instructions into a Stim circuit."""
    ...
```

The trajectory is precomputed by the noise generator, then wired into Stim's
per-instruction error rate support. This decouples noise generation
(SciPy / NumPy) from circuit simulation (Stim) cleanly.

## Open design questions

- Should `alpha` be exposed as a user parameter, or derived from a single
  "fluctuation strength" knob? Decide at start of Week 7.
- For multi-qubit gates, does each qubit get its own trajectory or do they
  share? Lean toward independent for v1; revisit if results look weird.
