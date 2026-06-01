# Noise injection layer: design note

This note specifies how a noise trajectory X(t) produced by `OUProcess` or
`SumOfOUProcess` is translated into per-gate Pauli error probabilities that a
Stim circuit can consume. It is the analytic specification; implementation
follows in `src/nonmarkov_qec/noise/injection.py`.

## Motivation

Stim is a Clifford-only simulator: it accepts discrete Pauli error channels at
per-instruction rates, not continuous Hamiltonian dynamics. The noise generators
produce continuous stochastic trajectories. The injection layer is the bridge:
it converts a precomputed trajectory X(t) into a sequence of per-cycle,
per-qubit Z-error probabilities that are wired into a Stim circuit.

This is the first QEC-facing component in the library. Its design fixes the
controlled comparison the headline experiment relies on: same mean error rate,
same fluctuation amplitude, only temporal correlation differs.

## The mapping: trajectory to per-gate Z-error probability

At gate cycle k, the noise trajectory on qubit q takes value X_q(k). We map
this to a per-gate Z-error probability via the linear model chosen in
`docs/noise_model.md`:

    p_{q,k} = clip(p_0 + alpha * X_q(k), 0, 1)

where:

- `p_0` is the mean physical error rate (the x-axis of threshold plots),
- `alpha` is the sensitivity of the error rate to the noise trajectory,
- `clip(·, 0, 1)` enforces that the probability stays in [0, 1].

Since X has mean zero by construction, the time-average of `p_{q,k}` over a
long trajectory equals `p_0` exactly. This mean-matching property is what makes
the Markovian-vs-non-Markovian comparison clean: both cases share the same mean
error rate.

## Modulation depth: the user-facing knob for alpha

`alpha` is not exposed directly. Its natural units are (probability) / (noise
amplitude), which depend on the chosen noise process and make cross-process
comparison unintuitive. Instead the user specifies a dimensionless **modulation
depth**:

    m = alpha * sigma / p_0

which is the relative fluctuation amplitude of the error rate. A value of
`m = 0.5` means the error rate fluctuates by roughly ±50% of its mean.
Rearranging, the internal conversion is:

    alpha = m * p_0 / sigma

where `sigma` is the stationary standard deviation of the noise process
(`OUProcess.sigma` for a single component; `sigma_total` for
`SumOfOUProcess`). This conversion is performed once at injection time; the
user never handles `alpha` directly.

### Why this parameterization is cleaner

Three properties make `m` the right user-facing knob:

1. **Dimensionless and interpretable.** `m` has a direct physical reading
   (relative fluctuation of the error rate) regardless of whether the
   underlying process is a single OU or a sum-of-OU with a different
   `sigma_total`.

2. **Automatic scaling.** If the user changes the noise amplitude (e.g.,
   increases `sigma_total` to model noisier hardware), `alpha` adjusts
   automatically to keep the fluctuation fraction constant. Without this,
   changing `sigma` while holding `alpha` fixed would change the effective
   fluctuation depth and confound the comparison.

3. **Clean controlled comparison.** For the Markovian-vs-non-Markovian
   experiment, both baselines are run at the same `(p_0, m)` pair. The only
   free variable is the correlation time `tau_c`. This is the minimal
   experiment: one knob changed, all others held fixed.

### Validity range of m and the clipping constraint

The linear model is valid only when `p_{q,k}` stays away from the [0, 1]
boundaries. Near-zero probabilities are clipped to zero (error rate saturates,
losing trajectory information); near-one probabilities are clipped to one
(error rate saturates in the other direction). Both distort the intended
statistics.

The clip at 0 fires when `p_0 + alpha * X_k < 0`, i.e., when
`X_k < -p_0 / alpha = -sigma / m` (substituting `alpha = m * p_0 / sigma`).
This is a Gaussian tail at `-(1/m)` standard deviations of X. The clip at 1
fires when `X_k > (1 - p_0) / alpha = (1 - p_0) * sigma / (m * p_0)`, a
much larger threshold for the small `p_0` values on threshold plots, so the
lower clip dominates. In terms of m:

- **`m <= 0.5` is safe** for the error rates of interest (`p_0` in the range
  0.001–0.1): the clip fires when X_k falls below -2 sigma, a few percent of
  cycles and small enough that the distortion to the trajectory statistics is
  negligible.
- **`m > 1` should be flagged** in the implementation: at this depth the
  trajectory regularly pushes `p_k` to zero or one, and the linear model no
  longer faithfully represents the intended fluctuation statistics.
- **`m > 1.5` is likely invalid** for any `p_0` in the threshold-plot range:
  the clipped and unclipped distributions diverge substantially.

The implementation will warn if the `(p_0, m, sigma)` triple implies that more
than ~1% of cycle probabilities would be clipped. This is a soft check, not
a hard error — the user may deliberately explore the nonlinear regime — but
silence would be misleading.

## Per-qubit independence: the v1 multi-qubit convention

Each physical qubit `q` gets its own independent noise trajectory `X_q(t)`,
sampled from the same process parameters `(tau_c, sigma)` or
`(tau_min, tau_max, n_components, sigma_total)` but with an independent
random seed. The set of trajectories is therefore a matrix of shape
`(n_qubits, n_cycles + 1)` as returned by `sample()` (see trajectory shape
convention below).

**For a single-qubit gate on qubit q at cycle k,** the Z-error probability is
`p_{q,k}` as above.

**For a two-qubit gate on qubits A and B at cycle k,** each qubit dephases
according to its own environment: qubit A uses `p_{A,k}` and qubit B uses
`p_{B,k}`. The joint error channel is the product of two independent single-
qubit Z channels.

### Physical motivation

Each superconducting qubit couples to its own local set of two-level
fluctuators (charge traps, surface spins, junction defects). While
fluctuators on nearby qubits can share a bath through the substrate, the
dominant 1/f dephasing in state-of-the-art devices is qubit-local. Independent
trajectories per qubit is therefore the physically well-motivated v1 baseline.

### QEC-specific motivation

Independent trajectories isolate **temporal non-Markovianity** as the single
experimental variable. This is important for two reasons:

1. **Decoder validity.** PyMatching's minimum-weight perfect matching assumes
   that errors on distinct qubits are spatially independent (the matching
   graph's edge weights are calibrated per-edge, not per-correlated-cluster).
   Spatially correlated trajectories would violate this assumption, causing the
   decoder to underperform for reasons unrelated to temporal memory. Using
   independent trajectories keeps the decoder in its intended operating regime
   and ensures any threshold shift is attributable to temporal correlation
   alone.

2. **Interpretability.** If we introduced both temporal and spatial correlation
   simultaneously, the headline result ("1/f noise shifts the threshold") would
   be ambiguous: is it the memory, the spatial correlation, or their
   interaction? v1 separates these by holding spatial structure fixed
   (independent) and varying only `tau_c`.

Spatial correlation (correlated baths across qubits) is listed as a v2
extension in the Assumptions section below.

## Markovian vs. non-Markovian baseline at matched mean

The controlled comparison is between two injection modes:

| Property | Markovian baseline | Non-Markovian |
|---|---|---|
| Trajectory | White noise: X_k ~ N(0, sigma^2) i.i.d. | OU or sum-of-OU trajectory |
| Temporal correlation | None (tau_c -> 0) | tau_c set by process |
| Mean error rate | p_0 | p_0 (same) |
| Modulation depth | m | m (same) |

The Markovian baseline can be constructed in two equivalent ways:

1. Draw `X_k` i.i.d. N(0, sigma^2) at each cycle. This is mathematically
   equivalent to an OU process in the limit `tau_c -> 0` with **sigma held
   fixed**, so the per-cycle marginal distribution N(0, sigma^2) is preserved
   and the error-rate variance is unchanged between the two modes. Note that
   this is a different limit from the one used in the Markovian-limit
   validation test in `test_ornstein_uhlenbeck.py`, where the diffusion
   coefficient D = sigma^2 * tau_c is held fixed as tau_c -> 0 (so
   sigma = sqrt(D/tau_c) grows without bound while D stays constant). Both
   limits are physically meaningful; the injection baseline uses the
   fixed-sigma version because it preserves the per-cycle error-rate
   distribution and makes the Markovian-vs-non-Markovian comparison a
   controlled one: same variance of p_k, only the temporal memory differs.

2. Set `p_k = p_0` constant (zero modulation, `m = 0`). This is a simpler
   baseline but conflates "no correlation" with "no fluctuation" — it removes
   the variance of `p_k` as well as its memory. Mode 1 is the controlled
   comparison; mode 2 is the pure-Markovian reference also sometimes shown.

The headline experiment uses mode 1 for the Markovian baseline: same `(p_0, m)`, same per-cycle variance of `p_k`, only `tau_c` differs. Mode 2 (constant p_0) is included as an additional reference curve on threshold plots.

## Stim integration sketch

The trajectory is precomputed by the noise generator and passed to the injection
function alongside circuit metadata. The function iterates over the circuit's
gate cycles and inserts `Z_ERROR(p_{q,k})` instructions after each data-gate
moment on qubit q.

Only **data-gate dephasing** is modulated by the trajectory. Measurement
errors and reset errors use a constant Markovian rate (a separate parameter
`p_meas`) in v1. This reflects the physical picture: flux noise drives
qubit-frequency fluctuations between gates; readout fidelity is limited by
other mechanisms (photon shot noise, dispersive coupling) that are not modeled
by the OU process.

### Trajectory shape and index convention

`OUProcess.sample()` (and `SumOfOUProcess.sample()`) returns an array of shape
`(n_trajectories, n_steps + 1)`, where index 0 is the initial condition `x0`
and index `k >= 1` is the state after `k` time steps of size `dt`.

The injection function expects trajectories of shape `(n_qubits, n_cycles)`,
where `trajectories[q, k]` is the noise value for qubit `q` at circuit cycle
`k` (zero-indexed, covering cycles 0 through n_cycles - 1). The initial
condition at index 0 of `sample()` output is **excluded**: the caller is
responsible for passing `raw_samples[:, 1:]` to strip it. This convention must
be documented at the call site to prevent silent off-by-one misalignment between
the noise trajectory and the circuit cycle it modulates.

The number of time steps `n_steps` passed to `sample()` must equal `n_cycles`,
the number of gate cycles in the circuit. The time step `dt` must equal the
gate cycle duration. Both must be set by the caller; the injection function
does not infer them.

### Intended API

```python
def inject_dephasing_noise(
    base_circuit: stim.Circuit,
    trajectories: np.ndarray,  # shape (n_qubits, n_cycles), X_q(k) for cycle k;
                               # initial condition excluded (pass sample()[:, 1:])
    p_0: float,                # mean Z-error rate
    m: float,                  # modulation depth (dimensionless)
    sigma: float,              # noise process stationary std dev
    p_meas: float,             # constant measurement error rate
) -> stim.Circuit:
    """
    Insert per-cycle Z_ERROR(p_{q,k}) instructions into a Stim circuit.

    alpha = m * p_0 / sigma is computed internally.
    trajectories[q, k] is X_q at circuit cycle k (0-indexed).
    Measurement and reset moments use p_meas unchanged.
    """
    ...
```

The `base_circuit` is a Stim circuit with the QEC code's stabilizer structure
but no per-qubit error channels. The function produces a new circuit with
`Z_ERROR` instructions inserted at the appropriate moments. This decouples
noise generation (NumPy / SciPy) from circuit simulation (Stim) cleanly and
allows the same trajectory to be reused across multiple circuit variants
(e.g., different code distances) without regenerating samples.

## Assumptions and limitations

1. **Pauli-twirl approximation.** Real flux noise produces continuous Z
   rotations, not discrete Z flips. The mapping to a Pauli Z channel invokes
   an implicit Pauli twirl, which is standard for QEC threshold estimation
   but discards coherence information. The Pauli twirl is exact in the limit
   of fast randomization; for slow (non-Markovian) noise it is an
   approximation whose error is not quantified in v1.

2. **No spatial correlations.** Each qubit gets an independent trajectory.
   Correlated baths (e.g., two qubits sharing a flux line or a substrate
   phonon mode) are out of scope for v1. Spatial correlation is the most
   physically motivated v2 extension: it would further stress the decoder
   (whose matching weights assume spatial independence) and is expected to
   produce an additional threshold shift, separable from the temporal-memory
   effect studied in v1.

3. **Measurement and reset noise is constant.** Only data-gate dephasing is
   modulated. This is a simplifying assumption, not a fundamental one; a v2
   extension could model readout noise with a separate correlated process.

4. **Per-cycle granularity.** The trajectory is evaluated once per gate cycle.
   Sub-cycle dynamics (e.g., noise that varies within a single gate duration)
   are averaged out. This is exact for instantaneous Pauli errors and is a
   standard approximation at the gate-cycle level of description.

5. **Linear model validity.** The mapping `p_k = p_0 + alpha * X_k` is linear
   and is valid only for small `m` (see the clipping constraint discussion
   above). A quadratic / pure-dephasing model `p_k = (1 - exp(-X_k^2 t_gate))
   / 2` is a more physically faithful alternative; it is deferred to v1.1.

## Open questions resolved by this note

- **Is alpha user-facing?** No. The user specifies modulation depth `m`;
  `alpha` is derived internally as `m * p_0 / sigma`.

- **Do multi-qubit gates share a trajectory?** No. Each qubit has its own
  independent trajectory in v1. The physical motivation is local fluctuators;
  the QEC motivation is keeping temporal memory as the single experimental
  variable and preserving the decoder's spatial-independence assumption.
