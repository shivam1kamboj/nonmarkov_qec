# Benchmarking harness + headline threshold experiment

## 0. Purpose & scope

Produce the headline result: the surface-code logical-error threshold under 1/f (sum-of-OU) dephasing versus a matched Markovian baseline, via the existing noise-generator → injection → surface-code → MWPM-decoder pipeline. Two prerequisites fold into this phase: a d=5 surface-code patch (a second distance is required to locate a threshold crossing) and the Monte Carlo sweep harness.

Non-goals (v1): full finite-size scaling collapse (needs ≥3 distances; d=7 parked), the v1.1 injection refinements (quadratic map, reset noise, spatial correlation).

## 1. The controlled comparison (validity core)

The experiment isolates one variable: the temporal correlation structure (memory) of the noise. The marginal per-cycle error process is held identical across the two arms.

Per data qubit per cycle the injection layer sets

```
p_q(k) = clip(p_0 + alpha * X_q(k), 0, 1),   alpha = m * p_0 / sigma_X
```

where sigma_X is the standard deviation of the supplied trajectory X. Because alpha * X has standard deviation alpha * sigma_X = m * p_0 independent of sigma_X, the marginal distribution of p_q(k) is N(p_0, (m*p_0)^2) regardless of which process generated X. Therefore:

- **Mean matched:** E[p] = p_0 in both arms (X is zero-mean).
- **Marginal variance matched:** Var(p) = (m*p_0)^2 in both arms, by construction.
- **Sole difference:** the autocorrelation of X(k) — delta-correlated (white) in the Markovian arm, a sum of decaying exponentials (1/f over the band, correlation time ~tau_c) in the non-Markovian arm.

This is noise_injection.md "mode 1." The tau_c↔sigma subtlety on record (D = sigma^2 * tau_c, so sigma = sqrt(D/tau_c) → ∞ as tau_c → 0) does not arise here: we hold sigma fixed through the alpha-normalization and never fix a diffusion constant D, so there is no blow-up as the Markovian arm goes white.

What is matched is the instantaneous per-cycle error power, not the accumulated Ramsey/T2 dephasing over a cycle. This is deliberate: correlated noise accumulates phase differently (the variance of a sum of correlated increments is not the sum of variances), and that difference is precisely the effect the experiment measures. Physically the comparison is temporally-independent errors versus bursty errors (runs of high-p cycles clustered in time) at identical instantaneous noise power.

## 2. Sweep axes and point → circuit mapping

Axes: p_0 (per-data-gate baseline dephasing probability, log-ish grid bracketing the threshold), d ∈ {3, 5}, model ∈ {markovian, sum_of_ou}. Held fixed across all points: m (modulation depth), tau_c (active only in the OU arm), rounds, shots, N_traj, base seed.

**rounds convention:** rounds = d (square spacetime patch; temporal extent equals spatial distance so timelike and spacelike protection are comparable). Note tau_c is fixed in absolute (cycle) units and not rescaled with d, so the fraction of the patch correlated by the slow drift changes with distance — physical and intended.

**Two-layer Monte Carlo (both layers required).** The quantity we report is the logical error rate under the noise *process*, `p_L = E_X[ p_L(X) ]`, not the rate under one noise realization. This requires two nested averages:

- **Layer 1 (shots, within a fixed trajectory):** one trajectory X freezes the per-gate Z_ERROR probabilities, producing one noisy circuit. Sampling `shots` records from it estimates `p_L(X)`, the rate *conditional on that trajectory*. `DecodeResult.stderr` is this layer's binomial error only.
- **Layer 2 (trajectories, over the ensemble):** draw `N_traj` independent trajectories and average their conditional rates. This is the average over noise realizations.

Why both layers are mandatory, and why one trajectory is wrong: a single **white** trajectory nearly self-averages — its d^2·rounds per-gate probabilities are i.i.d., so by the law of large numbers `p_L(X)` for one Markovian draw lies close to the ensemble mean. A **1/f** trajectory does **not** self-average: temporal correlation lets an entire trajectory sit in a high-p excursion for many consecutive cycles, so the between-trajectory variance `A = Var_X[p_L(X)]` is substantial in the OU arm. A one-trajectory design therefore understates uncertainty precisely in the arm that carries the headline, and can manufacture or erase an apparent threshold shift purely from which OU trajectory was drawn. Layer 2 is what makes the comparison valid.

**Decoder calibration (fixed on p_0, shared across trajectories).** The MWPM decoder is built **once per (d, p_0) point** from a constant-p_0 circuit (every data gate at the baseline p_0), and reused for every trajectory at that point. The decoder is *not* rebuilt per trajectory. Rationale: (i) physical honesty — a real decoder is calibrated to the average noise model and cannot see the instantaneous correlated burst; rebuilding it per trajectory would let it adapt to the very excursion the experiment is meant to measure, laundering away the effect; (ii) efficiency — the Matching object is shared, so each trajectory costs only sample + decode, not a matcher rebuild.

A single point (p_0, m, tau_c, d, rounds, model) maps to a rate by:

1. `surface_code(d, rounds)` → bare circuit (once per point).
2. Build the decoder once: inject at constant p_0 → `matching_from_circuit` → fixed Matching `M`.
3. For i in 1..N_traj:
   - draw trajectory X^(i) from the model's process (q over d^2 data qubits, k over rounds cycles);
   - `inject_dephasing_noise(circuit, X^(i), p_0, m, p_meas, ...)` → noisy circuit C_i;
   - sample `shots` records from C_i, decode with the fixed `M`;
   - record r_i = errors_i / shots.
4. Combine: `p_L_hat = mean_i(r_i)`; `stderr = std_i(r_i) / sqrt(N_traj)`.

The spread of the r_i carries both layers automatically: within-trajectory binomial noise and between-trajectory variance both enter std_i(r_i), so no separate variance-component model is needed.

## 3. The Markovian baseline arm

Reuse the identical injection path; only the trajectory source changes. The Markovian arm draws X_q(k) i.i.d. N(0,1) (sigma_X = 1, hence alpha = m*p_0), no temporal correlation. The OU arm uses `SumOfOUProcess.from_frequency_band(...)` as today. Both feed the same `inject_dephasing_noise`, so the matched marginals of §1 hold by construction.

The white source is implemented as a small process class in `noise/` exposing the same sampling interface as `SumOfOUProcess` (a `.sample(n_qubits, n_cycles, rng)`-style method returning an array with the same shape contract), so the harness is polymorphic over a process object and the noise model is just a parameter. A unit test asserts the matched-marginal property: across many samples, both processes produce per-cycle p with the same mean (p_0) and variance ((m*p_0)^2) to within tolerance.

## 4. d=5 patch — parameterized rotated surface code

Parameterize `surface_code(distance, rounds)` for arbitrary odd d via a coordinate-driven construction rather than hand-building d=5. Sizing: d^2 data, d^2 − 1 ancilla, 2d^2 − 1 qubits total (17 at d=3, 49 at d=5). Each data and ancilla qubit is assigned a lattice (row, col); each stabilizer's support and CX order are derived from geometry (a consistent per-ancilla traversal order) rather than hardcoded indices. The parameterized construction reproduces the rule behind the current d=3 hand-remap of Stim's rotated_memory_x schedule, not its literal indices.

**Validation gates** (required before the parameterized code is accepted):

- `shortest_graphlike_error(...) == d` at both d=3 (==3) and d=5 (==5). This certifies no hook-error distance collapse from a wrong CX order.
- d=3 regression: behavioral equivalence to the existing hand-built d=3 code — same distance, same detector/observable counts and structure, same decoded logical error rate within stderr. Bit-exact circuit reproduction is not required (the geometric CX order may differ from the hand-remap; behavioral equivalence is the property actually needed).

Fallback if the parameterized route fails the d=5 distance gate: a separately hand-built d=5 validated by the same gate. Parameterize is attempted first.

## 5. Threshold location & plotting

For each model separately: plot logical error rate p_L vs p_0, one curve per distance, with error bars from the two-layer stderr (§2 step 4). Below threshold the d=5 curve sits under d=3; above, it rises above. The crossing of the d=3 and d=5 curves is the threshold estimate p_th.

With two distances the method is a crossing-bracket, not a scaling collapse (collapse needs ≥3 distances; parked). Procedure: locate the p_0 interval where the two central curves cross, locally interpolate each curve, take the intersection as p_th, and obtain its uncertainty by parametric bootstrap — resample each point's rate within its stderr, recompute the crossing, take the spread. Report p_th for each arm with bootstrap CIs; the headline is the difference between the two arms' thresholds and its significance.

The expected physics story (to be measured, not assumed): temporal correlations redistribute error weight across the decoding graph relative to white noise at the same marginal rate, shifting the crossing. Report whatever sign and magnitude is found.

## 6. Harness API / data flow

A `run_sweep(...)` over (model, d, p_0-grid) that, per point, composes the §2 steps and emits a row: (model, d, p_0, shots, N_traj, rate, stderr, seed). Output is a tidy results table (list-of-dataclass or array) consumed by the plotting/threshold code. The harness is model-agnostic: it takes a process object (§3) and a list of distances; nothing about a specific noise type is hardwired. Per-point and per-trajectory seeds are derived deterministically from the base seed for reproducibility.

## 7. Statistics / sample size

Two budgets, total cost ∝ N_traj · shots:

- **N_traj (Layer 2):** controls the between-trajectory variance `A = Var_X[p_L(X)]`, which shrinks only as A/N_traj. Shots cannot reduce it. This is the dominant uncertainty in the OU arm, so N_traj is the priority knob there.
- **shots (Layer 1):** controls the within-trajectory binomial noise, ~p_L(1−p_L)/shots. Set shots large enough that this is comparable to the between-trajectory spread √A — not far below it, since driving binomial noise to zero while N_traj is small wastes budget on a layer that isn't limiting.

Reported uncertainty is the two-layer `stderr = std_i(r_i)/sqrt(N_traj)` from §2 step 4, which folds both layers. Starting scale: shots ∈ [1e2, 1e3], N_traj ∈ [1e2, 1e3], with the OU arm carrying more N_traj than the Markovian arm (which self-averages and tolerates fewer). Plan: coarse scan to bracket p_th, then raise N_traj in the bracket where the curves separate.

## 8. Decisions (consolidated)

- rounds = d.
- Two-layer Monte Carlo: average over N_traj trajectories, each with `shots` samples; `stderr = std_i(r_i)/sqrt(N_traj)`. N_traj is the priority knob (especially OU arm).
- Decoder built once per (d, p_0) on a constant-p_0 circuit and shared across all trajectories (not rebuilt per trajectory).
- tau_c is a harness parameter from the start (headline run at a single tau_c; p_th(tau_c) curve is a no-refactor byproduct).
- White source as a shared-interface process class (harness polymorphic over a process object).
- d=3 regression is behavioral equivalence, not bit-exact.
- Threshold via two-distance crossing-bracket + parametric-bootstrap CI; d=7/collapse parked.

## 9. Build order

note → parameterized `surface_code(d, ...)` with distance gate at d=3 and d=5 + d=3 behavioral regression → white-noise process + matched-marginal unit test → `run_sweep` harness (two-layer MC, fixed shared decoder) → coarse scan → fine scan + threshold extraction + plot. Each step lands ruff + mypy + pytest clean before the next.
