# Small QEC codes: design note

This note specifies the first QEC-facing circuit layer: three small codes
implemented as Stim circuits, to be exercised by the noise injection layer
(`src/nonmarkov_qec/noise/injection.py`). It is the analytic/structural
specification; implementation follows in `src/nonmarkov_qec/codes/`.

Design-note-first discipline: no code until this is reviewed.

## Scope

**In scope (v1).** Three memory experiments (encode → rounds of syndrome
extraction → readout), as `stim.Circuit` objects with detector/observable
annotations:

1. **3-qubit bit-flip** (repetition code, Z-basis memory) — via Stim's
   generator.
2. **3-qubit phase-flip** (X-basis memory) — hand-built.
3. **Shor 9-qubit** — hand-built.

Plus the metadata each circuit must expose downstream, and the reconciliation
of the injection layer's gate-classification sets against real Stim output.

**Out of scope.** Decoding (separate roadmap item — lookup tables for these
small codes, PyMatching for the surface code). Logical error *rates* (need a
decoder). The surface-code patch (later phase). Any noise other than the
Z-dephasing channel the injection layer already emits.

## 1. Code / noise pairing — the physics that drives the rest

The injection layer emits **pure Z (dephasing)** errors. How each code responds
to a Z channel is not incidental — it determines what each code is *for* in v1.

### Bit-flip code is blind to our noise (intended)

Stabilizers are Z-type: `Z0Z1`, `Z1Z2`. A Z error on a data qubit commutes with
every stabilizer, so it produces **no detection events**. Worse (for
"protection") and better (for a clean null): the bit-flip memory prepares
`|0_L⟩`, a +1 eigenstate of `Z_L`, and reads out in the Z basis. A Z error acts
as `Z|0⟩ = |0⟩` and `Z|1⟩ = -|1⟩` — a *phase* the computational-basis
measurement cannot see. So a data-qubit Z error is invisible **twice over**:
undetected by the syndrome and inconsequential to the logical Z-basis outcome.

**Consequence:** under our Z-injection, the bit-flip code's logical error rate
is independent of `(p_0, m)` of the dephasing channel. Any residual logical
failures come from `p_meas` on the final readout alone. This is the intended,
expected-null result.

**What the bit-flip code is for in v1:** a *plumbing and vocabulary* test, not a
protection demonstration. It is the simplest real Stim code circuit, so it is
where we (a) exercise the injection layer against genuine Stim-emitted
vocabulary (`MR`, detector/observable structure, Stim's TICK idiom), and (b)
assert the noise is correctly Z-type and the code correctly Z-stabilized: "we
inject dephasing, we see zero detection events, the observable is unmoved." A
nonzero detection rate here would be a bug in the injection layer or the noise
model.

(A bit-flip code that *does* respond would require injecting X errors; the
injection layer does not emit those in v1. An X-noise variant is a documented
v2 item, below.)

### Phase-flip code corrects our noise (the first real demonstration)

The Hadamard conjugate of the bit-flip code: stabilizers are X-type, `X0X1`,
`X1X2`; logical `X_L` = X on a data qubit, `Z_L = Z0Z1Z2`. It prepares `|+_L⟩`
(= `|+++⟩`, a +1 eigenstate of `X_L`) and reads out in the X basis.

A Z error **anticommutes** with the X-stabilizers it touches, so it flips
stabilizer parity and **is detected**. A single data-qubit Z is a correctable
weight-1 error in this frame. This is the code that actually pairs with our
dephasing noise — the first circuit where injected noise produces syndromes a
decoder could act on.

### Shor 9-qubit handles both (the capstone)

Concatenation of phase-flip (outer) over bit-flip (inner): six weight-2 Z
stabilizers (within each block of three) and two weight-6 X stabilizers (across
blocks). A single data Z anticommutes with the X stabilizer(s) covering it →
detected. Shor corrects our Z-noise with more overhead and is the
"corrects-arbitrary-single-qubit-error" demonstration.

### Summary

| Code | Stabilizers | Prep / readout basis | Response to injected Z | Role in v1 |
|---|---|---|---|---|
| Bit-flip | `Z0Z1`, `Z1Z2` | Z | none (blind, by construction) | plumbing / vocabulary test, expected-null |
| Phase-flip | `X0X1`, `X1X2` | X | detected, correctable | first real correction demonstration |
| Shor 9q | 6× Z (weight-2), 2× X (weight-6) | — | detected via X-checks | both-error capstone |

## 2. Circuit structure and the cycle boundary

This is the section the injection-layer alignment depends on, so it is grounded
in the **actual** flattened output of
`stim.Circuit.generated("repetition_code:memory", distance=3, rounds=3)`
(stim 1.16.0), reproduced here in abbreviated form:

```
R 0 1 2 3 4            # k = 0  (PASSTHROUGH: no Z_ERROR, no clock advance)
TICK                   # -> k = 1
CX 0 1 2 3             # data gate, dephased at k = 1
TICK                   # -> k = 2
CX 2 1 4 3             # dephased at k = 2
TICK                   # -> k = 3
MR 1 3                 # measure+reset ancillas: M(p_meas), reset ideal
DETECTOR ... ; DETECTOR ...
TICK                   # -> k = 4   (round 2 begins) ...
...                    # rounds repeat: 3 TICKs/round
M 0 2 4                # final data readout at k = 9: M(p_meas)
DETECTOR ... ; OBSERVABLE_INCLUDE(0) ...
```

### Qubit layout

Stim interleaves: data on **even** indices (0, 2, 4), ancillas on **odd** (1,
3). The injection layer is layout-agnostic — it maps Stim qubit index `q` to
trajectory row `q` — so what matters is only that **every physical qubit that
ever appears in a data gate has a trajectory row**, including ancillas (the CX
targets above are ancillas, and they receive their own `Z_ERROR`). For a
distance-`d`, `r`-round repetition code there are `2d - 1` physical qubits;
`trajectories.shape[0]` must be `2d - 1`, not `d`.

### What "one cycle" is

One **moment** = the span between consecutive TICKs (with the pre-first-TICK
span as moment 0 and the post-last-TICK span as the final moment). The
injection layer's clock `k` increments on each TICK, so:

```
n_cycles = (number of TICKs) + 1
```

For the generated repetition code, TICKs = `3 · rounds`, hence
`n_cycles = 3·rounds + 1` (= 10 for `rounds=3`). The injection layer counts
TICKs itself and validates `trajectories.shape[1] == n_ticks + 1`; the **caller
must not hardcode** this — read it off the built circuit.

### Which moments carry dephasing

Crucially, **not every cycle column is read by a data gate**:

- **k = 0** is the reset moment (`R`, PASSTHROUGH). Trajectory column 0 is read
  by nothing in a generated circuit. It still counts toward `n_cycles`; it is
  simply unused. (Sampling it costs nothing and keeps the moment↔column
  bijection clean.)
- **Measurement moments** (`MR`, final `M`) take `p_meas`, never the
  trajectory.
- Only **data-gate moments** (the CX layers, k ∈ {1,2,4,5,7,8} above) inject
  `Z_ERROR(clip(p_0 + α·X_{q,k}))`.

### v1 approximation to name explicitly: no idle dephasing

During an `MR` moment the *data* qubits are idling, and under real flux noise
they would accumulate phase. v1 injects **no** Z_ERROR on idling data qubits —
consistent with the injection layer's "only data-gate dephasing is modulated"
convention, but it means the dephasing the circuit sees is undercounted at
measurement moments. This is a known v1 limitation (companion to the
measurement/reset-noise-is-constant and per-cycle-granularity assumptions in
`docs/noise_injection.md`). Idle dephasing is a candidate v1.1/v2 refinement.

Physical reading of the trajectory across resets: `X_q(t)` represents the
*environment* (e.g. flux noise), not the qubit state. Resetting the ancilla
wipes the qubit but not its environment, so the trajectory advances continuously
across `MR`. Ancilla trajectory rows are therefore physically well-defined.

### Caller recipe (to be documented at the call site)

```
build circuit  ->  n_ticks = circuit.flattened().count("TICK")
n_cycles = n_ticks + 1
raw = process.sample(n_steps=n_cycles, dt=dt, n_trajectories=n_qubits, rng=rng)
trajectories = raw[:, 1:]              # strip x0; shape (n_qubits, n_cycles)
noisy = inject_dephasing_noise(circuit, trajectories, p_0, m, sigma, p_meas)
```

`dt` is the per-moment (per-TICK) duration. v1 treats all moments as equal `dt`
even though a CX layer and an `MR` differ in physical duration — another
per-cycle-granularity approximation.

## 3. Hand-built vs. generated (decision)

Settled:

- **Bit-flip → `stim.Circuit.generated("repetition_code:memory", ...)`.** Used
  *because* it emits real Stim vocabulary and TICK placement, making the
  bit-flip case the live re-validation of the injection layer against genuine
  Stim output (the deferred re-check), executed for real rather than against
  hand-built fixtures.
- **Phase-flip → hand-built.** Stim's repetition generator only does the
  Z-basis memory; there is no built-in X-basis task, and this is the code that
  actually corrects our noise.
- **Shor → hand-built.** No generator exists.

**Binding constraint on hand-built circuits:** use **only** the vocabulary the
generators emit — `R, TICK, H, CX, MR, M, DETECTOR, OBSERVABLE_INCLUDE` — and
**mirror Stim's TICK idiom** (a TICK before each gate layer and before each
`MR`). This guarantees the injection layer's gate sets cover generated and
hand-built circuits uniformly, and prevents reintroducing the
fixtures-don't-match-Stim gap.

Pedagogical arc (intended): read the canonical generated bit-flip circuit to
learn the Stim idiom, then hand-build phase-flip mirroring its structure, then
Shor.

## 4. Gate-vocabulary reconciliation (resolves the deferred re-check)

Verified by generating real circuits (stim 1.16.0):

- `repetition_code:memory` emits: `R, TICK, CX, MR, DETECTOR, M,
  OBSERVABLE_INCLUDE`.
- `surface_code:rotated_memory_z` / `unrotated_memory_z` additionally emit
  `QUBIT_COORDS, H`.

Every token except **`MR`** is already in the injection layer's classification
sets. `MR` (measure-and-reset, the standard ancilla operation) is in none of
them, so the current code raises `ValueError: unrecognised gate 'MR'` on the
first real code circuit. The hand-built fixtures used separate `M` + `R`, which
is why no test exercised this path.

### Resolution: treat `MR` as `M(p_meas)` + ideal reset

`MR q` is a measurement (gets `p_meas`, exactly like `M`) immediately followed
by an ideal reset (no error, consistent with `R` being PASSTHROUGH in v1). This
is the only choice consistent with the existing model: measurements use constant
`p_meas`, resets are ideal in v1, only data gates are trajectory-modulated.

Required change to `injection.py` (to be applied in the implementation step,
reviewed as a diff):

- Add `MR` (and, for completeness against hand-rolled circuits and other
  generators, `MRX, MRY, MRZ`) to the measurement-handling path.
- The measurement branch already emits `out.append(name, targets, p_meas)`;
  `MR(p_meas)` is valid Stim and performs the measure-with-flip-probability then
  ideal reset. So the fix is to recognize the `MR*` names as measurements, not
  to add a new branch.
- `MR` must **not** advance the clock (only TICK does) and must **not** read the
  trajectory — same as `M`.

This belongs in `MEASUREMENT_GATES` semantics; document the
measure-then-ideal-reset reading in the injection docstring so the convention is
explicit rather than implied.

## 5. Module output contract

Each code constructor in `codes/` returns a `stim.Circuit` plus metadata the
injection layer and the future decoder both need. Proposed structure (a small
frozen dataclass, exact form to be settled at implementation):

- `circuit: stim.Circuit` — annotated base circuit, **no error channels**
  (injection adds them).
- `data_qubits: tuple[int, ...]` — Stim indices of data qubits.
- `ancilla_qubits: tuple[int, ...]` — Stim indices of ancillas.
- `n_qubits: int` — total physical qubits = `max(index) + 1` (the row count the
  trajectory array must match).
- `n_cycles: int` — `circuit.flattened()` TICK count + 1 (the column count the
  trajectory array must match). Provided so the caller never recomputes it
  inconsistently.
- `rounds: int`, `distance: int` — for bookkeeping and plot axes.
- A note on the detector/observable layout (how many detectors, what the single
  observable is) — needed by the lookup-table decoder later.

Rationale: this decouples noise generation (NumPy) from circuit construction
(Stim) and lets the same trajectory feed multiple circuit variants. It also
makes the two shape contracts (`n_qubits` rows, `n_cycles` columns) explicit at
the boundary, where the off-by-one risks live.

## 6. Per-code specifications (structural)

Enough to implement; full Stim code is the implementation step, not this note.

### Bit-flip (generated)

`stim.Circuit.generated("repetition_code:memory", distance=3, rounds=r,
after_clifford_depolarization=0, before_round_data_depolarization=0,
before_measure_flip_probability=0, after_reset_flip_probability=0)` — all noise
args zero; the injection layer is the sole noise source. Read `data_qubits =
(0,2,4)`, `ancilla_qubits = (1,3)` from the layout. Validation expectation:
injected Z noise produces **zero** detection events and an unmoved observable.

### Phase-flip (hand-built)

Mirror the bit-flip structure under Hadamard conjugation. Proposed contiguous
layout (readability; injection is layout-agnostic): data `0,1,2`; ancillas
`3,4` for `X0X1` and `X1X2`.

- **Encode `|+_L⟩`:** `R 0 1 2`, then `H 0 1 2` (giving `|+++⟩`). (For the
  repetition-style memory this is the prepared logical state; the codespace is
  `span{|+++⟩, |---⟩}`.)
- **One syndrome round** (measure `X_aX_b` with an ancilla): `R` ancilla,
  `H` ancilla (→ `|+⟩`), `CX(anc, a)`, `CX(anc, b)`, `H` ancilla, `MR` ancilla.
  TICK between layers, TICK before `MR`, mirroring Stim's idiom.
- **Readout in X basis:** `H 0 1 2` then `M 0 2 ...` (or `MX`); pick the form
  whose canonical Stim name is in the gate sets — prefer `H` + `M` to stay in
  the verified vocabulary.
- **Detectors:** compare each ancilla `MR` outcome to the same stabilizer in the
  previous round (deterministic = 0 in the noiseless circuit); final detectors
  tie the last round to the data readout. `OBSERVABLE_INCLUDE` = the `X_L`
  parity.

### Shor 9-qubit (hand-built)

Data `0..8` in three blocks `{0,1,2}, {3,4,5}, {6,7,8}`. Stabilizers: per-block
`Z` checks `Z0Z1, Z1Z2, Z3Z4, Z4Z5, Z6Z7, Z7Z8` (bit-flip layer, six weight-2);
cross-block `X` checks `X0..X5` and `X3..X8` (phase-flip layer, two weight-6).
One ancilla per stabilizer (indices `9..16`) for clarity in v1; ancilla reuse is
a later optimization. Encode the Shor state, run `r` rounds, read out. The X
checks are the ones that detect our Z-noise. Exact gate ordering and detector
wiring are the implementation step.

## 7. Validation plan (decoder-free)

Decoding is out of scope, so validation checks **circuit correctness and
injection plumbing**, not logical error rates:

1. **Noiseless determinism.** With injection off (or `p_0 = m = p_meas = 0`),
   every `DETECTOR` is deterministically 0 — Stim will error at compile time if
   a detector is non-deterministic, and `circuit.detector_error_model()` should
   build cleanly. Confirms stabilizer/detector wiring.
2. **Detector/observable counts** match the expected `stabilizers × (rounds+1)`
   structure and a single logical observable.
3. **Single-fault response (the physics check).** Insert one `Z` on a data qubit
   and inspect which detectors fire:
   - bit-flip: **no** detector fires and the observable is unchanged (the
     blind-by-construction assertion);
   - phase-flip / Shor: the expected X-stabilizer detector(s) fire.
4. **Observable flips** under a logical-weight error in the detectable cases.
5. **Injection-layer integration on a real circuit.** Run
   `inject_dephasing_noise` on the generated bit-flip circuit (post-`MR` fix) and
   confirm: no `unrecognised gate` error, `n_cycles`/`n_qubits` contracts hold,
   and detection-event rate is zero for the bit-flip Z case.

These run in CI without a decoder and lock the circuit layer before the decoding
phase.

## 8. Open questions and deferrals

- **X-noise variant for the bit-flip code.** Would make bit-flip a *protection*
  demonstration rather than a null. Requires the injection layer to emit
  `X_ERROR`. Deferred (v1.1) — out of scope for the Z-dephasing v1.
- **Idle dephasing.** Data qubits idling during `MR` moments are not dephased in
  v1 (§2). Candidate v1.1/v2 refinement.
- **Ancilla reuse in Shor** (fewer than one ancilla per stabilizer) — later
  optimization; v1 favors clarity.
- **Metadata dataclass exact fields** (§5) — settle at implementation.
- **Spatial correlation across qubits** — already deferred to v2 in
  `docs/noise_injection.md`; unchanged here.

## Next step after review

Implementation order, one reviewed diff at a time: (1) `MR` fix + docstring in
`injection.py` with a regression test on a generated circuit; (2) bit-flip
constructor + metadata + decoder-free validations; (3) phase-flip; (4) Shor.
Plus the stale-CLAUDE.md update, batched with (1).
