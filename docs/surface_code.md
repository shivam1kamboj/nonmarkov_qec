# Surface code: design note (distance-3 rotated patch, X-memory)

This note specifies the distance-3 rotated surface code patch, its stabilizer
layout, the per-round CX schedule, detector/observable wiring, and how the Stim
detector error model (DEM) maps onto a PyMatching decoding graph. It is the
analytic specification; implementation in `src/nonmarkov_qec/codes/surface_code.py`
and the decoder in `src/nonmarkov_qec/decoders/` follow only after review.

This is the first code in the project with a *real decoder*. The small codes were
decoder-free (contrast tests on fired-detector sets). Here the deliverable is a
working memory experiment whose logical error rate can be measured, which is what
the headline threshold experiment consumes.

## 1. Two decisions, stated up front

**Patch: rotated d=3.** 9 data + 8 ancilla = 17 qubits, vs 13 data + 12 ancilla
= 25 for the unrotated d=3. Same distance, same logical protection, fewer qubits
and stabilizers. It is also what Stim's `surface_code:rotated_memory_x` generator
produces, so we can cross-validate our hand-built version against a reference DEM.

**Experiment: X-memory.** Our injected noise is dephasing (`Z_ERROR`). Physical
Z-errors anticommute with X-type stabilizers, so the X-stabilizers are the checks
that catch our noise, and the logical failure mode is a vertical Z-string forming
logical Z-bar. To make that failure *observable* we prepare an eigenstate of the
logical operator it anticommutes with — X-bar — i.e. the |+_L> state, and read out
in the X basis.

This mirrors the Shor experiment exactly: there we prepared the Z-bar eigenstate
|0_L> (with Z-bar = X^9) and let the Z-noise-induced logical (X-bar, the Z-type
logical) threaten it. Same principle, relabeled: prepare the eigenstate of the
logical that anticommutes with the noise-induced logical. Had we reflexively run
`memory_z`, dephasing would be the invisible null all over again (the bit-flip
plumbing case).

**On the word "single-shot."** v1 runs `r` rounds of *noisy* syndrome extraction
followed by a *noiseless* transversal data-qubit readout that reconstructs the
final stabilizer layer perfectly. This is the standard memory experiment with a
clean time boundary. It is **not** "single-shot QEC" in the Bombin sense (which
needs metachecks / 3D codes); the decode is still minimum-weight perfect matching
on the full (r+1)-layer space-time graph. The term is avoided in the rest of this
note.

## 2. Qubit layout

Data qubits 0–8 on a 3x3 grid; ancilla qubits 9–16, one per stabilizer.
`n_qubits = 17`.

```
        col0   col1   col2
row0     q0 --- q1 --- q2
         |      |      |
row1     q3 --- q4 --- q5
         |      |      |
row2     q6 --- q7 --- q8
```

Ancilla assignment (one ancilla per stabilizer; reuse is a later optimization):

| Ancilla | Type | Data qubits            | Weight | Location          |
|---------|------|------------------------|--------|-------------------|
| 9       | X    | q0, q1, q3, q4         | 4      | top-left bulk     |
| 10      | X    | q4, q5, q7, q8         | 4      | bottom-right bulk |
| 11      | X    | q2, q5                 | 2      | right boundary    |
| 12      | X    | q3, q6                 | 2      | left boundary     |
| 13      | Z    | q1, q2, q4, q5         | 4      | top-right bulk    |
| 14      | Z    | q3, q4, q6, q7         | 4      | bottom-left bulk  |
| 15      | Z    | q0, q1                 | 2      | top boundary      |
| 16      | Z    | q7, q8                 | 2      | bottom boundary   |

4 X-stabilizers + 4 Z-stabilizers = 8 stabilizers on 9 data qubits → 1 logical
qubit (9 − 8 = 1). All X-stabilizers commute with all Z-stabilizers (each pair
overlaps on an even number of data qubits — verified pairwise; the only nonzero
overlaps are weight 2).

## 3. Logical operators and distance

- **Logical X-bar = X0 X1 X2** (top row). Commutes with every Z-stabilizer
  (even overlap with each), not in the stabilizer group.
- **Logical Z-bar = Z0 Z3 Z6** (left column). Commutes with every X-stabilizer,
  anticommutes with X-bar on q0 (single-qubit overlap) as a logical pair must.

Both have weight 3 = d. The Z-distance (minimum-weight logical Z operator) is 3:
dephasing needs a full column of 3 Z-errors to produce an undetected logical
Z-bar. This weight-3 Z-distance is the protection the experiment measures.

Distance is not assumed — it is a validation gate (Section 7): the implementation
must confirm `len(dem.shortest_graphlike_error()) == 3` for the X-detector graph.

## 4. Stabilizer extraction circuits

Ancilla conventions carried over from the small codes:
- **Z-stabilizer** (ancilla in |0>): `CX data -> ancilla` for each data qubit,
  then `MR ancilla`. Measures the Z-parity.
- **X-stabilizer** (ancilla in |+>): `H ancilla`; `CX ancilla -> data` for each
  data qubit; `H ancilla`; `MR ancilla`. Phase kickback measures the X-parity.

Dephasing lands only at CX moments (no idle dephasing in v1), TICK-delimited
rounds, trajectory rows cover all 17 physical qubits including ancillas, so
metadata reports `n_qubits = 17` (consistent with the injection layer's shape
contract).

## 5. CX schedule and hook errors (the decoder-critical part)

This is the section the determinism test cannot police. A noiseless circuit is
all-zero detectors regardless of CX order; only a *fault-sensitive* check catches
a bad schedule. The risk is a **hook error**: a single ancilla fault midway
through its 4 CXs propagates to a weight-2 data error. If that weight-2 error is
oriented *along* the logical operator, it combines with one more fault to cause a
logical error — effectively reducing the distance from 3 to 2. The schedule must
orient hook errors *perpendicular* to the logical they could otherwise feed.

The distance-preserving CX order for the rotated surface code is well established
(Tomita & Svore 2014; the Google "exponential suppression" schedule) and is what
Stim's generator emits. Rather than re-derive it from memory and risk an aligned
schedule, the implementation fixes the order via a **discovery spike** (consistent
with the project's "verify API facts empirically" rule):

> **Implementation Step 0 — schedule spike.** Generate
> `stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=3)`,
> read off its per-round CX time-ordering and qubit coordinates, and adopt that
> ordering for the hand-built patch (remapped onto our 0–16 indexing). Hand-built
> remains the deliverable — the spike supplies only the *order*, not the circuit.

The order is then frozen in the note's revision and the constructor implements it
explicitly (4 CX time-steps per stabilizer type, interleaved so X- and
Z-stabilizer extraction overlap where the schedule allows without targeting a
data qubit twice in one moment).

**Acceptance gate for the schedule:** `shortest_graphlike_error()` on the DEM must
have length 3. If the hand-built order yields 2, the schedule is wrong (aligned
hooks) and is corrected before proceeding. This is the test that would have caught
a bad order; it is mandatory, not optional.

## 6. Detector and observable wiring

Preparation: `R` all 17; `H` on the 9 data qubits → |+>^9 = |+_L>, a +1
eigenstate of all four X-stabilizers and of X-bar. It is *not* an eigenstate of
the Z-stabilizers (their round-1 outcomes are random).

- **Round 1 detectors — X-stabilizers only.** Each X-ancilla (9–12) is compared
  to its known +1 (deterministic 0 on a clean encode). Z-stabilizers get **no**
  round-1 detector: |+_L> is not a Z-eigenstate, so their first outcomes are
  random and carry no syndrome information yet.
- **Rounds 2..r detectors — all 8 stabilizers.** Each ancilla compared to its
  own previous-round outcome (XOR). From round 2 the Z-stabilizers are
  well-defined relative to round 1 and contribute.
- **Final layer — noiseless X-basis readout.** `MX` on all 9 data qubits. Each
  X-stabilizer is reconstructed as the parity of its data qubits' `MX` outcomes
  and compared to that stabilizer's last ancilla round (final X-detectors). No
  final Z-detectors — X-basis readout cannot reconstruct Z-parities.
- **Observable:** `OBSERVABLE_INCLUDE` = parity of `MX(q0), MX(q1), MX(q2)` =
  logical X-bar. A logical Z-bar error (the dephasing failure mode) anticommutes
  with X-bar and flips this readout.

Record-index arithmetic (the Shor-style trap) is derived analytically at
implementation time and pinned by the fault-contrast test, not eyeballed.

## 7. Stim DEM → PyMatching graph

The decoder path:

1. Build the noisy circuit (inject dephasing via the existing injection layer).
2. `dem = circuit.detector_error_model(decompose_errors=True)`. For a pure-Z
   (dephasing) channel the X-detector graph is **graphlike** — every error
   mechanism flips at most two detectors — so MWPM applies directly without
   hyperedge decomposition trouble. (`decompose_errors=True` is still set so any
   stray correlated mechanism is split rather than silently dropped.)
3. `matching = pymatching.Matching.from_detector_error_model(dem)`.
4. Sample detector shots + observable flips from the circuit; `matching.decode`
   each shot; compare the predicted observable flip to the actual; the logical
   error rate is the mismatch fraction.

Because v1 injects only Z-errors, only the X-stabilizer subgraph carries signal;
the Z-stabilizer detectors will be event-free in a pure-dephasing run. This is
expected and is itself a check (a Z-detector firing under pure-Z injection would
indicate a wiring bug). The matching graph's boundary edges correspond to the two
Z-boundaries (top/bottom in this layout) where a Z-string can terminate.

## 8. v1 scope and deferrals

In scope for v1:
- Rotated d=3 patch, X-memory, variable `rounds`.
- Noiseless final readout (clean time boundary).
- MWPM decode via PyMatching from the Stim DEM.
- Logical error rate as a measured output.

Deferred:
- **d=5 patch** — needed for the threshold experiment (two distances to locate a
  crossing). The d=3 constructor should be written so the layout generalizes, but
  d=5 is a separate step.
- **Noisy final readout / repeated-readout time boundary** — v1's perfect readout
  is the standard simplification; a fully noisy boundary is a later refinement.
- **Z-memory variant** — not needed for dephasing; trivial to add later by
  symmetry if a bit-flip study is ever wanted.
- **Ancilla reuse / qubit-count optimization** — one ancilla per stabilizer in v1
  for transparency.

## 9. Validation plan

Decoder-aware, unlike the small codes:

1. **Metadata** — 17 qubits, 8 ancillas (4 X + 4 Z), 1 observable, `n_cycles`
   consistency.
2. **No pre-existing noise** — bare circuit has no error instructions.
3. **Noiseless determinism** — all detectors 0, observable never flips, DEM builds.
4. **Distance gate** — `len(dem.shortest_graphlike_error()) == 3`. The schedule
   acceptance test of Section 5.
5. **Single-fault contrast** — a deterministic Z on a data qubit fires exactly the
   X-stabilizer(s) containing it and no Z-stabilizers; exact fired-detector sets
   asserted (Shor-style), including a qubit shared by two X-stabilizers.
6. **Decoder smoke / sub-threshold suppression** — at a low physical error rate,
   the logical error rate is well below the physical rate, and (once d=5 exists)
   decreases with distance. This is the end-to-end check that the matching graph
   is wired correctly; a mis-wired observable typically pins the logical rate near
   0.5.

Implementation order (to be confirmed after this note is signed off): schedule
spike → constructor → metadata/determinism/distance tests → fault-contrast test →
PyMatching decoder wiring → decoder smoke test. One step at a time, as usual.
