"""Shor 9-qubit code constructor — distance 3, variable rounds."""

from __future__ import annotations

import stim

from nonmarkov_qec.codes.base import CodeCircuit


def shor_code(rounds: int) -> CodeCircuit:
    """Return a noise-free CodeCircuit for the Shor 9-qubit code.

    Hand-built X-basis memory circuit.  Encodes the logical |0_L⟩ state and
    measures six weight-2 Z-stabilizers (ancillas 9-14) and two weight-6
    X-stabilizers (ancillas 15-16) each round.

    Generalisation to arbitrary distance is deferred; this constructor is
    distance-3 only.

    Layout
    ------
    Data qubits    : 0-8, in blocks {0,1,2}, {3,4,5}, {6,7,8}
    Ancilla qubits : 9  (Z0Z1),  10 (Z1Z2),  11 (Z3Z4),  12 (Z4Z5),
                     13 (Z6Z7),  14 (Z7Z8),
                     15 (X0X1X2X3X4X5),  16 (X3X4X5X6X7X8)
    n_qubits = 17

    Circuit vocabulary
    ------------------
    R, H, CX, MR, MX, TICK, DETECTOR, OBSERVABLE_INCLUDE only.

    Encoding for |0_L⟩
    -------------------
    Phase-flip (outer) layer then bit-flip (inner) layer, separated by TICKs::

        CX 0 3 ; CX 0 6                                        TICK
        H 0 ; H 3 ; H 6                                        TICK
        CX 0 1 ; CX 0 2 ; CX 3 4 ; CX 3 5 ; CX 6 7 ; CX 6 8  TICK

    Per-round syndrome structure
    ----------------------------
    Z-checks leg 1 → TICK → Z-checks leg 2 → TICK →
    H 15 16 → TICK →
    CX 15 0, 16 3 → TICK → CX 15 1, 16 4 → TICK →
    CX 15 2, 16 5 → TICK → CX 15 3, 16 6 → TICK →
    CX 15 4, 16 7 → TICK → CX 15 5, 16 8 → TICK →
    H 15 16 → TICK →
    MR 9..16 → detectors → TICK (round boundary)

    Gives 11 TICKs per round; 3 encoding TICKs; n_cycles = 4 + 11*rounds.

    Detector record indices
    -----------------------
    Immediately after MR 9 10 11 12 13 14 15 16 in round t:
      rec[-8] = anc9,  rec[-7] = anc10, …, rec[-2] = anc15, rec[-1] = anc16
      prev round (t>=1): rec[-16] = anc9_prev, …, rec[-9] = anc16_prev

    After final MX 0 1 2 3 4 5 6 7 8 (9 records appended):
      rec[-9] = MX(q0), …, rec[-1] = MX(q8)
      rec[-11] = anc15_last,  rec[-10] = anc16_last

    Parameters
    ----------
    rounds
        Number of syndrome measurement rounds; must be >= 1.

    Returns
    -------
    CodeCircuit
        circuit has no error channels; inject_dephasing_noise is the sole
        noise source.

    Raises
    ------
    ValueError
        If rounds < 1.
    """
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")

    c = stim.Circuit()

    # ------------------------------------------------------------------
    # Reset all 17 qubits
    # ------------------------------------------------------------------
    c.append("R", list(range(17)))

    # ------------------------------------------------------------------
    # Encode |0_L⟩: phase-flip layer (outer), then bit-flip layer (inner)
    # ------------------------------------------------------------------
    # Outer: spread q0 to q3, q6 (no-op for |0⟩ input, but part of general encoder)
    c.append("CX", [0, 3, 0, 6])
    c.append("TICK", [])

    # Hadamard on outer-code representatives → |+⟩|+⟩|+⟩
    c.append("H", [0, 3, 6])
    c.append("TICK", [])

    # Inner: bit-flip each block into (|000⟩+|111⟩)/√2
    c.append("CX", [0, 1, 0, 2, 3, 4, 3, 5, 6, 7, 6, 8])
    c.append("TICK", [])

    # ------------------------------------------------------------------
    # Syndrome rounds
    # ------------------------------------------------------------------
    for t in range(rounds):
        # Z-checks, leg 1: data→ancilla (data is control)
        c.append("CX", [0, 9, 1, 10, 3, 11, 4, 12, 6, 13, 7, 14])
        c.append("TICK", [])

        # Z-checks, leg 2
        c.append("CX", [1, 9, 2, 10, 4, 11, 5, 12, 7, 13, 8, 14])
        c.append("TICK", [])

        # X-checks, prep: rotate ancillas into |+⟩
        c.append("H", [15, 16])
        c.append("TICK", [])

        # X-checks, 6 CX moments — ancilla is control, data is target;
        # staggered so no data qubit appears as target twice in one moment
        c.append("CX", [15, 0, 16, 3])
        c.append("TICK", [])
        c.append("CX", [15, 1, 16, 4])
        c.append("TICK", [])
        c.append("CX", [15, 2, 16, 5])
        c.append("TICK", [])
        c.append("CX", [15, 3, 16, 6])
        c.append("TICK", [])
        c.append("CX", [15, 4, 16, 7])
        c.append("TICK", [])
        c.append("CX", [15, 5, 16, 8])
        c.append("TICK", [])

        # X-checks, unprep: rotate ancillas back to Z basis
        c.append("H", [15, 16])
        c.append("TICK", [])

        # Measure-and-reset all 8 ancillas; records (in order): anc9…anc16
        c.append("MR", [9, 10, 11, 12, 13, 14, 15, 16])
        # Immediately after MR: rec[-8]=anc9, …, rec[-1]=anc16

        if t == 0:
            # Round 1: |0_L⟩ is a +1 eigenstate of all 8 stabilizers → det = 0
            for i in range(8):
                c.append("DETECTOR", [stim.target_rec(i - 8)])
        else:
            # Rounds 2…r: XOR current with previous round
            # rec[i-8]  = ancilla i this round  (i=0→anc9, i=7→anc16)
            # rec[i-16] = ancilla i last round
            for i in range(8):
                c.append(
                    "DETECTOR",
                    [stim.target_rec(i - 16), stim.target_rec(i - 8)],
                )

        c.append("TICK", [])  # round boundary

    # ------------------------------------------------------------------
    # Readout: measure all data qubits in X basis
    # ------------------------------------------------------------------
    c.append("MX", list(range(9)))
    # Records: rec[-9]=MX(q0), rec[-8]=MX(q1), …, rec[-1]=MX(q8)
    # Last-round ancillas shifted by 9: rec[-11]=anc15_last, rec[-10]=anc16_last

    # Final detector A — X0X1X2X3X4X5:
    #   (MX parity of q0..q5) XOR (last-round anc15)
    c.append(
        "DETECTOR",
        [
            stim.target_rec(-11),  # anc15 last round
            stim.target_rec(-9),   # MX(q0)
            stim.target_rec(-8),   # MX(q1)
            stim.target_rec(-7),   # MX(q2)
            stim.target_rec(-6),   # MX(q3)
            stim.target_rec(-5),   # MX(q4)
            stim.target_rec(-4),   # MX(q5)
        ],
    )

    # Final detector B — X3X4X5X6X7X8:
    #   (MX parity of q3..q8) XOR (last-round anc16)
    c.append(
        "DETECTOR",
        [
            stim.target_rec(-10),  # anc16 last round
            stim.target_rec(-6),   # MX(q3)
            stim.target_rec(-5),   # MX(q4)
            stim.target_rec(-4),   # MX(q5)
            stim.target_rec(-3),   # MX(q6)
            stim.target_rec(-2),   # MX(q7)
            stim.target_rec(-1),   # MX(q8)
        ],
    )

    # Observable: Z̄ = X⊗9; XOR of all 9 MX outcomes
    c.append(
        "OBSERVABLE_INCLUDE",
        [stim.target_rec(-9 + i) for i in range(9)],
        0,
    )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    flat = c.flattened()
    n_ticks: int = sum(
        1
        for item in flat
        if isinstance(item, stim.CircuitInstruction) and item.name == "TICK"
    )

    return CodeCircuit(
        circuit=c,
        data_qubits=tuple(range(9)),
        ancilla_qubits=tuple(range(9, 17)),
        n_qubits=c.num_qubits,
        n_cycles=n_ticks + 1,
        rounds=rounds,
        distance=3,
    )
