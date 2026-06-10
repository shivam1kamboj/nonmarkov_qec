"""Distance-3 rotated surface code constructor — variable rounds, X-basis memory."""

from __future__ import annotations

import stim

from nonmarkov_qec.codes.base import CodeCircuit


def surface_code(rounds: int) -> CodeCircuit:
    """Return a noise-free CodeCircuit for the distance-3 rotated surface code.

    Hand-built X-basis memory circuit.  Encodes the logical |+_L⟩ state and
    measures four weight-2/4 X-stabilizers (ancillas 9-12) and four weight-2/4
    Z-stabilizers (ancillas 13-16) each round.

    Layout
    ------
    Data qubits    : 0-8, row-major 3x3::

        0  1  2
        3  4  5
        6  7  8

    Ancilla qubits
        X-type (Hadamard-prepped, ancilla-as-control):
            9  : X_{0,1}
            10 : X_{1,2,4,5}
            11 : X_{3,4,6,7}
            12 : X_{7,8}
        Z-type (data-as-control, ancilla-as-target):
            13 : Z_{0,1,3,4}
            14 : Z_{4,5,7,8}
            15 : Z_{3,6}
            16 : Z_{2,5}

    n_qubits = 17

    Circuit vocabulary
    ------------------
    R, H, CX, MR, MX, TICK, DETECTOR, OBSERVABLE_INCLUDE only.

    Encoding for |+_L⟩
    -------------------
    Reset all qubits, then Hadamard all 9 data qubits::

        R  0..16
        H  0 1 2 3 4 5 6 7 8   TICK

    Per-round syndrome structure
    ----------------------------
    Four frozen CX layers (A-D) implement all eight stabilizer checks in
    parallel.  Ancilla-as-control for X checks; data-as-control for Z checks::

        H 9 10 11 12                                         TICK
        CX 9 1  11 7  10 5  6 15  4 13  8 14                TICK   # A
        CX 9 0  11 6  10 4  3 15  1 13  5 14                TICK   # B
        CX 11 4  10 2  12 8  3 13  7 14  5 16               TICK   # C
        CX 11 3  10 1  12 7  0 13  4 14  2 16               TICK   # D
        H 9 10 11 12                                         TICK
        MR 9 10 11 12 13 14 15 16   <detectors>             TICK

    Gives 7 TICKs per round; 1 encoding TICK; n_cycles = 2 + 7·rounds.

    Detector record indices
    -----------------------
    Immediately after MR 9..16 in round t:
      rec[-8] = anc9,  rec[-7] = anc10, …,  rec[-1] = anc16
      prev round (t≥1):  rec[-16] = anc9_prev, …, rec[-9] = anc16_prev

    Round 0 emits four X-ancilla detectors only (Z-ancilla outcomes are
    random on |+⟩^9 initialization; their XOR with round 1 is deterministic).

    After final MX 0..8 (9 records appended):
      rec[-9] = MX(q0), …, rec[-1] = MX(q8)
      last-round MR shifted by 9:
        anc9_last = rec[-17], anc10_last = rec[-16],
        anc11_last = rec[-15], anc12_last = rec[-14]

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
    # Reset all 17 qubits and prepare data in |+⟩ for X-basis memory
    # ------------------------------------------------------------------
    c.append("R", list(range(17)))
    c.append("H", list(range(9)))
    c.append("TICK", [])

    # ------------------------------------------------------------------
    # Syndrome rounds
    # ------------------------------------------------------------------
    for t in range(rounds):
        # X-ancilla prep: rotate into |+⟩
        c.append("H", [9, 10, 11, 12])
        c.append("TICK", [])

        # Layer A
        c.append("CX", [9, 1, 11, 7, 10, 5, 6, 15, 4, 13, 8, 14])
        c.append("TICK", [])

        # Layer B
        c.append("CX", [9, 0, 11, 6, 10, 4, 3, 15, 1, 13, 5, 14])
        c.append("TICK", [])

        # Layer C
        c.append("CX", [11, 4, 10, 2, 12, 8, 3, 13, 7, 14, 5, 16])
        c.append("TICK", [])

        # Layer D
        c.append("CX", [11, 3, 10, 1, 12, 7, 0, 13, 4, 14, 2, 16])
        c.append("TICK", [])

        # X-ancilla unprep: rotate back to Z basis
        c.append("H", [9, 10, 11, 12])
        c.append("TICK", [])

        # Measure-and-reset all 8 ancillas; records: anc9…anc16
        c.append("MR", [9, 10, 11, 12, 13, 14, 15, 16])
        # Immediately after MR: rec[-8]=anc9, rec[-7]=anc10, …, rec[-1]=anc16

        if t == 0:
            # Round 0: |+⟩^9 is a +1 eigenstate of all X-stabilizers → det = 0.
            # Z-ancilla outcomes are random on this state; omit to preserve
            # noiseless determinism.  rec[i-8] for i=0..3 → anc9..anc12.
            for i in range(4):
                c.append("DETECTOR", [stim.target_rec(i - 8)])
        else:
            # Rounds 1…r-1: XOR current with previous round.
            # rec[i-8]  = ancilla i this round  (i=0 → anc9, i=7 → anc16)
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
    # Last-round MR shifted by 9:
    #   anc9_last=rec[-17], anc10_last=rec[-16], anc11_last=rec[-15], anc12_last=rec[-14]

    # Final detector for anc9 — X_{0,1}:  anc9_last ⊕ MX(q0) ⊕ MX(q1)
    c.append(
        "DETECTOR",
        [
            stim.target_rec(-17),  # anc9_last
            stim.target_rec(-9),   # MX(q0)
            stim.target_rec(-8),   # MX(q1)
        ],
    )

    # Final detector for anc10 — X_{1,2,4,5}:  anc10_last ⊕ MX(q1) ⊕ MX(q2) ⊕ MX(q4) ⊕ MX(q5)
    c.append(
        "DETECTOR",
        [
            stim.target_rec(-16),  # anc10_last
            stim.target_rec(-8),   # MX(q1)
            stim.target_rec(-7),   # MX(q2)
            stim.target_rec(-5),   # MX(q4)
            stim.target_rec(-4),   # MX(q5)
        ],
    )

    # Final detector for anc11 — X_{3,4,6,7}:  anc11_last ⊕ MX(q3) ⊕ MX(q4) ⊕ MX(q6) ⊕ MX(q7)
    c.append(
        "DETECTOR",
        [
            stim.target_rec(-15),  # anc11_last
            stim.target_rec(-6),   # MX(q3)
            stim.target_rec(-5),   # MX(q4)
            stim.target_rec(-3),   # MX(q6)
            stim.target_rec(-2),   # MX(q7)
        ],
    )

    # Final detector for anc12 — X_{7,8}:  anc12_last ⊕ MX(q7) ⊕ MX(q8)
    c.append(
        "DETECTOR",
        [
            stim.target_rec(-14),  # anc12_last
            stim.target_rec(-2),   # MX(q7)
            stim.target_rec(-1),   # MX(q8)
        ],
    )

    # Observable: X̄ = X_0 X_3 X_6 (left column)
    c.append(
        "OBSERVABLE_INCLUDE",
        [
            stim.target_rec(-9),  # MX(q0)
            stim.target_rec(-6),  # MX(q3)
            stim.target_rec(-3),  # MX(q6)
        ],
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
