"""Phase-flip (X-basis repetition) code constructor — distance 3, variable rounds."""

from __future__ import annotations

import stim

from nonmarkov_qec.codes.base import CodeCircuit


def phase_flip_code(rounds: int) -> CodeCircuit:
    """Return a noise-free CodeCircuit for the distance-3 phase-flip code.

    Hand-built X-basis repetition memory circuit.  Encodes the logical |+_L⟩
    state and measures X0X1 (ancilla 3) and X1X2 (ancilla 4) each round.

    Generalisation to arbitrary odd distance is deferred; this constructor is
    distance-3 only.

    Layout
    ------
    Data qubits   : 0, 1, 2
    Ancilla qubits: 3 (checks X0 X1), 4 (checks X1 X2)

    Circuit vocabulary
    ------------------
    R, H, CX, MR, M, TICK, DETECTOR, OBSERVABLE_INCLUDE only.

    TICK placement mirrors Stim's idiom: TICK after each gate layer and after
    each MR (round boundary).

    Per-round syndrome structure
    ----------------------------
    H 3 4 → TICK → CX 3 0 / CX 4 1 → TICK → CX 3 1 / CX 4 2 → TICK →
    H 3 4 → MR 3 4 → detectors → TICK

    Detector record indices
    -----------------------
    MR 3 4 produces two records per round: ancilla 3 then ancilla 4.
    Immediately after MR 3 4 in round t:
      rec[-2] = anc3 round t,  rec[-1] = anc4 round t
      rec[-4] = anc3 round t-1, rec[-3] = anc4 round t-1   (t >= 1)
    After the final M 0 1 2 (no measurements between last MR and M):
      rec[-3] = data0, rec[-2] = data1, rec[-1] = data2
      rec[-5] = anc3 last round, rec[-4] = anc4 last round

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
    # Encode: data qubits 0 1 2 → |+++⟩ = |+_L⟩; ancillas reset by R
    # ------------------------------------------------------------------
    c.append("R", [0, 1, 2, 3, 4])
    c.append("H", [0, 1, 2])
    c.append("TICK", [])

    # ------------------------------------------------------------------
    # Syndrome rounds
    # ------------------------------------------------------------------
    for t in range(rounds):
        # Rotate ancillas into |+⟩ (first round: R above put them in |0⟩;
        # subsequent rounds: previous MR reset them to |0⟩)
        c.append("H", [3, 4])
        c.append("TICK", [])

        # CX layer 1: ancilla as control
        c.append("CX", [3, 0, 4, 1])
        c.append("TICK", [])

        # CX layer 2
        c.append("CX", [3, 1, 4, 2])
        c.append("TICK", [])

        # Rotate ancillas back to Z basis and measure-and-reset
        c.append("H", [3, 4])
        c.append("MR", [3, 4])  # records: anc3, then anc4

        if t == 0:
            # First round: |+_L⟩ is a +1 eigenstate of X0X1 and X1X2 → det = 0
            c.append("DETECTOR", [stim.target_rec(-2)])  # anc3 round 0
            c.append("DETECTOR", [stim.target_rec(-1)])  # anc4 round 0
        else:
            # Subsequent rounds: XOR current with previous
            c.append("DETECTOR", [stim.target_rec(-4), stim.target_rec(-2)])  # anc3 t XOR t-1
            c.append("DETECTOR", [stim.target_rec(-3), stim.target_rec(-1)])  # anc4 t XOR t-1

        c.append("TICK", [])  # round boundary

    # ------------------------------------------------------------------
    # Readout: rotate data to Z basis and measure
    # ------------------------------------------------------------------
    c.append("H", [0, 1, 2])
    c.append("M", [0, 1, 2])  # records: data0, data1, data2

    # Final detectors: last-round ancilla parity vs data parity
    # After M 0 1 2: rec[-3]=data0, rec[-2]=data1, rec[-1]=data2
    #                rec[-5]=anc3_last, rec[-4]=anc4_last
    c.append(
        "DETECTOR",
        [stim.target_rec(-5), stim.target_rec(-3), stim.target_rec(-2)],
    )  # X0X1: anc3_last XOR data0 XOR data1
    c.append(
        "DETECTOR",
        [stim.target_rec(-4), stim.target_rec(-2), stim.target_rec(-1)],
    )  # X1X2: anc4_last XOR data1 XOR data2
    c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-3)], 0)  # X_L = X0

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
        data_qubits=(0, 1, 2),
        ancilla_qubits=(3, 4),
        n_qubits=c.num_qubits,
        n_cycles=n_ticks + 1,
        rounds=rounds,
        distance=3,
    )
