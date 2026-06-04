"""Bit-flip (Z-basis repetition) code constructor."""

from __future__ import annotations

import stim

from nonmarkov_qec.codes.base import CodeCircuit


def bit_flip_code(distance: int, rounds: int) -> CodeCircuit:
    """Return a noise-free CodeCircuit for the Z-basis repetition (bit-flip) code.

    Uses stim.Circuit.generated("repetition_code:memory") with all built-in
    noise args zeroed out.  Layout is extracted from the generated circuit
    without assuming any qubit-index parity: data_qubits are the targets of
    the final M instruction; ancilla_qubits are the union of targets across
    all MR instructions in the flattened circuit.

    Parameters
    ----------
    distance
        Code distance; must be >= 3 and odd.
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
        If distance < 3, distance is even, or rounds < 1.
    """
    if distance < 3 or distance % 2 == 0:
        raise ValueError(f"distance must be >= 3 and odd, got {distance}")
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")

    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=0,
        before_round_data_depolarization=0,
        before_measure_flip_probability=0,
        after_reset_flip_probability=0,
    )

    flat = circuit.flattened()

    ancilla_set: set[int] = set()
    last_m_targets: list[int] = []
    n_ticks: int = 0

    for item in flat:
        if isinstance(item, stim.CircuitRepeatBlock):
            raise AssertionError("unexpected CircuitRepeatBlock after flattened()")
        if item.name == "TICK":
            n_ticks += 1
        elif item.name == "MR":
            for t in item.targets_copy():
                if t.is_qubit_target:
                    ancilla_set.add(t.value)
        elif item.name == "M":
            last_m_targets = [t.value for t in item.targets_copy() if t.is_qubit_target]

    return CodeCircuit(
        circuit=circuit,
        data_qubits=tuple(sorted(last_m_targets)),
        ancilla_qubits=tuple(sorted(ancilla_set)),
        n_qubits=circuit.num_qubits,
        n_cycles=n_ticks + 1,
        rounds=rounds,
        distance=distance,
    )
