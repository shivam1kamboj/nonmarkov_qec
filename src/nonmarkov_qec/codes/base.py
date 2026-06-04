"""Shared return type for QEC code constructors."""

from __future__ import annotations

from dataclasses import dataclass

import stim


@dataclass(frozen=True)
class CodeCircuit:
    """A base QEC circuit (no error channels) plus the metadata the injection
    layer and decoders need.

    Attributes
    ----------
    circuit
        Annotated stim.Circuit with stabilizer structure but NO error channels;
        inject_dephasing_noise is the sole noise source.
    data_qubits, ancilla_qubits
        Stim qubit indices, ascending.
    n_qubits
        Total physical qubits = circuit.num_qubits. Row count the trajectory
        array must match.
    n_cycles
        TICK count of the flattened circuit + 1. Column count the trajectory
        array must match.
    rounds, distance
        Bookkeeping for plot axes and sanity checks.
    """

    circuit: stim.Circuit
    data_qubits: tuple[int, ...]
    ancilla_qubits: tuple[int, ...]
    n_qubits: int
    n_cycles: int
    rounds: int
    distance: int
