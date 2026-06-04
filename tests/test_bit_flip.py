"""Tests for the bit-flip code constructor."""

from __future__ import annotations

import numpy as np
import stim

from nonmarkov_qec.codes.bit_flip import bit_flip_code
from nonmarkov_qec.noise.injection import inject_dephasing_noise


def test_metadata() -> None:
    """bit_flip_code(3, 3) returns correct metadata."""
    code = bit_flip_code(3, 3)
    assert code.n_qubits == 5
    assert code.data_qubits == (0, 2, 4)
    assert code.ancilla_qubits == (1, 3)
    assert code.n_cycles == 10
    assert code.rounds == 3
    assert code.distance == 3
    assert code.circuit.num_observables == 1
    assert code.circuit.num_detectors > 0


def test_no_preexisting_noise() -> None:
    """Generated circuit has no error channels; injection is the sole noise source."""
    code = bit_flip_code(3, 3)
    for item in code.circuit.flattened():
        if isinstance(item, stim.CircuitInstruction):
            assert not item.name.endswith("_ERROR"), (
                f"found preexisting error channel: {item.name}"
            )
            assert not item.name.startswith("DEPOLARIZE"), (
                f"found preexisting depolarizing channel: {item.name}"
            )


def test_z_blind_under_injection() -> None:
    """Z dephasing is invisible to the ZZ-stabilizer repetition code."""
    code = bit_flip_code(3, 3)
    noisy = inject_dephasing_noise(
        code.circuit,
        trajectories=np.zeros((code.n_qubits, code.n_cycles)),
        p_0=0.3,
        m=0.0,
        sigma=1.0,
        p_meas=0.0,
    )
    det_samples = noisy.compile_detector_sampler().sample(shots=1024)
    assert not det_samples.any(), (
        "expected all-zero detectors; Z dephasing is invisible to ZZ-stabilizer code"
    )


def _inject_single_fault(
    code_circuit: stim.Circuit, fault_name: str, qubit: int
) -> stim.Circuit:
    """Return a copy of code_circuit with one fault inserted after the first TICK."""
    out = stim.Circuit()
    first_tick_seen = False
    for item in code_circuit.flattened():
        if isinstance(item, stim.CircuitRepeatBlock):
            raise AssertionError("unexpected CircuitRepeatBlock after flattened()")
        out.append(item)
        if not first_tick_seen and item.name == "TICK":
            first_tick_seen = True
            out.append(fault_name, [qubit], 1.0)
    return out


def test_single_fault_contrast() -> None:
    """X fault fires detectors; Z fault fires none — detectors are live, Z is blind."""
    code = bit_flip_code(3, 3)
    qubit = code.data_qubits[0]

    x_circuit = _inject_single_fault(code.circuit, "X_ERROR", qubit)
    z_circuit = _inject_single_fault(code.circuit, "Z_ERROR", qubit)

    x_det = x_circuit.compile_detector_sampler().sample(shots=256)
    z_det = z_circuit.compile_detector_sampler().sample(shots=256)

    assert x_det.any(), "X fault produced no detector events — detectors may be dead"
    assert not z_det.any(), (
        "Z fault triggered a detector — Z dephasing should be invisible to bit-flip code"
    )
