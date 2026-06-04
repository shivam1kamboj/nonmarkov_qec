"""Tests for the phase-flip code constructor."""

from __future__ import annotations

import numpy as np
import stim

from nonmarkov_qec.codes.phase_flip import phase_flip_code
from nonmarkov_qec.noise.injection import inject_dephasing_noise


def test_metadata() -> None:
    """phase_flip_code(3) returns correct metadata."""
    code = phase_flip_code(3)
    assert code.n_qubits == 5
    assert code.data_qubits == (0, 1, 2)
    assert code.ancilla_qubits == (3, 4)
    assert code.rounds == 3
    assert code.distance == 3
    assert code.circuit.num_observables == 1
    assert code.circuit.num_detectors > 0

    flat = code.circuit.flattened()
    n_ticks = sum(
        1
        for item in flat
        if isinstance(item, stim.CircuitInstruction) and item.name == "TICK"
    )
    assert code.n_cycles == n_ticks + 1


def test_no_preexisting_noise() -> None:
    """Generated circuit has no error channels; injection is the sole noise source."""
    code = phase_flip_code(3)
    for item in code.circuit.flattened():
        if isinstance(item, stim.CircuitInstruction):
            assert not item.name.endswith("_ERROR"), (
                f"found preexisting error channel: {item.name}"
            )
            assert not item.name.startswith("DEPOLARIZE"), (
                f"found preexisting depolarizing channel: {item.name}"
            )


def test_noiseless_determinism() -> None:
    """Noiseless circuit produces all-zero detectors; detector_error_model builds cleanly."""
    code = phase_flip_code(3)
    det_samples = code.circuit.compile_detector_sampler().sample(shots=1024)
    assert not det_samples.any(), (
        "expected all-zero detectors from noiseless phase-flip circuit — "
        "check detector wiring"
    )
    # Validates that the circuit is a well-formed stabilizer circuit with
    # consistent detector structure.
    code.circuit.detector_error_model(decompose_errors=False)


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
    """Z fault fires detectors; X fault fires none — detectors are live, X is blind."""
    code = phase_flip_code(3)
    qubit = code.data_qubits[0]  # data qubit 0

    z_circuit = _inject_single_fault(code.circuit, "Z_ERROR", qubit)
    x_circuit = _inject_single_fault(code.circuit, "X_ERROR", qubit)

    z_det = z_circuit.compile_detector_sampler().sample(shots=256)
    x_det = x_circuit.compile_detector_sampler().sample(shots=256)

    assert z_det.any(), "Z fault produced no detector events — detectors may be dead"
    assert not x_det.any(), (
        "X fault triggered a detector — X errors should be invisible to phase-flip code"
    )


def test_z_visible_under_injection() -> None:
    """Z dephasing IS detected by the phase-flip code (direct contrast to bit-flip)."""
    code = phase_flip_code(3)
    noisy = inject_dephasing_noise(
        code.circuit,
        trajectories=np.zeros((code.n_qubits, code.n_cycles)),
        p_0=0.3,
        m=0.0,
        sigma=1.0,
        p_meas=0.0,
    )
    det_samples = noisy.compile_detector_sampler().sample(shots=1024)
    assert det_samples.any(), (
        "expected some detector events; Z dephasing should be visible to X-stabilizer code"
    )
