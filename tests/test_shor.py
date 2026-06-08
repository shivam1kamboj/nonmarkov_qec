"""Tests for the Shor 9-qubit code constructor."""

from __future__ import annotations

import numpy as np
import stim

from nonmarkov_qec.codes.shor import shor_code
from nonmarkov_qec.noise.injection import inject_dephasing_noise


def test_metadata() -> None:
    """shor_code(3) returns correct metadata."""
    code = shor_code(3)
    assert code.n_qubits == 17
    assert code.data_qubits == tuple(range(9))
    assert code.ancilla_qubits == tuple(range(9, 17))
    assert len(code.ancilla_qubits) == 8  # 6 Z-checks + 2 X-checks
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
    code = shor_code(3)
    for item in code.circuit.flattened():
        if isinstance(item, stim.CircuitInstruction):
            assert not item.name.endswith("_ERROR"), (
                f"found preexisting error channel: {item.name}"
            )
            assert not item.name.startswith("DEPOLARIZE"), (
                f"found preexisting depolarizing channel: {item.name}"
            )


def test_noiseless_determinism() -> None:
    """Noiseless circuit produces all-zero detectors and observable never flips."""
    code = shor_code(3)
    sampler = code.circuit.compile_detector_sampler()
    det_samples, obs_samples = sampler.sample(shots=256, separate_observables=True)
    assert not det_samples.any(), (
        "expected all-zero detectors from noiseless Shor circuit — "
        "check detector wiring"
    )
    assert not obs_samples.any(), (
        "expected logical observable = 0 for noiseless |0_L⟩ — "
        "check observable wiring"
    )
    # Validates well-formed stabilizer structure.
    code.circuit.detector_error_model(decompose_errors=False)


def _inject_single_fault_at_tick(
    code_circuit: stim.Circuit,
    fault_name: str,
    qubit: int,
    tick_index: int,
) -> stim.Circuit:
    """Return a copy of code_circuit with one deterministic fault after the
    (tick_index)-th TICK (0-indexed).

    tick_index=2 is after the full Shor encoding (3 encoder TICKs at indices
    0, 1, 2), placing the fault at the very start of the first syndrome round
    when data qubits are in the encoded codeword.
    """
    out = stim.Circuit()
    ticks_seen = 0
    for item in code_circuit.flattened():
        if isinstance(item, stim.CircuitRepeatBlock):
            raise AssertionError("unexpected CircuitRepeatBlock after flattened()")
        out.append(item)
        if item.name == "TICK":
            if ticks_seen == tick_index:
                out.append(fault_name, [qubit], 1.0)
            ticks_seen += 1
    return out


def test_single_fault_contrast() -> None:
    """Each single-qubit fault fires EXACTLY the stabilizers it anticommutes with.

    rounds=1 detector column layout:
        0: Z0Z1   1: Z1Z2   2: Z3Z4   3: Z4Z5   4: Z6Z7   5: Z7Z8
        6: X0X1X2X3X4X5 (round 1)    7: X3X4X5X6X7X8 (round 1)
        8: final X0..X5 detector     9: final X3..X8 detector

    Fault injected after the 3rd encoder TICK (tick_index=2): on the encoded
    |0_L> at the start of round 1.  Because the fault is static and rounds=1,
    each final detector cancels (last-round ancilla XOR readout parity = 0),
    so a fault fires ONLY its round-1 syndrome column(s).  Asserting the exact
    fired set therefore checks the ancilla->stabilizer wiring, the X-check
    staggering, and the final-detector record arithmetic all at once.
    """
    code = shor_code(1)

    def fired(circuit: stim.Circuit) -> set[int]:
        # Fault prob = 1.0 on a Clifford stabilizer circuit -> deterministic,
        # so the set of columns that fire is identical across all shots.
        det = circuit.compile_detector_sampler().sample(shots=256)
        return {int(c) for c in np.flatnonzero(det.any(axis=0))}

    # --- qubit 0: in Z0Z1 (col 0) and X0..X5 (col 6) only ---
    z0 = _inject_single_fault_at_tick(code.circuit, "Z_ERROR", 0, tick_index=2)
    x0 = _inject_single_fault_at_tick(code.circuit, "X_ERROR", 0, tick_index=2)
    assert fired(z0) == {6}, (
        f"Z on q0 should fire only the X0..X5 check (col 6); got {fired(z0)}"
    )
    assert fired(x0) == {0}, (
        f"X on q0 should fire only the Z0Z1 check (col 0); got {fired(x0)}"
    )

    # --- qubit 3: the overlap qubit, in BOTH weight-6 X-checks (cols 6 and 7)
    #     and in Z3Z4 (col 2).  This is the case that proves the staggered
    #     X-check schedule wires q3 into both phase-flip stabilizers. ---
    z3 = _inject_single_fault_at_tick(code.circuit, "Z_ERROR", 3, tick_index=2)
    x3 = _inject_single_fault_at_tick(code.circuit, "X_ERROR", 3, tick_index=2)
    assert fired(z3) == {6, 7}, (
        f"Z on q3 (in both X-checks) should fire cols 6 and 7; got {fired(z3)}"
    )
    assert fired(x3) == {2}, (
        f"X on q3 should fire only the Z3Z4 check (col 2); got {fired(x3)}"
    )


def test_z_visible_under_injection() -> None:
    """Z dephasing IS detected by the Shor code's X-check stabilizers.

    Uses the same flat-trajectory / p_0=0.3 pattern as test_phase_flip.py:
    trajectories=zeros gives a constant Z-error rate of p_0 at every cycle
    (alpha=0 when m=0), which should reliably fire X-check detectors.
    """
    code = shor_code(3)
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
        "expected some detector events; Z dephasing should be visible to "
        "X-stabilizer checks in the Shor code"
    )
