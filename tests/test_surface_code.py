"""Tests for the distance-3 rotated surface code constructor."""

from __future__ import annotations

import numpy as np
import stim

from nonmarkov_qec.codes.surface_code import surface_code
from nonmarkov_qec.noise.injection import inject_dephasing_noise


def test_metadata() -> None:
    """surface_code(3) returns correct metadata and n_cycles = n_ticks + 1."""
    code = surface_code(3)
    assert code.n_qubits == 17
    assert code.data_qubits == tuple(range(9))
    assert code.ancilla_qubits == tuple(range(9, 17))
    assert len(code.ancilla_qubits) == 8  # 4 X-checks + 4 Z-checks
    assert code.rounds == 3
    assert code.distance == 3
    assert code.circuit.num_observables == 1
    assert code.circuit.num_detectors == 24  # 4 + 8 + 8 + 4

    flat = code.circuit.flattened()
    n_ticks = sum(
        1
        for item in flat
        if isinstance(item, stim.CircuitInstruction) and item.name == "TICK"
    )
    assert code.n_cycles == n_ticks + 1


def test_no_preexisting_noise() -> None:
    """Generated circuit has no error channels; injection is the sole noise source."""
    code = surface_code(3)
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
    code = surface_code(3)
    sampler = code.circuit.compile_detector_sampler()
    det_samples, obs_samples = sampler.sample(shots=256, separate_observables=True)
    assert not det_samples.any(), (
        "expected all-zero detectors from noiseless surface code circuit — "
        "check detector wiring"
    )
    assert not obs_samples.any(), (
        "expected logical observable = 0 for noiseless |+_L⟩ — "
        "check observable wiring"
    )
    code.circuit.detector_error_model(decompose_errors=False)


def test_distance_is_three() -> None:
    """Schedule-acceptance gate: DEM under flat Z-dephasing has graphlike distance 3.

    A noiseless circuit has an empty DEM; inject flat Z-noise (p_0=0.001,
    m=0, sigma=1) to populate error mechanisms before computing the DEM.
    """
    code = surface_code(rounds=3)
    noisy = inject_dephasing_noise(
        code.circuit,
        trajectories=np.zeros((code.n_qubits, code.n_cycles)),
        p_0=0.001,
        m=0.0,
        sigma=1.0,
        p_meas=0.0,
    )
    dem = noisy.detector_error_model(decompose_errors=True)
    assert len(dem.shortest_graphlike_error()) == 3


def _inject_single_fault_at_tick(
    code_circuit: stim.Circuit,
    fault_name: str,
    qubit: int,
    tick_index: int,
) -> stim.Circuit:
    """Return a copy of code_circuit with one deterministic fault after the
    (tick_index)-th TICK (0-indexed).

    tick_index=0 is after the single prep TICK (R + H), placing the fault at
    the very start of the first syndrome round when data qubits are in |+⟩^9.
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
    """Each Z fault fires exactly the X-stabilizers it anticommutes with.

    rounds=1 detector column layout:
        0: round-0 anc9  (X_{0,1})
        1: round-0 anc10 (X_{1,2,4,5})
        2: round-0 anc11 (X_{3,4,6,7})
        3: round-0 anc12 (X_{7,8})
        4: final anc9  ⊕ MX{0,1}
        5: final anc10 ⊕ MX{1,2,4,5}
        6: final anc11 ⊕ MX{3,4,6,7}
        7: final anc12 ⊕ MX{7,8}

    Fault injected after the prep TICK (tick_index=0): data in |+⟩^9 at the
    start of round 0.  The fault is static and rounds=1, so each final
    detector cancels (last-round ancilla XOR readout parity = 0), and only
    the round-0 syndrome columns fire.
    """
    code = surface_code(1)

    def fired(circuit: stim.Circuit) -> set[int]:
        det = circuit.compile_detector_sampler().sample(shots=256)
        return {int(c) for c in np.flatnonzero(det.any(axis=0))}

    # q0 is in X_{0,1} (anc9, col 0) only
    z0 = _inject_single_fault_at_tick(code.circuit, "Z_ERROR", 0, tick_index=0)
    assert fired(z0) == {0}, f"Z on q0 should fire col 0 only; got {fired(z0)}"

    # q4 is in X_{1,2,4,5} (anc10, col 1) and X_{3,4,6,7} (anc11, col 2)
    z4 = _inject_single_fault_at_tick(code.circuit, "Z_ERROR", 4, tick_index=0)
    assert fired(z4) == {1, 2}, f"Z on q4 should fire cols 1,2; got {fired(z4)}"


def test_z_visible_under_injection() -> None:
    """Z dephasing IS detected by the surface code's X-check stabilizers.

    Flat trajectory at p_0=0.3 (m=0, sigma=1, p_meas=0) gives a constant
    Z-error rate of 0.3 at every cycle, which should reliably fire X-check
    detectors.
    """
    code = surface_code(3)
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
        "X-stabilizer checks in the surface code"
    )
