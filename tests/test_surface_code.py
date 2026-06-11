"""Tests for the parameterized rotated surface code constructor (any odd d).

Distance gates lead the file: they are the primary acceptance oracle. The
fault-contrast test inverts the empirical Z-fault -> detector incidence and
compares it against an independently derived plaquette enumeration, so the
schedule wiring is checked against geometry without reusing constructor
internals.
"""

from __future__ import annotations

import numpy as np
import pytest
import stim

from nonmarkov_qec.codes.surface_code import surface_code
from nonmarkov_qec.decoders import estimate_logical_error_rate
from nonmarkov_qec.noise.injection import inject_dephasing_noise


# ----------------------------------------------------------------------------
# Independent geometric oracle (does NOT import constructor internals)
# ----------------------------------------------------------------------------
def _expected_x_supports(d: int) -> set[frozenset[int]]:
    """X-stabilizer supports derived purely from plaquette geometry.

    Frame: data(r, c) = r*d + c, row 0 north, row index increases south,
    column index increases east.  X-stabilizers sit on plaquettes (pr, pc) with
    (pr + pc) odd, excluding the left/right code boundaries (pc in {-1, d-1});
    top/bottom boundary plaquettes (pr in {-1, d-1}) appear as weight-2 checks.
    """
    supports: set[frozenset[int]] = set()
    for pr in range(-1, d):
        for pc in range(-1, d):
            if (pr + pc) % 2 != 1:
                continue  # Z-type, not X
            if pc in (-1, d - 1):
                continue  # left/right boundary carries Z, not X
            cells = [
                (pr + dr) * d + (pc + dc)
                for dr in (0, 1)
                for dc in (0, 1)
                if 0 <= pr + dr < d and 0 <= pc + dc < d
            ]
            if len(cells) >= 2:
                supports.add(frozenset(cells))
    return supports


def _inject_single_fault_at_tick(
    code_circuit: stim.Circuit,
    fault_name: str,
    qubit: int,
    tick_index: int,
) -> stim.Circuit:
    """Copy of code_circuit with one deterministic fault after the
    (tick_index)-th TICK (0-indexed).  tick_index=0 is after the single prep
    TICK (R + H), i.e. data in |+> at the start of round 0.
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


def _fired_detectors(circuit: stim.Circuit) -> set[int]:
    """Detector indices that fire (circuit is deterministic under Z_ERROR=1.0)."""
    det = circuit.compile_detector_sampler().sample(shots=8)
    return {int(i) for i in np.flatnonzero(det.any(axis=0))}


# ----------------------------------------------------------------------------
# Primary acceptance gate: code distance
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("d", [3, 5])
def test_distance_gate(d: int) -> None:
    """Graphlike distance under flat Z-dephasing equals d (no hook collapse)."""
    code = surface_code(distance=d, rounds=d)
    noisy = inject_dephasing_noise(
        code.circuit,
        trajectories=np.zeros((code.n_qubits, code.n_cycles)),
        p_0=0.001,
        m=0.0,
        sigma=1.0,
        p_meas=0.0,
    )
    dem = noisy.detector_error_model(decompose_errors=True)
    assert len(dem.shortest_graphlike_error()) == d


# ----------------------------------------------------------------------------
# Structure / metadata
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("d", [3, 5])
def test_structure(d: int) -> None:
    code = surface_code(distance=d, rounds=d)
    assert code.circuit.num_qubits == 2 * d * d - 1
    assert code.n_qubits == 2 * d * d - 1
    assert code.distance == d
    assert code.rounds == d
    assert code.data_qubits == tuple(range(d * d))
    assert code.ancilla_qubits == tuple(range(d * d, 2 * d * d - 1))
    assert len(code.ancilla_qubits) == d * d - 1
    assert code.circuit.num_detectors == (d * d - 1) * d
    assert code.circuit.num_observables == 1

    flat = code.circuit.flattened()
    n_ticks = sum(
        1
        for item in flat
        if isinstance(item, stim.CircuitInstruction) and item.name == "TICK"
    )
    assert code.n_cycles == n_ticks + 1


@pytest.mark.parametrize("d", [3, 5])
def test_no_preexisting_noise(d: int) -> None:
    """Bare circuit carries no error channels; injection is the sole source."""
    code = surface_code(distance=d, rounds=d)
    for item in code.circuit.flattened():
        if isinstance(item, stim.CircuitInstruction):
            assert not item.name.endswith("_ERROR"), (
                f"preexisting error channel: {item.name}"
            )
            assert not item.name.startswith("DEPOLARIZE"), (
                f"preexisting depolarizing channel: {item.name}"
            )


@pytest.mark.parametrize("d", [3, 5])
def test_noiseless_determinism(d: int) -> None:
    """Noiseless circuit: all detectors zero, observable never flips."""
    code = surface_code(distance=d, rounds=d)
    sampler = code.circuit.compile_detector_sampler()
    det, obs = sampler.sample(shots=256, separate_observables=True)
    assert not det.any(), "expected all-zero detectors — check detector wiring"
    assert not obs.any(), "expected observable = 0 for noiseless |+_L> — check observable"


# ----------------------------------------------------------------------------
# Fault contrast: schedule wiring vs independent geometry
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("d", [3, 5])
def test_x_stabilizer_incidence(d: int) -> None:
    """A single Z fault fires exactly the X-stabilizers whose support contains
    the faulted qubit; final detectors cancel for a static rounds=1 fault.

    The empirical fault->detector incidence is inverted to recover each
    X-stabilizer's support and compared (unordered) to _expected_x_supports,
    which is derived independently of the constructor.
    """
    code = surface_code(distance=d, rounds=1)
    n_x = (d * d - 1) // 2

    circuit_support: dict[int, set[int]] = {i: set() for i in range(n_x)}
    for q in range(d * d):
        faulted = _inject_single_fault_at_tick(code.circuit, "Z_ERROR", q, tick_index=0)
        fired = _fired_detectors(faulted)
        # rounds=1: detectors [0, n_x) are round-0 X-syndromes; [n_x, 2 n_x) are
        # final reconstructions, which must cancel for a static |+> fault.
        assert all(i < n_x for i in fired), (
            f"d={d}, Z on q{q}: a final detector fired {sorted(fired)}; "
            f"static rounds=1 fault should cancel at readout"
        )
        for i in fired:
            circuit_support[i].add(q)

    circuit_supports = {frozenset(s) for s in circuit_support.values()}
    expected = _expected_x_supports(d)
    assert len(expected) == n_x  # oracle self-check
    assert circuit_supports == expected, (
        f"d={d}: circuit X-stabilizer supports {circuit_supports} "
        f"!= geometric expectation {expected}"
    )


# ----------------------------------------------------------------------------
# End-to-end decode path
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("d", [3, 5])
def test_subthreshold_decode(d: int) -> None:
    """Decoder runs end to end and beats coin-flip well below threshold."""
    code = surface_code(distance=d, rounds=d)
    noisy = inject_dephasing_noise(
        code.circuit,
        trajectories=np.zeros((code.n_qubits, code.n_cycles)),
        p_0=0.001,
        m=0.0,
        sigma=1.0,
        p_meas=0.0,
    )
    result = estimate_logical_error_rate(noisy, 2000, seed=0)
    assert result.shots == 2000
    assert result.rate < 0.5


# ----------------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("bad_distance", [2, 4, 1, 0, -1])
def test_invalid_distance_raises(bad_distance: int) -> None:
    with pytest.raises(ValueError):
        surface_code(distance=bad_distance, rounds=3)


def test_invalid_rounds_raises() -> None:
    with pytest.raises(ValueError):
        surface_code(distance=3, rounds=0)
