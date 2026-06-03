"""Tests for the noise injection layer (inject_dephasing_noise).

Circuit fixture has 4 TICKs → 5 moments → trajectories shape (3, 5).

Moment layout:
  k=0: H 0 1 2          (data gates)
  TICK
  k=1: CX 0 1  /  H 2  (data gates)
  TICK
  TICK                  (idle: no data gates, k advances 2→3)
  k=3: CX 1 2          (data gates)
  TICK
  k=4: M 0 1 2          (measurement layer)
"""

from __future__ import annotations

import numpy as np
import pytest
import stim

from nonmarkov_qec.noise.injection import inject_dephasing_noise

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fixture() -> stim.Circuit:
    c = stim.Circuit()
    c.append("H", [0, 1, 2])     # moment 0
    c.append("TICK", [])
    c.append("CX", [0, 1])       # moment 1
    c.append("H", [2])
    c.append("TICK", [])
    c.append("TICK", [])         # moment 2: idle (no data gates)
    c.append("CX", [1, 2])       # moment 3
    c.append("TICK", [])
    c.append("M", [0, 1, 2])     # moment 4: measurement layer
    return c


def extract_z_errors(circuit: stim.Circuit) -> list[tuple[int, float]]:
    """Return (qubit_index, probability) for every Z_ERROR target in circuit order."""
    result: list[tuple[int, float]] = []
    for item in circuit:
        if not isinstance(item, stim.CircuitInstruction):
            continue
        if item.name != "Z_ERROR":
            continue
        p = item.gate_args_copy()[0]
        for t in item.targets_copy():
            result.append((t.value, p))
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_moment_mismatch_raises() -> None:
    """trajectories with 4 columns vs 5 moments → ValueError naming both numbers."""
    base = make_fixture()
    trajectories = np.zeros((3, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="5") as exc_info:
        inject_dephasing_noise(base, trajectories, 0.01, 0.5, 1.0, 0.01)
    assert "4" in str(exc_info.value)


def test_qubit_out_of_range_raises() -> None:
    """trajectories with 2 rows vs circuit using qubit 2 → ValueError."""
    base = make_fixture()
    trajectories = np.zeros((2, 5), dtype=np.float64)
    with pytest.raises(ValueError, match="qubit index"):
        inject_dephasing_noise(base, trajectories, 0.01, 0.5, 1.0, 0.01)


def test_z_error_values_match_formula() -> None:
    """Every emitted Z_ERROR p equals clip(p_0 + alpha * trajectories[q, k])."""
    p_0, m, sigma = 0.05, 0.3, 1.0
    alpha = m * p_0 / sigma  # = 0.015

    # All-unique, non-clipping trajectory values so no Stim instruction fusion
    # confounds the ordering check.
    trajectories = np.array(
        [
            [0.00, 0.01, 0.02, 0.03, 0.04],  # qubit 0
            [0.05, 0.06, 0.07, 0.08, 0.09],  # qubit 1
            [0.10, 0.11, 0.12, 0.13, 0.14],  # qubit 2
        ],
        dtype=np.float64,
    )

    result = inject_dephasing_noise(make_fixture(), trajectories, p_0, m, sigma, p_meas=0.02)
    z_errors = extract_z_errors(result)

    # Expected sequence derived from fixture moment layout:
    #   k=0: H[0,1,2] → q=0,1,2 read col 0
    #   k=1: CX[0,1]  → q=0,1   read col 1
    #        H[2]      → q=2     reads col 1
    #   k=2: idle (no gates)
    #   k=3: CX[1,2]  → q=1,2   read col 3
    expected_qk: list[tuple[int, int]] = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 3), (2, 3),
    ]
    expected = [
        (q, float(np.clip(p_0 + alpha * trajectories[q, k], 0.0, 1.0)))
        for q, k in expected_qk
    ]

    assert len(z_errors) == len(expected), (
        f"expected {len(expected)} Z_ERROR entries, got {len(z_errors)}"
    )
    for i, ((got_q, got_p), (exp_q, exp_p)) in enumerate(zip(z_errors, expected, strict=True)):
        assert got_q == exp_q, f"index {i}: qubit {got_q} != expected {exp_q}"
        assert np.isclose(got_p, exp_p), (
            f"index {i}: p={got_p} != expected {exp_p}"
        )


def test_clip_bounds() -> None:
    """Large-positive trajectory → p==1.0; large-negative → p==0.0."""
    p_0, m, sigma = 0.05, 0.5, 1.0

    trajectories = np.zeros((3, 5), dtype=np.float64)
    trajectories[0, 0] = 1e6   # clip to 1.0
    trajectories[1, 0] = -1e6  # clip to 0.0
    # qubit 2 stays at 0.0 → p = p_0

    result = inject_dephasing_noise(make_fixture(), trajectories, p_0, m, sigma, p_meas=0.02)
    z_errors = extract_z_errors(result)

    # First three Z_ERRORs are from H[0,1,2] at k=0.
    # Find them by qubit index (order: q=0, q=1, q=2).
    first_three = z_errors[:3]
    q0_p = next(p for q, p in first_three if q == 0)
    q1_p = next(p for q, p in first_three if q == 1)

    assert np.isclose(q0_p, 1.0), f"expected 1.0, got {q0_p}"
    assert np.isclose(q1_p, 0.0), f"expected 0.0, got {q1_p}"


def test_idle_moment_advances_clock() -> None:
    """The idle TICK (moment 2) advances k so CX[1,2] at moment 3 reads col 3, not col 2."""
    p_0, m, sigma = 0.05, 0.3, 1.0
    alpha = m * p_0 / sigma  # = 0.015

    trajectories = np.zeros((3, 5), dtype=np.float64)
    # Column 2 (idle moment): use a large value that would give p≈1 if mistakenly read.
    trajectories[:, 2] = 1e6
    # Column 3 (CX[1,2] moment): use small distinct values.
    trajectories[1, 3] = 0.5
    trajectories[2, 3] = -0.2

    result = inject_dephasing_noise(make_fixture(), trajectories, p_0, m, sigma, p_meas=0.02)
    z_errors = extract_z_errors(result)

    # The last two Z_ERROR pairs come from CX[1,2] at k=3.
    got_q1, got_p1 = z_errors[-2]
    got_q2, got_p2 = z_errors[-1]

    exp_p1 = float(np.clip(p_0 + alpha * 0.5, 0.0, 1.0))
    exp_p2 = float(np.clip(p_0 + alpha * (-0.2), 0.0, 1.0))
    wrong_p = 1.0  # what reading col 2 (=1e6) would yield

    assert got_q1 == 1
    assert got_q2 == 2
    # Must match col 3, not col 2.
    assert np.isclose(got_p1, exp_p1), f"q=1: got {got_p1}, expected {exp_p1}"
    assert np.isclose(got_p2, exp_p2), f"q=2: got {got_p2}, expected {exp_p2}"
    assert not np.isclose(got_p1, wrong_p), "q=1 looks like it read col 2 (idle)"
    assert not np.isclose(got_p2, wrong_p), "q=2 looks like it read col 2 (idle)"


def test_two_qubit_gate_emits_both() -> None:
    """CX 0 1 at moment 1 emits Z_ERROR for q=0 (traj[0,1]) and q=1 (traj[1,1])."""
    p_0, m, sigma = 0.05, 0.3, 1.0
    alpha = m * p_0 / sigma

    trajectories = np.zeros((3, 5), dtype=np.float64)
    trajectories[0, 1] = 0.5   # distinct value for control qubit at k=1
    trajectories[1, 1] = -0.3  # distinct value for target qubit at k=1

    result = inject_dephasing_noise(make_fixture(), trajectories, p_0, m, sigma, p_meas=0.02)
    z_errors = extract_z_errors(result)

    # Z_ERROR ordering:
    #   [0,1,2]: from H[0,1,2] at k=0 (all traj=0 → p=p_0)
    #   [3]:     from CX[0,1] at k=1 for q=0
    #   [4]:     from CX[0,1] at k=1 for q=1
    exp_p_q0 = float(np.clip(p_0 + alpha * 0.5, 0.0, 1.0))
    exp_p_q1 = float(np.clip(p_0 + alpha * (-0.3), 0.0, 1.0))

    assert z_errors[3][0] == 0, f"expected qubit 0, got {z_errors[3][0]}"
    assert np.isclose(z_errors[3][1], exp_p_q0), (
        f"q=0 p={z_errors[3][1]} != expected {exp_p_q0}"
    )
    assert z_errors[4][0] == 1, f"expected qubit 1, got {z_errors[4][0]}"
    assert np.isclose(z_errors[4][1], exp_p_q1), (
        f"q=1 p={z_errors[4][1]} != expected {exp_p_q1}"
    )


def test_measurement_uses_p_meas() -> None:
    """The M instruction carries p_meas, not any value from the trajectory."""
    p_meas = 0.007
    trajectories = np.ones((3, 5), dtype=np.float64) * 0.5  # arbitrary non-zero

    result = inject_dephasing_noise(
        make_fixture(), trajectories, 0.05, 0.3, 1.0, p_meas=p_meas
    )

    meas_args: list[float] = []
    for item in result:
        if isinstance(item, stim.CircuitInstruction) and item.name in (
            "M", "MZ", "MX", "MY"
        ):
            meas_args.extend(item.gate_args_copy())

    assert len(meas_args) == 1, f"expected 1 measurement instruction, got {len(meas_args)}"
    assert np.isclose(meas_args[0], p_meas), (
        f"measurement p={meas_args[0]}, expected p_meas={p_meas}"
    )


def test_ticks_preserved() -> None:
    """Output has the same number of TICK instructions as the input."""
    trajectories = np.zeros((3, 5), dtype=np.float64)
    base = make_fixture()

    def count_ticks(circuit: stim.Circuit) -> int:
        return sum(
            1
            for item in circuit
            if isinstance(item, stim.CircuitInstruction) and item.name == "TICK"
        )

    result = inject_dephasing_noise(base, trajectories, 0.01, 0.3, 1.0, 0.01)
    assert count_ticks(result) == count_ticks(base)


def test_base_circuit_unmodified() -> None:
    """inject_dephasing_noise does not mutate its base_circuit argument."""
    base = make_fixture()
    before = repr(base)
    trajectories = np.zeros((3, 5), dtype=np.float64)
    inject_dephasing_noise(base, trajectories, 0.01, 0.3, 1.0, 0.01)
    assert repr(base) == before
