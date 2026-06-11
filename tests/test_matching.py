"""Tests for the MWPM matching decoder."""

from __future__ import annotations

import math

import numpy as np
import pytest
import stim

from nonmarkov_qec.codes.surface_code import surface_code
from nonmarkov_qec.decoders.matching import (
    DecodeResult,
    estimate_logical_error_rate,
    matching_from_circuit,
)
from nonmarkov_qec.noise.injection import inject_dephasing_noise


def test_decode_result_arithmetic() -> None:
    """rate and stderr properties compute correctly; zero-shot case is safe."""
    r = DecodeResult(shots=1000, errors=50)
    assert r.rate == pytest.approx(0.05)
    assert r.stderr == pytest.approx(math.sqrt(0.05 * 0.95 / 1000))

    zero = DecodeResult(shots=0, errors=0)
    assert zero.rate == 0.0
    assert zero.stderr == 0.0


def test_matching_rejects_multiobservable() -> None:
    """matching_from_circuit asserts when num_observables != 1."""
    trivial = stim.Circuit()
    with pytest.raises(AssertionError):
        matching_from_circuit(trivial)


def test_noiseless_decodes_to_zero_errors() -> None:
    """Noiseless circuit is deterministic: decoder always predicts correctly."""
    code = surface_code(distance=3, rounds=3)
    result = estimate_logical_error_rate(code.circuit, shots=512)
    assert result.errors == 0


def test_seeded_reproducibility() -> None:
    """Same seed produces identical error counts on a noisy circuit."""
    code = surface_code(distance=3, rounds=3)
    noisy = inject_dephasing_noise(
        code.circuit,
        trajectories=np.zeros((code.n_qubits, code.n_cycles)),
        p_0=0.05,
        m=0.0,
        sigma=1.0,
        p_meas=0.0,
    )
    r1 = estimate_logical_error_rate(noisy, shots=200, seed=42)
    r2 = estimate_logical_error_rate(noisy, shots=200, seed=42)
    assert r1.errors == r2.errors


@pytest.mark.statistical
def test_sub_threshold_below_half() -> None:
    """At p_0=0.01 the decoded logical error rate is well below 0.5.

    A mis-wired observable or broken decoder pins near 0.5; rate < 0.4 catches
    that without being sensitive enough to flake at a genuinely sub-threshold
    operating point.
    """
    code = surface_code(distance=3, rounds=3)
    noisy = inject_dephasing_noise(
        code.circuit,
        trajectories=np.zeros((code.n_qubits, code.n_cycles)),
        p_0=0.01,
        m=0.0,
        sigma=1.0,
        p_meas=0.0,
    )
    result = estimate_logical_error_rate(noisy, shots=20000, seed=0)
    assert result.rate < 0.4
