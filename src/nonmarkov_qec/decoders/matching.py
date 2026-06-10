"""Minimum-weight perfect matching decoder (PyMatching) for stabilizer circuits."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pymatching
import stim


@dataclass(frozen=True)
class DecodeResult:
    """Outcome of a Monte Carlo logical-error-rate estimate.

    Attributes
    ----------
    shots
        Number of decoded samples.
    errors
        Count of shots where the predicted logical flip differed from the
        actual observable outcome.
    """

    shots: int
    errors: int

    @property
    def rate(self) -> float:
        """Logical error rate, errors / shots (0.0 if shots == 0)."""
        if self.shots == 0:
            return 0.0
        return self.errors / self.shots

    @property
    def stderr(self) -> float:
        """Binomial standard error sqrt(p*(1-p)/shots) (0.0 if shots == 0)."""
        if self.shots == 0:
            return 0.0
        p = self.errors / self.shots
        return math.sqrt(p * (1.0 - p) / self.shots)


def matching_from_circuit(circuit: stim.Circuit) -> pymatching.Matching:
    """Build a PyMatching graph from a circuit's detector error model.

    Uses decompose_errors=True so any non-graphlike mechanism is split rather
    than dropped. Asserts the circuit has exactly one logical observable.

    Parameters
    ----------
    circuit
        A stim.Circuit (typically with injected noise channels) whose
        detector error model is used to build the MWPM graph.

    Returns
    -------
    pymatching.Matching
        Configured matching graph ready for decoding.

    Raises
    ------
    AssertionError
        If circuit.num_observables != 1.
    """
    assert circuit.num_observables == 1, (
        f"matching_from_circuit requires exactly 1 logical observable, "
        f"got {circuit.num_observables}"
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    return pymatching.Matching.from_detector_error_model(dem)


def estimate_logical_error_rate(
    circuit: stim.Circuit,
    shots: int,
    *,
    seed: int | None = None,
) -> DecodeResult:
    """Sample *circuit*, MWPM-decode each shot, count logical mismatches.

    Samples (detectors, observables) with a seeded compiled sampler, decodes
    the detector batch, and compares the predicted observable-0 flip to the
    actual. Assumes exactly one logical observable.

    Parameters
    ----------
    circuit
        A stim.Circuit with noise channels already injected.
    shots
        Number of Monte Carlo samples to draw and decode.
    seed
        Optional integer seed for the stim detector sampler; None gives a
        random seed. Identical seeds reproduce identical sample sequences.

    Returns
    -------
    DecodeResult
        shots, errors, and derived rate / stderr.
    """
    matching = matching_from_circuit(circuit)
    sampler = circuit.compile_detector_sampler(seed=seed)
    det, obs = sampler.sample(shots, separate_observables=True)
    pred = matching.decode_batch(det)
    errors = int(np.count_nonzero(pred[:, 0] != obs[:, 0]))
    return DecodeResult(shots=shots, errors=errors)
