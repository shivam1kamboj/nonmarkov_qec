"""Equivalence: the reusable template path == the reference injection path."""

from __future__ import annotations

import numpy as np

from nonmarkov_qec.codes.surface_code import surface_code
from nonmarkov_qec.noise.injection import (
    compile_injection_template,
    inject_dephasing_noise,
    inject_from_template,
)


def test_template_matches_reference_dem_and_samples() -> None:
    code = surface_code(3, 3)
    bare = code.circuit
    n_qubits, n_cycles = code.n_qubits, code.n_cycles
    rng = np.random.default_rng(7)
    traj = rng.standard_normal((n_qubits, n_cycles)) * 0.8
    p_0, m, sigma, p_meas = 0.013, 0.5, 1.0, 0.013

    ref = inject_dephasing_noise(bare, traj, p_0=p_0, m=m, sigma=sigma, p_meas=p_meas)
    tmpl = compile_injection_template(bare, p_meas=p_meas)
    new = inject_from_template(tmpl, traj, p_0=p_0, m=m, sigma=sigma)

    assert str(ref.detector_error_model(decompose_errors=True)) == str(
        new.detector_error_model(decompose_errors=True)
    )

    seed = 12345
    dr, orr = ref.compile_detector_sampler(seed=seed).sample(500, separate_observables=True)
    dn, onn = new.compile_detector_sampler(seed=seed).sample(500, separate_observables=True)
    assert np.array_equal(dr, dn)
    assert np.array_equal(orr, onn)
