"""Noise injection layer: bridges SDE trajectories to per-gate Stim error channels.

Design note: docs/noise_injection.md
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import stim
from numpy.typing import NDArray
from scipy.stats import norm

# Gate classification sets.  Stim normalises aliases at insert time
# (e.g. CNOT → CX, MZ → M), so instr.name always reflects the canonical form.
# MR/MRX/MRY/MRZ are Stim's measure-and-reset gates used for ancilla qubits in
# generated circuits; they are treated identically to plain measurement gates:
# emitted as MR(p_meas) with ideal reset, never reading the trajectory.
# These sets will be re-validated against real Stim-generated code circuits
# during the codes phase.  Idle moments (two consecutive TICKs with no data
# gates between them) advance the cycle counter k but emit no Z_ERROR in v1.
DATA_GATES: frozenset[str] = frozenset({
    "H", "X", "Y", "Z", "S", "S_DAG", "SQRT_X",
    "CX", "CNOT", "CZ", "SWAP",
})
MEASUREMENT_GATES: frozenset[str] = frozenset({
    "M", "MZ", "MX", "MY",
    "MR", "MRX", "MRY", "MRZ",  # measure-and-reset; treated as M(p_meas) + ideal reset
})
PASSTHROUGH: frozenset[str] = frozenset({
    "R", "RX", "RY", "RZ",
    "DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS",
})


def inject_dephasing_noise(
    base_circuit: stim.Circuit,
    trajectories: NDArray[np.float64],
    p_0: float,
    m: float,
    sigma: float,
    p_meas: float,
) -> stim.Circuit:
    """Insert per-cycle Z_ERROR(p_{q,k}) instructions into a Stim circuit.

    Implements the linear dephasing model from docs/noise_injection.md:

        p_{q,k} = clip(p_0 + alpha * trajectories[q, k], 0, 1)
        alpha   = m * p_0 / sigma   (computed internally; m is user-facing)

    The base circuit is flattened (REPEAT blocks unrolled) and traversed in a
    single pass.  A fresh output circuit is returned; the input is not modified.

    Gate-then-error ordering: each data gate is appended first, then one
    Z_ERROR per qubit target follows immediately in the same moment.  Two-qubit
    gates emit one Z_ERROR per qubit, each reading its own trajectory row.
    Idle moments (consecutive TICKs with no intervening data gates) advance k
    without emitting any Z_ERROR.

    Measurement gates (M, MZ, MX, MY) and measure-and-reset gates (MR, MRX,
    MRY, MRZ) are emitted with p_meas as the flip probability and never read
    the trajectory.  MR* gates retain their built-in ideal reset; no separate R
    instruction is emitted.
    Resets and annotation instructions are passed through unchanged with no
    error appended and no clock advance.

    Gate sets (DATA_GATES, MEASUREMENT_GATES, PASSTHROUGH) will be
    re-validated against real Stim-generated code circuits during the codes
    phase.

    Parameters
    ----------
    base_circuit
        Stim circuit with stabilizer structure but no per-qubit error channels.
        May contain REPEAT blocks (they are flattened before processing).
    trajectories
        Float array of shape (n_qubits, n_cycles) where trajectories[q, k] is
        X_q at gate cycle k (zero-indexed).  The caller must strip the initial
        condition: pass raw_samples[:, 1:].  n_cycles must equal the number of
        TICKs in base_circuit plus one.
    p_0
        Mean Z-error rate (x-axis of threshold plots).
    m
        Dimensionless modulation depth; alpha = m * p_0 / sigma internally.
    sigma
        Stationary standard deviation of the noise process.
    p_meas
        Constant measurement flip probability; not trajectory-modulated in v1.

    Returns
    -------
    stim.Circuit
        New circuit with Z_ERROR instructions inserted after each data gate and
        M(p_meas) for each measurement.  base_circuit is not modified.

    Raises
    ------
    ValueError
        If the number of moments in the circuit does not match
        trajectories.shape[1], or if a qubit index in the circuit exceeds
        trajectories.shape[0] - 1, or if an unrecognised gate is encountered.
    """
    circuit = base_circuit.flattened()
    alpha: float = m * p_0 / sigma

    if m != 0.0:
        lower = float(norm.cdf(-1.0 / m))
        upper = float(norm.cdf(-(1.0 - p_0) / (m * p_0)))
        clip_fraction = lower + upper
        if clip_fraction > 0.05:
            warnings.warn(
                f"linear noise model: ~{clip_fraction:.1%} of cycles will clip at "
                f"(p_0={p_0}, m={m}); the trajectory statistics are distorted in this regime",
                stacklevel=2,
            )

    n_qubits: int = trajectories.shape[0]
    n_cycles: int = trajectories.shape[1]

    # Moments = (number of TICKs) + 1.
    n_ticks: int = 0
    for _item in circuit:
        if isinstance(_item, stim.CircuitInstruction) and _item.name == "TICK":
            n_ticks += 1
    n_moments: int = n_ticks + 1

    if n_moments != n_cycles:
        raise ValueError(
            f"circuit has {n_moments} moments but trajectory has {n_cycles} columns"
        )

    # Validate all qubit indices used by data and measurement gates.
    for _item in circuit:
        if isinstance(_item, stim.CircuitRepeatBlock):
            continue
        if _item.name in DATA_GATES or _item.name in MEASUREMENT_GATES:
            for _t in _item.targets_copy():
                if _t.is_qubit_target and _t.value >= n_qubits:
                    raise ValueError(
                        f"qubit index {_t.value} in gate '{_item.name}' exceeds "
                        f"trajectories row count {n_qubits}"
                    )

    out = stim.Circuit()
    k: int = 0

    for item in circuit:
        if isinstance(item, stim.CircuitRepeatBlock):
            raise AssertionError("unexpected CircuitRepeatBlock after flattened()")

        name: str = item.name

        if name == "TICK":
            out.append(item)
            k += 1

        elif name in DATA_GATES:
            out.append(item)
            for t in item.targets_copy():
                if t.is_qubit_target:
                    q: int = t.value
                    p: float = float(
                        np.clip(p_0 + alpha * float(trajectories[q, k]), 0.0, 1.0)
                    )
                    out.append("Z_ERROR", [q], p)

        elif name in MEASUREMENT_GATES:
            targets: list[int] = [
                t.value for t in item.targets_copy() if t.is_qubit_target
            ]
            out.append(name, targets, p_meas)

        elif name in PASSTHROUGH:
            out.append(item)

        else:
            raise ValueError(f"unrecognised gate '{name}' in base circuit")

    return out


@dataclass(frozen=True)
class InjectionTemplate:
    """Precompiled, reusable injection plan for one (circuit, p_meas) point.

    Built once per sweep point by :func:`compile_injection_template`, then
    reused across all Monte-Carlo trajectories via :func:`inject_from_template`.
    The structural circuit text (gates, measurements, detectors, p_meas) is
    fixed; only the per-cell ``Z_ERROR`` probabilities change per trajectory.
    A single ``stim.Circuit(text)`` parse per trajectory replaces ~10^3 Python
    ``append`` calls, the measured hot path (~20x faster at d=5).
    """

    template: str
    qk_q: NDArray[np.intp]
    qk_k: NDArray[np.intp]
    n_qubits: int
    n_cycles: int


def compile_injection_template(
    base_circuit: stim.Circuit, p_meas: float
) -> InjectionTemplate:
    """Compile a base circuit into a reusable :class:`InjectionTemplate`.

    Performs the trajectory-independent work once: flatten, moment count,
    qubit-index validation, and assembly of a Stim circuit text in which each
    data-gate ``Z_ERROR`` probability is a ``{:.17g}`` placeholder (17 sig figs
    -> exact IEEE-double round-trip). Placeholders appear in the same
    qubit/cycle order recorded in ``qk_q``/``qk_k``.
    """
    circuit = base_circuit.flattened()

    n_ticks = sum(
        1
        for it in circuit
        if isinstance(it, stim.CircuitInstruction) and it.name == "TICK"
    )
    n_cycles = n_ticks + 1

    n_qubits = 0
    for it in circuit:
        if isinstance(it, stim.CircuitRepeatBlock):
            continue
        if it.name in DATA_GATES or it.name in MEASUREMENT_GATES:
            for t in it.targets_copy():
                if t.is_qubit_target:
                    n_qubits = max(n_qubits, t.value + 1)

    lines: list[str] = []
    qk_q: list[int] = []
    qk_k: list[int] = []
    k = 0
    for it in circuit:
        if isinstance(it, stim.CircuitRepeatBlock):
            raise AssertionError("unexpected CircuitRepeatBlock after flattened()")
        name = it.name
        if name == "TICK":
            lines.append("TICK")
            k += 1
        elif name in DATA_GATES:
            lines.append(str(it))
            for t in it.targets_copy():
                if t.is_qubit_target:
                    lines.append("Z_ERROR({:.17g}) " + str(t.value))
                    qk_q.append(t.value)
                    qk_k.append(k)
        elif name in MEASUREMENT_GATES:
            targets = " ".join(
                str(t.value) for t in it.targets_copy() if t.is_qubit_target
            )
            lines.append(f"{name}({p_meas!r}) {targets}")
        elif name in PASSTHROUGH:
            lines.append(str(it))
        else:
            raise ValueError(f"unrecognised gate '{name}' in base circuit")

    return InjectionTemplate(
        template="\n".join(lines),
        qk_q=np.asarray(qk_q, dtype=np.intp),
        qk_k=np.asarray(qk_k, dtype=np.intp),
        n_qubits=n_qubits,
        n_cycles=n_cycles,
    )


def inject_from_template(
    tmpl: InjectionTemplate,
    trajectories: NDArray[np.float64],
    p_0: float,
    m: float,
    sigma: float,
) -> stim.Circuit:
    """Inject one trajectory using a precompiled template (the fast path).

    Equivalent to :func:`inject_dephasing_noise` (identical detector error
    model and sampling distribution) but reuses ``tmpl`` across trajectories.
    """
    if trajectories.shape[1] != tmpl.n_cycles:
        raise ValueError(
            f"template expects {tmpl.n_cycles} cycles but trajectory has "
            f"{trajectories.shape[1]} columns"
        )
    if trajectories.shape[0] < tmpl.n_qubits:
        raise ValueError(
            f"trajectory has {trajectories.shape[0]} rows but circuit uses "
            f"{tmpl.n_qubits} qubits"
        )
    alpha = m * p_0 / sigma
    probs = np.clip(p_0 + alpha * trajectories, 0.0, 1.0)
    vals = probs[tmpl.qk_q, tmpl.qk_k].tolist()
    return stim.Circuit(tmpl.template.format(*vals))
