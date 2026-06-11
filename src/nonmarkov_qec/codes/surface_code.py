"""Rotated surface code constructor — arbitrary odd distance, X-basis memory."""

from __future__ import annotations

from dataclasses import dataclass

import stim

from nonmarkov_qec.codes.base import CodeCircuit

# CX layer assignment keyed by the compass direction of a data qubit relative to
# its ancilla, in the frame where the row index increases SOUTH (row 0 = north)
# and the column index increases EAST. This mapping is extracted from — and
# bit-for-bit reproduces — the validated hand-built distance-3 schedule; the
# same geometric rule was confirmed distance-preserving at d=5 against Stim's
# rotated_memory_x generator. X- and Z-type stabilizers use *transposed* orders
# (X latitude-paired, Z longitude-paired) so the two types never collide on a
# shared data qubit within a layer — the hook-avoiding property that keeps the
# effective code distance equal to d.
_X_LAYER: dict[str, int] = {"SE": 0, "SW": 1, "NE": 2, "NW": 3}
_Z_LAYER: dict[str, int] = {"SE": 0, "NE": 1, "SW": 2, "NW": 3}


@dataclass(frozen=True)
class _Ancilla:
    """Internal record for one stabilizer ancilla."""

    qubit: int                     # stim qubit index
    kind: str                      # "X" or "Z"
    support: tuple[int, ...]       # data qubit indices, ascending
    schedule: dict[int, int]       # CX layer index (0-3) -> data qubit index
    coord: tuple[float, float]     # (x, y) for QUBIT_COORDS, larger y = north


def _build_ancillas(d: int) -> list[_Ancilla]:
    """Enumerate the d**2 - 1 stabilizer ancillas from the plaquette geometry.

    Plaquettes live on a grid of (pr, pc) positions. Interior plaquettes
    (pr, pc in 0..d-2) are weight-4; boundary plaquettes (one of pr, pc equal to
    -1 or d-1) are weight-2. Type is set by a checkerboard parity: X where
    (pr + pc) is odd, Z where even. Top/bottom boundaries carry only X
    stabilizers; left/right boundaries carry only Z. This yields exactly
    (d-1)**2 interior + 2*(d-1) boundary = d**2 - 1 stabilizers.

    Returns the ancillas with all X-type first (ascending plaquette position),
    then all Z-type, and assigns qubit indices d**2 .. 2*d**2 - 2 in that order.
    """
    raw_x: list[tuple[int, int, tuple[int, ...], dict[int, int], tuple[float, float]]] = []
    raw_z: list[tuple[int, int, tuple[int, ...], dict[int, int], tuple[float, float]]] = []

    for pr in range(-1, d):
        for pc in range(-1, d):
            support_rc = [
                (pr + dr, pc + dc)
                for dr in (0, 1)
                for dc in (0, 1)
                if 0 <= pr + dr < d and 0 <= pc + dc < d
            ]
            if len(support_rc) < 2:
                continue  # corner / degenerate position, not a stabilizer

            is_x = (pr + pc) % 2 == 1
            on_tb = pr in (-1, d - 1)   # top or bottom boundary
            on_lr = pc in (-1, d - 1)   # left or right boundary
            if on_tb and not is_x:
                continue  # top/bottom boundaries carry only X stabilizers
            if on_lr and is_x:
                continue  # left/right boundaries carry only Z stabilizers

            layer_map = _X_LAYER if is_x else _Z_LAYER
            center_row = pr + 0.5
            center_col = pc + 0.5
            schedule: dict[int, int] = {}
            support_list: list[int] = []
            for (r, c) in support_rc:
                q = r * d + c
                support_list.append(q)
                ns = "N" if r < center_row else "S"
                ew = "E" if c > center_col else "W"
                schedule[layer_map[ns + ew]] = q

            coord = (pc + 0.5, (d - 1) - (pr + 0.5))
            entry = (pr, pc, tuple(sorted(support_list)), schedule, coord)
            (raw_x if is_x else raw_z).append(entry)

    raw_x.sort(key=lambda e: (e[0], e[1]))
    raw_z.sort(key=lambda e: (e[0], e[1]))

    ancillas: list[_Ancilla] = []
    q = d * d
    for kind, raw in (("X", raw_x), ("Z", raw_z)):
        for (_pr, _pc, support, schedule, coord) in raw:
            ancillas.append(_Ancilla(qubit=q, kind=kind, support=support,
                                     schedule=schedule, coord=coord))
            q += 1
    return ancillas


def surface_code(distance: int, rounds: int) -> CodeCircuit:
    """Return a noise-free CodeCircuit for the rotated surface code, X-basis memory.

    Hand-built, coordinate-driven constructor for any odd ``distance`` >= 3.
    Encodes the logical |+_L> state, measures (d**2 - 1)/2 X-stabilizers and
    (d**2 - 1)/2 Z-stabilizers each round via a fixed four-layer CX schedule,
    and reads out all data qubits in the X basis. The logical observable is
    X-bar = product of X over data column 0.

    Geometry is in the frame where the data row index increases SOUTH (row 0 is
    north) and the column index increases EAST. Data qubits occupy indices
    0 .. d**2 - 1 (row-major: data(r, c) = r*d + c); ancillas occupy
    d**2 .. 2*d**2 - 2 (all X-type first, then all Z-type). Top/bottom code
    boundaries are X-type (weight-2), left/right are Z-type (weight-2).

    Parameters
    ----------
    distance
        Code distance; must be an odd integer >= 3.
    rounds
        Number of syndrome measurement rounds; must be >= 1.

    Returns
    -------
    CodeCircuit
        ``circuit`` has no error channels; inject_dephasing_noise is the sole
        noise source. ``n_qubits`` = 2*d**2 - 1, ``n_cycles`` = TICK count + 1.

    Raises
    ------
    ValueError
        If ``distance`` is even or < 3, or if ``rounds`` < 1.
    """
    if distance < 3 or distance % 2 == 0:
        raise ValueError(f"distance must be an odd integer >= 3, got {distance}")
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")

    d = distance
    n_data = d * d
    n_total = 2 * d * d - 1
    m = d * d - 1  # ancillas = measurement records per round

    data_qubits = tuple(range(n_data))
    ancillas = _build_ancillas(d)
    anc_qubits = tuple(a.qubit for a in ancillas)
    x_anc_qubits = [a.qubit for a in ancillas if a.kind == "X"]
    x_positions = [j for j, a in enumerate(ancillas) if a.kind == "X"]

    c = stim.Circuit()

    # Qubit coordinates (annotation only; larger y = north).
    for r in range(d):
        for col in range(d):
            c.append("QUBIT_COORDS", [r * d + col], [float(col), float(d - 1 - r)])
    for a in ancillas:
        c.append("QUBIT_COORDS", [a.qubit], [a.coord[0], a.coord[1]])

    # Reset everything; prepare data in |+> for X-basis memory.
    c.append("R", list(range(n_total)))
    c.append("H", list(data_qubits))
    c.append("TICK", [])

    for t in range(rounds):
        # X-ancillas into |+>.
        c.append("H", x_anc_qubits)
        c.append("TICK", [])

        # Four CX layers. X-stabs: ancilla controls data. Z-stabs: data controls
        # ancilla. Each physical qubit appears at most once per layer.
        for layer in range(4):
            pairs: list[int] = []
            for a in ancillas:
                data_q = a.schedule.get(layer)
                if data_q is None:
                    continue
                if a.kind == "X":
                    pairs += [a.qubit, data_q]
                else:
                    pairs += [data_q, a.qubit]
            if len(set(pairs)) != len(pairs):
                raise RuntimeError(
                    f"CX layer {layer} double-books a qubit at distance {d}; "
                    f"schedule rule failed to generalize."
                )
            if pairs:
                c.append("CX", pairs)
            c.append("TICK", [])

        # X-ancillas back to Z basis, then measure-and-reset all ancillas.
        c.append("H", x_anc_qubits)
        c.append("TICK", [])
        c.append("MR", list(anc_qubits))
        # Records: ancillas[j] -> rec(j - m), ascending in list order.

        if t == 0:
            # |+>^(d**2) is a +1 eigenstate of every X-stabilizer -> deterministic.
            # Z-stabilizer outcomes are random on this state; omit them in round 0.
            for j in x_positions:
                c.append("DETECTOR", [stim.target_rec(j - m)])
        else:
            # XOR each ancilla's outcome with its previous-round outcome.
            for j in range(m):
                c.append(
                    "DETECTOR",
                    [stim.target_rec(j - 2 * m), stim.target_rec(j - m)],
                )

        c.append("TICK", [])

    # Final X-basis data readout (records: data q -> rec(q - n_data)).
    c.append("MX", list(data_qubits))

    # Final detector per X-ancilla: last syndrome XOR reconstructed X-stabilizer.
    # Last-round ancilla records are shifted back by n_data MX records.
    for j in x_positions:
        a = ancillas[j]
        targets = [stim.target_rec(j - m - n_data)]
        targets += [stim.target_rec(q - n_data) for q in a.support]
        c.append("DETECTOR", targets)

    # Logical observable X-bar = X over data column 0.
    obs = [stim.target_rec((r * d) - n_data) for r in range(d)]
    c.append("OBSERVABLE_INCLUDE", obs, 0)

    flat = c.flattened()
    n_ticks: int = sum(
        1
        for item in flat
        if isinstance(item, stim.CircuitInstruction) and item.name == "TICK"
    )

    return CodeCircuit(
        circuit=c,
        data_qubits=data_qubits,
        ancilla_qubits=anc_qubits,
        n_qubits=c.num_qubits,
        n_cycles=n_ticks + 1,
        rounds=rounds,
        distance=distance,
    )
