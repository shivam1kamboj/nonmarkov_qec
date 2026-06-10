"""Decoders: lookup-table for small codes, MWPM (PyMatching) for surface codes."""

from nonmarkov_qec.decoders.matching import (
    DecodeResult,
    estimate_logical_error_rate,
    matching_from_circuit,
)

__all__ = [
    "DecodeResult",
    "estimate_logical_error_rate",
    "matching_from_circuit",
]
