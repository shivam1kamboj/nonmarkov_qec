"""Smoke tests confirming the package imports and version is set."""

import nonmarkov_qec


def test_version_is_set() -> None:
    assert isinstance(nonmarkov_qec.__version__, str)
    assert nonmarkov_qec.__version__.count(".") >= 1


def test_subpackages_importable() -> None:
    from importlib import import_module

    for name in (
        "nonmarkov_qec.benchmarks",
        "nonmarkov_qec.codes",
        "nonmarkov_qec.decoders",
        "nonmarkov_qec.noise",
    ):
        assert import_module(name) is not None
