"""Tests for public dataset dispatch."""

from __future__ import annotations

import pytest

from cobrabox import datasets


def test_dataset_dispatches_structured_identifiers(monkeypatch: pytest.MonkeyPatch) -> None:
    """dataset() routes structured IDs to load_structured_dummy."""
    captured: list[str] = []

    def _fake(identifier: str) -> list[object]:
        captured.append(identifier)
        return [object()]

    monkeypatch.setattr(datasets, "load_structured_dummy", _fake)

    out = datasets.dataset("dummy_chain")

    assert len(out) == 1
    assert captured == ["dummy_chain"]


def test_dataset_dispatches_noise_identifier(monkeypatch: pytest.MonkeyPatch) -> None:
    """dataset() routes dummy_noise to load_noise_dummy."""
    captured: list[str] = []

    def _fake(identifier: str) -> list[object]:
        captured.append(identifier)
        return [object()]

    monkeypatch.setattr(datasets, "load_noise_dummy", _fake)

    out = datasets.dataset("dummy_noise")

    assert len(out) == 1
    assert captured == ["dummy_noise"]


def test_dataset_dispatches_realistic_identifier(monkeypatch: pytest.MonkeyPatch) -> None:
    """dataset() routes realistic_swiss to load_realistic_swiss."""
    captured: list[str] = []

    def _fake(identifier: str) -> list[object]:
        captured.append(identifier)
        return [object()]

    monkeypatch.setattr(datasets, "load_realistic_swiss", _fake)

    out = datasets.dataset("realistic_swiss")

    assert len(out) == 1
    assert captured == ["realistic_swiss"]


def test_dataset_raises_for_unknown_identifier() -> None:
    """dataset() rejects unsupported IDs with a clear message."""
    with pytest.raises(ValueError, match="Unknown dataset identifier"):
        datasets.dataset("not_a_dataset")
