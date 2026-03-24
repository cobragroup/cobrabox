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


def test_dataset_remote_verify_false_skips_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
) -> None:
    """verify=False passes straight through without prompting."""
    from pathlib import Path

    from cobrabox import downloader

    ensure_calls: list = []

    def _fake_ensure(  # type: ignore[return]
        spec: object, *, subset: object = None, repo_root: object = None, verify: bool = True
    ) -> Path:
        ensure_calls.append({"verify": verify})
        return tmp_path  # type: ignore[return-value]

    def _fake_loader(dataset_dir: Path, subset: object) -> list[object]:
        return [object()]

    fake_spec = downloader.RemoteDatasetSpec(
        identifier="swiss_eeg_short",
        local_rel_dir=tmp_path,
        files=[],
        loader=_fake_loader,
        description="test",
    )
    monkeypatch.setattr(datasets, "get_remote_dataset_spec", lambda _: fake_spec)
    monkeypatch.setattr(datasets, "ensure_remote_files", _fake_ensure)

    datasets.dataset("swiss_eeg_short", verify=False)

    assert ensure_calls == [{"verify": False}]


def test_dataset_remote_verify_true_passed_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
) -> None:
    """verify=True is forwarded to ensure_remote_files."""
    from pathlib import Path

    from cobrabox import downloader

    ensure_calls: list = []

    def _fake_ensure(  # type: ignore[return]
        spec: object, *, subset: object = None, repo_root: object = None, verify: bool = True
    ) -> Path:
        ensure_calls.append({"verify": verify})
        return tmp_path  # type: ignore[return-value]

    def _fake_loader(dataset_dir: Path, subset: object) -> list[object]:
        return [object()]

    fake_spec = downloader.RemoteDatasetSpec(
        identifier="swiss_eeg_short",
        local_rel_dir=tmp_path,
        files=[],
        loader=_fake_loader,
        description="test",
    )
    monkeypatch.setattr(datasets, "get_remote_dataset_spec", lambda _: fake_spec)
    monkeypatch.setattr(datasets, "ensure_remote_files", _fake_ensure)

    datasets.dataset("swiss_eeg_short", verify=True)

    assert ensure_calls == [{"verify": True}]
