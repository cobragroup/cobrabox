"""Tests for public dataset dispatch."""

from __future__ import annotations

from pathlib import Path

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
    """accept=True passes straight through without prompting."""

    from cobrabox import downloader

    ensure_calls: list = []

    def _fake_ensure(  # type: ignore[return]
        spec: object,
        *,
        subset: object = None,
        data_dir: object = None,
        accept: bool = False,
        force: bool = False,
    ) -> Path:
        ensure_calls.append({"accept": accept})
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

    datasets.dataset("swiss_eeg_short", accept=True)

    assert ensure_calls == [{"accept": True}]


def test_dataset_remote_verify_true_passed_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
) -> None:
    """accept=False is forwarded to ensure_remote_files."""

    from cobrabox import downloader

    ensure_calls: list = []

    def _fake_ensure(  # type: ignore[return]
        spec: object,
        *,
        subset: object = None,
        data_dir: object = None,
        accept: bool = False,
        force: bool = False,
    ) -> Path:
        ensure_calls.append({"accept": accept})
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

    datasets.dataset("swiss_eeg_short", accept=False)

    assert ensure_calls == [{"accept": False}]


# ---------------------------------------------------------------------------
# download()
# ---------------------------------------------------------------------------


def test_download_returns_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """download() returns the dataset directory path without loading data."""
    from cobrabox import datasets, downloader

    fake_spec = downloader.RemoteDatasetSpec(
        identifier="bonn_eeg",
        local_rel_dir=tmp_path,
        files=[],
        loader=lambda d, s: (_ for _ in ()).throw(AssertionError("loader must not be called")),  # type: ignore[arg-type]
        description="test",
    )
    monkeypatch.setattr(datasets, "get_remote_dataset_spec", lambda _: fake_spec)
    monkeypatch.setattr(datasets, "ensure_remote_files", lambda *a, **kw: tmp_path)

    result = datasets.download("bonn_eeg", accept=True)
    assert result == tmp_path


def test_download_raises_for_local_dataset() -> None:
    """download() raises ValueError for local datasets."""
    import cobrabox as cb

    with pytest.raises(ValueError, match="local dataset"):
        cb.download("dummy_chain")


def test_download_raises_for_unknown_dataset() -> None:
    """download() raises ValueError for unknown identifiers."""
    import cobrabox as cb

    with pytest.raises(ValueError, match="Unknown dataset"):
        cb.download("nonexistent_dataset")


def test_download_raises_for_invalid_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    """download() raises ValueError for invalid subset keys."""
    from cobrabox import datasets, downloader

    fake_spec = downloader.RemoteDatasetSpec(
        identifier="bonn_eeg",
        local_rel_dir=Path("bonn_eeg"),
        files=[],
        loader=lambda d, s: [],  # type: ignore[arg-type]
        description="test",
        known_subset_keys=("Z", "S"),
        subset_key_name="sets",
    )
    monkeypatch.setattr(datasets, "get_remote_dataset_spec", lambda _: fake_spec)

    with pytest.raises(ValueError, match="Unknown subset keys"):
        datasets.download("bonn_eeg", subset=["INVALID"])


# ---------------------------------------------------------------------------
# describe_all()
# ---------------------------------------------------------------------------


def test_describe_all_prints_all_datasets(capsys: pytest.CaptureFixture[str]) -> None:
    """describe_all() prints a table containing all known dataset identifiers."""
    import cobrabox as cb

    cb.describe_all()
    output = capsys.readouterr().out

    for ident in cb.list_datasets()["local"] + cb.list_datasets()["remote"]:
        assert ident in output


def test_describe_all_includes_header(capsys: pytest.CaptureFixture[str]) -> None:
    """describe_all() prints column headers."""
    import cobrabox as cb

    cb.describe_all()
    output = capsys.readouterr().out
    assert "Dataset" in output
    assert "Type" in output
    assert "Size" in output
    assert "License" in output
