"""Tests for public dataset dispatch."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from cobrabox import datasets


def test_dataset_dispatches_structured_identifiers(monkeypatch: pytest.MonkeyPatch) -> None:
    """dataset() routes structured IDs to load_structured_dummy."""
    captured: list[str] = []

    def _fake(identifier: str) -> list[object]:
        captured.append(identifier)
        return [object()]

    monkeypatch.setattr(datasets, "load_structured_dummy", _fake)

    out = datasets.load_dataset("dummy_chain")

    assert len(out) == 1
    assert captured == ["dummy_chain"]


def test_dataset_dispatches_noise_identifier(monkeypatch: pytest.MonkeyPatch) -> None:
    """dataset() routes dummy_noise to load_noise_dummy."""
    captured: list[str] = []

    def _fake(identifier: str) -> list[object]:
        captured.append(identifier)
        return [object()]

    monkeypatch.setattr(datasets, "load_noise_dummy", _fake)

    out = datasets.load_dataset("dummy_noise")

    assert len(out) == 1
    assert captured == ["dummy_noise"]


def test_dataset_dispatches_realistic_identifier(monkeypatch: pytest.MonkeyPatch) -> None:
    """dataset() routes realistic_swiss to load_realistic_swiss."""
    captured: list[str] = []

    def _fake(identifier: str) -> list[object]:
        captured.append(identifier)
        return [object()]

    monkeypatch.setattr(datasets, "load_realistic_swiss", _fake)

    out = datasets.load_dataset("realistic_swiss")

    assert len(out) == 1
    assert captured == ["realistic_swiss"]


def test_dataset_raises_for_unknown_identifier() -> None:
    """dataset() rejects unsupported IDs with a clear message."""
    with pytest.raises(ValueError, match="Unknown dataset identifier"):
        datasets.load_dataset("not_a_dataset")


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

    datasets.load_dataset("swiss_eeg_short", accept=True)

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

    datasets.load_dataset("swiss_eeg_short", accept=False)

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

    result = datasets.download_dataset("bonn_eeg", accept=True)
    assert result == tmp_path


def test_download_raises_for_local_dataset() -> None:
    """download() raises ValueError for local datasets."""
    import cobrabox as cb

    with pytest.raises(ValueError, match="local dataset"):
        cb.download_dataset("dummy_chain")


def test_download_raises_for_unknown_dataset() -> None:
    """download() raises ValueError for unknown identifiers."""
    import cobrabox as cb

    with pytest.raises(ValueError, match="Unknown dataset"):
        cb.download_dataset("nonexistent_dataset")


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
        datasets.download_dataset("bonn_eeg", subset=["INVALID"])


# ---------------------------------------------------------------------------
# show_datasets()
# ---------------------------------------------------------------------------


def test_describe_all_prints_all_datasets(capsys: pytest.CaptureFixture[str]) -> None:
    """show_datasets() prints a table containing all known dataset identifiers."""
    import cobrabox as cb

    cb.show_datasets()
    output = capsys.readouterr().out

    for ident in cb.list_datasets()["local"] + cb.list_datasets()["remote"]:
        assert ident in output


def test_describe_all_includes_header(capsys: pytest.CaptureFixture[str]) -> None:
    """show_datasets() prints column headers."""
    import cobrabox as cb

    cb.show_datasets()
    output = capsys.readouterr().out
    assert "Dataset" in output
    assert "Type" in output
    assert "Size" in output
    assert "License" in output


def test_describe_all_includes_cached_header(capsys: pytest.CaptureFixture[str]) -> None:
    """show_datasets() prints a Cached column header."""
    import cobrabox as cb

    cb.show_datasets()
    output = capsys.readouterr().out
    assert "Cached" in output


def test_describe_all_returns_list_of_dicts(capsys: pytest.CaptureFixture[str]) -> None:
    """show_datasets() returns one dict per dataset with the expected keys."""
    import cobrabox as cb

    rows = cb.show_datasets()
    capsys.readouterr()  # discard printed output

    assert isinstance(rows, list)
    assert len(rows) > 0
    for row in rows:
        assert "identifier" in row
        assert "type" in row
        assert "cached" in row
        assert "size" in row
        assert "subsets" in row
        assert "license" in row


def test_describe_all_local_datasets_have_null_cached(capsys: pytest.CaptureFixture[str]) -> None:
    """Local datasets have cached=None in the returned rows."""
    import cobrabox as cb

    rows = cb.show_datasets()
    capsys.readouterr()

    local_rows = [r for r in rows if r["type"] == "local"]
    assert local_rows, "expected at least one local dataset"
    for row in local_rows:
        assert row["cached"] is None


def test_describe_all_remote_datasets_have_string_cached(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Remote datasets have cached='yes' or 'no' in the returned rows."""
    import cobrabox as cb

    rows = cb.show_datasets()
    capsys.readouterr()

    remote_rows = [r for r in rows if r["type"] == "remote"]
    assert remote_rows, "expected at least one remote dataset"
    for row in remote_rows:
        assert row["cached"] in ("yes", "no")


# ---------------------------------------------------------------------------
# _is_dataset_cached
# ---------------------------------------------------------------------------


def test_is_dataset_cached_false_when_dir_missing(tmp_path: Path) -> None:
    """_is_dataset_cached returns False when the dataset directory does not exist."""
    from cobrabox.downloader import RemoteDatasetSpec, RemoteFile, _is_dataset_cached

    spec = RemoteDatasetSpec(
        identifier="bonn_eeg",
        local_rel_dir=Path("nonexistent_dir"),
        files=[RemoteFile(url="http://example.com/a.zip", filename="a.zip")],
        loader=lambda d, s: [],  # type: ignore[arg-type]
    )
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("cobrabox.downloader._data_dir", tmp_path)
        assert _is_dataset_cached(spec) is False


def test_is_dataset_cached_false_when_dir_empty(tmp_path: Path) -> None:
    """_is_dataset_cached returns False when the directory exists but has no data files."""
    from cobrabox.downloader import RemoteDatasetSpec, RemoteFile, _is_dataset_cached

    dataset_dir = tmp_path / "bonn_eeg"
    dataset_dir.mkdir()

    spec = RemoteDatasetSpec(
        identifier="bonn_eeg",
        local_rel_dir=Path("bonn_eeg"),
        files=[RemoteFile(url="http://example.com/a.zip", filename="a.zip")],
        loader=lambda d, s: [],  # type: ignore[arg-type]
    )
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("cobrabox.downloader._data_dir", tmp_path)
        assert _is_dataset_cached(spec) is False


def test_is_dataset_cached_false_manifest_only(tmp_path: Path) -> None:
    """_is_dataset_cached returns False when only _manifest.json is present."""
    from cobrabox.downloader import RemoteDatasetSpec, RemoteFile, _is_dataset_cached

    dataset_dir = tmp_path / "bonn_eeg"
    dataset_dir.mkdir()
    (dataset_dir / "_manifest.json").write_text("{}", encoding="utf-8")

    spec = RemoteDatasetSpec(
        identifier="bonn_eeg",
        local_rel_dir=Path("bonn_eeg"),
        files=[RemoteFile(url="http://example.com/a.zip", filename="a.zip")],
        loader=lambda d, s: [],  # type: ignore[arg-type]
    )
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("cobrabox.downloader._data_dir", tmp_path)
        assert _is_dataset_cached(spec) is False


def test_is_dataset_cached_true_when_file_present(tmp_path: Path) -> None:
    """_is_dataset_cached returns True when a data file exists in the directory."""
    from cobrabox.downloader import RemoteDatasetSpec, RemoteFile, _is_dataset_cached

    dataset_dir = tmp_path / "bonn_eeg"
    dataset_dir.mkdir()
    (dataset_dir / "S.zip").write_bytes(b"data")

    spec = RemoteDatasetSpec(
        identifier="bonn_eeg",
        local_rel_dir=Path("bonn_eeg"),
        files=[RemoteFile(url="http://example.com/S.zip", filename="S.zip")],
        loader=lambda d, s: [],  # type: ignore[arg-type]
    )
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("cobrabox.downloader._data_dir", tmp_path)
        assert _is_dataset_cached(spec) is True


# ---------------------------------------------------------------------------
# delete_remote_files
# ---------------------------------------------------------------------------


def test_delete_remote_files_removes_entire_dir(tmp_path: Path) -> None:
    """delete_remote_files with subset=None removes the full dataset directory."""
    from cobrabox.downloader import RemoteDatasetSpec, RemoteFile, delete_remote_files

    dataset_dir = tmp_path / "bonn_eeg"
    dataset_dir.mkdir()
    (dataset_dir / "S.zip").write_bytes(b"data")

    spec = RemoteDatasetSpec(
        identifier="bonn_eeg",
        local_rel_dir=Path("bonn_eeg"),
        files=[RemoteFile(url="http://example.com/S.zip", filename="S.zip")],
        loader=lambda d, s: [],  # type: ignore[arg-type]
    )

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("cobrabox.downloader._data_dir", tmp_path)
        delete_remote_files(spec, confirm=False)

    assert not dataset_dir.exists()


def test_delete_remote_files_noop_when_missing(tmp_path: Path) -> None:
    """delete_remote_files is a no-op when the dataset directory does not exist."""
    from cobrabox.downloader import RemoteDatasetSpec, RemoteFile, delete_remote_files

    spec = RemoteDatasetSpec(
        identifier="bonn_eeg",
        local_rel_dir=Path("nonexistent"),
        files=[RemoteFile(url="http://example.com/S.zip", filename="S.zip")],
        loader=lambda d, s: [],  # type: ignore[arg-type]
    )

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("cobrabox.downloader._data_dir", tmp_path)
        delete_remote_files(spec, confirm=False)  # should not raise


def test_delete_remote_files_subset_removes_only_matching_files(tmp_path: Path) -> None:
    """delete_remote_files with subset only deletes files for those keys."""
    from cobrabox.downloader import RemoteDatasetSpec, RemoteFile, delete_remote_files

    dataset_dir = tmp_path / "chb_mit"
    dataset_dir.mkdir()
    (dataset_dir / "chb01.edf").write_bytes(b"data1")
    (dataset_dir / "chb02.edf").write_bytes(b"data2")
    (dataset_dir / "chb03.edf").write_bytes(b"data3")

    spec = RemoteDatasetSpec(
        identifier="chb_mit",
        local_rel_dir=Path("chb_mit"),
        files=[
            RemoteFile(url="http://x.com/chb01.edf", filename="chb01.edf", subset_key="chb01"),
            RemoteFile(url="http://x.com/chb02.edf", filename="chb02.edf", subset_key="chb02"),
            RemoteFile(url="http://x.com/chb03.edf", filename="chb03.edf", subset_key="chb03"),
        ],
        loader=lambda d, s: [],  # type: ignore[arg-type]
    )

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("cobrabox.downloader._data_dir", tmp_path)
        delete_remote_files(spec, subset=["chb01", "chb02"], confirm=False)

    assert not (dataset_dir / "chb01.edf").exists()
    assert not (dataset_dir / "chb02.edf").exists()
    assert (dataset_dir / "chb03.edf").exists()


def test_delete_remote_files_confirm_cancel_raises(tmp_path: Path) -> None:
    """Declining the delete prompt raises RuntimeError."""
    from cobrabox.downloader import RemoteDatasetSpec, RemoteFile, delete_remote_files

    dataset_dir = tmp_path / "bonn_eeg"
    dataset_dir.mkdir()
    (dataset_dir / "S.zip").write_bytes(b"data")

    spec = RemoteDatasetSpec(
        identifier="bonn_eeg",
        local_rel_dir=Path("bonn_eeg"),
        files=[RemoteFile(url="http://example.com/S.zip", filename="S.zip")],
        loader=lambda d, s: [],  # type: ignore[arg-type]
    )

    with pytest.MonkeyPatch().context() as mp, patch("builtins.input", return_value="n"):
        mp.setattr("cobrabox.downloader._data_dir", tmp_path)
        with pytest.raises(RuntimeError, match="cancelled by user"):
            delete_remote_files(spec, confirm=True)

    assert (dataset_dir / "S.zip").exists()  # file must not have been deleted


def test_delete_remote_files_updates_manifest_on_subset_delete(tmp_path: Path) -> None:
    """Subset deletion removes deleted files from _manifest.json."""
    import json

    from cobrabox.downloader import RemoteDatasetSpec, RemoteFile, delete_remote_files

    dataset_dir = tmp_path / "chb_mit"
    dataset_dir.mkdir()
    (dataset_dir / "chb01.edf").write_bytes(b"a")
    (dataset_dir / "chb02.edf").write_bytes(b"b")
    (dataset_dir / "_manifest.json").write_text(
        json.dumps({"chb01.edf": 1, "chb02.edf": 1}), encoding="utf-8"
    )

    spec = RemoteDatasetSpec(
        identifier="chb_mit",
        local_rel_dir=Path("chb_mit"),
        files=[
            RemoteFile(url="http://x.com/chb01.edf", filename="chb01.edf", subset_key="chb01"),
            RemoteFile(url="http://x.com/chb02.edf", filename="chb02.edf", subset_key="chb02"),
        ],
        loader=lambda d, s: [],  # type: ignore[arg-type]
    )

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("cobrabox.downloader._data_dir", tmp_path)
        delete_remote_files(spec, subset=["chb01"], confirm=False)

    manifest = json.loads((dataset_dir / "_manifest.json").read_text(encoding="utf-8"))
    assert "chb01.edf" not in manifest
    assert "chb02.edf" in manifest


# ---------------------------------------------------------------------------
# delete_dataset
# ---------------------------------------------------------------------------


def test_delete_dataset_raises_for_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """delete_dataset raises ValueError for local (non-downloadable) datasets."""
    import cobrabox as cb

    with pytest.raises(ValueError, match="local dataset"):
        cb.delete_dataset("dummy_chain")


def test_delete_dataset_raises_for_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """delete_dataset raises ValueError for unknown identifiers."""
    import cobrabox as cb

    with pytest.raises(ValueError, match="Unknown dataset"):
        cb.delete_dataset("not_a_dataset")


def test_delete_dataset_delegates_to_delete_remote_files(monkeypatch: pytest.MonkeyPatch) -> None:
    """delete_dataset calls delete_remote_files with the right arguments."""
    from cobrabox import datasets

    calls: list[dict] = []

    def _fake_delete(spec: object, *, subset: object = None, confirm: bool = True) -> None:
        calls.append({"subset": subset, "confirm": confirm})

    monkeypatch.setattr(datasets, "delete_remote_files", _fake_delete)

    datasets.delete_dataset("bonn_eeg", subset=["S"], confirm=False)

    assert calls == [{"subset": ["S"], "confirm": False}]
