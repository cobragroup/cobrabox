"""Tests for dataset loader helpers."""

from __future__ import annotations

import io
import json
import lzma
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cobrabox import datasets
from cobrabox.dataset_loader import (
    _sampling_rate_from_info,
    _sidecar_json_for_csv,
    load_noise_dummy,
    load_realistic_swiss,
    load_structured_dummy,
)
from cobrabox.remote_datasets import (
    RemoteDatasetSpec,
    RemoteFile,
    ensure_remote_files,
    get_remote_dataset_spec,
)


def test_load_structured_dummy_reads_matching_files(tmp_path: Path) -> None:
    """Structured loader returns one Data object per matching non-empty file."""
    struct_dir = tmp_path / "data" / "synthetic" / "dummy" / "struct"
    struct_dir.mkdir(parents=True)

    pd.DataFrame({"ch0": [1.0, 2.0], "ch1": [3.0, 4.0]}).to_csv(
        struct_dir / "dummy_struct_VAR_chain_1.csv.xz", index=False, compression="xz"
    )
    pd.DataFrame({"ch0": [5.0], "ch1": [6.0]}).to_csv(
        struct_dir / "dummy_struct_VAR_chain_2.csv.xz", index=False, compression="xz"
    )

    out = load_structured_dummy("dummy_chain", repo_root=tmp_path)

    assert len(out) == 2
    assert out[0].data.dims == ("space", "time")
    assert list(out[0].data.coords["space"].values) == ["ch0", "ch1"]
    np.testing.assert_allclose(out[0].to_numpy(), np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert out[0].data.attrs["identifier"] == "dummy_chain"


def test_load_structured_dummy_raises_when_no_files(tmp_path: Path) -> None:
    """Structured loader raises if no matching files exist."""
    (tmp_path / "data" / "synthetic" / "dummy" / "struct").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="No files found"):
        load_structured_dummy("dummy_chain", repo_root=tmp_path)


def test_load_structured_dummy_raises_when_all_files_empty(tmp_path: Path) -> None:
    """Structured loader raises if all matching files are empty."""
    struct_dir = tmp_path / "data" / "synthetic" / "dummy" / "struct"
    struct_dir.mkdir(parents=True)
    pd.DataFrame(columns=["ch0", "ch1"]).to_csv(
        struct_dir / "dummy_struct_VAR_star_1.csv.xz", index=False, compression="xz"
    )

    with pytest.raises(ValueError, match="All files for 'dummy_star' are empty"):
        load_structured_dummy("dummy_star", repo_root=tmp_path)


def test_load_noise_dummy_reads_all_noise_files(tmp_path: Path) -> None:
    """Noise loader returns one Data object per non-empty noise file."""
    noise_dir = tmp_path / "data" / "synthetic" / "dummy" / "noise"
    noise_dir.mkdir(parents=True)
    a_csv = noise_dir / "a.csv.xz"
    b_csv = noise_dir / "b.csv.xz"
    pd.DataFrame({"n0": [0.1, 0.2]}).to_csv(a_csv, index=False, compression="xz")
    pd.DataFrame({"n0": [0.3, 0.4]}).to_csv(b_csv, index=False, compression="xz")

    # Sidecar JSON with settings metadata
    a_json = noise_dir / "info_a.json.xz"
    with lzma.open(a_json, "wt", encoding="utf-8") as f:
        json.dump({"Settings": {"Seizure start (sec)": 8, "SOZ": "[1 0 1 0]", "fs": 256}}, f)

    out = load_noise_dummy(repo_root=tmp_path)

    assert len(out) == 2
    np.testing.assert_allclose(out[0].to_numpy(), np.array([[0.1, 0.2]]))
    assert out[0].data.attrs["identifier"] == "dummy_noise"
    assert out[0].sampling_rate == 256.0
    # Extra metadata propagated from JSON, including SOZ array string
    assert "Settings" in out[0].extra
    settings = out[0].extra["Settings"]
    assert settings["Seizure start (sec)"] == 8
    assert settings["SOZ"] == "[1 0 1 0]"


def test_load_noise_dummy_raises_when_all_files_empty(tmp_path: Path) -> None:
    """Noise loader raises if all discovered files are empty."""
    noise_dir = tmp_path / "data" / "synthetic" / "dummy" / "noise"
    noise_dir.mkdir(parents=True)
    pd.DataFrame(columns=["n0"]).to_csv(noise_dir / "a.csv.xz", index=False, compression="xz")

    with pytest.raises(ValueError, match="All files for 'dummy_noise' are empty"):
        load_noise_dummy(repo_root=tmp_path)


def test_load_noise_dummy_raises_when_no_csv_xz_files(tmp_path: Path) -> None:
    """Noise loader raises FileNotFoundError when the noise dir has no .csv.xz files."""
    noise_dir = tmp_path / "data" / "synthetic" / "dummy" / "noise"
    noise_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match=r"No \.csv\.xz files found"):
        load_noise_dummy(repo_root=tmp_path)


def test_load_realistic_swiss_reads_all_matching_files(tmp_path: Path) -> None:
    """Realistic loader returns one Data object per non-empty Swiss VAR file."""
    realistic_dir = tmp_path / "data" / "synthetic" / "realistic"
    realistic_dir.mkdir(parents=True)
    csv1 = realistic_dir / "fit_Swiss_VAR_ID1_sz13_simulated_data_1.csv.xz"
    csv2 = realistic_dir / "fit_Swiss_VAR_ID1_sz7_simulated_data_3.csv.xz"
    pd.DataFrame({"ch0": [1.0, 2.0]}).to_csv(csv1, index=False, compression="xz")
    pd.DataFrame({"ch0": [3.0, 4.0]}).to_csv(csv2, index=False, compression="xz")

    json1 = realistic_dir / "info_fit_Swiss_VAR_ID1_sz13_simulated_data_1.json.xz"
    with lzma.open(json1, "wt", encoding="utf-8") as f:
        json.dump({"Settings": {"Seizure start (sec)": 5, "SOZ": "[0 1 0]", "fs": 512}}, f)

    out = load_realistic_swiss(repo_root=tmp_path)

    assert len(out) == 2
    np.testing.assert_allclose(out[0].to_numpy(), np.array([[1.0, 2.0]]))
    assert out[0].data.attrs["identifier"] == "realistic_swiss"
    assert out[0].sampling_rate == 512.0
    assert "Settings" in out[0].extra
    settings = out[0].extra["Settings"]
    assert settings["Seizure start (sec)"] == 5
    assert settings["SOZ"] == "[0 1 0]"


def test_load_realistic_swiss_raises_when_no_matching_files(tmp_path: Path) -> None:
    """Realistic loader raises when there are no matching Swiss VAR files."""
    realistic_dir = tmp_path / "data" / "synthetic" / "realistic"
    realistic_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match=r"No \.csv\.xz files found"):
        load_realistic_swiss(repo_root=tmp_path)


def test_load_realistic_swiss_raises_when_all_files_empty(tmp_path: Path) -> None:
    """Realistic loader raises when all matching files are empty."""
    realistic_dir = tmp_path / "data" / "synthetic" / "realistic"
    realistic_dir.mkdir(parents=True)
    pd.DataFrame(columns=["ch0"]).to_csv(
        realistic_dir / "fit_Swiss_VAR_ID1_sz13_simulated_data_1.csv.xz",
        index=False,
        compression="xz",
    )

    with pytest.raises(ValueError, match="All files for 'realistic_swiss' are empty"):
        load_realistic_swiss(repo_root=tmp_path)


def test_sidecar_json_for_csv_non_csv_xz_extension(tmp_path: Path) -> None:
    """_sidecar_json_for_csv falls back to path.stem for non-.csv.xz files."""
    p = tmp_path / "myfile.csv"
    result = _sidecar_json_for_csv(p)
    assert result.name == "info_myfile.json.xz"


def test_sampling_rate_from_info_raises_for_non_numeric_fs() -> None:
    """_sampling_rate_from_info raises ValueError when fs is not numeric."""
    with pytest.raises(ValueError, match="Invalid sampling rate"):
        _sampling_rate_from_info({"fs": "not-a-number"})


def test_sampling_rate_from_info_raises_for_unconvertible_type() -> None:
    """_sampling_rate_from_info raises ValueError when fs cannot be float-cast."""
    with pytest.raises(ValueError, match="Invalid sampling rate"):
        _sampling_rate_from_info({"fs": [1, 2]})


def test_load_structured_dummy_ignores_malformed_json_sidecar(tmp_path: Path) -> None:
    """Malformed JSON sidecar is silently ignored; data still loads."""
    struct_dir = tmp_path / "data" / "synthetic" / "dummy" / "struct"
    struct_dir.mkdir(parents=True)
    pd.DataFrame({"ch0": [1.0, 2.0]}).to_csv(
        struct_dir / "dummy_struct_VAR_chain_1.csv.xz", index=False, compression="xz"
    )
    json_path = struct_dir / "info_dummy_struct_VAR_chain_1.json.xz"
    with lzma.open(json_path, "wt", encoding="utf-8") as f:
        f.write("not valid json {{{")

    out = load_structured_dummy("dummy_chain", repo_root=tmp_path)
    assert len(out) == 1
    assert out[0].sampling_rate is None


def test_load_noise_dummy_ignores_malformed_json_sidecar(tmp_path: Path) -> None:
    """Malformed JSON sidecar in noise loader is silently ignored."""
    noise_dir = tmp_path / "data" / "synthetic" / "dummy" / "noise"
    noise_dir.mkdir(parents=True)
    pd.DataFrame({"n0": [1.0, 2.0]}).to_csv(noise_dir / "a.csv.xz", index=False, compression="xz")
    json_path = noise_dir / "info_a.json.xz"
    with lzma.open(json_path, "wt", encoding="utf-8") as f:
        f.write("{{invalid")

    out = load_noise_dummy(repo_root=tmp_path)
    assert len(out) == 1
    assert out[0].sampling_rate is None


def test_load_realistic_swiss_ignores_malformed_json_sidecar(tmp_path: Path) -> None:
    """Malformed JSON sidecar in realistic loader is silently ignored."""
    realistic_dir = tmp_path / "data" / "synthetic" / "realistic"
    realistic_dir.mkdir(parents=True)
    csv_path = realistic_dir / "fit_Swiss_VAR_ID1_sz1_simulated_data_1.csv.xz"
    pd.DataFrame({"ch0": [1.0, 2.0]}).to_csv(csv_path, index=False, compression="xz")
    json_path = realistic_dir / "info_fit_Swiss_VAR_ID1_sz1_simulated_data_1.json.xz"
    with lzma.open(json_path, "wt", encoding="utf-8") as f:
        f.write("{{invalid")

    out = load_realistic_swiss(repo_root=tmp_path)
    assert len(out) == 1
    assert out[0].sampling_rate is None


@pytest.mark.slow
def test_load_structured_dummy_default_repo_root() -> None:
    """load_structured_dummy infers repo_root from package path and loads real data."""
    from cobrabox.data import Data

    out = load_structured_dummy("dummy_chain")
    assert len(out) > 0
    assert all(isinstance(d, Data) for d in out)


@pytest.mark.slow
def test_load_noise_dummy_default_repo_root() -> None:
    """load_noise_dummy infers repo_root from package path and loads real data."""
    from cobrabox.data import Data

    try:
        out = load_noise_dummy()
    except FileNotFoundError:
        pytest.skip("Real dummy noise dataset files not available (likely LFS not fetched).")
    assert len(out) > 0
    assert all(isinstance(d, Data) for d in out)


@pytest.mark.slow
def test_load_realistic_swiss_default_repo_root() -> None:
    """load_realistic_swiss infers repo_root and loads real data."""
    from cobrabox.data import Data

    try:
        out = load_realistic_swiss()
    except FileNotFoundError:
        pytest.skip("Real realistic_swiss dataset files not available (likely LFS not fetched).")
    assert len(out) > 0
    assert all(isinstance(d, Data) for d in out)


class _NoOpBar:
    """Silent stand-in for tqdm progress bars used in remote-download tests."""

    def __enter__(self) -> _NoOpBar:
        return self

    def __exit__(self, *exc_info: object) -> None:
        pass

    def update(self, n: int) -> None:
        pass


def test_ensure_remote_files_downloads_missing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ensure_remote_files downloads missing remote files into the dataset directory."""
    # Arrange a simple spec with two files.
    files = [
        RemoteFile(url="http://example.com/a.bin", filename="a.bin"),
        RemoteFile(url="http://example.com/b.bin", filename="b.bin"),
    ]
    spec = RemoteDatasetSpec(
        identifier="test_remote",
        local_rel_dir=Path("data") / "remote" / "test_remote",
        files=files,
        loader=lambda _p: [],
    )

    # Fake HTTP responses with small byte payloads.
    payloads = {"http://example.com/a.bin": b"AAA", "http://example.com/b.bin": b"BBB"}

    class _FakeHeaders:
        def get(self, key: str, default: str | None = None) -> str | None:
            return default

    class _FakeResponse(io.BytesIO):
        headers = _FakeHeaders()

        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *exc_info: object) -> None:  # type: ignore[override]
            self.close()

    def _fake_urlopen(url: str, *args: object, **kwargs: object) -> _FakeResponse:
        try:
            return _FakeResponse(payloads[url])
        except KeyError as exc:
            raise AssertionError(f"Unexpected URL requested: {url!r}") from exc

    import cobrabox.remote_datasets as remote_datasets

    monkeypatch.setattr(remote_datasets.urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(remote_datasets, "tqdm", lambda *a, **kw: _NoOpBar())

    # Act
    dataset_dir = ensure_remote_files(spec, repo_root=tmp_path)

    # Assert
    assert dataset_dir == tmp_path / spec.local_rel_dir
    a_path = dataset_dir / "a.bin"
    b_path = dataset_dir / "b.bin"
    assert a_path.exists()
    assert b_path.exists()
    assert a_path.read_bytes() == b"AAA"
    assert b_path.read_bytes() == b"BBB"


def test_ensure_remote_files_skips_existing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing files are not re-downloaded by ensure_remote_files."""
    files = [RemoteFile(url="http://example.com/a.bin", filename="a.bin")]
    spec = RemoteDatasetSpec(
        identifier="test_remote",
        local_rel_dir=Path("data") / "remote" / "test_remote",
        files=files,
        loader=lambda _p: [],
    )

    dataset_dir = tmp_path / spec.local_rel_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)
    a_path = dataset_dir / "a.bin"
    a_path.write_bytes(b"ORIGINAL")

    import cobrabox.remote_datasets as remote_datasets

    def _failing_urlopen(url: str, *args: object, **kwargs: object) -> io.BytesIO:
        raise AssertionError("urlopen should not be called when files already exist")

    monkeypatch.setattr(remote_datasets.urllib.request, "urlopen", _failing_urlopen)

    result_dir = ensure_remote_files(spec, repo_root=tmp_path)

    assert result_dir == dataset_dir
    assert a_path.read_bytes() == b"ORIGINAL"


def test_ensure_remote_files_auth_hint_shown_on_401_403(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """auth_hint is surfaced when the server responds with a 401 or 403."""
    files = [RemoteFile(url="https://example.com/dataset.zip", filename="dataset.zip")]
    spec = RemoteDatasetSpec(
        identifier="some_dataset",
        local_rel_dir=Path("data") / "remote" / "some_dataset",
        files=files,
        loader=lambda _p: [],
        auth_hint="You need credentials to download this dataset.",
    )

    import cobrabox.remote_datasets as remote_datasets

    def _raise_http_error(url: str, *args: object, **kwargs: object) -> io.BytesIO:
        raise remote_datasets.urllib.error.HTTPError(
            url=url, code=403, msg="Forbidden", hdrs=None, fp=None
        )

    monkeypatch.setattr(remote_datasets.urllib.request, "urlopen", _raise_http_error)

    with pytest.raises(RuntimeError, match="credentials") as excinfo:
        ensure_remote_files(spec, repo_root=tmp_path)

    assert "Expected file location" in str(excinfo.value)


def test_ensure_remote_files_no_auth_hint_generic_error_on_401_403(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without auth_hint, a generic HTTP error is raised for 401/403."""
    files = [RemoteFile(url="https://example.com/dataset.zip", filename="dataset.zip")]
    spec = RemoteDatasetSpec(
        identifier="some_dataset",
        local_rel_dir=Path("data") / "remote" / "some_dataset",
        files=files,
        loader=lambda _p: [],
    )

    import cobrabox.remote_datasets as remote_datasets

    def _raise_http_error(url: str, *args: object, **kwargs: object) -> io.BytesIO:
        raise remote_datasets.urllib.error.HTTPError(
            url=url, code=403, msg="Forbidden", hdrs=None, fp=None
        )

    monkeypatch.setattr(remote_datasets.urllib.request, "urlopen", _raise_http_error)

    with pytest.raises(RuntimeError, match="HTTP 403"):
        ensure_remote_files(spec, repo_root=tmp_path)


def test_dataset_uses_remote_spec_for_known_identifier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """dataset() consults the remote dataset registry for known remote identifiers."""
    called: list[str] = []

    def _fake_loader(path: Path) -> list[object]:
        called.append(str(path))
        return [object()]

    fake_spec = RemoteDatasetSpec(
        identifier="swiss_eeg_short",
        local_rel_dir=Path("data") / "remote" / "swiss_eeg_short",
        files=[],
        loader=_fake_loader,
    )

    def _fake_get_remote_dataset_spec(identifier: str) -> RemoteDatasetSpec | None:
        return fake_spec if identifier == "swiss_eeg_short" else None

    def _fake_ensure_remote_files(
        spec: RemoteDatasetSpec, *, repo_root: Path | None = None
    ) -> Path:
        assert spec is fake_spec
        base = tmp_path if repo_root is None else repo_root
        path = base / spec.local_rel_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    # datasets.py imports these names directly, so patch them in that module's namespace.
    monkeypatch.setattr(datasets, "get_remote_dataset_spec", _fake_get_remote_dataset_spec)
    monkeypatch.setattr(datasets, "ensure_remote_files", _fake_ensure_remote_files)

    out = datasets.dataset("swiss_eeg_short")

    assert len(out) == 1
    assert called


def test_ensure_remote_files_uses_index_when_no_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ensure_remote_files resolves files from a remote index when files is None."""
    spec = RemoteDatasetSpec(
        identifier="test_index",
        local_rel_dir=Path("data") / "remote" / "test_index",
        files=None,
        loader=lambda _p: [],
        file_index_url="http://example.com/index.txt",
    )

    index_body = b"http://example.com/a.bin\nhttp://example.com/b.bin\n"
    payloads = {
        "http://example.com/index.txt": index_body,
        "http://example.com/a.bin": b"AAA",
        "http://example.com/b.bin": b"BBB",
    }

    class _FakeHeaders:
        def get(self, key: str, default: str | None = None) -> str | None:
            return default

    class _FakeResponse(io.BytesIO):
        headers = _FakeHeaders()

        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *exc_info: object) -> None:  # type: ignore[override]
            self.close()

    def _fake_urlopen(url: str, *args: object, **kwargs: object) -> _FakeResponse:
        try:
            return _FakeResponse(payloads[url])
        except KeyError as exc:
            raise AssertionError(f"Unexpected URL requested: {url!r}") from exc

    import cobrabox.remote_datasets as remote_datasets

    monkeypatch.setattr(remote_datasets.urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(remote_datasets, "tqdm", lambda *a, **kw: _NoOpBar())

    dataset_dir = ensure_remote_files(spec, repo_root=tmp_path)

    assert (dataset_dir / "a.bin").read_bytes() == b"AAA"
    assert (dataset_dir / "b.bin").read_bytes() == b"BBB"


def test_swiss_eeg_short_loader_reads_csv_from_zip(tmp_path: Path) -> None:
    """Swiss short EEG loader reads numeric CSV inside a zip archive."""
    from cobrabox.data import SignalData

    dataset_dir = tmp_path / "data" / "remote" / "swiss_eeg_short"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dataset_dir / "ID1.zip"
    with zipfile.ZipFile(zip_path, mode="w") as zf:
        zf.writestr("ID1_signal.csv", "ch0,ch1\n1.0,2.0\n3.0,4.0\n")

    spec = get_remote_dataset_spec("swiss_eeg_short")
    assert spec is not None

    out = spec.loader(dataset_dir)

    assert len(out) == 1
    assert isinstance(out[0], SignalData)
    assert out[0].subjectID == "ID1"
    np.testing.assert_allclose(out[0].to_numpy(), np.array([[1.0, 3.0], [2.0, 4.0]]))
    assert out[0].data.attrs["source_archive"] == "ID1.zip"


def test_swiss_eeg_short_loader_raises_when_no_zip_files(tmp_path: Path) -> None:
    """Swiss short EEG loader raises when no zip files are available."""
    dataset_dir = tmp_path / "data" / "remote" / "swiss_eeg_short"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    spec = get_remote_dataset_spec("swiss_eeg_short")
    assert spec is not None

    with pytest.raises(FileNotFoundError, match=r"No \.zip files found"):
        spec.loader(dataset_dir)
