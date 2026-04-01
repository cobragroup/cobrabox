"""Tests for dataset loader helpers."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cobrabox import datasets
from cobrabox.dataset import Dataset
from cobrabox.dataset_loader import (
    _sampling_rate_from_info,
    _sidecar_json_for_csv,
    load_noise_dummy,
    load_realistic_swiss,
    load_structured_dummy,
)
from cobrabox.downloader import (
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
    a_json = noise_dir / "info_a.json"
    with open(a_json, "w", encoding="utf-8") as f:
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

    json1 = realistic_dir / "info_fit_Swiss_VAR_ID1_sz13_simulated_data_1.json"
    with open(json1, "w", encoding="utf-8") as f:
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
    assert result.name == "info_myfile.json"


def test_sampling_rate_from_info_raises_for_non_numeric_fs() -> None:
    """_sampling_rate_from_info raises ValueError when fs is not numeric."""
    with pytest.raises(ValueError, match="Invalid sampling rate"):
        _sampling_rate_from_info({"fs": "not-a-number"})


def test_sampling_rate_from_info_raises_for_unconvertible_type() -> None:
    """_sampling_rate_from_info raises ValueError when fs cannot be float-cast."""
    with pytest.raises(ValueError, match="Invalid sampling rate"):
        _sampling_rate_from_info({"fs": [1, 2]})


def test_sampling_rate_from_info_falls_back_to_top_level_fs() -> None:
    """Top-level fs is used when Settings exists but has no fs key."""
    info = {"Settings": {"Seizure start (sec)": 8}, "fs": 256}
    assert _sampling_rate_from_info(info) == 256.0


def test_load_structured_dummy_ignores_malformed_json_sidecar(tmp_path: Path) -> None:
    """Malformed JSON sidecar is silently ignored; data still loads."""
    struct_dir = tmp_path / "data" / "synthetic" / "dummy" / "struct"
    struct_dir.mkdir(parents=True)
    pd.DataFrame({"ch0": [1.0, 2.0]}).to_csv(
        struct_dir / "dummy_struct_VAR_chain_1.csv.xz", index=False, compression="xz"
    )
    json_path = struct_dir / "info_dummy_struct_VAR_chain_1.json"
    with open(json_path, "w", encoding="utf-8") as f:
        f.write("not valid json {{{")

    out = load_structured_dummy("dummy_chain", repo_root=tmp_path)
    assert len(out) == 1
    assert out[0].sampling_rate is None


def test_load_noise_dummy_ignores_malformed_json_sidecar(tmp_path: Path) -> None:
    """Malformed JSON sidecar in noise loader is silently ignored."""
    noise_dir = tmp_path / "data" / "synthetic" / "dummy" / "noise"
    noise_dir.mkdir(parents=True)
    pd.DataFrame({"n0": [1.0, 2.0]}).to_csv(noise_dir / "a.csv.xz", index=False, compression="xz")
    json_path = noise_dir / "info_a.json"
    with open(json_path, "w", encoding="utf-8") as f:
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
    json_path = realistic_dir / "info_fit_Swiss_VAR_ID1_sz1_simulated_data_1.json"
    with open(json_path, "w", encoding="utf-8") as f:
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

    def _fake_urlopen(url: object, *args: object, **kwargs: object) -> _FakeResponse:
        try:
            return _FakeResponse(payloads[getattr(url, "full_url", url)])
        except KeyError as exc:
            raise AssertionError(f"Unexpected URL requested: {url!r}") from exc

    import cobrabox.downloader as downloader

    monkeypatch.setattr(downloader.urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(downloader, "tqdm", lambda *a, **kw: _NoOpBar())

    # Act
    dataset_dir = ensure_remote_files(spec, data_dir=tmp_path, accept=True)

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

    import cobrabox.downloader as downloader

    def _failing_urlopen(url: str, *args: object, **kwargs: object) -> io.BytesIO:
        raise AssertionError("urlopen should not be called when files already exist")

    monkeypatch.setattr(downloader.urllib.request, "urlopen", _failing_urlopen)

    result_dir = ensure_remote_files(spec, data_dir=tmp_path)

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

    import cobrabox.downloader as downloader

    def _raise_http_error(url: str, *args: object, **kwargs: object) -> io.BytesIO:
        raise downloader.urllib.error.HTTPError(
            url=url, code=403, msg="Forbidden", hdrs=None, fp=None
        )

    monkeypatch.setattr(downloader.urllib.request, "urlopen", _raise_http_error)

    with pytest.raises(RuntimeError, match="credentials") as excinfo:
        ensure_remote_files(spec, data_dir=tmp_path, accept=True)

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

    import cobrabox.downloader as downloader

    def _raise_http_error(url: str, *args: object, **kwargs: object) -> io.BytesIO:
        raise downloader.urllib.error.HTTPError(
            url=url, code=403, msg="Forbidden", hdrs=None, fp=None
        )

    monkeypatch.setattr(downloader.urllib.request, "urlopen", _raise_http_error)

    with pytest.raises(RuntimeError, match="HTTP 403"):
        ensure_remote_files(spec, data_dir=tmp_path, accept=True)


def test_dataset_uses_remote_spec_for_known_identifier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """dataset() consults the remote dataset registry for known remote identifiers."""
    import xarray as xr

    from cobrabox.data import SignalData

    called: list[tuple[str, object]] = []

    def _fake_loader(path: Path, subset: object = None) -> Dataset:
        called.append((str(path), subset))
        da = xr.DataArray([[1.0, 2.0]], dims=["time", "space"])
        return Dataset([SignalData.from_xarray(da)])

    fake_spec = RemoteDatasetSpec(
        identifier="swiss_eeg_short",
        local_rel_dir=Path("data") / "remote" / "swiss_eeg_short",
        files=[],
        loader=_fake_loader,
    )

    def _fake_get_remote_dataset_spec(identifier: str) -> RemoteDatasetSpec | None:
        return fake_spec if identifier == "swiss_eeg_short" else None

    def _fake_ensure_remote_files(
        spec: RemoteDatasetSpec,
        *,
        subset: object = None,
        data_dir: Path | None = None,
        accept: bool = False,
        force: bool = False,
    ) -> Path:
        assert spec is fake_spec
        base = tmp_path if data_dir is None else data_dir
        path = base / spec.local_rel_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    # datasets.py imports these names directly, so patch them in that module's namespace.
    monkeypatch.setattr(datasets, "get_remote_dataset_spec", _fake_get_remote_dataset_spec)
    monkeypatch.setattr(datasets, "ensure_remote_files", _fake_ensure_remote_files)

    out = datasets.dataset("swiss_eeg_short")

    assert len(out) == 1
    assert called


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

    out = spec.loader(dataset_dir, None)

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
        spec.loader(dataset_dir, None)


def test_swiss_eeg_short_loader_subset_filters_by_stem(tmp_path: Path) -> None:
    """Passing subset to the loader only loads matching zip archives."""

    dataset_dir = tmp_path / "data" / "remote" / "swiss_eeg_short"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for subject in ("ID1", "ID2", "ID5"):
        zip_path = dataset_dir / f"{subject}.zip"
        with zipfile.ZipFile(zip_path, mode="w") as zf:
            zf.writestr(f"{subject}_signal.csv", "ch0\n1.0\n2.0\n")

    spec = get_remote_dataset_spec("swiss_eeg_short")
    assert spec is not None

    out = spec.loader(dataset_dir, ["ID1", "ID5"])

    assert len(out) == 2
    subject_ids = {item.subjectID for item in out}
    assert subject_ids == {"ID1", "ID5"}


def test_ensure_remote_files_subset_filters_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ensure_remote_files only downloads files matching the requested subset."""
    files = [
        RemoteFile(url="http://example.com/ID1.zip", filename="ID1.zip", subset_key="ID1"),
        RemoteFile(url="http://example.com/ID2.zip", filename="ID2.zip", subset_key="ID2"),
        RemoteFile(url="http://example.com/ID5.zip", filename="ID5.zip", subset_key="ID5"),
    ]
    spec = RemoteDatasetSpec(
        identifier="test_subset",
        local_rel_dir=Path("data") / "remote" / "test_subset",
        files=files,
        loader=lambda _p, _s: Dataset([]),
    )

    downloaded: list[str] = []
    payloads = {"http://example.com/ID1.zip": b"ZIP1", "http://example.com/ID5.zip": b"ZIP5"}

    class _FakeHeaders:
        def get(self, key: str, default: str | None = None) -> str | None:
            return default

    class _FakeResponse(io.BytesIO):
        headers = _FakeHeaders()

        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *exc_info: object) -> None:  # type: ignore[override]
            self.close()

    def _fake_urlopen(url: object, *args: object, **kwargs: object) -> _FakeResponse:
        downloaded.append(getattr(url, "full_url", url))
        try:
            return _FakeResponse(payloads[getattr(url, "full_url", url)])
        except KeyError as exc:
            raise AssertionError(f"Unexpected URL requested: {url!r}") from exc

    import cobrabox.downloader as downloader

    monkeypatch.setattr(downloader.urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(downloader, "tqdm", lambda *a, **kw: _NoOpBar())

    dataset_dir = ensure_remote_files(spec, subset=["ID1", "ID5"], data_dir=tmp_path, accept=True)

    assert (dataset_dir / "ID1.zip").exists()
    assert (dataset_dir / "ID5.zip").exists()
    assert not (dataset_dir / "ID2.zip").exists()
    assert set(downloaded) == {"http://example.com/ID1.zip", "http://example.com/ID5.zip"}


# ---------------------------------------------------------------------------
# SubsetSpec dict form — _filter_files_by_dict_subset unit tests
# ---------------------------------------------------------------------------


def test_filter_files_by_dict_subset_none_includes_all_subject_files() -> None:
    """None value returns all files for that subject key."""
    from cobrabox.downloader import RemoteFile, _filter_files_by_dict_subset

    files = [
        RemoteFile(url="u", filename="a.mat", subset_key="ID01"),
        RemoteFile(url="u", filename="b.mat", subset_key="ID01"),
        RemoteFile(url="u", filename="c.mat", subset_key="ID02"),
    ]
    result = _filter_files_by_dict_subset({"ID01": None}, files)
    assert [f.filename for f in result if f.subset_key == "ID01"] == ["a.mat", "b.mat"]
    assert not any(f.subset_key == "ID02" for f in result)


def test_filter_files_by_dict_subset_int_returns_first_n() -> None:
    """Int value returns the first N files for that subject key."""
    from cobrabox.downloader import RemoteFile, _filter_files_by_dict_subset

    files = [RemoteFile(url="u", filename=f"ID01_{i}h.mat", subset_key="ID01") for i in range(1, 6)]
    result = _filter_files_by_dict_subset({"ID01": 3}, files)
    assert [f.filename for f in result] == ["ID01_1h.mat", "ID01_2h.mat", "ID01_3h.mat"]


def test_filter_files_by_dict_subset_int_zero_raises() -> None:
    """Int value of 0 raises ValueError."""
    from cobrabox.downloader import RemoteFile, _filter_files_by_dict_subset

    files = [RemoteFile(url="u", filename="ID01_1h.mat", subset_key="ID01")]
    with pytest.raises(ValueError, match="must be >= 1"):
        _filter_files_by_dict_subset({"ID01": 0}, files)


def test_filter_files_by_dict_subset_list_filters_by_filename() -> None:
    """List value returns only the named files."""
    from cobrabox.downloader import RemoteFile, _filter_files_by_dict_subset

    files = [
        RemoteFile(url="u", filename="ID01_1h.mat", subset_key="ID01"),
        RemoteFile(url="u", filename="ID01_2h.mat", subset_key="ID01"),
        RemoteFile(url="u", filename="ID01_3h.mat", subset_key="ID01"),
    ]
    result = _filter_files_by_dict_subset({"ID01": ["ID01_1h.mat", "ID01_3h.mat"]}, files)
    assert [f.filename for f in result] == ["ID01_1h.mat", "ID01_3h.mat"]


def test_filter_files_by_dict_subset_empty_list_raises() -> None:
    """Empty list value raises ValueError."""
    from cobrabox.downloader import RemoteFile, _filter_files_by_dict_subset

    files = [RemoteFile(url="u", filename="ID01_1h.mat", subset_key="ID01")]
    with pytest.raises(ValueError, match="non-empty"):
        _filter_files_by_dict_subset({"ID01": []}, files)


def test_filter_files_by_dict_subset_unknown_filename_raises() -> None:
    """Unknown filename raises ValueError when the file list is known."""
    from cobrabox.downloader import RemoteFile, _filter_files_by_dict_subset

    files = [RemoteFile(url="u", filename="ID01_1h.mat", subset_key="ID01")]
    with pytest.raises(ValueError, match="Unknown filenames"):
        _filter_files_by_dict_subset({"ID01": ["ID01_99h.mat"]}, files)


def test_filter_files_by_dict_subset_no_subset_key_always_included() -> None:
    """Files with no subset_key are always included regardless of the dict."""
    from cobrabox.downloader import RemoteFile, _filter_files_by_dict_subset

    sidecar = RemoteFile(url="u", filename="info.mat", subset_key=None)
    subject = RemoteFile(url="u", filename="ID01_1h.mat", subset_key="ID01")
    result = _filter_files_by_dict_subset({"ID01": 1}, [sidecar, subject])
    assert any(f.filename == "info.mat" for f in result)


# ---------------------------------------------------------------------------
# SubsetSpec dict form — ensure_remote_files integration
# ---------------------------------------------------------------------------


def test_ensure_remote_files_dict_subset_int_downloads_first_n(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dict subset with int value only downloads the first N files per subject."""
    files = [
        RemoteFile(url=f"http://x.com/ID01_{i}h.mat", filename=f"ID01_{i}h.mat", subset_key="ID01")
        for i in range(1, 5)
    ]
    spec = RemoteDatasetSpec(
        identifier="test_dict_int",
        local_rel_dir=Path("data") / "remote" / "test_dict_int",
        files=files,
        loader=lambda _p, _s: Dataset([]),
    )

    downloaded: list[str] = []

    class _FakeHeaders:
        def get(self, key: str, default: str | None = None) -> str | None:
            return default

    class _FakeResponse(io.BytesIO):
        headers = _FakeHeaders()

        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    def _fake_urlopen(url: object, *a: object, **kw: object) -> _FakeResponse:
        downloaded.append(getattr(url, "full_url", url))
        return _FakeResponse(b"DATA")

    import cobrabox.downloader as downloader

    monkeypatch.setattr(downloader.urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(downloader, "tqdm", lambda *a, **kw: _NoOpBar())

    ensure_remote_files(spec, subset={"ID01": 2}, data_dir=tmp_path, accept=True)

    assert len(downloaded) == 2
    assert any("ID01_1h.mat" in u for u in downloaded)
    assert any("ID01_2h.mat" in u for u in downloaded)


def test_ensure_remote_files_dict_subset_list_downloads_named_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dict subset with list value downloads only the explicitly named files."""
    files = [
        RemoteFile(url=f"http://x.com/ID01_{i}h.mat", filename=f"ID01_{i}h.mat", subset_key="ID01")
        for i in range(1, 4)
    ]
    spec = RemoteDatasetSpec(
        identifier="test_dict_list",
        local_rel_dir=Path("data") / "remote" / "test_dict_list",
        files=files,
        loader=lambda _p, _s: Dataset([]),
    )

    downloaded: list[str] = []

    class _FakeHeaders:
        def get(self, key: str, default: str | None = None) -> str | None:
            return default

    class _FakeResponse(io.BytesIO):
        headers = _FakeHeaders()

        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    def _fake_urlopen(url: object, *a: object, **kw: object) -> _FakeResponse:
        downloaded.append(getattr(url, "full_url", url))
        return _FakeResponse(b"DATA")

    import cobrabox.downloader as downloader

    monkeypatch.setattr(downloader.urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(downloader, "tqdm", lambda *a, **kw: _NoOpBar())

    ensure_remote_files(
        spec, subset={"ID01": ["ID01_1h.mat", "ID01_3h.mat"]}, data_dir=tmp_path, accept=True
    )

    assert len(downloaded) == 2
    assert any("ID01_1h.mat" in u for u in downloaded)
    assert any("ID01_3h.mat" in u for u in downloaded)
    assert not any("ID01_2h.mat" in u for u in downloaded)


# ---------------------------------------------------------------------------
# SubsetSpec dict form — dataset() validation and loader forwarding
# ---------------------------------------------------------------------------


def test_dataset_dict_subset_invalid_subject_key_raises() -> None:
    """dataset() raises ValueError for unknown subject keys in dict form."""
    with pytest.raises(ValueError, match="Unknown subset keys"):
        datasets.dataset("swiss_eeg_short", subset={"INVALID_ID": 2})


def test_dataset_dict_subset_int_zero_raises() -> None:
    """dataset() raises ValueError when a dict value is 0."""
    with pytest.raises(ValueError, match="must be >= 1"):
        datasets.dataset("swiss_eeg_short", subset={"ID1": 0})


def test_dataset_dict_subset_empty_list_raises() -> None:
    """dataset() raises ValueError when a dict value is an empty list."""
    with pytest.raises(ValueError, match="non-empty"):
        datasets.dataset("swiss_eeg_short", subset={"ID1": []})


def test_dataset_dict_subset_passes_stems_to_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """dataset() passes file stems (not subject keys) to the loader for dict-form subsets."""
    import xarray as xr

    from cobrabox.data import SignalData

    captured: dict[str, object] = {}

    def _fake_loader(path: Path, subset: object = None) -> Dataset:
        captured["loader_subset"] = subset
        da = xr.DataArray([[1.0]], dims=["time", "space"])
        return Dataset([SignalData.from_xarray(da)])

    spec = get_remote_dataset_spec("swiss_eeg_short")
    assert spec is not None
    assert spec.files is not None

    patched_spec = RemoteDatasetSpec(
        identifier=spec.identifier,
        local_rel_dir=spec.local_rel_dir,
        files=spec.files,
        loader=_fake_loader,
        description=spec.description,
        subset_key_name=spec.subset_key_name,
    )

    def _fake_get_spec(identifier: str) -> RemoteDatasetSpec | None:
        return patched_spec if identifier == "swiss_eeg_short" else None

    def _fake_ensure(
        s: RemoteDatasetSpec,
        *,
        subset: object = None,
        data_dir: Path | None = None,
        accept: bool = False,
        force: bool = False,
    ) -> Path:
        p = tmp_path / s.local_rel_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    monkeypatch.setattr(datasets, "get_remote_dataset_spec", _fake_get_spec)
    monkeypatch.setattr(datasets, "ensure_remote_files", _fake_ensure)

    datasets.dataset("swiss_eeg_short", subset={"ID1": 1})

    # The loader should receive a list of stems, not the dict.
    loader_subset = captured["loader_subset"]
    assert isinstance(loader_subset, list)
    assert loader_subset == ["ID1"]  # stem of "ID1.zip" is "ID1"


def test_dataset_info_returns_local_dataset_info() -> None:
    """dataset_info returns description for local datasets."""
    from cobrabox.datasets import dataset_info

    info = dataset_info("dummy_chain")
    assert info.identifier == "dummy_chain"
    assert info.description
    assert info.subset_key_name is None
    assert info.subsets is None


def test_dataset_info_returns_remote_dataset_info() -> None:
    """dataset_info returns subsets and subset_key_name for remote datasets."""
    from cobrabox.datasets import dataset_info

    info = dataset_info("swiss_eeg_short")
    assert info.identifier == "swiss_eeg_short"
    assert info.description
    assert info.subset_key_name == "subjects"
    assert info.subsets is not None
    assert "ID1" in info.subsets
    assert "ID16" in info.subsets
    assert len(info.subsets) == 18


def test_dataset_info_str_contains_usage_hint() -> None:
    """DatasetInfo.__str__ includes a usage hint showing how to pass subset."""
    from cobrabox.datasets import dataset_info

    text = str(dataset_info("swiss_eeg_short"))
    assert "cb.dataset" in text
    assert "subset=" in text
    assert "ID1" in text


def test_dataset_info_raises_for_unknown_identifier() -> None:
    """dataset_info raises ValueError for unknown identifiers."""
    from cobrabox.datasets import dataset_info

    with pytest.raises(ValueError, match="Unknown dataset identifier"):
        dataset_info("nonexistent_dataset_xyz")


def test_dataset_subset_raises_for_invalid_keys() -> None:
    """dataset() raises ValueError when unknown subset keys are passed."""
    with pytest.raises(ValueError, match="Unknown subset keys"):
        datasets.dataset("swiss_eeg_short", subset=["INVALID_ID"])


def test_dataset_subset_passes_to_loader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """dataset() forwards the subset to both ensure_remote_files and the loader."""
    import xarray as xr

    from cobrabox.data import SignalData

    captured: dict[str, object] = {}

    def _fake_loader(path: Path, subset: object = None) -> Dataset:
        captured["loader_subset"] = subset
        da = xr.DataArray([[1.0]], dims=["time", "space"])
        return Dataset([SignalData.from_xarray(da)])

    spec = get_remote_dataset_spec("swiss_eeg_short")
    assert spec is not None

    patched_spec = RemoteDatasetSpec(
        identifier=spec.identifier,
        local_rel_dir=spec.local_rel_dir,
        files=spec.files,
        loader=_fake_loader,
        description=spec.description,
        subset_key_name=spec.subset_key_name,
    )

    def _fake_get_spec(identifier: str) -> RemoteDatasetSpec | None:
        return patched_spec if identifier == "swiss_eeg_short" else None

    def _fake_ensure(
        s: RemoteDatasetSpec,
        *,
        subset: object = None,
        data_dir: Path | None = None,
        accept: bool = False,
        force: bool = False,
    ) -> Path:
        captured["ensure_subset"] = subset
        p = tmp_path / s.local_rel_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    monkeypatch.setattr(datasets, "get_remote_dataset_spec", _fake_get_spec)
    monkeypatch.setattr(datasets, "ensure_remote_files", _fake_ensure)

    datasets.dataset("swiss_eeg_short", subset=["ID1", "ID2"])

    assert captured["loader_subset"] == ["ID1", "ID2"]
    assert captured["ensure_subset"] == ["ID1", "ID2"]


# ---------------------------------------------------------------------------
# _sidecar_json_for_csv — else branch (non-csv/non-csv.xz extension)
# ---------------------------------------------------------------------------


def test_sidecar_json_for_csv_non_csv_extension(tmp_path: Path) -> None:
    """_sidecar_json_for_csv uses path.stem for non-.csv and non-.csv.xz files."""
    p = tmp_path / "myfile.mat"
    result = _sidecar_json_for_csv(p)
    assert result.name == "info_myfile.json"


# ---------------------------------------------------------------------------
# _sampling_rate_from_info — returns None when no fs key present
# ---------------------------------------------------------------------------


def test_sampling_rate_from_info_returns_none_when_no_fs() -> None:
    """_sampling_rate_from_info returns None when neither Settings.fs nor fs exist."""
    from cobrabox.dataset_loader import _sampling_rate_from_info

    assert _sampling_rate_from_info({}) is None
    assert _sampling_rate_from_info({"Settings": {"other_key": 1}}) is None


# ---------------------------------------------------------------------------
# _extract_numeric_from_csv_bytes
# ---------------------------------------------------------------------------


def test_extract_numeric_from_csv_bytes_raises_when_no_numeric_columns() -> None:
    """_extract_numeric_from_csv_bytes raises when no numeric columns are present."""
    from cobrabox.dataset_loader import _extract_numeric_from_csv_bytes

    raw = b"name,label\nfoo,bar\n"
    with pytest.raises(ValueError, match="no numeric columns"):
        _extract_numeric_from_csv_bytes(raw)


def test_extract_numeric_from_csv_bytes_returns_values_and_channels() -> None:
    """_extract_numeric_from_csv_bytes extracts numeric columns and channel names."""
    from cobrabox.dataset_loader import _extract_numeric_from_csv_bytes

    raw = b"ch0,ch1\n1.0,2.0\n3.0,4.0\n"
    values, channels = _extract_numeric_from_csv_bytes(raw)
    assert values.shape == (2, 2)
    assert channels == ["ch0", "ch1"]


# ---------------------------------------------------------------------------
# _extract_numeric_from_npy_bytes
# ---------------------------------------------------------------------------


def test_extract_numeric_from_npy_bytes_2d_array() -> None:
    """_extract_numeric_from_npy_bytes handles a plain 2D .npy array."""
    from cobrabox.dataset_loader import _extract_numeric_from_npy_bytes

    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    buf = io.BytesIO()
    np.save(buf, arr)
    values, channels = _extract_numeric_from_npy_bytes(buf.getvalue())
    np.testing.assert_allclose(values, arr)
    assert channels == ["ch0", "ch1"]


def test_extract_numeric_from_npy_bytes_1d_array_reshaped() -> None:
    """_extract_numeric_from_npy_bytes reshapes a 1D array to a column vector."""
    from cobrabox.dataset_loader import _extract_numeric_from_npy_bytes

    arr = np.array([1.0, 2.0, 3.0])
    buf = io.BytesIO()
    np.save(buf, arr)
    values, channels = _extract_numeric_from_npy_bytes(buf.getvalue())
    assert values.shape == (3, 1)
    assert channels == ["ch0"]


def test_extract_numeric_from_npy_bytes_npz_file() -> None:
    """_extract_numeric_from_npy_bytes reads the first array from an .npz file."""
    from cobrabox.dataset_loader import _extract_numeric_from_npy_bytes

    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    buf = io.BytesIO()
    np.savez(buf, signal=arr)
    values, channels = _extract_numeric_from_npy_bytes(buf.getvalue())
    assert values.shape == (2, 2)
    assert channels == ["ch0", "ch1"]


def test_extract_numeric_from_npy_bytes_empty_npz_raises() -> None:
    """_extract_numeric_from_npy_bytes raises when the NPZ archive is empty."""
    from cobrabox.dataset_loader import _extract_numeric_from_npy_bytes

    buf = io.BytesIO()
    np.savez(buf)  # no arrays
    with pytest.raises(ValueError, match="contains no arrays"):
        _extract_numeric_from_npy_bytes(buf.getvalue())


def test_extract_numeric_from_npy_bytes_3d_array_raises() -> None:
    """_extract_numeric_from_npy_bytes raises for arrays with more than 2 dimensions."""
    from cobrabox.dataset_loader import _extract_numeric_from_npy_bytes

    arr = np.ones((2, 3, 4))
    buf = io.BytesIO()
    np.save(buf, arr)
    with pytest.raises(ValueError, match="Expected a 2D array"):
        _extract_numeric_from_npy_bytes(buf.getvalue())


# ---------------------------------------------------------------------------
# _extract_numeric_from_mat_bytes
# ---------------------------------------------------------------------------


def test_extract_numeric_from_mat_bytes_extracts_signal_and_fs() -> None:
    """_extract_numeric_from_mat_bytes reads signal data and sampling rate from .mat."""
    import scipy.io

    from cobrabox.dataset_loader import _extract_numeric_from_mat_bytes

    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    buf = io.BytesIO()
    scipy.io.savemat(buf, {"EEG": arr, "fs": np.array([[512.0]])})
    values, channels, fs = _extract_numeric_from_mat_bytes(buf.getvalue())
    assert values.shape == (2, 2)
    assert fs == pytest.approx(512.0)
    assert channels == ["ch0", "ch1"]


def test_extract_numeric_from_mat_bytes_row_vector_signal() -> None:
    """_extract_numeric_from_mat_bytes handles a row-vector signal (scipy returns (1, N))."""
    import scipy.io

    from cobrabox.dataset_loader import _extract_numeric_from_mat_bytes

    buf = io.BytesIO()
    # MATLAB/scipy always stores 1D arrays as row vectors → (1, N) on load
    scipy.io.savemat(buf, {"signal": np.array([1.0, 2.0, 3.0])})
    values, _channels, fs = _extract_numeric_from_mat_bytes(buf.getvalue())
    assert values.shape == (1, 3)
    assert fs is None


def test_extract_numeric_from_mat_bytes_raises_when_no_2d_array() -> None:
    """_extract_numeric_from_mat_bytes raises when no 2D numeric array is found."""
    import scipy.io

    from cobrabox.dataset_loader import _extract_numeric_from_mat_bytes

    buf = io.BytesIO()
    scipy.io.savemat(buf, {"label": np.array([[[1.0, 2.0]]])})  # 3D, won't qualify
    with pytest.raises(ValueError, match="does not contain a numeric 2D array"):
        _extract_numeric_from_mat_bytes(buf.getvalue())


# ---------------------------------------------------------------------------
# _load_swez_sampling_rate
# ---------------------------------------------------------------------------


def test_load_swez_sampling_rate_returns_none_when_no_info_file(tmp_path: Path) -> None:
    """_load_swez_sampling_rate returns None when no IDxx_info.mat exists."""
    from cobrabox.dataset_loader import _load_swez_sampling_rate

    assert _load_swez_sampling_rate(tmp_path, "ID01") is None


def test_load_swez_sampling_rate_reads_fs_from_info_mat(tmp_path: Path) -> None:
    """_load_swez_sampling_rate reads fs from an IDxx_info.mat sidecar."""
    import scipy.io

    from cobrabox.dataset_loader import _load_swez_sampling_rate

    scipy.io.savemat(str(tmp_path / "ID01_info.mat"), {"fs": np.array([[512.0]])})
    assert _load_swez_sampling_rate(tmp_path, "ID01") == pytest.approx(512.0)


# ---------------------------------------------------------------------------
# _load_swiss_eeg_long
# ---------------------------------------------------------------------------


def test_load_swiss_eeg_long_reads_mat_files(tmp_path: Path) -> None:
    """_load_swiss_eeg_long loads .mat files and returns one SignalData per file."""
    import scipy.io

    from cobrabox.data import SignalData
    from cobrabox.dataset_loader import _load_swiss_eeg_long

    dataset_dir = tmp_path / "swiss_eeg_long"
    dataset_dir.mkdir()

    # channels x time (SWEZ convention)
    eeg = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2 channels, 3 samples
    scipy.io.savemat(str(dataset_dir / "ID01_1h.mat"), {"EEG": eeg})
    scipy.io.savemat(str(dataset_dir / "ID01_info.mat"), {"fs": np.array([[512.0]])})

    out = _load_swiss_eeg_long(dataset_dir)

    assert len(out) == 1
    assert isinstance(out[0], SignalData)
    assert out[0].subjectID == "ID01"
    assert out[0].sampling_rate == pytest.approx(512.0)
    # After transpose: time x channels → shape (3, 2)
    assert out[0].data.sizes["time"] == 3
    assert out[0].data.sizes["space"] == 2


def test_load_swiss_eeg_long_subset_filters_by_subject(tmp_path: Path) -> None:
    """_load_swiss_eeg_long only loads files for subjects in the subset list."""
    import scipy.io

    from cobrabox.dataset_loader import _load_swiss_eeg_long

    dataset_dir = tmp_path / "swiss_eeg_long"
    dataset_dir.mkdir()
    eeg = np.ones((2, 4))
    for subj in ("ID01", "ID02", "ID03"):
        scipy.io.savemat(str(dataset_dir / f"{subj}_1h.mat"), {"EEG": eeg})

    out = _load_swiss_eeg_long(dataset_dir, subset=["ID01", "ID03"])

    subject_ids = {item.subjectID for item in out}
    assert subject_ids == {"ID01", "ID03"}
    assert len(out) == 2


def test_load_swiss_eeg_long_subset_by_stem_filters_exact_files(tmp_path: Path) -> None:
    """_load_swiss_eeg_long with stem-based subset loads only the exact named files."""
    import scipy.io

    from cobrabox.dataset_loader import _load_swiss_eeg_long

    dataset_dir = tmp_path / "swiss_eeg_long"
    dataset_dir.mkdir()
    eeg = np.ones((2, 4))
    for hour in (1, 2, 3):
        scipy.io.savemat(str(dataset_dir / f"ID01_{hour}h.mat"), {"EEG": eeg})

    out = _load_swiss_eeg_long(dataset_dir, subset=["ID01_1h", "ID01_3h"])

    assert len(out) == 2
    # Both items belong to ID01 (subject ID is still derived from the stem).
    assert all(item.subjectID == "ID01" for item in out)


def test_load_swiss_eeg_long_subject_key_subset_still_works(tmp_path: Path) -> None:
    """_load_swiss_eeg_long with subject-key subset still loads all hours for that subject."""
    import scipy.io

    from cobrabox.dataset_loader import _load_swiss_eeg_long

    dataset_dir = tmp_path / "swiss_eeg_long"
    dataset_dir.mkdir()
    eeg = np.ones((2, 4))
    for hour in (1, 2):
        scipy.io.savemat(str(dataset_dir / f"ID01_{hour}h.mat"), {"EEG": eeg})
    scipy.io.savemat(str(dataset_dir / "ID02_1h.mat"), {"EEG": eeg})

    out = _load_swiss_eeg_long(dataset_dir, subset=["ID01"])

    assert len(out) == 2
    assert all(item.subjectID == "ID01" for item in out)


def test_load_swiss_eeg_long_raises_when_no_mat_files(tmp_path: Path) -> None:
    """_load_swiss_eeg_long raises FileNotFoundError when the directory has no .mat files."""
    from cobrabox.dataset_loader import _load_swiss_eeg_long

    dataset_dir = tmp_path / "swiss_eeg_long"
    dataset_dir.mkdir()
    with pytest.raises(FileNotFoundError, match=r"No \.mat files found"):
        _load_swiss_eeg_long(dataset_dir)


def test_load_swiss_eeg_long_raises_when_all_files_unparsable(tmp_path: Path) -> None:
    """_load_swiss_eeg_long raises ValueError when all EEG arrays are empty/missing."""
    import scipy.io

    from cobrabox.dataset_loader import _load_swiss_eeg_long

    dataset_dir = tmp_path / "swiss_eeg_long"
    dataset_dir.mkdir()
    # Save a file with no EEG variable and no other 2D numeric array
    scipy.io.savemat(str(dataset_dir / "ID01_1h.mat"), {"note": np.array([0.0])})

    with pytest.raises(ValueError, match="empty or unparsable"):
        _load_swiss_eeg_long(dataset_dir)


def test_load_swiss_eeg_long_raises_on_hdf5_mat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_load_swiss_eeg_long raises a clear RuntimeError for v7.3 HDF5 .mat files."""
    import scipy.io

    from cobrabox.dataset_loader import _load_swiss_eeg_long

    dataset_dir = tmp_path / "swiss_eeg_long"
    dataset_dir.mkdir()

    # Create a valid mat file so glob finds it
    scipy.io.savemat(str(dataset_dir / "ID01_1h.mat"), {"EEG": np.ones((2, 4))})

    # Simulate scipy raising NotImplementedError for v7.3 files
    original_loadmat = scipy.io.loadmat

    def _raise_not_implemented(path: str, **kwargs: object) -> object:
        if "variable_names" in kwargs:
            raise NotImplementedError("HDF5")
        return original_loadmat(path, **kwargs)

    monkeypatch.setattr(scipy.io, "loadmat", _raise_not_implemented)

    with pytest.raises(RuntimeError, match=r"MATLAB v7\.3"):
        _load_swiss_eeg_long(dataset_dir)


# ---------------------------------------------------------------------------
# _load_swiss_eeg_short — empty members branch
# ---------------------------------------------------------------------------


def test_swiss_eeg_short_loader_skips_zip_with_no_members(tmp_path: Path) -> None:
    """Swiss short EEG loader skips archives that contain no files."""
    from cobrabox.data import SignalData
    from cobrabox.dataset_loader import _load_swiss_eeg_short

    dataset_dir = tmp_path / "swiss_eeg_short"
    dataset_dir.mkdir()

    # Empty zip (no members)
    with zipfile.ZipFile(dataset_dir / "ID99.zip", mode="w"):
        pass

    # Valid zip alongside it
    with zipfile.ZipFile(dataset_dir / "ID1.zip", mode="w") as zf:
        zf.writestr("signal.csv", "ch0\n1.0\n2.0\n")

    out = _load_swiss_eeg_short(dataset_dir)
    assert len(out) == 1
    assert isinstance(out[0], SignalData)
    assert out[0].subjectID == "ID1"


def test_swiss_eeg_short_loader_raises_when_no_supported_member(tmp_path: Path) -> None:
    """Swiss short EEG loader raises when a zip contains only unsupported file types."""
    from cobrabox.dataset_loader import _load_swiss_eeg_short

    dataset_dir = tmp_path / "swiss_eeg_short"
    dataset_dir.mkdir()

    with zipfile.ZipFile(dataset_dir / "ID1.zip", mode="w") as zf:
        zf.writestr("readme.pdf", b"not a signal")

    with pytest.raises(ValueError, match="no supported numeric member"):
        _load_swiss_eeg_short(dataset_dir)


def test_ensure_remote_files_raises_on_url_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ensure_remote_files raises RuntimeError when a file download hits a network error."""
    files = [RemoteFile(url="http://example.com/a.bin", filename="a.bin")]
    spec = RemoteDatasetSpec(
        identifier="test_url_error",
        local_rel_dir=Path("data") / "remote" / "test_url_error",
        files=files,
        loader=lambda _p, _s=None: Dataset([]),
    )

    import cobrabox.downloader as downloader

    monkeypatch.setattr(
        downloader.urllib.request,
        "urlopen",
        lambda url, *a, **kw: (_ for _ in ()).throw(downloader.urllib.error.URLError("timeout")),
    )

    with pytest.raises(RuntimeError, match="Network error"):
        ensure_remote_files(spec, data_dir=tmp_path, accept=True)


# ---------------------------------------------------------------------------
# RemoteDatasetSpec.subset_keys — edge cases
# ---------------------------------------------------------------------------


def test_subset_keys_returns_none_when_no_subset_key_name() -> None:
    """subset_keys() returns None when subset_key_name is not set."""
    spec = RemoteDatasetSpec(
        identifier="test",
        local_rel_dir=Path("data/test"),
        files=[RemoteFile(url="http://x.com/a.bin", filename="a.bin", subset_key="S1")],
        loader=lambda _p, _s=None: Dataset([]),
    )
    assert spec.subset_keys() is None


def test_subset_keys_returns_none_when_all_files_have_no_subset_key() -> None:
    """subset_keys() returns None when files exist but none have a subset_key."""
    spec = RemoteDatasetSpec(
        identifier="test",
        local_rel_dir=Path("data/test"),
        files=[RemoteFile(url="http://x.com/a.bin", filename="a.bin")],
        loader=lambda _p, _s=None: Dataset([]),
        subset_key_name="subjects",
    )
    assert spec.subset_keys() is None


# ---------------------------------------------------------------------------
# DatasetInfo.__str__ and __repr__
# ---------------------------------------------------------------------------


def test_dataset_info_str_for_dataset_without_subsets() -> None:
    """DatasetInfo.__str__ shows 'none' message when subset_key_name is None."""
    from cobrabox.datasets import DatasetInfo

    info = DatasetInfo(
        identifier="dummy_chain",
        description="Synthetic chain dataset.",
        subset_key_name=None,
        subsets=None,
    )
    text = str(info)
    assert "none" in text
    assert "cb.dataset" in text


def test_dataset_info_repr_equals_str() -> None:
    """DatasetInfo.__repr__ returns the same output as __str__."""
    from cobrabox.datasets import DatasetInfo

    info = DatasetInfo(
        identifier="dummy_chain", description="Test.", subset_key_name=None, subsets=None
    )
    assert repr(info) == str(info)


# ---------------------------------------------------------------------------
# dataset() — local dataset branches and unknown identifier
# ---------------------------------------------------------------------------


def test_dataset_loads_dummy_noise(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """dataset() delegates to load_noise_dummy for 'dummy_noise'."""
    import xarray as xr

    from cobrabox.data import SignalData

    da = xr.DataArray([[1.0, 2.0]], dims=["time", "space"])
    fake_ds = Dataset([SignalData.from_xarray(da)])

    monkeypatch.setattr(datasets, "load_noise_dummy", lambda ident: fake_ds)

    out = datasets.dataset("dummy_noise")
    assert len(out) == 1


def test_dataset_loads_realistic_swiss(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """dataset() delegates to load_realistic_swiss for 'realistic_swiss'."""
    import xarray as xr

    from cobrabox.data import SignalData

    da = xr.DataArray([[1.0, 2.0]], dims=["time", "space"])
    fake_ds = Dataset([SignalData.from_xarray(da)])

    monkeypatch.setattr(datasets, "load_realistic_swiss", lambda ident: fake_ds)

    out = datasets.dataset("realistic_swiss")
    assert len(out) == 1


def test_dataset_raises_for_unknown_identifier() -> None:
    """dataset() raises ValueError for completely unknown identifiers."""
    with pytest.raises(ValueError, match="Unknown dataset identifier"):
        datasets.dataset("nonexistent_xyz_123")


# ---------------------------------------------------------------------------
# _swez_long_subject_key
# ---------------------------------------------------------------------------


def test_swez_long_subject_key_parses_subject_from_filename() -> None:
    """_swez_long_subject_key extracts the subject ID from a SWEZ filename."""
    from cobrabox.downloader import _swez_long_subject_key

    assert _swez_long_subject_key("ID01_1h.mat") == "ID01"
    assert _swez_long_subject_key("ID18_295h.mat") == "ID18"


def test_swez_long_subject_key_returns_none_for_no_underscore() -> None:
    """_swez_long_subject_key returns None when the filename has no underscore."""
    from cobrabox.downloader import _swez_long_subject_key

    assert _swez_long_subject_key("nounderscorefile.mat") is None


# ---------------------------------------------------------------------------
# _load_bonn_eeg
# ---------------------------------------------------------------------------


def _make_bonn_zip(path: Path, set_letter: str, n_files: int = 3) -> None:
    """Write a minimal Bonn EEG zip with n_files single-column .txt recordings."""
    signal = "\n".join(str(i) for i in range(4096))
    with zipfile.ZipFile(path, mode="w") as zf:
        for i in range(n_files):
            zf.writestr(f"{set_letter}{i:03d}.txt", signal)


def test_load_bonn_eeg_reads_txt_from_zips(tmp_path: Path) -> None:
    """_load_bonn_eeg produces one SignalData per .txt file across all sets."""
    from cobrabox.data import SignalData
    from cobrabox.dataset_loader import _load_bonn_eeg

    dataset_dir = tmp_path / "bonn_eeg"
    dataset_dir.mkdir()
    _make_bonn_zip(dataset_dir / "Z.zip", "Z", n_files=2)
    _make_bonn_zip(dataset_dir / "S.zip", "S", n_files=3)

    out = _load_bonn_eeg(dataset_dir)

    assert len(out) == 5
    assert all(isinstance(item, SignalData) for item in out)


def test_load_bonn_eeg_sets_metadata_correctly(tmp_path: Path) -> None:
    """_load_bonn_eeg assigns correct subjectID, groupID, condition, and sampling rate."""
    from cobrabox.dataset_loader import _BONN_SAMPLING_RATE, _load_bonn_eeg

    dataset_dir = tmp_path / "bonn_eeg"
    dataset_dir.mkdir()
    _make_bonn_zip(dataset_dir / "S.zip", "S", n_files=1)

    out = _load_bonn_eeg(dataset_dir)

    assert len(out) == 1
    item = out[0]
    assert item.subjectID == "S000"
    assert item.groupID == "ictal"
    assert item.condition == "ictal"
    assert item.sampling_rate == pytest.approx(_BONN_SAMPLING_RATE)
    assert item.data.sizes["time"] == 4096
    assert item.data.sizes["space"] == 1
    assert list(item.data.coords["space"].values) == ["ch0"]


def test_load_bonn_eeg_healthy_set_metadata(tmp_path: Path) -> None:
    """_load_bonn_eeg assigns healthy groupID and descriptive condition for set Z."""
    from cobrabox.dataset_loader import _load_bonn_eeg

    dataset_dir = tmp_path / "bonn_eeg"
    dataset_dir.mkdir()
    _make_bonn_zip(dataset_dir / "Z.zip", "Z", n_files=1)

    out = _load_bonn_eeg(dataset_dir)

    assert out[0].groupID == "healthy"
    assert out[0].condition == "healthy_eyes_open"


def test_load_bonn_eeg_subset_filters_by_set(tmp_path: Path) -> None:
    """_load_bonn_eeg only loads zips for sets in the subset list."""
    from cobrabox.dataset_loader import _load_bonn_eeg

    dataset_dir = tmp_path / "bonn_eeg"
    dataset_dir.mkdir()
    for letter in ("Z", "O", "S"):
        _make_bonn_zip(dataset_dir / f"{letter}.zip", letter, n_files=2)

    out = _load_bonn_eeg(dataset_dir, subset=["S"])

    assert len(out) == 2
    assert all(item.subjectID.startswith("S") for item in out)


def test_load_bonn_eeg_raises_when_no_zips(tmp_path: Path) -> None:
    """_load_bonn_eeg raises FileNotFoundError when no zip files are present."""
    from cobrabox.dataset_loader import _load_bonn_eeg

    dataset_dir = tmp_path / "bonn_eeg"
    dataset_dir.mkdir()

    with pytest.raises(FileNotFoundError, match=r"No \.zip files found"):
        _load_bonn_eeg(dataset_dir)


def test_load_bonn_eeg_raises_when_all_txt_unparsable(tmp_path: Path) -> None:
    """_load_bonn_eeg raises ValueError when no .txt files can be parsed."""
    from cobrabox.dataset_loader import _load_bonn_eeg

    dataset_dir = tmp_path / "bonn_eeg"
    dataset_dir.mkdir()

    with zipfile.ZipFile(dataset_dir / "Z.zip", mode="w") as zf:
        zf.writestr("Z000.txt", "not a number\nnot a number\n")

    with pytest.raises(ValueError, match="empty or unparsable"):
        _load_bonn_eeg(dataset_dir)


def test_bonn_eeg_spec_has_five_sets() -> None:
    """bonn_eeg spec has exactly 5 RemoteFile entries, one per set."""
    spec = get_remote_dataset_spec("bonn_eeg")
    assert spec is not None
    assert spec.files is not None
    assert len(spec.files) == 5
    assert {f.subset_key for f in spec.files} == {"Z", "O", "N", "F", "S"}
    assert spec.subset_key_name == "sets"


def test_dataset_info_bonn_eeg_lists_sets() -> None:
    """dataset_info returns the 5 set letters as subsets for bonn_eeg."""
    from cobrabox.datasets import dataset_info

    info = dataset_info("bonn_eeg")
    assert info.subsets is not None
    assert set(info.subsets) == {"Z", "O", "N", "F", "S"}


# ---------------------------------------------------------------------------
# _load_edf_dataset / _load_chb_mit / _load_siena_eeg / _load_sleep_ieeg
# ---------------------------------------------------------------------------


class _MockRaw:
    """Minimal MNE Raw-like object for testing EDF loaders."""

    def __init__(self, n_channels: int = 3, n_samples: int = 512, sfreq: float = 256.0) -> None:
        self._data = np.ones((n_channels, n_samples))
        self.ch_names = [f"EEG {i}" for i in range(n_channels)]
        self.info = {"sfreq": sfreq}

    def get_data(self) -> np.ndarray:
        return self._data


def _patch_mne_read_raw_edf(monkeypatch: pytest.MonkeyPatch, mock_raw: _MockRaw) -> None:
    """Monkeypatch mne.io.read_raw_edf to return mock_raw for any path."""
    import mne

    monkeypatch.setattr(mne.io, "read_raw_edf", lambda path, **kwargs: mock_raw)


def test_load_chb_mit_reads_edf_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_load_chb_mit loads EDF files and returns one SignalData per file."""
    from cobrabox.data import SignalData
    from cobrabox.dataset_loader import _load_chb_mit

    dataset_dir = tmp_path / "chb_mit"
    dataset_dir.mkdir()
    (dataset_dir / "chb01_01.edf").write_bytes(b"fake")
    (dataset_dir / "chb01_02.edf").write_bytes(b"fake")

    mock_raw = _MockRaw(n_channels=3, n_samples=512, sfreq=256.0)
    _patch_mne_read_raw_edf(monkeypatch, mock_raw)

    out = _load_chb_mit(dataset_dir)

    assert len(out) == 2
    assert all(isinstance(item, SignalData) for item in out)
    assert out[0].subjectID == "chb01"
    assert out[0].sampling_rate == pytest.approx(256.0)
    assert out[0].data.sizes["time"] == 512
    assert out[0].data.sizes["space"] == 3


def test_load_chb_mit_subject_id_from_stem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_load_chb_mit extracts subject ID as the part before the first underscore."""
    from cobrabox.dataset_loader import _load_chb_mit

    dataset_dir = tmp_path / "chb_mit"
    dataset_dir.mkdir()
    (dataset_dir / "chb05_17.edf").write_bytes(b"fake")

    _patch_mne_read_raw_edf(monkeypatch, _MockRaw())

    out = _load_chb_mit(dataset_dir)

    assert out[0].subjectID == "chb05"


def test_load_chb_mit_subset_filters_by_subject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_load_chb_mit only loads files for subjects in the subset list."""
    from cobrabox.dataset_loader import _load_chb_mit

    dataset_dir = tmp_path / "chb_mit"
    dataset_dir.mkdir()
    for name in ("chb01_01.edf", "chb02_01.edf", "chb03_01.edf"):
        (dataset_dir / name).write_bytes(b"fake")

    _patch_mne_read_raw_edf(monkeypatch, _MockRaw())

    out = _load_chb_mit(dataset_dir, subset=["chb01", "chb03"])

    subject_ids = {item.subjectID for item in out}
    assert subject_ids == {"chb01", "chb03"}
    assert len(out) == 2


def test_load_chb_mit_raises_when_no_edf_files(tmp_path: Path) -> None:
    """_load_chb_mit raises FileNotFoundError when no EDF files exist."""
    from cobrabox.dataset_loader import _load_chb_mit

    dataset_dir = tmp_path / "chb_mit"
    dataset_dir.mkdir()

    with pytest.raises(FileNotFoundError, match=r"No \.edf files found"):
        _load_chb_mit(dataset_dir)


def test_load_chb_mit_raises_when_all_files_unparsable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_load_chb_mit raises ValueError when MNE fails to parse every file."""
    from cobrabox.dataset_loader import _load_chb_mit

    dataset_dir = tmp_path / "chb_mit"
    dataset_dir.mkdir()
    (dataset_dir / "chb01_01.edf").write_bytes(b"corrupt")

    import mne

    monkeypatch.setattr(
        mne.io, "read_raw_edf", lambda path, **kwargs: (_ for _ in ()).throw(OSError("bad"))
    )  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="empty or unparsable"):
        _load_chb_mit(dataset_dir)


def test_load_chb_mit_raises_on_missing_mne(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_load_chb_mit raises RuntimeError with a helpful message when MNE is unavailable."""
    import builtins

    from cobrabox.dataset_loader import _load_chb_mit

    dataset_dir = tmp_path / "chb_mit"
    dataset_dir.mkdir()
    (dataset_dir / "chb01_01.edf").write_bytes(b"fake")

    original_import = builtins.__import__

    def _block_mne(name: str, *args: object, **kwargs: object) -> object:
        if name == "mne":
            raise ImportError("No module named 'mne'")
        return original_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _block_mne)

    with pytest.raises(RuntimeError, match="MNE"):
        _load_chb_mit(dataset_dir)


# --- Siena ---


def test_load_siena_eeg_reads_edf_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_load_siena_eeg loads EDF files and returns one SignalData per file."""
    from cobrabox.data import SignalData
    from cobrabox.dataset_loader import _load_siena_eeg

    dataset_dir = tmp_path / "siena_eeg"
    dataset_dir.mkdir()
    (dataset_dir / "PN00-1.edf").write_bytes(b"fake")
    (dataset_dir / "PN01-1.edf").write_bytes(b"fake")

    mock_raw = _MockRaw(n_channels=21, n_samples=1024, sfreq=512.0)
    _patch_mne_read_raw_edf(monkeypatch, mock_raw)

    out = _load_siena_eeg(dataset_dir)

    assert len(out) == 2
    assert all(isinstance(item, SignalData) for item in out)
    assert out[0].subjectID == "PN00"
    assert out[0].sampling_rate == pytest.approx(512.0)
    assert out[0].data.sizes["space"] == 21


def test_load_siena_eeg_subject_id_from_stem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_load_siena_eeg extracts subject ID as the part before the first hyphen."""
    from cobrabox.dataset_loader import _load_siena_eeg

    dataset_dir = tmp_path / "siena_eeg"
    dataset_dir.mkdir()
    (dataset_dir / "PN13-2.edf").write_bytes(b"fake")

    _patch_mne_read_raw_edf(monkeypatch, _MockRaw())

    out = _load_siena_eeg(dataset_dir)

    assert out[0].subjectID == "PN13"


def test_load_siena_eeg_subset_filters_by_subject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_load_siena_eeg only loads files for subjects in the subset list."""
    from cobrabox.dataset_loader import _load_siena_eeg

    dataset_dir = tmp_path / "siena_eeg"
    dataset_dir.mkdir()
    for name in ("PN00-1.edf", "PN01-1.edf", "PN02-1.edf"):
        (dataset_dir / name).write_bytes(b"fake")

    _patch_mne_read_raw_edf(monkeypatch, _MockRaw())

    out = _load_siena_eeg(dataset_dir, subset=["PN00", "PN02"])

    assert {item.subjectID for item in out} == {"PN00", "PN02"}


def test_load_siena_eeg_raises_when_no_edf_files(tmp_path: Path) -> None:
    """_load_siena_eeg raises FileNotFoundError when no EDF files exist."""
    from cobrabox.dataset_loader import _load_siena_eeg

    dataset_dir = tmp_path / "siena_eeg"
    dataset_dir.mkdir()

    with pytest.raises(FileNotFoundError, match=r"No \.edf files found"):
        _load_siena_eeg(dataset_dir)


# --- Sleep iEEG ---


def test_load_sleep_ieeg_reads_edf_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_load_sleep_ieeg loads EDF files and sets subject ID from filename prefix."""
    from cobrabox.data import SignalData
    from cobrabox.dataset_loader import _load_sleep_ieeg

    dataset_dir = tmp_path / "sleep_ieeg"
    dataset_dir.mkdir()
    (dataset_dir / "sub-Detroit001_ses-01_task-sleep_ieeg.edf").write_bytes(b"fake")
    (dataset_dir / "sub-UCLA01_ses-01_task-sleep_ieeg.edf").write_bytes(b"fake")

    mock_raw = _MockRaw(n_channels=64, n_samples=1000, sfreq=1000.0)
    _patch_mne_read_raw_edf(monkeypatch, mock_raw)

    out = _load_sleep_ieeg(dataset_dir)

    assert len(out) == 2
    assert all(isinstance(item, SignalData) for item in out)
    subject_ids = {item.subjectID for item in out}
    assert subject_ids == {"sub-Detroit001", "sub-UCLA01"}
    assert out[0].sampling_rate == pytest.approx(1000.0)


def test_load_sleep_ieeg_subject_id_is_bids_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_load_sleep_ieeg extracts the sub-XXX prefix as the subject ID."""
    from cobrabox.dataset_loader import _load_sleep_ieeg

    dataset_dir = tmp_path / "sleep_ieeg"
    dataset_dir.mkdir()
    (dataset_dir / "sub-Detroit042_ses-01_task-sleep_ieeg.edf").write_bytes(b"fake")

    _patch_mne_read_raw_edf(monkeypatch, _MockRaw())

    out = _load_sleep_ieeg(dataset_dir)

    assert out[0].subjectID == "sub-Detroit042"


def test_load_sleep_ieeg_subset_filters_by_subject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_load_sleep_ieeg only loads files for subjects in the subset list."""
    from cobrabox.dataset_loader import _load_sleep_ieeg

    dataset_dir = tmp_path / "sleep_ieeg"
    dataset_dir.mkdir()
    for subject in ("sub-Detroit001", "sub-Detroit002", "sub-UCLA01"):
        (dataset_dir / f"{subject}_ses-01_task-sleep_ieeg.edf").write_bytes(b"fake")

    _patch_mne_read_raw_edf(monkeypatch, _MockRaw())

    out = _load_sleep_ieeg(dataset_dir, subset=["sub-Detroit001", "sub-UCLA01"])

    assert {item.subjectID for item in out} == {"sub-Detroit001", "sub-UCLA01"}


def test_load_sleep_ieeg_raises_when_no_edf_files(tmp_path: Path) -> None:
    """_load_sleep_ieeg raises FileNotFoundError when no EDF files exist."""
    from cobrabox.dataset_loader import _load_sleep_ieeg

    dataset_dir = tmp_path / "sleep_ieeg"
    dataset_dir.mkdir()

    with pytest.raises(FileNotFoundError, match=r"No \.edf files found"):
        _load_sleep_ieeg(dataset_dir)


def test_sleep_ieeg_subject_key_parses_bids_prefix() -> None:
    """_sleep_ieeg_subject_key extracts the sub-XXX portion of a BIDS filename."""
    from cobrabox.downloader import _sleep_ieeg_subject_key

    assert _sleep_ieeg_subject_key("sub-Detroit001_ses-01_task-sleep_ieeg.edf") == "sub-Detroit001"
    assert _sleep_ieeg_subject_key("sub-UCLA01_ses-01_task-sleep_ieeg.edf") == "sub-UCLA01"


def test_sleep_ieeg_spec_is_registered() -> None:
    """sleep_ieeg is registered in REMOTE_DATASETS with correct metadata."""
    spec = get_remote_dataset_spec("sleep_ieeg")
    assert spec is not None
    assert spec.identifier == "sleep_ieeg"
    assert spec.subset_key_name == "subjects"
    assert spec.files is not None
    assert spec.size_hint == "~13 GB"


# --- Spec registrations ---


def test_chb_mit_spec_is_registered() -> None:
    """chb_mit is registered in REMOTE_DATASETS with correct metadata."""
    spec = get_remote_dataset_spec("chb_mit")
    assert spec is not None
    assert spec.identifier == "chb_mit"
    assert spec.subset_key_name == "subjects"
    assert spec.files is not None


def test_siena_eeg_spec_is_registered() -> None:
    """siena_eeg is registered in REMOTE_DATASETS with correct metadata."""
    spec = get_remote_dataset_spec("siena_eeg")
    assert spec is not None
    assert spec.identifier == "siena_eeg"
    assert spec.subset_key_name == "subjects"
    assert spec.files is not None


def test_list_datasets_returns_sorted_list() -> None:
    """list_datasets() returns a dict with sorted lists for 'local' and 'remote'."""
    from cobrabox.datasets import list_datasets

    result = list_datasets()
    assert isinstance(result, dict)
    assert set(result.keys()) == {"local", "remote"}
    assert result["local"] == sorted(result["local"])
    assert result["remote"] == sorted(result["remote"])


def test_list_datasets_includes_local_datasets() -> None:
    """list_datasets() includes all built-in local datasets under 'local'."""
    from cobrabox.datasets import list_datasets

    result = list_datasets()
    for name in ("dummy_chain", "dummy_random", "dummy_star", "dummy_noise", "realistic_swiss"):
        assert name in result["local"]


def test_list_datasets_includes_remote_datasets() -> None:
    """list_datasets() includes all registered remote datasets under 'remote'."""
    from cobrabox.datasets import list_datasets
    from cobrabox.downloader import REMOTE_DATASETS

    result = list_datasets()
    for name in REMOTE_DATASETS:
        assert name in result["remote"]


def test_list_datasets_accessible_via_cb() -> None:
    """list_datasets is accessible at the top-level cb namespace."""
    import cobrabox as cb

    assert callable(cb.list_datasets)
    assert cb.list_datasets() == cb.datasets.list_datasets()


# ---------------------------------------------------------------------------
# default repo_root resolution
# ---------------------------------------------------------------------------


def test_load_structured_dummy_uses_default_repo_root() -> None:
    """load_structured_dummy works when repo_root is not provided."""
    from cobrabox.dataset_loader import load_structured_dummy

    result = load_structured_dummy("dummy_chain")
    assert len(result) > 0


def test_load_noise_dummy_uses_default_repo_root() -> None:
    """load_noise_dummy works when repo_root is not provided."""
    from cobrabox.dataset_loader import load_noise_dummy

    result = load_noise_dummy()
    assert len(result) > 0


def test_load_realistic_swiss_uses_default_repo_root() -> None:
    """load_realistic_swiss works when repo_root is not provided."""
    from cobrabox.dataset_loader import load_realistic_swiss

    result = load_realistic_swiss()
    assert len(result) > 0


# ---------------------------------------------------------------------------
# _extract_numeric_from_mat_bytes edge cases
# ---------------------------------------------------------------------------


def test_extract_numeric_from_mat_bytes_ignores_unconvertible_fs() -> None:
    """Sampling rate is None when the fs key cannot be converted to float."""
    import scipy.io

    from cobrabox.dataset_loader import _extract_numeric_from_mat_bytes

    buf = io.BytesIO()
    # String-valued fs raises ValueError on float()
    scipy.io.savemat(buf, {"fs": np.array(["bad"]), "signal": np.ones((10, 2))})
    values, _channels, fs = _extract_numeric_from_mat_bytes(buf.getvalue())
    assert fs is None
    assert values.shape == (10, 2)


def test_extract_numeric_from_mat_bytes_1d_signal_becomes_column_vector() -> None:
    """A 1-D signal array is reshaped to (N, 1) with one generated channel."""
    from unittest.mock import patch

    from cobrabox.dataset_loader import _extract_numeric_from_mat_bytes

    # scipy always loads MAT arrays as at least 2-D, so we mock loadmat to
    # supply a genuinely 1-D array and exercise the reshape branch.
    fake_data = {
        "__header__": b"",
        "__version__": "1.0",
        "__globals__": [],
        "signal": np.arange(50, dtype=float),
    }
    with patch("cobrabox.dataset_loader.scipy.io.loadmat", return_value=fake_data):
        values, channels, _ = _extract_numeric_from_mat_bytes(b"fake")
    assert values.shape == (50, 1)
    assert channels == ["ch0"]


# ---------------------------------------------------------------------------
# _load_swez_sampling_rate — exception path
# ---------------------------------------------------------------------------


def test_load_swez_sampling_rate_returns_none_on_corrupt_mat(tmp_path: Path) -> None:
    """Returns None without raising when the info .mat file is unreadable."""
    from cobrabox.dataset_loader import _load_swez_sampling_rate

    info_path = tmp_path / "ID01_info.mat"
    info_path.write_bytes(b"this is not a valid mat file")
    result = _load_swez_sampling_rate(tmp_path, "ID01")
    assert result is None


# ---------------------------------------------------------------------------
# _load_swiss_eeg_long — non-2D EEG array
# ---------------------------------------------------------------------------


def test_load_swiss_eeg_long_skips_non_2d_eeg(tmp_path: Path) -> None:
    """Files whose EEG variable is not 2-D are silently skipped."""
    import scipy.io

    from cobrabox.dataset_loader import _load_swiss_eeg_long

    dataset_dir = tmp_path / "swiss_eeg_long"
    dataset_dir.mkdir()

    # One bad file (3-D EEG, ndim != 2) and one good file.
    bad = dataset_dir / "ID01_1h.mat"
    scipy.io.savemat(str(bad), {"EEG": np.ones((2, 4, 50), dtype=float)})

    good = dataset_dir / "ID01_2h.mat"
    scipy.io.savemat(str(good), {"EEG": np.ones((4, 200), dtype=float)})

    result = _load_swiss_eeg_long(dataset_dir)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# _load_bonn_eeg — multi-dimensional txt signal skipped
# ---------------------------------------------------------------------------


def test_load_bonn_eeg_skips_multidimensional_signal(tmp_path: Path) -> None:
    """Text members that produce a 2-D array are silently skipped."""
    from cobrabox.dataset_loader import _load_bonn_eeg

    dataset_dir = tmp_path / "bonn_eeg"
    dataset_dir.mkdir()
    zip_path = dataset_dir / "S.zip"

    with zipfile.ZipFile(zip_path, "w") as zf:
        # Two columns → np.loadtxt gives shape (N, 2), ndim==2 → skipped.
        zf.writestr("S000.txt", "1.0 2.0\n3.0 4.0\n5.0 6.0\n")
        # One column → valid 1-D signal.
        zf.writestr("S001.txt", "1.0\n2.0\n3.0\n")

    result = _load_bonn_eeg(dataset_dir)
    assert len(result) == 1
    assert result[0].subjectID == "S001"


# ---------------------------------------------------------------------------
# _load_edf_dataset — zero-sample recording skipped
# ---------------------------------------------------------------------------


def test_load_edf_dataset_skips_zero_sample_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EDF files whose data array has 0 samples are silently skipped."""
    from cobrabox.dataset_loader import _load_chb_mit

    dataset_dir = tmp_path / "chb_mit"
    dataset_dir.mkdir()
    (dataset_dir / "chb01_01.edf").write_bytes(b"fake")
    (dataset_dir / "chb01_02.edf").write_bytes(b"fake")

    empty_raw = _MockRaw(n_channels=3, n_samples=0, sfreq=256.0)
    good_raw = _MockRaw(n_channels=3, n_samples=512, sfreq=256.0)
    call_count = 0

    import mne

    def _alternating(path: str, **kwargs: object) -> _MockRaw:
        nonlocal call_count
        call_count += 1
        return empty_raw if call_count == 1 else good_raw

    monkeypatch.setattr(mne.io, "read_raw_edf", _alternating)

    result = _load_chb_mit(dataset_dir)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# _load_swiss_eeg_short — .npy / .mat members and empty-values path
# ---------------------------------------------------------------------------


def test_swiss_eeg_short_loader_reads_npy_from_zip(tmp_path: Path) -> None:
    """_load_swiss_eeg_short can parse a .npy member inside a zip archive."""
    from cobrabox.dataset_loader import _load_swiss_eeg_short

    dataset_dir = tmp_path / "swiss_eeg_short"
    dataset_dir.mkdir(parents=True)
    zip_path = dataset_dir / "ID1.zip"

    arr = np.ones((50, 3), dtype=float)
    buf = io.BytesIO()
    np.save(buf, arr)

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("signal.npy", buf.getvalue())

    result = _load_swiss_eeg_short(dataset_dir)
    assert len(result) == 1
    assert result[0].subjectID == "ID1"


def test_swiss_eeg_short_loader_reads_mat_from_zip(tmp_path: Path) -> None:
    """_load_swiss_eeg_short can parse a .mat member inside a zip archive."""
    import scipy.io

    from cobrabox.dataset_loader import _load_swiss_eeg_short

    dataset_dir = tmp_path / "swiss_eeg_short"
    dataset_dir.mkdir(parents=True)
    zip_path = dataset_dir / "ID2.zip"

    buf = io.BytesIO()
    scipy.io.savemat(buf, {"EEG": np.ones((4, 100), dtype=float), "fs": np.array(256.0)})

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("signal.mat", buf.getvalue())

    result = _load_swiss_eeg_short(dataset_dir)
    assert len(result) == 1
    assert result[0].subjectID == "ID2"
    assert result[0].sampling_rate == 256.0


def test_swiss_eeg_short_loader_skips_member_with_empty_values(tmp_path: Path) -> None:
    """Members whose data has 0 time samples are skipped; next member is tried."""
    from cobrabox.dataset_loader import _load_swiss_eeg_short

    dataset_dir = tmp_path / "swiss_eeg_short"
    dataset_dir.mkdir(parents=True)
    zip_path = dataset_dir / "ID1.zip"

    empty_arr = np.ones((0, 3), dtype=float)
    good_arr = np.ones((20, 2), dtype=float)
    empty_buf = io.BytesIO()
    good_buf = io.BytesIO()
    np.save(empty_buf, empty_arr)
    np.save(good_buf, good_arr)

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("empty.npy", empty_buf.getvalue())
        zf.writestr("signal.npy", good_buf.getvalue())

    result = _load_swiss_eeg_short(dataset_dir)
    assert len(result) == 1


def test_swiss_eeg_short_loader_raises_when_all_archives_empty(tmp_path: Path) -> None:
    """ValueError is raised when every zip archive contains no members."""
    from cobrabox.dataset_loader import _load_swiss_eeg_short

    dataset_dir = tmp_path / "swiss_eeg_short"
    dataset_dir.mkdir(parents=True)

    # Two empty zip files (no members at all) → both skipped → items empty.
    for name in ("ID1.zip", "ID2.zip"):
        with zipfile.ZipFile(dataset_dir / name, "w"):
            pass

    with pytest.raises(ValueError, match="empty or unparsable"):
        _load_swiss_eeg_short(dataset_dir)


# ---------------------------------------------------------------------------
# datasets.py — realistic_swiss and seizures display
# ---------------------------------------------------------------------------


def test_dataset_loads_realistic_swiss_via_cb() -> None:
    """cb.dataset('realistic_swiss') routes to load_realistic_swiss."""
    import cobrabox as cb

    result = cb.dataset("realistic_swiss")
    assert len(result) > 0


def test_dataset_info_str_shows_seizures_per_subject() -> None:
    """DatasetInfo.__str__ includes a seizures/subject block when data is present."""
    from cobrabox.datasets import dataset_info

    info = dataset_info("bonn_eeg")
    text = str(info)
    assert "seizures/subject" in text


# ---------------------------------------------------------------------------
# _load_zurich_ieeg — BrainVision loader
# ---------------------------------------------------------------------------


def _patch_mne_read_raw_brainvision(monkeypatch: pytest.MonkeyPatch, mock_raw: _MockRaw) -> None:
    """Monkeypatch mne.io.read_raw_brainvision to return mock_raw for any path."""
    import mne

    monkeypatch.setattr(mne.io, "read_raw_brainvision", lambda path, **kwargs: mock_raw)


def test_load_zurich_ieeg_reads_vhdr_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_load_zurich_ieeg loads .vhdr files and returns one SignalData per run."""
    from cobrabox.data import SignalData
    from cobrabox.dataset_loader import _load_zurich_ieeg

    dataset_dir = tmp_path / "zurich_ieeg"
    dataset_dir.mkdir()
    for name in (
        "sub-01_ses-interictalsleep_run-01_ieeg.vhdr",
        "sub-01_ses-interictalsleep_run-02_ieeg.vhdr",
    ):
        (dataset_dir / name).write_bytes(b"fake")

    mock_raw = _MockRaw(n_channels=50, n_samples=600000, sfreq=2000.0)
    _patch_mne_read_raw_brainvision(monkeypatch, mock_raw)

    out = _load_zurich_ieeg(dataset_dir)

    assert len(out) == 2
    assert all(isinstance(item, SignalData) for item in out)
    assert out[0].subjectID == "sub-01"
    assert out[0].sampling_rate == pytest.approx(2000.0)
    assert out[0].data.sizes["time"] == 600000
    assert out[0].data.sizes["space"] == 50


def test_load_zurich_ieeg_subject_id_is_bids_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_load_zurich_ieeg extracts the sub-XX prefix as the subject ID."""
    from cobrabox.dataset_loader import _load_zurich_ieeg

    dataset_dir = tmp_path / "zurich_ieeg"
    dataset_dir.mkdir()
    (dataset_dir / "sub-10_ses-interictalsleep_run-03_ieeg.vhdr").write_bytes(b"fake")

    _patch_mne_read_raw_brainvision(monkeypatch, _MockRaw())

    out = _load_zurich_ieeg(dataset_dir)

    assert out[0].subjectID == "sub-10"


def test_load_zurich_ieeg_subset_filters_by_subject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_load_zurich_ieeg only loads files whose subject ID is in the subset list."""
    from cobrabox.dataset_loader import _load_zurich_ieeg

    dataset_dir = tmp_path / "zurich_ieeg"
    dataset_dir.mkdir()
    for name in (
        "sub-01_ses-interictalsleep_run-01_ieeg.vhdr",
        "sub-02_ses-interictalsleep_run-01_ieeg.vhdr",
        "sub-03_ses-interictalsleep_run-01_ieeg.vhdr",
    ):
        (dataset_dir / name).write_bytes(b"fake")

    _patch_mne_read_raw_brainvision(monkeypatch, _MockRaw())

    out = _load_zurich_ieeg(dataset_dir, subset=["sub-01", "sub-03"])

    assert {item.subjectID for item in out} == {"sub-01", "sub-03"}
    assert len(out) == 2


def test_load_zurich_ieeg_raises_when_no_vhdr_files(tmp_path: Path) -> None:
    """_load_zurich_ieeg raises FileNotFoundError when no .vhdr files exist."""
    from cobrabox.dataset_loader import _load_zurich_ieeg

    dataset_dir = tmp_path / "zurich_ieeg"
    dataset_dir.mkdir()

    with pytest.raises(FileNotFoundError, match=r"No \.vhdr files found"):
        _load_zurich_ieeg(dataset_dir)


def test_load_zurich_ieeg_skips_zero_sample_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_load_zurich_ieeg silently skips runs with 0 samples."""
    from cobrabox.dataset_loader import _load_zurich_ieeg

    dataset_dir = tmp_path / "zurich_ieeg"
    dataset_dir.mkdir()
    for name in (
        "sub-01_ses-interictalsleep_run-01_ieeg.vhdr",
        "sub-01_ses-interictalsleep_run-02_ieeg.vhdr",
    ):
        (dataset_dir / name).write_bytes(b"fake")

    empty_raw = _MockRaw(n_channels=50, n_samples=0, sfreq=2000.0)
    good_raw = _MockRaw(n_channels=50, n_samples=600000, sfreq=2000.0)
    call_count = 0

    import mne

    def _alternating(path: str, **kwargs: object) -> _MockRaw:
        nonlocal call_count
        call_count += 1
        return empty_raw if call_count == 1 else good_raw

    monkeypatch.setattr(mne.io, "read_raw_brainvision", _alternating)

    result = _load_zurich_ieeg(dataset_dir)
    assert len(result) == 1
