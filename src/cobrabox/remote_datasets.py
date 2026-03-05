from __future__ import annotations

import io
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import xarray as xr
from tqdm import tqdm

from .data import Data, SignalData


@dataclass(slots=True)
class RemoteFile:
    """Description of a single remote file belonging to a dataset."""

    url: str
    filename: str  # Relative to the dataset's local directory


RemoteLoader = Callable[[Path], list[Data] | list[SignalData]]


@dataclass(slots=True)
class RemoteDatasetSpec:
    """Specification for a remotely hosted dataset.

    Either ``files`` or ``file_index_url`` must be provided.
    ``file_index_url`` points to a text file containing one absolute URL per
    line; the list of files is resolved lazily on first download.
    ``auth_hint`` is an optional message shown to users when the server
    responds with a 401 or 403 status, e.g. to explain how to obtain
    credentials.
    """

    identifier: str
    local_rel_dir: Path
    files: Sequence[RemoteFile] | None
    loader: RemoteLoader
    file_index_url: str | None = None
    auth_hint: str | None = None

    def __post_init__(self) -> None:
        if self.files is None and self.file_index_url is None:
            raise ValueError(
                f"RemoteDatasetSpec '{self.identifier}' must define either "
                "'files' or 'file_index_url'."
            )


def _default_repo_root() -> Path:
    """Infer the repository root from the package path."""
    return Path(__file__).resolve().parents[2]


def _resolve_files_from_index(spec: RemoteDatasetSpec) -> Sequence[RemoteFile]:
    """Resolve RemoteFile entries from a remote index text file.

    The index is expected to contain one absolute URL per line. Empty lines are
    ignored. The local filename is taken as the last path component.
    """
    assert spec.file_index_url is not None  # guarded by __post_init__

    try:
        with urllib.request.urlopen(spec.file_index_url) as response:
            raw = response.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Failed to download file index for remote dataset '{spec.identifier}' "
            f"from {spec.file_index_url!r}: HTTP {e.code}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Network error while downloading file index for remote dataset "
            f"'{spec.identifier}' from {spec.file_index_url!r}: {e.reason!r}"
        ) from e

    urls = [line.strip() for line in raw.decode("utf-8").splitlines() if line.strip()]
    files = [RemoteFile(url=url, filename=url.rsplit("/", 1)[-1]) for url in urls]
    # Cache the resolved files on the spec for subsequent calls.
    spec.files = files
    return files


def ensure_remote_files(spec: RemoteDatasetSpec, *, repo_root: Path | None = None) -> Path:
    """Ensure all files for a remote dataset are present locally.

    Files are stored under ``repo_root / spec.local_rel_dir``. Existing files
    are left untouched; missing files are streamed down in parallel and written
    atomically via a ``.part`` temp file.

    Returns the resolved local dataset directory.
    """
    if repo_root is None:
        repo_root = _default_repo_root()

    dataset_dir = repo_root / spec.local_rel_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)

    files = spec.files if spec.files is not None else _resolve_files_from_index(spec)

    def _download_one(remote_file: RemoteFile, position: int) -> None:
        dest_path = dataset_dir / remote_file.filename
        if _has_valid_local_copy(remote_file):
            return

        url = remote_file.url
        tmp_path = dest_path.with_name(dest_path.name + ".part")

        try:
            with urllib.request.urlopen(url) as response:
                content_length = response.headers.get("Content-Length")
                total_size = int(content_length) if content_length else None
                with (
                    open(tmp_path, "wb") as f,
                    tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=remote_file.filename,
                        position=position,
                        leave=True,
                    ) as bar,
                ):
                    for chunk in iter(lambda: response.read(65536), b""):
                        f.write(chunk)
                        bar.update(len(chunk))
        except urllib.error.HTTPError as e:
            tmp_path.unlink(missing_ok=True)
            if spec.auth_hint and e.code in {401, 403}:
                raise RuntimeError(f"{spec.auth_hint}\nExpected file location: {dest_path}") from e
            raise RuntimeError(
                f"Failed to download remote dataset file for '{spec.identifier}' "
                f"from {url!r}: HTTP {e.code}"
            ) from e
        except urllib.error.URLError as e:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Network error while downloading remote dataset file for "
                f"'{spec.identifier}' from {url!r}: {e.reason!r}"
            ) from e
        except Exception as e:  # pragma: no cover - defensive catch-all
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Unexpected error while downloading remote dataset file for "
                f"'{spec.identifier}' from {url!r}: {e!r}"
            ) from e

        tmp_path.replace(dest_path)

    def _has_valid_local_copy(remote_file: RemoteFile) -> bool:
        path = dataset_dir / remote_file.filename
        if not path.exists():
            return False
        if path.suffix.lower() != ".zip":
            return True
        try:
            with zipfile.ZipFile(path) as zf:
                # testzip() returns first bad member name or None if all are valid.
                return zf.testzip() is None
        except Exception:
            return False

    to_download = [f for f in files if not _has_valid_local_copy(f)]

    if not to_download:
        return dataset_dir

    max_workers = min(4, len(to_download))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_download_one, remote_file, position)
            for position, remote_file in enumerate(to_download)
        ]
        for fut in futures:
            fut.result()

    return dataset_dir


def _placeholder_loader(identifier: str) -> RemoteLoader:
    """Create a loader that raises a clear message until implemented."""

    def _loader(_dataset_dir: Path) -> list[Data] | list[SignalData]:
        raise NotImplementedError(
            f"Loader for remote dataset '{identifier}' is not implemented yet. "
            f"Files should now be available under the configured 'data/remote' "
            f"directory. Inspect the raw data layout and implement a proper "
            f"loader to convert it into cobrabox Data/SignalData objects."
        )

    return _loader


def _extract_numeric_from_csv_bytes(raw: bytes) -> tuple[np.ndarray, list[str]]:
    """Parse a CSV-like payload and return numeric values plus channel names."""
    df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        raise ValueError("CSV member has no numeric columns")
    values = numeric_df.to_numpy(dtype=float, copy=False)
    channels = [str(c) for c in numeric_df.columns]
    return values, channels


def _extract_numeric_from_npy_bytes(raw: bytes) -> tuple[np.ndarray, list[str]]:
    """Parse .npy/.npz payload and return a 2D array plus generated channel names."""
    obj = np.load(io.BytesIO(raw), allow_pickle=False)
    if isinstance(obj, np.lib.npyio.NpzFile):
        if not obj.files:
            raise ValueError("NPZ archive contains no arrays")
        arr = np.asarray(obj[obj.files[0]], dtype=float)
    else:
        arr = np.asarray(obj, dtype=float)

    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {arr.shape}")

    channels = [f"ch{i}" for i in range(arr.shape[1])]
    return arr, channels


def _extract_numeric_from_mat_bytes(raw: bytes) -> tuple[np.ndarray, list[str], float | None]:
    """Parse .mat payload and return signal data, channel names, and optional sampling rate."""
    data = scipy.io.loadmat(io.BytesIO(raw))
    sampling_rate: float | None = None
    for key in ("fs", "sampling_rate", "Fs", "srate"):
        if key in data:
            try:
                sampling_rate = float(np.asarray(data[key]).squeeze())
                break
            except TypeError:
                pass
            except ValueError:
                pass

    candidate: np.ndarray | None = None
    for key, value in data.items():
        if key.startswith("__"):
            continue
        arr = np.asarray(value)
        if not np.issubdtype(arr.dtype, np.number):
            continue
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        if arr.ndim == 2:
            candidate = np.asarray(arr, dtype=float)
            break

    if candidate is None:
        raise ValueError("MAT member does not contain a numeric 2D array")

    channels = [f"ch{i}" for i in range(candidate.shape[1])]
    return candidate, channels, sampling_rate


def _load_swiss_eeg_short(dataset_dir: Path) -> list[SignalData]:
    """Load Swiss short EEG zip archives into SignalData objects.

    One SignalData object is produced per archive by reading the first supported
    numeric member found inside each zip file.
    """
    zip_paths = sorted(dataset_dir.glob("*.zip"))
    if not zip_paths:
        raise FileNotFoundError(f"No .zip files found for 'swiss_eeg_short' in {dataset_dir}.")

    datasets: list[SignalData] = []
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as zf:
            members = [name for name in zf.namelist() if not name.endswith("/")]
            if not members:
                continue

            parsed = False
            for member in members:
                suffix = Path(member).suffix.lower()
                if suffix not in {".csv", ".txt", ".tsv", ".npy", ".npz", ".mat"}:
                    continue

                raw = zf.read(member)
                sampling_rate: float | None = None
                if suffix in {".csv", ".txt", ".tsv"}:
                    values, channels = _extract_numeric_from_csv_bytes(raw)
                elif suffix in {".npy", ".npz"}:
                    values, channels = _extract_numeric_from_npy_bytes(raw)
                else:
                    values, channels, sampling_rate = _extract_numeric_from_mat_bytes(raw)

                if values.shape[0] == 0:
                    continue

                time = (
                    np.arange(values.shape[0], dtype=float) / sampling_rate
                    if sampling_rate
                    else np.arange(values.shape[0], dtype=float)
                )
                da = xr.DataArray(
                    values,
                    dims=["time", "space"],
                    coords={"time": time, "space": channels},
                    attrs={
                        "identifier": "swiss_eeg_short",
                        "source_archive": zip_path.name,
                        "source_member": member,
                    },
                )
                datasets.append(
                    SignalData.from_xarray(da, sampling_rate=sampling_rate, subjectID=zip_path.stem)
                )
                parsed = True
                break

            if not parsed:
                raise ValueError(f"{zip_path.name}: no supported numeric member found in archive.")

    if not datasets:
        raise ValueError("All swiss_eeg_short archives were empty or unparsable.")
    return datasets


def _swiss_eeg_short_spec() -> RemoteDatasetSpec:
    base_url = "https://iis-people.ee.ethz.ch/~ieeg/BioCAS2018/dataset"
    ids = [
        "ID1",
        "ID2",
        "ID4a",
        "ID4b",
        "ID5",
        "ID6",
        "ID7",
        "ID8",
        "ID9",
        "ID10",
        "ID11",
        "ID12",
        "ID13a",
        "ID13b",
        "ID14a",
        "ID14b",
        "ID15",
        "ID16",
    ]
    return RemoteDatasetSpec(
        identifier="swiss_eeg_short",
        local_rel_dir=Path("data") / "remote" / "swiss_eeg_short",
        files=[RemoteFile(url=f"{base_url}/{id_}.zip", filename=f"{id_}.zip") for id_ in ids],
        loader=_load_swiss_eeg_short,
    )


def _swiss_eeg_long_spec() -> RemoteDatasetSpec:
    return RemoteDatasetSpec(
        identifier="swiss_eeg_long",
        local_rel_dir=Path("data") / "remote" / "swiss_eeg_long",
        files=None,
        loader=_placeholder_loader("swiss_eeg_long"),
        file_index_url="http://ieeg-swez.ethz.ch/longterm-files.txt",
    )


def _bonn_eeg_spec() -> RemoteDatasetSpec:
    return RemoteDatasetSpec(
        identifier="bonn_eeg",
        local_rel_dir=Path("data") / "remote" / "bonn_eeg",
        files=[
            RemoteFile(
                url="https://www.kaggle.com/api/v1/datasets/download/quands/eeg-dataset",
                filename="bonn_eeg.zip",
            )
        ],
        loader=_placeholder_loader("bonn_eeg"),
        auth_hint=(
            "Remote dataset 'bonn_eeg' appears to require Kaggle authentication. "
            "Download the dataset from https://www.kaggle.com/datasets/quands/eeg-dataset "
            "and place the zip file at the path shown below, or configure Kaggle API "
            "credentials so the URL can be accessed directly."
        ),
    )


REMOTE_DATASETS: dict[str, RemoteDatasetSpec] = {
    "swiss_eeg_short": _swiss_eeg_short_spec(),
    "swiss_eeg_long": _swiss_eeg_long_spec(),
    "bonn_eeg": _bonn_eeg_spec(),
}


def get_remote_dataset_spec(identifier: str) -> RemoteDatasetSpec | None:
    """Return the RemoteDatasetSpec for a given identifier, if known."""
    return REMOTE_DATASETS.get(identifier)
