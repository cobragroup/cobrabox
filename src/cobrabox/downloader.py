from __future__ import annotations

import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from .data import SignalData
from .dataset import Dataset


@dataclass(slots=True)
class RemoteFile:
    """Description of a single remote file belonging to a dataset."""

    url: str
    filename: str  # Relative to the dataset's local directory
    subset_key: str | None = None  # Subset identifier this file belongs to (e.g. "ID1")


RemoteLoader = Callable[[Path, "Sequence[str] | None"], Dataset[SignalData]]


@dataclass(slots=True)
class RemoteDatasetSpec:
    """Specification for a remotely hosted dataset.

    Either ``files`` or ``file_index_url`` must be provided.
    ``file_index_url`` points to a text file containing one absolute URL per
    line; the list of files is resolved lazily on first download.
    ``auth_hint`` is an optional message shown to users when the server
    responds with a 401 or 403 status, e.g. to explain how to obtain
    credentials.
    ``subset_key_name`` names what the subset dimension represents (e.g.
    ``"subjects"``). ``None`` means the dataset has no subset concept.
    ``subset_key_fn`` is an optional callable that maps a filename to a subset
    key string; used when resolving files from a remote index so that each
    ``RemoteFile`` gets its ``subset_key`` set automatically.
    ``description`` is a short human-readable description of the dataset.
    """

    identifier: str
    local_rel_dir: Path
    files: Sequence[RemoteFile] | None
    loader: RemoteLoader
    file_index_url: str | None = None
    auth_hint: str | None = None
    description: str = ""
    subset_key_name: str | None = None  # e.g. "subjects"
    subset_key_fn: Callable[[str], str | None] | None = None  # filename -> subset key

    def __post_init__(self) -> None:
        if self.files is None and self.file_index_url is None:
            raise ValueError(
                f"RemoteDatasetSpec '{self.identifier}' must define either "
                "'files' or 'file_index_url'."
            )

    def subset_keys(self) -> list[str] | None:
        """Return the list of available subset keys, or None if unknown/not applicable."""
        if self.files is None or self.subset_key_name is None:
            return None
        keys = [f.subset_key for f in self.files if f.subset_key is not None]
        return keys if keys else None


def _default_repo_root() -> Path:
    """Infer the repository root from the package path."""
    return Path(__file__).resolve().parents[2]


def _resolve_files_from_index(spec: RemoteDatasetSpec) -> Sequence[RemoteFile]:
    """Resolve RemoteFile entries from a remote index text file.

    The index is expected to contain one absolute URL per line. Empty lines are
    ignored. The local filename is taken as the last path component. If
    ``spec.subset_key_fn`` is set, it is called with each filename to populate
    ``RemoteFile.subset_key``.
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
    files = [
        RemoteFile(
            url=url,
            filename=url.rsplit("/", 1)[-1],
            subset_key=spec.subset_key_fn(url.rsplit("/", 1)[-1]) if spec.subset_key_fn else None,
        )
        for url in urls
    ]
    # Cache the resolved files on the spec for subsequent calls.
    spec.files = files
    return files


def ensure_remote_files(
    spec: RemoteDatasetSpec, *, subset: Sequence[str] | None = None, repo_root: Path | None = None
) -> Path:
    """Ensure all files for a remote dataset are present locally.

    Files are stored under ``repo_root / spec.local_rel_dir``. Existing files
    are left untouched; missing files are streamed down in parallel and written
    atomically via a ``.part`` temp file.

    Args:
        spec: The remote dataset specification.
        subset: If given, only download files whose ``subset_key`` is in this
            list. Files without a ``subset_key`` are always included.
        repo_root: Override the inferred repository root.

    Returns the resolved local dataset directory.
    """
    if repo_root is None:
        repo_root = _default_repo_root()

    dataset_dir = repo_root / spec.local_rel_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)

    all_files = spec.files if spec.files is not None else _resolve_files_from_index(spec)

    if subset is not None:
        subset_set = set(subset)
        files = [f for f in all_files if f.subset_key is None or f.subset_key in subset_set]
    else:
        files = list(all_files)

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

    def _loader(_dataset_dir: Path, _subset: Sequence[str] | None = None) -> Dataset[SignalData]:
        raise NotImplementedError(
            f"Loader for remote dataset '{identifier}' is not implemented yet. "
            f"Files should now be available under the configured 'data/remote' "
            f"directory. Inspect the raw data layout and implement a proper "
            f"loader to convert it into cobrabox Data/SignalData objects."
        )

    return _loader


_SWISS_EEG_SHORT_IDS = [
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


def _swiss_eeg_short_spec() -> RemoteDatasetSpec:
    from .dataset_loader import _load_swiss_eeg_short  # avoid circular import at module level

    base_url = "https://iis-people.ee.ethz.ch/~ieeg/BioCAS2018/dataset"
    return RemoteDatasetSpec(
        identifier="swiss_eeg_short",
        local_rel_dir=Path("data") / "remote" / "swiss_eeg_short",
        files=[
            RemoteFile(url=f"{base_url}/{id_}.zip", filename=f"{id_}.zip", subset_key=id_)
            for id_ in _SWISS_EEG_SHORT_IDS
        ],
        loader=_load_swiss_eeg_short,
        description=(
            "Short-term scalp EEG recordings from the BioCAS 2018 challenge "
            f"({len(_SWISS_EEG_SHORT_IDS)} subjects, ictal/interictal)."
        ),
        subset_key_name="subjects",
    )


def _swez_long_subject_key(filename: str) -> str | None:
    """Parse subject ID from a SWEZ long-term filename (e.g. 'ID01_1h.mat' -> 'ID01')."""
    stem = filename.rsplit(".", 1)[0]  # strip extension
    parts = stem.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else None


def _swiss_eeg_long_spec() -> RemoteDatasetSpec:
    from .dataset_loader import _load_swiss_eeg_long  # avoid circular import at module level

    return RemoteDatasetSpec(
        identifier="swiss_eeg_long",
        local_rel_dir=Path("data") / "remote" / "swiss_eeg_long",
        files=None,
        loader=_load_swiss_eeg_long,
        file_index_url="http://ieeg-swez.ethz.ch/longterm-files.txt",
        description=(
            "Long-term intracranial EEG recordings from the SWEZ dataset "
            "(ETH Zurich, 18 subjects, ictal/interictal)."
        ),
        subset_key_name="subjects",
        subset_key_fn=_swez_long_subject_key,
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
        description="Bonn EEG dataset (normal/interictal/ictal) from Kaggle.",
    )


REMOTE_DATASETS: dict[str, RemoteDatasetSpec] = {
    "swiss_eeg_short": _swiss_eeg_short_spec(),
    "swiss_eeg_long": _swiss_eeg_long_spec(),
    "bonn_eeg": _bonn_eeg_spec(),
}


def get_remote_dataset_spec(identifier: str) -> RemoteDatasetSpec | None:
    """Return the RemoteDatasetSpec for a given identifier, if known."""
    return REMOTE_DATASETS.get(identifier)
