from __future__ import annotations

import errno
import socket
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


def _format_bytes(n: int) -> str:
    """Format a byte count as a human-readable string."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} B"  # unreachable, satisfies type checker


def _head_size(url: str, timeout: int = 5) -> int | None:
    """Return Content-Length from a HEAD request, or None if unavailable."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            cl = resp.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception:
        return None


def _prompt_download_verify(spec: RemoteDatasetSpec, to_download: list[RemoteFile]) -> bool:
    """Print dataset info and download summary, then ask the user to confirm.

    Returns ``True`` if the user confirmed, ``False`` otherwise.
    """
    print(f"\nDataset: {spec.identifier}")
    if spec.description:
        print(f"  {spec.description}")

    n = len(to_download)
    print(f"\n  Files to download: {n}")

    # Attempt parallel HEAD requests to estimate total size.
    max_head = min(16, n)
    sample = to_download[:max_head]
    with ThreadPoolExecutor(max_workers=max_head) as executor:
        sizes = list(executor.map(lambda f: _head_size(f.url), sample))

    if None in sizes:
        size_str = "unknown"
    else:
        total_sample = sum(s for s in sizes if s is not None)
        if max_head < n:
            # Extrapolate from sample.
            estimated = int(total_sample * n / max_head)
            size_str = f"~{_format_bytes(estimated)} (estimated)"
        else:
            size_str = _format_bytes(total_sample)

    print(f"  Estimated download size: {size_str}")

    answer = input("\nProceed with download? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


@dataclass(slots=True)
class RemoteFile:
    """Description of a single remote file belonging to a dataset."""

    url: str
    filename: str  # Relative to the dataset's local directory
    subset_key: str | None = None  # Subset identifier this file belongs to (e.g. "ID1")


RemoteLoader = Callable[[Path, "Sequence[str] | None"], Dataset[SignalData]]

# A callable that dynamically resolves the list of remote files for a dataset.
FileIndexFn = Callable[[], "Sequence[RemoteFile]"]

# Type for the subset parameter accepted by :func:`ensure_remote_files` and
# :func:`~cobrabox.datasets.dataset`:
#
#   list[str]
#       All files for those subset keys (existing behaviour).
#
#   dict[str, int | list[str] | None]
#       Per-key file-level selection:
#         int        → first N files for that key (in file-index order)
#         list[str]  → specific filenames for that key
#         None       → all files for that key (same as including it in a plain list)
SubsetSpec = list[str] | dict[str, int | list[str] | None]


@dataclass(slots=True)
class RemoteDatasetSpec:
    """Specification for a remotely hosted dataset.

    Exactly one of ``files``, ``file_index_url``, or ``file_index_fn`` must be
    provided (``files`` may also be ``None`` until lazily resolved).
    ``file_index_url`` points to a text file containing one absolute URL per
    line; the list of files is resolved lazily on first download.
    ``file_index_fn`` is a zero-argument callable that returns the file list;
    use this when the index format requires custom parsing (e.g. JSON APIs).
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
    file_index_fn: FileIndexFn | None = None
    auth_hint: str | None = None
    description: str = ""
    subset_key_name: str | None = None  # e.g. "subjects"
    subset_key_fn: Callable[[str], str | None] | None = None  # filename -> subset key
    known_subset_keys: tuple[str, ...] | None = None  # static list when known upfront
    size_hint: str | None = None  # Approximate total download size, e.g. "~10 MB"
    subset_size_hint: str | None = None  # Approximate size per subset, e.g. "~2 MB per set"
    seizures_per_subject: dict[str, int] | None = None  # Seizure count keyed by subset key
    seizure_info_url: str | None = None  # URL where seizure count information was sourced
    license: str | None = None  # License name / terms, e.g. "CC BY 4.0"
    max_parallel_downloads: int = 4  # Max concurrent file downloads

    def __post_init__(self) -> None:
        if self.files is None and self.file_index_url is None and self.file_index_fn is None:
            raise ValueError(
                f"RemoteDatasetSpec '{self.identifier}' must define either "
                "'files', 'file_index_url', or 'file_index_fn'."
            )

    def subset_keys(self) -> list[str] | None:
        """Return the list of available subset keys, or None if unknown/not applicable."""
        if self.known_subset_keys is not None:
            return list(self.known_subset_keys)
        if self.files is None or self.subset_key_name is None:
            return None
        keys = list(dict.fromkeys(f.subset_key for f in self.files if f.subset_key is not None))
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
        with urllib.request.urlopen(spec.file_index_url, timeout=30) as response:
            raw = response.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Failed to download file index for remote dataset '{spec.identifier}' "
            f"from {spec.file_index_url!r}: HTTP {e.code}"
        ) from e
    except (TimeoutError, urllib.error.URLError) as e:
        raise RuntimeError(
            f"Network error while downloading file index for remote dataset "
            f"'{spec.identifier}' from {spec.file_index_url!r}: {e!r}"
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


def _filter_files_by_dict_subset(
    subset: dict[str, int | list[str] | None], all_files: Sequence[RemoteFile]
) -> list[RemoteFile]:
    """Return the ``RemoteFile`` entries selected by a dict-form subset spec.

    Files without a ``subset_key`` (e.g. shared sidecar files) are always
    included.  For each subject key in *subset*:

    - ``None``      → all files for that key
    - ``int N``     → first *N* files (in file-index order); raises if N < 1
    - ``list[str]`` → files whose ``filename`` is in the list; unknown names
                      raise ``ValueError`` when the full file list is known
    """
    result: list[RemoteFile] = []
    # Sidecar / shared files (no subset_key) are always included.
    result.extend(f for f in all_files if f.subset_key is None)

    for subject_key, value in subset.items():
        subject_files = [f for f in all_files if f.subset_key == subject_key]
        if value is None:
            result.extend(subject_files)
        elif isinstance(value, int):
            if value < 1:
                raise ValueError(
                    f"File count for subset key '{subject_key}' must be >= 1, got {value!r}."
                )
            result.extend(subject_files[:value])
        else:  # list[str]
            if not value:
                raise ValueError(
                    f"File list for subset key '{subject_key}' must be non-empty; "
                    "use None to include all files."
                )
            by_name = {f.filename: f for f in subject_files}
            if by_name:  # only validate when the file list is known
                unknown = [fn for fn in value if fn not in by_name]
                if unknown:
                    raise ValueError(
                        f"Unknown filenames for subset key '{subject_key}': {unknown}.\n"
                        f"Known files: {sorted(by_name)}"
                    )
            result.extend(by_name[fn] for fn in value if fn in by_name)
    return result


def ensure_remote_files(
    spec: RemoteDatasetSpec,
    *,
    subset: SubsetSpec | None = None,
    repo_root: Path | None = None,
    verify: bool = True,
) -> Path:
    """Ensure all files for a remote dataset are present locally.

    Files are stored under ``repo_root / spec.local_rel_dir``. Existing files
    are left untouched; missing files are streamed down in parallel and written
    atomically via a ``.part`` temp file.

    Args:
        spec: The remote dataset specification.
        subset: If given, restrict which files are downloaded.  Accepts either
            a ``list[str]`` of subset keys (e.g. subject IDs) to download all
            files for those keys, or a ``dict`` for file-level control — see
            :data:`SubsetSpec`.  Files without a ``subset_key`` are always
            included.
        repo_root: Override the inferred repository root.
        verify: If ``True`` (default) and there are files to download, show
            dataset info and an estimated download size, then ask the user to
            confirm before proceeding.  Set to ``False`` to skip the prompt.

    Returns the resolved local dataset directory.

    Raises:
        RuntimeError: If ``verify=True`` and the user declines the download.
    """
    if repo_root is None:
        repo_root = _default_repo_root()

    dataset_dir = repo_root / spec.local_rel_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if spec.files is not None:
        all_files = spec.files
    elif spec.file_index_url is not None:
        all_files = _resolve_files_from_index(spec)
    else:
        assert spec.file_index_fn is not None
        resolved = list(spec.file_index_fn())
        spec.files = resolved  # cache for subsequent calls
        all_files = resolved

    if subset is not None:
        if isinstance(subset, dict):
            files = _filter_files_by_dict_subset(subset, all_files)
        else:
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
            with urllib.request.urlopen(url, timeout=120) as response:
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
        except (TimeoutError, urllib.error.URLError) as e:
            tmp_path.unlink(missing_ok=True)
            reason = getattr(e, "reason", str(e))
            if isinstance(e, socket.timeout) or isinstance(reason, socket.timeout):
                raise RuntimeError(
                    f"Download timed out for '{remote_file.filename}' "
                    f"(no data received for 120 s). "
                    f"Check your connection and retry."
                ) from e
            raise RuntimeError(
                f"Network error while downloading remote dataset file for "
                f"'{spec.identifier}' from {url!r}: {reason!r}"
            ) from e
        except OSError as e:
            tmp_path.unlink(missing_ok=True)
            if e.errno == errno.ENOSPC:
                raise RuntimeError(
                    f"No space left on device while downloading '{remote_file.filename}' "
                    f"for dataset '{spec.identifier}'. "
                    f"Free up disk space and retry, or use subset= to download fewer files."
                ) from e
            raise RuntimeError(
                f"Unexpected OS error while downloading remote dataset file for "
                f"'{spec.identifier}' from {url!r}: {e!r}"
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

    if verify:
        confirmed = _prompt_download_verify(spec, to_download)
        if not confirmed:
            raise RuntimeError(
                f"Download of '{spec.identifier}' cancelled by user. "
                "Pass verify=False to skip this prompt."
            )

    max_workers = min(spec.max_parallel_downloads, len(to_download))
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
        size_hint="~11 GB",
        subset_size_hint="~100 MB - 1 GB per subject",
        # Per-subject counts are in Burrello et al. TBME 2019 (doi:10.1109/TBME.2019.2921940).
        seizure_info_url="https://iis-people.ee.ethz.ch/~ieeg/BioCAS2018/",
        license="Free for research and education only; commercial and military use prohibited.",
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
        size_hint=">1 TB (hundreds of hourly files per subject)",
        subset_size_hint="~100-200 GB per subject (~619 MB per hourly file)",
        # 116 seizures total across 18 subjects (Burrello et al., DATE 2019).
        # Per-subject table is in the Laelaps paper; see seizure_info_url.
        seizure_info_url="http://ieeg-swez.ethz.ch/",
        license="Free for research and education only; commercial and military use prohibited.",
    )


# UPF DSpace persistent bitstream UUIDs for the five Bonn EEG sets.
# Source: https://repositori.upf.edu/handle/10230/42894 (DOI: 10.34810/data490)
_BONN_UPF_BITSTREAMS: dict[str, str] = {
    "Z": "bb2c41b0-6c8b-4c9c-b7c0-62f4f87c9b48",  # healthy, eyes open
    "O": "ca3563e5-09bb-4373-84b3-c707e724432a",  # healthy, eyes closed
    "N": "8b2c6173-2538-41ac-bff9-d569755ade66",  # interictal, contralateral
    "F": "91342498-960f-410f-9df8-42f3c4da2b45",  # interictal, focal
    "S": "500b0f1d-aa9b-49c2-abd8-0410acd4b86c",  # ictal
}


def _bonn_eeg_spec() -> RemoteDatasetSpec:
    from .dataset_loader import _load_bonn_eeg  # avoid circular import at module level

    base = "https://repositori.upf.edu/server/api/core/bitstreams"
    return RemoteDatasetSpec(
        identifier="bonn_eeg",
        local_rel_dir=Path("data") / "remote" / "bonn_eeg",
        files=[
            RemoteFile(url=f"{base}/{uuid}/content", filename=f"{letter}.zip", subset_key=letter)
            for letter, uuid in _BONN_UPF_BITSTREAMS.items()
        ],
        loader=_load_bonn_eeg,
        description=(
            "Bonn University EEG dataset (Andrzejak et al. 2001): 5 sets of 100 "
            "single-channel recordings (healthy, interictal, ictal). "
            "Hosted by Universitat Pompeu Fabra (DOI: 10.34810/data490)."
        ),
        subset_key_name="sets",
        size_hint="~10 MB",
        subset_size_hint="~2 MB per set",
        # Set S contains 100 single-channel ictal recordings; Z/O/N/F are seizure-free.
        # Source: Andrzejak et al. 2001 (DOI: 10.34810/data490).
        seizures_per_subject={"Z": 0, "O": 0, "N": 0, "F": 0, "S": 100},
        seizure_info_url="https://repositori.upf.edu/handle/10230/42894",
        license="Free for research and education only; commercial and military use prohibited.",
        max_parallel_downloads=8,
    )


_CHB_MIT_SUBJECTS: tuple[str, ...] = tuple(f"chb{i:02d}" for i in range(1, 25))


def _chb_mit_file_index() -> Sequence[RemoteFile]:
    """Fetch the CHB-MIT file list from the PhysioNet RECORDS file.

    The RECORDS file lists one record name per line in the format
    ``subjectdir/recordname`` (without extension), e.g. ``chb01/chb01_01``.
    Each record corresponds to one EDF file.
    """
    base_url = "https://physionet.org/files/chbmit/1.0.0"
    records_url = f"{base_url}/RECORDS"
    try:
        with urllib.request.urlopen(records_url, timeout=30) as resp:
            lines = resp.read().decode("utf-8").splitlines()
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Failed to fetch CHB-MIT file index from {records_url!r}: HTTP {e.code}"
        ) from e
    except (TimeoutError, urllib.error.URLError) as e:
        raise RuntimeError(
            f"Network error fetching CHB-MIT file index from {records_url!r}: {e!r}"
        ) from e

    files: list[RemoteFile] = []
    for line in lines:
        record = line.strip()
        if not record:
            continue
        parts = record.split("/")
        if len(parts) != 2:
            continue
        subject_id, filename = parts
        if not filename.lower().endswith(".edf"):
            filename = f"{filename}.edf"
        files.append(
            RemoteFile(
                url=f"{base_url}/{subject_id}/{filename}", filename=filename, subset_key=subject_id
            )
        )
    if not files:
        raise RuntimeError("CHB-MIT file index returned no valid records.")
    return files


def _chb_mit_spec() -> RemoteDatasetSpec:
    from .dataset_loader import _load_chb_mit  # avoid circular import at module level

    return RemoteDatasetSpec(
        identifier="chb_mit",
        local_rel_dir=Path("data") / "remote" / "chb_mit",
        files=None,
        loader=_load_chb_mit,
        file_index_fn=_chb_mit_file_index,
        description=(
            "CHB-MIT Scalp EEG Database: pediatric patients with intractable seizures "
            "(24 subjects, 256 Hz, 23 channels, ictal/interictal). "
            "Children's Hospital Boston / MIT. "
            "License: Open Data Commons Attribution License v1.0."
        ),
        subset_key_name="subjects",
        known_subset_keys=_CHB_MIT_SUBJECTS,
        size_hint="~30 GB",
        subset_size_hint="~1.5 GB per subject",
        # Counts sourced from per-subject summary files (chbXX/chbXX-summary.txt).
        # chb21 is a repeat recording from the same patient as chb01.
        seizures_per_subject={
            "chb01": 7,
            "chb02": 3,
            "chb03": 7,
            "chb04": 4,
            "chb05": 5,
            "chb06": 10,
            "chb07": 3,
            "chb08": 5,
            "chb09": 4,
            "chb10": 7,
            "chb11": 3,
            "chb12": 40,
            "chb13": 12,
            "chb14": 8,
            "chb15": 22,
            "chb16": 10,
            "chb17": 3,
            "chb18": 6,
            "chb19": 3,
            "chb20": 8,
            "chb21": 4,
            "chb22": 3,
            "chb23": 7,
            "chb24": 16,
        },
        seizure_info_url="https://physionet.org/content/chbmit/1.0.0/",
        license="Open Data Commons Attribution License v1.0 (ODC-By-1.0)",
    )


_SIENA_SUBJECTS: tuple[str, ...] = (
    "PN00",
    "PN01",
    "PN03",
    "PN05",
    "PN06",
    "PN07",
    "PN09",
    "PN10",
    "PN11",
    "PN12",
    "PN13",
    "PN14",
    "PN16",
    "PN17",
)


def _siena_file_index() -> Sequence[RemoteFile]:
    """Fetch the Siena Scalp EEG file list from the PhysioNet RECORDS file.

    The RECORDS file lists one record name per line in the format
    ``subjectdir/recordname`` (without extension), e.g. ``PN00/PN00-1``.
    """
    base_url = "https://physionet.org/files/siena-scalp-eeg/1.0.0"
    records_url = f"{base_url}/RECORDS"
    try:
        with urllib.request.urlopen(records_url, timeout=30) as resp:
            lines = resp.read().decode("utf-8").splitlines()
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Failed to fetch Siena EEG file index from {records_url!r}: HTTP {e.code}"
        ) from e
    except (TimeoutError, urllib.error.URLError) as e:
        raise RuntimeError(
            f"Network error fetching Siena EEG file index from {records_url!r}: {e!r}"
        ) from e

    files: list[RemoteFile] = []
    for line in lines:
        record = line.strip()
        if not record:
            continue
        parts = record.split("/")
        if len(parts) != 2:
            continue
        subject_id, filename = parts
        if not filename.lower().endswith(".edf"):
            filename = f"{filename}.edf"
        files.append(
            RemoteFile(
                url=f"{base_url}/{subject_id}/{filename}", filename=filename, subset_key=subject_id
            )
        )
    if not files:
        raise RuntimeError("Siena EEG file index returned no valid records.")
    return files


def _siena_eeg_spec() -> RemoteDatasetSpec:
    from .dataset_loader import _load_siena_eeg  # avoid circular import at module level

    return RemoteDatasetSpec(
        identifier="siena_eeg",
        local_rel_dir=Path("data") / "remote" / "siena_eeg",
        files=None,
        loader=_load_siena_eeg,
        file_index_fn=_siena_file_index,
        description=(
            "Siena Scalp EEG Database: adult epilepsy patients with annotated seizures "
            "(14 subjects, 512 Hz, 21+ channels, ictal/interictal). "
            "University of Siena. "
            "License: Creative Commons Attribution 4.0 International (CC BY 4.0)."
        ),
        subset_key_name="subjects",
        known_subset_keys=_SIENA_SUBJECTS,
        size_hint="~15 GB",
        subset_size_hint="~1 GB per subject",
        # Sourced from subject_info.csv (Detti et al. 2020, 47 seizures total).
        # Note: subject IDs skip some numbers (no PN02, PN04, PN08, etc.).
        seizures_per_subject={
            "PN00": 5,
            "PN01": 2,
            "PN03": 2,
            "PN05": 3,
            "PN06": 5,
            "PN07": 1,
            "PN09": 3,
            "PN10": 10,
            "PN11": 1,
            "PN12": 4,
            "PN13": 3,
            "PN14": 4,
            "PN16": 2,
            "PN17": 2,
        },
        seizure_info_url="https://physionet.org/content/siena-scalp-eeg/1.0.0/subject_info.csv",
        license="Creative Commons Attribution 4.0 International (CC-BY-4.0)",
    )


def _open_ieeg_subject_key(filename: str) -> str | None:
    """Parse subject ID from an Open iEEG filename.

    e.g. ``'sub-Detroit001_ses-01_task-sleep_ieeg.edf'`` → ``'sub-Detroit001'``
    """
    stem = filename.rsplit(".", 1)[0]
    return stem.split("_", 1)[0]


def _open_ieeg_file_index() -> Sequence[RemoteFile]:
    """Fetch the subject list from participants.tsv and build the file index.

    Reads the public S3-hosted participants.tsv to discover all subject IDs,
    then constructs one :class:`RemoteFile` per subject pointing to the
    interictal sleep EDF recording.
    """
    base_url = "https://s3.amazonaws.com/openneuro.org/ds005398"
    participants_url = f"{base_url}/participants.tsv"
    try:
        with urllib.request.urlopen(participants_url, timeout=30) as resp:
            content = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Failed to fetch Open iEEG participant list from {participants_url!r}: HTTP {e.code}"
        ) from e
    except (TimeoutError, urllib.error.URLError) as e:
        raise RuntimeError(
            f"Network error fetching Open iEEG participant list from {participants_url!r}: {e!r}"
        ) from e

    files: list[RemoteFile] = []
    for line in content.splitlines()[1:]:  # skip header row
        parts = line.split("\t")
        if not parts or not parts[0].strip().startswith("sub-"):
            continue
        subject = parts[0].strip()  # e.g. "sub-Detroit001"
        filename = f"{subject}_ses-01_task-sleep_ieeg.edf"
        files.append(
            RemoteFile(
                url=f"{base_url}/{subject}/ses-01/ieeg/{filename}",
                filename=filename,
                subset_key=subject,
            )
        )

    if not files:
        raise RuntimeError("Open iEEG participant list returned no valid subjects.")
    return files


def _open_ieeg_spec() -> RemoteDatasetSpec:
    from .dataset_loader import _load_open_ieeg  # avoid circular import at module level

    return RemoteDatasetSpec(
        identifier="open_ieeg",
        local_rel_dir=Path("data") / "remote" / "open_ieeg",
        files=None,
        loader=_load_open_ieeg,
        file_index_fn=_open_ieeg_file_index,
        description=(
            "Open iEEG Dataset: interictal iEEG during slow-wave sleep from 185 epilepsy "
            "patients (135 Detroit at 1000 Hz, 50 UCLA at 2000 Hz). ECoG/sEEG recordings. "
            "License: CC0 (public domain). "
            "DOI: 10.18112/openneuro.ds005398.v1.0.1."
        ),
        subset_key_name="subjects",
        subset_key_fn=_open_ieeg_subject_key,
        size_hint="~13 GB",
        subset_size_hint="~70 MB per subject",
        # Interictal-only dataset: recordings are sleep segments with no ictal events.
        seizures_per_subject=None,
        seizure_info_url="https://openneuro.org/datasets/ds005398/versions/1.0.1",
        license="CC0 1.0 Universal (public domain)",
        max_parallel_downloads=8,
    )


REMOTE_DATASETS: dict[str, RemoteDatasetSpec] = {
    "swiss_eeg_short": _swiss_eeg_short_spec(),
    "swiss_eeg_long": _swiss_eeg_long_spec(),
    "bonn_eeg": _bonn_eeg_spec(),
    "chb_mit": _chb_mit_spec(),
    "siena_eeg": _siena_eeg_spec(),
    "open_ieeg": _open_ieeg_spec(),
}


def get_remote_dataset_spec(identifier: str) -> RemoteDatasetSpec | None:
    """Return the RemoteDatasetSpec for a given identifier, if known."""
    return REMOTE_DATASETS.get(identifier)
