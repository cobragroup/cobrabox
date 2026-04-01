from __future__ import annotations

import errno
import importlib.resources
import json
import os
import platform
import socket
import threading
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


def _load_file_index() -> dict[str, list[RemoteFile]]:
    """Load the bundled file index from file_index.json."""
    ref = importlib.resources.files(__package__).joinpath("file_index.json")
    with importlib.resources.as_file(ref) as path:
        raw: dict[str, list[dict[str, str | None]]] = json.loads(path.read_text(encoding="utf-8"))
    return {
        dataset_id: [
            RemoteFile(url=e["url"], filename=e["filename"], subset_key=e.get("subset_key"))
            for e in entries
        ]
        for dataset_id, entries in raw.items()
    }


_FILE_INDEX: dict[str, list[RemoteFile]] | None = None


def _get_file_index() -> dict[str, list[RemoteFile]]:
    global _FILE_INDEX
    if _FILE_INDEX is None:
        _FILE_INDEX = _load_file_index()
    return _FILE_INDEX


def _format_bytes(n: int) -> str:
    """Format a byte count as a human-readable string."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} B"  # unreachable, satisfies type checker


def _prompt_download_verify(spec: RemoteDatasetSpec, to_download: list[RemoteFile] | None) -> bool:
    """Print dataset info, license, and download summary, then ask the user to confirm.

    ``to_download`` may be ``None`` when the file index has not been fetched yet
    (dynamic datasets).  In that case the prompt shows size hints only.

    Returns ``True`` if the user confirmed, ``False`` otherwise.
    """
    print(f"\nDataset: {spec.identifier}")
    if spec.description:
        print(f"  {spec.description}")

    if spec.license is not None:
        print(f"\n  License: {spec.license}")
    if spec.info_url is not None:
        print(f"  More info: {spec.info_url}")

    if to_download is None:
        # File index not yet fetched — show static size hints only.
        print(f"\n  Estimated download size: {spec.size_hint or 'unknown'}")
    else:
        n = len(to_download)
        total_files = len(spec.files) if spec.files is not None else None
        files_str = f"{n}" if total_files is None else f"{n} of {total_files}"
        print(f"\n  Files to download: {files_str}")
        if spec.subset_size_hint is not None and total_files is not None and n < total_files:
            print(f"  Estimated download size: {spec.subset_size_hint} x {n} files")
        else:
            print(f"  Estimated download size: {spec.size_hint or 'unknown'}")

    answer = input("\nProceed with download? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


@dataclass(slots=True)
class RemoteFile:
    """Description of a single remote file belonging to a dataset."""

    url: str
    filename: str  # Relative to the dataset's local directory
    subset_key: str | None = None  # Subset identifier this file belongs to (e.g. "ID1")


RemoteLoader = Callable[[Path, "Sequence[str] | None"], Dataset[SignalData]]

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

    ``auth_hint`` is an optional message shown to users when the server
    responds with a 401 or 403 status, e.g. to explain how to obtain
    credentials.
    ``subset_key_name`` names what the subset dimension represents (e.g.
    ``"subjects"``). ``None`` means the dataset has no subset concept.
    ``subset_key_fn`` is an optional callable that maps a filename to a subset
    key string.
    ``description`` is a short human-readable description of the dataset.
    """

    identifier: str
    local_rel_dir: Path
    files: Sequence[RemoteFile] | None
    loader: RemoteLoader
    auth_hint: str | None = None
    description: str = ""
    subset_key_name: str | None = None  # e.g. "subjects"
    subset_key_fn: Callable[[str], str | None] | None = None  # filename -> subset key
    known_subset_keys: tuple[str, ...] | None = None  # static list when known upfront
    size_hint: str | None = None  # Approximate total download size, e.g. "~10 MB"
    subset_size_hint: str | None = None  # Approximate size per subset, e.g. "~2 MB per set"
    seizures_per_subject: dict[str, int] | None = None  # Seizure count keyed by subset key
    seizure_info_url: str | None = None  # URL where seizure count information was sourced
    info_url: str | None = None  # Landing page / homepage for the dataset
    license: str | None = None  # License name / terms, e.g. "CC BY 4.0"
    max_parallel_downloads: int = 4  # Max concurrent file downloads

    def subset_keys(self) -> list[str] | None:
        """Return the list of available subset keys, or None if unknown/not applicable."""
        if self.known_subset_keys is not None:
            return list(self.known_subset_keys)
        if self.files is None or self.subset_key_name is None:
            return None
        keys = list(dict.fromkeys(f.subset_key for f in self.files if f.subset_key is not None))
        return keys if keys else None


_COBRABOX_CONFIG_PATH = Path.home() / ".cobrabox" / "config.json"


def _read_config_data_dir() -> Path | None:
    """Read the persisted data_dir from ~/.cobrabox/config.json, or return None."""
    try:
        config = json.loads(_COBRABOX_CONFIG_PATH.read_text(encoding="utf-8"))
        value = config.get("data_dir")
        if value:
            return Path(value)
    except Exception:
        pass
    return None


def _default_data_dir() -> Path:
    """Return the default directory for storing downloaded datasets.

    Resolution order:
    1. ``COBRABOX_DATA_DIR`` environment variable (if set).
    2. In-process ``_data_dir`` set via :func:`set_data_dir` (checked in :func:`get_data_dir`).
    3. ``~/.cobrabox/config.json`` → ``{"data_dir": "/some/path"}``
    4. OS-specific user cache directory:
       - Linux  : ``~/.cache/cobrabox``
       - macOS  : ``~/Library/Caches/cobrabox``
       - Windows: ``%LOCALAPPDATA%\\cobrabox``
    """
    env = os.environ.get("COBRABOX_DATA_DIR")
    if env:
        return Path(env)
    config_dir = _read_config_data_dir()
    if config_dir is not None:
        return config_dir
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Caches" / "cobrabox"
    if system == "Windows":
        local_app_data = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(local_app_data) / "cobrabox"
    # Linux and everything else
    xdg = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(xdg) / "cobrabox"


_data_dir: Path | None = None


def get_data_dir() -> Path:
    """Return the directory where downloaded datasets are stored.

    Returns the value set by :func:`set_data_dir`, or the default platform
    cache directory if none has been set.  The directory is not created here;
    it is created on first download.

    Returns:
        Resolved :class:`~pathlib.Path` to the data directory.
    """
    return _data_dir if _data_dir is not None else _default_data_dir()


def set_data_dir(path: str | Path, *, persist: bool = True) -> None:
    """Override the directory where downloaded datasets are stored.

    Call this before :func:`~cobrabox.dataset` to redirect all downloads to a
    custom location.  The directory is created automatically when needed.

    Args:
        path: Absolute or relative path to the desired data directory.
        persist: If ``True`` (default), write the setting to
            ``~/.cobrabox/config.json`` so it survives process restarts.
            Set to ``False`` to only change the in-process value.

    Example::

        cb.set_data_dir("/mnt/data/cobrabox")
        ds = cb.dataset("bonn_eeg", subset=["S"], verify=False)
    """
    global _data_dir
    _data_dir = Path(path)
    if persist:
        config_dir = _COBRABOX_CONFIG_PATH.parent
        config_dir.mkdir(parents=True, exist_ok=True)
        try:
            existing = json.loads(_COBRABOX_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        existing["data_dir"] = str(_data_dir)
        _COBRABOX_CONFIG_PATH.write_text(json.dumps(existing), encoding="utf-8")


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
    data_dir: Path | None = None,
    accept: bool = False,
    force: bool = False,
) -> Path:
    """Ensure all files for a remote dataset are present locally.

    Files are stored under ``data_dir / spec.local_rel_dir``. Existing files
    are left untouched; missing files are streamed down in parallel and written
    atomically via a ``.part`` temp file.  Interrupted downloads are resumed
    automatically using HTTP range requests.

    Args:
        spec: The remote dataset specification.
        subset: If given, restrict which files are downloaded.  Accepts either
            a ``list[str]`` of subset keys (e.g. subject IDs) to download all
            files for those keys, or a ``dict`` for file-level control — see
            :data:`SubsetSpec`.  Files without a ``subset_key`` are always
            included.
        data_dir: Override the data directory for this call.  Defaults to
            :func:`get_data_dir`.
        accept: If ``False`` (default) and there are files to download, show
            the dataset license, estimated download size, and ask the user to
            confirm before proceeding.  Set to ``True`` to skip the prompt
            (e.g. in scripts where you have already accepted the license).
        force: If ``True``, delete any existing local files for the selected
            subset and re-download from scratch.  Useful when a previous
            download is suspected to be corrupt.

    Returns the resolved local dataset directory.

    Raises:
        RuntimeError: If ``accept=False`` and the user declines the download.
    """
    if data_dir is None:
        data_dir = get_data_dir()

    dataset_dir = data_dir / spec.local_rel_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)

    all_files: Sequence[RemoteFile] = spec.files if spec.files is not None else []

    def _get_files() -> list[RemoteFile]:
        if subset is not None:
            if isinstance(subset, dict):
                return _filter_files_by_dict_subset(subset, all_files)
            subset_set = set(subset)
            return [f for f in all_files if f.subset_key is None or f.subset_key in subset_set]
        return list(all_files)

    files: list[RemoteFile] = _get_files()

    manifest_path = dataset_dir / "_manifest.json"
    manifest_lock = threading.Lock()

    def _load_manifest() -> dict[str, int]:
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _update_manifest(filename: str, size: int) -> None:
        with manifest_lock:
            manifest = _load_manifest()
            manifest[filename] = size
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def _download_one(remote_file: RemoteFile, position: int) -> None:
        dest_path = dataset_dir / remote_file.filename
        if _has_valid_local_copy(remote_file):
            return

        url = remote_file.url
        tmp_path = dest_path.with_name(dest_path.name + ".part")

        # Resume from a partial download if one exists.
        resumed_bytes = tmp_path.stat().st_size if tmp_path.exists() else 0

        try:
            request = urllib.request.Request(url)
            if resumed_bytes:
                request.add_header("Range", f"bytes={resumed_bytes}-")
            with urllib.request.urlopen(request, timeout=120) as response:
                content_length = response.headers.get("Content-Length")
                remaining = int(content_length) if content_length else None
                total_size = (resumed_bytes + remaining) if remaining is not None else None
                with (
                    open(tmp_path, "ab" if resumed_bytes else "wb") as f,
                    tqdm(
                        total=total_size,
                        initial=resumed_bytes,
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
        _update_manifest(remote_file.filename, dest_path.stat().st_size)

    def _has_valid_local_copy(remote_file: RemoteFile) -> bool:
        path = dataset_dir / remote_file.filename
        if not path.exists():
            return False
        # Fast path: check the manifest first.
        with manifest_lock:
            manifest = _load_manifest()
        if remote_file.filename in manifest:
            if manifest[remote_file.filename] == path.stat().st_size:
                return True
        # Fallback: for zip files, validate the archive.
        if path.suffix.lower() != ".zip":
            return True
        try:
            with zipfile.ZipFile(path) as zf:
                # testzip() returns first bad member name or None if all are valid.
                return zf.testzip() is None
        except Exception:
            return False

    if force:
        for f in files:
            dest = dataset_dir / f.filename
            dest.unlink(missing_ok=True)
            dest.with_name(dest.name + ".part").unlink(missing_ok=True)

    to_download = [f for f in files if not _has_valid_local_copy(f)]

    if not to_download:
        return dataset_dir

    if not accept:
        confirmed = _prompt_download_verify(spec, to_download)
        if not confirmed:
            raise RuntimeError(
                f"Download of '{spec.identifier}' cancelled by user. "
                "Pass accept=True to skip this prompt."
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
        local_rel_dir=Path("swiss_eeg_short"),
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
        known_subset_keys=tuple(_SWISS_EEG_SHORT_IDS),
        size_hint="~11 GB",
        subset_size_hint="~100 MB - 1 GB per subject",
        # Per-subject counts are in Burrello et al. TBME 2019 (doi:10.1109/TBME.2019.2921940)
        # but the paper PDF is not publicly accessible as plain text.
        seizure_info_url="https://iis-people.ee.ethz.ch/~ieeg/BioCAS2018/",
        info_url="https://iis-people.ee.ethz.ch/~ieeg/BioCAS2018/",
        license="Free for research and education only; commercial and military use prohibited.",
    )


_SWEZ_LONG_SUBJECTS: tuple[str, ...] = tuple(f"ID{i:02d}" for i in range(1, 19))


def _swez_long_subject_key(filename: str) -> str | None:
    """Parse subject ID from a SWEZ long-term filename (e.g. 'ID01_1h.mat' -> 'ID01')."""
    stem = filename.rsplit(".", 1)[0]  # strip extension
    parts = stem.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else None


def _swiss_eeg_long_spec() -> RemoteDatasetSpec:
    from .dataset_loader import _load_swiss_eeg_long  # avoid circular import at module level

    return RemoteDatasetSpec(
        identifier="swiss_eeg_long",
        local_rel_dir=Path("swiss_eeg_long"),
        files=_get_file_index()["swiss_eeg_long"],
        loader=_load_swiss_eeg_long,
        description=(
            "Long-term intracranial EEG recordings from the SWEZ dataset "
            "(ETH Zurich, 18 subjects, ictal/interictal)."
        ),
        subset_key_name="subjects",
        subset_key_fn=_swez_long_subject_key,
        known_subset_keys=_SWEZ_LONG_SUBJECTS,
        size_hint=">1 TB (hundreds of hourly files per subject)",
        subset_size_hint="~100-200 GB per subject (~619 MB per hourly file)",
        # 116 seizures total across 18 subjects (Burrello et al., DATE 2019).
        # Per-subject breakdown is in the Laelaps paper, but the SWEZ website
        # (seizure_info_url) has TLS issues preventing automated access.
        seizure_info_url="http://ieeg-swez.ethz.ch/",
        info_url="http://ieeg-swez.ethz.ch/",
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
        local_rel_dir=Path("bonn_eeg"),
        files=[
            RemoteFile(url=f"{base}/{uuid}/content", filename=f"{letter}.zip", subset_key=letter)
            for letter, uuid in _BONN_UPF_BITSTREAMS.items()
        ],
        loader=_load_bonn_eeg,
        description=(
            "Bonn University EEG dataset (Andrzejak et al. 2001): 5 sets of 100 "
            "single-channel recordings. Sets: Z = healthy eyes open, O = healthy eyes closed, "
            "N = interictal (seizure-free zone), F = interictal (epileptogenic zone), "
            "S = ictal (seizure). "
            "Hosted by Universitat Pompeu Fabra (DOI: 10.34810/data490)."
        ),
        subset_key_name="sets",
        size_hint="~10 MB",
        subset_size_hint="~2 MB per set",
        # Set S contains 100 single-channel ictal recordings; Z/O/N/F are seizure-free.
        # Source: Andrzejak et al. 2001 (DOI: 10.34810/data490).
        seizures_per_subject={"Z": 0, "O": 0, "N": 0, "F": 0, "S": 100},
        seizure_info_url="https://repositori.upf.edu/handle/10230/42894",
        info_url="https://repositori.upf.edu/handle/10230/42894",
        license="Free for research and education only; commercial and military use prohibited.",
        max_parallel_downloads=8,
    )


_CHB_MIT_SUBJECTS: tuple[str, ...] = tuple(f"chb{i:02d}" for i in range(1, 25))


def _chb_mit_spec() -> RemoteDatasetSpec:
    from .dataset_loader import _load_chb_mit  # avoid circular import at module level

    return RemoteDatasetSpec(
        identifier="chb_mit",
        local_rel_dir=Path("chb_mit"),
        files=_get_file_index()["chb_mit"],
        loader=_load_chb_mit,
        description=(
            "CHB-MIT Scalp EEG Database: pediatric patients with intractable seizures "
            "(24 subjects, 256 Hz, 23 channels, ictal/interictal). "
            "Children's Hospital Boston / MIT."
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
        info_url="https://physionet.org/content/chbmit/1.0.0/",
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


def _siena_eeg_spec() -> RemoteDatasetSpec:
    from .dataset_loader import _load_siena_eeg  # avoid circular import at module level

    return RemoteDatasetSpec(
        identifier="siena_eeg",
        local_rel_dir=Path("siena_eeg"),
        files=_get_file_index()["siena_eeg"],
        loader=_load_siena_eeg,
        description=(
            "Siena Scalp EEG Database: adult epilepsy patients with annotated seizures "
            "(14 subjects, 512 Hz, 21+ channels, ictal/interictal). "
            "University of Siena."
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
        info_url="https://physionet.org/content/siena-scalp-eeg/1.0.0/",
        license="Creative Commons Attribution 4.0 International (CC-BY-4.0)",
    )


_OPEN_IEEG_SUBJECTS: tuple[str, ...] = (
    *[f"sub-Detroit{i:03d}" for i in range(1, 96)],
    *[f"sub-UCLA{i:02d}" for i in range(1, 51)],
    *[f"sub-Detroit{i:03d}" for i in range(96, 136)],
)


def _sleep_ieeg_subject_key(filename: str) -> str | None:
    """Parse subject ID from an Sleep iEEG filename.

    e.g. ``'sub-Detroit001_ses-01_task-sleep_ieeg.edf'`` → ``'sub-Detroit001'``
    """
    stem = filename.rsplit(".", 1)[0]
    return stem.split("_", 1)[0]


def _sleep_ieeg_spec() -> RemoteDatasetSpec:
    from .dataset_loader import _load_sleep_ieeg  # avoid circular import at module level

    return RemoteDatasetSpec(
        identifier="sleep_ieeg",
        local_rel_dir=Path("sleep_ieeg"),
        files=_get_file_index()["sleep_ieeg"],
        loader=_load_sleep_ieeg,
        description=(
            "Sleep iEEG Dataset: interictal iEEG during slow-wave sleep from 185 epilepsy "
            "patients (135 Detroit at 1000 Hz, 50 UCLA at 2000 Hz). ECoG/sEEG recordings. "
            "DOI: 10.18112/openneuro.ds005398.v1.0.1."
        ),
        subset_key_name="subjects",
        subset_key_fn=_sleep_ieeg_subject_key,
        known_subset_keys=_OPEN_IEEG_SUBJECTS,
        size_hint="~13 GB",
        subset_size_hint="~70 MB per subject",
        # Interictal-only dataset: recordings are sleep segments with no ictal events.
        seizures_per_subject=None,
        info_url="https://openneuro.org/datasets/ds005398/versions/1.0.1",
        license="CC0 1.0 Universal (public domain)",
        max_parallel_downloads=8,
    )


_ZURICH_IEEG_SUBJECTS: tuple[str, ...] = tuple(f"sub-{i:02d}" for i in range(1, 21))


def _zurich_ieeg_subject_key(filename: str) -> str | None:
    """Parse subject ID from a Zurich iEEG filename.

    e.g. ``'sub-01_ses-interictalsleep_run-01_ieeg.vhdr'`` → ``'sub-01'``
    """
    return filename.split("_", 1)[0]


def _zurich_ieeg_spec() -> RemoteDatasetSpec:
    from .dataset_loader import _load_zurich_ieeg  # avoid circular import at module level

    return RemoteDatasetSpec(
        identifier="zurich_ieeg",
        local_rel_dir=Path("zurich_ieeg"),
        files=_get_file_index()["zurich_ieeg"],
        loader=_load_zurich_ieeg,
        description=(
            "Zurich iEEG HFO Dataset: interictal ECoG during slow-wave sleep from 20 epilepsy "
            "patients (TLE and extra-temporal), with HFO event markings. "
            "2000 Hz, BrainVision format. DOI: 10.18112/openneuro.ds003498.v1.1.1."
        ),
        subset_key_name="subjects",
        subset_key_fn=_zurich_ieeg_subject_key,
        known_subset_keys=_ZURICH_IEEG_SUBJECTS,
        size_hint="~60 GB",
        subset_size_hint="~3 GB per subject (varies by recording nights)",
        info_url="https://openneuro.org/datasets/ds003498/versions/1.1.1",
        license="CC0 1.0 Universal (public domain)",
        max_parallel_downloads=8,
    )


REMOTE_DATASETS: dict[str, RemoteDatasetSpec] = {
    "swiss_eeg_short": _swiss_eeg_short_spec(),
    "swiss_eeg_long": _swiss_eeg_long_spec(),
    "bonn_eeg": _bonn_eeg_spec(),
    "chb_mit": _chb_mit_spec(),
    "siena_eeg": _siena_eeg_spec(),
    "sleep_ieeg": _sleep_ieeg_spec(),
    "zurich_ieeg": _zurich_ieeg_spec(),
}


def get_remote_dataset_spec(identifier: str) -> RemoteDatasetSpec | None:
    """Return the RemoteDatasetSpec for a given identifier, if known."""
    return REMOTE_DATASETS.get(identifier)
