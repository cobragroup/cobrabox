from __future__ import annotations

import errno
import importlib.resources
import json
import os
import platform
import shutil
import socket
import sys
import threading
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from .data import SignalData
from .dataset import Dataset


class DownloadCancelled(Exception):
    """Raised when the user declines a download or deletion at the confirmation prompt."""


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


def _prompt_download_verify(
    spec: RemoteDatasetSpec, to_download: list[RemoteFile] | None, dataset_dir: Path
) -> bool:
    """Show a Rich panel with dataset info and ask the user to confirm the download.

    Returns ``True`` if the user confirmed, ``False`` otherwise.
    """
    lines: list[str] = []
    if spec.description:
        lines.append(spec.description)
        lines.append("")
    if spec.license is not None:
        lines.append(f"[dim]License :[/dim] {spec.license}")
    if spec.info_url is not None:
        lines.append(f"[dim]More info:[/dim] {spec.info_url}")
    lines.append(f"[dim]Save to :[/dim] {dataset_dir}")

    if to_download is None:
        lines.append(f"[dim]Size    :[/dim] {spec.size_hint or 'unknown'}")
    else:
        n_files = len(to_download)
        total_files = len(spec.files) if spec.files is not None else None
        files_str = f"{n_files}" if total_files is None else f"{n_files} of {total_files}"
        n_subjects = len({f.subset_key for f in to_download if f.subset_key is not None})
        is_subset = total_files is not None and n_files < total_files

        subj_str = (
            f" ({n_subjects} subject{'s' if n_subjects != 1 else ''})"
            if n_subjects > 0 and n_subjects < n_files
            else ""
        )
        lines.append(f"[dim]Files   :[/dim] {files_str}{subj_str}")

        if is_subset and spec.subset_size_bytes is not None and n_subjects > 0:
            total_fmt = _format_bytes(spec.subset_size_bytes * n_subjects)
            per_fmt = _format_bytes(spec.subset_size_bytes)
            lines.append(
                f"[dim]Size    :[/dim] {total_fmt}"
                f"  ({per_fmt} \u00d7 {n_subjects} subject{'s' if n_subjects != 1 else ''})"
            )
        elif is_subset and spec.subset_size_hint is not None:
            count = n_subjects if n_subjects > 0 else n_files
            label = "subject" if n_subjects > 0 else "file"
            lines.append(
                f"[dim]Size    :[/dim] {count} \u00d7 {spec.subset_size_hint}"
                f"  ({count} {label}{'s' if count != 1 else ''})"
            )
        else:
            lines.append(f"[dim]Size    :[/dim] {spec.size_hint or 'unknown'}")

    lines.append("\n[dim]Tip: pass [bold]accept=True[/bold] to skip this prompt.[/dim]")
    Console(file=sys.stdout, highlight=False).print(
        Panel("\n".join(lines), title=f"[bold cyan]{spec.identifier}[/bold cyan]")
    )
    answer = input("Proceed with download? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


@dataclass(slots=True)
class RemoteFile:
    """Description of a single remote file belonging to a dataset."""

    url: str
    filename: str  # Relative to the dataset's local directory
    subset_key: str | None = None  # Subset identifier this file belongs to (e.g. "ID1")


RemoteLoader = Callable[[Path, "Sequence[str] | None"], Dataset[SignalData]]

# Type for the subset parameter accepted by :func:`ensure_remote_files` and
# :func:`~cobrabox.datasets.load_dataset`:
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
    ``description`` is a short human-readable description of the dataset.
    """

    identifier: str
    local_rel_dir: Path
    files: Sequence[RemoteFile] | None
    loader: RemoteLoader
    auth_hint: str | None = None
    description: str = ""
    subset_key_name: str | None = None  # e.g. "subjects"
    known_subset_keys: tuple[str, ...] | None = None  # static list when known upfront
    size_hint: str | None = None  # Approximate total download size, e.g. "~10 MB"
    subset_size_hint: str | None = None  # Approximate size per subset, e.g. "~2 MB per set"
    subset_size_bytes: int | None = None  # Per-subject byte count for computed size estimates
    seizures_per_subject: dict[str, int] | None = None  # Seizure count keyed by subset key
    seizure_info_url: str | None = None  # URL where seizure count information was sourced
    info_url: str | None = None  # Landing page / homepage for the dataset
    license: str | None = None  # License name / terms, e.g. "CC BY 4.0"
    max_parallel_downloads: int = 4  # Max concurrent file downloads
    ilae_per_subject: dict[str, int] | None = None  # ILAE surgical outcome per subject
    resected_zone_per_subject: dict[str, list[str]] | None = None  # Resected channels per subject
    excluded_channels_per_subject: dict[str, list[str]] | None = (
        None  # Excluded channels per subject
    )
    all_channels_per_subject: dict[str, list[str]] | None = (
        None  # All bipolar channel pairs per subject
    )

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
    2. In-process ``_data_dir`` set via :func:`set_dataset_dir`
       (checked in :func:`get_dataset_dir`).
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


def get_dataset_dir() -> Path:
    """Return the directory where downloaded datasets are stored.

    Returns the value set by :func:`set_dataset_dir`, or the default platform
    cache directory if none has been set.  The directory is not created here;
    it is created on first download.

    Returns:
        Resolved :class:`~pathlib.Path` to the data directory.
    """
    return _data_dir if _data_dir is not None else _default_data_dir()


def set_dataset_dir(path: str | Path, *, persist: bool = True) -> None:
    """Override the directory where downloaded datasets are stored.

    Call this before :func:`~cobrabox.datasets.load_dataset` to redirect all downloads to a
    custom location.  The directory is created automatically when needed.

    Args:
        path: Absolute or relative path to the desired data directory.
        persist: If ``True`` (default), write the setting to
            ``~/.cobrabox/config.json`` so it survives process restarts.
            Set to ``False`` to only change the in-process value.

    Example::

        cb.set_dataset_dir("/mnt/data/cobrabox")
        ds = cb.load_dataset("bonn_eeg", subset=["S"], accept=True)
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


def _is_dataset_cached(spec: RemoteDatasetSpec) -> bool:
    """Return True if the dataset has any locally cached data files."""
    dataset_dir = get_dataset_dir() / spec.local_rel_dir
    if not dataset_dir.is_dir():
        return False
    return any(
        f
        for f in dataset_dir.iterdir()
        if f.is_file() and f.name != "_manifest.json" and not f.name.endswith(".part")
    )


def delete_remote_files(
    spec: RemoteDatasetSpec,
    *,
    subset: list[str] | None = None,
    confirm: bool = True,
    data_dir: Path | None = None,
) -> None:
    """Delete locally cached files for a remote dataset.

    Args:
        spec: The remote dataset specification.
        subset: If given, only delete files for the listed subset keys.
            ``None`` (default) deletes the entire dataset directory.
        confirm: If ``True`` (default), print a summary and prompt the user
            before deleting.  Set to ``False`` to skip the prompt.
        data_dir: Override the data directory for this call.  Defaults to
            :func:`get_dataset_dir`.

    Raises:
        RuntimeError: If ``confirm=True`` and the user declines the deletion.
    """
    if data_dir is None:
        data_dir = get_dataset_dir()

    dataset_dir = data_dir / spec.local_rel_dir

    if not dataset_dir.exists():
        return

    if subset is None:
        # Delete entire dataset directory.
        data_files = [
            f
            for f in dataset_dir.rglob("*")
            if f.is_file() and f.name != "_manifest.json" and not f.name.endswith(".part")
        ]
        total_size = sum(f.stat().st_size for f in dataset_dir.rglob("*") if f.is_file())

        if confirm:
            lines = [
                f"[dim]Local path:[/dim] {dataset_dir}",
                f"[dim]Files     :[/dim] {len(data_files)}",
                f"[dim]Disk space:[/dim] {_format_bytes(total_size)}",
                "\n[dim]Tip: pass [bold]confirm=False[/bold] to"
                " [bold]cb.delete_dataset()[/bold] to skip this prompt.[/dim]",
            ]
            Console(file=sys.stdout, highlight=False).print(
                Panel("\n".join(lines), title=f"[bold red]{spec.identifier}[/bold red]")
            )
            answer = input("Delete all local files? [y/N] ").strip().lower()
            if answer not in {"y", "yes"}:
                raise DownloadCancelled(
                    f"Deletion of '{spec.identifier}' cancelled. "
                    "Pass confirm=False to skip this prompt."
                )

        shutil.rmtree(dataset_dir)

    else:
        if spec.files is None:
            return

        files: Sequence[RemoteFile] = spec.files
        subset_set = set(subset)
        candidates = [f for f in files if f.subset_key in subset_set]
        existing = [f for f in candidates if (dataset_dir / f.filename).exists()]

        if not existing:
            return

        total_size = sum((dataset_dir / f.filename).stat().st_size for f in existing)

        if confirm:
            lines = [
                f"[dim]Subjects  :[/dim] {', '.join(sorted(subset_set))}",
                f"[dim]Files     :[/dim] {len(existing)}",
                f"[dim]Disk space:[/dim] {_format_bytes(total_size)}",
                "\n[dim]Tip: pass [bold]confirm=False[/bold] to"
                " [bold]cb.delete_dataset()[/bold] to skip this prompt.[/dim]",
            ]
            Console(file=sys.stdout, highlight=False).print(
                Panel("\n".join(lines), title=f"[bold red]{spec.identifier}[/bold red]")
            )
            answer = input("Delete these files? [y/N] ").strip().lower()
            if answer not in {"y", "yes"}:
                raise DownloadCancelled(
                    f"Deletion of '{spec.identifier}' (subset: {sorted(subset_set)}) "
                    "cancelled. Pass confirm=False to skip this prompt."
                )

        manifest_path = dataset_dir / "_manifest.json"
        try:
            manifest: dict[str, int] = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

        for f in existing:
            (dataset_dir / f.filename).unlink(missing_ok=True)
            manifest.pop(f.filename, None)

        if manifest:
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        else:
            manifest_path.unlink(missing_ok=True)


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
            :func:`get_dataset_dir`.
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
        data_dir = get_dataset_dir()

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
    _cancel = threading.Event()

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

    def _download_one(
        remote_file: RemoteFile,
        file_progress: Progress,
        overall_progress: Progress,
        overall_task: TaskID,
    ) -> None:
        dest_path = dataset_dir / remote_file.filename
        if _has_valid_local_copy(remote_file):
            overall_progress.advance(overall_task)
            return

        url = remote_file.url
        tmp_path = dest_path.with_name(dest_path.name + ".part")

        # Resume from a partial download if one exists.
        resumed_bytes = tmp_path.stat().st_size if tmp_path.exists() else 0

        file_task = file_progress.add_task(remote_file.filename, total=None)
        try:
            try:
                request = urllib.request.Request(url)
                if resumed_bytes:
                    request.add_header("Range", f"bytes={resumed_bytes}-")
                with urllib.request.urlopen(request, timeout=120) as response:
                    content_length = response.headers.get("Content-Length")
                    remaining = int(content_length) if content_length else None
                    total_size = (resumed_bytes + remaining) if remaining is not None else None
                    file_progress.update(file_task, total=total_size, completed=resumed_bytes)
                    with open(tmp_path, "ab" if resumed_bytes else "wb") as f:
                        for chunk in iter(lambda: response.read(65536), b""):
                            if _cancel.is_set():
                                return
                            f.write(chunk)
                            file_progress.update(file_task, advance=len(chunk))
            except urllib.error.HTTPError as e:
                tmp_path.unlink(missing_ok=True)
                if spec.auth_hint and e.code in {401, 403}:
                    raise RuntimeError(
                        f"{spec.auth_hint}\nExpected file location: {dest_path}"
                    ) from e
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
        finally:
            file_progress.remove_task(file_task)

        tmp_path.replace(dest_path)
        _update_manifest(remote_file.filename, dest_path.stat().st_size)
        overall_progress.advance(overall_task)

    def _has_valid_local_copy(
        remote_file: RemoteFile, manifest: dict[str, int] | None = None
    ) -> bool:
        path = dataset_dir / remote_file.filename
        if not path.exists():
            return False
        # Use the provided manifest snapshot, or load fresh under the lock.
        if manifest is not None:
            m = manifest
        else:
            with manifest_lock:
                m = _load_manifest()
        if remote_file.filename in m:
            if m[remote_file.filename] == path.stat().st_size:
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

    # Load the manifest once for the pre-flight check (avoids one disk read per file).
    preflight_manifest = _load_manifest()
    to_download = [f for f in files if not _has_valid_local_copy(f, preflight_manifest)]

    if not to_download:
        return dataset_dir

    if not accept:
        confirmed = _prompt_download_verify(spec, to_download, dataset_dir)
        if not confirmed:
            raise DownloadCancelled(
                f"Download of '{spec.identifier}' cancelled. Pass accept=True to skip this prompt."
            )

    max_workers = min(spec.max_parallel_downloads, len(to_download))
    _console = Console(file=sys.stdout)
    overall_progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=_console,
    )
    file_progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}", table_column=None),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=_console,
    )
    progress_group = Group(overall_progress, file_progress)
    try:
        with Live(progress_group, console=_console, refresh_per_second=10):
            overall_task = overall_progress.add_task(spec.identifier, total=len(to_download))
            executor = ThreadPoolExecutor(max_workers=max_workers)
            futures = [
                executor.submit(_download_one, rf, file_progress, overall_progress, overall_task)
                for rf in to_download
            ]
            try:
                for fut in futures:
                    fut.result()
            except KeyboardInterrupt:
                _cancel.set()
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=False)
    except KeyboardInterrupt:
        _console.print(
            "\n[yellow]Download interrupted \u2014"
            " partial files will be resumed on next run.[/yellow]"
        )
        raise

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
        subset_key_name="subsets (subjects)",
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
        subset_key_name="subsets (subjects)",
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
        subset_key_name="subsets",
        size_hint="~10 MB",
        subset_size_hint="~2 MB per subset",
        subset_size_bytes=2 * 1024 * 1024,  # ~2 MB per set
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
        subset_key_name="subsets (subjects)",
        known_subset_keys=_CHB_MIT_SUBJECTS,
        size_hint="~30 GB",
        subset_size_hint="~1.5 GB per subject",
        subset_size_bytes=int(1.5 * 1024**3),  # ~1.5 GB per subject
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
        subset_key_name="subsets (subjects)",
        known_subset_keys=_SIENA_SUBJECTS,
        size_hint="~15 GB",
        subset_size_hint="~1 GB per subject",
        subset_size_bytes=1024**3,  # ~1 GB per subject
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
        subset_key_name="subsets (subjects)",
        known_subset_keys=_OPEN_IEEG_SUBJECTS,
        size_hint="~13 GB",
        subset_size_hint="~70 MB per subject",
        subset_size_bytes=70 * 1024 * 1024,  # ~70 MB per subject
        # Interictal-only dataset: recordings are sleep segments with no ictal events.
        seizures_per_subject=None,
        info_url="https://openneuro.org/datasets/ds005398/versions/1.0.1",
        license="CC0 1.0 Universal (public domain)",
        max_parallel_downloads=8,
    )


_ZURICH_IEEG_SUBJECTS: tuple[str, ...] = tuple(f"sub-{i:02d}" for i in range(1, 21))

# ILAE surgical outcome score (1=seizure-free, 6=no improvement) per subject.
# Source: Patient_Info_Zurich_iEEG_HFOs.xlsx, column "Surgical Outcome".
_ZURICH_ILAE_PER_SUBJECT: dict[str, int] = {
    "sub-01": 1,
    "sub-02": 1,
    "sub-03": 1,
    "sub-04": 1,
    "sub-05": 1,
    "sub-06": 1,
    "sub-07": 3,
    "sub-08": 3,
    "sub-09": 5,
    "sub-10": 1,
    "sub-11": 1,
    "sub-12": 1,
    "sub-13": 1,
    "sub-14": 1,
    "sub-15": 1,
    "sub-16": 1,
    "sub-17": 5,
    "sub-18": 5,
    "sub-19": 6,
    "sub-20": 5,
}

# Bipolar channel pairs in the surgically resected zone per subject.
# Source: Patient_Info_Zurich_iEEG_HFOs.xlsx, column "Resected Electrodes/Channels".
_ZURICH_RESECTED_ZONE_PER_SUBJECT: dict[str, list[str]] = {
    "sub-01": [
        "AHR1-AHR2",
        "AHR2-AHR3",
        "AHR3-AHR4",
        "AR1-AR2",
        "AR2-AR3",
        "AR3-AR4",
        "PHR1-PHR2",
        "PHR2-PHR3",
        "PHR3-PHR4",
    ],
    "sub-02": [
        "AHR1-AHR2",
        "AHR2-AHR3",
        "AHR3-AHR4",
        "AR1-AR2",
        "AR2-AR3",
        "AR3-AR4",
        "ECR1-ECR2",
        "ECR2-ECR3",
        "ECR3-ECR4",
        "PHR1-PHR2",
        "PHR2-PHR3",
        "PHR3-PHR4",
    ],
    "sub-03": [
        "AHL1-AHL2",
        "AHL2-AHL3",
        "AHL3-AHL4",
        "AL1-AL2",
        "AL2-AL3",
        "AL3-AL4",
        "ECL1-ECL2",
        "ECL2-ECL3",
        "ECL3-ECL4",
        "PHL1-PHL2",
        "PHL2-PHL3",
        "PHL3-PHL4",
    ],
    "sub-04": [
        "AHR1-AHR2",
        "AHR2-AHR3",
        "AHR3-AHR4",
        "AR1-AR2",
        "AR2-AR3",
        "AR3-AR4",
        "ECR1-ECR2",
        "ECR2-ECR3",
        "ECR3-ECR4",
        "PHR1-PHR2",
        "PHR2-PHR3",
        "PHR3-PHR4",
    ],
    "sub-05": [
        "AHR1-AHR2",
        "AHR2-AHR3",
        "AHR3-AHR4",
        "AR1-AR2",
        "AR2-AR3",
        "AR3-AR4",
        "ECR1-ECR2",
        "ECR2-ECR3",
        "ECR3-ECR4",
        "PHR1-PHR2",
        "PHR2-PHR3",
        "PHR3-PHR4",
    ],
    "sub-06": [
        "AHR1-AHR2",
        "AHR2-AHR3",
        "AHR3-AHR4",
        "AR1-AR2",
        "AR2-AR3",
        "AR3-AR4",
        "ECR1-ECR2",
        "ECR2-ECR3",
        "ECR3-ECR4",
        "PHR1-PHR2",
        "PHR2-PHR3",
        "PHR3-PHR4",
    ],
    "sub-07": [
        "AHL1-AHL2",
        "AHL2-AHL3",
        "AHL3-AHL4",
        "AL1-AL2",
        "AL2-AL3",
        "AL3-AL4",
        "ECL1-ECL2",
        "ECL2-ECL3",
        "ECL3-ECL4",
        "PHL1-PHL2",
        "PHL2-PHL3",
        "PHL3-PHL4",
    ],
    "sub-08": [
        "AHL1-AHL2",
        "AHL2-AHL3",
        "AHL3-AHL4",
        "AL1-AL2",
        "AL2-AL3",
        "AL3-AL4",
        "ECL1-ECL2",
        "ECL2-ECL3",
        "ECL3-ECL4",
        "PHL1-PHL2",
        "PHL2-PHL3",
        "PHL3-PHL4",
    ],
    "sub-09": [
        "AHL1-AHL2",
        "AHL2-AHL3",
        "AHL3-AHL4",
        "AL1-AL2",
        "AL2-AL3",
        "AL3-AL4",
        "ECL1-ECL2",
        "ECL2-ECL3",
        "ECL3-ECL4",
        "PHL1-PHL2",
        "PHL2-PHL3",
        "PHL3-PHL4",
    ],
    "sub-10": ["IPR1-IPR2", "IPR2-IPR3", "IPR3-IPR4"],
    "sub-11": ["TR1-TR2", "TR2-TR3", "TR3-TR4"],
    "sub-12": [
        "TL1-TL2",
        "TL2-TL3",
        "TL3-TL4",
        "GL1-GL2",
        "GL2-GL3",
        "GL9-GL10",
        "GL10-GL11",
        "GL11-GL12",
        "GL12-GL13",
        "GL17-GL18",
        "GL18-GL19",
        "GL19-GL20",
        "GL20-GL21",
        "GL21-GL22",
        "GL25-GL26",
        "GL26-GL27",
        "GL27-GL28",
        "GL28-GL29",
        "GL29-GL30",
        "GL30-GL31",
        "GL31-GL32",
    ],
    "sub-13": ["TR1-TR2", "TR2-TR3", "TR3-TR4"],
    "sub-14": ["IPR3-IPR4"],
    "sub-15": ["TBAL2-3", "TBAL3-4", "TLL1-2", "TLL9-10"],
    "sub-16": ["TL1-TL2", "TL2-TL3", "TL3-TL4"],
    "sub-17": [
        "TR1-TR2",
        "TR2-TR3",
        "TR3-TR4",
        "TR4-TR5",
        "TR5-TR6",
        "FAR1-FAR2",
        "FAR2-FAR3",
        "FAR3-FAR4",
        "FAR4-FAR5",
        "FAR5-FAR6",
        "FAR6-FAR7",
        "FAR7-FAR8",
        "FAR8-FAR9",
        "FAR9-FAR10",
        "FAR10-FAR11",
        "FAR11-FAR12",
        "FAR12-FAR13",
        "FAR13-FAR14",
        "FAR14-FAR15",
        "FAR15-FAR16",
        "FPR3-FPR4",
        "FPR4-FPR5",
        "FPR5-FPR6",
        "FPR6-FPR7",
        "FPR7-FPR8",
        "FPR11-FPR12",
        "FPR12-FPR13",
        "FPR13-FPR14",
        "FPR14-FPR15",
        "FPR15-FPR16",
    ],
    "sub-18": ["TL1-TL2", "TL2-TL3", "TL3-TL4", "TL4-TL5"],
    "sub-19": ["TL1-TL2", "TL2-TL3", "TL3-TL4", "TL9-TL10", "TL10-TL11", "TL11-TL12"],
    "sub-20": ["OTL4-OTL5", "OTL5-OTL6", "OTL6-OTL7", "OTL13-OTL14", "OTL14-OTL15"],
}

# Bipolar channel pairs excluded due to eloquent cortex (motor/language responses) per subject.
# Source: Patient_Info_Zurich_iEEG_HFOs.xlsx, column "Excluded Electrodes/Channels".
_ZURICH_EXCLUDED_CHANNELS_PER_SUBJECT: dict[str, list[str]] = {
    "sub-01": [],
    "sub-02": [],
    "sub-03": [],
    "sub-04": [],
    "sub-05": [],
    "sub-06": [],
    "sub-07": [],
    "sub-08": [],
    "sub-09": [],
    "sub-10": [],
    "sub-11": [],
    "sub-12": [],
    "sub-13": [
        "GR4-GR5",
        "GR5-GR6",
        "GR6-GR7",
        "GR7-GR8",
        "GR12-GR13",
        "GR13-GR14",
        "GR14-GR15",
        "GR15-GR16",
        "GR20-GR21",
        "GR21-GR22",
        "GR22-GR23",
        "GR23-GR24",
    ],
    "sub-14": [],
    "sub-15": ["TLL6-7", "TLL7-8", "TLL14-15", "TLL15-16", "TLL22-23", "TLL23-24", "TLL31-32"],
    "sub-16": [],
    "sub-17": [],
    "sub-18": ["PML1-PML2", "PML2-PML3"],
    "sub-19": [],
    "sub-20": ["OTL1-OTL2"],
}

# All bipolar channel pairs implanted per subject.
# Source: Patient_Info_Zurich_iEEG_HFOs.xlsx, column "All Electrodes/Channels".
_ZURICH_ALL_CHANNELS_PER_SUBJECT: dict[str, list[str]] = {
    "sub-01": [
        "AHL1-AHL2",
        "AHL2-AHL3",
        "AHL3-AHL4",
        "AHL4-AHL5",
        "AHL5-AHL6",
        "AHL6-AHL7",
        "AHL7-AHL8",
        "AHR1-AHR2",
        "AHR2-AHR3",
        "AHR3-AHR4",
        "AHR4-AHR5",
        "AHR5-AHR6",
        "AHR6-AHR7",
        "AHR7-AHR8",
        "AL1-AL2",
        "AL2-AL3",
        "AL3-AL4",
        "AL4-AL5",
        "AL5-AL6",
        "AL6-AL7",
        "AL7-AL8",
        "AR1-AR2",
        "AR2-AR3",
        "AR3-AR4",
        "AR4-AR5",
        "AR5-AR6",
        "AR6-AR7",
        "AR7-AR8",
        "PHR1-PHR2",
        "PHR2-PHR3",
        "PHR3-PHR4",
        "PHR4-PHR5",
        "PHR5-PHR6",
        "PHR6-PHR7",
        "PHR7-PHR8",
        "IAR1-IAR2",
        "IAR2-IAR3",
        "IAR3-IAR4",
        "IAR4-IAR5",
        "IAR5-IAR6",
        "IPR1-IPR2",
        "IPR2-IPR3",
        "IPR3-IPR4",
    ],
    "sub-02": [
        "AHL1-AHL2",
        "AHL2-AHL3",
        "AHL3-AHL4",
        "AHL4-AHL5",
        "AHL5-AHL6",
        "AHL6-AHL7",
        "AHL7-AHL8",
        "AHR1-AHR2",
        "AHR2-AHR3",
        "AHR3-AHR4",
        "AHR4-AHR5",
        "AHR5-AHR6",
        "AHR6-AHR7",
        "AHR7-AHR8",
        "AL1-AL2",
        "AL2-AL3",
        "AL3-AL4",
        "AL4-AL5",
        "AL5-AL6",
        "AL6-AL7",
        "AL7-AL8",
        "AR1-AR2",
        "AR2-AR3",
        "AR3-AR4",
        "AR4-AR5",
        "AR5-AR6",
        "AR6-AR7",
        "AR7-AR8",
        "ECL1-ECL2",
        "ECL2-ECL3",
        "ECL3-ECL4",
        "ECL4-ECL5",
        "ECL5-ECL6",
        "ECL6-ECL7",
        "ECL7-ECL8",
        "ECR1-ECR2",
        "ECR2-ECR3",
        "ECR3-ECR4",
        "ECR4-ECR5",
        "ECR5-ECR6",
        "ECR6-ECR7",
        "ECR7-ECR8",
        "PHL1-PHL2",
        "PHL2-PHL3",
        "PHL3-PHL4",
        "PHL4-PHL5",
        "PHL5-PHL6",
        "PHL6-PHL7",
        "PHL7-PHL8",
        "PHR1-PHR2",
        "PHR2-PHR3",
        "PHR3-PHR4",
        "PHR4-PHR5",
        "PHR5-PHR6",
        "PHR6-PHR7",
        "PHR7-PHR8",
    ],
    "sub-03": [
        "AHL1-AHL2",
        "AHL2-AHL3",
        "AHL3-AHL4",
        "AHL4-AHL5",
        "AHL5-AHL6",
        "AHL6-AHL7",
        "AHL7-AHL8",
        "AHR1-AHR2",
        "AHR2-AHR3",
        "AHR3-AHR4",
        "AHR4-AHR5",
        "AHR5-AHR6",
        "AHR6-AHR7",
        "AHR7-AHR8",
        "AL1-AL2",
        "AL2-AL3",
        "AL3-AL4",
        "AL4-AL5",
        "AL5-AL6",
        "AL6-AL7",
        "AL7-AL8",
        "ECL1-ECL2",
        "ECL2-ECL3",
        "ECL3-ECL4",
        "ECL4-ECL5",
        "ECL5-ECL6",
        "ECL6-ECL7",
        "ECL7-ECL8",
        "PHL1-PHL2",
        "PHL2-PHL3",
        "PHL3-PHL4",
        "PHL4-PHL5",
        "PHL5-PHL6",
        "PHL6-PHL7",
        "PHL7-PHL8",
    ],
    "sub-04": [
        "AHL1-AHL2",
        "AHL2-AHL3",
        "AHL3-AHL4",
        "AHL4-AHL5",
        "AHL5-AHL6",
        "AHL6-AHL7",
        "AHL7-AHL8",
        "AHR1-AHR2",
        "AHR2-AHR3",
        "AHR3-AHR4",
        "AHR4-AHR5",
        "AHR5-AHR6",
        "AHR6-AHR7",
        "AHR7-AHR8",
        "AL1-AL2",
        "AL2-AL3",
        "AL3-AL4",
        "AL4-AL5",
        "AL5-AL6",
        "AL6-AL7",
        "AL7-AL8",
        "AR1-AR2",
        "AR2-AR3",
        "AR3-AR4",
        "AR4-AR5",
        "AR5-AR6",
        "AR6-AR7",
        "AR7-AR8",
        "ECL1-ECL2",
        "ECL2-ECL3",
        "ECL3-ECL4",
        "ECL4-ECL5",
        "ECL5-ECL6",
        "ECL6-ECL7",
        "ECL7-ECL8",
        "ECR1-ECR2",
        "ECR2-ECR3",
        "ECR3-ECR4",
        "ECR4-ECR5",
        "ECR5-ECR6",
        "ECR6-ECR7",
        "ECR7-ECR8",
        "PHL1-PHL2",
        "PHL2-PHL3",
        "PHL3-PHL4",
        "PHL4-PHL5",
        "PHL5-PHL6",
        "PHL6-PHL7",
        "PHL7-PHL8",
        "PHR1-PHR2",
        "PHR2-PHR3",
        "PHR3-PHR4",
        "PHR4-PHR5",
        "PHR5-PHR6",
        "PHR6-PHR7",
        "PHR7-PHR8",
    ],
    "sub-05": [
        "AHL1-AHL2",
        "AHL2-AHL3",
        "AHL3-AHL4",
        "AHL4-AHL5",
        "AHL5-AHL6",
        "AHL6-AHL7",
        "AHL7-AHL8",
        "AHR1-AHR2",
        "AHR2-AHR3",
        "AHR3-AHR4",
        "AHR4-AHR5",
        "AHR5-AHR6",
        "AHR6-AHR7",
        "AHR7-AHR8",
        "AL1-AL2",
        "AL2-AL3",
        "AL3-AL4",
        "AL4-AL5",
        "AL5-AL6",
        "AL6-AL7",
        "AL7-AL8",
        "AR1-AR2",
        "AR2-AR3",
        "AR3-AR4",
        "AR4-AR5",
        "AR5-AR6",
        "AR6-AR7",
        "AR7-AR8",
        "ECL1-ECL2",
        "ECL2-ECL3",
        "ECL3-ECL4",
        "ECL4-ECL5",
        "ECL5-ECL6",
        "ECL6-ECL7",
        "ECL7-ECL8",
        "ECR1-ECR2",
        "ECR2-ECR3",
        "ECR3-ECR4",
        "ECR4-ECR5",
        "ECR5-ECR6",
        "ECR6-ECR7",
        "ECR7-ECR8",
        "PHL1-PHL2",
        "PHL2-PHL3",
        "PHL3-PHL4",
        "PHL4-PHL5",
        "PHL5-PHL6",
        "PHL6-PHL7",
        "PHL7-PHL8",
        "PHR1-PHR2",
        "PHR2-PHR3",
        "PHR3-PHR4",
        "PHR4-PHR5",
        "PHR5-PHR6",
        "PHR6-PHR7",
        "PHR7-PHR8",
    ],
    "sub-06": [
        "AHL1-AHL2",
        "AHL2-AHL3",
        "AHL3-AHL4",
        "AHL4-AHL5",
        "AHL5-AHL6",
        "AHL6-AHL7",
        "AHL7-AHL8",
        "AHR1-AHR2",
        "AHR2-AHR3",
        "AHR3-AHR4",
        "AHR4-AHR5",
        "AHR5-AHR6",
        "AHR6-AHR7",
        "AHR7-AHR8",
        "AL1-AL2",
        "AL2-AL3",
        "AL3-AL4",
        "AL4-AL5",
        "AL5-AL6",
        "AL6-AL7",
        "AL7-AL8",
        "AR1-AR2",
        "AR2-AR3",
        "AR3-AR4",
        "AR4-AR5",
        "AR5-AR6",
        "AR6-AR7",
        "AR7-AR8",
        "ECL1-ECL2",
        "ECL2-ECL3",
        "ECL3-ECL4",
        "ECL4-ECL5",
        "ECL5-ECL6",
        "ECL6-ECL7",
        "ECL7-ECL8",
        "ECR1-ECR2",
        "ECR2-ECR3",
        "ECR3-ECR4",
        "ECR4-ECR5",
        "ECR5-ECR6",
        "ECR6-ECR7",
        "ECR7-ECR8",
        "PHL1-PHL2",
        "PHL2-PHL3",
        "PHL3-PHL4",
        "PHL4-PHL5",
        "PHL5-PHL6",
        "PHL6-PHL7",
        "PHL7-PHL8",
        "PHR1-PHR2",
        "PHR2-PHR3",
        "PHR3-PHR4",
        "PHR4-PHR5",
        "PHR5-PHR6",
        "PHR6-PHR7",
        "PHR7-PHR8",
    ],
    "sub-07": [
        "AHL1-AHL2",
        "AHL2-AHL3",
        "AHL3-AHL4",
        "AHL4-AHL5",
        "AHL5-AHL6",
        "AHL6-AHL7",
        "AHL7-AHL8",
        "AHR1-AHR2",
        "AHR2-AHR3",
        "AHR3-AHR4",
        "AHR4-AHR5",
        "AHR5-AHR6",
        "AHR6-AHR7",
        "AHR7-AHR8",
        "AL1-AL2",
        "AL2-AL3",
        "AL3-AL4",
        "AL4-AL5",
        "AL5-AL6",
        "AL6-AL7",
        "AL7-AL8",
        "AR1-AR2",
        "AR2-AR3",
        "AR3-AR4",
        "AR4-AR5",
        "AR5-AR6",
        "AR6-AR7",
        "AR7-AR8",
        "ECL1-ECL2",
        "ECL2-ECL3",
        "ECL3-ECL4",
        "ECL4-ECL5",
        "ECL5-ECL6",
        "ECL6-ECL7",
        "ECL7-ECL8",
        "ECR1-ECR2",
        "ECR2-ECR3",
        "ECR3-ECR4",
        "ECR4-ECR5",
        "ECR5-ECR6",
        "ECR6-ECR7",
        "ECR7-ECR8",
        "PHL1-PHL2",
        "PHL2-PHL3",
        "PHL3-PHL4",
        "PHL4-PHL5",
        "PHL5-PHL6",
        "PHL6-PHL7",
        "PHL7-PHL8",
        "PHR1-PHR2",
        "PHR2-PHR3",
        "PHR3-PHR4",
        "PHR4-PHR5",
        "PHR5-PHR6",
        "PHR6-PHR7",
        "PHR7-PHR8",
    ],
    "sub-08": [
        "AHL1-AHL2",
        "AHL2-AHL3",
        "AHL3-AHL4",
        "AHL4-AHL5",
        "AHL5-AHL6",
        "AHL6-AHL7",
        "AHL7-AHL8",
        "AHR1-AHR2",
        "AHR2-AHR3",
        "AHR3-AHR4",
        "AHR4-AHR5",
        "AHR5-AHR6",
        "AHR6-AHR7",
        "AHR7-AHR8",
        "AL1-AL2",
        "AL2-AL3",
        "AL3-AL4",
        "AL4-AL5",
        "AL5-AL6",
        "AL6-AL7",
        "AL7-AL8",
        "AR1-AR2",
        "AR2-AR3",
        "AR3-AR4",
        "AR4-AR5",
        "AR5-AR6",
        "AR6-AR7",
        "AR7-AR8",
        "ECL1-ECL2",
        "ECL2-ECL3",
        "ECL3-ECL4",
        "ECL4-ECL5",
        "ECL5-ECL6",
        "ECL6-ECL7",
        "ECL7-ECL8",
        "ECR1-ECR2",
        "ECR2-ECR3",
        "ECR3-ECR4",
        "ECR4-ECR5",
        "ECR5-ECR6",
        "ECR6-ECR7",
        "ECR7-ECR8",
        "PHL1-PHL2",
        "PHL2-PHL3",
        "PHL3-PHL4",
        "PHL4-PHL5",
        "PHL5-PHL6",
        "PHL6-PHL7",
        "PHL7-PHL8",
        "PHR1-PHR2",
        "PHR2-PHR3",
        "PHR3-PHR4",
        "PHR4-PHR5",
        "PHR5-PHR6",
        "PHR6-PHR7",
        "PHR7-PHR8",
    ],
    "sub-09": [
        "AHL1-AHL2",
        "AHL2-AHL3",
        "AHL3-AHL4",
        "AHL4-AHL5",
        "AHL5-AHL6",
        "AHL6-AHL7",
        "AHL7-AHL8",
        "AHR1-AHR2",
        "AHR2-AHR3",
        "AHR3-AHR4",
        "AHR4-AHR5",
        "AHR5-AHR6",
        "AHR6-AHR7",
        "AHR7-AHR8",
        "AL1-AL2",
        "AL2-AL3",
        "AL3-AL4",
        "AL4-AL5",
        "AL5-AL6",
        "AL6-AL7",
        "AL7-AL8",
        "AR1-AR2",
        "AR2-AR3",
        "AR3-AR4",
        "AR4-AR5",
        "AR5-AR6",
        "AR6-AR7",
        "AR7-AR8",
        "ECL1-ECL2",
        "ECL2-ECL3",
        "ECL3-ECL4",
        "ECL4-ECL5",
        "ECL5-ECL6",
        "ECL6-ECL7",
        "ECL7-ECL8",
        "ECR1-ECR2",
        "ECR2-ECR3",
        "ECR3-ECR4",
        "ECR4-ECR5",
        "ECR5-ECR6",
        "ECR6-ECR7",
        "ECR7-ECR8",
        "PHL1-PHL2",
        "PHL2-PHL3",
        "PHL3-PHL4",
        "PHL4-PHL5",
        "PHL5-PHL6",
        "PHL6-PHL7",
        "PHL7-PHL8",
        "PHR1-PHR2",
        "PHR2-PHR3",
        "PHR3-PHR4",
        "PHR4-PHR5",
        "PHR5-PHR6",
        "PHR6-PHR7",
        "PHR7-PHR8",
    ],
    "sub-10": [
        "GR1-GR2",
        "GR2-GR3",
        "GR3-GR4",
        "GR4-GR5",
        "GR5-GR6",
        "GR6-GR7",
        "GR7-GR8",
        "GR8-GR9",
        "GR9-GR10",
        "GR10-GR11",
        "GR11-GR12",
        "GR12-GR13",
        "GR13-GR14",
        "GR14-GR15",
        "GR15-GR16",
        "GR16-GR17",
        "GR17-GR18",
        "GR18-GR19",
        "GR19-GR20",
        "GR20-GR21",
        "GR21-GR22",
        "GR22-GR23",
        "GR23-GR24",
        "GR24-GR25",
        "GR25-GR26",
        "GR26-GR27",
        "GR27-GR28",
        "GR28-GR29",
        "GR29-GR30",
        "GR30-GR31",
        "GR31-GR32",
        "IAR1-IAR2",
        "IAR2-IAR3",
        "IAR3-IAR4",
        "IPR1-IPR2",
        "IPR2-IPR3",
        "IPR3-IPR4",
    ],
    "sub-11": [
        "TR1-TR2",
        "TR2-TR3",
        "TR3-TR4",
        "TR4-TR5",
        "TR5-TR6",
        "TR6-TR7",
        "TR7-TR8",
        "TR8-TR9",
        "TR9-TR10",
        "GR1-GR2",
        "GR2-GR3",
        "GR3-GR4",
        "GR4-GR5",
        "GR5-GR6",
        "GR6-GR7",
        "GR7-GR8",
        "GR8-GR9",
        "GR9-GR10",
        "GR10-GR11",
        "GR11-GR12",
        "GR12-GR13",
        "GR13-GR14",
        "GR14-GR15",
        "GR15-GR16",
        "GR16-GR17",
        "GR17-GR18",
        "GR18-GR19",
        "GR19-GR20",
        "GR20-GR21",
        "GR21-GR22",
        "GR22-GR23",
        "GR23-GR24",
        "GR24-GR25",
        "GR25-GR26",
        "GR26-GR27",
        "GR27-GR28",
        "GR28-GR29",
        "GR29-GR30",
        "GR30-GR31",
        "GR31-GR32",
        "GR32-GR33",
        "GR33-GR34",
        "GR34-GR35",
        "GR35-GR36",
        "GR36-GR37",
        "GR37-GR38",
        "GR38-GR39",
        "GR39-GR40",
        "GR40-GR41",
        "GR41-GR42",
        "GR42-GR43",
        "GR43-GR44",
        "GR44-GR45",
        "GR45-GR46",
        "GR46-GR47",
        "GR47-GR48",
        "GR48-GR49",
        "GR49-GR50",
        "GR50-GR51",
        "GR51-GR52",
        "GR52-GR53",
        "GR53-GR54",
        "GR54-GR55",
        "GR55-GR56",
        "GR56-GR57",
        "GR57-GR58",
        "GR58-GR59",
        "GR59-GR60",
        "GR60-GR61",
        "GR61-GR62",
        "GR62-GR63",
        "GR63-GR64",
    ],
    "sub-12": [
        "TL1-TL2",
        "TL2-TL3",
        "TL3-TL4",
        "TL4-TL5",
        "TL5-TL6",
        "TL6-TL7",
        "TL7-TL8",
        "TL8-TL9",
        "TL9-TL10",
        "GL1-GL2",
        "GL2-GL3",
        "GL3-GL4",
        "GL4-GL5",
        "GL5-GL6",
        "GL6-GL7",
        "GL7-GL8",
        "GL8-GL9",
        "GL9-GL10",
        "GL10-GL11",
        "GL11-GL12",
        "GL12-GL13",
        "GL13-GL14",
        "GL14-GL15",
        "GL15-GL16",
        "GL16-GL17",
        "GL17-GL18",
        "GL18-GL19",
        "GL19-GL20",
        "GL20-GL21",
        "GL21-GL22",
        "GL22-GL23",
        "GL23-GL24",
        "GL24-GL25",
        "GL25-GL26",
        "GL26-GL27",
        "GL27-GL28",
        "GL28-GL29",
        "GL29-GL30",
        "GL30-GL31",
        "GL31-GL32",
    ],
    "sub-13": [
        "TR1-TR2",
        "TR2-TR3",
        "TR3-TR4",
        "TR4-TR5",
        "TR5-TR6",
        "TR6-TR7",
        "TR7-TR8",
        "TR8-TR9",
        "TR9-TR10",
        "GR1-GR2",
        "GR2-GR3",
        "GR3-GR4",
        "GR4-GR5",
        "GR5-GR6",
        "GR6-GR7",
        "GR7-GR8",
        "GR8-GR9",
        "GR9-GR10",
        "GR10-GR11",
        "GR11-GR12",
        "GR12-GR13",
        "GR13-GR14",
        "GR14-GR15",
        "GR15-GR16",
        "GR16-GR17",
        "GR17-GR18",
        "GR18-GR19",
        "GR19-GR20",
        "GR20-GR21",
        "GR21-GR22",
        "GR22-GR23",
        "GR23-GR24",
        "GR24-GR25",
        "GR25-GR26",
        "GR26-GR27",
        "GR27-GR28",
        "GR28-GR29",
        "GR29-GR30",
        "GR30-GR31",
        "GR31-GR32",
        "GR32-GR33",
        "GR33-GR34",
        "GR34-GR35",
        "GR35-GR36",
        "GR36-GR37",
        "GR37-GR38",
        "GR38-GR39",
        "GR39-GR40",
        "GR40-GR41",
        "GR41-GR42",
        "GR42-GR43",
        "GR43-GR44",
        "GR44-GR45",
        "GR45-GR46",
        "GR46-GR47",
        "GR47-GR48",
        "GR48-GR49",
        "GR49-GR50",
        "GR50-GR51",
        "GR51-GR52",
        "GR52-GR53",
        "GR53-GR54",
        "GR54-GR55",
        "GR55-GR56",
        "GR56-GR57",
        "GR57-GR58",
        "GR58-GR59",
        "GR59-GR60",
        "GR60-GR61",
        "GR61-GR62",
        "GR62-GR63",
        "GR63-GR64",
    ],
    "sub-14": [
        "TR1-TR2",
        "TR2-TR3",
        "TR3-TR4",
        "TR4-TR5",
        "TR5-TR6",
        "TR6-TR7",
        "TR7-TR8",
        "TR8-TR9",
        "TR9-TR10",
        "IAR1-IAR2",
        "IAR2-IAR3",
        "IAR3-IAR4",
        "IAR4-IAR5",
        "IAR5-IAR6",
        "IPR1-IPR2",
        "IPR2-IPR3",
        "IPR3-IPR4",
        "PLR1-PLR2",
        "PLR2-PLR3",
        "PLR3-PLR4",
        "PLR4-PLR5",
        "PLR5-PLR6",
        "PLR6-PLR7",
        "PLR7-PLR8",
        "PLR8-PLR9",
        "PLR9-PLR10",
        "PLR10-PLR11",
        "PLR11-PLR12",
        "PLR12-PLR13",
        "PLR13-PLR14",
        "PLR14-PLR15",
        "PLR15-PLR16",
        "PMR1-PMR2",
        "PMR2-PMR3",
        "PMR3-PMR4",
        "PMR4-PMR5",
        "PMR5-PMR6",
        "PMR6-PMR7",
        "PMR7-PMR8",
        "PMR8-PMR9",
        "PMR9-PMR10",
        "PMR10-PMR11",
        "PMR11-PMR12",
        "PMR12-PMR13",
        "PMR13-PMR14",
        "PMR14-PMR15",
        "PMR15-PMR16",
    ],
    "sub-15": [
        "TBAL1-TBAL2",
        "TBAL2-TBAL3",
        "TBAL3-TBAL4",
        "TBPL1-TBPL2",
        "TBPL2-TBPL3",
        "TBPL3-TBPL4",
        "TLL1-TLL2",
        "TLL2-TLL3",
        "TLL3-TLL4",
        "TLL4-TLL5",
        "TLL5-TLL6",
        "TLL6-TLL7",
        "TLL7-TLL8",
        "TLL8-TLL9",
        "TLL9-TLL10",
        "TLL10-TLL11",
        "TLL11-TLL12",
        "TLL12-TLL13",
        "TLL13-TLL14",
        "TLL14-TLL15",
        "TLL15-TLL16",
        "TLL16-TLL17",
        "TLL17-TLL18",
        "TLL18-TLL19",
        "TLL19-TLL20",
        "TLL20-TLL21",
        "TLL21-TLL22",
        "TLL22-TLL23",
        "TLL23-TLL24",
        "TLL24-TLL25",
        "TLL25-TLL26",
        "TLL26-TLL27",
        "TLL27-TLL28",
        "TLL28-TLL29",
        "TLL29-TLL30",
        "TLL30-TLL31",
        "TLL31-TLL32",
    ],
    "sub-16": [
        "TL1-TL2",
        "TL2-TL3",
        "TL3-TL4",
        "TL4-TL5",
        "TL5-TL6",
        "TL6-TL7",
        "TL7-TL8",
        "TL8-TL9",
        "TL9-TL10",
        "GL1-GL2",
        "GL2-GL3",
        "GL3-GL4",
        "GL4-GL5",
        "GL5-GL6",
        "GL6-GL7",
        "GL7-GL8",
        "GL8-GL9",
        "GL9-GL10",
        "GL10-GL11",
        "GL11-GL12",
        "GL12-GL13",
        "GL13-GL14",
        "GL14-GL15",
        "GL15-GL16",
        "GL16-GL17",
        "GL17-GL18",
        "GL18-GL19",
        "GL19-GL20",
        "GL20-GL21",
        "GL21-GL22",
        "GL22-GL23",
        "GL23-GL24",
        "GL24-GL25",
        "GL25-GL26",
        "GL26-GL27",
        "GL27-GL28",
        "GL28-GL29",
        "GL29-GL30",
        "GL30-GL31",
        "GL31-GL32",
    ],
    "sub-17": [
        "TR1-TR2",
        "TR2-TR3",
        "TR3-TR4",
        "TR4-TR5",
        "TR5-TR6",
        "TR6-TR7",
        "TR7-TR8",
        "TR8-TR9",
        "TR9-TR10",
        "FAR1-FAR2",
        "FAR2-FAR3",
        "FAR3-FAR4",
        "FAR4-FAR5",
        "FAR5-FAR6",
        "FAR6-FAR7",
        "FAR7-FAR8",
        "FAR8-FAR9",
        "FAR9-FAR10",
        "FAR10-FAR11",
        "FAR11-FAR12",
        "FAR12-FAR13",
        "FAR13-FAR14",
        "FAR14-FAR15",
        "FAR15-FAR16",
        "FPR1-FPR2",
        "FPR2-FPR3",
        "FPR3-FPR4",
        "FPR4-FPR5",
        "FPR5-FPR6",
        "FPR6-FPR7",
        "FPR7-FPR8",
        "FPR8-FPR9",
        "FPR9-FPR10",
        "FPR10-FPR11",
        "FPR11-FPR12",
        "FPR12-FPR13",
        "FPR13-FPR14",
        "FPR14-FPR15",
        "FPR15-FPR16",
    ],
    "sub-18": [
        "TL1-TL2",
        "TL2-TL3",
        "TL3-TL4",
        "TL4-TL5",
        "TL5-TL6",
        "TL6-TL7",
        "TL7-TL8",
        "TL8-TL9",
        "TL9-TL10",
        "IHAL1-IHAL2",
        "IHAL2-IHAL3",
        "IHAL3-IHAL4",
        "IHPL1-IHPL2",
        "IHPL2-IHPL3",
        "IHPL3-IHPL4",
        "PLL1-PLL2",
        "PLL2-PLL3",
        "PLL3-PLL4",
        "PLL4-PLL5",
        "PLL5-PLL6",
        "PML1-PML2",
        "PML2-PML3",
        "PML3-PML4",
        "PML4-PML5",
        "PML5-PML6",
    ],
    "sub-19": [
        "PL1-PL2",
        "PL2-PL3",
        "PL3-PL4",
        "PL4-PL5",
        "PL5-PL6",
        "PL6-PL7",
        "PL7-PL8",
        "PL8-PL9",
        "PL9-PL10",
        "PL10-PL11",
        "PL11-PL12",
        "PL12-PL13",
        "PL13-PL14",
        "PL14-PL15",
        "PL15-PL16",
        "PL16-PL17",
        "PL17-PL18",
        "PL18-PL19",
        "PL19-PL20",
        "PL20-PL21",
        "PL21-PL22",
        "PL22-PL23",
        "PL23-PL24",
        "PL24-PL25",
        "PL25-PL26",
        "PL26-PL27",
        "PL27-PL28",
        "PL28-PL29",
        "PL29-PL30",
        "PL30-PL31",
        "PL31-PL32",
        "TL1-TL2",
        "TL2-TL3",
        "TL3-TL4",
        "TL4-TL5",
        "TL5-TL6",
        "TL6-TL7",
        "TL7-TL8",
        "TL8-TL9",
        "TL9-TL10",
        "TL10-TL11",
        "TL11-TL12",
        "TL12-TL13",
        "TL13-TL14",
        "TL14-TL15",
        "TL15-TL16",
    ],
    "sub-20": [
        "OTL1-OTL2",
        "OTL2-OTL3",
        "OTL3-OTL4",
        "OTL4-OTL5",
        "OTL5-OTL6",
        "OTL6-OTL7",
        "OTL7-OTL8",
        "OTL8-OTL9",
        "OTL9-OTL10",
        "OTL10-OTL11",
        "OTL11-OTL12",
        "OTL12-OTL13",
        "OTL13-OTL14",
        "OTL14-OTL15",
        "OTL15-OTL16",
    ],
}


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
        subset_key_name="subsets (subjects)",
        known_subset_keys=_ZURICH_IEEG_SUBJECTS,
        size_hint="~60 GB",
        subset_size_hint="~3 GB per subject (varies by recording nights)",
        subset_size_bytes=3 * 1024**3,  # ~3 GB per subject
        info_url="https://openneuro.org/datasets/ds003498/versions/1.1.1",
        license="CC0 1.0 Universal (public domain)",
        max_parallel_downloads=8,
        ilae_per_subject=_ZURICH_ILAE_PER_SUBJECT,
        resected_zone_per_subject=_ZURICH_RESECTED_ZONE_PER_SUBJECT,
        excluded_channels_per_subject=_ZURICH_EXCLUDED_CHANNELS_PER_SUBJECT,
        all_channels_per_subject=_ZURICH_ALL_CHANNELS_PER_SUBJECT,
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
