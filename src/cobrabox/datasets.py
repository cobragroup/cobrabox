from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console, ConsoleOptions, RenderResult

from .data import SignalData
from .dataset import Dataset
from .downloader import (
    REMOTE_DATASETS,
    LargeLoadError,
    RemoteDatasetSpec,
    SubsetSpec,
    _dataset_cache_status,
    _filter_files_by_dict_subset,
    _format_bytes,
    _prompt_large_load_verify,
    delete_remote_files,
    ensure_remote_files,
    get_dataset_dir,
    get_remote_dataset_spec,
)

# ---------------------------------------------------------------------------
# DatasetInfo
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetInfo:
    """Human-readable summary of a dataset, returned by :func:`dataset_info`.

    Attributes:
        identifier: The string key passed to :func:`dataset`.
        description: Short description of the dataset.
        subset_key_name: What the subset dimension is called (e.g. ``"subjects"``),
            or ``None`` if no subset concept exists.
        subsets: Available subset keys, or ``None`` if unknown or not applicable.
        size_hint: Approximate total download size (e.g. ``"~10 MB"``), or
            ``None`` if unknown.
        subset_size_hint: Approximate size per subset (e.g. ``"~2 MB per set"``),
            or ``None`` if unknown.
        seizures_per_subject: Number of seizures per subset key, or ``None`` if
            not available.  Keys match the subset keys shown under *subsets*.
        seizure_info_url: URL where the seizure count information was sourced,
            or ``None`` if not available.
        info_url: General landing page / homepage for the dataset, or ``None``
            if not available.
        auth_hint: Optional hint shown when a download fails with 401/403,
            e.g. how to obtain credentials.  ``None`` if not applicable.
        ilae_per_subject: ILAE surgical outcome score per subject (1=seizure-free,
            6=no improvement), or ``None`` if not available.
        resected_zone_per_subject: Bipolar channel pairs in the resected zone per
            subject, or ``None`` if not available.
        excluded_channels_per_subject: Bipolar channel pairs excluded due to eloquent
            cortex stimulation per subject, or ``None`` if not available.
        all_channels_per_subject: All implanted bipolar channel pairs per subject,
            or ``None`` if not available.
    """

    identifier: str
    description: str
    subset_key_name: str | None
    subsets: tuple[str, ...] | None
    size_hint: str | None = None
    subset_size_hint: str | None = None
    data_type: str | None = None
    seizures_per_subject: dict[str, int] | None = None
    seizure_info_url: str | None = None
    info_url: str | None = None
    license: str | None = None
    auth_hint: str | None = None
    local_path: Path | None = None
    ilae_per_subject: dict[str, int] | None = None
    resected_zone_per_subject: dict[str, list[str]] | None = None
    excluded_channels_per_subject: dict[str, list[str]] | None = None
    all_channels_per_subject: dict[str, list[str]] | None = None

    def _rich_renderable(self) -> RenderResult:

        from rich.panel import Panel
        from rich.table import Table

        lines: list[str] = [self.description]
        if self.data_type is not None:
            lines.append(f"\n[dim]Data type:[/dim] {self.data_type}")
        if self.info_url is not None:
            lines.append(f"\n[dim]Source  :[/dim] {self.info_url}")
        if self.size_hint is not None or self.subset_size_hint is not None:
            parts = []
            if self.size_hint is not None:
                parts.append(f"total {self.size_hint}")
            if self.subset_size_hint is not None:
                parts.append(self.subset_size_hint)
            lines.append(f"[dim]Size    :[/dim] {', '.join(parts)} (approximate)")
        if self.subset_key_name is None or self.subsets is None:
            lines.append(
                f"[dim]Subsets :[/dim] none \u2014"
                f" call cb.load_dataset({self.identifier!r}) to load all."
            )
        else:
            n = len(self.subsets)
            lines.append(f"[dim]{self.subset_key_name} ({n})[/dim]: {', '.join(self.subsets)}")
            second = self.subsets[1] if len(self.subsets) > 1 else self.subsets[0]
            example = f'["{self.subsets[0]}", "{second}"]'
            lines.append(
                f'[dim]Usage   :[/dim] cb.load_dataset("{self.identifier}", subset={example})'
            )
        if self.seizures_per_subject is not None:
            counts = self.seizures_per_subject
            total = sum(counts.values())
            max_key = max(len(k) for k in counts)
            max_val = len(str(max(counts.values())))
            n_cols = 5
            pairs = list(counts.items())
            lines.append(f"  seizures/subject ({total} total):")
            for i in range(0, len(pairs), n_cols):
                chunk = pairs[i : i + n_cols]
                row = "    " + "   ".join(f"{k:<{max_key}} {v:>{max_val}}" for k, v in chunk)
                lines.append(row)
        if self.ilae_per_subject is not None:
            ilae = self.ilae_per_subject
            max_key = max(len(k) for k in ilae)
            n_cols = 5
            pairs = list(ilae.items())
            lines.append("  ILAE outcome/subject (1=seizure-free, 6=no improvement):")
            for i in range(0, len(pairs), n_cols):
                chunk = pairs[i : i + n_cols]
                row = "    " + "   ".join(f"{k:<{max_key}} {v}" for k, v in chunk)
                lines.append(row)
        if self.license is not None:
            lines.append(f"[dim]License :[/dim] {self.license}")
        if self.auth_hint is not None:
            lines.append(f"[dim]Auth    :[/dim] {self.auth_hint}")
        if self.local_path is not None:
            lines.append(f"[dim]Cached  :[/dim] {self.local_path}")

        content: object = "\n".join(lines)

        if self.seizures_per_subject is not None:
            from rich.console import Group

            counts = self.seizures_per_subject
            total = sum(counts.values())
            seiz_table = Table(box=None, padding=(0, 2, 0, 0), show_header=False)
            seiz_table.add_column(style="dim", no_wrap=True)
            seiz_table.add_column(justify="right")
            for k, v in counts.items():
                seiz_table.add_row(k, str(v))
            content = Group(content, f"\n[dim]seizures/subject ({total} total):[/dim]", seiz_table)

        yield Panel(content, title=f"[bold]{self.identifier}[/bold]")

    def __str__(self) -> str:
        from io import StringIO

        from rich.console import Console

        sio = StringIO()
        console = Console(file=sio, highlight=False, no_color=True, width=88)
        for renderable in self._rich_renderable():
            console.print(renderable)
        return sio.getvalue().rstrip("\n")

    def __repr__(self) -> str:
        return self.__str__()

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield from self._rich_renderable()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_subset(spec: RemoteDatasetSpec, subset: SubsetSpec) -> None:
    """Raise ValueError if *subset* contains keys or values invalid for *spec*."""
    if isinstance(subset, dict) and spec.files is None:
        raise ValueError(
            f"'{spec.identifier}' has no file index, so file-level subsets are not "
            f"supported. Pass a list of keys instead, e.g. subset={list(subset)!r}."
        )
    keys_to_validate = list(subset.keys()) if isinstance(subset, dict) else list(subset)
    known = spec.subset_keys()
    if known is not None:
        invalid = [s for s in keys_to_validate if s not in known]
        if invalid:
            raise ValueError(
                f"Unknown subset keys for '{spec.identifier}': {invalid}.\n"
                f"Valid {spec.subset_key_name or 'keys'}: {known}"
            )
    if isinstance(subset, dict):
        for key, value in subset.items():
            if isinstance(value, int) and value < 1:
                raise ValueError(f"File count for subset key '{key}' must be >= 1, got {value!r}.")
            if isinstance(value, list) and not value:
                raise ValueError(
                    f"File list for subset key '{key}' must be non-empty; "
                    "use None to include all files."
                )


_LARGE_LOAD_WARN_BYTES = 4 * 1024**3  # 4 GB


def _check_large_load(spec: RemoteDatasetSpec, subset: SubsetSpec | None, accept: bool) -> None:
    """Warn and ask for confirmation before a load that would use a lot of memory at once.

    Uses spec.subset_size_bytes (on-disk, per-subject estimate) as a proxy; the
    in-memory footprint after decoding can be larger. Skipped when accept=True,
    or when the dataset has no per-subject size estimate to check against.
    Raises LargeLoadError if the user declines the prompt.
    """
    if accept or spec.subset_size_bytes is None:
        return
    if subset is None:
        keys = spec.subset_keys()
        n = len(keys) if keys else 0
    elif isinstance(subset, dict):
        n = len(subset)
    else:
        n = len(set(subset))
    if n == 0:
        return
    total_bytes = spec.subset_size_bytes * n
    if total_bytes <= _LARGE_LOAD_WARN_BYTES:
        return
    if _prompt_large_load_verify(spec, n, total_bytes):
        return
    raise LargeLoadError(
        f"Load of '{spec.identifier}' cancelled at the memory-size prompt "
        f"({n} subject{'s' if n != 1 else ''}, ~{_format_bytes(total_bytes)} estimated). "
        "Pass a smaller subset=..., accept=True to skip this prompt, or call "
        "download_dataset() instead to fetch the files without loading them into memory."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_dataset(
    identifier: str, *, subset: SubsetSpec | None = None, accept: bool = False, force: bool = False
) -> Dataset[SignalData]:
    """Load a dataset by identifier.

    Args:
        identifier: Dataset name, e.g. ``"swiss_demo"`` or
            ``"swiss_eeg_short"``.
        subset: Restrict which data is downloaded and loaded.  Two forms:

            - ``list[str]``: subject / set keys to load in full, e.g.
              ``["ID1", "ID2"]``.  Call :func:`dataset_info` to see available
              keys.
            - ``dict[str, int | list[str] | None]``: per-key file-level
              control.  Map each key to:

              - ``int N``     — first *N* files (in file-index order)
              - ``list[str]`` — specific filenames
              - ``None``      — all files for that key

              Example::

                  cb.load_dataset("swiss_eeg_long", subset={"ID01": 2})
                  cb.load_dataset("swiss_eeg_long", subset={"ID01": ["ID01_1h.mat"]})
                  cb.load_dataset("swiss_eeg_long", subset={"ID01": None, "ID02": 3})

            ``None`` loads everything.
        accept: If ``False`` (default), show a confirmation prompt whenever
            the resolved subset is estimated to pull more than 4 GB into
            memory at once, and — if files also need to be downloaded — the
            dataset license and estimated download size. Set to ``True`` to
            skip both prompts (e.g. in scripts where you have already
            reviewed the memory cost and accepted the license).
        force: If ``True``, delete any existing local files for the selected
            subset and re-download from scratch.

    Returns:
        :class:`~cobrabox.Dataset` of :class:`~cobrabox.SignalData` objects.

    Raises:
        ValueError: If ``identifier`` is unknown, or if ``subset`` contains
            keys not present in the dataset.
        LargeLoadError: If ``accept=False`` and the user declines the
            large-load memory prompt. Pass a smaller ``subset``,
            ``accept=True``, or use :func:`download_dataset` instead to fetch
            the files without loading them into memory.
        DownloadCancelled: If ``accept=False`` and the user declines the download.
    """
    spec = get_remote_dataset_spec(identifier)
    if spec is not None:
        # Validate subset keys early, before triggering any downloads.
        if subset is not None:
            _validate_subset(spec, subset)
        _check_large_load(spec, subset, accept)

        dataset_dir = ensure_remote_files(spec, subset=subset, accept=accept, force=force)

        # Derive the subset to pass to the loader.
        # For the dict form: expand to a flat list of file stems so the loader
        # can filter by exact filename rather than by subject key.
        # spec.files is now populated (ensure_remote_files resolved it above).
        if isinstance(subset, dict):
            selected = _filter_files_by_dict_subset(subset, spec.files or [])
            loader_subset: Sequence[str] | None = [
                Path(f.filename).stem for f in selected if f.subset_key is not None
            ]
        else:
            loader_subset = list(subset) if subset is not None else None

        return spec.loader(dataset_dir, loader_subset)

    raise ValueError(
        f"Unknown dataset identifier: {identifier!r}. Known identifiers: {sorted(REMOTE_DATASETS)}"
    )


def download_dataset(
    identifier: str,
    *,
    subset: SubsetSpec | None = None,
    accept: bool = False,
    force: bool = False,
    dry_run: bool = False,
) -> Path:
    """Download a remote dataset without loading it into memory.

    Useful for pre-fetching large datasets before analysis, or for downloading
    to a shared location.  Interrupted downloads are resumed automatically.

    Args:
        identifier: Remote dataset name, e.g. ``"chb_mit"``.
        subset: Restrict which files are downloaded.  Two forms:

            - ``list[str]``: subject / set keys to download in full, e.g.
              ``["chb01", "chb02"]``.  Call :func:`dataset_info` to see
              available keys.
            - ``dict[str, int | list[str] | None]``: per-key file-level
              control.  Map each key to:

              - ``int N``     — first *N* files (in file-index order)
              - ``list[str]`` — specific filenames
              - ``None``      — all files for that key

              Example::

                  cb.download_dataset("swiss_eeg_long", subset={"ID01": 2})
                  cb.download_dataset("swiss_eeg_long", subset={"ID01": None, "ID02": 3})

            ``None`` downloads everything.
        accept: Skip the confirmation prompt.
        force: Delete existing local files and re-download from scratch.
        dry_run: If ``True``, print a summary of what would be downloaded and
            return immediately without downloading anything.  Useful for
            checking file counts and size before committing to a large download.

    Returns:
        Path to the local dataset directory (may not exist if ``dry_run=True``
        and nothing has been downloaded yet).

    Raises:
        ValueError: If ``identifier`` is unknown, or if ``subset`` contains
            invalid keys.
        DownloadCancelled: If ``accept=False`` and the user declines the download.
    """
    spec = get_remote_dataset_spec(identifier)
    if spec is not None:
        if subset is not None:
            _validate_subset(spec, subset)
        return ensure_remote_files(spec, subset=subset, accept=accept, force=force, dry_run=dry_run)

    raise ValueError(
        f"Unknown dataset identifier: {identifier!r}. Known identifiers: {sorted(REMOTE_DATASETS)}"
    )


def list_datasets() -> list[str]:
    """Return all known dataset identifiers, sorted.

    Every dataset is downloaded on first use; none ship with the package.

    Example::

        cb.list_datasets()
        # ['bonn_eeg', 'chb_mit', ..., 'swiss_demo', ...]

    Returns:
        Sorted list of identifier strings accepted by :func:`load_dataset`.
    """
    return sorted(REMOTE_DATASETS)


def show_datasets() -> list[dict[str, str | None]]:
    """Print a summary table of all available datasets and return the data.

    Example::

        cb.show_datasets()
        # Dataset     Data type         Cached  Seizures  Size     Subsets  License
        # bonn_eeg    ictal/interictal  yes     100       ~10 MB   5 sets   ...
        # chb_mit     ictal/interictal  3/24    200       ~30 GB   24 subj  ...

    Returns:
        List of dicts with keys ``"identifier"``, ``"data_type"``, ``"cached"``,
        ``"seizures"``, ``"size"``, ``"subsets"``, and ``"license"``.  Suitable
        for programmatic use in notebooks or scripts.  ``"cached"`` is
        ``"yes"``/``"no"``/``"N/M"``.
    """
    from rich.console import Console
    from rich.table import Table

    table = Table(title="Available Datasets", show_lines=False)
    table.add_column("Dataset", style="bold", no_wrap=True)
    table.add_column("Data type")
    table.add_column("Cached")
    table.add_column("Seizures", justify="right")
    table.add_column("Size")
    table.add_column("Subsets")
    table.add_column("License")

    rows: list[dict[str, str | None]] = []
    for ident in sorted(REMOTE_DATASETS):
        spec = REMOTE_DATASETS[ident]
        data_type_str = spec.data_type or "—"
        cached_val = _dataset_cache_status(spec)
        cached_cell = (
            f"[green]{cached_val}[/green]" if cached_val == "yes" else f"[dim]{cached_val}[/dim]"
        )
        if spec.seizures_per_subject is not None:
            seizures_str = str(sum(spec.seizures_per_subject.values()))
        else:
            seizures_str = "—"
        size = spec.size_hint or "—"
        subset_keys = spec.subset_keys()
        n_keys = len(subset_keys) if subset_keys else None
        subsets = f"{n_keys} {spec.subset_key_name or 'subsets'}" if n_keys else "—"
        lic = spec.license or "—"
        table.add_row(ident, data_type_str, cached_cell, seizures_str, size, subsets, lic)
        rows.append(
            {
                "identifier": ident,
                "data_type": None if data_type_str == "—" else data_type_str,
                "cached": cached_val,
                "seizures": None if seizures_str == "—" else seizures_str,
                "size": None if size == "—" else size,
                "subsets": None if subsets == "—" else subsets,
                "license": None if lic == "—" else lic,
            }
        )

    console = Console(file=sys.stdout)
    console.print(table)
    console.print(f"[dim]Data directory:[/dim] {get_dataset_dir()}")
    return rows


def dataset_info(identifier: str) -> DatasetInfo:
    """Return metadata for a dataset, including available subset keys.

    Example::

        info = cb.dataset_info("swiss_eeg_short")
        print(info)
        # DatasetInfo: swiss_eeg_short
        #   description  : Short-term scalp EEG ...
        #   subjects (18): ID1, ID2, ID4a, ...
        #   usage        : cb.load_dataset("swiss_eeg_short", subset=["ID1", "ID2"])

    Args:
        identifier: Dataset name, e.g. ``"swiss_eeg_short"``.

    Returns:
        :class:`DatasetInfo` with description and available subset keys.

    Raises:
        ValueError: If the identifier is unknown.
    """
    spec = get_remote_dataset_spec(identifier)
    if spec is not None:
        keys = spec.subset_keys()
        cached_path = get_dataset_dir() / spec.local_rel_dir
        return DatasetInfo(
            identifier=identifier,
            description=spec.description,
            subset_key_name=spec.subset_key_name,
            subsets=tuple(keys) if keys is not None else None,
            size_hint=spec.size_hint,
            subset_size_hint=spec.subset_size_hint,
            data_type=spec.data_type,
            seizures_per_subject=spec.seizures_per_subject,
            seizure_info_url=spec.seizure_info_url,
            info_url=spec.info_url,
            license=spec.license,
            auth_hint=spec.auth_hint,
            local_path=cached_path if _dataset_cache_status(spec) != "no" else None,
            ilae_per_subject=spec.ilae_per_subject,
            resected_zone_per_subject=spec.resected_zone_per_subject,
            excluded_channels_per_subject=spec.excluded_channels_per_subject,
            all_channels_per_subject=spec.all_channels_per_subject,
        )

    raise ValueError(
        f"Unknown dataset identifier: {identifier!r}. Known identifiers: {sorted(REMOTE_DATASETS)}"
    )


def delete_dataset(
    identifier: str, *, subset: list[str] | None = None, confirm: bool = True
) -> None:
    """Delete locally cached files for a remote dataset.

    Useful for freeing disk space after analysis is complete.  Use
    :func:`dataset_info` to inspect what subset keys are available, and
    :func:`show_datasets` to see which datasets are currently cached.

    Args:
        identifier: Remote dataset name, e.g. ``"bonn_eeg"``.
        subset: If given, only delete files for the listed subset keys (e.g.
            ``["chb01", "chb02"]``).  ``None`` (default) deletes everything
            for this dataset.
        confirm: If ``True`` (default), print a summary and prompt before
            deleting.  Set to ``False`` to skip the prompt (e.g. in
            automated pipelines).

    Raises:
        ValueError: If ``identifier`` is unknown.
        DownloadCancelled: If ``confirm=True`` and the user declines the deletion.

    Example::

        # Delete a full dataset
        cb.delete_dataset("bonn_eeg")

        # Delete specific subjects only
        cb.delete_dataset("chb_mit", subset=["chb01", "chb02"])

        # Skip confirmation
        cb.delete_dataset("bonn_eeg", confirm=False)
    """
    spec = get_remote_dataset_spec(identifier)
    if spec is None:
        raise ValueError(
            f"Unknown dataset identifier: {identifier!r}. "
            f"Known identifiers: {sorted(REMOTE_DATASETS)}"
        )

    delete_remote_files(spec, subset=subset, confirm=confirm)
