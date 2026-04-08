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
from .dataset_loader import load_noise_dummy, load_realistic_swiss, load_structured_dummy
from .downloader import (
    REMOTE_DATASETS,
    RemoteDatasetSpec,
    SubsetSpec,
    _filter_files_by_dict_subset,
    _is_dataset_cached,
    delete_remote_files,
    ensure_remote_files,
    get_dataset_dir,
    get_remote_dataset_spec,
)

# ---------------------------------------------------------------------------
# Metadata for built-in local datasets (no subset concept)
# ---------------------------------------------------------------------------

_LOCAL_DATASET_INFO: dict[str, str] = {
    "dummy_chain": "Synthetic chain-topology VAR time-series (3 subjects).",
    "dummy_random": "Synthetic random-topology VAR time-series (3 subjects).",
    "dummy_star": "Synthetic star-topology VAR time-series (3 subjects).",
    "dummy_noise": "Synthetic uncorrelated noise time-series (10 subjects).",
    "realistic_swiss": "Simulated realistic Swiss VAR time-series (1 subject).",
}


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
    """

    identifier: str
    description: str
    subset_key_name: str | None
    subsets: tuple[str, ...] | None
    size_hint: str | None = None
    subset_size_hint: str | None = None
    seizures_per_subject: dict[str, int] | None = None
    seizure_info_url: str | None = None
    info_url: str | None = None
    license: str | None = None
    auth_hint: str | None = None
    local_path: Path | None = None

    def _rich_renderable(self) -> RenderResult:

        from rich.panel import Panel
        from rich.table import Table

        lines: list[str] = [self.description]
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_dataset(
    identifier: str, *, subset: SubsetSpec | None = None, accept: bool = False, force: bool = False
) -> Dataset[SignalData]:
    """Load a dataset by identifier.

    Args:
        identifier: Dataset name, e.g. ``"dummy_chain"`` or
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
        accept: If ``False`` (default) and files need to be downloaded, show
            the dataset license, estimated download size, and ask for
            confirmation before proceeding.  Set to ``True`` to skip the
            prompt (e.g. in scripts where you have already accepted the
            license).
        force: If ``True``, delete any existing local files for the selected
            subset and re-download from scratch.

    Returns:
        :class:`~cobrabox.Dataset` of :class:`~cobrabox.SignalData` objects.

    Raises:
        ValueError: If ``identifier`` is unknown, or if ``subset`` contains
            keys not present in the dataset.
        DownloadCancelled: If ``accept=False`` and the user declines the download.
    """
    if identifier in {"dummy_chain", "dummy_random", "dummy_star"}:
        return load_structured_dummy(identifier)
    if identifier == "dummy_noise":
        return load_noise_dummy(identifier)
    if identifier == "realistic_swiss":
        return load_realistic_swiss(identifier)

    spec = get_remote_dataset_spec(identifier)
    if spec is not None:
        # Validate subset keys early, before triggering any downloads.
        if subset is not None:
            _validate_subset(spec, subset)

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
        f"Unknown dataset identifier: {identifier!r}. "
        f"Known identifiers: {[*sorted(_LOCAL_DATASET_INFO), *REMOTE_DATASETS]}"
    )


def download_dataset(
    identifier: str, *, subset: SubsetSpec | None = None, accept: bool = False, force: bool = False
) -> Path:
    """Download a remote dataset without loading it into memory.

    Useful for pre-fetching large datasets before analysis, or for downloading
    to a shared location.  Interrupted downloads are resumed automatically.

    Args:
        identifier: Remote dataset name, e.g. ``"chb_mit"``.
        subset: Restrict which files are downloaded — same syntax as
            :func:`dataset`.
        accept: Skip the confirmation prompt.
        force: Delete existing local files and re-download from scratch.

    Returns:
        Path to the local dataset directory.

    Raises:
        ValueError: If ``identifier`` is a local dataset (no download needed)
            or unknown, or if ``subset`` contains invalid keys.
        DownloadCancelled: If ``accept=False`` and the user declines the download.
    """
    if identifier in {*_LOCAL_DATASET_INFO}:
        raise ValueError(f"'{identifier}' is a local dataset — no download needed.")

    spec = get_remote_dataset_spec(identifier)
    if spec is not None:
        if subset is not None:
            _validate_subset(spec, subset)
        return ensure_remote_files(spec, subset=subset, accept=accept, force=force)

    raise ValueError(
        f"Unknown dataset identifier: {identifier!r}. "
        f"Known identifiers: {[*sorted(_LOCAL_DATASET_INFO), *REMOTE_DATASETS]}"
    )


def list_datasets() -> dict[str, list[str]]:
    """Return all known dataset identifiers grouped by type.

    Example::

        cb.list_datasets()
        # {
        #   'local':  ['dummy_chain', 'dummy_noise', ...],
        #   'remote': ['bonn_eeg', 'chb_mit', ...],
        # }

    Returns:
        Dict with keys ``"local"`` (no download required) and ``"remote"``,
        each mapping to a sorted list of identifier strings.
    """
    return {"local": sorted(_LOCAL_DATASET_INFO), "remote": sorted(REMOTE_DATASETS)}


def show_datasets() -> list[dict[str, str | None]]:
    """Print a summary table of all available datasets and return the data.

    Example::

        cb.show_datasets()

    Returns:
        List of dicts with keys ``"identifier"``, ``"type"``, ``"cached"``,
        ``"size"``, ``"subsets"``, and ``"license"``.  Suitable for
        programmatic use in notebooks or scripts.  ``"cached"`` is
        ``"yes"``/``"no"`` for remote datasets and ``None`` for local ones.
    """
    from rich.console import Console
    from rich.table import Table

    table = Table(title="Available Datasets", show_lines=False)
    table.add_column("Dataset", style="bold", no_wrap=True)
    table.add_column("Type", style="dim")
    table.add_column("Cached")
    table.add_column("Size")
    table.add_column("Subsets")
    table.add_column("License")

    rows: list[dict[str, str | None]] = []
    for ident in sorted(_LOCAL_DATASET_INFO) + sorted(REMOTE_DATASETS):
        if ident in _LOCAL_DATASET_INFO:
            table.add_row(ident, "local", "\u2014", "\u2014", "\u2014", "\u2014")
            rows.append(
                {
                    "identifier": ident,
                    "type": "local",
                    "cached": None,
                    "size": None,
                    "subsets": None,
                    "license": None,
                }
            )
        else:
            spec = get_remote_dataset_spec(ident)
            if spec is None:
                continue
            cached_val = "yes" if _is_dataset_cached(spec) else "no"
            cached_cell = (
                f"[green]{cached_val}[/green]"
                if cached_val == "yes"
                else f"[dim]{cached_val}[/dim]"
            )
            size = spec.size_hint or "\u2014"
            subset_keys = spec.subset_keys()
            n_keys = len(subset_keys) if subset_keys else None
            subsets = f"{n_keys} {spec.subset_key_name or 'subsets'}" if n_keys else "\u2014"
            lic = spec.license or "\u2014"
            table.add_row(ident, "remote", cached_cell, size, subsets, lic)
            rows.append(
                {
                    "identifier": ident,
                    "type": "remote",
                    "cached": cached_val,
                    "size": None if size == "\u2014" else size,
                    "subsets": None if subsets == "\u2014" else subsets,
                    "license": None if lic == "\u2014" else lic,
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
    if identifier in _LOCAL_DATASET_INFO:
        return DatasetInfo(
            identifier=identifier,
            description=_LOCAL_DATASET_INFO[identifier],
            subset_key_name=None,
            subsets=None,
        )

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
            seizures_per_subject=spec.seizures_per_subject,
            seizure_info_url=spec.seizure_info_url,
            info_url=spec.info_url,
            license=spec.license,
            auth_hint=spec.auth_hint,
            local_path=cached_path if _is_dataset_cached(spec) else None,
        )

    raise ValueError(
        f"Unknown dataset identifier: {identifier!r}. "
        f"Known identifiers: {[*sorted(_LOCAL_DATASET_INFO), *REMOTE_DATASETS]}"
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
        ValueError: If ``identifier`` is a local dataset (nothing to delete)
            or unknown.
        DownloadCancelled: If ``confirm=True`` and the user declines the deletion.

    Example::

        # Delete a full dataset
        cb.delete_dataset("bonn_eeg")

        # Delete specific subjects only
        cb.delete_dataset("chb_mit", subset=["chb01", "chb02"])

        # Skip confirmation
        cb.delete_dataset("bonn_eeg", confirm=False)
    """
    if identifier in _LOCAL_DATASET_INFO:
        raise ValueError(f"'{identifier}' is a local dataset — nothing to delete.")

    spec = get_remote_dataset_spec(identifier)
    if spec is None:
        raise ValueError(
            f"Unknown dataset identifier: {identifier!r}. "
            f"Known identifiers: {[*sorted(_LOCAL_DATASET_INFO), *REMOTE_DATASETS]}"
        )

    delete_remote_files(spec, subset=subset, confirm=confirm)
