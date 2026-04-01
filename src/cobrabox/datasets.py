from __future__ import annotations

import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .data import SignalData
from .dataset import Dataset
from .dataset_loader import load_noise_dummy, load_realistic_swiss, load_structured_dummy
from .downloader import (
    REMOTE_DATASETS,
    SubsetSpec,
    _filter_files_by_dict_subset,
    _is_dataset_cached,
    delete_remote_files,
    ensure_remote_files,
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

    def __str__(self) -> str:
        lines = [f"DatasetInfo: {self.identifier}"]
        lines.append(f"  description : {self.description}")
        if self.info_url is not None:
            lines.append(f"  source      : {self.info_url}")
        if self.size_hint is not None or self.subset_size_hint is not None:
            parts = []
            if self.size_hint is not None:
                parts.append(f"total {self.size_hint}")
            if self.subset_size_hint is not None:
                parts.append(self.subset_size_hint)
            lines.append(f"  size        : {', '.join(parts)} (approximate)")
        if self.subset_key_name is None or self.subsets is None:
            lines.append(
                f"  subsets     : none — call cb.dataset({self.identifier!r}) to load all."
            )
        else:
            n = len(self.subsets)
            keys_str = ", ".join(self.subsets)
            wrapped = textwrap.fill(
                keys_str,
                width=72,
                initial_indent=f"  {self.subset_key_name} ({n}): ",
                subsequent_indent=" " * (len(self.subset_key_name) + 7),
                break_on_hyphens=False,
            )
            lines.append(wrapped)
            second = self.subsets[1] if len(self.subsets) > 1 else self.subsets[0]
            example = f'["{self.subsets[0]}", "{second}"]'
            lines.append(f'  usage       : cb.dataset("{self.identifier}", subset={example})')
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
        if self.license is not None:
            lines.append(f"  license     : {self.license}")
            if self.info_url is not None:
                lines.append(f"  license url : {self.info_url}")
        elif self.info_url is not None:
            lines.append(f"  info        : {self.info_url}")
        if self.auth_hint is not None:
            lines.append(f"  auth        : {self.auth_hint}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def dataset(
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

                  cb.dataset("swiss_eeg_long", subset={"ID01": 2})
                  cb.dataset("swiss_eeg_long", subset={"ID01": ["ID01_1h.mat"]})
                  cb.dataset("swiss_eeg_long", subset={"ID01": None, "ID02": 3})

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
        RuntimeError: If ``accept=False`` and the user declines the download.
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
            keys_to_validate = list(subset.keys()) if isinstance(subset, dict) else list(subset)
            known = spec.subset_keys()
            if known is not None:
                invalid = [s for s in keys_to_validate if s not in known]
                if invalid:
                    raise ValueError(
                        f"Unknown subset keys for '{identifier}': {invalid}.\n"
                        f"Valid {spec.subset_key_name or 'keys'}: {known}"
                    )
            # Validate dict values before any download.
            if isinstance(subset, dict):
                for key, value in subset.items():
                    if isinstance(value, int) and value < 1:
                        raise ValueError(
                            f"File count for subset key '{key}' must be >= 1, got {value!r}."
                        )
                    if isinstance(value, list) and not value:
                        raise ValueError(
                            f"File list for subset key '{key}' must be non-empty; "
                            "use None to include all files."
                        )

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


def download(
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
        RuntimeError: If ``accept=False`` and the user declines the download.
    """
    if identifier in {*_LOCAL_DATASET_INFO}:
        raise ValueError(f"'{identifier}' is a local dataset — no download needed.")

    spec = get_remote_dataset_spec(identifier)
    if spec is not None:
        if subset is not None:
            keys_to_validate = list(subset.keys()) if isinstance(subset, dict) else list(subset)
            known = spec.subset_keys()
            if known is not None:
                invalid = [s for s in keys_to_validate if s not in known]
                if invalid:
                    raise ValueError(
                        f"Unknown subset keys for '{identifier}': {invalid}.\n"
                        f"Valid {spec.subset_key_name or 'keys'}: {known}"
                    )
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


def describe_all() -> list[dict[str, str | None]]:
    """Print a compact summary table of all available datasets and return the data.

    Example::

        cb.describe_all()
        # Dataset               Type    Cached  Size           Subsets       License
        # --------------------  ------  ------  -------------  ------------  ----------
        # bonn_eeg              remote  yes     ~10 MB         5 sets        CC BY ...
        # dummy_chain           local   —       —              —             —
        # ...

    Returns:
        List of dicts with keys ``"identifier"``, ``"type"``, ``"cached"``,
        ``"size"``, ``"subsets"``, and ``"license"``.  Suitable for
        programmatic use in notebooks or scripts.  ``"cached"`` is
        ``"yes"``/``"no"`` for remote datasets and ``None`` for local ones.
    """
    col_id = 22
    col_type = 8
    col_cached = 8
    col_size = 15
    col_subsets = 14
    header = (
        f"{'Dataset':<{col_id}}{'Type':<{col_type}}{'Cached':<{col_cached}}"
        f"{'Size':<{col_size}}{'Subsets':<{col_subsets}}License"
    )
    sep = (
        f"{'-' * col_id}{'-' * col_type}{'-' * col_cached}"
        f"{'-' * col_size}{'-' * col_subsets}{'-' * 20}"
    )
    print(header)
    print(sep)

    rows: list[dict[str, str | None]] = []
    all_ids = sorted(_LOCAL_DATASET_INFO) + sorted(REMOTE_DATASETS)
    for ident in all_ids:
        if ident in _LOCAL_DATASET_INFO:
            row_type = "local"
            cached_str = "—"
            cached_val: str | None = None
            size = "—"
            subsets = "—"
            license_str = "—"
        else:
            spec = get_remote_dataset_spec(ident)
            if spec is None:
                continue
            row_type = "remote"
            cached_val = "yes" if _is_dataset_cached(spec) else "no"
            cached_str = cached_val
            raw_size = spec.size_hint or "—"
            size = raw_size[:14] + "…" if len(raw_size) > 15 else raw_size
            n = len(spec.subset_keys() or []) if spec.subset_keys() else None
            subsets = f"{n} {spec.subset_key_name or 'subsets'}" if n else "—"
            lic = spec.license or "—"
            license_str = lic[:30] + "…" if len(lic) > 31 else lic

        print(
            f"{ident:<{col_id}}{row_type:<{col_type}}{cached_str:<{col_cached}}"
            f"{size:<{col_size}}{subsets:<{col_subsets}}{license_str}"
        )
        rows.append(
            {
                "identifier": ident,
                "type": row_type,
                "cached": cached_val,
                "size": None if size == "—" else size,
                "subsets": None if subsets == "—" else subsets,
                "license": None if license_str == "—" else license_str,
            }
        )

    return rows


def dataset_info(identifier: str) -> DatasetInfo:
    """Return metadata for a dataset, including available subset keys.

    Example::

        info = cb.dataset_info("swiss_eeg_short")
        print(info)
        # DatasetInfo: swiss_eeg_short
        #   description  : Short-term scalp EEG ...
        #   subjects (18): ID1, ID2, ID4a, ...
        #   usage        : cb.dataset("swiss_eeg_short", subset=["ID1", "ID2"])

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
    :func:`describe_all` to see which datasets are currently cached.

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
        RuntimeError: If ``confirm=True`` and the user declines the deletion.

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
