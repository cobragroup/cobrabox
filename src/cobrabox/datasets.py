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

    def __str__(self) -> str:
        lines = [f"DatasetInfo: {self.identifier}"]
        lines.append(f"  description : {self.description}")
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
            if self.seizure_info_url is not None:
                lines.append(f"  seizure src : {self.seizure_info_url}")
        elif self.seizure_info_url is not None:
            lines.append(f"  seizure info: {self.seizure_info_url}")
        if self.info_url is not None:
            lines.append(f"  info        : {self.info_url}")
        if self.license is not None:
            lines.append(f"  license     : {self.license}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def dataset(
    identifier: str, *, subset: SubsetSpec | None = None, verify: bool = True
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
        verify: If ``True`` (default) and files need to be downloaded, show
            dataset info and an estimated download size and ask for
            confirmation before proceeding.  Set to ``False`` to skip the
            prompt (e.g. in scripts).

    Returns:
        :class:`~cobrabox.Dataset` of :class:`~cobrabox.SignalData` objects.

    Raises:
        ValueError: If ``identifier`` is unknown, or if ``subset`` contains
            keys not present in the dataset.
        RuntimeError: If ``verify=True`` and the user declines the download.
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

        dataset_dir = ensure_remote_files(spec, subset=subset, verify=verify)

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


def list_datasets() -> list[str]:
    """Return all known dataset identifiers.

    Includes built-in local datasets (no download required) and all
    registered remote datasets.

    Example::

        cb.list_datasets()
        # ['dummy_chain', 'dummy_noise', 'dummy_random', 'dummy_star',
        #  'realistic_swiss', 'bonn_eeg', 'chb_mit', ...]

    Returns:
        Sorted list of dataset identifier strings.
    """
    return sorted([*_LOCAL_DATASET_INFO, *REMOTE_DATASETS])


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
        )

    raise ValueError(
        f"Unknown dataset identifier: {identifier!r}. "
        f"Known identifiers: {[*sorted(_LOCAL_DATASET_INFO), *REMOTE_DATASETS]}"
    )
