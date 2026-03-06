from __future__ import annotations

import textwrap
from collections.abc import Sequence
from dataclasses import dataclass

from .data import SignalData
from .dataset import Dataset
from .dataset_loader import load_noise_dummy, load_realistic_swiss, load_structured_dummy
from .downloader import ensure_remote_files, get_remote_dataset_spec

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
    """

    identifier: str
    description: str
    subset_key_name: str | None
    subsets: tuple[str, ...] | None

    def __str__(self) -> str:
        lines = [f"DatasetInfo: {self.identifier}"]
        lines.append(f"  description : {self.description}")
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
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def dataset(identifier: str, *, subset: Sequence[str] | None = None) -> Dataset[SignalData]:
    """Load a dataset by identifier.

    Args:
        identifier: Dataset name, e.g. ``"dummy_chain"`` or
            ``"swiss_eeg_short"``.
        subset: Optional list of subset keys to load (e.g. subject IDs for
            ``"swiss_eeg_short"``).  Only files matching these keys are
            downloaded and loaded.  Call :func:`dataset_info` to see which
            keys are available.  ``None`` loads everything.

    Returns:
        :class:`~cobrabox.Dataset` of :class:`~cobrabox.SignalData` objects.

    Raises:
        ValueError: If ``identifier`` is unknown, or if ``subset`` contains
            keys not present in the dataset.
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
            known = spec.subset_keys()
            if known is not None:
                invalid = [s for s in subset if s not in known]
                if invalid:
                    raise ValueError(
                        f"Unknown subset keys for '{identifier}': {invalid}.\n"
                        f"Valid {spec.subset_key_name or 'keys'}: {known}"
                    )
        dataset_dir = ensure_remote_files(spec, subset=subset)
        return spec.loader(dataset_dir, subset)

    _remote = ["swiss_eeg_short", "swiss_eeg_long", "bonn_eeg"]
    raise ValueError(
        f"Unknown dataset identifier: {identifier!r}. "
        f"Known identifiers: {[*sorted(_LOCAL_DATASET_INFO), *_remote]}"
    )


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
        )

    _remote = ["swiss_eeg_short", "swiss_eeg_long", "bonn_eeg"]
    raise ValueError(
        f"Unknown dataset identifier: {identifier!r}. "
        f"Known identifiers: {[*sorted(_LOCAL_DATASET_INFO), *_remote]}"
    )
