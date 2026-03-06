from __future__ import annotations

from .data import SignalData
from .dataset import Dataset
from .dataset_loader import load_noise_dummy, load_realistic_swiss, load_structured_dummy
from .downloader import ensure_remote_files, get_remote_dataset_spec


def dataset(identifier: str) -> Dataset[SignalData]:
    """Load one logical dataset identifier as a Dataset of Data parts.

    Supports both built-in synthetic datasets and registered remote datasets.
    """
    if identifier in {"dummy_chain", "dummy_random", "dummy_star"}:
        return load_structured_dummy(identifier)
    if identifier == "dummy_noise":
        return load_noise_dummy(identifier)
    if identifier == "realistic_swiss":
        return load_realistic_swiss(identifier)

    spec = get_remote_dataset_spec(identifier)
    if spec is not None:
        dataset_dir = ensure_remote_files(spec)
        return spec.loader(dataset_dir)

    raise ValueError(f"Unknown dataset identifier: {identifier}")
