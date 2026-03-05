from __future__ import annotations

from .data import SignalData
from .dataset import Dataset
from .dataset_loader import load_noise_dummy, load_realistic_swiss, load_structured_dummy


def dataset(identifier: str) -> Dataset[SignalData]:
    """Load one logical dataset identifier as a Dataset of Data parts."""
    if identifier in {"dummy_chain", "dummy_random", "dummy_star"}:
        return load_structured_dummy(identifier)
    if identifier == "dummy_noise":
        return load_noise_dummy(identifier)
    if identifier == "realistic_swiss":
        return load_realistic_swiss(identifier)
    raise ValueError(f"Unknown dataset identifier: {identifier}")
