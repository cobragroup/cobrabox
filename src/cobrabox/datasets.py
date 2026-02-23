from __future__ import annotations

from .data import Dataset
from .dataset_loader import load_noise_dummy, load_structured_dummy


def dataset(identifier: str) -> list[Dataset]:
    """Load one logical dataset identifier as a list of Dataset parts."""
    if identifier in {"dummy_chain", "dummy_random", "dummy_star"}:
        return load_structured_dummy(identifier)
    if identifier == "dummy_noise":
        return load_noise_dummy(identifier)
    raise ValueError(f"Unknown dataset identifier: {identifier}")
