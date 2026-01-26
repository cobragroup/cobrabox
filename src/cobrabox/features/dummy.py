from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import Feature
from ..data import TimeSeriesDataset


@dataclass(frozen=True)
class DummyFeature(Feature):
    """Simple placeholder feature: overall mean across all channels."""

    def calculate(self, dataset: TimeSeriesDataset) -> np.ndarray:
        value = float(np.mean(dataset.data))
        return np.asarray([[value]])

