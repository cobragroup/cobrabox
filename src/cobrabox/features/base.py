from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..data import TimeSeriesDataset


@dataclass(frozen=True)
class Feature(ABC):
    """Base class for time-series features."""

    name: str

    @abstractmethod
    def calculate(self, dataset: TimeSeriesDataset) -> np.ndarray:
        """Compute feature(s) from a dataset."""
