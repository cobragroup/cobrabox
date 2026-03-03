from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import xarray as xr

from .data import Data


@dataclass
class Feature(ABC):
    """Base class for all cobrabox features.

    Features are dataclasses that store configuration at initialization
    and implement transform logic in __call__.
    """

    # Marker for dynamic feature discovery (replaces _is_cobrabox_feature)
    _is_cobrabox_feature: bool = field(default=True, init=False, repr=False)

    def __or__(self, other: Feature) -> Pipeline:
        """Enable pipe syntax: Mean(dim="time") | SlidingWindow(...)"""
        return Pipeline([self, other])

    @abstractmethod
    def __call__(self, data: Data) -> xr.DataArray | Data:
        """Apply the feature transformation. Subclasses implement this."""
        pass

    def apply(self, data: Data) -> Data:
        """Apply feature and wrap result in Data with history tracking."""
        result = self(data)
        if not isinstance(result, (xr.DataArray, Data)):
            raise TypeError(
                f"Feature '{self.__class__.__name__}' must return "
                f"xarray.DataArray or Data, got {type(result)}"
            )
        return data._copy_with_new_data(new_data=result, operation_name=self.__class__.__name__)


class Pipeline:
    """Sequential pipeline of features using pipe syntax."""

    def __init__(self, features: list[Feature]) -> None:
        self.features = features

    def __or__(self, other: Feature) -> Pipeline:
        """Enable chaining: pipeline | AnotherFeature()"""
        return Pipeline(self.features + [other])

    def apply(self, data: Data) -> Data:
        """Apply all features in sequence."""
        result = data
        for feature in self.features:
            result = feature.apply(result)
        return result

    def __call__(self, data: Data) -> Data:
        """Allow pipeline(data) syntax."""
        return self.apply(data)
