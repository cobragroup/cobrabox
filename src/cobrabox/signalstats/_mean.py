from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import xarray as xr

from .._functional import functional
from ..base_feature import BaseFeature
from ..data import Data


@dataclass
class Mean(BaseFeature[Data]):
    """Compute the mean value across a dimension.

    Args:
        dim: Name of the dimension to reduce over (e.g. ``"time"``).

    Returns:
        :class:`~cobrabox.Data` with ``dim`` removed. Shape is the input shape
        minus the reduced dimension. Values are the per-position arithmetic
        mean in the same units as the input signal.

    Raises:
        ValueError: If the specified dimension is not found in the data.

    Example:
        >>> result = cb.Mean(dim="time").apply(data)
    """

    _tags: ClassVar[list[str]] = ["reduction", "io:scalar"]

    dim: str

    def __call__(self, data: Data) -> xr.DataArray:
        if self.dim not in data.data.dims:
            raise ValueError(f"dim '{self.dim}' not found in data dimensions {data.data.dims}")
        return data.data.mean(dim=self.dim)


@functional(Mean)
def mean(data: Data, dim: str) -> Data:
    """Compute the mean value across a dimension.

    Args:
        data: The input data to process, as a :class:`~cobrabox.Data`.
        dim: Name of the dimension to reduce over (e.g. ``"time"``).

    Returns:
        :class:`~cobrabox.Data` with ``dim`` removed. Shape is the input shape
        minus the reduced dimension. Values are the per-position arithmetic
        mean in the same units as the input signal.

    Raises:
        ValueError: If the specified dimension is not found in the data.

    Example:
        >>> result = cb.mean(data, dim="time")
    """
    return Mean(dim=dim).apply(data)
