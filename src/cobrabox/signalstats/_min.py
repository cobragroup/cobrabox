from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import xarray as xr

from .._functional import functional
from ..base_feature import BaseFeature
from ..data import Data


@dataclass
class Min(BaseFeature[Data]):
    """Compute the minimum value across a dimension.

    Args:
        dim: Name of the dimension to reduce over (e.g. ``"time"``).

    Returns:
        :class:`~cobrabox.Data` with ``dim`` removed. Shape is the input shape
        minus the reduced dimension. Values are the per-position minimum
        in the same units as the input signal.

    Example:
        >>> result = cb.Min(dim="time").apply(data)
    """

    _tags: ClassVar[list[str]] = ["reduction", "io:scalar"]

    dim: str

    def __call__(self, data: Data) -> xr.DataArray:
        if self.dim not in data.data.dims:
            raise ValueError(f"dim '{self.dim}' not found in data dimensions {data.data.dims}")
        return data.data.min(dim=self.dim)


@functional(Min)
def min(data: Data, dim: str) -> Data:
    """Compute the minimum value across a dimension.

    Args:
        data: The input data to process, as a :class:`~cobrabox.Data`.
        dim: Name of the dimension to reduce over (e.g. ``"time"``).

    Returns:
        :class:`~cobrabox.Data` with ``dim`` removed. Shape is the input shape
        minus the reduced dimension. Values are the per-position minimum
        in the same units as the input signal.

    Example:
        >>> result = cb.min(data, dim="time")
    """
    return Min(dim=dim).apply(data)
