from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import xarray as xr

from .._functional import functional
from ..base_feature import BaseFeature
from ..data import Data


@dataclass
class Max(BaseFeature[Data]):
    """Compute the maximum value across a dimension.

    Args:
        dim: Name of the dimension to reduce over (e.g. ``"time"``).

    Returns:
        :class:`~cobrabox.Data` with ``dim`` removed. Shape is the input shape
        minus the reduced dimension. Values are the per-position maximum
        in the same units as the input data.

    Raises:
        ValueError: If the specified dimension is not found in the data.

    Example:
        >>> result = cb.Max(dim="time").apply(data)
    """

    _tags: ClassVar[list[str]] = ["reduction", "io:scalar"]

    output_type: ClassVar[type[Data]] = Data

    dim: str

    def __call__(self, data: Data) -> xr.DataArray:
        if self.dim not in data.data.dims:
            raise ValueError(f"dim '{self.dim}' not found in data dimensions {data.data.dims}")
        return data.data.max(dim=self.dim)


@functional(Max)
def max(data: Data, dim: str) -> Data:
    """Compute the maximum value across a dimension.

    Args:
        data: The input data to process, as a :class:`~cobrabox.Data`.
        dim: Name of the dimension to reduce over (e.g. ``"time"``).

    Returns:
        :class:`~cobrabox.Data` with ``dim`` removed. Shape is the input shape
        minus the reduced dimension. Values are the per-position maximum
        in the same units as the input data.

    Raises:
        ValueError: If the specified dimension is not found in the data.

    Example:
        >>> result = cb.max(data, dim="time")
    """
    return Max(dim=dim).apply(data)
