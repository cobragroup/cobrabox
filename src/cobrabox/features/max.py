from __future__ import annotations

from dataclasses import dataclass

import xarray as xr

from ..base_feature import BaseFeature
from ..data import Data


@dataclass
class Max(BaseFeature):
    """Compute the maximum value across a dimension.

    Args:
        dim: Name of the dimension to reduce over (e.g. ``"time"``).

    Returns:
        xarray DataArray with ``dim`` removed. Shape is the input shape
        minus the reduced dimension. Values are the per-position maximum
        in the same units as the input signal.

    Example:
        >>> result = cb.feature.Max(dim="time").apply(data)
    """

    dim: str

    def __call__(self, data: Data) -> xr.DataArray:
        if self.dim not in data.data.dims:
            raise ValueError(f"dim '{self.dim}' not found in data dimensions {data.data.dims}")
        return data.data.max(dim=self.dim)
