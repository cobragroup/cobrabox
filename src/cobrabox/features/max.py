from __future__ import annotations

from dataclasses import dataclass

import xarray as xr

from ..base_feature import BaseFeature
from ..data import Data


@dataclass
class Max(BaseFeature):
    """Compute maximum across a dimension."""

    dim: str

    def __call__(self, data: Data) -> xr.DataArray:
        if self.dim not in data.data.dims:
            raise ValueError(f"dim '{self.dim}' not found in data dimensions {data.data.dims}")
        return data.data.max(dim=self.dim)
