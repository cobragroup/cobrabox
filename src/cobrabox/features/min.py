from __future__ import annotations

import xarray as xr

from ..data import Data
from ..function_wrapper import feature


@feature
def min(data: Data, dim: str) -> xr.DataArray:
    """Compute minimum across a required dimension."""
    xr_data = data.data
    if dim not in xr_data.dims:
        raise ValueError(f"dim '{dim}' not found in data dimensions {xr_data.dims}")
    return xr_data.min(dim=dim)
