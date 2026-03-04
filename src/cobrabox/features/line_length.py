from __future__ import annotations

from dataclasses import dataclass

import xarray as xr

from ..base_feature import BaseFeature
from ..data import Data


@dataclass
class LineLength(BaseFeature):
    """Compute line length feature.

    Line length is the sum of absolute differences between consecutive
    timepoints, summed over the time dimension.

    Args:
        data: Data with 'time' and 'space' dimensions
            (may also have 'window_index' if from SlidingWindow)

    Returns:
        xarray DataArray with 'time' dimension removed (or 'window_index' preserved)

    Example:
        >>> wdata = cb.feature.SlidingWindow().apply(data)
        >>> out = cb.feature.LineLength().apply(wdata)
    """

    def __call__(self, data: Data) -> xr.DataArray:
        xr_data = data.data

        if "time" not in xr_data.dims:
            raise ValueError("data must have 'time' dimension")

        diff = xr_data.diff(dim="time")
        return abs(diff).sum(dim="time")
