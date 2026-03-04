from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import xarray as xr

from ..base_feature import BaseFeature
from ..data import Data, SignalData


@dataclass
class LineLength(BaseFeature[SignalData]):
    """Compute line length over the time dimension.

    Line length is the sum of absolute differences between consecutive
    timepoints. A larger value indicates a more rapidly varying signal.

    Args:
        None

    Returns:
        xarray DataArray with the ``time`` dimension removed. Shape is
        ``(space,)`` for standard input, or ``(*extra_dims, space)`` if
        additional dimensions are present (e.g. ``window_index``). Values
        are in the same units as the input signal.

    Example:
        >>> result = cb.feature.LineLength().apply(data)
    """

    output_type: ClassVar[type[Data]] = Data

    def __call__(self, data: SignalData) -> xr.DataArray:
        xr_data = data.data
        diff = xr_data.diff(dim="time")
        return abs(diff).sum(dim="time")
