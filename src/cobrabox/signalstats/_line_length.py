from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import xarray as xr

from .._functional import functional
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
        :class:`~cobrabox.Data` with the ``time`` dimension removed. Shape is
        ``(space,)`` for standard input, or ``(*extra_dims, space)`` if
        additional dimensions are present (e.g. ``window_index``). Values
        are in the same units as the input signal.

    Example:
        >>> result = cb.LineLength().apply(data)
    """

    _tags: ClassVar[list[str]] = [
        "variability",
        "signal-complexity",
        "epilepsy",
        "seizure-detection",
        "eeg",
        "io:scalar-per-channel",
    ]

    output_type: ClassVar[type[Data]] = Data

    def __call__(self, data: SignalData) -> xr.DataArray:
        xr_data = data.data
        diff = xr_data.diff(dim="time")
        return abs(diff).sum(dim="time")


@functional(LineLength)
def line_length(data: SignalData) -> Data:
    """Compute line length over the time dimension.

    Line length is the sum of absolute differences between consecutive
    timepoints. A larger value indicates a more rapidly varying signal.

    Args:
        data: The input time-series signal to process, as a
            :class:`~cobrabox.SignalData` (or any :class:`~cobrabox.Data`
            carrying a ``time`` dimension).

    Returns:
        :class:`~cobrabox.Data` with the ``time`` dimension removed. Shape is
        ``(space,)`` for standard input, or ``(*extra_dims, space)`` if
        additional dimensions are present (e.g. ``window_index``). Values
        are in the same units as the input signal.

    Example:
        >>> result = cb.line_length(data)
    """
    return LineLength().apply(data)
