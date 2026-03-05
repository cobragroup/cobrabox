from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import xarray as xr

from ..base_feature import BaseFeature
from ..data import Data, SignalData


@dataclass
class AmplitudeVariation(BaseFeature[SignalData]):
    """Compute amplitude variation over the time dimension.

    Amplitude variation is the standard deviation of the EEG signal within
    the calculation window. A larger value indicates a more variable signal.

    Args:
        None

    Returns:
        xarray DataArray with the ``time`` dimension removed. Shape is
        ``(space,)`` for standard input, or ``(*extra_dims, space)`` if
        additional dimensions are present (e.g. ``window_index``). Values
        are in the same units as the input signal.

    Example:
        >>> result = cb.feature.AmplitudeVariation().apply(data)
    """

    output_type: ClassVar[type[Data]] = Data  # pyright: ignore[reportIncompatibleVariableOverride]

    def __call__(self, data: SignalData) -> xr.DataArray:
        return data.data.std(dim="time")
