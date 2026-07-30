from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import xarray as xr

from .._functional import functional
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
        :class:`~cobrabox.Data` with the ``time`` dimension removed. Shape is
        ``(space,)`` for standard input, or ``(*extra_dims, space)`` if
        additional dimensions are present (e.g. ``window_index``). Values
        are in the same units as the input signal.

    Example:
        >>> result = cb.AmplitudeVariation().apply(data)
    """

    _tags: ClassVar[list[str]] = [
        "variability",
        "standard-deviation",
        "anesthesia",
        "eeg",
        "io:scalar-per-channel",
    ]

    output_type: ClassVar[type[Data]] = Data

    def __call__(self, data: SignalData) -> xr.DataArray:
        return data.data.std(dim="time")


@functional(AmplitudeVariation)
def amplitude_variation(data: SignalData) -> Data:
    """Compute amplitude variation over the time dimension.

    Amplitude variation is the standard deviation of the EEG signal within
    the calculation window. A larger value indicates a more variable signal.

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
        >>> result = cb.amplitude_variation(data)
    """
    return AmplitudeVariation().apply(data)
