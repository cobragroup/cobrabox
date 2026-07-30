from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import xarray as xr

from .._functional import functional
from ..base_feature import BaseFeature
from ..data import Data


@dataclass
class SpikeCount(BaseFeature[Data]):
    """Calculate spikes in the input data using the IQR method.

    Detects outliers as values falling outside ±1.5*IQR from Q1/Q3.
    Returns a scalar count of detected spikes.

    Args:
        None

    Returns:
        :class:`~cobrabox.Data` with shape (), dims (),
        containing the spike count as a scalar float value.

    Raises:
        ValueError: If input data is empty.

    Example:
        >>> result = SpikeCount().apply(data)
    """

    _tags: ClassVar[list[str]] = [
        "outlier-detection",
        "artifact",
        "epilepsy",
        "spike-detection",
        "eeg",
        "io:scalar-per-channel",
    ]

    output_type: ClassVar[type[Data]] = Data

    def __call__(self, data: Data) -> xr.DataArray:
        a = data.data.values

        if a.size == 0:
            raise ValueError("Input data cannot be empty")

        # Calculate IQR bounds
        q1 = np.quantile(a, 0.25)
        q3 = np.quantile(a, 0.75)
        iqr = q3 - q1

        low_bound = q1 - 1.5 * iqr
        up_bound = q3 + 1.5 * iqr

        # Count outliers
        spike_count = np.sum((a > up_bound) | (a < low_bound))

        # Return as 0-dimensional scalar array
        return xr.DataArray(float(spike_count))


@functional(SpikeCount)
def spike_count(data: Data) -> Data:
    """Calculate spikes in the input data using the IQR method.

    Detects outliers as values falling outside ±1.5*IQR from Q1/Q3.
    Returns a scalar count of detected spikes.

    Args:
        data: The input data to process, as a :class:`~cobrabox.Data`.

    Returns:
        :class:`~cobrabox.Data` with shape (), dims (),
        containing the spike count as a scalar float value.

    Raises:
        ValueError: If input data is empty.

    Example:
        >>> result = SpikeCount().apply(data)
    """
    return SpikeCount().apply(data)
