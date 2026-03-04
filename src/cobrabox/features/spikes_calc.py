from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from ..base_feature import BaseFeature
from ..data import Data


@dataclass
class SpikesCalc(BaseFeature[Data]):
    """Calculate spikes in the input data using the IQR method.

    Detects outliers as values falling outside ±1.5*IQR from Q1/Q3.
    Returns a 2D array shaped (1, 1) containing the spike count.

    Args:
        None

    Returns:
        xr.DataArray with shape (1, 1), dims ["time", "space"],
        containing the spike count as a float value.

    Example:
        >>> result = SpikesCalc().apply(data)
    """

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

        # Reshape to 2D array (1, 1) required by Data architecture
        result = np.array([[spike_count]], dtype=float)

        return xr.DataArray(result, dims=["time", "space"], coords={"time": [0]})
