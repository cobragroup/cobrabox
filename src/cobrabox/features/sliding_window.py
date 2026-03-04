from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from ..base_feature import BaseFeature
from ..data import Data


@dataclass
class SlidingWindow(BaseFeature):
    """Apply sliding window to time-series data.

    Creates overlapping windows along the 'time' dimension.
    Output uses:
    - 'time': window sequence index (one entry per window)
    - 'window_index': local index inside each window

    Args:
        window_size: Size of each window in timepoints
        step_size: Step size between windows in timepoints

    Example:
        >>> data = cb.dataset("fMRI_sim2")
        >>> wdata = cb.feature.SlidingWindow(window_size=20, step_size=10).apply(data)
    """

    window_size: int = field(default=10)
    step_size: int = field(default=5)

    def __call__(self, data: Data) -> xr.DataArray:
        xr_data = data.data

        if "time" not in xr_data.dims:
            raise ValueError("data must have 'time' dimension")

        n_time = xr_data.sizes["time"]

        n_windows = (n_time - self.window_size) // self.step_size + 1
        if n_windows <= 0:
            raise ValueError(f"window_size ({self.window_size}) must be <= n_time ({n_time})")

        window_starts = np.arange(0, n_time - self.window_size + 1, self.step_size)
        window_starts = window_starts[:n_windows]

        windows = []
        for start in window_starts:
            end = start + self.window_size
            window = xr_data.isel(time=slice(start, end)).rename({"time": "window_index"})
            window = window.assign_coords(window_index=np.arange(self.window_size))
            windows.append(window)

        stacked = xr.concat(windows, dim="time", join="inner")
        return stacked.assign_coords(time=xr_data.coords["time"].values[window_starts])
