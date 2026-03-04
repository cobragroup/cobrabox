import numpy as np
import xarray as xr

from ..data import Data
from ..function_wrapper import feature


@feature
def sliding_window(data: Data, window_size: int = 10, step_size: int = 5) -> xr.DataArray:
    """Apply sliding window to time-series data.

    Creates overlapping windows along the 'time' dimension.
    Output uses:
    - 'time': window sequence index (one entry per window)
    - 'window_index': local index inside each window

    Args:
        data: Data with 'time' and 'space' dimensions
        window_size: Size of each window in timepoints
        step_size: Step size between windows in timepoints

    Returns:
        xarray DataArray with added 'window_index' dimension

    Example:
        >>> data = cb.dataset("fMRI_sim2")
        >>> wdata = cb.feature.sliding_window(data, window_size=20, step_size=10)
    """
    # Extract xarray DataArray
    xr_data = data.data

    # Validate dimensions
    if "time" not in xr_data.dims:
        raise ValueError("data must have 'time' dimension")

    n_time = xr_data.sizes["time"]

    # Calculate number of windows
    n_windows = (n_time - window_size) // step_size + 1
    if n_windows <= 0:
        raise ValueError(f"window_size ({window_size}) must be <= n_time ({n_time})")

    # Create window indices
    window_starts = np.arange(0, n_time - window_size + 1, step_size)
    window_starts = window_starts[:n_windows]  # Ensure we don't exceed

    # Extract windows
    windows = []
    for start in window_starts:
        end = start + window_size
        # Convert each [time, ...] slice into [window_index, ...] so reducing
        # over window_index means "reduce within each window".
        window = xr_data.isel(time=slice(start, end)).rename({"time": "window_index"})
        window = window.assign_coords(window_index=np.arange(window_size))
        windows.append(window)

    # Stack windows along the time axis (one timepoint per window start).
    stacked = xr.concat(windows, dim="time", join="inner")
    return stacked.assign_coords(time=xr_data.coords["time"].values[window_starts])
