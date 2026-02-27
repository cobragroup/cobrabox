import xarray as xr

from ..data import Data
from ..function_wrapper import feature


@feature
def line_length(data: Data) -> xr.DataArray:
    """Compute line length feature.

    Line length is the sum of absolute differences between consecutive
    timepoints, summed over the time dimension. This is a dummy implementation
    for now.

    Args:
        data: Data with 'time' and 'space' dimensions
            (may also have 'window_index' if from sliding_window)

    Returns:
        xarray DataArray with 'time' dimension removed (or 'window_index' preserved)

    Example:
        >>> wdata = cb.feature.sliding_window(data)
        >>> feature = cb.feature.line_length(wdata)
    """
    # Extract xarray DataArray
    xr_data = data.data

    # Validate dimensions
    if "time" not in xr_data.dims:
        raise ValueError("data must have 'time' dimension")

    # Compute line length: sum of absolute differences along time
    # For each spatial location, compute sum(|x[t+1] - x[t]|)
    diff = xr_data.diff(dim="time")
    line_length_values = abs(diff).sum(dim="time")

    # If window_index exists, preserve it
    if "window_index" in xr_data.dims:
        # Line length per window
        return line_length_values
    # Single line length value per spatial location
    return line_length_values
