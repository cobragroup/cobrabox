import numpy as np
import xarray as xr

from ..data import Data
from ..function_wrapper import feature


@feature
def spikes_calc(data: Data, mandatory_arg: int) -> xr.DataArray:
    """Calculate spikes in the input data using the IQR method.
    Detects outliers as values falling outside ±1.5*IQR from Q1/Q3.
    Returns a 2D array shaped (1, 1) containing the spike count.
    Parameters:
    -----------
    data : Data
        Input data with 'time' and 'space' dimensions.
    mandatory_arg : int
        Placeholder for potential future use (not used currently).
    Returns:
    --------
    xr.DataArray
        2D array of shape (1, 1) containing the count of detected spikes.
    """
    a = data.to_numpy()

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
