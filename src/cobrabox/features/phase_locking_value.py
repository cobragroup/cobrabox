import numpy as np
import xarray as xr
from scipy.signal import hilbert

from ..data import Data
from ..function_wrapper import feature


def _compute_phase_locking_value(x: np.ndarray, y: np.ndarray) -> float:
    """Compute phase locking value between x and y.

    Args:
        x: 1D array of shape (n_samples,)
        y: 1D array of shape (n_samples,)

    Returns:
        Phase locking value (float)
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if np.allclose(x, y):
        return 1.0

    return np.abs(np.mean(np.exp(1j * (np.angle(hilbert(x)) - np.angle(hilbert(y))))))


@feature
def phase_locking_value(data: Data, coord_x: str, coord_y: str) -> xr.DataArray:
    """Compute phase locking value (PLV) between two coordinates.

    All variables (x, y) are coordinate names from the `space` dimension.
    Samples are drawn from the `time` dimension.

    Args:
        data: Data object with time and space dimensions
        coord_x: Name of first coordinate (from space dimension)
        coord_y: Name of second coordinate (from space dimension)

    Returns:
        xarray DataArray with a single value (PLV)

    Raises:
        ValueError: If space dimension not in data
        ValueError: If any coordinate not found in space dimension
    """
    xr_data = data.data
    space_dim = "space"

    if space_dim not in xr_data.dims:
        raise ValueError(f"dimension '{space_dim}' not found in data dimensions {xr_data.dims}")

    time_dim = "time"
    if time_dim not in xr_data.dims:
        raise ValueError(f"data must have '{time_dim}' dimension")

    space_coords = xr_data.coords[space_dim].values

    if coord_x not in space_coords:
        raise ValueError(f"coordinate '{coord_x}' not found in space dimension: {space_coords}")
    if coord_y not in space_coords:
        raise ValueError(f"coordinate '{coord_y}' not found in space dimension: {space_coords}")

    x_series = xr_data.sel({space_dim: coord_x}).values
    y_series = xr_data.sel({space_dim: coord_y}).values

    result = _compute_phase_locking_value(x_series, y_series)

    time_coord = xr_data.coords[time_dim].values
    return (
        xr.DataArray(result, dims=[])
        .expand_dims(time_dim, axis=0)
        .assign_coords({time_dim: [time_coord[0]]})
        .expand_dims(space_dim, axis=0)
        .assign_coords({space_dim: [coord_x]})
    )


@feature
def phase_locking_value_matrix(data: Data, coords: list[str]) -> xr.DataArray:
    """Compute pairwise phase locking value matrix.

    Computes phase locking value for every pair of coordinates in coords.
    All coordinates must be from the space dimension.

    Args:
        data: Data object with time and space dimensions
        coords: List of coordinate names to compute pairwise phase locking values

    Returns:
        xarray DataArray with shape (len(coords), len(coords))

    Raises:
        ValueError: If coords is empty
        ValueError: If any coordinate not found in space dimension
    """
    xr_data = data.data
    space_dim = "space"

    if space_dim not in xr_data.dims:
        raise ValueError(f"dimension '{space_dim}' not found in data dimensions {xr_data.dims}")

    time_dim = "time"
    if time_dim not in xr_data.dims:
        raise ValueError(f"data must have '{time_dim}' dimension")

    if not coords:
        raise ValueError("coords must have at least one coordinate")

    space_coords = xr_data.coords[space_dim].values

    for c in coords:
        if c not in space_coords:
            raise ValueError(f"coordinate '{c}' not found in space dimension: {space_coords}")

    n = len(coords)
    result = np.full((n, n), np.nan)

    for i, coord_i in enumerate(coords):
        x_series = xr_data.sel({space_dim: coord_i}).values
        for j, coord_j in enumerate(coords):
            y_series = xr_data.sel({space_dim: coord_j}).values
            result[i, j] = _compute_phase_locking_value(x_series, y_series)

    time_coord = xr_data.coords[time_dim].values
    return (
        xr.DataArray(
            result, dims=["coord_i", "coord_j"], coords={"coord_i": coords, "coord_j": coords}
        )
        .expand_dims(time_dim, axis=0)
        .assign_coords({time_dim: [time_coord[0]]})
        .expand_dims(space_dim, axis=0)
        .assign_coords({space_dim: [coords[0]]})
    )
