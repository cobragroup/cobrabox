import numpy as np
import xarray as xr

from ..data import Data
from ..function_wrapper import feature


def _compute_partial_correlation(x: np.ndarray, y: np.ndarray, controls: list[np.ndarray]) -> float:
    """Compute partial correlation between x and y controlling for variables in controls.

    Args:
        x: 1D array of shape (n_samples,)
        y: 1D array of shape (n_samples,)
        controls: list of 1D arrays, each of shape (n_samples,)

    Returns:
        Partial correlation coefficient (float)
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    controls = [np.asarray(c).ravel() for c in controls]

    if np.allclose(x, y):
        return 1.0

    all_vars_arr = np.column_stack([x, y, *controls])

    corr_matrix = np.corrcoef(all_vars_arr, rowvar=False)

    precision_matrix = np.linalg.inv(corr_matrix)

    r_xy_z = -precision_matrix[0, 1] / np.sqrt(precision_matrix[0, 0] * precision_matrix[1, 1])

    return float(r_xy_z)


@feature
def partial_correlation(
    data: Data, coord_x: str, coord_y: str, control_vars: list[str]
) -> xr.DataArray:
    """Compute partial correlation between two coordinates while controlling for others.

    All variables (x, y, controls) are coordinate names from the space dimension.
    Samples are drawn from the time dimension.

    Args:
        data: Data object with time and space dimensions
        coord_x: Name of first coordinate (from space dimension)
        coord_y: Name of second coordinate (from space dimension)
        control_vars: List of coordinate names to control for (from space dimension, min 1)

    Returns:
        xarray DataArray with a single value (partial correlation coefficient)

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

    if not control_vars:
        raise ValueError("control_vars must have at least one coordinate")

    for cv in control_vars:
        if cv not in space_coords:
            raise ValueError(
                f"control coordinate '{cv}' not found in space dimension: {space_coords}"
            )

    x_series = xr_data.sel({space_dim: coord_x}).values
    y_series = xr_data.sel({space_dim: coord_y}).values

    control_series = [xr_data.sel({space_dim: cv}).values for cv in control_vars]

    result = _compute_partial_correlation(x_series, y_series, control_series)

    time_coord = xr_data.coords[time_dim].values
    return (
        xr.DataArray(result, dims=[])
        .expand_dims(time_dim, axis=0)
        .assign_coords({time_dim: [time_coord[0]]})
        .expand_dims(space_dim, axis=0)
        .assign_coords({space_dim: [coord_x]})
    )


@feature
def partial_correlation_matrix(
    data: Data, coords: list[str], control_vars: list[str]
) -> xr.DataArray:
    """Compute pairwise partial correlation matrix for multiple coordinates.

    Computes partial correlation for every pair of coordinates in coords,
    controlling for the coordinates in control_vars. All coordinates must be
    from the space dimension.

    Args:
        data: Data object with time and space dimensions
        coords: List of coordinate names to compute pairwise correlations
        control_vars: List of coordinate names to control for (from space dimension)

    Returns:
        xarray DataArray with shape (len(coords), len(coords))

    Raises:
        ValueError: If coords or control_vars is empty
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

    if not control_vars:
        raise ValueError("control_vars must have at least one coordinate")

    space_coords = xr_data.coords[space_dim].values

    for c in coords:
        if c not in space_coords:
            raise ValueError(f"coordinate '{c}' not found in space dimension: {space_coords}")

    for cv in control_vars:
        if cv not in space_coords:
            raise ValueError(
                f"control coordinate '{cv}' not found in space dimension: {space_coords}"
            )

    n = len(coords)
    result = np.full((n, n), np.nan)

    control_series = [xr_data.sel({space_dim: cv}).values for cv in control_vars]

    for i, coord_i in enumerate(coords):
        x_series = xr_data.sel({space_dim: coord_i}).values
        for j, coord_j in enumerate(coords):
            y_series = xr_data.sel({space_dim: coord_j}).values
            result[i, j] = _compute_partial_correlation(x_series, y_series, control_series)

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
