from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import xarray as xr

from ...base_feature import BaseFeature
from ...data import Data, SignalData


def _compute_partial_correlation(x: np.ndarray, y: np.ndarray, controls: list[np.ndarray]) -> float:
    """Compute partial correlation between x and y controlling for variables in controls.

    Args:
        x: 1D array of shape (n_samples,)
        y: 1D array of shape (n_samples,)
        controls: list of 1D arrays, each of shape (n_samples,)

    Returns:
        Partial correlation coefficient (float)

    Raises:
        ValueError: If correlation matrix is singular (non-invertible)
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    controls = [np.asarray(c).ravel() for c in controls]

    if np.allclose(x, y):
        return 1.0

    all_vars_arr = np.column_stack([x, y, *controls])

    corr_matrix = np.corrcoef(all_vars_arr, rowvar=False)

    try:
        precision_matrix = np.linalg.inv(corr_matrix)
    except np.linalg.LinAlgError as e:
        raise ValueError(
            f"Correlation matrix is singular (non-invertible). "
            f"This can happen when variables are perfectly correlated or "
            f"when there are insufficient samples. "
            f"Original error: {e}"
        ) from e

    r_xy_z = -precision_matrix[0, 1] / np.sqrt(precision_matrix[0, 0] * precision_matrix[1, 1])

    return float(r_xy_z)


@dataclass
class PartialCorrelation(BaseFeature[SignalData]):
    """Compute partial correlation between two coordinates while controlling for others.

    All variables (x, y, controls) are coordinate names from the space dimension.
    Samples are drawn from the time dimension.

    Args:
        coord_x: Name of first coordinate (from space dimension)
        coord_y: Name of second coordinate (from space dimension)
        control_vars: List of coordinate names to control for (from space dimension, min 1)

    Returns:
        xarray DataArray with a single value (partial correlation coefficient)

    Raises:
        ValueError: If space dimension not in data
        ValueError: If any coordinate not found in space dimension

    Example:
        >>> result = cb.feature.PartialCorrelation(
        ...     coord_x=0, coord_y=1, control_vars=[2]
        ... ).apply(data)
    """

    output_type: ClassVar[type[Data]] = Data

    coord_x: str | int
    coord_y: str | int
    control_vars: list[str] | list[int]

    def __call__(self, data: SignalData) -> xr.DataArray:
        xr_data = data.data
        space_dim = "space"

        if space_dim not in xr_data.dims:
            raise ValueError(f"dimension '{space_dim}' not found in data dimensions {xr_data.dims}")

        time_dim = "time"
        if time_dim not in xr_data.dims:
            raise ValueError(f"data must have '{time_dim}' dimension")

        space_coords = xr_data.coords[space_dim].values

        if self.coord_x not in space_coords:
            raise ValueError(
                f"coordinate '{self.coord_x}' not found in space dimension: {space_coords}"
            )
        if self.coord_y not in space_coords:
            raise ValueError(
                f"coordinate '{self.coord_y}' not found in space dimension: {space_coords}"
            )

        if not self.control_vars:
            raise ValueError("control_vars must have at least one coordinate")

        for cv in self.control_vars:
            if cv not in space_coords:
                raise ValueError(
                    f"control coordinate '{cv}' not found in space dimension: {space_coords}"
                )

        x_series = xr_data.sel({space_dim: self.coord_x}).values
        y_series = xr_data.sel({space_dim: self.coord_y}).values

        control_series = [xr_data.sel({space_dim: cv}).values for cv in self.control_vars]

        result = _compute_partial_correlation(x_series, y_series, control_series)

        return xr.DataArray(result)


@dataclass
class PartialCorrelationMatrix(BaseFeature[SignalData]):
    """Compute pairwise partial correlation matrix for multiple coordinates.

    Computes partial correlation for every pair of coordinates in coords,
    controlling for the coordinates in control_vars. All coordinates must be
    from the space dimension.

    Args:
        control_vars: List of coordinate names to control for (from space dimension)
        coords: List of coordinate names to compute pairwise correlations.
                If None, uses all space coordinates excluding control_vars.

    Returns:
        xarray DataArray with shape (len(coords), len(coords))

    Raises:
        ValueError: If coords is an empty list.
        ValueError: If control_vars is empty
        ValueError: If any coordinate not found in space dimension

    Example:
        >>> # Test specific coordinates
        >>> result = cb.feature.PartialCorrelationMatrix(
        ...     control_vars=[3], coords=[0, 1, 2]
        ... ).apply(data)
        >>> # Test all coordinates (default)
        >>> result_all = cb.feature.PartialCorrelationMatrix(control_vars=[3]).apply(data)
    """

    output_type: ClassVar[type[Data]] = Data

    control_vars: list[str] | list[int]
    coords: list[str] | list[int] | None = None

    def __call__(self, data: SignalData) -> xr.DataArray:
        xr_data = data.data
        space_dim = "space"

        if space_dim not in xr_data.dims:
            raise ValueError(f"dimension '{space_dim}' not found in data dimensions {xr_data.dims}")

        time_dim = "time"
        if time_dim not in xr_data.dims:
            raise ValueError(f"data must have '{time_dim}' dimension")

        if not self.control_vars:
            raise ValueError("control_vars must have at least one coordinate")

        space_coords = xr_data.coords[space_dim].values

        for cv in self.control_vars:
            if cv not in space_coords:
                raise ValueError(
                    f"control coordinate '{cv}' not found in space dimension: {space_coords}"
                )

        if self.coords is None:
            coords_list = [c for c in space_coords if c not in set(self.control_vars)]
        else:
            if not self.coords:
                raise ValueError("coords must have at least one coordinate")
            coords_list = self.coords

        for c in coords_list:
            if c not in space_coords:
                raise ValueError(f"coordinate '{c}' not found in space dimension: {space_coords}")

        n = len(coords_list)
        result = np.full((n, n), np.nan)

        control_series = [xr_data.sel({space_dim: cv}).values for cv in self.control_vars]

        for i, coord_i in enumerate(coords_list):
            x_series = xr_data.sel({space_dim: coord_i}).values
            for j, coord_j in enumerate(coords_list):
                y_series = xr_data.sel({space_dim: coord_j}).values
                result[i, j] = _compute_partial_correlation(x_series, y_series, control_series)

        return xr.DataArray(
            result,
            dims=["coord_i", "coord_j"],
            coords={"coord_i": coords_list, "coord_j": coords_list},
        )
