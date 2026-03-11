from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import xarray as xr
from scipy.signal import hilbert

from ...base_feature import BaseFeature
from ...data import Data, SignalData


def _compute_plv(x: np.ndarray, y: np.ndarray) -> float:
    """Compute phase locking value between x and y.

    Args:
        x: 1D array of shape (n_samples,).
        y: 1D array of shape (n_samples,).

    Returns:
        Phase locking value (float).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if np.allclose(x, y):
        return 1.0

    return float(np.abs(np.mean(np.exp(1j * (np.angle(hilbert(x)) - np.angle(hilbert(y)))))))


@dataclass
class PhaseLockingValue(BaseFeature[SignalData]):
    """Compute phase locking value (PLV) between two coordinates.

    Both coordinates must come from the ``space`` dimension; samples are drawn
    from the ``time`` dimension. PLV measures phase synchrony between two signals
    in [0, 1], where 1 indicates perfect phase locking.

    Args:
        coord_x: Name of the first coordinate (from space dimension).
        coord_y: Name of the second coordinate (from space dimension).

    Returns:
        Scalar xarray DataArray (0-dimensional) containing the PLV value.

    Raises:
        ValueError: If ``space`` dimension is not in data or either coordinate is missing.

    Example:
        >>> result = cb.feature.PhaseLockingValue(coord_x=0, coord_y=1).apply(data)
    """

    output_type: ClassVar[type[Data]] = Data

    coord_x: str | int
    coord_y: str | int

    def __call__(self, data: SignalData) -> xr.DataArray:
        xr_data = data.data
        space_dim = "space"

        if space_dim not in xr_data.dims:
            raise ValueError(f"dimension '{space_dim}' not found in data dimensions {xr_data.dims}")

        space_coords = xr_data.coords[space_dim].values

        if self.coord_x not in space_coords:
            raise ValueError(
                f"coordinate '{self.coord_x}' not found in space dimension: {space_coords}"
            )
        if self.coord_y not in space_coords:
            raise ValueError(
                f"coordinate '{self.coord_y}' not found in space dimension: {space_coords}"
            )

        x_series = xr_data.sel({space_dim: self.coord_x}).values
        y_series = xr_data.sel({space_dim: self.coord_y}).values

        return xr.DataArray(_compute_plv(x_series, y_series))


@dataclass
class PhaseLockingValueMatrix(BaseFeature[SignalData]):
    """Compute pairwise phase locking value matrix for multiple coordinates.

    Computes PLV for every pair of coordinates in ``coords``. All coordinates
    must be from the ``space`` dimension.

    Args:
        coords: List of coordinate names to compute pairwise PLV for.

    Returns:
        xarray DataArray with dims ``(coord_i, coord_j)`` and shape
        ``(len(coords), len(coords))``.

    Raises:
        ValueError: If ``coords`` is empty.
        ValueError: If any coordinate is not found in the space dimension.

    Example:
        >>> result = cb.feature.PhaseLockingValueMatrix(coords=[0, 1, 2]).apply(data)
    """

    output_type: ClassVar[type[Data]] = Data

    coords: list[str] | list[int]

    def __call__(self, data: SignalData) -> xr.DataArray:
        xr_data = data.data
        space_dim = "space"

        if space_dim not in xr_data.dims:
            raise ValueError(f"dimension '{space_dim}' not found in data dimensions {xr_data.dims}")

        if not self.coords:
            raise ValueError("coords must have at least one coordinate")

        space_coords = xr_data.coords[space_dim].values

        for c in self.coords:
            if c not in space_coords:
                raise ValueError(f"coordinate '{c}' not found in space dimension: {space_coords}")

        n = len(self.coords)
        result = np.full((n, n), np.nan)

        for i, coord_i in enumerate(self.coords):
            x_series = xr_data.sel({space_dim: coord_i}).values
            for j, coord_j in enumerate(self.coords):
                y_series = xr_data.sel({space_dim: coord_j}).values
                result[i, j] = _compute_plv(x_series, y_series)

        return xr.DataArray(
            result,
            dims=["coord_i", "coord_j"],
            coords={"coord_i": self.coords, "coord_j": self.coords},
        )
