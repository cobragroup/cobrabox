from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import xarray as xr

from cobrabox.base_feature import BaseFeature
from cobrabox.data import Data, SignalData


def _granger_log_ratio(x: np.ndarray, y: np.ndarray, lag: int) -> tuple[float, float, float, float]:
    """Compute Granger causality using log-ratio of prediction error variances.

    Measures whether y Granger-causes x by comparing prediction error variances
    from two VAR models:
    - Restricted: predict x(t) using past x values only
    - Unrestricted: predict x(t) using past x and y values

    Args:
        x: Target series of shape (n_samples,)
        y: Source series of shape (n_samples,)
        lag: Lag order for VAR models

    Returns:
        (granger_causality, ssr_restricted, ssr_unrestricted, variance_ratio) tuple
        where granger_causality = log(ssr_restricted / ssr_unrestricted)

    Formula:
        GC = log(SSR_r / SSR_u) = log(sigma_r² / sigma_u²)

    Where:
        SSR_r = sum of squared residuals (restricted model)
        SSR_u = sum of squared residuals (unrestricted model)
        sigma_r², sigma_u² = prediction error variances

    Interpretation:
        - GC > 0: y Granger-causes x (including y's past improves prediction)
        - GC = 0: no Granger causality
        - GC < 0: adding y's past degrades prediction (rare in practice)
        - Larger values indicate stronger causality
    """
    n = len(x)

    X_restricted = np.column_stack([x[lag - i : n - i] for i in range(1, lag + 1)])

    X_unrestricted = np.column_stack(
        [x[lag - i : n - i] for i in range(1, lag + 1)]
        + [y[lag - i : n - i] for i in range(1, lag + 1)]
    )

    y_target = x[lag:]

    coeffs_r, _, _, _ = np.linalg.lstsq(X_restricted, y_target, rcond=None)
    residuals_r = y_target - X_restricted @ coeffs_r
    ssr_r = np.sum(residuals_r**2)

    coeffs_u, _, _, _ = np.linalg.lstsq(X_unrestricted, y_target, rcond=None)
    residuals_u = y_target - X_unrestricted @ coeffs_u
    ssr_u = np.sum(residuals_u**2)

    variance_ratio = ssr_r / ssr_u
    granger_causality = np.log(variance_ratio)

    return float(granger_causality), float(ssr_r), float(ssr_u), float(variance_ratio)


@dataclass
class GrangerCausality(BaseFeature[SignalData]):
    """Test whether one coordinate Granger-causes another.

    Uses Granger causality measured as the log-ratio of prediction error
    variances to determine if past values of coord_y contain predictive
    information for coord_x beyond what is contained in past values
    of coord_x alone.

    Args:
        coord_x: Target variable coordinate (from space dimension)
        coord_y: Source/causal variable coordinate (from space dimension)
        lag: Specific lag order to test. If provided, only tests this lag.
        maxlag: Maximum lag to test. If lag is not provided, tests all lags
                from 1 to maxlag. Default: 1.

    Note:
        - If both lag and maxlag are provided, only lag is used
        - Granger causality = log(SSR_restricted / SSR_unrestricted)
        - Positive values indicate y Granger-causes x (including y's past improves prediction)
        - Larger values indicate stronger causality
        - If neither lag nor maxlag provided, defaults to maxlag=1
        - Choose lag based on data sampling rate and expected causal delay timescales

    Returns:
        If lag is specified: scalar xarray.DataArray with Granger causality value
        If maxlag is specified: xarray.DataArray with dims (lag_index,)
                                  containing results for lags 1 to maxlag

    Example:
        >>> # Test lag 2 only
        >>> result = cb.feature.GrangerCausality(
        ...     coord_x=0, coord_y=1, lag=2
        ... ).apply(data)
        >>> # Test range of lags explicitly
        >>> results_all = cb.feature.GrangerCausality(
        ...     coord_x=0, coord_y=1, maxlag=4
        ... ).apply(data)
    """

    output_type: ClassVar[type[Data]] = Data

    coord_x: str | int | None = None
    coord_y: str | int | None = None
    lag: int | None = None
    maxlag: int = 1

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.maxlag < 1:
            raise ValueError(f"maxlag must be >= 1, got {self.maxlag}")
        if self.lag is not None and self.lag < 1:
            raise ValueError(f"lag must be >= 1, got {self.lag}")

    def __call__(self, data: SignalData) -> xr.DataArray:
        """Apply Granger causality test using log-ratio method."""
        xr_data = data.data
        xr_data = xr_data.transpose(..., "time")
        x_data = xr_data.sel(space=self.coord_x).values.flatten()
        y_data = xr_data.sel(space=self.coord_y).values.flatten()

        if self.lag is not None:
            lags_to_test = [self.lag]
        else:
            lags_to_test = list(range(1, self.maxlag + 1))

        results = []
        for lag in lags_to_test:
            gc, _, _, _ = _granger_log_ratio(x_data, y_data, lag)
            results.append(gc)

        if len(lags_to_test) == 1:
            return xr.DataArray(results[0])
        return xr.DataArray(results, dims=("lag_index",), coords={"lag_index": lags_to_test})


_is_cobrabox_feature = True
GrangerCausality._is_cobrabox_feature = True


@dataclass
class GrangerCausalityMatrix(BaseFeature[SignalData]):
    """Compute Granger causality matrix for all coordinate pairs.

    Tests whether each coordinate Granger-causes each other coordinate
    for a specified lag or range of lags using the log-ratio of prediction
    error variances. The standard convention matches
    partial_correlation_matrix: row i represents the target/outcome,
    column j represents the source/cause.

    Args:
        coords: List of coordinates to test. If None, uses all space
                coordinates.
        lag: Specific lag to test. If provided, only tests this lag.
        maxlag: Maximum lag to test. If lag is not provided, tests all
                lags from 1 to maxlag. Default: 1.

    Returns:
        xarray.DataArray with dims (coord_i, coord_j) where:
        - result[i,j] = causality from j to i (j → i)
        - result[i,i] = NaN (self-causality undefined)
        - Positive values indicate causality; larger values are stronger

        If testing multiple lags: dims (coord_i, coord_j, lag_index)

    Example:
        >>> # Test lag 2 for all pairs
        >>> result = cb.feature.GrangerCausalityMatrix(
        ...     coords=[0, 1, 2], lag=2
        ... ).apply(data)
        >>> # result[0,1] > 0 means channel 1 causes channel 0
    """

    output_type: ClassVar[type[Data]] = Data

    coords: list[str] | list[int] | None = None
    lag: int | None = None
    maxlag: int = 1

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.maxlag < 1:
            raise ValueError(f"maxlag must be >= 1, got {self.maxlag}")
        if self.lag is not None and self.lag < 1:
            raise ValueError(f"lag must be >= 1, got {self.lag}")

    def __call__(self, data: SignalData) -> xr.DataArray:
        """Apply Granger causality test to all coordinate pairs using log-ratio method."""
        xr_data = data.data
        xr_data = xr_data.transpose(..., "time")

        if self.coords is None:
            coords_list = list(xr_data.space.values)
        else:
            coords_list = self.coords

        n_coords = len(coords_list)

        if self.lag is not None:
            lags_to_test = [self.lag]
        else:
            lags_to_test = list(range(1, self.maxlag + 1))

        if len(lags_to_test) == 1:
            result_matrix = np.full((n_coords, n_coords), np.nan, dtype=float)
        else:
            result_matrix = np.full((n_coords, n_coords, len(lags_to_test)), np.nan, dtype=float)

        for i, coord_i in enumerate(coords_list):
            for j, coord_j in enumerate(coords_list):
                if i == j:
                    continue

                x_data = xr_data.sel(space=coord_i).values.flatten()
                y_data = xr_data.sel(space=coord_j).values.flatten()

                for lag_idx, lag in enumerate(lags_to_test):
                    gc, _, _, _ = _granger_log_ratio(x_data, y_data, lag)

                    if len(lags_to_test) == 1:
                        result_matrix[i, j] = gc
                    else:
                        result_matrix[i, j, lag_idx] = gc

        if len(lags_to_test) == 1:
            return xr.DataArray(
                result_matrix,
                dims=("coord_i", "coord_j"),
                coords={"coord_i": coords_list, "coord_j": coords_list},
            )
        return xr.DataArray(
            result_matrix,
            dims=("coord_i", "coord_j", "lag_index"),
            coords={"coord_i": coords_list, "coord_j": coords_list, "lag_index": lags_to_test},
        )


GrangerCausalityMatrix._is_cobrabox_feature = True
