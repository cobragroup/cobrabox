from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import xarray as xr

from ..base_feature import BaseFeature
from ..data import Data, SignalData
from ._mvar import granger_log_ratio


@dataclass
class GrangerCausality(BaseFeature[SignalData]):
    """Compute the Granger causality matrix across channels.

    For each ordered pair ``(i, j)``, measures whether channel ``j``
    Granger-causes channel ``i`` via the log-ratio of prediction-error
    variances of two VAR models (restricted: past of ``i`` only; unrestricted:
    past of both ``i`` and ``j``). Self-causality is undefined and reported
    as NaN on the diagonal.

    Args:
        coords: Channels to include. ``None`` (default) computes the full
            ``(K, K)`` matrix across all space coordinates. Pass a list of
            coordinate names to restrict the output (each pair is an
            independent OLS solve, so this is a real compute saving).
        lag: Specific lag order to test. When set, returns a 2-D matrix.
        maxlag: Maximum lag to test when ``lag`` is ``None``. The output gains
            a ``lag_index`` dimension when ``maxlag > 1``. Default: 1.

    Returns:
        xarray DataArray with dims ``(space_to, space_from)`` (or
        ``(space_to, space_from, lag_index)`` when multiple lags are tested).
        ``result.sel(space_to=A, space_from=B)`` is the causal influence
        ``B → A``. Diagonal entries are NaN.

    Raises:
        ValueError: If ``maxlag < 1``, ``lag < 1`` when provided, or ``coords``
            is an empty list.

    Example:
        >>> # Full matrix at lag 2
        >>> result = cb.feature.GrangerCausality(lag=2).apply(data)
        >>> # Restricted to two channels at the default lag
        >>> result = cb.feature.GrangerCausality(coords=["F3", "F4"]).apply(data)
        >>> # Range of lags
        >>> result = cb.feature.GrangerCausality(maxlag=4).apply(data)
    """

    _tags: ClassVar[list[str]] = [
        "mvar",
        "var",
        "directed",
        "prediction-error",
        "epilepsy",
        "seizure-propagation",
        "resting-state",
        "eeg",
        "io:matrix",
        "req:multichannel",
        "req:sampling-rate",
    ]

    output_type: ClassVar[type[Data]] = Data

    coords: list[str] | list[int] | None = None
    lag: int | None = None
    maxlag: int = 1

    def __post_init__(self) -> None:
        if self.maxlag < 1:
            raise ValueError(f"maxlag must be >= 1, got {self.maxlag}")
        if self.lag is not None and self.lag < 1:
            raise ValueError(f"lag must be >= 1, got {self.lag}")
        if self.coords is not None and len(self.coords) == 0:
            raise ValueError("coords cannot be an empty list")

    def __call__(self, data: SignalData) -> xr.DataArray:
        xr_data = data.data.transpose(..., "time")

        if self.coords is None:
            coords_list = list(xr_data.coords["space"].values)
        else:
            coords_list = list(self.coords)
            space_coords = set(xr_data.coords["space"].values.tolist())
            for c in coords_list:
                if c not in space_coords:
                    raise ValueError(f"coordinate {c!r} not found in space dimension")

        n = len(coords_list)
        lags_to_test = [self.lag] if self.lag is not None else list(range(1, self.maxlag + 1))

        if len(lags_to_test) == 1:
            result = np.full((n, n), np.nan)
        else:
            result = np.full((n, n, len(lags_to_test)), np.nan)

        for i, ci in enumerate(coords_list):
            x = xr_data.sel(space=ci).values.flatten()
            for j, cj in enumerate(coords_list):
                if i == j:
                    continue
                y = xr_data.sel(space=cj).values.flatten()
                for k, lag in enumerate(lags_to_test):
                    gc, _, _, _ = granger_log_ratio(x, y, lag)
                    if len(lags_to_test) == 1:
                        result[i, j] = gc
                    else:
                        result[i, j, k] = gc

        if len(lags_to_test) == 1:
            return xr.DataArray(
                result,
                dims=("space_to", "space_from"),
                coords={"space_to": coords_list, "space_from": coords_list},
            )
        return xr.DataArray(
            result,
            dims=("space_to", "space_from", "lag_index"),
            coords={"space_to": coords_list, "space_from": coords_list, "lag_index": lags_to_test},
        )
