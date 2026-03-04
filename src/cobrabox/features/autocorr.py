from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from ..base_feature import BaseFeature
from ..data import Data


@dataclass
class Autocorr(BaseFeature[Data]):
    """Compute normalized autocorrelation at a given lag along a required dimension.

    MATLAB equivalent from mapping_seizure_dynamics/calc_feat:
        acf = autocorr(d,'NumLags',round(fs/1000*5));
        autocorrel = acf(end);

    Specify either ``lag_steps`` or ``lag_ms``, not both.
    If neither is specified, the default lag of 5 ms is used.

    Args:
        dim: Dimension to compute autocorrelation along.
        fs: Sampling frequency in Hz. Must be positive.
        lag_steps: Lag in number of samples. Mutually exclusive with ``lag_ms``.
        lag_ms: Lag in milliseconds. Mutually exclusive with ``lag_steps``.

    Returns:
        xarray DataArray with the ``dim`` dimension removed. Shape is the input shape
        minus the size of ``dim``; each element is the normalized autocorrelation at the
        computed lag for that position in the remaining dimensions.

    Example:
        >>> result = cb.feature.Autocorr(dim="time", fs=1000.0, lag_steps=5).apply(data)
    """

    dim: str
    fs: float
    lag_steps: int | None = None
    lag_ms: float | None = None

    def __post_init__(self) -> None:
        if self.lag_steps is not None and self.lag_ms is not None:
            raise ValueError("Specify either 'lag_steps' or 'lag_ms', not both.")
        if self.fs <= 0:
            raise ValueError(f"fs must be positive, got {self.fs}")

    @staticmethod
    def _acf_numpy(x: np.ndarray, lag: int) -> float:
        """Compute normalized autocorrelation using numpy."""
        x = x.astype(float)

        if np.all(np.isnan(x)):
            return np.nan

        x = x - np.nanmean(x)

        # replace NaNs with zero for correlation
        x = np.nan_to_num(x)

        corr = np.correlate(x, x, mode="full")
        corr = corr[corr.size // 2 :]  # keep non-negative lags

        if corr[0] == 0:
            return np.nan

        corr = corr / corr[0]
        return float(corr[lag])

    def __call__(self, data: Data) -> xr.DataArray:
        xr_data = data.data

        if self.dim not in xr_data.dims:
            raise ValueError(f"dim '{self.dim}' not found in data dimensions {xr_data.dims}")

        # determine lag
        if self.lag_steps is not None:
            lag = int(self.lag_steps)
        else:
            lag_ms_value: float = 5.0 if self.lag_ms is None else float(self.lag_ms)
            lag = round(self.fs * lag_ms_value / 1000.0)

        # Validate lag against length along the requested dimension
        n = xr_data.sizes[self.dim]
        if lag <= 0 or lag >= n:
            raise ValueError(f"lag must be between 1 and {n - 1} along dim '{self.dim}'")

        def _wrapper(x: np.ndarray) -> float:
            return self._acf_numpy(x, lag)

        return xr.apply_ufunc(_wrapper, xr_data, input_core_dims=[[self.dim]], vectorize=True)
