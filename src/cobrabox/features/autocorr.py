from __future__ import annotations

import numpy as np
import xarray as xr

from ..data import Data
from ..function_wrapper import feature


@feature
def autocorrelation(
    data: Data, dim: str, lag_steps: int | None = None, lag_ms: float | None = None
) -> xr.DataArray:
    """Compute normalized autocorrelation at a given lag along a required dimension.

    MATLAB equivalent from mapping_seizure_dynamics/calc_feat:
        acf = autocorr(d,'NumLags',round(fs/1000*5));
        autocorrel = acf(end);

    Notes
    -----
    Specify either `lag_steps` or `lag_ms`, not both.
    If neither is specified, the default lag of 5 ms is used.
    """
    xr_data = data.data

    if dim not in xr_data.dims:
        raise ValueError(f"dim '{dim}' not found in data dimensions {xr_data.dims}")

    if lag_steps is not None and lag_ms is not None:
        raise ValueError("Specify either 'lag_steps' or 'lag_ms', not both.")

    fs_value = data.sampling_rate
    if fs_value is None:
        raise ValueError("sampling_rate must be defined for autocorrelation.")

    fs = float(fs_value)

    # determine lag
    if lag_steps is not None:
        lag = int(lag_steps)
    else:
        lag_ms_value: float = 5.0 if lag_ms is None else float(lag_ms)
        lag = round(fs * lag_ms_value / 1000.0)

    # Validate lag against length along the requested dimension
    n = xr_data.sizes[dim]
    if lag <= 0 or lag >= n:
        raise ValueError(f"lag must be between 1 and {n - 1} along dim '{dim}'")

    def _acf_numpy(x: np.ndarray) -> float:
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

    return xr.apply_ufunc(_acf_numpy, xr_data, input_core_dims=[[dim]], vectorize=True)
