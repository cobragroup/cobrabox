from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import xarray as xr
from scipy.signal import hilbert as _scipy_hilbert

from ..base_feature import BaseFeature
from ..data import SignalData

_VALID_FEATURES = ("analytic", "envelope", "phase", "frequency")


@dataclass
class Hilbert(BaseFeature[SignalData]):
    """Apply the Hilbert transform to extract analytic signal representations.

    Computes the analytic signal via :func:`scipy.signal.hilbert` along the
    ``time`` axis, then derives the requested representation. The output always
    has the same shape and dimensions as the input — the ``time`` dimension is
    preserved.

    Args:
        feature: Which representation to extract. One of:

            * ``'analytic'``  — the complex analytic signal as a ``complex128``
              DataArray with the same dims and coords as the input.
            * ``'envelope'``  — amplitude envelope (``|analytic|``), always
              non-negative.
            * ``'phase'``     — instantaneous phase in radians (``∈ [-pi, pi]``).
            * ``'frequency'`` — instantaneous frequency in Hz, computed as the
              time-derivative of the unwrapped phase divided by 2π. Requires
              ``data.sampling_rate`` to be set.

    Returns:
        xarray DataArray with the same dims and coords as the input. Dtype is
        ``complex128`` for ``'analytic'``, ``float64`` for all other modes.

    Raises:
        ValueError: If ``feature`` is not one of the four valid options.
        ValueError: If ``feature='frequency'`` but ``data.sampling_rate`` is
            ``None``.

    Example:
        >>> analytic = cb.feature.Hilbert().apply(data)  # default: analytic signal
        >>> env = cb.feature.Hilbert(feature="envelope").apply(data)
        >>> phase = cb.feature.Hilbert(feature="phase").apply(data)
        >>> freq = cb.feature.Hilbert(feature="frequency").apply(data)
    """

    feature: Literal["analytic", "envelope", "phase", "frequency"] = "analytic"

    def __post_init__(self) -> None:
        if self.feature not in _VALID_FEATURES:
            raise ValueError(
                f"Invalid feature {self.feature!r}. "
                f"Must be one of: {', '.join(repr(f) for f in _VALID_FEATURES)}."
            )

    def __call__(self, data: SignalData) -> xr.DataArray:
        xr_data = data.data
        time_axis = xr_data.dims.index("time")
        values = xr_data.values

        analytic = _scipy_hilbert(values, axis=time_axis)

        if self.feature == "analytic":
            return xr.DataArray(analytic, dims=xr_data.dims, coords=xr_data.coords)
        if self.feature == "envelope":
            result = np.abs(analytic)
        elif self.feature == "phase":
            result = np.angle(analytic)
        else:  # "frequency"
            if data.sampling_rate is None:
                raise ValueError(
                    "feature='frequency' requires data.sampling_rate to be set, but it is None."
                )
            unwrapped = np.unwrap(np.angle(analytic), axis=time_axis)
            result = np.gradient(unwrapped, 1.0 / data.sampling_rate, axis=time_axis) / (2 * np.pi)

        return xr.DataArray(result, dims=xr_data.dims, coords=xr_data.coords)
