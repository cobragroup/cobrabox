from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
import xarray as xr
from scipy.signal import hilbert as _scipy_hilbert

from .._functional import functional
from ..base_feature import BaseFeature
from ..data import Data, SignalData

_VALID_REPRESENTATIONS = ("analytic", "envelope", "phase", "frequency")


def _analytic_signal_1d(
    signal: np.ndarray,
    representation: Literal["analytic", "envelope", "phase", "frequency"] = "analytic",
    *,
    sampling_rate: float | None = None,
    axis: int = -1,
) -> np.ndarray:
    """Compute an analytic-signal representation along *axis*.

    Used internally by :class:`AnalyticSignal` and by other features that need
    Hilbert-derived quantities (PhaseLockingValue, RecurrenceMatrix) without
    going through the xarray wrapper.

    Args:
        signal: Real-valued ndarray.
        representation: One of "analytic", "envelope", "phase", "frequency".
        sampling_rate: Required when ``representation == "frequency"``.
        axis: Axis along which to compute.

    Returns:
        ndarray with the same shape as ``signal``; dtype is ``complex128`` for
        ``"analytic"`` and ``float64`` otherwise.
    """
    if representation not in _VALID_REPRESENTATIONS:
        raise ValueError(
            f"Invalid representation {representation!r}. "
            f"Must be one of: {', '.join(repr(r) for r in _VALID_REPRESENTATIONS)}."
        )
    analytic = _scipy_hilbert(signal, axis=axis)
    if representation == "analytic":
        return analytic
    if representation == "envelope":
        return np.abs(analytic)
    if representation == "phase":
        return np.angle(analytic)
    if sampling_rate is None:
        raise ValueError("representation='frequency' requires sampling_rate, but it is None.")
    unwrapped = np.unwrap(np.angle(analytic), axis=axis)
    return np.gradient(unwrapped, 1.0 / sampling_rate, axis=axis) / (2 * np.pi)


@dataclass
class AnalyticSignal(BaseFeature[SignalData]):
    """Extract analytic-signal representations along the time axis.

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
        >>> analytic = cb.AnalyticSignal().apply(data)
        >>> env = cb.AnalyticSignal(feature="envelope").apply(data)
        >>> phase = cb.AnalyticSignal(feature="phase").apply(data)
        >>> freq = cb.AnalyticSignal(feature="frequency").apply(data)
    """

    _tags: ClassVar[list[str]] = [
        "hilbert",
        "instantaneous-phase",
        "envelope",
        "instantaneous-frequency",
        "time-frequency",
        "eeg",
        "fmri",
        "io:preserves-time",
    ]

    feature: Literal["analytic", "envelope", "phase", "frequency"] = "analytic"

    def __post_init__(self) -> None:
        if self.feature not in _VALID_REPRESENTATIONS:
            raise ValueError(
                f"Invalid feature {self.feature!r}. "
                f"Must be one of: {', '.join(repr(f) for f in _VALID_REPRESENTATIONS)}."
            )

    def __call__(self, data: SignalData) -> xr.DataArray:
        xr_data = data.data
        time_axis = xr_data.dims.index("time")
        result = _analytic_signal_1d(
            xr_data.values,
            representation=self.feature,
            sampling_rate=data.sampling_rate,
            axis=time_axis,
        )
        return xr.DataArray(result, dims=xr_data.dims, coords=xr_data.coords)


@functional(AnalyticSignal)
def analytic_signal(
    data: SignalData, feature: Literal["analytic", "envelope", "phase", "frequency"] = "analytic"
) -> Data:
    """Extract analytic-signal representations along the time axis.

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
        >>> analytic = cb.analytic_signal(data)
        >>> env = cb.analytic_signal(data, feature="envelope")
        >>> phase = cb.analytic_signal(data, feature="phase")
        >>> freq = cb.analytic_signal(data, feature="frequency")
    """
    return AnalyticSignal(feature=feature).apply(data)
