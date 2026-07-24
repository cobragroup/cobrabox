"""Fractal dimension feature: Higuchi and Katz algorithms.

Higuchi algorithm adapted from MATLAB code by Jesús Monge Álvarez (2014),
itself based on Higuchi (1988).

References:
    Higuchi, T. (1988). Approach to an irregular time series on the basis of
    the fractal theory. Physica D, 31(2), 277-283.

    Katz, M. J. (1988). Fractals and the analysis of waveforms. Computers in
    Biology and Medicine, 18(3), 145-156.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import ClassVar, Literal

import numpy as np
import xarray as xr

from .._functional import functional
from ..base_feature import BaseFeature
from ..data import Data, SignalData

_VALID_METHODS = ("higuchi", "katz")


def _higuchi_1d(signal: np.ndarray, k_max: int) -> float:
    """Compute Higuchi fractal dimension for a single 1-D signal."""
    N = len(signal)
    if N <= k_max:
        raise ValueError(
            f"Signal length ({N}) must be greater than k_max ({k_max}). "
            "Use a longer window or reduce k_max."
        )
    L = np.empty(k_max)
    for k in range(1, k_max + 1):
        lm = np.empty(k)
        for m in range(1, k + 1):
            n_steps = (N - m) // k
            if n_steps == 0:
                lm[m - 1] = 0.0
                continue
            idx = np.arange(m - 1, m - 1 + n_steps * k + 1, k)
            norm = (N - 1) / (n_steps * k)
            lm[m - 1] = np.sum(np.abs(np.diff(signal[idx]))) * norm / k
        L[k - 1] = np.sum(lm) / k
    log_k_inv = np.log(1.0 / np.arange(1, k_max + 1))
    return float(np.polyfit(log_k_inv, np.log(L), 1)[0])


def _katz_1d(signal: np.ndarray) -> float:
    """Compute Katz fractal dimension for a single 1-D signal."""
    N = len(signal)
    x = np.arange(N, dtype=float)
    eu_length = float(np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(signal) ** 2)))
    max_dist = float(np.max(np.sqrt((x[1:] - x[0]) ** 2 + (signal[1:] - signal[0]) ** 2)))
    n = N - 1
    return float(np.log10(n) / (np.log10(n) + np.log10(max_dist / eu_length)))


@dataclass
class FractalDimension(BaseFeature[SignalData]):
    """Compute fractal dimension over the time dimension.

    Selects the algorithm via the ``method`` parameter:

    * ``"higuchi"`` — Higuchi Fractal Dimension. Constructs k sub-series for
      each interval k = 1…k_max, computes their normalised average curve
      length L(k), then estimates the fractal dimension as the slope of
      log(L(k)) vs log(1/k) via least-squares regression. Tunable via
      ``k_max``. Values close to 1 indicate a smooth signal; values close
      to 2 indicate a highly irregular signal.

    * ``"katz"`` — Katz Fractal Dimension. Models the signal as a 2-D curve
      and estimates fractal dimension from the total Euclidean path length,
      step count, and maximum planar distance from the first sample.
      Parameter-free, O(N).

    Args:
        method: Which fractal dimension algorithm to use. One of ``"higuchi"``
            (default) or ``"katz"``.
        k_max: Maximum interval for the Higuchi algorithm. Must be >= 2.
            Default is 10. **Ignored when ``method="katz"``** (a warning is
            issued when a non-default value is passed but unused).

    Returns:
        xarray DataArray with the ``time`` dimension removed. Shape is
        ``(space,)`` for standard input. Values are dimensionless floats
        typically in [1, 2] for Higuchi and >= 1 for Katz.

    Raises:
        ValueError: If ``method`` is not one of the valid options, or if
            ``k_max < 2`` when using the Higuchi method.

    References:
        Higuchi, T. (1988). Approach to an irregular time series on the
        basis of the fractal theory. Physica D, 31(2), 277-283.

        Katz, M. J. (1988). Fractals and the analysis of waveforms.
        Computers in Biology and Medicine, 18(3), 145-156.

    Example:
        >>> result = cb.FractalDimension().apply(data)  # Higuchi, k_max=10
        >>> result = cb.FractalDimension(method="higuchi", k_max=20).apply(data)
        >>> result = cb.FractalDimension(method="katz").apply(data)
    """

    _tags: ClassVar[list[str]] = [
        "fractal",
        "self-similarity",
        "higuchi",
        "katz",
        "box-counting",
        "aging",
        "anesthesia",
        "dementia",
        "eeg",
        "io:scalar-per-channel",
    ]

    method: Literal["higuchi", "katz"] = "higuchi"
    k_max: int = field(default=10)

    output_type: ClassVar[type[Data]] = Data  # pyright: ignore[reportIncompatibleVariableOverride]

    def __post_init__(self) -> None:
        if self.method not in _VALID_METHODS:
            raise ValueError(
                f"Invalid method {self.method!r}. "
                f"Must be one of: {', '.join(repr(m) for m in _VALID_METHODS)}."
            )
        if self.method == "higuchi" and self.k_max < 2:
            raise ValueError(f"k_max must be >= 2, got {self.k_max}")
        if self.method == "katz" and self.k_max != 10:
            warnings.warn(
                f"k_max={self.k_max} is ignored when method='katz' (Katz fractal "
                "dimension is parameter-free).",
                UserWarning,
                stacklevel=2,
            )

    def __call__(self, data: SignalData) -> xr.DataArray:
        if self.method == "higuchi":
            return xr.apply_ufunc(
                lambda sig: _higuchi_1d(sig, self.k_max),
                data.data,
                input_core_dims=[["time"]],
                vectorize=True,
            )
        return xr.apply_ufunc(_katz_1d, data.data, input_core_dims=[["time"]], vectorize=True)


@functional(FractalDimension)
def fractal_dimension(
    data: SignalData, method: Literal["higuchi", "katz"] = "higuchi", k_max: int = 10
) -> Data:
    """Compute fractal dimension over the time dimension.

    Selects the algorithm via the ``method`` parameter:

    * ``"higuchi"`` — Higuchi Fractal Dimension. Constructs k sub-series for
      each interval k = 1…k_max, computes their normalised average curve
      length L(k), then estimates the fractal dimension as the slope of
      log(L(k)) vs log(1/k) via least-squares regression. Tunable via
      ``k_max``. Values close to 1 indicate a smooth signal; values close
      to 2 indicate a highly irregular signal.

    * ``"katz"`` — Katz Fractal Dimension. Models the signal as a 2-D curve
      and estimates fractal dimension from the total Euclidean path length,
      step count, and maximum planar distance from the first sample.
      Parameter-free, O(N).

    Args:
        method: Which fractal dimension algorithm to use. One of ``"higuchi"``
            (default) or ``"katz"``.
        k_max: Maximum interval for the Higuchi algorithm. Must be >= 2.
            Default is 10. **Ignored when ``method="katz"``** (a warning is
            issued when a non-default value is passed but unused).

    Returns:
        xarray DataArray with the ``time`` dimension removed. Shape is
        ``(space,)`` for standard input. Values are dimensionless floats
        typically in [1, 2] for Higuchi and >= 1 for Katz.

    Raises:
        ValueError: If ``method`` is not one of the valid options, or if
            ``k_max < 2`` when using the Higuchi method.

    References:
        Higuchi, T. (1988). Approach to an irregular time series on the
        basis of the fractal theory. Physica D, 31(2), 277-283.

        Katz, M. J. (1988). Fractals and the analysis of waveforms.
        Computers in Biology and Medicine, 18(3), 145-156.

    Example:
        >>> result = cb.fractal_dimension(data)  # Higuchi, k_max=10
        >>> result = cb.fractal_dimension(data, method="higuchi", k_max=20)
        >>> result = cb.fractal_dimension(data, method="katz")
    """
    return FractalDimension(method=method, k_max=k_max).apply(data)
