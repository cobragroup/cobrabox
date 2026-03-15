"""Fractal dimension features: Higuchi and Katz algorithms.

Higuchi algorithm adapted from MATLAB code by Jesús Monge Álvarez (2014),
itself based on Higuchi (1988).

References:
    Higuchi, T. (1988). Approach to an irregular time series on the basis of
    the fractal theory. Physica D, 31(2), 277-283.

    Katz, M. J. (1988). Fractals and the analysis of waveforms. Computers in
    Biology and Medicine, 18(3), 145-156.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import xarray as xr

from ...base_feature import BaseFeature
from ...data import Data, SignalData


@dataclass
class FractalDimHiguchi(BaseFeature[SignalData]):
    """Compute Higuchi Fractal Dimension (HFD) over the time dimension.

    Constructs k sub-series from the signal for each interval k = 1…k_max,
    computes their normalised average curve length L(k), then estimates the
    fractal dimension as the slope of log(L(k)) vs log(1/k) via least-squares
    regression.

    Values close to 1 indicate a smooth, low-complexity signal; values close
    to 2 indicate a highly irregular signal. Typical EEG values lie between
    1.0 and 2.0.

    Args:
        k_max: Maximum interval (number of sub-series). Must be >= 2.
            Default is 10, following common EEG literature recommendations.

    Returns:
        xarray DataArray with the ``time`` dimension removed. Shape is
        ``(space,)`` for standard input. Values are dimensionless floats
        typically in [1, 2].

    References:
        Higuchi, T. (1988). Approach to an irregular time series on the
        basis of the fractal theory. Physica D, 31(2), 277-283.

    Example:
        >>> result = cb.feature.FractalDimHiguchi().apply(data)
        >>> result = cb.feature.FractalDimHiguchi(k_max=20).apply(data)
    """

    k_max: int = field(default=10)

    output_type: ClassVar[type[Data]] = Data  # pyright: ignore[reportIncompatibleVariableOverride]

    def __post_init__(self) -> None:
        if self.k_max < 2:
            raise ValueError(f"k_max must be >= 2, got {self.k_max}")

    def __call__(self, data: SignalData) -> xr.DataArray:
        return xr.apply_ufunc(
            lambda sig: self._higuchi_1d(sig, self.k_max),
            data.data,
            input_core_dims=[["time"]],
            vectorize=True,
        )

    @staticmethod
    def _higuchi_1d(signal: np.ndarray, k_max: int) -> float:
        """Compute HFD for a single 1-D signal."""
        N = len(signal)
        if N <= k_max:
            raise ValueError(
                f"Signal length ({N}) must be greater than k_max ({k_max}). "
                "Use a longer window or reduce k_max."
            )
        L = np.empty(k_max)
        for k in range(1, k_max + 1):
            lm = np.empty(k)
            for m in range(1, k + 1):  # 1-indexed m, as in the original
                n_steps = (N - m) // k
                if n_steps == 0:
                    lm[m - 1] = 0.0
                    continue
                # 0-indexed sub-series: m-1, m-1+k, ..., m-1+n_steps*k
                idx = np.arange(m - 1, m - 1 + n_steps * k + 1, k)
                norm = (N - 1) / (n_steps * k)
                lm[m - 1] = np.sum(np.abs(np.diff(signal[idx]))) * norm / k
            L[k - 1] = np.sum(lm) / k  # equivalent to mean since len(lm) == k
        log_k_inv = np.log(1.0 / np.arange(1, k_max + 1))
        return float(np.polyfit(log_k_inv, np.log(L), 1)[0])


@dataclass
class FractalDimKatz(BaseFeature[SignalData]):
    """Compute Katz Fractal Dimension (KFD) over the time dimension.

    Models the signal as a 2-D curve (sample index vs amplitude) and estimates
    fractal dimension from the total Euclidean path length (L), the number of
    steps (n = N - 1), and the maximum planar distance from the first sample to
    any other sample (d):

        KFD = log10(n) / (log10(n) + log10(d / L))

    Values close to 1 indicate a smooth, regular signal; values above 1 indicate
    greater irregularity. Unlike Higuchi FD, KFD has no tuning parameters and is
    O(N), making it fast even for long signals.

    Args:
        None

    Returns:
        xarray DataArray with the ``time`` dimension removed. Shape is
        ``(space,)`` for standard input. Values are dimensionless floats
        typically >= 1.

    References:
        Katz, M. J. (1988). Fractals and the analysis of waveforms.
        Computers in Biology and Medicine, 18(3), 145-156.

    Example:
        >>> result = cb.feature.FractalDimKatz().apply(data)
    """

    output_type: ClassVar[type[Data]] = Data  # pyright: ignore[reportIncompatibleVariableOverride]

    def __call__(self, data: SignalData) -> xr.DataArray:
        return xr.apply_ufunc(self._katz_1d, data.data, input_core_dims=[["time"]], vectorize=True)

    @staticmethod
    def _katz_1d(signal: np.ndarray) -> float:
        """Compute KFD for a single 1-D signal.

        Adapted from MATLAB code (Sara-Kamali / Jesús Monge Álvarez).
        Sample indices are used as the time (x) axis.
        """
        N = len(signal)
        x = np.arange(N, dtype=float)  # time axis: sample indices 0, 1, ..., N-1

        # Total Euclidean length of the curve
        eu_length = float(np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(signal) ** 2)))

        # Max Euclidean distance from the first point to any other point
        max_dist = float(np.max(np.sqrt((x[1:] - x[0]) ** 2 + (signal[1:] - signal[0]) ** 2)))

        n = N - 1
        return float(np.log10(n) / (np.log10(n) + np.log10(max_dist / eu_length)))
