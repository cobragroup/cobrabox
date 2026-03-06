from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import xarray as xr

from ..base_feature import BaseFeature
from ..data import Data


@dataclass
class AmplitudeEntropy(BaseFeature[Data]):
    """
    Compute amplitude entropy from time-series data using histogram-based probability estimation.

    Amplitude entropy quantifies the randomness or unpredictability of the amplitude distribution
    in the data. It uses a histogram-based approach where the data is binned according to the
    specified band_width, and Shannon entropy is computed from the resulting probability
    distribution.

    The entropy for each row (time point) is calculated as H = -sum(p_i * log2(p_i)), where p_i
    is the probability of the i-th bin. The final result is the mean entropy across all time
    points, returned as a scalar value.

    Args:
        band_width: The width of histogram bins for discretizing the data. Must be positive.

    Returns:
        A 0-dimensional xarray.DataArray containing the mean amplitude entropy as a scalar value.

    Example:
        >>> import cobrabox as cb
        >>> import numpy as np
        >>> data = cb.Data.from_numpy(np.random.randn(50, 10), dims=["time", "space"])
        >>> result = cb.feature.AmplitudeEntropy(band_width=0.5).apply(data)
        >>> float(result.to_numpy())  # Scalar entropy value

    """

    output_type: ClassVar[type[Data] | None] = Data

    band_width: float

    def __post_init__(self) -> None:
        if self.band_width <= 0:
            raise ValueError("band_width must be positive")

    def __call__(self, data: Data) -> xr.DataArray:
        raw = data.to_numpy()
        a = np.asarray(raw)

        # Validate that data has at least 2 dimensions for row indexing
        if a.ndim < 2:
            raise ValueError("Input data must have at least 2 dimensions")

        # MATLAB size(data_window, 1) is the number of rows.
        # NumPy equivalent is a.shape[0].
        n_timepoints = a.shape[0]  # rows interpreted as "time points"

        # MATLAB: ta = zeros(T, 1) -> a column vector (T x 1)
        # NumPy: np.zeros(T) -> 1D array of shape (T,)
        # This is usually fine unless later code expects (T, 1).
        ta = np.zeros(n_timepoints, dtype=float)

        # MATLAB variables are in scope; in a class, use self.band_width.
        bw = float(self.band_width)

        for i in range(n_timepoints):
            # MATLAB uses 1-based indexing: data_window(i,:)
            # Python uses 0-based indexing: a[i, :]
            dataww = a[i, :]

            mn = dataww.min()
            mx = dataww.max()

            # MATLAB edges:
            #   min(dataww) : band_width : max(dataww) + band_width
            #
            # Python/NumPy note:
            #   np.arange(start, stop, step) EXCLUDES 'stop' in ideal math,
            #   and floating point rounding can cause the last edge to be missed.
            #
            # Adding a tiny epsilon helps match MATLAB's intention of including
            # the final edge at (max + band_width).
            edges = np.arange(mn, mx + bw + 1e-12, bw)

            # MATLAB: histcounts(..., 'Normalization','probability')
            # returns probabilities directly.
            # NumPy: np.histogram returns raw COUNTS, so we normalize ourselves.
            counts, _ = np.histogram(dataww, bins=edges)

            total = counts.sum()
            if total == 0:
                # Defensive guard; usually not hit unless the row is empty or weird.
                ta[i] = 0.0
                continue

            probs = counts / total  # equivalent to MATLAB 'Normalization','probability'

            # MATLAB: e2 = N2(N2 > 0) actually REMOVES zeros.
            #
            # In Python, doing probs * (probs > 0) would keep zeros in-place.
            # That can cause issues if log2 is ever applied before masking.
            # Best match is to FILTER:
            e2 = probs[probs > 0]

            # MATLAB: -sum(e2 .* log2(e2))
            # NumPy equivalent uses elementwise multiply and np.log2
            ta[i] = -np.sum(e2 * np.log2(e2))

        tabar = float(np.mean(ta))

        return xr.DataArray(tabar)
