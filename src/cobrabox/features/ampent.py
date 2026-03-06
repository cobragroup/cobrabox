from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import xarray as xr

from ..base_feature import BaseFeature
from ..data import Data


@dataclass
class Ampent(BaseFeature[Data]):
    output_type: ClassVar[type[Data]] = Data

    band_width: float

    def __call__(self, data: Data) -> xr.DataArray:
        raw = data.to_numpy()
        a = np.asarray(raw)

        # MATLAB size(data_window, 1) is the number of rows.
        # NumPy equivalent is a.shape[0].
        T = a.shape[0]  # rows interpreted as "time points"

        # MATLAB: ta = zeros(T, 1) -> a column vector (T x 1)
        # NumPy: np.zeros(T) -> 1D array of shape (T,)
        # This is usually fine unless later code expects (T, 1).
        ta = np.zeros(T, dtype=float)

        # MATLAB variables are in scope; in a class, use self.band_width.
        bw = float(self.band_width)

        for i in range(T):
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

        return xr.DataArray([[tabar]], dims=["time", "space"], coords={"time": [0], "space": [0]})
