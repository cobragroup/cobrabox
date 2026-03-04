from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from ..base_feature import BaseFeature
from ..data import Data


@dataclass
class Coherence(BaseFeature):
    """Compute magnitude-squared coherence for all pairwise channel combinations.

    Uses Welch's method (50 % overlap, Hann window) to estimate the
    magnitude-squared coherence between each unique pair of spatial channels,
    then averages over frequency bins to produce a single scalar per pair.
    Because coherence is symmetric the result is a symmetric NxN matrix;
    the diagonal is set to NaN (self-coherence is uninformative).

    Extra dimensions beyond 'space' and 'time' (e.g. ``window_index``) are
    preserved — coherence is computed along the 'time' axis for every
    combination of those extra dimensions.

    Args:
        nperseg: Samples per FFT segment. Defaults to ``min(256, n_time)``.
            Must be >= 2 and <= n_time.

    Example:
        >>> data = cb.dataset("dummy_random")[0]
        >>> coh = cb.feature.Coherence().apply(data)
        >>> coh.data.dims
        ('space', 'space_to')
    """

    nperseg: int | None = field(default=None)

    def _mean_squared_coherence(self, x: np.ndarray, y: np.ndarray, nperseg: int) -> np.ndarray:
        """Compute frequency-averaged magnitude-squared coherence using Welch's method.

        Vectorised over leading dimensions: x and y must share shape (..., n_time)
        and the returned array has shape (...,) — one scalar per position.

        Args:
            x: First signal(s), shape (..., n_time).
            y: Second signal(s), shape (..., n_time).
            nperseg: Samples per FFT segment (50 % overlap, Hann window).

        Returns:
            Mean coherence in [0, 1], shape (...,).
        """
        step = nperseg - nperseg // 2
        n = x.shape[-1]
        win = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(nperseg) / (nperseg - 1)))

        extra = x.shape[:-1]
        n_freqs = nperseg // 2 + 1
        pxx = np.zeros((*extra, n_freqs))
        pyy = np.zeros((*extra, n_freqs))
        pxy = np.zeros((*extra, n_freqs), dtype=np.complex128)

        for start in range(0, n - nperseg + 1, step):
            xw = x[..., start : start + nperseg] * win
            yw = y[..., start : start + nperseg] * win
            X = np.fft.rfft(xw)
            Y = np.fft.rfft(yw)
            pxx += np.real(X * np.conj(X))
            pyy += np.real(Y * np.conj(Y))
            pxy += X * np.conj(Y)

        coh_per_freq = np.abs(pxy) ** 2 / (pxx * pyy)
        return coh_per_freq.mean(axis=-1)  # average over frequency bins

    def __call__(self, data: Data) -> xr.DataArray:
        xr_data = data.data

        n_time = xr_data.sizes["time"]
        space_coords = xr_data.coords["space"].values
        n_space = len(space_coords)

        if n_space < 2:
            raise ValueError(f"coherence requires at least 2 spatial channels, got {n_space}")

        seg = min(256, n_time) if self.nperseg is None else self.nperseg
        if seg < 2:
            raise ValueError("nperseg must be at least 2")
        if seg > n_time:
            raise ValueError(
                f"nperseg ({seg}) must not exceed the number of time samples ({n_time})"
            )

        # Transpose to (*extra_dims, space, time) so we can slice space and time cleanly
        extra_dims = [d for d in xr_data.dims if d not in ("space", "time")]
        xr_t = xr_data.transpose(*extra_dims, "space", "time")
        arr = xr_t.values  # shape: (*extra_shape, n_space, n_time)

        extra_shape = arr.shape[:-2]
        coh_matrix = np.full((*extra_shape, n_space, n_space), np.nan)

        for i, j in itertools.combinations(range(n_space), 2):
            x = arr[..., i, :]  # (*extra_shape, n_time)
            y = arr[..., j, :]
            val = self._mean_squared_coherence(x, y, nperseg=seg)
            coh_matrix[..., i, j] = val
            coh_matrix[..., j, i] = val  # symmetric

        extra_coords = {d: xr_data.coords[d].values for d in extra_dims if d in xr_data.coords}
        return xr.DataArray(
            coh_matrix,
            dims=(*extra_dims, "space", "space_to"),
            coords={**extra_coords, "space": space_coords, "space_to": space_coords},
        )
