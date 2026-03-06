from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from ..base_feature import SplitterFeature
from ..data import SignalData


@dataclass
class FourierTransformSurrogates(SplitterFeature[SignalData]):
    """Generate Fourier transform (i.e., preserving autocorrelation) of SignalData (surrogates
    the time axis).

    Args:
        n_surrogates: int
            Number of surrogate time series to generate.
        multivariate: bool = True
            If True (default) applies the same random phases to all the series. This ensures that
            also correlation is (approximately) preserved. This applies across all dimensions.
        return_data: bool = True
            If True (default) the generator has length 1+n_surrogates where the first element is
            the original data such that it can follow the same pipeline as the surrogates.
        random_state: np.random.Generator | int | None = None
            Initialiser for the pseudorandom number generator to ensure reproducibility.

    Yields:
        SignalData iterator containing the surrogate time series.

    Example:
        >>> import cobrabox as cb
        >>> data = cb.SignalData.from_numpy(
        ...     np.random.randn(100, 10),
        ...     dims=["time", "space"],
        ...     sampling_rate=100.0
        ... )
        >>> feature = cb.feature.FourierTransformSurrogates(
        ...     n_surrogates=5, random_state=42
        ... )
        >>> surrogates = list(feature(data))
    """

    n_surrogates: int
    multivariate: bool = True
    return_data: bool = True
    random_state: np.random.Generator | int | None = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.n_surrogates, int):
            raise ValueError("The number of surrogates must be an integer.")
        if self.n_surrogates < 0:
            raise ValueError("The number of surrogates cannot be negative.")
        object.__setattr__(self, "_rng", np.random.default_rng(self.random_state))

    def _surrogate(self, data: SignalData) -> SignalData:
        transposed = data.data.transpose(..., "time")
        tmp = np.reshape(transposed.data, [-1, transposed.shape[-1]])
        fft = np.fft.rfft(tmp, axis=1)
        extra_shape = 1 if self.multivariate else tmp.shape[0]

        rpha = np.exp(2 * np.pi * self._rng.random([extra_shape, int(tmp.shape[1] / 2 + 1)]) * 1.0j)
        fftX1 = fft * rpha

        xs = np.reshape(np.fft.irfft(fftX1, n=tmp.shape[1], axis=1), data.data.shape)

        result_da = xr.DataArray(
            xs, dims=data.data.dims, coords=data.data.coords, attrs=data.data.attrs
        )
        return data._copy_with_new_data(result_da)

    def __call__(self, data: SignalData) -> Iterator[SignalData]:

        if self.return_data:
            yield data
        for _ in range(self.n_surrogates):
            yield self._surrogate(data)
