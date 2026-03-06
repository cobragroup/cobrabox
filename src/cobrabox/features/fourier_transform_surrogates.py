from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from ..base_feature import SplitterFeature
from ..data import SignalData


@dataclass
class FourierTransformSurrogates(SplitterFeature[SignalData]):
    """Generate Fourier transform surrogates (preserving autocorrelation) of SignalData.

    Creates surrogate time series by randomizing the phases of the Fourier transform
    while preserving the power spectrum (and thus autocorrelation). This generates
    null-hypothesis data for testing whether observed effects exceed what would be
    expected from linear, stationary, Gaussian processes with the same correlation
    structure.

    Args:
        n_surrogates: Number of surrogate time series to generate. Must be a
            non-negative integer.
        multivariate: If True (default), applies the same random phases to all
            series. This ensures that cross-correlations are approximately preserved.
        return_data: If True (default), the generator yields the original data
            first, followed by the surrogates. This gives length 1 + n_surrogates.
        random_state: Initialiser for the pseudorandom number generator to
            ensure reproducibility. Can be a Generator, int seed, or None.

    Returns:
        Iterator yielding SignalData objects containing the surrogate time series
        (and optionally the original data first).

    Raises:
        ValueError: If n_surrogates is not an integer.
        ValueError: If n_surrogates is negative.

    References:
        Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., & Farmer, J. D.
        (1992). Testing for nonlinearity in time series: the method of surrogate
        data. Physica D: Nonlinear Phenomena, 58(1-4), 77-94.
        https://doi.org/10.1016/0167-2789(92)90102-S

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
