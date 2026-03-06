from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from ..base_feature import SplitterFeature
from ..data import SignalData


@dataclass
class FourierTransformSurrogates(SplitterFeature[SignalData]):
    """Generate Fourier transform (i.e., preserving autocorrelation) of SignalData (surrogates
    the time axis).

    Parameters:
    ---------
        n_surrogates: int
        multivariate: bool = True
            If True (default) applies the same random phases to all the series. This ensures that
            also correlation is (approximately) preserved. This applies across all dimensions.
        return_data: bool = True
            If True (default) the generator has length 1+n_surrogates where the first element is
            the original data such that it can follow the same pipeline as the surrogates.
        random_state: np.random.Generator | int | None = None
            Initialiser for the pseudorandom number generator to ensure reproducibility.

    Output:
    ----------
    SignalData iterator containing the surrogate time series.
    """

    n_surrogates: int
    multivariate: bool = True
    return_data: bool = True
    random_state: np.random.Generator | int | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.n_surrogates, int), "The number of surrogates must be an integer."
        assert self.n_surrogates >= 0, "The number of surrogates cannot be negative."

    def _surrogate(self, data: SignalData) -> SignalData:
        rng = np.random.default_rng(self.random_state)
        data.data.transpose(..., "time")
        tmp = np.reshape(data.data.data, [-1, data.data.shape[-1]])
        fft = np.fft.rfft(tmp, axis=1)
        extra_shape = 1 if self.multivariate else tmp.shape[0]

        rpha = np.exp(2 * np.pi * rng.random([extra_shape, int(tmp.shape[1] / 2 + 1)]) * 1.0j)
        fftX1 = fft * rpha

        xs = np.reshape(np.fft.irfft(fftX1, n=tmp.shape[1], axis=1), data.data.shape)

        return SignalData.from_numpy(xs, dims=list(data.data.dims))

    def __call__(self, data: SignalData) -> Iterator[SignalData]:

        if self.return_data:
            yield data
        for _ in range(self.n_surrogates):
            yield self._surrogate(data)
