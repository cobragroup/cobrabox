from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import xarray as xr

from .._functional import functional
from ..base_feature import BaseFeature
from ..data import Data, SignalData


def _rfft_1d(signal: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Real-input FFT along ``axis``. Wraps :func:`numpy.fft.rfft`."""
    return np.fft.rfft(signal, axis=axis)


def _irfft_1d(coeffs: np.ndarray, *, n: int | None = None, axis: int = -1) -> np.ndarray:
    """Inverse of :func:`_rfft_1d`. Wraps :func:`numpy.fft.irfft`."""
    return np.fft.irfft(coeffs, n=n, axis=axis)


def _rfftfreq(n: int, d: float = 1.0) -> np.ndarray:
    """Return the frequency bins for an ``_rfft_1d`` output of length ``n``."""
    return np.fft.rfftfreq(n, d=d)


@dataclass
class FourierTransform(BaseFeature[SignalData]):
    """Real-valued FFT along the time axis.

    Produces a frequency-domain representation of every channel. Output dims
    are ``(*non_time_dims, "frequency")``; ``time`` is consumed. When
    ``data.sampling_rate`` is available the ``frequency`` coordinate is in Hz,
    otherwise it falls back to cycles-per-sample.

    Args:
        return_magnitude: If ``True`` (default), return ``|FFT|`` as a
            real-valued array. If ``False``, return the raw complex
            coefficients.

    Returns:
        :class:`~cobrabox.Data` with dims ``(*non_time_dims, "frequency")``. Dtype is
        ``complex128`` when ``return_magnitude=False`` and ``float64``
        otherwise.
    """

    _tags: ClassVar[list[str]] = ["fft", "frequency-domain", "eeg", "fmri", "io:frequency-output"]

    return_magnitude: bool = True

    output_type: ClassVar[type[Data]] = Data

    def __call__(self, data: SignalData) -> xr.DataArray:
        xr_data = data.data
        if "time" not in xr_data.dims:
            raise ValueError("FourierTransform requires a 'time' dimension")

        ordered = xr_data.transpose(..., "time")
        n_time = ordered.sizes["time"]
        sr = float(data.sampling_rate) if data.sampling_rate is not None else 1.0

        coeffs = _rfft_1d(ordered.values, axis=-1)
        result = np.abs(coeffs) if self.return_magnitude else coeffs

        freqs = _rfftfreq(n_time, d=1.0 / sr)
        non_time_dims = [d for d in ordered.dims if d != "time"]
        coords = {d: ordered.coords[d] for d in non_time_dims if d in ordered.coords}
        coords["frequency"] = freqs
        return xr.DataArray(result, dims=(*non_time_dims, "frequency"), coords=coords)


@functional(FourierTransform)
def fourier_transform(data: SignalData, return_magnitude: bool = True) -> Data:
    """Real-valued FFT along the time axis.

    Produces a frequency-domain representation of every channel. Output dims
    are ``(*non_time_dims, "frequency")``; ``time`` is consumed. When
    ``data.sampling_rate`` is available the ``frequency`` coordinate is in Hz,
    otherwise it falls back to cycles-per-sample.

    Args:
        data: The input time-series signal to process, as a
            :class:`~cobrabox.SignalData` (or any :class:`~cobrabox.Data`
            carrying a ``time`` dimension).
        return_magnitude: If ``True`` (default), return ``|FFT|`` as a
            real-valued array. If ``False``, return the raw complex
            coefficients.

    Returns:
        :class:`~cobrabox.Data` with dims ``(*non_time_dims, "frequency")``. Dtype is
        ``complex128`` when ``return_magnitude=False`` and ``float64``
        otherwise.
    """
    return FourierTransform(return_magnitude=return_magnitude).apply(data)
