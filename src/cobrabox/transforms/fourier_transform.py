from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import xarray as xr

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
        xarray DataArray with dims ``(*non_time_dims, "frequency")``. Dtype is
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


@dataclass
class InverseFourierTransform(BaseFeature[Data]):
    """Inverse of :class:`FourierTransform`.

    Takes an array with a ``frequency`` dimension (complex coefficients) and
    returns a time-domain reconstruction.

    Args:
        n: Length of the output time axis. If ``None``, defaults to
            ``2 * (n_freq - 1)``.
        sampling_rate: Sampling rate to attach to the reconstructed time axis.
            When ``None`` the resulting Data has no sampling rate.

    Returns:
        xarray DataArray with the ``frequency`` dim replaced by ``time``.
    """

    _tags: ClassVar[list[str]] = ["ifft", "time-domain", "io:time-output", "req:frequency-input"]

    n: int | None = None
    sampling_rate: float | None = None

    output_type: ClassVar[type[Data]] = Data

    def __call__(self, data: Data) -> xr.DataArray:
        xr_data = data.data
        if "frequency" not in xr_data.dims:
            raise ValueError(
                f"InverseFourierTransform requires a 'frequency' dimension, got {xr_data.dims}"
            )
        ordered = xr_data.transpose(..., "frequency")
        if not np.iscomplexobj(ordered.values):
            raise ValueError(
                "InverseFourierTransform requires complex coefficients; pass "
                "FourierTransform(return_magnitude=False) upstream."
            )
        reconstructed = _irfft_1d(ordered.values, n=self.n, axis=-1)
        n_time = reconstructed.shape[-1]

        non_freq_dims = [d for d in ordered.dims if d != "frequency"]
        coords = {d: ordered.coords[d] for d in non_freq_dims if d in ordered.coords}
        if self.sampling_rate is not None:
            coords["time"] = np.arange(n_time) / float(self.sampling_rate)
        return xr.DataArray(reconstructed, dims=(*non_freq_dims, "time"), coords=coords)
