from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar
from warnings import warn

import numpy as np
import xarray as xr

from .._functional import functional
from ..base_feature import BaseFeature
from ..data import Data, SignalData


def _rfft_1d(signal: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Real-input FFT along ``axis``. Wraps :func:`numpy.fft.rfft`."""
    return np.fft.rfft(signal, axis=axis)

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
        norm: If ``None`` (default), returns the raw complex coefficients.
            If ``"real"`` returns ``|FFT|`` as a real-valued array.
            If ``"psd"`` returns the power spectral density ``|FFT|^2``.
        cutoff: None (default) or ``float`` (in Hz). Largest frequency returned.
            If ``None`` or ``cutoff`` > Nyquist freq, parameter is ignored and
            output contains frequencies up to Nyquist frequency ``self.sampling_rate // 2``.
            Given value must be positive.

    Returns:
        :class:`~cobrabox.Data` with dims ``(*non_time_dims, "frequency")``. Dtype is
        ``complex128`` when ``norm=None`` and ``float64`` otherwise.

    Raises:
        ValueError: If the input ``Data`` has no known ``sampling_rate``.
        ValueError: If ``norm`` is not one of [None, 'real', 'psd'].
        ValueError: If negative ``cutoff`` is given.
        UserWarning: If ``data`` does not have ``sampling_rate`` defined.
        UserWarning: If ``cutoff`` larger than Nyquist frequency given.
    """

    _tags: ClassVar[list[str]] = ["fft", "frequency-domain", "eeg", "fmri", "io:frequency-output"]

    norm: Literal["real", "psd"] | None = None
    cutoff: float | None = None

    output_type: ClassVar[type[Data]] = Data

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if self.cutoff is not None and self.cutoff < 0:
            raise ValueError(f"cutoff must be positive, got {self.cutoff}")

        if self.norm not in [None, "real", "psd"]:
            raise ValueError("norm must be one of [None, 'real', 'psd']")

    def __call__(self, data: SignalData) -> xr.DataArray:
        try:
            _sr = float(data.sampling_rate)
        except (TypeError, ValueError):
            _sr = 1.0
            _warn_msg1 = "Fourier transform requires 'sampling_rate' on input Data. None found, setting to 1.0"
            warn( _warn_msg1, category=UserWarning )

        _nyq = _sr / 2
        if self.cutoff is not None and self.cutoff > _nyq:
            _warn_msg2 = f"cutoff ({self.cutoff} Hz) larger than Nyquist frequency ({_nyq} Hz) for present data. \nIgnoring cutoff and using Nyquist."
            warn( _warn_msg2 ,category=UserWarning)
            self.cutoff = None

        xr_data = data.data
        if "time" not in xr_data.dims:
            raise ValueError("FourierTransform requires a 'time' dimension")

        ordered = xr_data.transpose(..., "time")
        n_time = ordered.sizes["time"]

        coeffs = _rfft_1d(ordered.values, axis=-1)
        if self.norm == 'real':
            coeffs = np.abs(coeffs)
        elif self.norm == "psd":
            ## TODO: Double-check this is the correct normalization
            coeffs = np.abs(coeffs)**2

        freqs = _rfftfreq(n_time, d=1.0 / _sr)
        non_time_dims = [d for d in ordered.dims if d != "time"]
        coords = {d: ordered.coords[d] for d in non_time_dims if d in ordered.coords}

        if self.cutoff is not None:
            mask = freqs <= self.cutoff
            maxidx = mask.sum() + 1
            freqs = freqs[:maxidx]
            coeffs = coeffs[:,:maxidx]

        coords["frequency"] = freqs
        return xr.DataArray(coeffs, dims=(*non_time_dims, "frequency"), coords=coords)


@functional(FourierTransform)
def fourier_transform(
    data: SignalData,
    norm: Literal["real", "psd"] | None = None,
    cutoff: float | None = None
) -> Data:
    """Real-valued FFT along the time axis.

    Produces a frequency-domain representation of every channel. Output dims
    are ``(*non_time_dims, "frequency")``; ``time`` is consumed. When
    ``data.sampling_rate`` is available the ``frequency`` coordinate is in Hz,
    otherwise it falls back to cycles-per-sample.

    Args:
        data: The input time-series signal to process, as a
            :class:`~cobrabox.SignalData` (or any :class:`~cobrabox.Data`
            carrying a ``time`` dimension).
        norm: If ``None`` (default), returns the raw complex coefficients.
            If ``"real"`` returns ``|FFT|`` as a real-valued array.
            If ``"psd"`` returns the power spectral density ``|FFT|^2``.
        cutoff: None (default) or ``float`` (in Hz). Largest frequency returned.
            If ``None`` or ``cutoff`` > Nyquist freq, parameter is ignored and
            output contains frequencies up to Nyquist frequency ``self.sampling_rate // 2``.
            Given value must be positive.

    Returns:
        :class:`~cobrabox.Data` with dims ``(*non_time_dims, "frequency")``. Dtype is
        ``complex128`` when ``norm=None`` and ``float64`` otherwise.

    Raises:
        ValueError: If the input ``Data`` has no known ``sampling_rate``.
        ValueError: If ``norm`` is not one of [None, 'real', 'psd'].
        ValueError: If negative ``cutoff`` is given.
        UserWarning: If ``data`` does not have ``sampling_rate`` defined.
        UserWarning: If ``cutoff`` larger than Nyquist frequency given.
    """
    return FourierTransform(norm=norm,cutoff=cutoff).apply(data)
