from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import xarray as xr

from .._functional import functional
from ..base_feature import BaseFeature
from ..data import Data, SignalData
from ..transforms._fourier_transform import FourierTransform


@dataclass
class PowerSpectralDensity(BaseFeature[SignalData]):
    """Power spectral density via FFT.

    Convenience wrapper around :class:`~cobrabox.FourierTransform` with
    ``norm="psd"``.  Computes the one-sided PSD as
    ``|FFT|^2 / (N * sampling_rate)`` with the interior bins doubled to
    account for the discarded negative frequencies.

    Args:
        cutoff: Optional upper frequency limit in Hz.  Frequencies above
            ``cutoff`` are discarded.  If ``None`` (default) the full
            spectrum up to the Nyquist frequency is returned.

    Returns:
        :class:`~cobrabox.Data` with dims ``(*non_time_dims, "frequency")``.
        Dtype is ``float64``.

    Raises:
        ValueError: If ``cutoff`` is negative.
        UserWarning: If ``data`` has no ``sampling_rate``.
        UserWarning: If ``cutoff`` exceeds the Nyquist frequency.
    """

    _tags: ClassVar[list[str]] = ["psd", "power-spectrum", "frequency-domain", "eeg", "fmri"]

    cutoff: float | None = None

    output_type: ClassVar[type[Data]] = Data

    def __call__(self, data: SignalData) -> xr.DataArray:
        return FourierTransform(norm="psd", cutoff=self.cutoff)(data)


@functional(PowerSpectralDensity)
def power_spectral_density(data: SignalData, cutoff: float | None = None) -> Data:
    """Power spectral density via FFT.

    Convenience wrapper around :func:`~cobrabox.fourier_transform` with
    ``norm="psd"``.  Computes the one-sided PSD as
    ``|FFT|^2 / (N * sampling_rate)`` with the interior bins doubled to
    account for the discarded negative frequencies.

    Args:
        data: The input time-series signal to process, as a
            :class:`~cobrabox.SignalData` (or any :class:`~cobrabox.Data`
            carrying a ``time`` dimension).
        cutoff: Optional upper frequency limit in Hz.  Frequencies above
            ``cutoff`` are discarded.  If ``None`` (default) the full
            spectrum up to the Nyquist frequency is returned.

    Returns:
        :class:`~cobrabox.Data` with dims ``(*non_time_dims, "frequency")``.
        Dtype is ``float64``.

    Raises:
        ValueError: If ``cutoff`` is negative.
        UserWarning: If ``data`` has no ``sampling_rate``.
        UserWarning: If ``cutoff`` exceeds the Nyquist frequency.
    """
    return PowerSpectralDensity(cutoff=cutoff).apply(data)
