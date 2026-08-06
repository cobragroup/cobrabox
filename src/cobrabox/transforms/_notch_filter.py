from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import xarray as xr
from scipy import signal

from .._functional import functional
from ..base_feature import BaseFeature
from ..data import Data, SignalData


@dataclass
class NotchFilter(BaseFeature[SignalData]):
    """Apply a notch filter to remove a specific frequency.

    Attenuates a narrow band centered on ``freq`` using :func:`scipy.signal.iirnotch`
    followed by filtering along the time axis. All channels are filtered independently.

    Args:
        freq: Center frequency in Hz to remove (e.g. 50 or 60 Hz for power-line noise).
        q: Quality factor controlling the notch width. Higher values create
            narrower notches. Defaults to 30.0.
        zero_phase: If ``True`` (default), uses :func:`scipy.signal.filtfilt` for
            zero-phase (forward-backward) filtering with no phase distortion —
            the MNE-style default for EEG preprocessing. If ``False``, uses
            :func:`scipy.signal.lfilter`, a causal forward-only IIR filter that
            introduces phase distortion but may be preferable for causal or
            online processing.

    Raises:
        ValueError: If ``freq`` is not positive, ``q`` is not positive,
            ``freq`` exceeds the Nyquist frequency, or the input has no
            ``sampling_rate``.

    Returns:
        :class:`~cobrabox.SignalData` with the same shape, dimensions, and
        metadata as the input.

    Example:
        >>> result = cb.NotchFilter(freq=50.0).apply(data)
        >>> result = cb.NotchFilter(freq=60.0, q=30.0).apply(data)
        >>> result = cb.NotchFilter(freq=50.0, zero_phase=False).apply(data)
    """

    _tags: ClassVar[list[str]] = ["filtering", "preprocessing", "notch", "eeg", "io:preserves-time"]

    freq: float
    q: float = 30.0
    zero_phase: bool = True

    def __post_init__(self) -> None:
        if self.freq <= 0.0:
            raise ValueError(f"freq must be positive, got {self.freq}")
        if self.q <= 0.0:
            raise ValueError(f"q must be positive, got {self.q}")

    def __call__(self, data: SignalData) -> xr.DataArray:
        if data.sampling_rate is None:
            raise ValueError("NotchFilter requires a known sampling_rate on the input Data object")
        nyquist = data.sampling_rate / 2.0
        if self.freq >= nyquist:
            raise ValueError(
                f"freq ({self.freq} Hz) must be less than the Nyquist frequency "
                f"({nyquist} Hz, half the sampling_rate of {data.sampling_rate} Hz)"
            )

        b, a = signal.iirnotch(self.freq, self.q, fs=data.sampling_rate)
        func = signal.filtfilt if self.zero_phase else signal.lfilter
        return xr.apply_ufunc(
            func,
            b,
            a,
            data.data,
            input_core_dims=[[], [], ["time"]],
            output_core_dims=[["time"]],
            vectorize=False,
        )


@functional(NotchFilter)
def notch_filter(data: SignalData, freq: float, q: float = 30.0, zero_phase: bool = True) -> Data:
    """Apply a notch filter to remove a specific frequency.

    Attenuates a narrow band centered on ``freq`` using :func:`scipy.signal.iirnotch`
    followed by filtering along the time axis. All channels are filtered independently.

    Args:
        data: The input time-series signal to process, as a
            :class:`~cobrabox.SignalData` (or any :class:`~cobrabox.Data`
            carrying a ``time`` dimension).
        freq: Center frequency in Hz to remove (e.g. 50 or 60 Hz for power-line noise).
        q: Quality factor controlling the notch width. Higher values create
            narrower notches. Defaults to 30.0.
        zero_phase: If ``True`` (default), uses :func:`scipy.signal.filtfilt` for
            zero-phase (forward-backward) filtering with no phase distortion —
            the MNE-style default for EEG preprocessing. If ``False``, uses
            :func:`scipy.signal.lfilter`, a causal forward-only IIR filter that
            introduces phase distortion but may be preferable for causal or
            online processing.

    Raises:
        ValueError: If ``freq`` is not positive, ``q`` is not positive,
            ``freq`` exceeds the Nyquist frequency, or the input has no
            ``sampling_rate``.

    Returns:
        :class:`~cobrabox.SignalData` with the same shape, dimensions, and
        metadata as the input.

    Example:
        >>> result = cb.notch_filter(data, freq=50.0)
        >>> result = cb.notch_filter(data, freq=60.0, q=30.0)
        >>> result = cb.notch_filter(data, freq=50.0, zero_phase=False)
    """
    return NotchFilter(freq=freq, q=q, zero_phase=zero_phase).apply(data)
