from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import xarray as xr
from scipy import signal

from .._functional import functional
from ..base_feature import BaseFeature
from ..data import Data, SignalData


@dataclass
class BandpassFilter(BaseFeature[SignalData]):
    """Filter a signal to keep one or more frequency ranges.

    Applies a Butterworth bandpass filter for each frequency range in
    ``bands`` and sums the filtered signals into a single output. When
    more than one range is given, the output is the sum of the
    individual band-filtered signals — a reconstruction of the signal
    from its selected frequency components.

    The output has the same shape and dimensions as the input (no
    ``band`` dimension is added).

    Args:
        bands: One or more ``[low_hz, high_hz]`` frequency ranges to keep,
            e.g. ``[[8, 12]]`` or ``[[1, 4], [8, 12]]``. A single range may
            also be given unnested as ``[8, 12]``. Each range is applied as
            a Butterworth bandpass filter and the results are summed.
        ord: Order of the filter.
            Defaults to 3.
        zero_phase: If ``True`` (default), uses :func:`scipy.signal.filtfilt` for
            zero-phase (forward-backward) filtering with no phase distortion —
            the MNE-style default for EEG preprocessing. If ``False``, uses
            :func:`scipy.signal.lfilter`, a causal forward-only IIR filter that
            introduces phase distortion but may be preferable for causal or
            online processing. Note that each range incurs a different delay
            under causal filtering, so summing several ranges with
            ``zero_phase=False`` does not reconstruct the signal.

    Raises:
        ValueError: If ``bands`` is ``None`` or empty, a range does not have
            exactly 2 frequencies, a frequency is not positive, ``low >= high``,
            a range reaches or exceeds the Nyquist frequency, or the input has
            no known ``sampling_rate``.
        TypeError: If ``bands`` is not a sequence of ``[low, high]`` pairs.

    Returns:
        :class:`~cobrabox.SignalData` with the same shape, dimensions, and
        metadata as the input. The values are the sum of the band-filtered
        signals.

    Example:
        >>> result = cb.BandpassFilter(bands=[[8, 12]]).apply(data)
        >>> result = cb.BandpassFilter(bands=[[1, 4], [8, 12]]).apply(data)
        >>> result = cb.BandpassFilter(bands=[8, 12]).apply(data)  # single range
    """

    _tags: ClassVar[list[str]] = [
        "filtering",
        "butterworth",
        "preprocessing",
        "eeg",
        "io:preserves-time",
    ]

    bands: list[list[float]]
    ord: int = 3
    zero_phase: bool = True

    def __post_init__(self) -> None:
        """Validate parameters and normalise ``bands`` to a list of pairs."""
        if self.ord <= 0:
            raise ValueError(f"ord must be positive, got {self.ord}")
        if self.bands is None:
            raise ValueError("bands cannot be None; pass one or more [low_hz, high_hz] ranges")
        if isinstance(self.bands, str | dict):
            raise TypeError(
                f"bands must be a sequence of [low_hz, high_hz] pairs, got "
                f"{type(self.bands).__name__}. BandpassFilter no longer takes a band "
                "mapping or an 'eeg' preset — use BandDecomposition for per-band "
                "output stacked along a 'band' dimension."
            )
        try:
            ranges: list[Any] = list(self.bands)
        except TypeError:
            raise TypeError(
                f"bands must be a sequence of [low_hz, high_hz] pairs, got "
                f"{type(self.bands).__name__}"
            ) from None
        if not ranges:
            raise ValueError("bands cannot be empty")
        # Accept a single unnested [low, high] pair for convenience.
        if all(not hasattr(freq, "__len__") for freq in ranges):
            if len(ranges) != 2:
                raise ValueError(
                    f"A single unnested range must be [low_hz, high_hz], got "
                    f"{len(ranges)} values. Pass a list of pairs for multiple ranges."
                )
            ranges = [ranges]
        normalised: list[list[float]] = []
        for range_i, freqs in enumerate(ranges):
            if isinstance(freqs, str | dict) or not hasattr(freqs, "__len__"):
                raise TypeError(
                    f"Range {range_i} must be a [low_hz, high_hz] pair, got {type(freqs).__name__}"
                )
            if len(freqs) != 2:
                raise ValueError(f"Range {range_i} must have exactly 2 frequencies [low, high]")
            low, high = (float(freq) for freq in freqs)
            if low <= 0 or high <= 0:
                raise ValueError(
                    f"Range {range_i} frequencies must be positive, got [{low}, {high}]"
                )
            if low >= high:
                raise ValueError(
                    f"Range {range_i} low frequency must be less than high, got [{low}, {high}]"
                )
            normalised.append([low, high])
        self.bands = normalised

    def __call__(self, data: SignalData) -> xr.DataArray:
        if data.sampling_rate is None:
            raise ValueError(
                "BandpassFilter requires a known sampling_rate on the input Data object"
            )
        nyquist = data.sampling_rate / 2.0
        for range_i, (_low, high) in enumerate(self.bands):
            if high >= nyquist:
                raise ValueError(
                    f"Range {range_i} high frequency ({high} Hz) must be less than the "
                    f"Nyquist frequency ({nyquist} Hz, half the sampling_rate of "
                    f"{data.sampling_rate} Hz)"
                )

        func = signal.filtfilt if self.zero_phase else signal.lfilter
        total = None
        for low, high in self.bands:
            b, a = signal.butter(self.ord, [low, high], btype="band", fs=data.sampling_rate)
            filtered = xr.apply_ufunc(
                func,
                b,
                a,
                data.data,
                input_core_dims=[[], [], ["time"]],
                output_core_dims=[["time"]],
                vectorize=False,
            )
            total = filtered if total is None else total + filtered

        assert total is not None  # bands is validated non-empty in __post_init__
        return total


@functional(BandpassFilter)
def bandpass_filter(
    data: SignalData, bands: list[list[float]], ord: int = 3, zero_phase: bool = True
) -> Data:
    """Filter a signal to keep one or more frequency ranges.

    Applies a Butterworth bandpass filter for each frequency range in
    ``bands`` and sums the filtered signals into a single output. When
    more than one range is given, the output is the sum of the
    individual band-filtered signals — a reconstruction of the signal
    from its selected frequency components.

    The output has the same shape and dimensions as the input (no
    ``band`` dimension is added).

    Args:
        data: The input time-series signal to process, as a
            :class:`~cobrabox.SignalData` (or any :class:`~cobrabox.Data`
            carrying a ``time`` dimension).
        bands: One or more ``[low_hz, high_hz]`` frequency ranges to keep,
            e.g. ``[[8, 12]]`` or ``[[1, 4], [8, 12]]``. A single range may
            also be given unnested as ``[8, 12]``. Each range is applied as
            a Butterworth bandpass filter and the results are summed.
        ord: Order of the filter.
            Defaults to 3.
        zero_phase: If ``True`` (default), uses :func:`scipy.signal.filtfilt` for
            zero-phase (forward-backward) filtering with no phase distortion —
            the MNE-style default for EEG preprocessing. If ``False``, uses
            :func:`scipy.signal.lfilter`, a causal forward-only IIR filter that
            introduces phase distortion but may be preferable for causal or
            online processing. Note that each range incurs a different delay
            under causal filtering, so summing several ranges with
            ``zero_phase=False`` does not reconstruct the signal.

    Raises:
        ValueError: If ``bands`` is ``None`` or empty, a range does not have
            exactly 2 frequencies, a frequency is not positive, ``low >= high``,
            a range reaches or exceeds the Nyquist frequency, or the input has
            no known ``sampling_rate``.
        TypeError: If ``bands`` is not a sequence of ``[low, high]`` pairs.

    Returns:
        :class:`~cobrabox.SignalData` with the same shape, dimensions, and
        metadata as the input. The values are the sum of the band-filtered
        signals.

    Example:
        >>> result = cb.bandpass_filter(data, bands=[[8, 12]])
        >>> result = cb.bandpass_filter(data, bands=[[1, 4], [8, 12]])
        >>> result = cb.bandpass_filter(data, bands=[8, 12])  # single range
    """
    return BandpassFilter(bands=bands, ord=ord, zero_phase=zero_phase).apply(data)
