from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

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
        bands: One or more ``[low_hz, high_hz]`` frequency ranges to keep.
            Each range is applied as a Butterworth bandpass filter and the
            results are summed.
        ord: Order of the filter.
            Defaults to 3.

    Raises:
        ValueError: If ``bands`` is empty, a range does not have exactly
            2 frequencies, a frequency is negative, ``low >= high``, a
            range exceeds the Nyquist frequency, or the input has no
            known ``sampling_rate``.

    Returns:
        :class:`~cobrabox.SignalData` with the same shape, dimensions, and
        metadata as the input. The values are the sum of the band-filtered
        signals.

    Example:
        >>> result = cb.BandpassFilter(bands=[[8, 12]]).apply(data)
        >>> result = cb.BandpassFilter(bands=[[1, 4], [8, 12]]).apply(data)
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

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if self.ord <= 0:
            raise ValueError(f"ord must be positive, got {self.ord}")
        if not self.bands:
            raise ValueError("bands cannot be empty")
        for range_i, freqs in enumerate(self.bands):
            if len(freqs) != 2:
                raise ValueError(f"Range {range_i} must have exactly 2 frequencies [low, high]")
            low, high = freqs
            if low < 0 or high < 0:
                raise ValueError(
                    f"Range {range_i} frequencies must be non-negative, got [{low}, {high}]"
                )
            if low >= high:
                raise ValueError(
                    f"Range {range_i} low frequency must be less than high, got [{low}, {high}]"
                )

    def __call__(self, data: SignalData) -> xr.DataArray:
        if data.sampling_rate is None:
            raise ValueError(
                "BandpassFilter requires a known sampling_rate on the input Data object"
            )
        nyquist = data.sampling_rate / 2.0
        for range_i, (_low, high) in enumerate(self.bands):
            if high > nyquist:
                raise ValueError(
                    f"Range {range_i} high frequency ({high} Hz) exceeds the Nyquist "
                    f"frequency ({nyquist} Hz, half the sampling_rate of "
                    f"{data.sampling_rate} Hz)"
                )

        total = None
        for low, high in self.bands:
            b, a = signal.butter(self.ord, [low, high], btype="band", fs=data.sampling_rate)
            filtered = xr.apply_ufunc(
                signal.lfilter,
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
def bandpass_filter(data: SignalData, bands: list[list[float]], ord: int = 3) -> Data:
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
        bands: One or more ``[low_hz, high_hz]`` frequency ranges to keep.
            Each range is applied as a Butterworth bandpass filter and the
            results are summed.
        ord: Order of the filter.
            Defaults to 3.

    Raises:
        ValueError: If ``bands`` is empty, a range does not have exactly
            2 frequencies, a frequency is negative, ``low >= high``, a
            range exceeds the Nyquist frequency, or the input has no
            known ``sampling_rate``.

    Returns:
        :class:`~cobrabox.SignalData` with the same shape, dimensions, and
        metadata as the input. The values are the sum of the band-filtered
        signals.

    Example:
        >>> result = cb.bandpass_filter(data, bands=[[8, 12]])
        >>> result = cb.bandpass_filter(data, bands=[[1, 4], [8, 12]])
    """
    return BandpassFilter(bands=bands, ord=ord).apply(data)
