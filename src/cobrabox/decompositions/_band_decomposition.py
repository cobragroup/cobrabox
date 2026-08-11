from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal

import xarray as xr
from scipy import signal

from .._functional import functional
from ..base_feature import BaseFeature
from ..data import Data, SignalData

_DEFAULTS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "gamma": (30.0, 45.0),
}

_BAND_PRESETS: dict[str, dict[str, tuple[float, float]]] = {"eeg": _DEFAULTS}


@dataclass
class BandDecomposition(BaseFeature[SignalData]):
    """Decompose a signal into frequency bands.

    Applies a Butterworth bandpass filter for each band and stacks
    the results along a new ``band`` dimension.

    Args:
        bands: Band specification. ``"eeg"`` computes the five standard EEG bands:

            - ``delta``:  1 - 4 Hz
            - ``theta``:  4 - 8 Hz
            - ``alpha``:  8 - 12 Hz
            - ``beta``:  12 - 30 Hz
            - ``gamma``: 30 - 45 Hz

            Or pass a mapping of band name to ``[low_hz, high_hz]``
            frequency edges. ``None`` is not allowed.
        ord: Order of the filter.
            Defaults to 3.
        keep_orig: Whether to keep the original signal as a "band" named "original".
            Defaults to ``False``.

    Raises:
        ValueError: If ``bands`` is ``None``, an unknown preset string, or invalid.
        ValueError: If the input ``Data`` has no known ``sampling_rate``.

    Returns:
        :class:`~cobrabox.SignalData`: The band-filtered signals stacked along a new
            ``band`` dimension. The ``band`` coordinate contains the band names
            (and "original" if ``keep_orig=True``). The shape is the same as
            the input data with an additional ``band`` dimension.

    Example:
        >>> result = cb.BandDecomposition(bands="eeg").apply(data)
        >>> result = cb.BandDecomposition(bands={"alpha": [8, 12]}).apply(data)
    """

    _tags: ClassVar[list[str]] = [
        "filtering",
        "butterworth",
        "frequency-band",
        "preprocessing",
        "eeg",
        "io:preserves-time",
        "io:adds-dimension",
    ]

    bands: Literal["eeg"] | dict[str, list[float]]
    ord: int = 3
    keep_orig: bool = False

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if self.ord <= 0:
            raise ValueError(f"ord must be positive, got {self.ord}")
        if self.bands is None:
            raise ValueError("bands cannot be None; pass 'eeg' or a band mapping")
        if isinstance(self.bands, str):
            if self.bands not in _BAND_PRESETS:
                raise ValueError(
                    f"Unknown bands preset {self.bands!r}. "
                    f"Available presets: {list(_BAND_PRESETS)}."
                )
            resolved: Mapping[str, Sequence[float]] = dict(_BAND_PRESETS[self.bands])
        else:
            resolved = self.bands
        if not resolved:
            raise ValueError("bands cannot be empty")
        for band_name, freqs in resolved.items():
            if len(freqs) != 2:
                raise ValueError(f"Band '{band_name}' must have exactly 2 frequencies [low, high]")
            low, high = freqs
            if low < 0 or high < 0:
                raise ValueError(
                    f"Band '{band_name}' frequencies must be non-negative, got [{low}, {high}]"
                )
            if low >= high:
                raise ValueError(
                    f"Band '{band_name}' low frequency must be less than high, got [{low}, {high}]"
                )

    def __call__(self, data: SignalData) -> xr.DataArray:
        if data.sampling_rate is None:
            raise ValueError(
                "BandDecomposition requires a known sampling_rate on the input Data object"
            )

        resolved: Mapping[str, Sequence[float]] = (
            dict(_BAND_PRESETS[self.bands]) if isinstance(self.bands, str) else self.bands
        )

        band_arrays = []
        if self.keep_orig:
            band_arrays.append(data.data.assign_coords({"band": "original"}).expand_dims("band"))
        for band_name, freqs in resolved.items():
            b, a = signal.butter(self.ord, freqs, btype="band", fs=data.sampling_rate)
            filtered = xr.apply_ufunc(
                signal.lfilter,
                b,
                a,
                data.data,
                input_core_dims=[[], [], ["time"]],
                output_core_dims=[["time"]],
                vectorize=False,
            )
            band_arrays.append(filtered.assign_coords({"band": band_name}).expand_dims("band"))

        return xr.concat(band_arrays, dim="band")


@functional(BandDecomposition)
def band_decomposition(
    data: SignalData,
    bands: Literal["eeg"] | dict[str, list[float]],
    ord: int = 3,
    keep_orig: bool = False,
) -> Data:
    """Decompose a signal into frequency bands.

    Applies a Butterworth bandpass filter for each band and stacks
    the results along a new ``band`` dimension.

    Args:
        data: The input time-series signal to process, as a
            :class:`~cobrabox.SignalData` (or any :class:`~cobrabox.Data`
            carrying a ``time`` dimension).
        bands: Band specification. ``"eeg"`` computes the five standard EEG bands:

            - ``delta``:  1 - 4 Hz
            - ``theta``:  4 - 8 Hz
            - ``alpha``:  8 - 12 Hz
            - ``beta``:  12 - 30 Hz
            - ``gamma``: 30 - 45 Hz

            Or pass a mapping of band name to ``[low_hz, high_hz]``
            frequency edges. ``None`` is not allowed.
        ord: Order of the filter.
            Defaults to 3.
        keep_orig: Whether to keep the original signal as a "band" named "original".
            Defaults to ``False``.

    Raises:
        ValueError: If ``bands`` is ``None``, an unknown preset string, or invalid.
        ValueError: If the input ``Data`` has no known ``sampling_rate``.

    Returns:
        :class:`~cobrabox.SignalData`: The band-filtered signals stacked along a new
            ``band`` dimension. The ``band`` coordinate contains the band names
            (and "original" if ``keep_orig=True``). The shape is the same as
            the input data with an additional ``band`` dimension.

    Example:
        >>> result = cb.band_decomposition(data, bands="eeg")
        >>> result = cb.band_decomposition(data, bands={"alpha": [8, 12]})
    """
    return BandDecomposition(bands=bands, ord=ord, keep_orig=keep_orig).apply(data)
