from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import xarray as xr
from scipy import signal

from ..base_feature import BaseFeature
from ..data import Data, SignalData

_DEFAULTS: dict[str, list[float]] = {
    "delta": [1.0, 4.0],
    "theta": [4.0, 8.0],
    "alpha": [8.0, 12.0],
    "beta": [12.0, 30.0],
    "gamma": [30.0, 45.0],
}


@dataclass
class BandFilter(BaseFeature[SignalData]):
    """Filter a signal into frequency bands.

    Applies a 3rd-order Butterworth bandpass filter for each band and stacks
    the results along a new ``band`` dimension placed first, before ``space``
    and ``time``.

    Dimension ordering convention: new dims first, ``space`` second-to-last,
    ``time`` always last — e.g. ``(band, space, time)``.

    Args:
        bands: Mapping of band name to ``[low_hz, high_hz]`` frequency edges.
            Defaults to the five standard EEG bands:
            delta (1–4 Hz), theta (4–8 Hz), alpha (8–12 Hz),
            beta (12–30 Hz), gamma (30–45 Hz).

    Raises:
        ValueError: If the input ``Data`` has no known ``sampling_rate``.

    Example:
        >>> result = cb.feature.BandFilter().apply(data)
        >>> result = cb.feature.BandFilter(bands={"alpha": [8, 12]}).apply(data)
    """

    bands: dict[str, list[float]] = field(default_factory=lambda: dict(_DEFAULTS))
    output_type: ClassVar[type[Data]] = Data

    def __call__(self, data: SignalData) -> xr.DataArray:
        if data.sampling_rate is None:
            raise ValueError("BandFilter requires a known sampling_rate on the input Data object")

        band_arrays = []
        for band_name, freqs in self.bands.items():
            b, a = signal.butter(  # type: ignore[misc]
                3, freqs, btype="band", fs=data.sampling_rate
            )
            # apply_ufunc routes lfilter along the "time" dimension by label —
            # no axis arithmetic needed.
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

        # concat places "band" first; apply_ufunc keeps "time" last —
        # giving the (band, ..., space, time) convention automatically.
        return xr.concat(band_arrays, dim="band")
