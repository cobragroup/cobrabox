from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, TypeAlias

import numpy as np
import pywt
import xarray as xr

from ..base_feature import BaseFeature
from ..data import Data, SignalData

_VALID_CWT_SCALINGS = ("magnitude", "power", "complex")

_CwtWavelet: TypeAlias = Literal[
    "morl",
    "mexh",
    "gaus1",
    "gaus2",
    "gaus3",
    "gaus4",
    "gaus5",
    "gaus6",
    "gaus7",
    "gaus8",
    "cgau1",
    "cgau2",
    "cgau3",
    "cgau4",
    "cgau5",
    "cgau6",
    "cgau7",
    "cgau8",
    "cmor",
    "fbsp",
    "shan",
]


@dataclass
class ContinuousWaveletTransform(BaseFeature[SignalData]):
    """Continuous wavelet transform (CWT) scalogram.

    Applies ``pywt.cwt`` along the ``time`` axis for each spatial channel,
    producing a 2-D time-scale representation. The output preserves the
    ``time`` dimension and adds a ``scale`` dimension, making it directly
    composable with downstream features that operate along time or space.

    Args:
        wavelet: Continuous wavelet name. Common choices for EEG analysis:

            * ``"morl"`` *(default)* — real Morlet; captures oscillatory
              components with strong time-frequency localisation.
            * ``"mexh"`` — Mexican Hat (second derivative of a Gaussian);
              highlights singularities and sharp transitions.
            * ``"cmor{B}-{C}"`` — complex Morlet (e.g. ``"cmor1.5-1.0"``);
              gives complex coefficients with better frequency resolution.

            See ``pywt.wavelist(kind='continuous')`` for all built-in options.
        scales: Explicit array of scales to analyse. If ``None`` (default),
            ``n_scales`` linearly-spaced integer scales ``[1, 2, ..., n_scales]``
            are used. Larger scales correspond to lower frequencies.
        n_scales: Number of scales when ``scales`` is ``None``. Defaults to 64.
        scaling: Output scaling. One of:

            * ``"magnitude"`` *(default)* — ``|CWT|``; real-valued, non-negative.
            * ``"power"`` — ``|CWT|²``; power scalogram.
            * ``"complex"`` — raw complex coefficients (dtype ``complex128``).

    Returns:
        xarray DataArray with dims ``(*extra_dims, "space", "scale", "time")``
        where:

        * ``scale`` holds the raw pywt scale values.
        * ``frequency`` is a non-index coordinate on the ``scale`` dimension
          giving the pseudo-frequency in Hz (from ``data.sampling_rate``;
          falls back to cycles/sample when ``sampling_rate`` is ``None``).
        * ``time`` preserves the original time coordinates from the input.

    Raises:
        ValueError: If ``wavelet`` is not a valid continuous wavelet, if
            ``scales`` is empty or contains non-positive values, if
            ``n_scales < 1``, or if ``scaling`` is not one of the three valid
            options.

    Example:
        >>> data = cb.load_dataset("dummy_random")[0]
        >>> cwt = cb.ContinuousWaveletTransform(n_scales=32).apply(data)
        >>> cwt.data.dims
        ('space', 'scale', 'time')
        >>> "frequency" in cwt.data.coords
        True
    """

    _tags: ClassVar[list[str]] = [
        "wavelet",
        "scalogram",
        "time-frequency",
        "scale-adaptive",
        "seizure-onset",
        "erps",
        "eeg",
        "io:frequency-time-output",
    ]

    wavelet: _CwtWavelet = "morl"
    scales: list[float] | None = None
    n_scales: int = 64
    scaling: Literal["magnitude", "power", "complex"] = "magnitude"

    output_type: ClassVar[type[Data] | None] = None

    def __post_init__(self) -> None:
        if self.scales is not None:
            if len(self.scales) == 0:
                raise ValueError("scales must not be empty")
            if any(s <= 0 for s in self.scales):
                raise ValueError("all scales must be positive")
        if self.n_scales < 1:
            raise ValueError(f"n_scales must be >= 1, got {self.n_scales}")
        if self.scaling not in _VALID_CWT_SCALINGS:
            raise ValueError(
                f"scaling must be one of {_VALID_CWT_SCALINGS!r}, got {self.scaling!r}"
            )
        try:
            pywt.ContinuousWavelet(self.wavelet)  # type: ignore
        except ValueError as exc:
            raise ValueError(
                f"Unknown continuous wavelet '{self.wavelet}'. "
                "Use pywt.wavelist(kind='continuous') to see valid options."
            ) from exc

    def __call__(self, data: SignalData) -> xr.DataArray:
        xr_data = data.data
        fs = data.sampling_rate if data.sampling_rate is not None else 1.0
        sampling_period = 1.0 / fs

        scales = (
            np.asarray(self.scales, dtype=np.float64)
            if self.scales is not None
            else np.arange(1, self.n_scales + 1, dtype=np.float64)
        )

        extra_dims = [d for d in xr_data.dims if d not in ("space", "time")]
        xr_t = xr_data.transpose(*extra_dims, "space", "time")
        arr = xr_t.values
        n_time = arr.shape[-1]
        time_coords = (
            xr_data.coords["time"].values
            if "time" in xr_data.coords
            else np.arange(n_time) * sampling_period
        )
        space_coords = xr_data.coords["space"].values
        extra_shape = arr.shape[:-2]
        n_space = arr.shape[-2]
        arr_flat = arr.reshape(-1, n_time)
        n_total = arr_flat.shape[0]
        n_scales_actual = len(scales)

        is_complex = self.scaling == "complex"
        dtype = np.complex128 if is_complex else np.float64
        out_flat = np.empty((n_total, n_scales_actual, n_time), dtype=dtype)

        frequencies: np.ndarray | None = None
        for i in range(n_total):
            coefs, freqs = pywt.cwt(
                arr_flat[i], scales, self.wavelet, sampling_period=sampling_period
            )
            if frequencies is None:
                frequencies = freqs
            if self.scaling == "magnitude":
                out_flat[i] = np.abs(coefs)
            elif self.scaling == "power":
                out_flat[i] = np.abs(coefs) ** 2
            else:
                out_flat[i] = coefs

        out = out_flat.reshape(*extra_shape, n_space, n_scales_actual, n_time)

        extra_coords = {d: xr_data.coords[d].values for d in extra_dims if d in xr_data.coords}
        return xr.DataArray(
            out,
            dims=(*extra_dims, "space", "scale", "time"),
            coords={
                **extra_coords,
                "space": space_coords,
                "scale": scales,
                "frequency": ("scale", frequencies),
                "time": time_coords,
            },
        )
