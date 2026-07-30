from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, TypeAlias

import numpy as np
import pywt
import xarray as xr

from .._functional import functional
from ..base_feature import BaseFeature
from ..data import Data, SignalData

_DwtWavelet: TypeAlias = Literal[
    "haar",
    "db1",
    "db2",
    "db3",
    "db4",
    "db5",
    "db6",
    "db7",
    "db8",
    "sym2",
    "sym3",
    "sym4",
    "sym5",
    "sym6",
    "sym7",
    "sym8",
    "coif1",
    "coif2",
    "coif3",
    "coif4",
    "coif5",
    "dmey",
]


@dataclass
class DiscreteWaveletTransform(BaseFeature[SignalData]):
    """Multi-level discrete wavelet decomposition (DWT).

    Uses ``pywt.wavedec`` to decompose the ``time`` axis into one approximation
    and ``level`` detail coefficient arrays. Because each decomposition level
    has a different number of coefficients, shorter arrays are right-padded with
    ``NaN`` to match the length of the finest-level (level 1) detail
    coefficients, producing a rectangular output that can be passed to
    downstream features.

    Args:
        wavelet: Discrete wavelet name (e.g. ``"db4"``, ``"haar"``, ``"sym5"``).
            See ``pywt.wavelist(kind='discrete')`` for all valid options.
            Daubechies-4 (``"db4"``) is a standard choice for EEG analysis.
        level: Decomposition level. ``None`` (default) uses the maximum level
            allowed by the signal length and wavelet filter length.
        mode: Signal extension mode for boundary handling (e.g. ``"symmetric"``,
            ``"periodization"``, ``"zero"``). See ``pywt.Modes`` for options.

    Returns:
        :class:`~cobrabox.Data` with dims ``(*extra_dims, "space", "wavelet_level",
        "coef_index")`` where:

        * ``wavelet_level`` has string coordinates following the ``pywt.wavedec``
          output order: ``["approx", "detail_L", ..., "detail_1"]``
          (coarsest approximation first, then details from coarsest to finest).
        * ``coef_index`` runs from 0 to the length of the finest detail minus 1.
          Shorter levels are NaN-padded on the right.

    Raises:
        ValueError: If ``wavelet`` is not a valid discrete wavelet name, if
            ``level < 1``, or if ``level`` exceeds the maximum possible for
            the signal length.

    Example:
        >>> data = cb.load_dataset("dummy_random")[0]
        >>> dwt = cb.DiscreteWaveletTransform(wavelet="db4", level=4).apply(data)
        >>> dwt.data.dims
        ('space', 'wavelet_level', 'coef_index')
        >>> list(dwt.data.coords["wavelet_level"].values)
        ['approx', 'detail_4', 'detail_3', 'detail_2', 'detail_1']
    """

    _tags: ClassVar[list[str]] = [
        "wavelet",
        "sub-band",
        "dyadic",
        "decomposition",
        "denoising",
        "compression",
        "eeg",
        "io:adds-dimension",
    ]

    wavelet: _DwtWavelet = "db4"
    level: int | None = None
    mode: Literal[
        "zero",
        "constant",
        "symmetric",
        "periodic",
        "smooth",
        "periodization",
        "reflect",
        "antisymmetric",
        "antireflect",
    ] = "symmetric"

    output_type: ClassVar[type[Data] | None] = Data

    def __post_init__(self) -> None:
        if self.level is not None and self.level < 1:
            raise ValueError(f"level must be >= 1, got {self.level}")
        if self.wavelet not in pywt.wavelist(kind="discrete"):  # type: ignore
            raise ValueError(
                f"Unknown discrete wavelet '{self.wavelet}'. "
                "Use pywt.wavelist(kind='discrete') to see valid options."
            )

    def __call__(self, data: SignalData) -> xr.DataArray:
        xr_data = data.data

        extra_dims = [d for d in xr_data.dims if d not in ("space", "time")]
        xr_t = xr_data.transpose(*extra_dims, "space", "time")
        arr = xr_t.values
        n_time = arr.shape[-1]

        max_level = pywt.dwt_max_level(n_time, self.wavelet)
        level = self.level if self.level is not None else max_level
        if level > max_level:
            raise ValueError(
                f"level ({level}) exceeds the maximum decomposition level "
                f"({max_level}) for signal length {n_time} and wavelet '{self.wavelet}'."
            )

        space_coords = xr_data.coords["space"].values
        extra_shape = arr.shape[:-2]
        n_space = arr.shape[-2]
        arr_flat = arr.reshape(-1, n_time)

        coeffs = pywt.wavedec(arr_flat, self.wavelet, mode=self.mode, level=level, axis=-1)
        n_decomp = len(coeffs)
        max_coef_len = max(c.shape[-1] for c in coeffs)

        out_flat = np.full((arr_flat.shape[0], n_decomp, max_coef_len), np.nan, dtype=np.float64)
        for j, c in enumerate(coeffs):
            out_flat[:, j, : c.shape[-1]] = c

        out = out_flat.reshape(*extra_shape, n_space, n_decomp, max_coef_len)

        level_labels = ["approx"] + [f"detail_{level - k}" for k in range(level)]

        extra_coords = {d: xr_data.coords[d].values for d in extra_dims if d in xr_data.coords}
        return xr.DataArray(
            out,
            dims=(*extra_dims, "space", "wavelet_level", "coef_index"),
            coords={**extra_coords, "space": space_coords, "wavelet_level": level_labels},
        )


@functional(DiscreteWaveletTransform)
def discrete_wavelet_transform(
    data: SignalData,
    wavelet: _DwtWavelet = "db4",
    level: int | None = None,
    mode: Literal[
        "zero",
        "constant",
        "symmetric",
        "periodic",
        "smooth",
        "periodization",
        "reflect",
        "antisymmetric",
        "antireflect",
    ] = "symmetric",
) -> Data:
    """Multi-level discrete wavelet decomposition (DWT).

    Uses ``pywt.wavedec`` to decompose the ``time`` axis into one approximation
    and ``level`` detail coefficient arrays. Because each decomposition level
    has a different number of coefficients, shorter arrays are right-padded with
    ``NaN`` to match the length of the finest-level (level 1) detail
    coefficients, producing a rectangular output that can be passed to
    downstream features.

    Args:
        data: The input time-series signal to process, as a
            :class:`~cobrabox.SignalData` (or any :class:`~cobrabox.Data`
            carrying a ``time`` dimension).
        wavelet: Discrete wavelet name (e.g. ``"db4"``, ``"haar"``, ``"sym5"``).
            See ``pywt.wavelist(kind='discrete')`` for all valid options.
            Daubechies-4 (``"db4"``) is a standard choice for EEG analysis.
        level: Decomposition level. ``None`` (default) uses the maximum level
            allowed by the signal length and wavelet filter length.
        mode: Signal extension mode for boundary handling (e.g. ``"symmetric"``,
            ``"periodization"``, ``"zero"``). See ``pywt.Modes`` for options.

    Returns:
        :class:`~cobrabox.Data` with dims ``(*extra_dims, "space", "wavelet_level",
        "coef_index")`` where:

        * ``wavelet_level`` has string coordinates following the ``pywt.wavedec``
          output order: ``["approx", "detail_L", ..., "detail_1"]``
          (coarsest approximation first, then details from coarsest to finest).
        * ``coef_index`` runs from 0 to the length of the finest detail minus 1.
          Shorter levels are NaN-padded on the right.

    Raises:
        ValueError: If ``wavelet`` is not a valid discrete wavelet name, if
            ``level < 1``, or if ``level`` exceeds the maximum possible for
            the signal length.

    Example:
        >>> data = cb.load_dataset("dummy_random")[0]
        >>> dwt = cb.discrete_wavelet_transform(data, wavelet="db4", level=4)
        >>> dwt.data.dims
        ('space', 'wavelet_level', 'coef_index')
        >>> list(dwt.data.coords["wavelet_level"].values)
        ['approx', 'detail_4', 'detail_3', 'detail_2', 'detail_1']
    """
    return DiscreteWaveletTransform(wavelet=wavelet, level=level, mode=mode).apply(data)
