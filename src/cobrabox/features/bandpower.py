from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr
from scipy.signal import welch

from ..base_feature import BaseFeature
from ..data import Data

_DEFAULTS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "gamma": (30.0, 45.0),
}


@dataclass
class Bandpower(BaseFeature):
    """Compute band power for specified frequency bands using Welch's method.

    For each frequency band, integrates the power spectral density (PSD)
    estimated via Welch's method over the band's frequency range. Operates
    over the ``time`` dimension, returning one scalar power value per
    (space, band) pair.

    Args:
        bands: Mapping of band name to frequency range ``[f_low, f_high]``
            in Hz, or ``True`` to use the default range for that band name.
            If ``None`` or empty, all five default bands are computed:

            - ``delta``:  1 - 4 Hz
            - ``theta``:  4 - 8 Hz
            - ``alpha``:  8 - 12 Hz
            - ``beta``:  12 - 30 Hz
            - ``gamma``: 30 - 45 Hz

        nperseg: Number of samples per Welch segment. Controls the trade-off
            between frequency resolution (larger → finer bins) and variance
            reduction (smaller → more segments to average). Defaults to
            ``min(n_time, 256)`` as chosen by :func:`scipy.signal.welch`.

    Example:
        >>> bp = cb.feature.Bandpower().apply(data)
        >>> bp_custom = cb.feature.Bandpower(bands={"alpha": True, "ripple": [45, 80]}).apply(data)
        >>> bp_fine = cb.feature.Bandpower(nperseg=512).apply(data)

    Returns:
        xarray DataArray with dims ``(band_index, space)`` (plus a singleton
        ``time`` dimension added by ``BaseFeature.apply``). The ``band_index``
        coordinate holds the band names. Values are absolute power in units of
        the input signal squared per Hz (signal² / Hz).
    """

    bands: dict[str, list[float] | bool] | None = field(default=None)
    nperseg: int | None = field(default=None)

    def __post_init__(self) -> None:
        if self.nperseg is not None and self.nperseg < 2:
            raise ValueError(f"nperseg must be >= 2, got {self.nperseg}")

    def __call__(self, data: Data) -> xr.DataArray:
        xr_data = data.data

        if "time" not in xr_data.dims:
            raise ValueError("data must have 'time' dimension")

        sampling_rate = data.sampling_rate
        if sampling_rate is None:
            raise ValueError(
                "sampling_rate must be set on data to compute bandpower; "
                "pass sampling_rate when constructing the Data object"
            )

        # Resolve bands: None / empty → all defaults
        if not self.bands:
            resolved: dict[str, tuple[float, float]] = dict(_DEFAULTS)
        else:
            resolved = {}
            for name, spec in self.bands.items():
                if spec is True:
                    if name not in _DEFAULTS:
                        raise ValueError(
                            f"Band '{name}' is not a known default band. "
                            f"Known bands: {list(_DEFAULTS)}. "
                            "Provide an explicit [f_low, f_high] range instead."
                        )
                    resolved[name] = _DEFAULTS[name]
                elif isinstance(spec, bool):
                    raise ValueError(
                        f"Band '{name}' value must be True (use default range) or a "
                        "[f_low, f_high] list, not False."
                    )
                else:
                    resolved[name] = (float(spec[0]), float(spec[1]))

        # Data always transposes to (..., time), so time is always the last axis
        values = xr_data.values  # shape (..., time)

        freqs, psd = welch(values, fs=sampling_rate, nperseg=self.nperseg, axis=-1)

        # Integrate power in each band
        band_powers = []
        for f_low, f_high in resolved.values():
            mask = (freqs >= f_low) & (freqs < f_high)
            if not mask.any():
                band_powers.append(np.zeros(values.shape[:-1]))
            else:
                band_powers.append(np.trapezoid(psd[..., mask], freqs[mask], axis=-1))

        result = np.stack(band_powers, axis=0)  # (n_bands, *non_time_dims)

        non_time_dims = [d for d in xr_data.dims if d != "time"]
        da = xr.DataArray(result, dims=["band_index", *non_time_dims])
        return da.assign_coords(
            {"band_index": np.array(list(resolved.keys()))}
            | {d: xr_data.coords[d] for d in non_time_dims if d in xr_data.coords}
        )
