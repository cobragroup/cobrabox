from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.signal import welch

from ..data import Data
from ..function_wrapper import feature

_DEFAULTS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "gamma": (30.0, 45.0),
}


@feature
def bandpower(
    data: Data, *, bands: dict[str, list[float] | bool] | None = None, nperseg: int | None = None
) -> xr.DataArray:
    """Compute band power for specified frequency bands using Welch's method.

    For each frequency band, integrates the power spectral density (PSD)
    estimated via Welch's method over the band's frequency range. Operates
    over the ``time`` dimension, returning one scalar power value per
    (space, band) pair.

    Args:
        data: Data with ``time`` and ``space`` dimensions and a valid
            ``sampling_rate``.
        bands: Mapping of band name to frequency range ``[f_low, f_high]``
            in Hz, or ``True`` to use the default range for that band name.
            If ``None`` or empty, all five default bands are computed:

            - ``delta``:  1 - 4 Hz
            - ``theta``:  4 - 8 Hz
            - ``alpha``:  8 - 12 Hz
            - ``beta``:  12 - 30 Hz
            - ``gamma``: 30 - 45 Hz

            Examples::

                bands={"gamma": [45, 80]}           # custom range
                bands={"alpha": True}               # default [8, 12]
                bands={"alpha": True, "ripple": [45, 80]}  # mixed

        nperseg: Number of samples per Welch segment. Controls the trade-off
            between frequency resolution (larger → finer bins) and variance
            reduction (smaller → more segments to average). Defaults to
            ``min(n_time, 256)`` as chosen by :func:`scipy.signal.welch`.

    Returns:
        xarray DataArray with dims ``(band_index, space)`` containing
        absolute power values in units of the input signal squared per Hz.
        The ``@feature`` wrapper appends a singleton ``time`` dimension,
        yielding final dims ``(band_index, space, time)``. To drop it::

            out.data.squeeze("time")

    Raises:
        ValueError: If ``data`` is missing the ``time`` dimension.
        ValueError: If ``data.sampling_rate`` is ``None``.
        ValueError: If a band is specified as ``True`` but its name is not
            in the default band dictionary.

    Example:
        >>> bp = cb.feature.bandpower(data)
        >>> bp_custom = cb.feature.bandpower(data, bands={"alpha": True, "ripple": [45, 80]})
        >>> bp_fine = cb.feature.bandpower(data, nperseg=512)
    """
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
    if not bands:
        resolved: dict[str, tuple[float, float]] = dict(_DEFAULTS)
    else:
        resolved = {}
        for name, spec in bands.items():
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

    freqs, psd = welch(values, fs=sampling_rate, nperseg=nperseg, axis=-1)

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
