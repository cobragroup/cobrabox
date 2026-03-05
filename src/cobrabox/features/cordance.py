from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import xarray as xr
from scipy.signal import welch

from ..base_feature import BaseFeature
from ..data import Data, SignalData

_DEFAULTS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "gamma": (30.0, 45.0),
}


@dataclass
class Cordance(BaseFeature[SignalData]):
    """Compute cordance, a qEEG measure combining absolute and relative bandpower.

    Cordance (Leuchter et al., 1999) integrates absolute and relative
    spectral power into a single index per channel per frequency band.
    It has been used extensively in pharmacoEEG and treatment-response
    prediction studies (e.g. predicting SSRI response from prefrontal
    theta cordance).

    Algorithm:
        1. Estimate the PSD via Welch's method.
        2. For each frequency band, compute **absolute power** (AP) by
           integrating the PSD over the band's frequency range.
        3. Compute **relative power** (RP) as the ratio of each band's
           absolute power to the total power across all requested bands.
        4. Log-transform AP and RP: ``anorm = log(AP)``, ``rnorm = log(RP)``.
        5. Z-score ``anorm`` and ``rnorm`` across the *space* (channel)
           dimension within each band.
        6. Cordance = ``z(anorm) + z(rnorm)``.

    The z-scoring step requires at least two spatial channels. Values are
    unitless and centred around zero — positive values indicate relatively
    high combined absolute and relative power for that channel/band.

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
            between frequency resolution and variance reduction. Defaults to
            ``min(n_time, 256)`` as chosen by :func:`scipy.signal.welch`.

    Example:
        >>> cord = cb.feature.Cordance().apply(data)
        >>> cord_theta = cb.feature.Cordance(bands={"theta": True}).apply(data)

    Returns:
        xarray DataArray with dims ``(band_index, space)`` (plus a singleton
        ``time`` dimension added by ``BaseFeature.apply``). The ``band_index``
        coordinate holds the band names. Values are unitless cordance scores.

    References:
        Leuchter, A. F., Cook, I. A., Lufkin, R. B., Dunkin, J.,
        Newton, T. F., Cummings, J. L., ... & Walter, D. O. (1994).
        Cordance: a new method for assessment of cerebral perfusion and
        metabolism using quantitative electroencephalography. *NeuroImage*,
        1(3), 208-219.

        Leuchter, A. F., Cook, I. A., Witte, E. A., Morgan, M., &
        Abrams, M. (2002). Changes in brain function of depressed
        subjects during treatment with placebo. *American Journal of
        Psychiatry*, 159(1), 122-129.
    """

    output_type: ClassVar[type[Data]] = Data

    bands: dict[str, list[float] | bool] | None = field(default=None)
    nperseg: int | None = field(default=None)

    def __post_init__(self) -> None:
        if self.nperseg is not None and self.nperseg < 2:
            raise ValueError(f"nperseg must be >= 2, got {self.nperseg}")

    def __call__(self, data: SignalData) -> xr.DataArray:
        xr_data = data.data
        sampling_rate = data.sampling_rate
        if sampling_rate is None:
            raise ValueError(
                "sampling_rate must be set on data to compute cordance; "
                "pass sampling_rate when constructing the Data object"
            )

        if "space" not in xr_data.dims:
            raise ValueError(
                "Cordance requires a 'space' dimension with at least 2 channels "
                "for cross-channel z-scoring"
            )

        n_space = xr_data.sizes["space"]
        if n_space < 2:
            raise ValueError(
                f"Cordance requires at least 2 spatial channels for z-scoring, got {n_space}"
            )

        # ── Resolve bands ────────────────────────────────────────────
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

        # ── PSD via Welch ────────────────────────────────────────────
        # Ensure time is the last axis
        if xr_data.dims[-1] != "time":
            xr_data = xr_data.transpose(..., "time")

        # Identify extra dims (anything that is not space or time)
        extra_dims = [d for d in xr_data.dims if d not in ("space", "time")]
        xr_ordered = xr_data.transpose(*extra_dims, "space", "time")
        values = xr_ordered.values  # shape (*extra, n_space, n_time)

        freqs, psd = welch(values, fs=sampling_rate, nperseg=self.nperseg, axis=-1)
        # psd shape: (*extra, n_space, n_freqs)

        # ── Absolute power per band ─────────────────────────────────
        abs_powers = []  # list of arrays, each (*extra, n_space)
        for f_low, f_high in resolved.values():
            mask = (freqs >= f_low) & (freqs < f_high)
            if not mask.any():
                abs_powers.append(np.zeros(values.shape[:-1]))
            else:
                abs_powers.append(np.trapezoid(psd[..., mask], freqs[mask], axis=-1))

        ap = np.stack(abs_powers, axis=0)  # (n_bands, *extra, n_space)

        # ── Relative power ───────────────────────────────────────────
        total_power = ap.sum(axis=0, keepdims=True)  # (1, *extra, n_space)
        # Guard against division by zero (e.g. flat signal)
        total_power = np.where(total_power == 0, 1.0, total_power)
        rp = ap / total_power  # (n_bands, *extra, n_space)

        # ── Log-transform ────────────────────────────────────────────
        # Clamp to a tiny positive floor to avoid log(0)
        _eps = np.finfo(np.float64).tiny
        anorm = np.log(np.maximum(ap, _eps))
        rnorm = np.log(np.maximum(rp, _eps))

        # ── Z-score across channels (space axis) per band ────────────
        # space is the last axis in our (n_bands, *extra, n_space) layout
        space_axis = -1
        a_mean = anorm.mean(axis=space_axis, keepdims=True)
        a_std = anorm.std(axis=space_axis, keepdims=True, ddof=0)
        a_std = np.where(a_std == 0, 1.0, a_std)
        z_anorm = (anorm - a_mean) / a_std

        r_mean = rnorm.mean(axis=space_axis, keepdims=True)
        r_std = rnorm.std(axis=space_axis, keepdims=True, ddof=0)
        r_std = np.where(r_std == 0, 1.0, r_std)
        z_rnorm = (rnorm - r_mean) / r_std

        # ── Cordance ─────────────────────────────────────────────────
        cordance = z_anorm + z_rnorm  # (n_bands, *extra, n_space)

        # ── Build xarray result ──────────────────────────────────────
        dims = ("band_index", *extra_dims, "space")
        coords: dict[str, np.ndarray] = {"band_index": np.array(list(resolved.keys()))}
        if "space" in xr_data.coords:
            coords["space"] = xr_data.coords["space"].values
        for d in extra_dims:
            if d in xr_data.coords:
                coords[d] = xr_data.coords[d].values

        return xr.DataArray(cordance, dims=dims, coords=coords)
