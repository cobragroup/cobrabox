from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import xarray as xr

from ..base_feature import BaseFeature
from ..data import Data, SignalData
from .bandpower import Bandpower


def _zscore(arr: xr.DataArray, dim: str) -> xr.DataArray:
    """Z-score normalize along a dimension."""
    mean = arr.mean(dim=dim)
    std = arr.std(dim=dim)
    # Avoid division by zero when std=0 (identical values)
    std = xr.where(std == 0, 1.0, std)
    return (arr - mean) / std


@dataclass
class Cordance(BaseFeature[SignalData]):
    """Compute cordance, a qEEG measure combining absolute and relative bandpower.

    Cordance (Leuchter et al., 1999) integrates absolute and relative
    spectral power into a single index per channel per frequency band.
    It has been used extensively in pharmacoEEG and treatment-response
    prediction studies (e.g. predicting SSRI response from prefrontal
    theta cordance).

    Algorithm:
        1. Compute per-band **absolute power** (AP) via
           :class:`~cobrabox.features.bandpower.Bandpower` (Welch's method).
        2. Compute **relative power** (RP) as the ratio of each band's
           absolute power to the total power across all requested bands.
        3. Log-transform AP and RP: ``anorm = log(AP)``, ``rnorm = log(RP)``.
        4. Z-score ``anorm`` and ``rnorm`` across the *space* (channel)
           dimension within each band.
        5. Cordance = ``z(anorm) + z(rnorm)``.

    The z-scoring step requires at least two spatial channels. Values are
    unitless and centred around zero - positive values indicate relatively
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

        nan_on_zero: If ``True``, channels with zero total bandpower output
            ``NaN`` instead of raising an error. Useful for batch processing
            where some channels may be silent. Defaults to ``False``.

    Example:
        >>> cord = cb.feature.Cordance().apply(data)
        >>> cord_theta = cb.feature.Cordance(bands={"theta": True}).apply(data)

    Returns:
        xarray DataArray with dims ``(band, space)``. The ``band``
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

    bands: dict[str, list[float] | bool] | None = None
    nperseg: int | None = None
    nan_on_zero: bool = False

    def __post_init__(self) -> None:
        if self.nperseg is not None and self.nperseg < 2:
            raise ValueError(f"nperseg must be >= 2, got {self.nperseg}")

    def __call__(self, data: SignalData) -> xr.DataArray:
        if "space" not in data.data.dims:
            raise ValueError(
                "Cordance requires a 'space' dimension with at least 2 channels "
                "for cross-channel z-scoring"
            )

        n_space = data.data.sizes["space"]
        if n_space < 2:
            raise ValueError(
                f"Cordance requires at least 2 spatial channels for z-scoring, got {n_space}"
            )

        # ── Absolute power via Bandpower ─────────────────────────────
        ap = Bandpower(bands=self.bands, nperseg=self.nperseg)(data)

        # ── Relative power ───────────────────────────────────────────
        total = ap.sum(dim="band_index")
        zero_mask = total == 0
        if zero_mask.any():
            if not self.nan_on_zero:
                raise ValueError(
                    "Total bandpower is zero for one or more channels. "
                    "Cordance requires non-zero spectral power (check for zero signals). "
                    "Set nan_on_zero=True to output NaN for these channels instead."
                )
            # Replace zeros with 1 to avoid division error; we'll mask with NaN later
            total = xr.where(zero_mask, 1.0, total)
        rp = ap / total

        # ── Log-transform ────────────────────────────────────────────
        eps = np.finfo(np.float64).tiny
        anorm = xr.apply_ufunc(np.log, xr.apply_ufunc(np.maximum, ap, eps))
        rnorm = xr.apply_ufunc(np.log, xr.apply_ufunc(np.maximum, rp, eps))

        # ── Z-score across space per band ────────────────────────────
        z_anorm = _zscore(anorm, dim="space")
        z_rnorm = _zscore(rnorm, dim="space")

        # ── Cordance ─────────────────────────────────────────────────
        cordance = z_anorm + z_rnorm

        # Mask zero-power channels with NaN if requested
        if self.nan_on_zero and zero_mask.any():
            cordance = xr.where(zero_mask, np.nan, cordance)

        # Rename band_index → band
        return cordance.rename({"band_index": "band"})
