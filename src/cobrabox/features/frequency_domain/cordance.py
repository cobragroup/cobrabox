from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
import xarray as xr

from ...base_feature import BaseFeature
from ...data import Data, SignalData
from .bandpower import Bandpower


@dataclass
class Cordance(BaseFeature[SignalData]):
    """Compute cordance, a qEEG measure combining absolute and relative bandpower.

    Cordance (Leuchter et al., 1994) integrates absolute and relative
    spectral power into a single index per channel per frequency band.

    Algorithm (Leuchter 1994 threshold-based):
        1. Compute per-band **absolute power** (AP) via
           :class:`~cobrabox.features.bandpower.Bandpower` (Welch's method).
        2. Compute **relative power** (RP) as the ratio of each band's
           absolute power to the total power across all requested bands.
        3. Normalize AP and RP by dividing by the **maximum** across channels:
           ``Anorm = AP / max(AP)``, ``Rnorm = RP / max(RP)``.
        4. Classify each channel using a **threshold of 0.5**:
           - **Concordant**: both Anorm > 0.5 and Rnorm > 0.5
           - **Discordant**: Anorm < 0.5 and Rnorm > 0.5
        5. Compute cordance scores as deviation from the threshold:
           - Concordance = ``(Anorm - 0.5) + (Rnorm - 0.5)``
           - Discordance = ``(0.5 - Anorm) + (Rnorm - 0.5)``

    By default, this returns a combined "cordance" value where concordant
    channels get positive scores and discordant channels get negative scores.
    Use the ``output`` parameter to get concordance or discordance maps
    separately.

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

        threshold: Threshold for classifying concordant vs discordant channels.
            Default is 0.5 (50% of normalized maximum) as in the original
            Leuchter 1994 algorithm.

        output: What to return:
            - ``"cordance"`` (default): Combined score where concordant channels
              are positive and discordant channels are negative. Channels that
              are neither concordant nor discordant return 0.
            - ``"concordance"``: Only concordance scores (0 for non-concordant).
            - ``"discordance"``: Only discordance scores (0 for non-discordant).

        nan_on_zero: If ``True``, channels with zero total bandpower output
            ``NaN`` instead of raising an error. Useful for batch processing
            where some channels may be silent. Defaults to ``False``.

    Example:
        >>> cord = cb.feature.Cordance().apply(data)
        >>> cord_theta = cb.feature.Cordance(bands={"theta": True}).apply(data)
        >>> disc = cb.feature.Cordance(output="discordance").apply(data)

    Returns:
        xarray DataArray with dims ``(band_index, space)``. The ``band_index``
        coordinate holds the band names. Values are unitless cordance scores:
        positive for concordant channels, negative for discordant channels.

    References:
        Leuchter, A. F., Cook, I. A., Lufkin, R. B., Dunkin, J.,
        Newton, T. F., Cummings, J. L., ... & Walter, D. O. (1994).
        Cordance: a new method for assessment of cerebral perfusion and
        metabolism using quantitative electroencephalography. *NeuroImage*,
        1(3), 208-219.

        US Patent 5,309,923 (1994). Method and apparatus for determining
        brain activity including the nature of brain lesions by
        electroencephalography.
    """

    output_type: ClassVar[type[Data]] = Data

    bands: dict[str, list[float] | bool] | None = None
    nperseg: int | None = None
    threshold: float = 0.5
    output: Literal["cordance", "concordance", "discordance"] = "cordance"
    nan_on_zero: bool = False

    def __post_init__(self) -> None:
        if self.nperseg is not None and self.nperseg < 2:
            raise ValueError(f"nperseg must be >= 2, got {self.nperseg}")
        if not 0 < self.threshold < 1:
            raise ValueError(f"threshold must be in (0, 1), got {self.threshold}")
        if self.output not in ("cordance", "concordance", "discordance"):
            raise ValueError(
                f"output must be 'cordance', 'concordance', or 'discordance', got {self.output!r}"
            )

    def __call__(self, data: SignalData) -> xr.DataArray:
        if "space" not in data.data.dims:
            raise ValueError(
                "Cordance requires a 'space' dimension with at least 2 channels "
                "for cross-channel normalization"
            )

        n_space = data.data.sizes["space"]
        if n_space < 2:
            raise ValueError(
                f"Cordance requires at least 2 spatial channels for normalization, got {n_space}"
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

        # ── Normalize by maximum across space per band ───────────────
        # aratio = AP / max(AP), rratio = RP / max(RP)
        ap_max = ap.max(dim="space")
        rp_max = rp.max(dim="space")

        # Avoid division by zero (shouldn't happen if data has variation)
        ap_max = xr.where(ap_max == 0, 1.0, ap_max)
        rp_max = xr.where(rp_max == 0, 1.0, rp_max)

        anorm = ap / ap_max
        rnorm = rp / rp_max

        # ── Classification and scoring ───────────────────────────────
        t = self.threshold

        # Concordant: both Anorm > t and Rnorm > t
        # Score: (Anorm - t) + (Rnorm - t)
        concordant_mask = (anorm > t) & (rnorm > t)
        concordance_score = (anorm - t) + (rnorm - t)
        concordance = xr.where(concordant_mask, concordance_score, 0.0)

        # Discordant: Anorm < t and Rnorm > t
        # Score: (t - Anorm) + (Rnorm - t)
        discordant_mask = (anorm < t) & (rnorm > t)
        discordance_score = (t - anorm) + (rnorm - t)
        discordance = xr.where(discordant_mask, discordance_score, 0.0)

        # ── Select output ────────────────────────────────────────────
        if self.output == "concordance":
            result = concordance
        elif self.output == "discordance":
            result = discordance
        else:  # "cordance"
            # Combined: positive for concordant, negative for discordant
            result = concordance - discordance

        # Mask zero-power channels with NaN if requested
        if self.nan_on_zero and zero_mask.any():
            result = xr.where(zero_mask, np.nan, result)

        return result
