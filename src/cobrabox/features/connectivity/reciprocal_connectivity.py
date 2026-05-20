from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import xarray as xr

from ...base_feature import BaseFeature
from ...data import Data, SignalData


@dataclass
class ReciprocalConnectivity(BaseFeature[Data]):
    """Compute per-channel Reciprocal Connectivity (RC) from a directed connectivity measure.

    **Reciprocal Connectivity** quantifies the net directional role of each channel:
    positive values indicate a net *sink* (more influence received than sent), and
    negative values indicate a net *source* (more influence sent than received).

    For each channel ``i``::

        RC[i] = in_strength[i] - out_strength[i]

    where (using PDC notation where ``mat[i, j]`` = flow *from j to i*):

    * ``in_strength[i]``  = mean over j of ``mat[i, j, band]``
    * ``out_strength[i]`` = mean over j of ``mat[j, i, band]``

    **Two usage modes:**

    1. **Time-series input** (``"time"`` dimension present): fits a VAR model and
       computes PDC internally using :class:`PartialDirectedCoherence`.
    2. **Pre-computed matrix input** (``"space_to"`` + ``"space_from"`` dimensions, no
       ``"time"``): uses the provided asymmetric connectivity matrix directly.

    Args:
        connectivity: Connectivity measure for time-series mode. Currently only
            ``"pdc"`` is supported. Ignored for pre-computed matrix input.
        freq_band: Frequency band ``(fmin, fmax)`` in Hz used to average out the
            frequency dimension. If the matrix has no ``"frequency"`` dimension,
            set to ``None``. Default ``(30.0, 80.0)`` Hz.
        var_order: VAR model order for time-series mode. ``None`` selects the
            order automatically via AIC. Passed to :class:`PartialDirectedCoherence`.
        n_freqs: Number of frequency bins for internal PDC computation. Default 128.
        normalize: If ``True``, z-score the connectivity matrix (excluding the
            diagonal) before computing strengths.

    Returns:
        xarray DataArray with dim ``("space",)``, shape ``(n_channels,)``, containing
        the Reciprocal Connectivity score for each channel.

    Raises:
        ValueError: If time-series input is provided but ``connectivity != "pdc"``.
        ValueError: If pre-computed matrix is symmetric (directional RC requires
            asymmetry).
        ValueError: If ``freq_band`` is set but the matrix has no ``"frequency"``
            dimension.
        ValueError: If ``freq_band`` is outside the available frequency range.
        ValueError: If ``freq_band=None`` but matrix has a ``"frequency"`` dimension.

    Example:
        >>> # From time-series SignalData
        >>> rc = cb.feature.ReciprocalConnectivity(freq_band=(30.0, 80.0)).apply(signal_data)
        >>> rc.data.dims  # ('space',)

        >>> # From pre-computed PDC matrix (no freq dim)
        >>> rc = cb.feature.ReciprocalConnectivity(freq_band=None).apply(pdc_matrix)
    """

    connectivity: str = "pdc"
    freq_band: tuple[float, float] | None = field(default=(30.0, 80.0))
    var_order: int | None = None
    n_freqs: int = 128
    normalize: bool = False

    output_type: ClassVar[type[Data]] = Data

    def __post_init__(self) -> None:
        if self.freq_band is not None:
            fmin, fmax = self.freq_band
            if fmin >= fmax:
                raise ValueError(f"freq_band must satisfy fmin < fmax, got ({fmin}, {fmax}).")

    def __call__(self, data: Data) -> xr.DataArray:
        from .partial_directed_coherence import PartialDirectedCoherence

        xr_data = data.data
        dims = xr_data.dims

        # ------------------------------------------------------------------ #
        # PATH A — time-series input
        # ------------------------------------------------------------------ #
        if "time" in dims:
            if self.connectivity != "pdc":
                raise ValueError(
                    f"Unsupported connectivity measure {self.connectivity!r} for "
                    f"time-series input. Only 'pdc' is currently supported."
                )
            # Validate that data has sampling_rate (required for PDC)
            if not isinstance(data, SignalData) or data.sampling_rate is None:
                raise ValueError(
                    "Time-series input requires data.sampling_rate to be set. "
                    "Use SignalData (e.g., EEG or FMRI) instead of plain Data."
                )
            pdc_feat = PartialDirectedCoherence(var_order=self.var_order, n_freqs=self.n_freqs)
            mat = pdc_feat(data)

        # ------------------------------------------------------------------ #
        # PATH B — pre-computed connectivity matrix
        # ------------------------------------------------------------------ #
        else:
            if "space_to" not in dims or "space_from" not in dims:
                raise ValueError(
                    "Pre-computed input must have 'space_to' and 'space_from' dimensions. "
                    f"Got dims: {dims}."
                )
            mat = xr_data

            # Symmetry check for 2-D matrices (no frequency dim)
            if "frequency" not in mat.dims:
                mat_vals = mat.values
                if mat_vals.ndim == 2 and np.allclose(mat_vals, mat_vals.T, atol=1e-6):
                    raise ValueError(
                        "Pre-computed connectivity matrix appears symmetric. "
                        "Reciprocal Connectivity requires a directed (asymmetric) matrix."
                    )

        # ------------------------------------------------------------------ #
        # Frequency-band averaging
        # ------------------------------------------------------------------ #
        if self.freq_band is not None:
            if "frequency" not in mat.dims:
                raise ValueError(
                    f"freq_band={self.freq_band} is set but the connectivity matrix "
                    "has no 'frequency' dimension. Pass freq_band=None to skip band averaging."
                )
            freqs = mat.coords["frequency"].values
            fmin, fmax = self.freq_band
            if fmin > freqs.max() or fmax < freqs.min():
                raise ValueError(
                    f"freq_band ({fmin}, {fmax}) Hz is outside the available "
                    f"frequency range [{freqs.min():.4g}, {freqs.max():.4g}] Hz."
                )
            mat = mat.sel(frequency=slice(fmin, fmax)).mean("frequency")
        else:
            if "frequency" in mat.dims:
                raise ValueError(
                    "The connectivity matrix has a 'frequency' dimension but "
                    "freq_band=None. Set freq_band to (fmin, fmax) to select a band."
                )

        # mat is now (space_to, space_from) — shape (K, K)
        mat_vals = mat.values.copy().astype(float)
        n_ch = mat_vals.shape[0]

        # Optional z-score normalisation (before diagonal masking)
        if self.normalize:
            mask = ~np.eye(n_ch, dtype=bool)
            off_diag = mat_vals[mask]
            mu, sigma = off_diag.mean(), off_diag.std()
            if sigma > 0:
                mat_vals = (mat_vals - mu) / sigma

        # Mask diagonal with NaN so it does not contribute to strength means
        np.fill_diagonal(mat_vals, np.nan)

        # in_strength[i]  = mean over j of mat[i, j]  (how much flows INTO i)
        in_strength = np.nanmean(mat_vals, axis=1)  # (K,) — average over space_from

        # out_strength[i] = mean over j of mat[j, i]  (how much flows OUT OF i)
        out_strength = np.nanmean(mat_vals, axis=0)  # (K,) — average over space_to

        rc = in_strength - out_strength  # (K,)

        if "space_to" in mat.coords:
            space_vals = mat.coords["space_to"].values
        else:
            space_vals = np.arange(n_ch)

        return xr.DataArray(rc, dims=["space"], coords={"space": space_vals})
