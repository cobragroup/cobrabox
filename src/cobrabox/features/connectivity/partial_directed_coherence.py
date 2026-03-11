from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import xarray as xr
from statsmodels.tsa.api import VAR

from ...base_feature import BaseFeature
from ...data import Data, SignalData


@dataclass
class PartialDirectedCoherence(BaseFeature[SignalData]):
    """Estimate the Partial Directed Coherence (PDC) between channels via a VAR model.

    Fits a multivariate autoregressive (VAR) model to the input time series using
    :class:`statsmodels.tsa.vector_ar.var_model.VAR`, then derives the PDC spectrum
    from the estimated coefficient matrices.

    For each frequency bin, ``PDC[i, j, f]`` represents the normalized directional
    influence **from channel j to channel i**. Values lie in ``[0, 1]`` and sum to 1
    across the ``space_to`` (sink) dimension for each source channel and frequency bin::

        sum_i PDC[i, j, f]^2 = 1  for all j, f

    Args:
        var_order: Number of lags for the VAR model. If ``None`` (default), the
            optimal order is selected automatically using the AIC criterion.
        n_freqs: Number of frequency bins in ``[0, sr/2]`` (Nyquist). Default 128.

    Returns:
        xarray DataArray with dims ``("space_to", "space_from", "frequency")``, shape
        ``(n_channels, n_channels, n_freqs)``. The ``frequency`` coordinate runs
        from 0 Hz to the Nyquist frequency. Values are real-valued in ``[0, 1]``.

    Raises:
        ValueError: If ``data.sampling_rate`` is ``None``.
        ValueError: If the input has fewer than 2 channels.
        ValueError: If the input is not 2-D (space x time).
        ValueError: If ``var_order < 1`` or ``n_freqs < 1``.

    References:
        Baccalá, L. A., & Sameshima, K. (2001). Partial directed coherence:
        a new concept in neural structure determination. Biological Cybernetics,
        84(6), 463-474. https://doi.org/10.1007/PL00007990

    Example:
        >>> pdc = cb.feature.PartialDirectedCoherence().apply(signal_data)
        >>> pdc.data.dims  # ('space_to', 'space_from', 'frequency')
        >>> pdc.data.shape  # (n_ch, n_ch, 128)
    """

    var_order: int | None = None
    n_freqs: int = 128

    output_type: ClassVar[type[Data]] = Data

    def __post_init__(self) -> None:
        if self.var_order is not None and self.var_order < 1:
            raise ValueError(f"var_order must be a positive integer, got {self.var_order}.")
        if self.n_freqs < 1:
            raise ValueError(f"n_freqs must be a positive integer, got {self.n_freqs}.")

    def __call__(self, data: SignalData) -> xr.DataArray:
        if data.sampling_rate is None:
            raise ValueError("PartialDirectedCoherence requires data.sampling_rate to be set.")

        xr_data = data.data
        dims = xr_data.dims

        if xr_data.ndim != 2:
            raise ValueError(
                f"PartialDirectedCoherence requires 2-D input (space x time), "
                f"got shape {xr_data.shape} with dims {dims}."
            )

        # SignalData always puts time last — shape is (K, T)
        n_ch, _n_times = xr_data.shape

        if n_ch < 2:
            raise ValueError(f"PartialDirectedCoherence requires at least 2 channels, got {n_ch}.")

        sr = data.sampling_rate
        values = xr_data.values  # (K, T)

        # --- Fit VAR model ---
        # statsmodels VAR expects (T, K)
        var_model = VAR(values.T)
        if self.var_order is None:
            fit = var_model.fit(ic="aic")
        else:
            fit = var_model.fit(maxlags=self.var_order)

        coefs = fit.coefs  # (p, K, K): coefs[k, i, j] = A_{k+1}[i, j]
        p = coefs.shape[0]

        # --- Build A(f) = I - sum_{k=1}^{p} A_k * exp(-2pi*i*k*f/sr) ---
        freqs = np.linspace(0.0, sr / 2.0, self.n_freqs)  # (n_freqs,)

        # A shape: (n_freqs, K, K)
        A = np.zeros((self.n_freqs, n_ch, n_ch), dtype=complex)
        np.einsum("fii->fi", A)[:] = 1.0  # identity diagonal

        for k in range(1, p + 1):
            phase = np.exp(-2j * np.pi * k * freqs / sr)  # (n_freqs,)
            # coefs[k-1]: (K, K); phase: (n_freqs,) → broadcast to (n_freqs, K, K)
            A -= coefs[k - 1][np.newaxis, :, :] * phase[:, np.newaxis, np.newaxis]

        # --- PDC[i, j, f] = |A[f, i, j]| / sqrt(sum_m |A[f, m, j]|^2) ---
        # Column-normalise over i (space / sink dimension)
        A_abs = np.abs(A)  # (n_freqs, K, K)
        col_norm = np.sqrt((A_abs**2).sum(axis=1, keepdims=True))  # (n_freqs, 1, K)
        # Avoid division by zero (shouldn't happen for non-degenerate VAR)
        col_norm = np.where(col_norm == 0.0, 1.0, col_norm)
        pdc = A_abs / col_norm  # (n_freqs, K, K)

        # Reorder to (K, K, n_freqs) → dims (space_to, space_from, frequency)
        pdc = pdc.transpose(1, 2, 0)  # (K, K, n_freqs)

        # --- Build coordinates ---
        if "space" in xr_data.coords:
            space_vals = xr_data.coords["space"].values
        else:
            space_vals = np.arange(n_ch)

        return xr.DataArray(
            pdc,
            dims=["space_to", "space_from", "frequency"],
            coords={"space_to": space_vals, "space_from": space_vals, "frequency": freqs},
        )
