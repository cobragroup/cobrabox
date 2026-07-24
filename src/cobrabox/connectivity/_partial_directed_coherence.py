from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import xarray as xr

from .._functional import functional
from ..base_feature import BaseFeature
from ..data import Data, SignalData
from ._mvar import fit_mvar, frequency_response, pdc_from_A


@dataclass
class PartialDirectedCoherence(BaseFeature[SignalData]):
    """Estimate the Partial Directed Coherence (PDC) between channels via a VAR model.

    Fits a multivariate autoregressive (VAR) model to the input time series and
    derives the PDC spectrum from the estimated coefficient matrices.

    For each frequency bin, ``PDC[i, j, f]`` represents the normalized directional
    influence **from channel j to channel i**. Values lie in ``[0, 1]`` and sum to 1
    across the ``space_to`` (sink) dimension for each source channel and frequency
    bin: ``sum_i PDC[i, j, f]^2 = 1`` for all ``j, f``.

    Args:
        var_order: Number of lags for the VAR model. ``None`` (default) lets
            :func:`statsmodels.tsa.api.VAR` pick the optimal order via AIC.
        n_freqs: Number of frequency bins in ``[0, sr/2]``. Default 128.

    Returns:
        xarray DataArray with dims ``("space_to", "space_from", "frequency")``.

    Raises:
        ValueError: If ``data.sampling_rate`` is ``None``, the input is not 2-D,
            there are fewer than 2 channels, or ``var_order < 1`` /
            ``n_freqs < 1``.

    References:
        Baccalá, L. A., & Sameshima, K. (2001). Partial directed coherence: a
        new concept in neural structure determination. Biological Cybernetics
        84(6), 463-474.
    """

    _tags: ClassVar[list[str]] = [
        "mvar",
        "var",
        "directed",
        "frequency-domain",
        "direct-causality",
        "epilepsy",
        "seizure-propagation",
        "eeg",
        "io:matrix-frequency",
        "req:multichannel",
        "req:sampling-rate",
    ]

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
        if xr_data.ndim != 2:
            raise ValueError(
                f"PartialDirectedCoherence requires 2-D input (space x time), "
                f"got shape {xr_data.shape} with dims {xr_data.dims}."
            )
        n_ch, _ = xr_data.shape
        if n_ch < 2:
            raise ValueError(f"PartialDirectedCoherence requires at least 2 channels, got {n_ch}.")

        coefs, _, _ = fit_mvar(xr_data.values, order=self.var_order)
        A_f, freqs = frequency_response(coefs, self.n_freqs, float(data.sampling_rate))
        pdc = pdc_from_A(A_f).transpose(1, 2, 0)  # (K, K, n_freqs)

        if "space" in xr_data.coords:
            space_vals = xr_data.coords["space"].values
        else:
            space_vals = np.arange(n_ch)

        return xr.DataArray(
            pdc,
            dims=["space_to", "space_from", "frequency"],
            coords={"space_to": space_vals, "space_from": space_vals, "frequency": freqs},
        )


@functional(PartialDirectedCoherence)
def partial_directed_coherence(
    data: SignalData, var_order: int | None = None, n_freqs: int = 128
) -> Data:
    """Estimate the Partial Directed Coherence (PDC) between channels via a VAR model.

    Fits a multivariate autoregressive (VAR) model to the input time series and
    derives the PDC spectrum from the estimated coefficient matrices.

    For each frequency bin, ``PDC[i, j, f]`` represents the normalized directional
    influence **from channel j to channel i**. Values lie in ``[0, 1]`` and sum to 1
    across the ``space_to`` (sink) dimension for each source channel and frequency
    bin: ``sum_i PDC[i, j, f]^2 = 1`` for all ``j, f``.

    Args:
        var_order: Number of lags for the VAR model. ``None`` (default) lets
            :func:`statsmodels.tsa.api.VAR` pick the optimal order via AIC.
        n_freqs: Number of frequency bins in ``[0, sr/2]``. Default 128.

    Returns:
        xarray DataArray with dims ``("space_to", "space_from", "frequency")``.

    Raises:
        ValueError: If ``data.sampling_rate`` is ``None``, the input is not 2-D,
            there are fewer than 2 channels, or ``var_order < 1`` /
            ``n_freqs < 1``.

    References:
        Baccalá, L. A., & Sameshima, K. (2001). Partial directed coherence: a
        new concept in neural structure determination. Biological Cybernetics
        84(6), 463-474.
    """
    return PartialDirectedCoherence(var_order=var_order, n_freqs=n_freqs).apply(data)
