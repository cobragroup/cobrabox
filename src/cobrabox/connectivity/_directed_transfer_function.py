from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import xarray as xr

from .._functional import functional
from ..base_feature import BaseFeature
from ..data import Data, SignalData
from ._mvar import dtf_from_A, fit_mvar, frequency_response


@dataclass
class DirectedTransferFunction(BaseFeature[SignalData]):
    """Estimate the Directed Transfer Function (DTF) between channels via a VAR model.

    Like :class:`PartialDirectedCoherence`, the DTF is derived from a multivariate
    autoregressive fit. Where PDC normalizes the coefficient matrix ``A(f)``
    column-wise (capturing direct causal influence), DTF inverts ``A(f)`` to the
    transfer function ``H(f)`` and normalizes its rows (capturing total causal
    influence, including indirect paths).

    For each frequency bin, ``DTF[i, j, f]`` represents the normalized total
    directional influence **from channel j to channel i**. Values lie in
    ``[0, 1]`` and the rows of the magnitude-squared DTF sum to 1.

    Args:
        var_order: Number of lags for the VAR model. ``None`` (default) lets
            statsmodels pick the optimal order via AIC.
        n_freqs: Number of frequency bins in ``[0, sr/2]``. Default 128.

    Returns:
        xarray DataArray with dims ``("space_to", "space_from", "frequency")``.

    References:
        Kamiński, M., & Blinowska, K. J. (1991). A new method of the description
        of the information flow in the brain structures. Biological Cybernetics
        65(3), 203-210.
    """

    _tags: ClassVar[list[str]] = [
        "mvar",
        "var",
        "directed",
        "frequency-domain",
        "total-causality",
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
            raise ValueError("DirectedTransferFunction requires data.sampling_rate to be set.")

        xr_data = data.data
        if xr_data.ndim != 2:
            raise ValueError(
                f"DirectedTransferFunction requires 2-D input (space x time), "
                f"got shape {xr_data.shape} with dims {xr_data.dims}."
            )
        n_ch, _ = xr_data.shape
        if n_ch < 2:
            raise ValueError(f"DirectedTransferFunction requires at least 2 channels, got {n_ch}.")

        coefs, _, _ = fit_mvar(xr_data.values, order=self.var_order)
        A_f, freqs = frequency_response(coefs, self.n_freqs, float(data.sampling_rate))
        dtf = dtf_from_A(A_f).transpose(1, 2, 0)  # (K, K, n_freqs)

        if "space" in xr_data.coords:
            space_vals = xr_data.coords["space"].values
        else:
            space_vals = np.arange(n_ch)

        return xr.DataArray(
            dtf,
            dims=["space_to", "space_from", "frequency"],
            coords={"space_to": space_vals, "space_from": space_vals, "frequency": freqs},
        )


@functional(DirectedTransferFunction)
def directed_transfer_function(
    data: SignalData, var_order: int | None = None, n_freqs: int = 128
) -> Data:
    """Estimate the Directed Transfer Function (DTF) between channels via a VAR model.

    Like :class:`PartialDirectedCoherence`, the DTF is derived from a multivariate
    autoregressive fit. Where PDC normalizes the coefficient matrix ``A(f)``
    column-wise (capturing direct causal influence), DTF inverts ``A(f)`` to the
    transfer function ``H(f)`` and normalizes its rows (capturing total causal
    influence, including indirect paths).

    For each frequency bin, ``DTF[i, j, f]`` represents the normalized total
    directional influence **from channel j to channel i**. Values lie in
    ``[0, 1]`` and the rows of the magnitude-squared DTF sum to 1.

    Args:
        var_order: Number of lags for the VAR model. ``None`` (default) lets
            statsmodels pick the optimal order via AIC.
        n_freqs: Number of frequency bins in ``[0, sr/2]``. Default 128.

    Returns:
        xarray DataArray with dims ``("space_to", "space_from", "frequency")``.

    References:
        Kamiński, M., & Blinowska, K. J. (1991). A new method of the description
        of the information flow in the brain structures. Biological Cybernetics
        65(3), 203-210.
    """
    return DirectedTransferFunction(var_order=var_order, n_freqs=n_freqs).apply(data)
