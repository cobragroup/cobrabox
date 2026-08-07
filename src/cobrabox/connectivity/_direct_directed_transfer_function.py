from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import xarray as xr

from .._functional import functional
from ..base_feature import BaseFeature
from ..data import Data, SignalData
from ._mvar import ddtf_from_A, fit_mvar, frequency_response


@dataclass
class DirectDirectedTransferFunction(BaseFeature[SignalData]):
    """Estimate the direct Directed Transfer Function (dDTF) between channels.

    The dDTF combines the standard DTF with partial coherence to isolate
    **direct** causal connections, filtering out indirect paths that the DTF
    would otherwise include.

    For each frequency bin, ``dDTF[i, j, f]`` represents the normalized direct
    directional influence **from channel j to channel i**. Values lie in
    ``[0, 1]``.

    Args:
        var_order: Number of lags for the VAR model. ``None`` (default) lets
            statsmodels pick the optimal order via AIC.
        n_freqs: Number of frequency bins in ``[0, sr/2]``. Default 128.

    Returns:
        :class:`~cobrabox.Data` with dims ``("space_to", "space_from", "frequency")``.

    Raises:
        ValueError: If ``data.sampling_rate`` is ``None``, the input is not 2-D,
            there are fewer than 2 channels, or ``var_order < 1`` /
            ``n_freqs < 1``.

    References:
        Korzeniewska, A., Mańczak, M., Kamiński, M., Blinowska, K. J., &
        Kasicki, S. (2003). Determination of information flow direction among
        brain structures by a modified directed transfer function (dDTF) method.
        Journal of Neuroscience Methods, 125(1-2), 195-207.
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
            raise ValueError(
                "DirectDirectedTransferFunction requires data.sampling_rate to be set."
            )

        xr_data = data.data
        if xr_data.ndim != 2:
            raise ValueError(
                f"DirectDirectedTransferFunction requires 2-D input (space x time), "
                f"got shape {xr_data.shape} with dims {xr_data.dims}."
            )
        n_ch, _ = xr_data.shape
        if n_ch < 2:
            raise ValueError(
                f"DirectDirectedTransferFunction requires at least 2 channels, got {n_ch}."
            )

        coefs, resid_cov, _ = fit_mvar(xr_data.values, order=self.var_order)
        A_f, freqs = frequency_response(coefs, self.n_freqs, float(data.sampling_rate))
        ddtf = ddtf_from_A(A_f, resid_cov).transpose(1, 2, 0)  # (K, K, n_freqs)

        if "space" in xr_data.coords:
            space_vals = xr_data.coords["space"].values
        else:
            space_vals = np.arange(n_ch)

        return xr.DataArray(
            ddtf,
            dims=["space_to", "space_from", "frequency"],
            coords={"space_to": space_vals, "space_from": space_vals, "frequency": freqs},
        )


@functional(DirectDirectedTransferFunction)
def direct_directed_transfer_function(
    data: SignalData, var_order: int | None = None, n_freqs: int = 128
) -> Data:
    """Estimate the direct Directed Transfer Function (dDTF) between channels.

    The dDTF combines the standard DTF with partial coherence to isolate
    **direct** causal connections, filtering out indirect paths that the DTF
    would otherwise include.

    For each frequency bin, ``dDTF[i, j, f]`` represents the normalized direct
    directional influence **from channel j to channel i**. Values lie in
    ``[0, 1]``.

    Args:
        data: The input time-series signal to process, as a
            :class:`~cobrabox.SignalData` (or any :class:`~cobrabox.Data`
            carrying a ``time`` dimension).
        var_order: Number of lags for the VAR model. ``None`` (default) lets
            statsmodels pick the optimal order via AIC.
        n_freqs: Number of frequency bins in ``[0, sr/2]``. Default 128.

    Returns:
        :class:`~cobrabox.Data` with dims ``("space_to", "space_from", "frequency")``.

    Raises:
        ValueError: If ``data.sampling_rate`` is ``None``, the input is not 2-D,
            there are fewer than 2 channels, or ``var_order < 1`` /
            ``n_freqs < 1``.

    References:
        Korzeniewska, A., Mańczak, M., Kamiński, M., Blinowska, K. J., &
        Kasicki, S. (2003). Determination of information flow direction among
        brain structures by a modified directed transfer function (dDTF) method.
        Journal of Neuroscience Methods, 125(1-2), 195-207.
    """
    return DirectDirectedTransferFunction(var_order=var_order, n_freqs=n_freqs).apply(data)
