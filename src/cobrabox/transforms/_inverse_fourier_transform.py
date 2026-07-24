from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import xarray as xr

from .._functional import functional
from ..base_feature import BaseFeature
from ..data import Data
from ._fourier_transform import _irfft_1d


@dataclass
class InverseFourierTransform(BaseFeature[Data]):
    """Inverse of :class:`~cobrabox.transforms.fourier_transform.FourierTransform`.

    Takes an array with a ``frequency`` dimension (complex coefficients) and
    returns a time-domain reconstruction.

    Args:
        n: Length of the output time axis. If ``None``, defaults to
            ``2 * (n_freq - 1)``.
        sampling_rate: Sampling rate to attach to the reconstructed time axis.
            When ``None`` the resulting Data has no sampling rate.

    Returns:
        xarray DataArray with the ``frequency`` dim replaced by ``time``.
    """

    _tags: ClassVar[list[str]] = ["ifft", "time-domain", "io:time-output", "req:frequency-input"]

    n: int | None = None
    sampling_rate: float | None = None

    output_type: ClassVar[type[Data]] = Data

    def __call__(self, data: Data) -> xr.DataArray:
        xr_data = data.data
        if "frequency" not in xr_data.dims:
            raise ValueError(
                f"InverseFourierTransform requires a 'frequency' dimension, got {xr_data.dims}"
            )
        ordered = xr_data.transpose(..., "frequency")
        if not np.iscomplexobj(ordered.values):
            raise ValueError(
                "InverseFourierTransform requires complex coefficients; pass "
                "FourierTransform(return_magnitude=False) upstream."
            )
        reconstructed = _irfft_1d(ordered.values, n=self.n, axis=-1)
        n_time = reconstructed.shape[-1]

        non_freq_dims = [d for d in ordered.dims if d != "frequency"]
        coords = {d: ordered.coords[d] for d in non_freq_dims if d in ordered.coords}
        if self.sampling_rate is not None:
            coords["time"] = np.arange(n_time) / float(self.sampling_rate)
        return xr.DataArray(reconstructed, dims=(*non_freq_dims, "time"), coords=coords)


@functional(InverseFourierTransform)
def inverse_fourier_transform(
    data: Data, n: int | None = None, sampling_rate: float | None = None
) -> Data:
    """Inverse of :class:`~cobrabox.transforms.fourier_transform.FourierTransform`.

    Takes an array with a ``frequency`` dimension (complex coefficients) and
    returns a time-domain reconstruction.

    Args:
        n: Length of the output time axis. If ``None``, defaults to
            ``2 * (n_freq - 1)``.
        sampling_rate: Sampling rate to attach to the reconstructed time axis.
            When ``None`` the resulting Data has no sampling rate.

    Returns:
        xarray DataArray with the ``frequency`` dim replaced by ``time``.
    """
    return InverseFourierTransform(n=n, sampling_rate=sampling_rate).apply(data)
