from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import xarray as xr

from ...base_feature import BaseFeature
from ...data import Data


@dataclass
class Covariance(BaseFeature[Data]):
    """Compute pairwise sample covariance between all channel pairs.

    Computes the sample covariance matrix along a chosen dimension (default:
    ``"time"``) producing a symmetric N x N matrix. Input data must be exactly
    2-dimensional: one dimension is the channel axis, the other is the axis
    along which covariance is computed.

    The diagonal contains the sample variance of each channel (``ddof=1``).
    The result is a plain :class:`~cobrabox.Data` object (``output_type = Data``)
    because the covariance axis is consumed.

    Args:
        dim: Name of the dimension to compute covariance along. Defaults to
            ``"time"``. Must be present in the input data. The remaining
            dimension becomes both ``<other_dim>_to`` and ``<other_dim>_from``
            in the output.

    Raises:
        ValueError: If input data is not exactly 2-dimensional.
        ValueError: If ``dim`` is not present in the input data's dimensions.

    Example:
        >>> data = cb.load_dataset("dummy_random")[0]
        >>> cov = cb.feature.Covariance().apply(data)
        >>> cov.data.dims
        ('space_to', 'space_from')

    Returns:
        :class:`~cobrabox.Data` with dims ``(<other_dim>_to, <other_dim>_from)``.
        Both coordinate axes carry the original channel labels. The diagonal
        contains per-channel sample variance; off-diagonal entries are
        covariances. The matrix is symmetric.
    """

    output_type: ClassVar[type[Data]] = Data

    dim: str = field(default="time")

    def __call__(self, data: Data) -> xr.DataArray:
        xr_data = data.data

        if xr_data.ndim != 2:
            raise ValueError(
                f"Covariance requires exactly 2-dimensional input, "
                f"got {xr_data.ndim}D data with dims {xr_data.dims}"
            )

        if self.dim not in xr_data.dims:
            raise ValueError(
                f"dimension {self.dim!r} not found in data dimensions {xr_data.dims}; "
                f"set dim= to the dimension you want to compute covariance along"
            )

        other_dim = next(d for d in xr_data.dims if d != self.dim)

        # Transpose to (other_dim, dim) -> shape (n_channels, n_samples)
        xr_t = xr_data.transpose(other_dim, self.dim)
        arr = xr_t.values  # (n_channels, n_samples)

        cov_matrix = np.cov(arr)  # (n_channels, n_channels), ddof=1 by default

        channel_coords = xr_t.coords[other_dim].values if other_dim in xr_t.coords else None

        coords: dict[str, np.ndarray] = {}
        if channel_coords is not None:
            coords[f"{other_dim}_to"] = channel_coords
            coords[f"{other_dim}_from"] = channel_coords

        return xr.DataArray(
            cov_matrix, dims=[f"{other_dim}_to", f"{other_dim}_from"], coords=coords
        )
