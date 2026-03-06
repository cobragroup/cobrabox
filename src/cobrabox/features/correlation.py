from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import xarray as xr

from ..base_feature import BaseFeature
from ..data import Data


@dataclass
class Correlation(BaseFeature[Data]):
    """Compute pairwise Pearson or Spearman correlation between all channel pairs.

    Correlates channels along a chosen dimension (default: ``"time"``) producing a
    symmetric N x N matrix. Input data must be exactly 2-dimensional: one dimension
    is the channel axis, the other is the axis along which correlation is computed.

    The diagonal (self-correlation) is set to ``1.0``. The result is a plain
    :class:`~cobrabox.Data` object (``output_type = Data``) because the correlation
    axis is consumed.

    Args:
        dim: Name of the dimension to correlate along. Defaults to ``"time"``.
            Must be present in the input data. The remaining dimension becomes
            both ``<other_dim>_to`` and ``<other_dim>_from`` in the output.
        method: Correlation method to use. ``"pearson"`` (default) uses
            :func:`numpy.corrcoef`. ``"spearman"`` converts values to ranks
            along ``dim`` first, then applies :func:`numpy.corrcoef`.

    Raises:
        ValueError: If ``method`` is not ``"pearson"`` or ``"spearman"``.
        ValueError: If input data is not exactly 2-dimensional.
        ValueError: If ``dim`` is not present in the input data's dimensions.

    Example:
        >>> data = cb.dataset("dummy_random")[0]
        >>> corr = cb.feature.Correlation().apply(data)
        >>> corr.data.dims
        ('space_to', 'space_from')

    Returns:
        :class:`~cobrabox.Data` with dims ``(<other_dim>_to, <other_dim>_from)``.
        Both coordinate axes carry the original channel labels. Values are in
        ``[-1, 1]``; the diagonal is ``1.0``. The matrix is symmetric.
    """

    output_type: ClassVar[type[Data]] = Data

    dim: str = field(default="time")
    method: str = field(default="pearson")

    def __post_init__(self) -> None:
        if self.method not in {"pearson", "spearman"}:
            raise ValueError(f"method must be 'pearson' or 'spearman', got {self.method!r}")

    def __call__(self, data: Data) -> xr.DataArray:
        xr_data = data.data

        if xr_data.ndim != 2:
            raise ValueError(
                f"Correlation requires exactly 2-dimensional input, "
                f"got {xr_data.ndim}D data with dims {xr_data.dims}"
            )

        if self.dim not in xr_data.dims:
            raise ValueError(
                f"dimension {self.dim!r} not found in data dimensions {xr_data.dims}; "
                f"set dim= to the dimension you want to correlate along"
            )

        other_dim = next(d for d in xr_data.dims if d != self.dim)

        # Transpose to (other_dim, dim) → shape (n_channels, n_samples)
        xr_t = xr_data.transpose(other_dim, self.dim)
        arr = xr_t.values  # (n_channels, n_samples)

        if self.method == "spearman":
            from scipy.stats import rankdata

            arr = np.apply_along_axis(rankdata, axis=1, arr=arr)

        corr_matrix = np.corrcoef(arr)  # (n_channels, n_channels)
        np.fill_diagonal(corr_matrix, 1.0)  # numerical safety

        channel_coords = xr_t.coords[other_dim].values if other_dim in xr_t.coords else None

        coords: dict[str, np.ndarray] = {}
        if channel_coords is not None:
            coords[f"{other_dim}_to"] = channel_coords
            coords[f"{other_dim}_from"] = channel_coords

        return xr.DataArray(
            corr_matrix, dims=[f"{other_dim}_to", f"{other_dim}_from"], coords=coords
        )
