from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import xarray as xr

from ..base_feature import BaseFeature
from ..data import Data, SignalData


@dataclass
class MutualInformation(BaseFeature[SignalData]):
    """
    Compute mutual information (MI) between all pairs of series along a specified dimension (by
    default 'time').

    Mutual information is a measure of the mutual dependence between two variables. It is defined
    as MI(X, Y) = H(X) + H(Y) - H(X, Y), where H(X) is the entropy of X and H(X, Y) is the joint
    entropy of X and Y. The entropy is computed by discretizing the data into bins and applying the
    formula H(X) = -sum(p_i * log(p_i)), where p_i is the empirical probability of the i-th bin.

    The number of bins can be specified by the user or determined heuristically based on the size
    of the data along the specified dimension as $n^{1/3}$. The bins can be equiprobable (i.e., each
    bin has the same number of samples) or equidistant (i.e., each bin has the same width).

    Args:
        dim: The dimension along which to compute MI. Default is "time".
        other_dim: The other dimension to consider when computing MI. This is mandatory when the
            input data has more than two dimensions. Default is None.
        bins: The number of bins to use for discretizing the data. If None, it will be determined
            heuristically based on the size of the data along the specified dimension. Default
            is None.
        equiprobable_bins: Whether to use equiprobable bins (True) or equidistant bins (False)
            for discretizing the data. Default is True.
        log_base: The logarithm base to use for computing MI. Default is 2.0.

    Returns:
        An xarray.DataArray with dimensions "space_from" and "space_to" containing the pairwise
        mutual information matrix. The shape is (..., n_space, n_space) where n_space is the size
        of the dimension specified by `other_dim`.

    Raises:
        ValueError: If bins is not a positive integer.
        ValueError: If dim is not a string.
        ValueError: If other_dim is not a string or None.
        ValueError: If dim is not found in the data.
        ValueError: If other_dim is not found in the data.
        ValueError: If other_dim must be specified for data with more than 2 dimensions.

    References:
        Shannon, C. E. (1948). A mathematical theory of communication.
        Bell System Technical Journal, 27(3), 379-423.

    Example:
        >>> import cobrabox as cb
        >>> import numpy as np
        >>> data = cb.SignalData.from_numpy(
        ...     np.random.randn(10, 100),
        ...     dims=["space", "time"],
        ...     sampling_rate=100.0
        ... )
        >>> result = cb.feature.MutualInformation().apply(data)
        >>> result.data.shape
        (10, 10)

    """

    dim: str = "time"
    other_dim: str | None = None
    bins: int | None = None
    equiprobable_bins: bool = True
    log_base: float = 2.0

    output_type: ClassVar[type[Data] | None] = Data

    def __post_init__(self) -> None:
        if self.bins is not None:
            if self.bins <= 0:
                raise ValueError("bins must be positive")
            if not isinstance(self.bins, int):
                raise ValueError("bins must be an integer")
        if not isinstance(self.dim, str):
            raise ValueError("dim must be a string")
        if self.other_dim is not None and not isinstance(self.other_dim, str):
            raise ValueError("other_dim must be a string or None")

    def __call__(self, data: SignalData) -> xr.DataArray:
        if self.dim not in data.data.dims:
            raise ValueError(f"Dimension '{self.dim}' not found in data")

        if self.other_dim is None:
            if data.data.ndim == 2:
                self.other_dim = next(
                    d for d in data.data.dims if d != self.dim and isinstance(d, str)
                )
            else:
                raise ValueError(
                    "self.other_dim must be specified for data with more than 2 dimensions"
                )
        if self.other_dim not in data.data.dims:
            raise ValueError(f"Dimension '{self.other_dim}' not found in data")

        n_bins = int(data.data[self.dim].size ** (1 / 3)) if self.bins is None else self.bins

        return self._get_MI(data.data, n_bins) / np.log(self.log_base)

    def _get_binned(self, x: np.ndarray, n_bins: int) -> np.ndarray:
        first_edge = x.min() - 1e-5
        last_edge = x.max() + 1e-5
        bin_edges = np.linspace(first_edge, last_edge, n_bins + 1, endpoint=True, dtype=np.float64)
        norm_denom = last_edge - first_edge
        indices = np.floor((x - first_edge) / norm_denom * n_bins).astype(int)
        indices[indices == n_bins] -= 1
        decrement = x < bin_edges[indices]
        indices[decrement] -= 1
        # The last bin includes the right edge. The other bins do not.
        increment = (x >= bin_edges[indices + 1]) & (indices != n_bins - 1)
        indices[increment] += 1
        return indices

    def _vector_entropy(self, counts: np.ndarray) -> float:
        probs = counts[counts > 0] / np.sum(counts)
        return -np.sum(probs * np.log(probs))

    def _get_MI(self, x: xr.DataArray, n_bins: int) -> xr.DataArray:
        x = x.transpose(..., self.other_dim, self.dim)
        data = np.reshape(x.data, [-1, x.shape[-1]])
        if self.equiprobable_bins:
            data = np.argsort(np.argsort(data, axis=-1), axis=-1)
        tmp = np.zeros_like(data, dtype=np.int64)
        ent = np.zeros([data.shape[0]], dtype=np.float64)
        for i in range(data.shape[0]):
            tmp[i] = self._get_binned(data[i], n_bins)
            ent[i] = self._vector_entropy(np.bincount(tmp[i]).astype(np.int64))

        tmp = np.reshape(tmp, [-1, x.shape[-2], x.shape[-1]])
        ent = np.reshape(ent, [-1, x.shape[-2]])
        MI = np.zeros([tmp.shape[0], tmp.shape[-2], tmp.shape[-2]], dtype=np.float64)
        for i in range(tmp.shape[0]):
            for j in range(tmp.shape[1]):
                for k in range(j + 1, tmp.shape[1]):
                    MI[i, j, k] = (
                        ent[i, j]
                        + ent[i, k]
                        - self._vector_entropy(np.bincount(tmp[i, j] * n_bins + tmp[i, k]))
                    )
            MI[i] += MI[i].T

        return xr.DataArray(
            np.reshape(MI, [*x.shape[:-2], x.shape[-2], x.shape[-2]]),
            dims=[*x.dims[:-2], "space_from", "space_to"],
        )
