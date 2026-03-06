from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal

import xarray as xr

from ..base_feature import BaseFeature
from ..data import Data, SignalData


@dataclass
class SlidingWindowReduce(BaseFeature[SignalData]):
    """Sliding window with automatic per-window reduction.

    Creates sliding windows over a dimension, applies an aggregation function
    to each window, and stacks results along a new "window" dimension.

    This combines windowing + aggregation into a single feature, avoiding
    the need for a Chord when you just want reduced window statistics.

    Args:
        window_size: Number of samples per window. Must be >= 1.
        step_size: Step between window starts in samples. Must be >= 1.
        dim: Name of the dimension to window over and reduce (default: "time").
        agg: Aggregation function to apply to each window. One of:
            "mean", "std", "sum", "min", "max".

    Returns:
        Data with the reduced dimension removed and a new "window" dimension
        added. The "window" dimension has length equal to the number of
        windows that fit in the input data.

    Example:
        >>> # Mean of each 100-sample window, stepping by 50
        >>> result = cb.feature.SlidingWindowReduce(
        ...     window_size=100, step_size=50, dim="time", agg="mean"
        ... ).apply(data)
        >>> result.data.dims
        ('window', 'channel')  # time dimension is reduced, window dim added
    """

    window_size: int = field(default=10)
    step_size: int = field(default=5)
    dim: str = field(default="time")
    agg: Literal["mean", "std", "sum", "min", "max"] = field(default="mean")

    output_type: ClassVar[type[Data]] = Data

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}")
        if self.step_size < 1:
            raise ValueError(f"step_size must be >= 1, got {self.step_size}")
        valid_aggs = ("mean", "std", "sum", "min", "max")
        if self.agg not in valid_aggs:
            raise ValueError(f"agg must be one of {valid_aggs}, got '{self.agg}'")

    def __call__(self, data: SignalData) -> xr.DataArray:
        xr_data = data.data

        if self.dim not in xr_data.dims:
            raise ValueError(f"dim '{self.dim}' not found in data dimensions {xr_data.dims}")

        n_dim = xr_data.sizes[self.dim]
        if self.window_size > n_dim:
            raise ValueError(
                f"window_size ({self.window_size}) must be <= {self.dim} length ({n_dim})"
            )

        # Use xarray rolling to compute aggregation at each position
        rolling = xr_data.rolling({self.dim: self.window_size}, min_periods=self.window_size)
        agg_func = getattr(rolling, self.agg)
        full_agg = agg_func()

        # Select valid windows (every step_size-th position, starting from window_size-1)
        window_starts = range(self.window_size - 1, n_dim, self.step_size)
        indexed = full_agg.isel({self.dim: list(window_starts)})

        # Rename dimension to 'window'
        return indexed.rename({self.dim: "window"})
