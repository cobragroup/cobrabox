from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from ..base_feature import SplitterFeature
from ..data import Data


@dataclass
class SlidingWindow(SplitterFeature):
    """Yield one Data per sliding window over the time dimension.

    Lazily generates windows to avoid materialising all windows in memory at once.

    Args:
        window_size: Number of timepoints per window.
        step_size: Step between window starts in timepoints.

    Example:
        >>> chord = Chord(
        ...     split=SlidingWindow(window_size=100, step_size=50),
        ...     pipeline=LineLength(),
        ...     aggregate=MeanAggregate(),
        ... )
    """

    window_size: int = field(default=10)
    step_size: int = field(default=5)

    def __call__(self, data: Data) -> Iterator[Data]:
        xr_data = data.data

        if "time" not in xr_data.dims:
            raise ValueError("data must have 'time' dimension")

        n_time = xr_data.sizes["time"]
        n_windows = (n_time - self.window_size) // self.step_size + 1
        if n_windows <= 0:
            raise ValueError(f"window_size ({self.window_size}) must be <= n_time ({n_time})")

        window_starts = np.arange(0, n_time - self.window_size + 1, self.step_size)
        window_starts = window_starts[:n_windows]

        for start in window_starts:
            end = start + self.window_size
            window_data = xr_data.isel(time=slice(start, end))
            yield data._copy_with_new_data(new_data=window_data, operation_name="SlidingWindow")
