from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import xarray as xr

from ..base_feature import AggregatorFeature
from ..data import Data


@dataclass
class ConcatAggregate(AggregatorFeature):
    """Aggregate a stream of per-window Data by stacking along a new 'window' dimension.

    Collects all windows and concatenates them along an integer-indexed 'window'
    dimension, preserving the per-window results rather than reducing them.
    Per-window pipeline history is propagated to the result.

    Returns:
        A new ``Data`` object with an additional leading ``window`` dimension
        (integer-indexed from 0). All metadata from the original ``data`` is
        preserved, including ``sampling_rate``. History includes all per-window
        operations followed by ``"ConcatAggregate"``.

    Example:
        >>> chord = (
        ...     cb.feature.SlidingWindow(window_size=100, step_size=50)
        ...     | cb.feature.LineLength()
        ...     | cb.feature.ConcatAggregate()
        ... )
        >>> result = chord.apply(data)
        >>> result.data.dims  # ('window', ...)
    """

    def __call__(self, data: Data, stream: Iterator[Data]) -> Data:
        items = list(stream)
        if not items:
            raise ValueError("ConcatAggregate received an empty stream")
        stacked = xr.concat([w.data for w in items], dim="window")
        stacked = stacked.assign_coords(window=list(range(len(items))))
        window_history = [op for op in items[0].history if op not in data.history]
        return Data(
            data=stacked,
            subjectID=data.subjectID,
            groupID=data.groupID,
            condition=data.condition,
            sampling_rate=data.sampling_rate,
            history=list(data.history) + window_history + ["ConcatAggregate"],
            extra=data.extra,
        )
