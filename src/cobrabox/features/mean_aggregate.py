from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import xarray as xr

from ..base_feature import AggregatorFeature
from ..data import Data


@dataclass
class MeanAggregate(AggregatorFeature):
    """Aggregate a stream of per-window Data by averaging across windows.

    Collects all windows, stacks them along a temporary 'window' dimension,
    and reduces with mean. Per-window pipeline history is propagated to the result.

    Example:
        >>> chord = Chord(
        ...     split=SlidingWindow(window_size=100, step_size=50),
        ...     pipeline=LineLength(),
        ...     aggregate=MeanAggregate(),
        ... )
    """

    def __call__(self, data: Data, stream: Iterator[Data]) -> Data:
        items = list(stream)
        if not items:
            raise ValueError("MeanAggregate received an empty stream")
        stacked = xr.concat([w.data for w in items], dim="window", join="override")
        averaged = stacked.mean(dim="window")
        # All windows share the same pipeline history — propagate ops not already in data
        window_history = [op for op in items[0].history if op not in data.history]
        return Data(
            data=averaged,
            subjectID=data.subjectID,
            groupID=data.groupID,
            condition=data.condition,
            history=list(data.history) + window_history + ["MeanAggregate"],
            extra=data.extra,
        )
