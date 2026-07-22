from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import ClassVar

import xarray as xr

from ..base_feature import AggregatorFeature
from ..data import Data


@dataclass
class ConcatAggregate(AggregatorFeature):
    """Aggregate a stream of per-window Data by stacking along a new 'window' dimension.

    Collects all windows and concatenates them along a new 'window' dimension,
    preserving the per-window results rather than reducing them. Per-window
    pipeline history is propagated to the result.

    When the windows carry ``window_start``/``window_end`` in ``extra`` — as
    those produced by :class:`SlidingWindow` do — the ``window`` coordinate is
    labelled with each window's **start time** on the original time axis rather
    than a bare index, and a non-dimension ``window_end`` coordinate carries the
    matching end times. This lets you plot against real time and locate an event
    ("the seizure starts at 4.2 s") on the window axis. Splitters that do not
    report window positions fall back to an integer index from 0.

    Returns:
        A new ``Data`` object with an additional leading ``window`` dimension.
        All metadata from the original ``data`` is preserved, including
        ``sampling_rate``. History includes all per-window operations followed
        by ``"ConcatAggregate"``.

    Example:
        >>> chord = (
        ...     cb.SlidingWindow(window_size=100, step_size=50)
        ...     | cb.LineLength()
        ...     | cb.ConcatAggregate()
        ... )
        >>> result = chord.apply(data)
        >>> result.data.dims  # ('window', ...)
        >>> result.data.window.values[:3]  # 100 Hz data, step of 50 samples
        array([0. , 0.5, 1. ])
    """

    _tags: ClassVar[list[str]] = ["aggregation", "io:preserves-time"]

    def __call__(self, data: Data, stream: Iterator[Data]) -> Data:
        items = list(stream)
        if not items:
            raise ValueError("ConcatAggregate received an empty stream")
        stacked = xr.concat([w.data for w in items], dim="window")

        # Label the window axis with real time when the splitter reported window positions
        # (SlidingWindow does); otherwise fall back to a plain integer index.
        starts = [w.extra.get("window_start") for w in items]
        ends = [w.extra.get("window_end") for w in items]
        if all(s is not None for s in starts):
            stacked = stacked.assign_coords(window=starts)
            if all(e is not None for e in ends):
                stacked = stacked.assign_coords(window_end=("window", ends))
        else:
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
