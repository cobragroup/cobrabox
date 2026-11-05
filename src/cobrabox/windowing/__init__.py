"""Windowing & Aggregation: How do I analyze temporal dynamics?

Tools for applying a feature over time. A splitter such as
``SlidingWindow`` cuts a signal into windows, and an aggregator such as
``MeanAggregate`` or ``ConcatAggregate`` folds the per-window results back
together. ``SlidingWindowReduce`` windows and applies a simple statistic (mean,
std, ...) in one step, with no chord needed.

Everything here is also exported at the top level. Splitters and
``SlidingWindowReduce`` come as a class and a one-shot function
(``cb.SlidingWindow`` / ``cb.sliding_window``). Aggregators are class-only,
since they only make sense at the end of a chord. ``cb.windowing.__all__``
lists them all, and ``help()`` on any of them gives its parameters.

Example:
    >>> chord = (
    ...     cb.SlidingWindow(window_size=250, step_size=125)
    ...     | cb.LineLength()
    ...     | cb.MeanAggregate()
    ... )
    >>> result = chord.apply(data)
"""

from ._concat_aggregate import ConcatAggregate
from ._mean_aggregate import MeanAggregate
from ._sliding_window import SlidingWindow, sliding_window
from ._sliding_window_reduce import SlidingWindowReduce, sliding_window_reduce

__all__ = [
    "ConcatAggregate",
    "MeanAggregate",
    "SlidingWindow",
    "SlidingWindowReduce",
    "sliding_window",
    "sliding_window_reduce",
]
