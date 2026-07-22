# Windowing features: segmentation and aggregation over time.
from ._concat_aggregate import ConcatAggregate
from ._mean_aggregate import MeanAggregate
from ._sliding_window import SlidingWindow
from ._sliding_window_reduce import SlidingWindowReduce

__all__ = ["ConcatAggregate", "MeanAggregate", "SlidingWindow", "SlidingWindowReduce"]

# One-shot functional wrappers (cb.line_length(d) beside cb.LineLength().apply(d)). See GH #116.
from .._functional import install as _install_functional

_install_functional(__name__)
