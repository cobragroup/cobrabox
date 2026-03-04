# Make feature module accessible as cb.feature
from . import feature  # noqa: F401
from .base_feature import AggregatorFeature, Chord, Pipeline, SplitterFeature
from .data import EEG, FMRI, Data
from .datasets import dataset
from .egg.gorkastyle import gorkastyle
from .features.line_length import LineLength
from .features.mean_aggregate import MeanAggregate
from .features.sliding_window import SlidingWindow, coherence

# Package-level aliases for class methods
from_numpy = Data.from_numpy
from_xarray = Data.from_xarray

__all__ = [
    "EEG",
    "FMRI",
    "AggregatorFeature",
    "Chord",
    "Data",
    "LineLength",
    "MeanAggregate",
    "Pipeline",
    "SlidingWindow",
    "SplitterFeature",
    "dataset",
    "from_numpy",
    "from_xarray",
    "gorkastyle",
]
