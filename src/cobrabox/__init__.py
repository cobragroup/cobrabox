# Make feature module accessible as cb.feature
from . import (
    feature,  # noqa: F401
    serialization,
)
from .base_feature import AggregatorFeature, Chord, Pipeline, SplitterFeature
from .data import EEG, FMRI, Data, SignalData
from .dataset import Dataset
from .datasets import dataset
from .egg.gorkastyle import gorkastyle
from .features.concat_aggregate import ConcatAggregate
from .features.line_length import LineLength
from .features.mean_aggregate import MeanAggregate
from .features.sliding_window import SlidingWindow
from .serialization import deserialize, load, save, serialize

# Package-level aliases for class methods
from_numpy = Data.from_numpy
from_xarray = Data.from_xarray

__all__ = [
    "EEG",
    "FMRI",
    "AggregatorFeature",
    "Chord",
    "ConcatAggregate",
    "Data",
    "Dataset",
    "LineLength",
    "MeanAggregate",
    "Pipeline",
    "SignalData",
    "SlidingWindow",
    "SplitterFeature",
    "dataset",
    "deserialize",
    "from_numpy",
    "from_xarray",
    "gorkastyle",
    "load",
    "save",
    "serialization",
    "serialize",
]
