# Make feature module accessible as cb.feature
from . import feature, serialization  # noqa: F401
from .base_feature import AggregatorFeature, Chord, Pipeline, SplitterFeature
from .data import EEG, FMRI, Data, SignalData
from .dataset import Dataset
from .datasets import dataset, dataset_info, list_datasets
from .egg.gorkastyle import gorkastyle
from .features.time_domain.line_length import LineLength
from .features.time_domain.nonreversibility import Nonreversibility
from .features.time_domain.recurrence_matrix import RecurrenceMatrix
from .features.windowing.concat_aggregate import ConcatAggregate
from .features.windowing.mean_aggregate import MeanAggregate
from .features.windowing.sliding_window import SlidingWindow
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
    "Nonreversibility",
    "Pipeline",
    "RecurrenceMatrix",
    "SignalData",
    "SlidingWindow",
    "SplitterFeature",
    "dataset",
    "dataset_info",
    "deserialize",
    "from_numpy",
    "from_xarray",
    "gorkastyle",
    "list_datasets",
    "load",
    "save",
    "serialization",
    "serialize",
]
