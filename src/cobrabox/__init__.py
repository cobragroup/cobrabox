# Make feature module accessible as cb.feature
from . import feature, serialization  # noqa: F401
from .base_feature import AggregatorFeature, Chord, Pipeline, SplitterFeature
from .data import EEG, FMRI, Data, SignalData
from .dataset import Dataset
from .datasets import (
    dataset_info,
    delete_dataset,
    download_dataset,
    list_datasets,
    load_dataset,
    show_datasets,
)
from .downloader import DownloadCancelled, get_dataset_dir, set_dataset_dir
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
    "DownloadCancelled",
    "LineLength",
    "MeanAggregate",
    "Nonreversibility",
    "Pipeline",
    "RecurrenceMatrix",
    "SignalData",
    "SlidingWindow",
    "SplitterFeature",
    "dataset_info",
    "delete_dataset",
    "deserialize",
    "download_dataset",
    "from_numpy",
    "from_xarray",
    "get_dataset_dir",
    "gorkastyle",
    "list_datasets",
    "load",
    "load_dataset",
    "save",
    "serialization",
    "serialize",
    "set_dataset_dir",
    "show_datasets",
]
