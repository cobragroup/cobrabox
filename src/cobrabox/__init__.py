# Domain subpackages, plus the flat `feature` convenience namespace
from . import (
    connectivity,
    decompositions,
    feature,
    infometrics,
    serialization,
    signalstats,
    spectral,
    surrogates,
    transforms,
    windowing,
)
from .base_feature import AggregatorFeature, BaseFeature, Chord, Pipeline, SplitterFeature
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

# Hardcoded re-exports of key feature classes — must come before `feature` import
# so the domain modules can be imported via normal Python machinery first.
from .infometrics.nonreversibility import Nonreversibility
from .infometrics.recurrence_matrix import RecurrenceMatrix
from .serialization import deserialize, load, save, serialize
from .signalstats.line_length import LineLength
from .windowing.concat_aggregate import ConcatAggregate
from .windowing.mean_aggregate import MeanAggregate
from .windowing.sliding_window import SlidingWindow

# Package-level aliases for class methods
from_numpy = Data.from_numpy
from_xarray = Data.from_xarray

__all__ = [
    "EEG",
    "FMRI",
    "AggregatorFeature",
    "BaseFeature",
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
    "connectivity",
    "dataset_info",
    "decompositions",
    "delete_dataset",
    "deserialize",
    "download_dataset",
    "feature",
    "from_numpy",
    "from_xarray",
    "get_dataset_dir",
    "gorkastyle",
    "infometrics",
    "list_datasets",
    "load",
    "load_dataset",
    "save",
    "serialization",
    "serialize",
    "set_dataset_dir",
    "show_datasets",
    "signalstats",
    "spectral",
    "surrogates",
    "transforms",
    "windowing",
]
