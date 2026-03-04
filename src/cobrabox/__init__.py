# Make feature module accessible as cb.feature
from . import feature  # noqa: F401
from .data import EEG, FMRI, Data
from .datasets import dataset
from .egg.gorkastyle import gorkastyle
from .features import line_length, sliding_window

# Package-level aliases for class methods
from_numpy = Data.from_numpy
from_xarray = Data.from_xarray

__all__ = [
    "EEG",
    "FMRI",
    "Data",
    "dataset",
    "from_numpy",
    "from_xarray",
    "gorkastyle",
    "line_length",
    "sliding_window",
]
