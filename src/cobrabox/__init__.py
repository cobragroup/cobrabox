# Make feature module accessible as cb.feature
from . import feature  # noqa: F401

from .data import Data, EEG, FMRI
from .datasets import dataset
from .feature import sliding_window, line_length

# Package-level aliases for class methods
from_numpy = Data.from_numpy
from_xarray = Data.from_xarray

__all__ = [
    "Data",
    "EEG",
    "FMRI",
    "dataset",
    "from_numpy",
    "from_xarray",
    "sliding_window",
    "line_length",
]
