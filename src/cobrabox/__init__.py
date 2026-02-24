from .data import Dataset
from .datasets import dataset
from .feature import sliding_window, line_length

# Package-level aliases for class methods
from_numpy = Dataset.from_numpy
from_xarray = Dataset.from_xarray

__all__ = [
    "Dataset",
    "dataset",
    "from_numpy",
    "from_xarray",
    "sliding_window",
    "line_length",
]

# Make feature module accessible as cb.feature
from . import feature
