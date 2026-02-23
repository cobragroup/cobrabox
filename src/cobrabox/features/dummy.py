from __future__ import annotations

import numpy as np
import xarray as xr

from ..data import Dataset
from .wrapper import feature


@feature
def dummy(
    data: Dataset,
) -> Dataset:
    """Dummy feature that returns the input data."""
    a = data.asnumpy()
    sampling_rate = data.sampling_rate

    
    return data
