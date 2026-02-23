from __future__ import annotations

from functools import wraps
from typing import Callable, Protocol

import xarray as xr

from .data import Dataset

# Type alias for valid feature return types
FeatureReturn = xr.DataArray | Dataset


# Protocol defining the feature function interface
class FeatureFunction(Protocol):
    """Protocol for feature functions that can be decorated with @feature."""

    def __call__(self, data: Dataset, *args, **kwargs) -> FeatureReturn:
        """Feature function signature: takes Dataset, returns DataArray or Dataset."""
        ...


def feature(feature_func: FeatureFunction) -> Callable[[Dataset, ...], Dataset]:
    """Decorator that handles Dataset repack for features."""
    feature_name = feature_func.__name__

    @wraps(feature_func)
    def wrapped(data: Dataset, *args, **kwargs) -> Dataset:
        result = feature_func(data, *args, **kwargs)
        if not isinstance(result, (xr.DataArray, Dataset)):
            raise TypeError(
                f"Feature function '{feature_name}' must return xarray.DataArray or Dataset, "
                f"got {type(result)}"
            )
        return data._copy_with_new_data(
            new_data=result,
            operation_name=feature_name,
        )

    return wrapped
