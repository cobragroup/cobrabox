from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Protocol

import xarray as xr

from .data import Data

# Type alias for valid feature return types
FeatureReturn = xr.DataArray | Data


# Protocol defining the feature function interface
class FeatureFunction(Protocol):
    """Protocol for feature functions that can be decorated with @feature."""

    def __call__(self, data: Data, *args: object, **kwargs: object) -> FeatureReturn:
        """Feature function signature: takes Data, returns DataArray or Data."""
        ...


def feature(feature_func: FeatureFunction) -> Callable[[Data, ...], Data]:
    """Decorator that handles Data repack for features."""
    feature_name = feature_func.__name__

    @wraps(feature_func)
    def wrapped(data: Data, *args: object, **kwargs: object) -> Data:
        result = feature_func(data, *args, **kwargs)
        if not isinstance(result, (xr.DataArray, Data)):
            raise TypeError(
                f"Feature function '{feature_name}' must return xarray.DataArray or Data, "
                f"got {type(result)}"
            )
        return data._copy_with_new_data(new_data=result, operation_name=feature_name)

    # Marker used by dynamic feature discovery.
    wrapped._is_cobrabox_feature = True

    return wrapped
