from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Concatenate, ParamSpec, TypeAlias

import xarray as xr

from .data import Data

P = ParamSpec("P")

# Type alias for valid feature return types
FeatureReturn: TypeAlias = xr.DataArray | Data
# Type aliased for valid feature function signature
FeatureFunction: TypeAlias = Callable[Concatenate[Data, P], FeatureReturn]


def feature(feature_func: FeatureFunction) -> Callable[Concatenate[Data, P], Data]:
    """Decorator that handles Data repack for features."""
    feature_name = feature_func.__name__

    @wraps(feature_func)
    def wrapped(data: Data, *args: P.args, **kwargs: P.kwargs) -> Data:
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
