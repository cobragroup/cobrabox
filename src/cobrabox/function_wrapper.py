from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, Concatenate, ParamSpec, TypeAlias, cast

import numpy as np
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
        if data.windowed:
            window_length = data.window_length
            window_stride = data.window_stride
            if window_length is None or window_stride is None:
                raise ValueError(
                    "window_length and window_stride must be set when windowed is True"
                )
            xr_data = data.data

            n_time = xr_data.sizes["time"]
            n_windows = (n_time - window_length) // window_stride + 1
            if n_windows <= 0:
                raise ValueError(f"window length ({window_length}) must be <= n_time ({n_time})")

            window_starts = np.arange(0, n_time - window_length + 1, window_stride)
            window_starts = window_starts[:n_windows]

            results: list[xr.DataArray] = []
            for start in window_starts:
                end = start + window_length
                window = xr_data.isel(time=slice(start, end))

                window_data = Data(
                    data=window,
                    sampling_rate=data.sampling_rate,
                    subjectID=data.subjectID,
                    groupID=data.groupID,
                    condition=data.condition,
                    history=data.history,
                )

                result = feature_func(window_data, *args, **kwargs)
                if not isinstance(result, (xr.DataArray, Data)):
                    raise TypeError(
                        f"Feature function '{feature_name}' must return xarray.DataArray or Data, "
                        f"got {type(result)}"
                    )

                if isinstance(result, Data):
                    result_data = result.data
                else:
                    result_data = result

                results.append(result_data)

            windowed_result = xr.concat(results, dim="time", join="inner")
            windowed_result = windowed_result.assign_coords(
                time=xr_data.coords["time"].values[window_starts]
            )

            return data._copy_with_new_data(new_data=windowed_result, operation_name=feature_name)
        result = feature_func(data, *args, **kwargs)
        if not isinstance(result, (xr.DataArray, Data)):
            raise TypeError(
                f"Feature function '{feature_name}' must return xarray.DataArray or Data, "
                f"got {type(result)}"
            )
        return data._copy_with_new_data(new_data=result, operation_name=feature_name)

    # Marker used by dynamic feature discovery.
    cast(Any, wrapped)._is_cobrabox_feature = True

    return wrapped
