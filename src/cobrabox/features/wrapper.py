from __future__ import annotations

from functools import wraps
from typing import Callable, Protocol

import xarray as xr

from ..data import Dataset

# Type alias for valid feature return types
FeatureReturn = xr.DataArray | Dataset

# Protocol defining the feature function interface
class FeatureFunction(Protocol):
    """Protocol for feature functions that can be decorated with @feature."""
    
    def __call__(self, data: Dataset, *args, **kwargs) -> FeatureReturn:
        """Feature function signature: takes Dataset, returns DataArray or Dataset."""
        ...


def feature(feature_func: FeatureFunction) -> Callable[[Dataset, ...], Dataset]:
    """Decorator that handles Dataset repack for features.
    
    Enforces that feature functions return either xarray.DataArray or Dataset
    (via FeatureFunction Protocol). Runtime validation ensures correct return types.
    
    This wrapper:
    1. Calls the feature function with the Dataset
    2. Repacks the result into a new Dataset preserving all metadata
    3. Updates history with the feature name
    
    Features must return either:
    - xarray.DataArray: Will be wrapped in a new Dataset with preserved metadata
    - Dataset: Will be merged with original Dataset, only replacing fields that
      are defined in the returned Dataset
    
    Args:
        feature_func: Feature function conforming to FeatureFunction protocol.
            Must take Dataset as first arg and return FeatureReturn
            (xr.DataArray | Dataset)
    
    Returns:
        Wrapped function that takes Dataset and returns Dataset
    
    Raises:
        TypeError: If feature function returns something other than DataArray or Dataset
    
    Example:
        @feature
        def my_feature(data: Dataset) -> xr.DataArray:
            return data.data.mean(dim="time")
        
        @feature
        def my_feature_with_metadata(data: Dataset) -> Dataset:
            result = data.data.mean(dim="time")
            return Dataset(data=result, condition="processed")
        
        # Usage:
        result = my_feature(dataset)  # dataset is Dataset, result is Dataset
    """
    # Get feature name from function name
    feature_name = feature_func.__name__
    
    @wraps(feature_func)
    def wrapped(data: Dataset, *args, **kwargs) -> Dataset:
        """Wrapped feature function that handles Dataset repack."""
        # Call feature function with Dataset
        result = feature_func(data, *args, **kwargs)
        
        # Ensure result is either DataArray or Dataset
        if not isinstance(result, (xr.DataArray, Dataset)):
            raise TypeError(
                f"Feature function '{feature_name}' must return xarray.DataArray or Dataset, "
                f"got {type(result)}"
            )
        
        # Repack: create/merge Dataset with preserved metadata
        new_dataset = data._copy_with_new_data(
            new_data=result,
            operation_name=feature_name,
        )
        
        return new_dataset
    
    return wrapped
