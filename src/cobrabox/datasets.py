from __future__ import annotations

import numpy as np
import xarray as xr

from .data import Dataset


def dataset(identifier: str) -> Dataset:
    """Create or load a Dataset instance.
    
    For now, this is a placeholder that creates a dummy dataset.
    Future: can load from files, databases, etc.
    
    Args:
        identifier: Dataset identifier (e.g., "fMRI_sim2")
    
    Returns:
        Dataset instance
    
    Example:
        >>> data = cb.dataset("fMRI_sim2")
    """
    # TODO: Implement actual dataset loading
    # For now, create a dummy dataset for testing
    # Create dummy data: 100 timepoints, 10 channels
    n_time = 100
    n_space = 10
    dummy_data = np.random.randn(n_time, n_space)
    
    # Create xarray DataArray with proper dimensions
    data_array = xr.DataArray(
        dummy_data,
        dims=["time", "space"],
        coords={
            "time": np.arange(n_time),
            "space": [f"ch{i}" for i in range(n_space)],
        },
        attrs={"identifier": identifier},
    )
    
    # Create time coordinates in seconds (for 100 Hz sampling rate)
    time_coords = np.arange(n_time) / 100.0  # 0.01s intervals = 100 Hz
    
    data_array = data_array.assign_coords(time=time_coords)
    
    return Dataset(
        data=data_array,
        subjectID=f"subj_{identifier}",
        groupID="group1",
        condition="baseline",
    )
