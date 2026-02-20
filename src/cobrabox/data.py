from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


class Dataset:
    """Container for labelled multidimensional time-series data.
    
    Immutable container wrapping xarray.DataArray with mandatory dimensions
    (time, space) and metadata attributes for EEG/fMRI analysis.
    
    Mandatory dimensions:
        - time: Temporal dimension
        - space: Spatial dimension (electrode/voxel)
    
    Optional dimensions:
        - spaceX, spaceY, spaceZ: Additional spatial dimensions
        - run_index: Run/block index
        - window_index: Window index (e.g., from sliding window)
        - band_index: Frequency band index
    
    Metadata attributes:
        - subjectID: Subject identifier
        - sampling_rate: Sampling rate in Hz (inferred from time coordinates)
        - groupID: Group identifier
        - condition: Experimental condition
        - history: List of operations applied (automatically maintained)
        - extra: User-defined dict for additional fields and arrays (any values)
    
    Note:
        This class is immutable. To create modified versions, use methods that
        return new Dataset instances (e.g., features create new Datasets).
    """
    
    __slots__ = ("_data", "_extra", "_frozen")

    def __init__(
        self,
        data: xr.DataArray,
        subjectID: str | None = None,
        groupID: str | None = None,
        condition: str | None = None,
        history: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ):
        """Initialize Dataset.
        
        Args:
            data: xarray DataArray with at least 'time' and 'space' dimensions
            subjectID: Subject identifier
            groupID: Group identifier
            condition: Experimental condition
            history: List of operation names applied (default: empty list)
            extra: Optional dict for additional fields and arrays (e.g. xr.DataArray, scalars)
        
        Note:
            sampling_rate is inferred from the time coordinates if they represent
            time in seconds. If time coordinates are indices (0, 1, 2, ...), sampling_rate
            will be None.
        """
        # Validate mandatory dimensions
        if "time" not in data.dims:
            raise ValueError("data must have 'time' dimension")
        if "space" not in data.dims:
            raise ValueError("data must have 'space' dimension")
        
        # Store xarray DataArray
        self._data = data
        
        # Store metadata in xarray attrs for persistence
        attrs = dict(data.attrs) if data.attrs else {}
        
        if subjectID is not None:
            attrs["subjectID"] = subjectID
        if groupID is not None:
            attrs["groupID"] = groupID
        if condition is not None:
            attrs["condition"] = condition
        
        # Initialize history if not provided
        if history is None:
            history = []
        attrs["history"] = history
        
        # Infer and store sampling rate from time coordinates when possible
        inferred_sampling_rate = self._infer_sampling_rate(data)
        if inferred_sampling_rate is not None:
            attrs["sampling_rate"] = inferred_sampling_rate
        
        # Update attrs
        self._data = self._data.assign_attrs(attrs)
        
        # User-defined extra fields and arrays (frozen copy)
        self._extra = dict(extra) if extra else {}
        self._frozen = True  # Mark as frozen after initialization
    
    @property
    def data(self) -> xr.DataArray:
        """Access underlying xarray DataArray."""
        return self._data
    
    @property
    def subjectID(self) -> str | None:
        """Subject identifier."""
        return self._data.attrs.get("subjectID")
    
    def _infer_sampling_rate(self, data: xr.DataArray) -> float | None:
        """Infer sampling rate from time coordinates.
        
        Returns sampling rate in Hz if time coordinates represent time in seconds
        and are evenly spaced. Returns None if time coordinates are indices or
        cannot be inferred.
        """
        # Check if already stored in attrs (for backwards compatibility)
        if "sampling_rate" in data.attrs:
            return data.attrs.get("sampling_rate")
        
        # Try to infer from time coordinates
        time_coords = data.coords["time"]
        
        # Need at least 2 points to compute delta
        if len(time_coords) < 2:
            return None
        
        # Convert to numpy array for computation
        time_values = np.asarray(time_coords)
        
        # Check if evenly spaced (within tolerance)
        deltas = np.diff(time_values)
        if len(deltas) == 0:
            return None
        
        # Check if deltas are approximately equal (within 1% tolerance)
        mean_delta = np.mean(deltas)
        if mean_delta <= 0:
            return None
        
        # If deltas vary significantly, time might not be evenly sampled
        if np.std(deltas) / mean_delta > 0.01:
            return None
        
        # If time values look like indices (0, 1, 2, ...), can't infer sampling rate
        # Assume if max time < 10, they're likely indices
        if np.max(time_values) < 10 and np.allclose(time_values, np.arange(len(time_values))):
            return None
        
        # Compute sampling rate: 1 / delta_time (assuming time is in seconds)
        sampling_rate = 1.0 / mean_delta
        
        return float(sampling_rate)
    
    @property
    def sampling_rate(self) -> float | None:
        """Sampling rate in Hz (stored at creation, inferred from time coords when possible)."""
        return self._data.attrs.get("sampling_rate")
    
    @property
    def groupID(self) -> str | None:
        """Group identifier."""
        return self._data.attrs.get("groupID")
    
    @property
    def condition(self) -> str | None:
        """Experimental condition."""
        return self._data.attrs.get("condition")
    
    @property
    def history(self) -> list[str]:
        """List of operations applied to this dataset."""
        return self._data.attrs.get("history", [])
    
    @property
    def extra(self) -> dict[str, Any]:
        """User-defined additional fields and arrays.
        
        Returns a copy of the extra dict. To add fields, create a new Dataset
        with updated extra dict.
        """
        return self._extra.copy()
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent modification of attributes after initialization."""
        if hasattr(self, "_frozen") and self._frozen:
            raise AttributeError(
                f"Cannot modify attribute '{name}'. Dataset is immutable. "
                f"Create a new Dataset instance instead."
            )
        super().__setattr__(name, value)
    
    def asnumpy(self) -> np.ndarray:
        """Convert to numpy array.
        
        Returns:
            numpy array with same shape as underlying DataArray
        """
        return self._data.values
    
    def aspandas(self) -> pd.DataFrame:
        """Convert to pandas DataFrame.
        
        Returns:
            pandas DataFrame with MultiIndex from dimensions
        """
        return self._data.to_pandas()
    
    def _copy_with_new_data(
        self,
        new_data: xr.DataArray | Dataset,
        operation_name: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Dataset:
        """Create a new Dataset with updated data, preserving metadata.
        
        Internal method used by feature wrapper to create new Dataset instances.
        Can merge metadata from a returned Dataset, only replacing fields that are
        defined in the returned Dataset.
        
        Args:
            new_data: New xarray DataArray or Dataset to merge
            operation_name: Name of operation to append to history
            extra: Optional extra dict to merge in
        
        Returns:
            New Dataset instance with preserved/merged metadata
        """
        # Handle both DataArray and Dataset returns
        if isinstance(new_data, Dataset):
            # Extract DataArray from returned Dataset
            result_data = new_data.data
            
            # Merge metadata: use values from returned Dataset only if they're defined
            # (not None), otherwise keep original values
            merged_subjectID = new_data.subjectID if new_data.subjectID is not None else self.subjectID
            merged_groupID = new_data.groupID if new_data.groupID is not None else self.groupID
            merged_condition = new_data.condition if new_data.condition is not None else self.condition
            
            # Merge history: combine both histories, then append operation name
            merged_history = list(self.history) + list(new_data.history)
            if operation_name:
                merged_history.append(operation_name)
            
            # Merge extra: combine dicts (later overrides)
            merged_extra = {**self._extra, **new_data._extra}
            if extra is not None:
                merged_extra.update(extra)
        else:
            # Original behavior: just DataArray
            result_data = new_data
            
            # Preserve all metadata from self
            merged_subjectID = self.subjectID
            merged_groupID = self.groupID
            merged_condition = self.condition
            
            # Update history
            merged_history = list(self.history)
            if operation_name:
                merged_history.append(operation_name)
            
            # Preserve extra
            if extra is None:
                merged_extra = dict(self._extra)
            else:
                merged_extra = {**self._extra, **extra}
        
        # Ensure result has time dimension (add singleton if missing)
        if "time" not in result_data.dims:
            # Preserve original sampling rate in attrs since we can't infer from singleton
            original_sampling_rate = self.sampling_rate
            if original_sampling_rate is not None:
                # Store sampling_rate in attrs so it doesn't need to be inferred
                result_attrs = dict(result_data.attrs) if result_data.attrs else {}
                result_attrs["sampling_rate"] = original_sampling_rate
                result_data = result_data.assign_attrs(result_attrs)
                # Use proper time coordinate for consistency
                time_delta = 1.0 / original_sampling_rate
                result_data = result_data.expand_dims("time", axis=0).assign_coords(time=[time_delta])
            else:
                # Fallback: use a small time value that suggests 100 Hz
                result_attrs = dict(result_data.attrs) if result_data.attrs else {}
                result_attrs["sampling_rate"] = 100.0
                result_data = result_data.assign_attrs(result_attrs)
                result_data = result_data.expand_dims("time", axis=0).assign_coords(time=[0.01])
        
        return Dataset(
            data=result_data,
            subjectID=merged_subjectID,
            groupID=merged_groupID,
            condition=merged_condition,
            history=merged_history,
            extra=merged_extra,
        )
