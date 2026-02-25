from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


class Data:
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
        return new Data instances (e.g., features create new Data objects).
    """

    __slots__ = ("_data", "_extra", "_frozen")

    def __init__(
        self,
        data: xr.DataArray,
        sampling_rate: float | None = None,
        subjectID: str | None = None,
        groupID: str | None = None,
        condition: str | None = None,
        history: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Initialize Data.

        Args:
            data: xarray DataArray with at least 'time' and 'space' dimensions
            sampling_rate: Sampling rate in Hz. If not provided, inferred from
                time coordinates when they represent time in seconds.
            subjectID: Subject identifier
            groupID: Group identifier
            condition: Experimental condition
            history: List of operation names applied (default: empty list)
            extra: Optional dict for additional fields and arrays (e.g. xr.DataArray, scalars)
        """
        # Validate mandatory dimensions
        if "time" not in data.dims:
            raise ValueError("data must have 'time' dimension")
        if "space" not in data.dims:
            raise ValueError("data must have 'space' dimension")

        if sampling_rate is not None and sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive when provided")

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

        # Sampling rate: use provided value, else try to infer from time coordinates
        if sampling_rate is not None:
            attrs["sampling_rate"] = sampling_rate
        else:
            inferred = self._infer_sampling_rate(data)
            if inferred is not None:
                attrs["sampling_rate"] = inferred

        # Update attrs
        self._data = self._data.assign_attrs(attrs)

        # User-defined extra fields and arrays (frozen copy)
        self._extra = dict(extra) if extra else {}
        self._frozen = True  # Mark as frozen after initialization

    @classmethod
    def from_numpy(
        cls,
        arr: np.ndarray,
        dims: list[str],
        *,
        sampling_rate: float | None = None,
        subjectID: str | None = None,
        groupID: str | None = None,
        condition: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Data:
        """Create a Data object from a numpy array.

        Requires at least 2 dimensions and that resulting dims include `time` and `space`.
        """
        arr = np.asarray(arr)
        if arr.ndim < 2:
            raise ValueError("array must have at least 2 dimensions (time, space)")

        if len(dims) != arr.ndim:
            raise ValueError("dims length must match array ndim")

        if "time" not in dims:
            raise ValueError("dims must include 'time'")

        time_axis = dims.index("time")

        coords: dict[str, Any] = {}
        if sampling_rate is not None and sampling_rate > 0:
            coords["time"] = np.arange(arr.shape[time_axis], dtype=float) / sampling_rate
        else:
            coords["time"] = np.arange(arr.shape[time_axis], dtype=float)

        data = xr.DataArray(arr, dims=dims, coords=coords)
        return cls(
            data=data,
            sampling_rate=sampling_rate,
            subjectID=subjectID,
            groupID=groupID,
            condition=condition,
            extra=extra,
        )

    @classmethod
    def from_xarray(
        cls,
        ar: xr.DataArray,
        *,
        subjectID: str | None = None,
        groupID: str | None = None,
        condition: str | None = None,
        history: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Data:
        """Create a Data object from an xarray DataArray.

        The DataArray must have 'time' and 'space' dimensions. Time coordinates
        should be in seconds if you want sampling_rate to be inferred.

        Args:
            ar: xarray DataArray with dims (time, space) and optional coords.
            subjectID: Subject identifier.
            groupID: Group identifier.
            condition: Experimental condition.
            history: List of operation names applied.
            extra: Optional extra dict.

        Returns:
            Data instance.

        Example:
            >>> ar = xr.DataArray(...)
            >>> ds = cb.from_xarray(ar)
        """
        return cls(
            data=ar,
            subjectID=subjectID,
            groupID=groupID,
            condition=condition,
            history=history,
            extra=extra,
        )

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

        # If time values look like plain indices (0, 1, 2, ...), don't infer sampling rate.
        if np.allclose(time_values, np.arange(len(time_values), dtype=float)):
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

        Returns a copy of the extra dict. To add fields, create a new Data object
        with updated extra dict.
        """
        return self._extra.copy()

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent modification of attributes after initialization."""
        if hasattr(self, "_frozen") and self._frozen:
            raise AttributeError(
                f"Cannot modify attribute '{name}'. Data is immutable. "
                f"Create a new Data instance instead."
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
        new_data: xr.DataArray | Data,
        operation_name: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Data:
        """Create a new Data object with updated data, preserving metadata.

        Internal method used by feature wrapper to create new Data objects.
        Can merge metadata from a returned Data object, only replacing fields that are
        defined in the returned Data object.

        Args:
            new_data: New xarray DataArray or Data object to merge
            operation_name: Name of operation to append to history
            extra: Optional extra dict to merge in

        Returns:
            New Data instance with preserved/merged metadata
        """
        # Handle both DataArray and Data returns
        if isinstance(new_data, Data):
            # Extract DataArray from returned Data
            result_data = new_data.data

            # Merge metadata: use values from returned Data only if they're defined
            # (not None), otherwise keep original values
            merged_subjectID = (
                new_data.subjectID if new_data.subjectID is not None else self.subjectID
            )
            merged_groupID = new_data.groupID if new_data.groupID is not None else self.groupID
            merged_condition = (
                new_data.condition if new_data.condition is not None else self.condition
            )

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
                result_data = result_data.expand_dims("time", axis=0).assign_coords(
                    time=[time_delta]
                )
            else:
                # Fallback: use a small time value that suggests 100 Hz
                result_attrs = dict(result_data.attrs) if result_data.attrs else {}
                result_attrs["sampling_rate"] = 100.0
                result_data = result_data.assign_attrs(result_attrs)
                result_data = result_data.expand_dims("time", axis=0).assign_coords(time=[0.01])

        return Data(
            data=result_data,
            subjectID=merged_subjectID,
            groupID=merged_groupID,
            condition=merged_condition,
            history=merged_history,
            extra=merged_extra,
        )


class EEG(Data):
    """EEG data container."""


class FMRI(Data):
    """fMRI data container."""
