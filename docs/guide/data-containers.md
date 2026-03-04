# Data Containers

CobraBox provides a hierarchy of data container classes for different use cases.

## Class Hierarchy

```
Data (general, no dimension requirements)
└── SignalData (requires 'time' dimension)
    ├── EEG   # EEG data (type marker)
    └── FMRI  # fMRI data (type marker)
```

| Class | Requirements | Use Case |
|-------|--------------|----------|
| `Data` | None | General multidimensional data |
| `SignalData` | Must have 'time' dimension | Time-series analysis (EEG, fMRI) |
| `EEG` | Must have 'time' dimension | EEG-specific data |
| `FMRI` | Must have 'time' dimension | fMRI-specific data |

## General Data Container

### Creating Data Objects

#### From NumPy Arrays

```python
import cobrabox as cb
import numpy as np

# 2D data with arbitrary dimensions
arr = np.random.normal(size=(100, 4))
data = cb.Data.from_numpy(
    arr=arr,
    dims=["time", "space"],
    sampling_rate=100.0,
    subjectID="sub-01",
    groupID="control",
    condition="baseline"
)

# 1D data (no time dimension)
arr_1d = np.random.normal(size=(50,))
data_1d = cb.Data.from_numpy(
    arr=arr_1d,
    dims=["channel"],
    subjectID="sub-01"
)
# data_1d.sampling_rate is None (no time dimension)
```

**Requirements:**

- `dims` length must match array `ndim`
- No mandatory dimensions

#### From xarray DataArray

```python
import xarray as xr

# General Data with any dimensions
xr_data = xr.DataArray(
    np.random.normal(size=(100, 4)),
    dims=["x", "y"],
    coords={"x": range(100), "y": ["A", "B", "C", "D"]}
)

data = cb.Data.from_xarray(xr_data, subjectID="sub-01")
```

## SignalData (Time-Series Container)

`SignalData` is for time-series data and requires a 'time' dimension. It automatically transposes data to put time last for performance.

### Creating SignalData Objects

```python
import cobrabox as cb
import numpy as np

# Create time-series data
arr = np.random.normal(size=(1000, 64))  # time x channels
data = cb.SignalData.from_numpy(
    arr=arr,
    dims=["time", "space"],
    sampling_rate=256.0,
    subjectID="sub-01",
    condition="task"
)

# Time dimension is automatically moved to last position
print(data.data.dims)  # ('space', 'time')
print(data.data.shape)  # (64, 1000)
```

**Requirements:**

- Must have a 'time' dimension
- `dims` length must match array `ndim`
- Time dimension will be transposed to last position

### EEG and FMRI Subclasses

```python
# EEG data (type marker)
eeg = cb.EEG.from_numpy(
    arr=arr,
    dims=["time", "space"],
    sampling_rate=256.0
)
print(isinstance(eeg, cb.SignalData))  # True
print(isinstance(eeg, cb.Data))        # True
print(type(eeg) == cb.EEG)             # True

# FMRI data (type marker)
fmri = cb.FMRI.from_numpy(
    arr=fmri_arr,
    dims=["time", "spaceX", "spaceY", "spaceZ"],
    sampling_rate=0.5  # TR = 2s
)
```

## Properties

### Core Properties

```python
# All containers share these properties
data.subjectID      # Subject identifier
data.groupID        # Group identifier  
data.condition      # Experimental condition
data.sampling_rate  # Sampling rate in Hz (None if no time dimension)
data.history        # List of applied operations
data.extra          # Custom metadata dict
```

### Data Access

```python
data.data           # Underlying xarray.DataArray
data.to_numpy()     # Convert to numpy array
data.to_pandas()    # Convert to pandas DataFrame
```

## Sampling Rate

### General Data

For `Data` without a time dimension, `sampling_rate` is `None`:

```python
data = cb.Data.from_numpy(arr, dims=["x", "y"])
print(data.sampling_rate)  # None
```

For `Data` with a time dimension, sampling_rate can be provided or inferred:

```python
# Explicit sampling rate
data = cb.Data.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

# Or inferred from time coordinates (if time is in seconds)
coords = {"time": np.linspace(0, 1, 100), "space": ["E1", "E2"]}
xr_data = xr.DataArray(arr, dims=["time", "space"], coords=coords)
data = cb.Data.from_xarray(xr_data)
print(data.sampling_rate)  # ~100.0 Hz (inferred)
```

### SignalData

`SignalData` requires a time dimension, so `sampling_rate` may be provided or inferred, but will never be `None` if inference succeeds:

```python
data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=256.0)
print(data.sampling_rate)  # 256.0
```

## Immutability

All data containers are immutable. Attempting to modify them raises an error:

```python
data = cb.Data.from_numpy(arr, dims=["time", "space"])

# This will fail:
data.subjectID = "sub-02"  # AttributeError!

# Instead, create a new Data object:
new_data = cb.Data.from_xarray(
    data.data,
    subjectID="sub-02",
    history=data.history,
    extra=data.extra
)
```

## Extra Fields

The `extra` dict stores custom metadata:

```python
data = cb.SignalData.from_numpy(
    arr,
    dims=["time", "space"],
    extra={
        "notes": "Filtered 1-40 Hz",
        "bad_channels": ["E12", "E15"]
    }
)

# Access (returns a copy)
extra = data.extra
print(extra["notes"])

# To modify, create a new Data object:
new_extra = {**data.extra, "new_field": "value"}
# (typically done via features that accept extra parameter)
```

## Conversion Methods

### To NumPy

```python
# Default: just data values
arr = data.to_numpy()

# Gorkastyle: (time, space, labels) - requires time and space dimensions
time, space, labels = data.to_numpy(style="gorkastyle")
```

### To Pandas

```python
df = data.to_pandas()
# Returns DataFrame with MultiIndex from dimensions
```

## Type Hints

Full type hints are provided:

```python
from cobrabox import Data, SignalData, EEG, FMRI

def process_general(data: Data) -> Data:
    """Works with any Data container."""
    return cb.feature.Mean(dim="time").apply(data)

def process_timeseries(data: SignalData) -> Data:
    """Requires time-series data."""
    return cb.feature.LineLength().apply(data)

def process_eeg(data: EEG) -> EEG:
    """EEG-specific processing."""
    return cb.feature.LineLength().apply(data)
```

## When to Use Each Container

- **Use `Data`** when you have general multidimensional data without time (e.g., cross-sectional data, images without temporal dimension)
- **Use `SignalData`** when you have time-series data (EEG, fMRI, other time-series)
- **Use `EEG`** or **Use `FMRI`** when you want explicit type markers for those modalities

Features that require time (like `LineLength`, `Bandpower`, `SlidingWindow`) are typed to accept `SignalData`, providing better IDE support and runtime validation.
