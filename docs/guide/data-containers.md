# Data Containers

CobraBox provides a hierarchy of data container classes for different use cases.

## Class Hierarchy

```text
Data (general, no dimension requirements)
└── SignalData (requires 'time' dimension)
    ├── EEG   # EEG data (type marker)
    └── FMRI  # fMRI data (type marker)
```

| Class        | Requirements               | Use Case                           |
| ------------ | -------------------------- | ---------------------------------- |
| `Data`       | None                       | General multidimensional data      |
| `SignalData` | Must have 'time' dimension | Time-series analysis (EEG, fMRI)   |
| `EEG`        | Must have 'time' dimension | EEG-specific data                  |
| `FMRI`       | Must have 'time' dimension | fMRI-specific data                 |

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

## Dimensions and Coordinates

Every `Data` object wraps an `xarray.DataArray` accessible as `data.data`. The DataArray carries
named dimensions and optional coordinate labels for each axis.

### Inspect dimensions

```python
item = cb.dataset("dummy_chain")[0]

# Dimension names as a list
dims = list(item.data.dims)          # e.g. ['space', 'time']

# Shape keyed by dimension name
sizes = dict(item.data.sizes)        # e.g. {'space': 4, 'time': 200}

# Which coordinates have labels attached?
coords = list(item.data.coords)      # e.g. ['time'] or ['time', 'space']
```

### Get coordinate values as a list

```python
# Space coordinate — returns a numpy array; call .tolist() for a plain list
space_labels = item.data.coords["space"].values.tolist()
# e.g. ['E1', 'E2', 'E3', 'E4']  or  [0, 1, 2, 3] if no labels were set

# Time coordinate in seconds
time_array = item.data.coords["time"].values       # numpy array
time_list  = time_array.tolist()                   # Python list
```

> **Note:** If you created `Data` with `from_numpy()` and did not supply coordinate labels for
> `space`, the space dimension will have no coordinates at all — `"space" not in item.data.coords`.
> To attach labels, build an `xr.DataArray` with explicit `coords` and use `from_xarray()`.

### Attach named coordinates (e.g., electrode labels)

```python
import xarray as xr
import numpy as np

arr = np.random.normal(size=(200, 8))   # 200 time steps, 8 channels
labels = [f"E{i+1}" for i in range(8)]

xr_arr = xr.DataArray(
    arr,
    dims=["time", "space"],
    coords={
        "time": np.arange(200) / 100.0,  # seconds
        "space": labels,
    },
)
data = cb.Data.from_xarray(xr_arr, sampling_rate=100.0, subjectID="sub-01")

# Now space has labels:
data.data.coords["space"].values.tolist()   # ['E1', 'E2', ..., 'E8']
```

### Select by coordinate value

```python
# Single channel
ch = data.data.sel(space="E3")            # xr.DataArray, shape (200,)

# Multiple channels
subset = data.data.sel(space=["E1", "E5"])  # shape (2, 200) after transpose

# Time window (0.5 s – 1.0 s)
window = data.data.sel(time=slice(0.5, 1.0))

# To wrap the result back into a Data object:
data_subset = cb.Data.from_xarray(
    subset.rename({"space": "space"}),   # keep dims intact
    subjectID=data.subjectID,
    sampling_rate=data.sampling_rate,
)
```

### Convert to numpy or pandas

```python
arr = data.to_numpy()          # plain numpy array, shape matches data.data.shape
df  = data.to_pandas()         # pandas DataFrame with MultiIndex from dimensions

# Access specific channels via pandas
df.xs("E1", level="space")    # time-series for channel E1
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
