# Data Containers

CobraBox provides three data container classes: `Data`, `EEG`, and `FMRI`.

## Creating Data Objects

### From NumPy Arrays

```python
import cobrabox as cb
import numpy as np

arr = np.random.normal(size=(100, 4))

data = cb.from_numpy(
    arr=arr,
    dims=["time", "space"],
    sampling_rate=100.0,
    subjectID="sub-01",
    groupID="control",
    condition="baseline"
)
```

**Requirements:**

- Array must have at least 2 dimensions
- `dims` must include `"time"` and `"space"`
- `dims` length must match array `ndim`

### From xarray DataArray

```python
import xarray as xr

xr_data = xr.DataArray(
    np.random.normal(size=(100, 4)),
    dims=["time", "space"],
    coords={
        "time": np.linspace(0, 1, 100),
        "space": ["E1", "E2", "E3", "E4"]
    }
)

data = cb.from_xarray(
    xr_data,
    subjectID="sub-01",
    condition="task"
)
```

## Class Hierarchy

```
Data
├── EEG   # EEG data (type marker)
└── FMRI  # fMRI data (type marker)
```

`EEG` and `FMRI` are empty subclasses that serve as type markers:

```python
eeg = EEG.from_numpy(arr, dims=["time", "space"])
fmri = FMRI.from_numpy(arr, dims=["time", "space", "spaceZ"])

print(isinstance(eeg, Data))   # True
print(type(eeg) == EEG)        # True
```

## Properties

### Core Properties

```python
data.subjectID      # Subject identifier
data.groupID        # Group identifier
data.condition      # Experimental condition
data.sampling_rate  # Sampling rate in Hz
data.history        # List of applied operations
data.extra          # Custom metadata dict
```

### Data Access

```python
data.data           # Underlying xarray.DataArray
data.to_numpy()     # Convert to numpy array
data.to_pandas()    # Convert to pandas DataFrame
```

## Immutability

`Data` objects are immutable. Attempting to modify them raises an error:

```python
data = cb.from_numpy(arr, dims=["time", "space"])

# This will fail:
data.subjectID = "sub-02"  # AttributeError!

# Instead, create a new Data object:
new_data = cb.from_xarray(
    data.data,
    subjectID="sub-02",
    history=data.history,
    extra=data.extra
)
```

## Sampling Rate Inference

If not provided, `sampling_rate` is inferred from time coordinates:

```python
# Time coordinates in seconds
coords = {"time": np.linspace(0, 1, 100), "space": ["E1", "E2"]}
xr_data = xr.DataArray(arr, dims=["time", "space"], coords=coords)

data = cb.from_xarray(xr_data)
print(data.sampling_rate)  # ~100.0 Hz (inferred)
```

If time coordinates look like indices (0, 1, 2, ...), no inference happens.

## Extra Fields

The `extra` dict stores custom metadata:

```python
data = cb.from_numpy(
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

# Gorkastyle: (time, space, labels)
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
from cobrabox import Data, EEG, FMRI

def process_eeg(data: EEG) -> Data:
    return cb.feature.line_length(data)
```
