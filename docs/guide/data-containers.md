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
    condition="baseline",
)

# 1D data (no time dimension)
arr_1d = np.random.normal(size=(50,))
data_1d = cb.Data.from_numpy(arr=arr_1d, dims=["channel"], subjectID="sub-01")
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
    coords={"x": range(100), "y": ["A", "B", "C", "D"]},
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
    arr=arr, dims=["time", "space"], sampling_rate=256.0, subjectID="sub-01", condition="task"
)

# Time dimension is automatically moved to last position
print(data.dims)  # ('space', 'time')
print(data.shape)  # (64, 1000)
```

**Requirements:**

- Must have a 'time' dimension
- `dims` length must match array `ndim`
- Time dimension will be transposed to last position

### EEG and FMRI Subclasses

```python
# EEG data (type marker)
eeg = cb.EEG.from_numpy(arr=arr, dims=["time", "space"], sampling_rate=256.0)
print(isinstance(eeg, cb.SignalData))  # True
print(isinstance(eeg, cb.Data))  # True
print(type(eeg) == cb.EEG)  # True

# FMRI data (type marker)
fmri = cb.FMRI.from_numpy(
    arr=fmri_arr,
    dims=["time", "spaceX", "spaceY", "spaceZ"],
    sampling_rate=0.5,  # TR = 2s
)
```

## Properties

### Core Properties

```python
# All containers share these properties
data.subjectID  # Subject identifier
data.groupID  # Group identifier
data.condition  # Experimental condition
data.runID  # Recording identifier within a subject
data.sampling_rate  # Sampling rate in Hz (None if no time dimension)
data.history  # List of applied operations
data.extra  # Custom metadata dict
```

### Metadata fields

Four fields identify a recording. All are optional and default to `None`.

| Field | Means | Example |
| ----- | ----- | ------- |
| `subjectID` | The participant. One person. | `"sub-01"`, `"ID1"` |
| `groupID` | How the item is classified — a cohort, or a generative condition for synthetic data. Classifies rather than identifies. | `"control"`, `"chain"` |
| `condition` | The state the recording was made in, or the experimental manipulation. | `"rest"`, `"sz13"` |
| `runID` | Which recording this is, when a subject has several. | `"03"`, `"7h"` |

The field names follow [BIDS](https://bids.neuroimaging.io/) entity conventions,
because several bundled datasets are already named that way.

#### Why `runID` exists

`subjectID` alone does not identify a recording. Long-term monitoring produces many
recordings per subject — 39 for one Zurich subject, 295 hourly segments for Swiss
`ID01` — so without `runID` those collapse into a single ambiguous label.

#### Run versus session versus split

BIDS distinguishes several things that all look like "another recording":

- **session** (`ses`) — a visit. The subject left and came back; the setup was
  re-established in between.
- **run** — a repetition of the *same* acquisition within one session. You ran the
  identical protocol again.
- **split** — one continuous acquisition divided across files, usually for file-size
  reasons. The signal is contiguous across the boundary.

CobraBox has no `sessionID`, deliberately. In every bundled dataset the session is
either absent or single-valued — Zurich is all `ses-interictalsleep`, `sleep_ieeg`
is all `ses-01` — so the field would discriminate nothing. Where a "session" value
describes a clinical state rather than a visit, as Zurich's does, it belongs in
`condition`.

`runID` therefore carries both true runs and splits. For `chb_mit`, `siena_eeg` and
`swiss_eeg_long` the repeated token is strictly a split — consecutive segments of
continuous monitoring — but it is what distinguishes those recordings, and a
separate field for the distinction would not change how anyone queries them.

#### How the bundled datasets map on

| Dataset | Filename | → fields |
| ------- | -------- | -------- |
| `zurich_ieeg` | `sub-01_ses-interictalsleep_run-03_ieeg.vhdr` | `subjectID="sub-01"`, `runID="03"` |
| `sleep_ieeg` | `sub-Detroit001_ses-01_task-sleep_ieeg.edf` | `subjectID="sub-Detroit001"` (one recording per subject) |
| `chb_mit` | `chb01_03.edf` | `subjectID="chb01"`, `runID="03"` |
| `siena_eeg` | `PN00-1.edf` | `subjectID="PN00"`, `runID="1"` |
| `swiss_eeg_long` | `ID01_7h.mat` | `subjectID="ID01"`, `runID="7h"` |
| `realistic_swiss` | `fit_Swiss_VAR_ID1_sz13_simulated_data_2.csv.xz` | `subjectID="ID1"`, `condition="sz13"`, `runID="2"` |
| `dummy_*` | `dummy_struct_VAR_chain_3.csv.xz` | `subjectID="sub-03"`, `groupID="chain"` |

`subjectID`, `condition` and `runID` compose into a `Dataset` label — see
[Working with Datasets](datasets.md#labels). `groupID` does not, since it classifies
items rather than identifying them.

Anything a dataset carries beyond these goes in `extra`, which `filter()` and
`groupby()` also accept.

### Data Access

A `Data` object is a thin wrapper. These accessors let you reach what's inside it — and ask the
usual questions about its shape — without reaching through `.data` yourself.

```python
# Get at the wrapped object
data.data  # Underlying xarray.DataArray
data.xarr  # The same DataArray, under a name that can't be misread
data.numpy  # Underlying numpy array, no copy

# Shape metadata, straight off the container
data.shape  # (64, 1000)                  axis lengths
data.size  # 64000                       total element count
data.dims  # ('space', 'time')           axis names
data.sizes  # {'space': 64, 'time': 1000} axis names → lengths

# Convert (these build a new object, unlike the accessors above)
data.to_numpy()  # numpy array
data.to_pandas()  # pandas DataFrame
```

Two distinctions are worth internalising, because both have bitten people:

| You want | Use | Not | Because |
|---|---|---|---|
| The numpy array | `.numpy` | `.data.data` | `.data.data` reads like a typo. `.xarr.data` says it out loud. |
| The axis lengths | `.shape` or `.sizes` | `.size` | `.size` is the element *count*, exactly as in numpy. |

`.numpy` and `.xarr` hand back the *live* underlying objects rather than copies, so they're free —
but `Data` is immutable, and writing into what they return breaks that guarantee. Use `.to_numpy()`
when you intend to modify the result. `.sizes` is the one exception: it returns a fresh `dict`, so
you can do what you like with it.

!!! tip "Which name should I reach for?"
    `.shape`, `.size` and `.numpy` mean exactly what they mean in numpy. `.dims` and `.sizes` mean
    exactly what they mean in xarray. Nothing here invents a third convention — if you know either
    library, the name you already expect is the right one.

## Dimensions and Coordinates

Every `Data` object wraps an `xarray.DataArray` accessible as `data.data`. The DataArray carries
named dimensions and optional coordinate labels for each axis.

**`dims` vs `coords`** — these answer different questions:

- **`dims`** is just a tuple of axis *names* (e.g. `("time", "space")`) — which axes exist, and in
  what order. No values are attached.
- **`coords`** maps a dimension name to the actual *label array* for that axis (e.g.
  `time: [0.0, 0.01, ...]`, `space: ["E1", "E2", ...]`) — the real tick values you'd use to select
  or display data. A dimension can exist without a coordinate (see the note below), so don't assume
  `dims` implies `coords`.

The usual pattern: check `dims` to confirm an axis exists, then read `coords` to get what's on it.

### Inspect dimensions

Dimension names and lengths come straight off the container — see
[Data Access](#data-access) above for the full set:

```python
item = cb.load_dataset("dummy_chain")[0]

item.dims  # e.g. ('space', 'time')
item.sizes  # e.g. {'space': 4, 'time': 200}

# Which coordinates have labels attached? (still via the DataArray)
coords = list(item.data.coords)  # e.g. ['time'] or ['time', 'space']
```

### Get coordinate values as a list

```python
# Space coordinate — returns a numpy array; call .tolist() for a plain list
space_labels = item.data.coords["space"].values.tolist()
# e.g. ['E1', 'E2', 'E3', 'E4']  or  [0, 1, 2, 3] if no labels were set

# Time coordinate in seconds
time_array = item.data.coords["time"].values  # numpy array
time_list = time_array.tolist()  # Python list
```

> **Note:** If you created `Data` with `from_numpy()` and did not supply coordinate labels for
> `space`, the space dimension will have no coordinates at all — `"space" not in item.data.coords`.
> To attach labels, build an `xr.DataArray` with explicit `coords` and use `from_xarray()`.

### Attach named coordinates (e.g., electrode labels)

```python
import xarray as xr
import numpy as np

arr = np.random.normal(size=(200, 8))  # 200 time steps, 8 channels
labels = [f"E{i + 1}" for i in range(8)]

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
data.data.coords["space"].values.tolist()  # ['E1', 'E2', ..., 'E8']
```

### Select by coordinate value

```python
# Single channel
ch = data.data.sel(space="E3")  # xr.DataArray, shape (200,)

# Multiple channels
subset = data.data.sel(space=["E1", "E5"])  # shape (2, 200) after transpose

# Time window (0.5 s – 1.0 s)
window = data.data.sel(time=slice(0.5, 1.0))

# To wrap the result back into a Data object:
data_subset = cb.Data.from_xarray(
    subset.rename({"space": "space"}),  # keep dims intact
    subjectID=data.subjectID,
    sampling_rate=data.sampling_rate,
)
```

### Convert to numpy or pandas

```python
arr = data.to_numpy()  # plain numpy array, shape matches data.data.shape
df = data.to_pandas()  # pandas DataFrame with MultiIndex from dimensions

# Access specific channels via pandas
df.xs("E1", level="space")  # time-series for channel E1
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
    data.data, subjectID="sub-02", history=data.history, extra=data.extra
)
```

## Extra Fields

The `extra` dict stores custom metadata:

```python
data = cb.SignalData.from_numpy(
    arr, dims=["time", "space"], extra={"notes": "Filtered 1-40 Hz", "bad_channels": ["E12", "E15"]}
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
    return cb.Mean(dim="time").apply(data)


def process_timeseries(data: SignalData) -> Data:
    """Requires time-series data."""
    return cb.LineLength().apply(data)


def process_eeg(data: EEG) -> EEG:
    """EEG-specific processing."""
    return cb.LineLength().apply(data)
```

## When to Use Each Container

- **Use `Data`** when you have general multidimensional data without time (e.g., cross-sectional data, images without temporal dimension)
- **Use `SignalData`** when you have time-series data (EEG, fMRI, other time-series)
- **Use `EEG`** or **Use `FMRI`** when you want explicit type markers for those modalities

Features that require time (like `LineLength`, `Bandpower`, `SlidingWindow`) are typed to accept `SignalData`, providing better IDE support and runtime validation.
