# Core Concepts

CobraBox is built on three key principles: **immutability**, **metadata preservation**, and **transparent pipelines**.

## Immutability

All `Data` objects are **immutable**. Once created, you cannot modify them. Instead, operations return new instances:

```python
import cobrabox as cb
import numpy as np

data = cb.from_numpy(
    arr=np.random.normal(size=(100, 4)),
    dims=["time", "space"]
)

# This creates a NEW Data object; data is unchanged
result = cb.feature.line_length(data)

# Original data is preserved
assert data.history == []
assert len(result.history) == 1
```

**Why immutability?**

- Prevents accidental data modification
- Makes debugging easier
- Enables reproducible pipelines
- Thread-safe by design

## Metadata Preservation

Every `Data` object carries metadata in its `attrs`:

```python
data = cb.from_numpy(
    arr=np.random.normal(size=(100, 4)),
    dims=["time", "space"],
    sampling_rate=100.0,
    subjectID="sub-01",
    groupID="control",
    condition="baseline"
)

print(f"Subject: {data.subjectID}")
print(f"Group: {data.groupID}")
print(f"Condition: {data.condition}")
print(f"Sampling rate: {data.sampling_rate}")
```

When you apply features, this metadata is automatically preserved in the returned `Data` object.

## History Tracking

Every operation applied to a `Data` object is recorded in its `history` attribute:

```python
data = cb.from_numpy(arr, dims=["time", "space"])
print(data.history)  # []

result = cb.feature.line_length(data)
print(result.history)  # ['line_length']

# Chain multiple operations
wdata = cb.feature.sliding_window(data, window_size=10)
win_min = cb.feature.min(wdata, dim="window_index")
print(win_min.history)  # ['sliding_window', 'min']
```

This provides a complete audit trail of your preprocessing pipeline.

## The Data Model

At its core, CobraBox wraps `xarray.DataArray`:

```
┌─────────────────────────────────────┐
│              Data                   │
│  ┌───────────────────────────────┐  │
│  │       xarray.DataArray        │  │
│  │  - dims: (time, space, ...)   │  │
│  │  - coords: labeled axes       │  │
│  │  - attrs: metadata            │  │
│  └───────────────────────────────┘  │
│  - subjectID                        │
│  - groupID                          │
│  - condition                        │
│  - sampling_rate                    │
│  - history: ['op1', 'op2', ...]     │
│  - extra: {custom fields}           │
└─────────────────────────────────────┘
```

## Dimensions

### Mandatory Dimensions

- **`time`** - Temporal dimension (samples, timepoints)
- **`space`** - Spatial dimension (electrodes, voxels, channels)

### Optional Dimensions

- **`spaceX`, `spaceY`, `spaceZ`** - Additional spatial dimensions (for fMRI)
- **`run_index`** - Run/block index
- **`window_index`** - Window index (from sliding window operations)
- **`band_index`** - Frequency band index

## Type Distinctions

CobraBox provides three classes:

- **`Data`** - Base class for all time-series data
- **`EEG`** - Subclass for EEG data (type marker)
- **`FMRI`** - Subclass for fMRI data (type marker)

```python
eeg_data = EEG.from_numpy(arr, dims=["time", "space"])
fmri_data = FMRI.from_numpy(arr, dims=["time", "space", "spaceZ"])
```

## Extra Fields

Use the `extra` dict for custom metadata that doesn't fit in standard attrs:

```python
data = cb.from_numpy(
    arr,
    dims=["time", "space"],
    extra={
        "preprocessing_notes": "Bandpass filtered 1-40 Hz",
        "bad_channels": ["E12", "E15"],
        "custom_array": np.array([1, 2, 3])
    }
)

# Access extra fields
notes = data.extra["preprocessing_notes"]
```

The `extra` dict is preserved and merged across operations.
