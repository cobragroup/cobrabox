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

# Creates a NEW Data object; data is unchanged
result = cb.feature.LineLength().apply(data)

assert data.history == []
assert result.history == ["LineLength"]
```

**Why immutability?**

- Prevents accidental data modification
- Makes debugging easier
- Enables reproducible pipelines
- Thread-safe by design

## Metadata Preservation

Every `Data` object carries metadata:

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

When you apply features, metadata is automatically preserved in the returned `Data` object.

## History Tracking

Every operation appends its class name to `history`:

```python
data = cb.from_numpy(arr, dims=["time", "space"])
print(data.history)  # []

result = cb.feature.LineLength().apply(data)
print(result.history)  # ['LineLength']

# Chord: SlidingWindow + per-window pipeline + aggregation
result2 = (
    cb.feature.SlidingWindow(window_size=10, step_size=5)
    | cb.feature.LineLength()
    | cb.feature.MeanAggregate()
).apply(data)
print(result2.history)  # ['SlidingWindow', 'LineLength', 'MeanAggregate', 'Chord']
```

## The Feature System

CobraBox has three feature types, composable with `|`:

```text
BaseFeature      Data → Data          standard transformation
SplitterFeature  Data → Iterator[Data]  lazy stream of windows
AggregatorFeature  (Data, stream) → Data  fold stream back to Data
```

### Sequential pipeline

```python
pipeline = cb.feature.Min(dim="time") | cb.feature.Max(dim="time")
result = pipeline.apply(data)
```

### Chord (fan-out → map → fan-in)

```python
chord = (
    cb.feature.SlidingWindow(window_size=20, step_size=10)
    | cb.feature.LineLength()
    | cb.feature.MeanAggregate()
)
result = chord.apply(data)
```

The `|` operator knows which type is on the left and produces either a `Pipeline`, a `_ChordBuilder`, or a finalised `Chord` accordingly.

## The Data Model

At its core, CobraBox wraps `xarray.DataArray`:

```text
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
│  - history: ['Op1', 'Op2', ...]     │
│  - extra: {custom fields}           │
└─────────────────────────────────────┘
```

## Dimensions

### Mandatory

- **`time`** — temporal dimension (samples, timepoints)
- **`space`** — spatial dimension (electrodes, voxels, channels)

### Optional

- **`spaceX`, `spaceY`, `spaceZ`** — additional spatial dimensions (fMRI)
- **`run_index`** — run/block index
- **`band_index`** — frequency band index

## Type Distinctions

- **`Data`** — base class for all data (general multidimensional container)
- **`SignalData`** — subclass for time-series data (requires 'time' dimension)
- **`EEG`** — subclass for EEG data (type marker, inherits from SignalData)
- **`FMRI`** — subclass for fMRI data (type marker, inherits from SignalData)

```python
eeg_data = cb.EEG(base.data, sampling_rate=256.0)
```

## Extra Fields

Use the `extra` dict for custom metadata:

```python
data = cb.from_numpy(
    arr,
    dims=["time", "space"],
    extra={
        "preprocessing_notes": "Bandpass filtered 1-40 Hz",
        "bad_channels": ["E12", "E15"],
    }
)

notes = data.extra["preprocessing_notes"]
```

The `extra` dict is preserved and merged across feature operations.
