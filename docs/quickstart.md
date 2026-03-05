# Quick Start

Get up and running with CobraBox in 5 minutes.

## 1. Create Data

Start with a numpy array and wrap it in a `Data` container:

```python
import cobrabox as cb
import numpy as np

# Create synthetic data: 100 timepoints, 4 channels
my_array = np.random.default_rng(seed=0).normal(size=(100, 4))

data = cb.from_numpy(
    arr=my_array,
    dims=["time", "space"],
    sampling_rate=100.0,  # Hz
    subjectID="sub-01",
    condition="baseline"
)
```

## 2. Apply a Feature

Call `.apply(data)` on any feature class:

```python
feat = cb.feature.LineLength().apply(data)

print(f"Shape: {feat.data.shape}")
print(f"History: {feat.history}")  # ['LineLength']
```

## 3. Chain Features with `|`

Use `|` to build a sequential `Pipeline`:

```python
pipeline = cb.feature.Min(dim="time") | cb.feature.Max(dim="time")
result = pipeline.apply(data)
print(result.history)  # ['Min', 'Max']
```

## 4. Windowed Pipelines with Chord

Pipe a `SplitterFeature` into a pipeline and close it with an `AggregatorFeature`:

```python
chord = (
    cb.feature.SlidingWindow(window_size=20, step_size=10)
    | cb.feature.LineLength()
    | cb.feature.MeanAggregate()
)
result = chord.apply(data)
print(result.history)  # ['SlidingWindow', 'LineLength', 'MeanAggregate', 'Chord']
```

A `Chord` is itself a `BaseFeature`, so it composes with `|` like any other step.

## 5. Apply to a Dataset

```python
datasets = cb.dataset("dummy_chain")

pipeline = (
    cb.feature.SlidingWindow(window_size=20, step_size=10)
    | cb.feature.LineLength()
    | cb.feature.MeanAggregate()
)

results = [pipeline.apply(d) for d in datasets]
```

## 6. Access Data

```python
# As numpy array
arr = result.to_numpy()

# Access underlying xarray.DataArray
xr_data = result.data
```

## What's Next?

- [Core Concepts](guide/core-concepts.md) — Immutability, metadata, history, and the feature system
- [Feature Guide](guide/features.md) — All feature types and how to create custom ones
- [Pipelines](guide/pipelines.md) — `|` syntax, Chord, and batch processing
- [Data Containers](guide/data-containers.md) — Deep dive into the `Data` class
