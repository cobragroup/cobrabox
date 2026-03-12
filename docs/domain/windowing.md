# Windowing & Aggregation Features

Windowing features split signals into overlapping windows, and aggregation features combine windowed results.

## Features

### SlidingWindow
Split data into overlapping windows (splitter).

### SlidingWindowReduce
Single-step windowing + aggregation (simpler alternative to Chord).

### MeanAggregate
Average windowed results (aggregator).

### ConcatAggregate
Stack windowed results along new dimension (aggregator).

## Usage

```python
import cobrabox as cb
import numpy as np

data = cb.from_numpy(np.random.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0)

# Using Chord (fan-out → map → fan-in)
result = cb.Chord(
    split=cb.feature.SlidingWindow(window_size=20, step_size=10),
    pipeline=cb.feature.LineLength(),
    aggregate=cb.feature.MeanAggregate(),
).apply(data)

# Using SlidingWindowReduce (simpler)
result = cb.feature.SlidingWindowReduce(
    feature=cb.feature.LineLength(),
    window_size=20,
    step_size=10,
    aggregate=cb.feature.MeanAggregate(),
).apply(data)
```

## See Also

- [Time Domain](time_domain.md) for features applied within windows
- [Pipelines](../guide/pipelines.md) for working with Chord and pipelines

