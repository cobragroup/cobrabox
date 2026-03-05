# Pipelines

Build reproducible analysis pipelines by chaining features with `|`.

## Sequential Pipeline

Chain `BaseFeature` instances with `|`:

```python
import cobrabox as cb

data = cb.dataset("dummy_chain")[0]

pipeline = cb.feature.Min(dim="time") | cb.feature.Max(dim="time")
result = pipeline.apply(data)
print(result.history)  # ['Min', 'Max']
```

Each step receives the output of the previous one. The pipeline is itself composable — you can store it and reuse it across subjects.

```python
pipeline = cb.feature.LineLength() | cb.feature.Mean(dim="space")

results = [pipeline.apply(d) for d in cb.dataset("dummy_chain")]
```

## Chord (fan-out → map → fan-in)

A `Chord` runs a `SplitterFeature` to produce a stream of windows, applies a per-window pipeline, and folds the results back into one `Data` with an `AggregatorFeature`.

```python
chord = (
    cb.feature.SlidingWindow(window_size=20, step_size=10)
    | cb.feature.LineLength()
    | cb.feature.MeanAggregate()
)
result = chord.apply(data)
print(result.history)  # ['SlidingWindow', 'LineLength', 'MeanAggregate', 'Chord']
```

The `|` operator builds the chord automatically:

| Left              | Right               | Result                              |
| ----------------- | ------------------- | ----------------------------------- |
| `BaseFeature`     | `BaseFeature`       | `Pipeline`                          |
| `SplitterFeature` | `BaseFeature`       | `_ChordBuilder` (intermediate)      |
| `_ChordBuilder`   | `BaseFeature`       | `_ChordBuilder` (extended pipeline) |
| `_ChordBuilder`   | `AggregatorFeature` | `Chord`                             |

A `Chord` is itself a `BaseFeature`, so it composes freely with `|`:

```python
full = (
    cb.feature.SlidingWindow(window_size=20, step_size=10)
    | cb.feature.LineLength()
    | cb.feature.MeanAggregate()
    | cb.feature.Mean(dim="space")   # post-chord step
)
result = full.apply(data)
print(result.history)  # ['SlidingWindow', 'LineLength', 'MeanAggregate', 'Chord', 'Mean']
```

## Multi-Step Chord

Pipe multiple `BaseFeature` steps between the splitter and the aggregator:

```python
chord = (
    cb.feature.SlidingWindow(window_size=20, step_size=10)
    | cb.feature.LineLength()
    | cb.feature.Mean(dim="time")
    | cb.feature.MeanAggregate()
)
result = chord.apply(data)
print(result.history)
# ['SlidingWindow', 'LineLength', 'Mean', 'MeanAggregate', 'Chord']
```

## Applying to a Dataset

```python
pipeline = (
    cb.feature.SlidingWindow(window_size=20, step_size=10)
    | cb.feature.LineLength()
    | cb.feature.MeanAggregate()
)

datasets = cb.dataset("dummy_chain")
results = [pipeline.apply(d) for d in datasets]
```

## History Tracking

Every step appends its class name to `history`:

```python
data = cb.from_numpy(arr, dims=["time", "space"])

result = (
    cb.feature.SlidingWindow(window_size=10, step_size=5)
    | cb.feature.LineLength()
    | cb.feature.MeanAggregate()
).apply(data)

print(result.history)
# ['SlidingWindow', 'LineLength', 'MeanAggregate', 'Chord']
```

Use history for logging, reproducibility checks, or debugging:

```python
for i, step in enumerate(result.history, 1):
    print(f"  {i}. {step}")
```

## Best Practices

1. **Build pipelines once, apply many times** — define the pipeline outside the loop, then apply it per subject
2. **Prefer `Chord` over manual loops** — it handles history propagation correctly
3. **Validate inputs in `__call__`** — raise `ValueError` early rather than producing silent bad output
4. **`AggregatorFeature` owns its history** — propagate per-window ops manually when writing a custom aggregator
5. **No side effects** — features must never mutate `data`; always return new objects
