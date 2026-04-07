# Pipelines

Build reproducible analysis pipelines by chaining features with `|`.

## Sequential Pipeline

Chain `BaseFeature` instances with `|`:

```python
import cobrabox as cb

data = cb.load_dataset("dummy_chain")[0]

pipeline = cb.feature.Min(dim="time") | cb.feature.Max(dim="time")
result = pipeline.apply(data)
print(result.history)  # ['Min', 'Max']
```

Each step receives the output of the previous one. The pipeline is itself composable — you can store it and reuse it across subjects.

```python
pipeline = cb.feature.LineLength() | cb.feature.Mean(dim="space")

results = [pipeline.apply(d) for d in cb.load_dataset("dummy_chain")]
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

## Simple Windowed Aggregation

For basic windowed statistics without per-window features, use `SlidingWindowReduce` — it's simpler than a full Chord:

```python
# Single-step: window + aggregate
result = cb.feature.SlidingWindowReduce(
    window_size=100, step_size=50, dim="time", agg="mean"
).apply(data)
print(result.history)  # ['SlidingWindowReduce']
```

This computes the mean of each 100-sample window (stepping by 50) and returns a `Data` with a `window` dimension. Supports `mean`, `std`, `sum`, `min`, `max`.

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

datasets = cb.load_dataset("dummy_chain")
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

## Serialization

Save any feature, pipeline, or chord to YAML or JSON and reload it later. Useful for sharing pipelines with collaborators, versioning analysis configurations, and reproducing results.

### File I/O

```python
# Save to file (format inferred from extension: .yaml / .yml / .json)
cb.save(pipeline, "my_pipeline.yaml")
cb.save(pipeline, "my_pipeline.json")

# Load from file
loaded = cb.load("my_pipeline.yaml")
result = loaded.apply(data)
```

### String / dict

```python
# Serialize to string
yaml_str = cb.serialize(pipeline)          # YAML (default)
json_str = cb.serialize(pipeline, fmt="json")

# Deserialize from string
restored = cb.deserialize(yaml_str)
restored = cb.deserialize(json_str, fmt="json")

# Method API — available on any feature, pipeline, or chord
yaml_str = pipeline.to_yaml()
pipeline  = cb.Pipeline.from_yaml(yaml_str)

d        = pipeline.to_dict()
pipeline  = cb.Pipeline.from_dict(d)
```

### YAML format

All objects serialize as a `pipeline:` list. A single feature or chord is a one-element list:

```yaml
cobrabox_version: "0.3.1"
schema_version: "1.0.0"

pipeline:
  - split:
      class: SlidingWindow
      module: cobrabox.features.sliding_window
      params:
        window_size: 20
        step_size: 10
    pipeline:
      - class: LineLength
        module: cobrabox.features.line_length
        params: {}
    aggregate:
      class: MeanAggregate
      module: cobrabox.features.mean_aggregate
      params: {}
```

### Security note

Callable parameters (lambdas, functions) are serialized with `dill`. Only load YAML/JSON files from trusted sources.

## Best Practices

1. **Build pipelines once, apply many times** — define the pipeline outside the loop, then apply it per subject
2. **Prefer `Chord` over manual loops** — it handles history propagation correctly
3. **Save pipelines alongside results** — `cb.save(pipeline, ...)` keeps your analysis fully reproducible
4. **Validate inputs in `__call__`** — raise `ValueError` early rather than producing silent bad output
5. **`AggregatorFeature` owns its history** — propagate per-window ops manually when writing a custom aggregator
6. **No side effects** — features must never mutate `data`; always return new objects
