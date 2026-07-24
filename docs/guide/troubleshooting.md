# Troubleshooting

Common issues and solutions when using CobraBox.

## Installation Issues

### Command `uv` not found

**Problem:** Running `uv sync` returns "command not found"  
**Solution:** Install uv first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart your terminal or run `source $HOME/.cargo/env`.

---

### Git LFS errors when loading datasets

**Problem:** Dataset loading fails with "smudge filter lfs failed" or missing data files  
**Solution:**

```bash
# Install git-lfs if not already installed
# macOS: brew install git-lfs
# Ubuntu/Debian: sudo apt-get install git-lfs

# Initialize and pull LFS files
git lfs install
git lfs pull
```

---

## Data Container Issues

### ValueError: sampling_rate must be set

**Problem:** Features like `Bandpower`, `BandFilter`, or `Spectrogram` require `sampling_rate` but your Data has `None`  
**Solution:** Create Data with explicit sampling_rate:

```python
data = cb.Data.from_numpy(
    arr,
    dims=["time", "space"],
    sampling_rate=100.0,  # Required for frequency-domain features
)
```

---

### Dimension not found errors

**Problem:** Feature expects "time" or "space" dimension but your data has different names  
**Solution:** Check and rename dimensions:

```python
# Check available dimensions
print(data.data.dims)  # e.g., ('channels', 'samples')

# Rename to expected names
data = cb.Data.from_xarray(
    data.data.rename({"channels": "space", "samples": "time"}), sampling_rate=100.0
)
```

---

### Cannot create SignalData without time dimension

**Problem:** `SignalData.from_numpy()` raises ValueError about missing 'time' dimension  
**Solution:** SignalData requires a 'time' dimension. Use generic `Data` instead:

```python
# For non-time-series data (e.g., spatial matrices)
data = cb.Data.from_numpy(arr, dims=["space", "feature"])

# For time-series data
data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)
```

---

## Feature Issues

### Pipeline produces unexpected shape

**Problem:** Output dimensions do not match your expectations  
**Solution:** Check the feature's `output_type`:

```python
# Features that remove time dimension should set:
output_type: ClassVar[type[Data] | None] = Data

# This means sampling_rate will be None in output
result = cb.LineLength().apply(data)
print(result.sampling_rate)  # None - time dim removed
```

---

### ValueError: no real time axis left

**Problem:** You applied a time-domain feature to the output of a chord whose per-window feature already consumed the time dimension:

```python
chord = (
    cb.feature.SlidingWindow(window_size=100, step_size=50)
    | cb.feature.LineLength()
    | cb.feature.ConcatAggregate()
)
result = chord.apply(data)

cb.feature.LineLength().apply(result)  # ValueError: no real time axis left
```

**Why:** `LineLength` reduced each window to a single number, so no time axis survives. A `Chord` returns the same container type it was given, and `SignalData` requires a `time` dimension — so a length-1 placeholder is inserted to keep that contract. It is not time you can compute over. Without this error, time-domain features would silently return zeros.

**Solution:** decide which axis you actually meant.

To reduce over windows, use a feature that takes a `dim` argument:

```python
cb.feature.Mean(dim="window").apply(result)  # average the time course
cb.feature.Max(dim="window").apply(result)  # peak across windows
```

To apply a second time-domain feature *per window*, put it inside the chord rather than after it:

```python
chord = (
    cb.feature.SlidingWindow(window_size=100, step_size=50)
    | cb.feature.BandpassFilter(bands={"alpha": [8.0, 12.0]})
    | cb.feature.LineLength()
    | cb.feature.ConcatAggregate()
)
```

Note that a feature keeping a real, multi-sample time axis per window (a filter, say) leaves the chord output with genuine time — the error only appears once time has actually been reduced away.

---

### ValueError: coords cannot be an empty list

**Problem:** Matrix features (e.g., `PhaseLockingValueMatrix`) receive empty coords  
**Solution:** Provide at least one coordinate:

```python
# Wrong
matrix = cb.PhaseLockingValue(coords=[]).apply(data)

# Correct
matrix = cb.PhaseLockingValue(coords=[0, 1, 2]).apply(data)
```

---

### MemoryError when processing large datasets

**Problem:** Processing exhausts RAM on large datasets  
**Solution:** Use `SlidingWindow` with lazy evaluation:

```python
# Memory-efficient approach with Chord
result = cb.Chord(
    split=cb.SlidingWindow(window_size=1000, step_size=500),
    pipeline=cb.LineLength(),
    aggregate=cb.MeanAggregate(),
).apply(data)  # Processes one window at a time

# Instead of loading all windows:
# windows = list(cb.SlidingWindow(...)(data))  # Don't do this for large data!
```

---

### Chord pipeline produces wrong history

**Problem:** History shows operations in unexpected order  
**Solution:** This is expected. Chord automatically appends its name last:

```python
result = (
    cb.SlidingWindow(window_size=20, step_size=10) | cb.LineLength() | cb.MeanAggregate()
).apply(data)

print(result.history)
# ['SlidingWindow', 'LineLength', 'MeanAggregate', 'Chord']  # Chord is always last
```

---

## Serialization Issues

### SchemaVersionError when loading old pipelines

**Problem:** Loading a saved pipeline raises SchemaVersionError  
**Solution:** The serialization format has changed. You need to recreate the pipeline:

```python
# Old pipeline may fail to load
# pipeline = cb.load("old_pipeline.yaml")  # Raises SchemaVersionError

# Recreate with current version
pipeline = cb.SlidingWindow(window_size=20, step_size=10) | cb.LineLength() | cb.MeanAggregate()
cb.save(pipeline, "new_pipeline.yaml")
```

---

### Loading fails with "No module named 'dill'"

**Problem:** Pipeline contains callables but dill is not installed  
**Solution:** Dill is a mandatory dependency. Reinstall:

```bash
uv sync  # Ensures all dependencies are installed
```

---

## Type Checking Issues

### IDE shows type errors with Data vs SignalData

**Problem:** Your IDE complains about passing Data to SignalData features  
**Solution:** This is usually a false positive. Ensure you use the right type:

```python
# SignalData features work with SignalData, not generic Data
data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)
result = cb.LineLength().apply(data)  # OK

# Generic Data features work with any Data
result = cb.Mean(dim="time").apply(data)  # Also OK
```

---

### Mypy/Pyright errors with feature generics

**Problem:** Type checker complains about BaseFeature[Data] vs BaseFeature[SignalData]  
**Solution:** These are often false positives. The runtime behavior is correct. Add `# type: ignore` if needed:

```python
from cobrabox.data import SignalData


@dataclass
class MyFeature(BaseFeature[SignalData]):  # type: ignore[type-var]
    ...
```

---

## Testing Issues

### Tests fail with "Fixture not found"

**Problem:** Running tests fails with fixture errors  
**Solution:** Ensure you're running from the repo root:

```bash
# Correct
uv run pytest tests/test_feature_line_length.py -v

# Also correct
uv run pytest tests/ -v
```

---

### Coverage below 95% threshold

**Problem:** Tests pass but coverage fails  
**Solution:** Check which lines are uncovered:

```bash
uv run pytest --cov-report=term-missing --cov=src/cobrabox/features/my_feature tests/test_feature_my_feature.py
```

Add tests for the missing lines.

---

## Performance Tips

### Slow feature computation

**Problem:** Feature takes too long to run  
**Solutions:**

1. **Check input size**: Large time dimensions are slow
2. **Use windowing**: Process in chunks with `SlidingWindow`
3. **Reduce channels**: Process fewer spatial channels at once
4. **Check method**: Some features have faster alternatives:
   - `FractalDimKatz` is faster than `FractalDimHiguchi`
   - `SlidingWindowReduce` is simpler than `Chord` for basic stats

---

### High memory usage with Chord

**Problem:** Chord pipelines consume too much memory  
**Solution:** The issue is usually in the aggregator. Use `ConcatAggregate` carefully:

```python
# This keeps all windows in memory:
result = cb.Chord(
    split=cb.SlidingWindow(window_size=100, step_size=10),
    pipeline=cb.LineLength(),
    aggregate=cb.ConcatAggregate(),  # Stacks all windows
).apply(data)

# This reduces to one result (memory-efficient):
result = cb.Chord(
    split=cb.SlidingWindow(window_size=100, step_size=10),
    pipeline=cb.LineLength(),
    aggregate=cb.MeanAggregate(),  # Averages windows
).apply(data)
```

---

## Getting Help

If your issue is not covered here:

1. Check the [API Reference](../api/index.md) for feature-specific documentation
2. Review [feature examples](https://github.com/cobragroup/cobrabox/tree/main/examples) in the repo
3. Look at the test files in `tests/` for usage patterns
4. Run the agent review for the feature: `/review-feature src/cobrabox/features/my_feature.py`
