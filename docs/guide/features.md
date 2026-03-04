# Features

Features are the core building blocks of CobraBox pipelines. They are classes that operate on `Data` objects and return new `Data` objects with updated history.

## Feature Types

CobraBox has three kinds of features, each with a different role:

| Type | Signature | Role |
|------|-----------|------|
| `BaseFeature` | `DataT → Data` | Standard transformation |
| `SplitterFeature` | `DataT → Iterator[Data]` | Splits data into a lazy stream (e.g. windows) |
| `AggregatorFeature` | `(Data, Iterator[Data]) → Data` | Folds a stream back into one `Data` |

Features are generic: they accept a type parameter `DataT` that specifies what kind of data they work with. Use `BaseFeature[SignalData]` for time-series features, or `BaseFeature[Data]` for generic features that work with any data.

## What is a Feature?

A feature is a `@dataclass` subclassing `BaseFeature`. Store configuration in fields; implement `__call__`:

```python
from __future__ import annotations
from dataclasses import dataclass
import xarray as xr
from cobrabox.base_feature import BaseFeature
from cobrabox.data import SignalData

@dataclass
class SpectralPower(BaseFeature[SignalData]):
    """Compute mean power in a frequency band."""

    low: float
    high: float

    def __call__(self, data: SignalData) -> xr.DataArray:
        xr_data = data.data
        # SignalData guarantees 'time' dimension exists
        # ... FFT, bandpass, etc.
        return xr_data.mean(dim="time")
```

Call `.apply(data)` — it handles wrapping the result and appending the class name to `history`:

```python
feat = SpectralPower(low=8.0, high=12.0).apply(data)
print(feat.history)  # ['SpectralPower']
```

## Generic Feature Typing

### Generic Features (Work with Any Data)

Use `BaseFeature[Data]` for features that work with any data container:

```python
from cobrabox.base_feature import BaseFeature
from cobrabox.data import Data

@dataclass
class Mean(BaseFeature[Data]):
    """Compute mean over any dimension."""
    dim: str

    def __call__(self, data: Data) -> xr.DataArray:
        return data.data.mean(dim=self.dim)
```

### Time-Series Features (Require SignalData)

Use `BaseFeature[SignalData]` for features that require time-series data:

```python
from cobrabox.base_feature import BaseFeature
from cobrabox.data import SignalData

@dataclass
class LineLength(BaseFeature[SignalData]):
    """Compute line length over time dimension."""

    def __call__(self, data: SignalData) -> xr.DataArray:
        xr_data = data.data
        diff = xr_data.diff(dim="time")
        return abs(diff).sum(dim="time")
```

The `SignalData` type ensures:
- Data has a 'time' dimension (validated at construction)
- `sampling_rate` may be available
- Better IDE support and type checking

## Built-in Features

### `LineLength`

```python
feat = cb.feature.LineLength().apply(data)
```

Sum of absolute differences between consecutive timepoints per channel.

### `Min` / `Max` / `Mean`

```python
min_val = cb.feature.Min(dim="time").apply(data)
max_val = cb.feature.Max(dim="time").apply(data)
mean_val = cb.feature.Mean(dim="time").apply(data)
```

Reduce over any dimension present in the data.

### `SlidingWindow` (splitter)

```python
windows = cb.feature.SlidingWindow(window_size=10, step_size=5)(data)
# yields one Data per window, lazily
for window in windows:
    print(window.data.shape)
```

Used inside a `Chord` — not called directly in typical pipelines.

### `Bandpower`

```python
bp = cb.feature.Bandpower().apply(data)                          # all five default bands
bp = cb.feature.Bandpower(bands={"alpha": True}).apply(data)     # single default band
bp = cb.feature.Bandpower(bands={"ripple": [45, 80]}).apply(data)  # custom range
```

Computes band power using Welch's method for each requested frequency band. Returns a
`(band_index, space)` array (plus a singleton `time` dimension). Requires `sampling_rate`
to be set on the `Data` object.

Default bands: `delta` (1–4 Hz), `theta` (4–8 Hz), `alpha` (8–12 Hz), `beta` (12–30 Hz),
`gamma` (30–45 Hz).

### `Coherence`

```python
coh = cb.feature.Coherence().apply(data)
coh = cb.feature.Coherence(nperseg=128).apply(data)
```

Computes magnitude-squared coherence for every unique pair of spatial channels using
Welch's method (50% overlap, Hann window). Returns a symmetric `(space, space_to)` matrix
in [0, 1] with NaN on the diagonal. Extra dimensions (e.g. `window_index`) are preserved.

### `MeanAggregate` (aggregator)

Averages a stream of per-window `Data` objects into one result. Used as the terminal step of a `Chord`.

## Pipe Syntax `|`

### Sequential pipeline

Chain `BaseFeature` instances with `|`:

```python
pipeline = cb.feature.Min(dim="time") | cb.feature.Max(dim="time")
result = pipeline.apply(data)
print(result.history)  # ['Min', 'Max']
```

### Chord (fan-out → map → fan-in)

Start a `Chord` by piping a `SplitterFeature` into a pipeline step, then into an `AggregatorFeature`:

```python
chord = (
    cb.feature.SlidingWindow(window_size=20, step_size=10)
    | cb.feature.LineLength()
    | cb.feature.MeanAggregate()
)
result = chord.apply(data)
print(result.history)  # ['SlidingWindow', 'LineLength', 'MeanAggregate', 'Chord']
```

The intermediate steps build a `_ChordBuilder`; piping into an `AggregatorFeature` finalises it into a `Chord`. A `Chord` is itself a `BaseFeature`, so it composes freely with `|`:

```python
full = (
    cb.feature.SlidingWindow(window_size=20, step_size=10)
    | cb.feature.LineLength()
    | cb.feature.MeanAggregate()
    | cb.feature.Mean(dim="time")   # post-chord step
)
```

## Feature Discovery

Features are auto-discovered from `src/cobrabox/features/`. Any class with `_is_cobrabox_feature = True` (inherited from all base classes) whose `__module__` matches its file is registered automatically under `cb.feature.*`.

```python
import cobrabox as cb
print(dir(cb.feature))
```

## Creating Custom Features

### Generic `BaseFeature` (works with any Data)

```python
# src/cobrabox/features/variance.py
from __future__ import annotations
from dataclasses import dataclass
import xarray as xr
from cobrabox.base_feature import BaseFeature
from cobrabox.data import Data

@dataclass
class Variance(BaseFeature[Data]):
    """Compute variance over a dimension."""

    dim: str

    def __call__(self, data: Data) -> xr.DataArray:
        if self.dim not in data.data.dims:
            raise ValueError(f"dim '{self.dim}' not found in {data.data.dims}")
        return data.data.var(dim=self.dim)
```

### Time-Series `BaseFeature` (requires SignalData)

```python
# src/cobrabox/features/band_power.py
from __future__ import annotations
from dataclasses import dataclass
import xarray as xr
from cobrabox.base_feature import BaseFeature
from cobrabox.data import SignalData

@dataclass
class BandPower(BaseFeature[SignalData]):
    """Compute power in a frequency band."""

    band: tuple[float, float]

    def __call__(self, data: SignalData) -> xr.DataArray:
        # SignalData guarantees 'time' dimension exists
        # No need to check: if "time" not in data.data.dims
        # ... compute power
        return result
```

### `SplitterFeature` (time-series)

```python
# src/cobrabox/features/trial_split.py
from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass
from cobrabox.base_feature import SplitterFeature
from cobrabox.data import Data, SignalData

@dataclass
class TrialSplit(SplitterFeature[SignalData]):
    """Yield one Data per trial block."""

    trial_length: int

    def __call__(self, data: SignalData) -> Iterator[Data]:
        n = data.data.sizes["time"]
        for start in range(0, n - self.trial_length + 1, self.trial_length):
            window = data.data.isel(time=slice(start, start + self.trial_length))
            yield data._copy_with_new_data(new_data=window, operation_name="TrialSplit")
```

### `AggregatorFeature`

```python
# src/cobrabox/features/max_aggregate.py
from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass
import xarray as xr
from cobrabox.base_feature import AggregatorFeature
from cobrabox.data import Data

@dataclass
class MaxAggregate(AggregatorFeature):
    """Take element-wise max across a stream of Data."""

    def __call__(self, data: Data, stream: Iterator[Data]) -> Data:
        items = list(stream)
        if not items:
            raise ValueError("MaxAggregate received an empty stream")
        stacked = xr.concat([w.data for w in items], dim="window", join="override")
        result = stacked.max(dim="window")
        window_history = [op for op in items[0].history if op not in data.history]
        return Data(
            data=result,
            subjectID=data.subjectID,
            groupID=data.groupID,
            condition=data.condition,
            sampling_rate=data.sampling_rate,
            history=list(data.history) + window_history + ["MaxAggregate"],
            extra=data.extra,
        )
```

## Accessing Feature History

```python
data = cb.from_numpy(arr, dims=["time", "space"])

feat = cb.feature.LineLength().apply(data)
print(feat.history)  # ['LineLength']

result = (
    cb.feature.SlidingWindow(window_size=10, step_size=5)
    | cb.feature.LineLength()
    | cb.feature.MeanAggregate()
).apply(data)
print(result.history)  # ['SlidingWindow', 'LineLength', 'MeanAggregate', 'Chord']
```

## Best Practices

1. **One class per file** — match filename to class name (snake_case file, PascalCase class)
2. **Use proper generic typing** — `BaseFeature[SignalData]` for time-series, `BaseFeature[Data]` for generic
3. **Let SignalData validate** — no need to check for 'time' dimension; `SignalData` validates at construction
4. **Document thoroughly** — Args, returns, and example in the docstring
5. **`AggregatorFeature` owns its history** — propagate per-window ops manually
6. **No side effects** — never mutate `data` in place; always return new objects
