# Migration Patterns

Full before/after examples for each feature type.

---

## BaseFeature — No Parameters

### Before

```python
# src/cobrabox/features/line_length.py
from __future__ import annotations

import xarray as xr

from cobrabox.function_wrapper import feature
from cobrabox.data import Data


@feature
def line_length(data: Data, **kwargs) -> xr.DataArray:
    """Compute line length (sum of absolute differences between consecutive timepoints)."""
    return data.data.diff(dim="time").abs().sum(dim="time")
```

### After

```python
# src/cobrabox/features/line_length.py
from __future__ import annotations

from dataclasses import dataclass

import xarray as xr

from ..base_feature import BaseFeature
from ..data import Data


@dataclass
class LineLength(BaseFeature):
    """Compute line length over the time dimension.

    Sum of absolute differences between consecutive timepoints per channel.

    Example:
        >>> result = cb.feature.LineLength().apply(data)
    """

    def __call__(self, data: Data) -> xr.DataArray:
        if "time" not in data.data.dims:
            raise ValueError("data must have 'time' dimension")
        return data.data.diff(dim="time").abs().sum(dim="time")
```

---

## BaseFeature — With Parameters

### Before

```python
# src/cobrabox/features/mean.py
from __future__ import annotations

import xarray as xr

from cobrabox.function_wrapper import feature
from cobrabox.data import Data


@feature
def mean(data: Data, dim: str = "time", **kwargs) -> xr.DataArray:
    """Compute mean over a dimension."""
    return data.data.mean(dim=dim)
```

### After

```python
# src/cobrabox/features/mean.py
from __future__ import annotations

from dataclasses import dataclass

import xarray as xr

from ..base_feature import BaseFeature
from ..data import Data


@dataclass
class Mean(BaseFeature):
    """Compute mean across a dimension.

    Args:
        dim: Dimension to reduce over.

    Example:
        >>> result = cb.feature.Mean(dim="time").apply(data)
    """

    dim: str

    def __call__(self, data: Data) -> xr.DataArray:
        if self.dim not in data.data.dims:
            raise ValueError(f"dim '{self.dim}' not found in data dimensions {data.data.dims}")
        return data.data.mean(dim=self.dim)
```

**Note on defaults:** If the old function had `dim: str = "time"`, use a dataclass default:

```python
from dataclasses import dataclass, field

@dataclass
class Mean(BaseFeature):
    dim: str = field(default="time")
```

Or, for simple scalar defaults, the shorthand `dim: str = "time"` also works.

---

## BaseFeature — Multiple Parameters

### Before

```python
@feature
def bandpass(data: Data, low: float = 1.0, high: float = 40.0, order: int = 4, **kwargs):
    ...
```

### After

```python
@dataclass
class Bandpass(BaseFeature):
    """Apply a Butterworth bandpass filter.

    Args:
        low: Low-cut frequency in Hz.
        high: High-cut frequency in Hz.
        order: Filter order.

    Example:
        >>> result = cb.feature.Bandpass(low=1.0, high=40.0).apply(data)
    """

    low: float = 1.0
    high: float = 40.0
    order: int = 4

    def __call__(self, data: Data) -> xr.DataArray:
        if "time" not in data.data.dims:
            raise ValueError("data must have 'time' dimension")
        if self.low >= self.high:
            raise ValueError(f"low ({self.low}) must be less than high ({self.high})")
        # ... filter logic ...
        return filtered
```

---

## SplitterFeature — Windowing

Use when the old feature sliced `data` into multiple windows or trials.

### Before

```python
@feature
def sliding_window(data: Data, window_size: int = 10, step_size: int = 5, **kwargs):
    """Slide a window over the time dimension, returning one Data per window."""
    windows = []
    n = data.data.sizes["time"]
    for start in range(0, n - window_size + 1, step_size):
        w = data.data.isel(time=slice(start, start + window_size))
        windows.append(data._copy_with_new_data(new_data=w, operation_name="sliding_window"))
    return windows  # returned a list, not a generator
```

### After

```python
# src/cobrabox/features/sliding_window.py
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from ..base_feature import SplitterFeature
from ..data import Data


@dataclass
class SlidingWindow(SplitterFeature):
    """Yield one Data per sliding window over the time dimension.

    Lazily generates windows to avoid materialising all windows in memory at once.

    Args:
        window_size: Number of timepoints per window.
        step_size: Step between window starts in timepoints.

    Example:
        >>> chord = (
        ...     cb.feature.SlidingWindow(window_size=100, step_size=50)
        ...     | cb.feature.LineLength()
        ...     | cb.feature.MeanAggregate()
        ... )
    """

    window_size: int = field(default=10)
    step_size: int = field(default=5)

    def __call__(self, data: Data) -> Iterator[Data]:
        xr_data = data.data
        if "time" not in xr_data.dims:
            raise ValueError("data must have 'time' dimension")
        n_time = xr_data.sizes["time"]
        if self.window_size > n_time:
            raise ValueError(f"window_size ({self.window_size}) must be <= n_time ({n_time})")
        for start in np.arange(0, n_time - self.window_size + 1, self.step_size):
            window_data = xr_data.isel(time=slice(start, start + self.window_size))
            yield data._copy_with_new_data(new_data=window_data, operation_name="SlidingWindow")
```

**Key differences from old pattern:**
- Returns a **generator** (`yield`), not a list — memory efficient
- Inherits from `SplitterFeature`, not `BaseFeature`
- Never called directly in pipelines — always used as the first step of a Chord via `|`

---

## AggregatorFeature — Folding a Stream

Use when the old feature received a list of windowed `Data` and reduced them to one.

### Before

```python
@feature
def mean_aggregate(data: Data, windows: list[Data], **kwargs) -> Data:
    """Average a list of windowed Data objects."""
    stacked = xr.concat([w.data for w in windows], dim="window")
    averaged = stacked.mean(dim="window")
    return data._copy_with_new_data(new_data=averaged, operation_name="mean_aggregate")
```

### After

```python
# src/cobrabox/features/mean_aggregate.py
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import xarray as xr

from ..base_feature import AggregatorFeature
from ..data import Data


@dataclass
class MeanAggregate(AggregatorFeature):
    """Aggregate a stream of per-window Data by averaging across windows.

    Collects all windows, stacks them along a temporary 'window' dimension,
    and reduces with mean. Per-window pipeline history is propagated to the result.

    Example:
        >>> chord = (
        ...     cb.feature.SlidingWindow(window_size=100, step_size=50)
        ...     | cb.feature.LineLength()
        ...     | cb.feature.MeanAggregate()
        ... )
    """

    def __call__(self, data: Data, stream: Iterator[Data]) -> Data:
        items = list(stream)
        if not items:
            raise ValueError("MeanAggregate received an empty stream")
        stacked = xr.concat([w.data for w in items], dim="window", join="override")
        averaged = stacked.mean(dim="window")
        # Propagate per-window pipeline history
        window_history = [op for op in items[0].history if op not in data.history]
        return Data(
            data=averaged,
            subjectID=data.subjectID,
            groupID=data.groupID,
            condition=data.condition,
            history=list(data.history) + window_history + ["MeanAggregate"],
            extra=data.extra,
        )
```

**Critical rule for `AggregatorFeature`:** Do NOT call `data._copy_with_new_data()`. Build the
`Data` object directly and propagate history manually:
1. Collect `items = list(stream)`
2. Compute `window_history = [op for op in items[0].history if op not in data.history]`
3. Return `Data(..., history=list(data.history) + window_history + ["MyAggregatorName"])`

`join="override"` on `xr.concat` suppresses a FutureWarning when windows have different time
coordinates (which they will, since each window is a slice).

---

## Test Migration

### Before

```python
def test_mean_reduces_time() -> None:
    data = cb.from_numpy(np.ones((10, 4)), dims=["time", "space"])
    result = cb.feature.mean(data, dim="time")
    assert result.history == ["mean"]
    assert "time" not in result.data.dims
```

### After

```python
def test_mean_reduces_time() -> None:
    data = cb.from_numpy(np.ones((10, 4)), dims=["time", "space"])
    result = cb.feature.Mean(dim="time").apply(data)
    assert result.history == ["Mean"]  # PascalCase
    assert "time" not in result.data.dims
```

### SplitterFeature test pattern

```python
def test_sliding_window_yields_correct_count() -> None:
    data = cb.from_numpy(np.ones((20, 4)), dims=["time", "space"])
    windows = list(cb.feature.SlidingWindow(window_size=10, step_size=5)(data))
    assert len(windows) == 3
    assert windows[0].data.sizes["time"] == 10
    assert windows[0].history == ["SlidingWindow"]
```

### AggregatorFeature test pattern (via Chord)

```python
def test_mean_aggregate_via_chord() -> None:
    data = cb.from_numpy(np.ones((20, 4)), dims=["time", "space"])
    result = (
        cb.feature.SlidingWindow(window_size=10, step_size=5)
        | cb.feature.LineLength()
        | cb.feature.MeanAggregate()
    ).apply(data)
    assert result.history == ["SlidingWindow", "LineLength", "MeanAggregate", "Chord"]
```

---

## Common Pitfalls

| Pitfall | Fix |
|---|---|
| Keeping `**kwargs` | Remove it; declare every parameter explicitly as a typed field |
| `data` as a dataclass field | `data` is only the argument to `__call__`, never a field |
| Calling `self.apply()` inside `__call__` | Never; `apply()` calls `__call__`, not the other way |
| Using `data._copy_with_new_data` in `AggregatorFeature` | Build `Data(...)` directly and set history manually |
| Forgetting `join="override"` in `xr.concat` | Add it to suppress FutureWarning on mismatched coords |
| History still lowercase after migration | History is the class name — PascalCase |
| `SplitterFeature` called with `.apply()` | Call it directly: `splitter(data)` returns `Iterator[Data]` |
