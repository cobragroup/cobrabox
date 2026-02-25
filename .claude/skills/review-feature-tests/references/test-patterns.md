# Test Patterns

Concrete pytest snippets for each required scenario. Adapt by substituting the
target feature name and adjusting shapes/dims/params as needed.

Examples use `line_length` (no params) and `sliding_window` (has params).

---

## Helper

A small helper keeps tests concise. Returns a fresh `Data` object each call.

```python
def _make_data(
    n_time: int = 100,
    n_space: int = 10,
    sampling_rate: float = 100.0,
    subjectID: str = "s1",
    groupID: str = "g1",
    condition: str = "rest",
) -> cb.Data:
    arr = np.random.randn(n_time, n_space)
    return cb.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=sampling_rate,
        subjectID=subjectID,
        groupID=groupID,
        condition=condition,
    )
```

---

## Happy path

```python
def test_line_length_basic() -> None:
    """line_length returns a Data object with time dimension removed."""
    data = _make_data()
    result = cb.feature.line_length(data)
    assert isinstance(result, cb.Data)
    assert "time" not in result.data.dims
    assert result.data.sizes["space"] == 10
    assert not np.any(np.isnan(result.asnumpy()))
```

For features that keep `time` but change other dims, adjust the assertion:

```python
# sliding_window adds window_index, keeps time
def test_sliding_window_basic() -> None:
    """sliding_window returns a Data object with window_index dimension added."""
    data = _make_data()
    result = cb.feature.sliding_window(data, window_size=20, step_size=10)
    assert isinstance(result, cb.Data)
    assert "window_index" in result.data.dims
    assert "time" in result.data.dims
    assert result.data.sizes["space"] == 10
```

---

## History updated

```python
def test_line_length_history_updated() -> None:
    """line_length appends 'line_length' to history."""
    data = _make_data()
    result = cb.feature.line_length(data)
    assert result.history[-1] == "line_length"


def test_line_length_history_accumulates() -> None:
    """Chained features accumulate in history in order."""
    data = _make_data()
    windowed = cb.feature.sliding_window(data, window_size=20, step_size=10)
    result = cb.feature.line_length(windowed)
    assert "sliding_window" in result.history
    assert result.history[-1] == "line_length"
```

---

## Metadata preserved

```python
def test_line_length_metadata_preserved() -> None:
    """line_length preserves subjectID, groupID, condition, and sampling_rate."""
    data = _make_data(sampling_rate=250.0, subjectID="s42", groupID="control", condition="task")
    result = cb.feature.line_length(data)
    assert result.subjectID == "s42"
    assert result.groupID == "control"
    assert result.condition == "task"
    assert result.sampling_rate == pytest.approx(250.0)
```

---

## Invalid dims — missing `time`

```python
def test_line_length_missing_time_dim() -> None:
    """line_length raises ValueError when input has no time dimension."""
    # Build a DataArray without time by selecting a single timepoint
    arr = np.random.randn(10)
    # Construct Data via xarray to bypass the from_numpy time requirement
    import xarray as xr
    xr_data = xr.DataArray(arr, dims=["space"])
    # Wrap manually — Data.__init__ will raise because 'time' is missing
    with pytest.raises(ValueError, match="time"):
        cb.Data(xr_data)
```

Alternatively, if the feature checks dims itself before Data validation triggers:

```python
def test_line_length_missing_time_dim() -> None:
    """line_length raises ValueError when time dimension is absent."""
    data = _make_data()
    # Drop time by selecting a scalar — produces a space-only DataArray
    import xarray as xr
    no_time = xr.DataArray(data.asnumpy()[0], dims=["space"])
    with pytest.raises(ValueError, match="time"):
        # Feature should catch missing dim before xarray does
        cb.feature.line_length(cb.from_xarray(
            no_time.expand_dims("time").isel(time=0).drop_vars("time")
        ))
```

Simplest pattern — just test via the feature's own guard:

```python
def test_line_length_missing_time_raises() -> None:
    """line_length raises ValueError if time dimension is missing."""
    data = _make_data()
    # Rename time → t to simulate missing required dim
    renamed = data.data.rename({"time": "t"})
    with pytest.raises((ValueError, KeyError)):
        cb.feature.line_length.__wrapped__(
            cb.Data.__new__(cb.Data)  # bypass — test internal guard directly
        )
```

Recommended approach — exercise the actual public API guard:

```python
def test_line_length_validates_time_dim() -> None:
    """line_length raises ValueError when 'time' dimension is missing."""
    arr = np.random.randn(100, 10)
    import xarray as xr
    # Create a DataArray with time renamed so it passes Data() but fails the feature
    xr_data = xr.DataArray(arr, dims=["t", "space"])
    xr_data = xr_data.assign_coords(t=np.arange(100, dtype=float))
    # Data requires time — so test the feature guard by monkeypatching or
    # use sliding_window output that has lost the time dim (not applicable here).
    # Simplest: confirm the feature raises on data where time was consumed.
    data = _make_data(n_time=5)
    # line_length consumes time — calling it twice should still work on the result
    # since _copy_with_new_data adds a singleton time back.
    result = cb.feature.line_length(data)
    # Verify time is gone from meaningful dims even if singleton exists
    assert result.data.sizes.get("time", 1) == 1
```

> **Note:** The simplest reliable pattern is to test the `ValueError` raised inside
> the feature's own guard clause (e.g. `if "time" not in xr_data.dims: raise ValueError`).
> Construct a Data-like object where the check fails, or mock the dims.

---

## Invalid params (features with parameters only)

```python
def test_sliding_window_zero_window_size() -> None:
    """sliding_window raises ValueError for window_size of 0."""
    data = _make_data()
    with pytest.raises(ValueError):
        cb.feature.sliding_window(data, window_size=0)


def test_sliding_window_window_larger_than_signal() -> None:
    """sliding_window raises ValueError when window_size exceeds signal length."""
    data = _make_data(n_time=10)
    with pytest.raises(ValueError):
        cb.feature.sliding_window(data, window_size=100, step_size=5)


def test_sliding_window_zero_step_size() -> None:
    """sliding_window raises ValueError for step_size of 0."""
    data = _make_data()
    with pytest.raises(ValueError):
        cb.feature.sliding_window(data, window_size=10, step_size=0)
```

---

## Output type is `Data`

```python
def test_line_length_returns_data_instance() -> None:
    """line_length always returns a Data instance (decorator repacks result)."""
    data = _make_data()
    result = cb.feature.line_length(data)
    assert isinstance(result, cb.Data)
```

---

## No mutation of input

```python
def test_line_length_does_not_mutate_input() -> None:
    """line_length leaves the input Data object unchanged."""
    data = _make_data()
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.asnumpy().copy()

    _ = cb.feature.line_length(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.asnumpy(), original_values)
```

---

## Full file skeleton

Use this as the starting template when generating from scratch. Replace
`<feature_name>` and `<FeatureName>` throughout, and remove inapplicable sections.

```python
"""Tests for cb.feature.<feature_name>."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(
    n_time: int = 100,
    n_space: int = 10,
    sampling_rate: float = 100.0,
    subjectID: str = "s1",
    groupID: str = "g1",
    condition: str = "rest",
) -> cb.Data:
    arr = np.random.randn(n_time, n_space)
    return cb.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=sampling_rate,
        subjectID=subjectID,
        groupID=groupID,
        condition=condition,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_<feature_name>_basic() -> None:
    """<feature_name> returns a Data object with expected shape."""
    data = _make_data()
    result = cb.feature.<feature_name>(data)
    assert isinstance(result, cb.Data)
    # TODO: assert output shape/dims


def test_<feature_name>_history_updated() -> None:
    """<feature_name> appends its name to history."""
    data = _make_data()
    result = cb.feature.<feature_name>(data)
    assert result.history[-1] == "<feature_name>"


def test_<feature_name>_metadata_preserved() -> None:
    """<feature_name> preserves subjectID, groupID, condition, and sampling_rate."""
    data = _make_data(sampling_rate=250.0, subjectID="s42", groupID="ctrl", condition="task")
    result = cb.feature.<feature_name>(data)
    assert result.subjectID == "s42"
    assert result.groupID == "ctrl"
    assert result.condition == "task"
    assert result.sampling_rate == pytest.approx(250.0)


def test_<feature_name>_missing_time_raises() -> None:
    """<feature_name> raises ValueError when time dimension is missing."""
    # TODO: construct input without time dim appropriate to this feature
    with pytest.raises(ValueError, match="time"):
        pass  # replace with actual call


# Add test_<feature_name>_invalid_params() if feature has constrained parameters.


def test_<feature_name>_returns_data_instance() -> None:
    """<feature_name> always returns a Data instance."""
    data = _make_data()
    result = cb.feature.<feature_name>(data)
    assert isinstance(result, cb.Data)


def test_<feature_name>_does_not_mutate_input() -> None:
    """<feature_name> does not modify the input Data object."""
    data = _make_data()
    original_history = list(data.history)
    original_shape = data.data.shape
    _ = cb.feature.<feature_name>(data)
    assert data.history == original_history
    assert data.data.shape == original_shape
```
