# Test Patterns

Concrete pytest snippets for each required scenario. Adapt by substituting the
target feature name and adjusting shapes/dims/params as needed.

Examples use `LineLength` (no params) and `SlidingWindow` + `MeanAggregate` (have params).

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
    """LineLength returns a Data object with time dimension removed."""
    data = _make_data()
    result = cb.feature.LineLength().apply(data)
    assert isinstance(result, cb.Data)
    assert "time" not in result.data.dims
    assert result.data.sizes["space"] == 10
    assert not np.any(np.isnan(result.asnumpy()))
```

For features that keep `time` but change other dims, adjust the assertion:

```python
# SlidingWindow is a SplitterFeature — test via a Chord
def test_sliding_window_basic() -> None:
    """SlidingWindow inside a Chord produces windowed output."""
    data = _make_data()
    chord = cb.SlidingWindow(window_size=20, step_size=10) | cb.LineLength() | cb.MeanAggregate()
    result = chord.apply(data)
    assert isinstance(result, cb.Data)
    assert result.data.sizes["space"] == 10
```

For a `SplitterFeature` tested directly (as a generator):

```python
def test_sliding_window_yields_data() -> None:
    """SlidingWindow yields Data objects as a generator."""
    data = _make_data()
    windows = list(cb.feature.SlidingWindow(window_size=20, step_size=10)(data))
    assert len(windows) > 0
    assert all(isinstance(w, cb.Data) for w in windows)
    assert all("time" in w.data.dims for w in windows)
```

---

## History updated

History entry is the **class name** (PascalCase), appended by `BaseFeature.apply`.

```python
def test_line_length_history_updated() -> None:
    """LineLength appends 'LineLength' to history."""
    data = _make_data()
    result = cb.feature.LineLength().apply(data)
    assert result.history[-1] == "LineLength"


def test_line_length_history_accumulates() -> None:
    """Chained features accumulate in history in order."""
    data = _make_data()
    chord = cb.feature.SlidingWindow(window_size=20, step_size=10) | \
            cb.feature.LineLength() | \
            cb.feature.MeanAggregate()
    result = chord.apply(data)
    assert "LineLength" in result.history
    assert result.history[-1] == "MeanAggregate"
```

---

## Metadata preserved

```python
def test_line_length_metadata_preserved() -> None:
    """LineLength preserves subjectID, groupID, condition, and sampling_rate."""
    data = _make_data(sampling_rate=250.0, subjectID="s42", groupID="control", condition="task")
    result = cb.feature.LineLength().apply(data)
    assert result.subjectID == "s42"
    assert result.groupID == "control"
    assert result.condition == "task"
    assert result.sampling_rate == pytest.approx(250.0)
```

---

## Invalid dims — missing `time`

Test the feature's own `__call__` guard clause (the `ValueError` it raises before xarray does).

```python
def test_line_length_missing_time_raises() -> None:
    """LineLength raises ValueError when 'time' dimension is missing."""
    import xarray as xr
    arr = np.random.randn(10)
    # Build a Data-like object whose underlying array lacks 'time'
    xr_data = xr.DataArray(arr, dims=["space"])
    # Bypass Data.__init__ validation to isolate the feature guard
    raw = cb.Data.__new__(cb.Data)
    object.__setattr__(raw, "_data", xr_data)
    with pytest.raises(ValueError, match="time"):
        cb.feature.LineLength().apply(raw)
```

Simpler alternative — rename dims on a valid Data object and call directly:

```python
def test_line_length_missing_time_raises() -> None:
    """LineLength raises ValueError when 'time' dimension is absent."""
    data = _make_data()
    # Construct feature and call __call__ directly with a bad xr.DataArray
    import xarray as xr
    bad = xr.DataArray(np.ones((10,)), dims=["space"])
    with pytest.raises((ValueError, KeyError)):
        cb.feature.LineLength()(data)  # still valid; test guard via monkeypatch if needed
```

Recommended: test via the actual public guard in `__call__`:

```python
def test_line_length_validates_time_dim() -> None:
    """LineLength raises ValueError when 'time' is not in data.data.dims."""
    import xarray as xr
    # Build a fresh Data with time renamed to 't' via __new__ bypass
    bad_xr = xr.DataArray(np.random.randn(100, 10), dims=["t", "space"])
    raw = cb.Data.__new__(cb.Data)
    object.__setattr__(raw, "_data", bad_xr)
    with pytest.raises(ValueError, match="time"):
        cb.feature.LineLength()(raw)
```

---

## Invalid params (features with parameters only)

`__post_init__` validation fires at construction time; runtime guards fire in `__call__`.

```python
def test_sliding_window_zero_window_size() -> None:
    """SlidingWindow raises ValueError for window_size of 0."""
    with pytest.raises(ValueError):
        cb.feature.SlidingWindow(window_size=0, step_size=10)


def test_sliding_window_zero_step_size() -> None:
    """SlidingWindow raises ValueError for step_size of 0."""
    with pytest.raises(ValueError):
        cb.feature.SlidingWindow(window_size=10, step_size=0)


def test_sliding_window_window_larger_than_signal() -> None:
    """SlidingWindow raises ValueError when window_size exceeds signal length."""
    data = _make_data(n_time=10)
    with pytest.raises(ValueError):
        list(cb.feature.SlidingWindow(window_size=100, step_size=5)(data))
```

---

## Output type is `Data`

`.apply()` always returns a `Data` instance (`_copy_with_new_data` wraps the result).

```python
def test_line_length_returns_data_instance() -> None:
    """LineLength.apply() always returns a Data instance."""
    data = _make_data()
    result = cb.feature.LineLength().apply(data)
    assert isinstance(result, cb.Data)
```

---

## No mutation of input

```python
def test_line_length_does_not_mutate_input() -> None:
    """LineLength.apply() leaves the input Data object unchanged."""
    data = _make_data()
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.asnumpy().copy()

    _ = cb.feature.LineLength().apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.asnumpy(), original_values)
```

---

## Full file skeleton

Use this as the starting template when generating from scratch. Replace
`<feature_name>` and `<FeatureName>` throughout, and remove inapplicable sections.

```python
"""Tests for cb.feature.<FeatureName>."""

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
    """<FeatureName> returns a Data object with expected shape."""
    data = _make_data()
    result = cb.feature.<FeatureName>().apply(data)
    assert isinstance(result, cb.Data)
    # TODO: assert output shape/dims


def test_<feature_name>_history_updated() -> None:
    """<FeatureName> appends '<FeatureName>' to history."""
    data = _make_data()
    result = cb.feature.<FeatureName>().apply(data)
    assert result.history[-1] == "<FeatureName>"


def test_<feature_name>_metadata_preserved() -> None:
    """<FeatureName> preserves subjectID, groupID, condition, and sampling_rate."""
    data = _make_data(sampling_rate=250.0, subjectID="s42", groupID="ctrl", condition="task")
    result = cb.feature.<FeatureName>().apply(data)
    assert result.subjectID == "s42"
    assert result.groupID == "ctrl"
    assert result.condition == "task"
    assert result.sampling_rate == pytest.approx(250.0)


def test_<feature_name>_missing_time_raises() -> None:
    """<FeatureName> raises ValueError when time dimension is missing."""
    # TODO: construct input without time dim and assert ValueError
    with pytest.raises(ValueError, match="time"):
        pass  # replace with actual call


# Add test_<feature_name>_invalid_params() if feature has constrained fields.


def test_<feature_name>_returns_data_instance() -> None:
    """<FeatureName>.apply() always returns a Data instance."""
    data = _make_data()
    result = cb.feature.<FeatureName>().apply(data)
    assert isinstance(result, cb.Data)


def test_<feature_name>_does_not_mutate_input() -> None:
    """<FeatureName> does not modify the input Data object."""
    data = _make_data()
    original_history = list(data.history)
    original_shape = data.data.shape
    _ = cb.feature.<FeatureName>().apply(data)
    assert data.history == original_history
    assert data.data.shape == original_shape
```
