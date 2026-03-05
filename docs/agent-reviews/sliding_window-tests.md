# Test Review: sliding_window

**Feature**: `src/cobrabox/features/sliding_window.py`
**Test file**: `tests/test_feature_sliding_window.py`
**Date**: 2025-03-04
**Verdict**: NEEDS WORK

## Summary

The test file covers the core functionality well: happy path, shape/values, metadata preservation, parameter validation, and lazy generator behavior. However, most tests lack docstrings, and there's no explicit test for input immutability. The existing tests are generally correct and follow naming conventions.

## Keep

Tests that are correct and complete — no changes needed:

- `test_sliding_window_yields_correct_number_of_windows` — Correctly verifies window count calculation `(n_time - window_size) // step_size + 1`
- `test_sliding_window_yields_correct_shape_and_values` — Comprehensive shape and value assertions with `np.testing.assert_allclose`
- `test_sliding_window_each_window_is_data` — Verifies each yielded item is a `Data` instance
- `test_sliding_window_is_lazy` — Correctly uses `inspect.isgenerator` to verify lazy behavior

## Fix

Tests that exist but need changes:

### `test_sliding_window_preserves_metadata`

**Issue**: Missing docstring and could be more explicit about sampling_rate preservation

```python
def test_sliding_window_preserves_metadata() -> None:
    """SlidingWindow preserves all metadata on each window."""
    arr = np.arange(20, dtype=float).reshape(10, 2)
    data = cb.SignalData.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=100.0,
        subjectID="sub-01",
        groupID="patient",
        condition="rest",
    )
    for w in cb.feature.SlidingWindow(window_size=4, step_size=2)(data):
        assert w.subjectID == "sub-01"
        assert w.groupID == "patient"
        assert w.condition == "rest"
        assert w.sampling_rate == pytest.approx(100.0)  # explicit approx
        assert w.history == ["SlidingWindow"]
```

### `test_sliding_window_raises_when_window_too_large`

**Issue**: Missing docstring

```python
def test_sliding_window_raises_when_window_too_large() -> None:
    """SlidingWindow raises ValueError when window_size exceeds signal length."""
    data = cb.SignalData.from_numpy(np.ones((5, 2)), dims=["time", "space"])
    with pytest.raises(ValueError, match="window_size"):
        list(cb.feature.SlidingWindow(window_size=10, step_size=1)(data))
```

### `test_sliding_window_raises_when_window_size_less_than_one`

**Issue**: Missing docstring

```python
def test_sliding_window_raises_when_window_size_less_than_one() -> None:
    """SlidingWindow raises ValueError for window_size < 1."""
    with pytest.raises(ValueError, match="window_size must be >= 1"):
        cb.feature.SlidingWindow(window_size=0)
```

### `test_sliding_window_raises_when_step_size_less_than_one`

**Issue**: Missing docstring

```python
def test_sliding_window_raises_when_step_size_less_than_one() -> None:
    """SlidingWindow raises ValueError for step_size < 1."""
    with pytest.raises(ValueError, match="step_size must be >= 1"):
        cb.feature.SlidingWindow(step_size=0)
```

### `test_sliding_window_yields_correct_number_of_windows`

**Issue**: Missing docstring

```python
def test_sliding_window_yields_correct_number_of_windows() -> None:
    """SlidingWindow yields correct number of windows based on formula."""
```

### `test_sliding_window_yields_correct_shape_and_values`

**Issue**: Missing docstring

```python
def test_sliding_window_yields_correct_shape_and_values() -> None:
    """SlidingWindow yields windows with correct shape and values."""
```

### `test_sliding_window_each_window_is_data`

**Issue**: Missing docstring

```python
def test_sliding_window_each_window_is_data() -> None:
    """SlidingWindow yields Data instances for each window."""
```

## Add

Missing scenarios — new tests to add:

### `test_sliding_window_does_not_mutate_input`

```python
def test_sliding_window_does_not_mutate_input() -> None:
    """SlidingWindow does not modify the input Data object."""
    arr = np.arange(20, dtype=float).reshape(10, 2)
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = list(cb.feature.SlidingWindow(window_size=4, step_size=2)(data))

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
```

### `test_sliding_window_preserves_sampling_rate`

```python
def test_sliding_window_preserves_sampling_rate() -> None:
    """SlidingWindow preserves sampling_rate on windows since they retain time dim."""
    data = cb.SignalData.from_numpy(
        np.ones((10, 2)),
        dims=["time", "space"],
        sampling_rate=250.0,
    )
    for w in cb.feature.SlidingWindow(window_size=4, step_size=2)(data):
        assert w.sampling_rate == pytest.approx(250.0)
```

## Action List

1. [Severity: MEDIUM] Add docstrings to all test functions (7 missing)
2. [Severity: MEDIUM] Add `test_sliding_window_does_not_mutate_input` to verify input immutability
3. [Severity: LOW] Add explicit `test_sliding_window_preserves_sampling_rate` or update existing metadata test to use `pytest.approx`
4. [Severity: LOW] Add `import pytest` if not already present (needed for `pytest.approx`)
