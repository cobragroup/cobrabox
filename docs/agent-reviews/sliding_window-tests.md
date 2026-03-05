# Test Review: sliding_window

**Feature**: `src/cobrabox/features/sliding_window.py`
**Test file**: `tests/test_feature_sliding_window.py`
**Date**: 2025-03-05
**Verdict**: NEEDS WORK

## Coverage

```
sliding_window.py: 100% (26 statements, 0 missing)
```

Coverage meets the ≥95% threshold. All lines are exercised.

## Summary

The test file covers most critical scenarios for the SlidingWindow SplitterFeature:
- Correct window count and slicing logic
- Output shape and value verification
- Metadata preservation across windows
- History propagation
- Parameter validation (__post_init__ and runtime guards)
- Lazy generator behavior

Main gaps: missing test for input immutability, and some tests lack docstrings.

## Keep

Tests that are correct and complete:

- `test_sliding_window_yields_correct_number_of_windows` — Correctly verifies window count formula `(n_time - window_size) // step_size + 1`
- `test_sliding_window_yields_correct_shape_and_values` — Comprehensive shape and value assertions including transposition handling
- `test_sliding_window_each_window_is_data` — Verifies each yielded item is a Data instance
- `test_sliding_window_preserves_metadata` — Checks all metadata fields (subjectID, groupID, condition, sampling_rate) and history
- `test_sliding_window_raises_when_window_too_large` — Runtime guard for window_size > n_time
- `test_sliding_window_raises_when_window_size_less_than_one` — __post_init__ validation
- `test_sliding_window_raises_when_step_size_less_than_one` — __post_init__ validation
- `test_sliding_window_is_lazy` — Verifies generator behavior

## Fix

Tests needing minor improvements:

### `test_sliding_window_is_lazy`
Issue: Missing docstring (only test with one)
```python
def test_sliding_window_is_lazy() -> None:
    """SlidingWindow returns a generator, not a materialized list."""
```

### `test_sliding_window_yields_correct_number_of_windows`
Issue: Missing docstring
```python
def test_sliding_window_yields_correct_number_of_windows() -> None:
    """SlidingWindow yields the expected number of windows."""
```

### `test_sliding_window_yields_correct_shape_and_values`
Issue: Missing docstring
```python
def test_sliding_window_yields_correct_shape_and_values() -> None:
    """Each window has correct shape and contains expected values."""
```

### `test_sliding_window_each_window_is_data`
Issue: Missing docstring
```python
def test_sliding_window_each_window_is_data() -> None:
    """Each yielded window is a Data instance."""
```

### `test_sliding_window_raises_when_window_too_large`
Issue: Missing docstring
```python
def test_sliding_window_raises_when_window_too_large() -> None:
    """SlidingWindow raises ValueError when window_size exceeds signal length."""
```

### `test_sliding_window_raises_when_window_size_less_than_one`
Issue: Missing docstring
```python
def test_sliding_window_raises_when_window_size_less_than_one() -> None:
    """SlidingWindow raises ValueError for window_size < 1."""
```

### `test_sliding_window_raises_when_step_size_less_than_one`
Issue: Missing docstring
```python
def test_sliding_window_raises_when_step_size_less_than_one() -> None:
    """SlidingWindow raises ValueError for step_size < 1."""
```

## Add

Missing scenarios to add:

### `test_sliding_window_does_not_mutate_input`

```python
def test_sliding_window_does_not_mutate_input() -> None:
    """SlidingWindow does not modify the input Data object."""
    arr = np.arange(20, dtype=float).reshape(10, 2)
    data = cb.SignalData.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=100.0,
        subjectID="sub-01",
        groupID="patient",
        condition="rest",
    )
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = list(cb.feature.SlidingWindow(window_size=4, step_size=2)(data))

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
    assert data.subjectID == "sub-01"
    assert data.groupID == "patient"
    assert data.condition == "rest"
    assert data.sampling_rate == 100.0
```

## Action List

1. [Severity: MEDIUM] Add missing docstrings to 7 test functions (test_sliding_window_yields_correct_number_of_windows, test_sliding_window_yields_correct_shape_and_values, test_sliding_window_each_window_is_data, test_sliding_window_raises_when_window_too_large, test_sliding_window_raises_when_window_size_less_than_one, test_sliding_window_raises_when_step_size_less_than_one)

2. [Severity: MEDIUM] Add `test_sliding_window_does_not_mutate_input` to verify input Data is not modified when iterating over windows (tests/test_feature_sliding_window.py)
