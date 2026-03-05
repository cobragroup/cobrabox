# Test Review: line_length

**Feature**: `src/cobrabox/features/line_length.py`
**Test file**: `tests/test_feature_line_length.py`
**Date**: 2026-03-05
**Verdict**: NEEDS WORK

## Coverage

```
LineLength: 100% (13 statements, 0 missing)
```

## Summary

The LineLength feature tests cover the core functionality well with 100% code coverage. All existing tests verify correct output values, shape/dimension handling, history tracking, and Chord integration. However, several required test scenarios from the criteria are missing: input mutation verification, full metadata preservation checks, and error handling for invalid dimensions.

## Keep

Tests that are correct and complete — no changes needed:

- `test_feature_line_length_single_channel_timeseries` — correctly tests single-channel (1D) signals with proper shape assertions and history verification
- `test_feature_line_length_via_chord` — correctly tests Chord composition with SlidingWindow and MeanAggregate

## Fix

Tests that exist but need changes:

### `test_feature_line_length_expected_values_and_history`

**Issue**: Function name uses wrong convention (`test_feature_*` instead of `test_line_length_*`)

**Issue**: Missing groupID and condition metadata preservation checks. Only checks subjectID and sampling_rate.

```python
def test_line_length_expected_values_and_history() -> None:
    """LineLength computes absolute temporal differences and wraps to Data."""
    arr = np.array([[0.0, 1.0], [1.0, 3.0], [3.0, 2.0], [6.0, 2.0]])
    # Expected line length per channel:
    # ch0: |1-0| + |3-1| + |6-3| = 1 + 2 + 3 = 6
    # ch1: |3-1| + |2-3| + |2-2| = 2 + 1 + 0 = 3
    expected = np.array([[6.0], [3.0]])

    data = cb.SignalData.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=200.0,
        subjectID="sub-02",
        groupID="group-A",
        condition="rest",
    )
    out = cb.feature.LineLength().apply(data)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space",)
    assert out.data.shape == (2,)
    np.testing.assert_allclose(out.to_numpy(), expected.ravel())
    assert out.subjectID == "sub-02"
    assert out.groupID == "group-A"
    assert out.condition == "rest"
    assert out.sampling_rate is None
    assert out.history == ["LineLength"]
```

## Add

Missing scenarios — new tests to add:

### `test_line_length_missing_time_raises`

```python
def test_line_length_missing_time_raises() -> None:
    """LineLength raises ValueError when time dimension is missing."""
    import xarray as xr

    bad_xr = xr.DataArray(np.random.randn(100, 10), dims=["t", "space"])
    raw = cb.Data.__new__(cb.Data)
    object.__setattr__(raw, "_data", bad_xr)
    with pytest.raises(ValueError, match="time"):
        cb.feature.LineLength()(raw)
```

### `test_line_length_does_not_mutate_input`

```python
def test_line_length_does_not_mutate_input() -> None:
    """LineLength.apply() leaves the input Data object unchanged."""
    arr = np.array([[0.0, 1.0], [1.0, 3.0], [3.0, 2.0], [6.0, 2.0]])
    data = cb.SignalData.from_numpy(
        arr, dims=["time", "space"], sampling_rate=200.0, subjectID="sub-02"
    )
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.LineLength().apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
```

### `test_line_length_returns_data_instance`

```python
def test_line_length_returns_data_instance() -> None:
    """LineLength.apply() always returns a Data instance."""
    arr = np.array([[0.0, 1.0], [1.0, 3.0], [3.0, 2.0], [6.0, 2.0]])
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=200.0)
    result = cb.feature.LineLength().apply(data)
    assert isinstance(result, cb.Data)
```

## Action List

1. [Severity: MEDIUM] Rename `test_feature_line_length_*` functions to `test_line_length_*` pattern (3 occurrences)
2. [Severity: HIGH] Add groupID and condition assertions to `test_line_length_expected_values_and_history`
3. [Severity: HIGH] Add `test_line_length_missing_time_raises` to test invalid dimension handling
4. [Severity: HIGH] Add `test_line_length_does_not_mutate_input` to verify input immutability
5. [Severity: MEDIUM] Add `test_line_length_returns_data_instance` for explicit output type check
6. [Severity: LOW] Add `import pytest` to imports (needed for pytest.raises)
