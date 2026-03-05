# Test Review: max

**Feature**: `src/cobrabox/features/max.py`
**Test file**: `tests/test_feature_max.py`
**Date**: 2025-03-05
**Verdict**: NEEDS WORK

## Coverage

```
max.py: 100% (12 statements, 0 missing)
```

## Summary

The test file has 100% line coverage but misses several required test patterns:
1. No mutation test to verify input Data is unchanged
2. No comprehensive metadata preservation test (groupID, condition, sampling_rate)
3. Missing helper function for cleaner test setup
4. Test names don't follow the `test_<feature>_<scenario>` pattern

## Keep

Tests that are correct and complete:
- `test_feature_max_reduces_extra_dimension` — correctly tests dimension reduction, history, and subjectID preservation
- `test_feature_max_raises_for_unknown_dimension` — correctly tests ValueError for missing dimension
- `test_feature_max_single_channel_timeseries_returns_single_value` — correctly tests single-value output case

## Fix

### Test naming convention
Issue: Test names use `test_feature_max_...` instead of `test_max_...`

**Current:**
```python
def test_feature_max_reduces_extra_dimension() -> None:
def test_feature_max_raises_for_unknown_dimension() -> None:
def test_feature_max_single_channel_timeseries_returns_single_value() -> None:
```

**Fixed:**
```python
def test_max_reduces_extra_dimension() -> None:
def test_max_raises_for_unknown_dimension() -> None:
def test_max_single_channel_timeseries_returns_single_value() -> None:
```

## Add

Missing scenarios — new tests to add:

### `test_max_does_not_mutate_input`

```python
def test_max_does_not_mutate_input() -> None:
    """Max.apply() leaves the input Data object unchanged."""
    data = cb.SignalData.from_numpy(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        dims=["time", "space"],
        sampling_rate=100.0,
        subjectID="s1",
        groupID="g1",
        condition="rest",
    )
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.Max(dim="time").apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
```

### `test_max_preserves_metadata`

```python
def test_max_preserves_metadata() -> None:
    """Max preserves subjectID, groupID, condition; sampling_rate becomes None when time removed."""
    data = cb.SignalData.from_numpy(
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        dims=["time", "space"],
        sampling_rate=250.0,
        subjectID="s42",
        groupID="control",
        condition="task",
    )
    result = cb.feature.Max(dim="time").apply(data)
    assert result.subjectID == "s42"
    assert result.groupID == "control"
    assert result.condition == "task"
    # Max removes the time dimension, so sampling_rate should be None
    assert result.sampling_rate is None
```

### `test_max_sampling_rate_preserved_when_time_kept`

```python
def test_max_sampling_rate_preserved_when_time_kept() -> None:
    """Max preserves sampling_rate when time dimension is not the reduced dimension."""
    import xarray as xr

    arr = np.arange(24, dtype=float).reshape(3, 4, 2)  # run_index, time, space
    xr_data = xr.DataArray(arr, dims=["run_index", "time", "space"])
    data = cb.SignalData(xr_data, sampling_rate=100.0, subjectID="s1")

    result = cb.feature.Max(dim="run_index").apply(data)

    # Time dimension is preserved, so sampling_rate should be kept
    assert result.sampling_rate == 100.0
```

## Action List

1. [Severity: MEDIUM] Rename tests to follow `test_max_<scenario>` pattern (remove `feature_` prefix)
2. [Severity: MEDIUM] Add `test_max_does_not_mutate_input` to verify input Data is not modified
3. [Severity: MEDIUM] Add `test_max_preserves_metadata` to verify all metadata fields are preserved correctly
4. [Severity: MEDIUM] Add `test_max_sampling_rate_preserved_when_time_kept` to verify sampling_rate behavior when time dim is kept
5. [Severity: LOW] Consider adding a `_make_data` helper function for cleaner test setup (optional)
