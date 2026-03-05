# Test Review: max

**Feature**: `src/cobrabox/features/max.py`
**Test file**: `tests/test_feature_max.py`
**Date**: 2025-03-04
**Verdict**: NEEDS WORK

## Summary

The Max feature has a solid foundation with 3 tests covering basic dimension reduction, error handling for unknown dimensions, and single-channel time-series reduction. However, it's missing a critical "no mutation" test and has incomplete metadata preservation coverage.

## Keep

Tests that are correct and complete:

- `test_feature_max_reduces_extra_dimension` — Correctly tests reducing a non-time dimension, verifies output shape, history update, and subjectID preservation.
- `test_feature_max_raises_for_unknown_dimension` — Properly validates that ValueError is raised with clear message when dimension is missing.
- `test_feature_max_single_channel_timeseries_returns_single_value` — Good edge case testing single-channel, single-value output.

## Fix

None — existing tests are correct but incomplete in coverage.

## Add

Missing required scenarios:

### `test_max_no_mutation`

The input Data object must remain unchanged after `.apply()`.

```python
def test_max_no_mutation() -> None:
    """Max does not modify the input Data object."""
    arr = np.arange(24, dtype=float).reshape(3, 4, 2)
    import xarray as xr
    xr_data = xr.DataArray(arr, dims=["run_index", "time", "space"])
    data = cb.SignalData(xr_data, sampling_rate=100.0, subjectID="sub-01")
    
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()
    
    _ = cb.feature.Max(dim="run_index").apply(data)
    
    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
```

### `test_max_preserves_all_metadata`

Current test only checks `subjectID`. Should verify `groupID`, `condition`, and `sampling_rate` (when time dim present).

```python
def test_max_preserves_all_metadata() -> None:
    """Max preserves all metadata fields."""
    arr = np.arange(24, dtype=float).reshape(3, 4, 2)
    import xarray as xr
    xr_data = xr.DataArray(arr, dims=["run_index", "time", "space"])
    data = cb.SignalData(
        xr_data,
        sampling_rate=100.0,
        subjectID="sub-01",
        groupID="group-A",
        condition="rest"
    )
    
    out = cb.feature.Max(dim="run_index").apply(data)
    
    assert out.subjectID == "sub-01"
    assert out.groupID == "group-A"
    assert out.condition == "rest"
    assert out.sampling_rate == pytest.approx(100.0)
```

## Action List

1. [Severity: HIGH] Add `test_max_no_mutation` to verify input Data is not modified (`tests/test_feature_max.py`)
2. [Severity: MEDIUM] Expand metadata test to check all fields (groupID, condition, sampling_rate) or add dedicated `test_max_preserves_all_metadata` test (`tests/test_feature_max.py`, lines 11-26)
