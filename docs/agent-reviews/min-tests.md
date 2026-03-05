# Test Review: min

**Feature**: `src/cobrabox/features/min.py`
**Test file**: `tests/test_feature_min.py`
**Date**: 2025-03-05
**Verdict**: NEEDS WORK

## Coverage

```
Min: 100% (12 statements, 0 missing)
```

## Summary

The `Min` feature has 100% coverage and tests the core functionality well. However, there are naming convention violations (using `test_feature_min_*` instead of `test_min_*`), incomplete metadata preservation tests (only `subjectID` checked), and a missing "no mutation of input" test. The tests use inconsistent data construction patterns (some use `cb.SignalData` constructor, others use `cb.SignalData.from_numpy`).

## Keep

Tests that are correct and complete:
- `test_feature_min_reduces_extra_dimension` — correctly tests reduction over non-time dim and history
- `test_feature_min_raises_for_unknown_dimension` — proper error case with `match=`
- `test_feature_min_single_channel_timeseries_returns_single_value` — good edge case for single-value output
- `test_feature_min_finds_smallest_value_with_negative_numbers` — correctly verifies min with negative values

## Fix

### `test_feature_min_reduces_extra_dimension`
Issue: Incomplete metadata preservation check. Only verifies `subjectID`, missing `groupID`, `condition`, `sampling_rate`.
```python
def test_min_reduces_extra_dimension() -> None:
    """Min reduces an extra dimension (run_index) and updates history."""
    import xarray as xr

    arr = np.arange(24, dtype=float).reshape(3, 4, 2)  # run_index, time, space
    xr_data = xr.DataArray(arr, dims=["run_index", "time", "space"])
    data = cb.SignalData(
        xr_data,
        sampling_rate=100.0,
        subjectID="sub-01",
        groupID="group-A",
        condition="rest",
    )

    out = cb.feature.Min(dim="run_index").apply(data)

    assert isinstance(out, cb.Data)
    assert "run_index" not in out.data.dims
    assert out.data.shape == (2, 4)
    np.testing.assert_allclose(out.to_numpy(), arr.min(axis=0).T)
    # Metadata preservation
    assert out.subjectID == "sub-01"
    assert out.groupID == "group-A"
    assert out.condition == "rest"
    assert out.sampling_rate == pytest.approx(100.0)
    assert out.history == ["Min"]
```

### Naming convention
Issue: All test functions use `test_feature_min_*` instead of `test_min_*` per project conventions.
Fix: Rename all functions:
- `test_feature_min_reduces_extra_dimension` → `test_min_reduces_extra_dimension`
- `test_feature_min_raises_for_unknown_dimension` → `test_min_raises_for_unknown_dimension`
- `test_feature_min_single_channel_timeseries_returns_single_value` → `test_min_single_channel_timeseries_returns_single_value`
- `test_feature_min_finds_smallest_value_with_negative_numbers` → `test_min_finds_smallest_value_with_negative_numbers`

## Add

Missing scenarios:

### `test_min_does_not_mutate_input`
```python
def test_min_does_not_mutate_input() -> None:
    """Min.apply() leaves the input Data object unchanged."""
    data = cb.SignalData.from_numpy(
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        dims=["time", "space"],
        sampling_rate=100.0,
    )
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.Min(dim="time").apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
```

### `test_min_preserves_metadata`
Dedicated test for metadata preservation (even though partially covered in existing tests):
```python
def test_min_preserves_metadata() -> None:
    """Min preserves all metadata fields correctly."""
    data = cb.SignalData.from_numpy(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        dims=["time", "space"],
        sampling_rate=250.0,
        subjectID="s42",
        groupID="control",
        condition="task",
    )
    result = cb.feature.Min(dim="time").apply(data)
    assert result.subjectID == "s42"
    assert result.groupID == "control"
    assert result.condition == "task"
    # sampling_rate preserved since time dim is removed, so None
    assert result.sampling_rate is None
```

## Action List

1. [Severity: MEDIUM] Rename all test functions from `test_feature_min_*` to `test_min_*` pattern (file: `tests/test_feature_min.py`)
2. [Severity: MEDIUM] Add complete metadata preservation assertions (`groupID`, `condition`, `sampling_rate`) to existing tests or add dedicated `test_min_preserves_metadata` test
3. [Severity: MEDIUM] Add `test_min_does_not_mutate_input` to verify input data is not modified
4. [Severity: LOW] Standardize data construction to use `cb.SignalData.from_numpy()` consistently instead of mixing `cb.SignalData()` constructor and `.from_numpy()`
