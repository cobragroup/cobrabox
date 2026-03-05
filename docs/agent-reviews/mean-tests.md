# Test Review: mean

**Feature**: `src/cobrabox/features/mean.py`
**Test file**: `tests/test_feature_mean.py`
**Date**: 2026-03-04
**Verdict**: NEEDS WORK

## Summary

The test file covers the core functionality well with 3 tests for happy paths (reducing run_index and time dimensions) and error handling (unknown dimension). However, it is missing the "no mutation of input" test which is a required scenario per the criteria. Additionally, metadata preservation is incomplete (only subjectID is checked, not groupID, condition, or sampling_rate).

## Keep

Tests that are correct and complete — no changes needed:

- `test_feature_mean_reduces_extra_dimension` — correctly tests reducing a non-time dimension, shape verification, history tracking, and basic metadata preservation (subjectID). Good use of xarray construction for multi-dimensional test data.

- `test_feature_mean_raises_for_unknown_dimension` — properly tests the ValueError guard with `match=` parameter. Validates the feature's runtime dimension check.

- `test_feature_mean_single_channel_timeseries_returns_single_value` — good edge case test for single-channel time-series data, verifies correct scalar output and shape.

## Fix

Tests that exist but need changes:

### `test_feature_mean_reduces_extra_dimension`

Issue: Incomplete metadata preservation check — only verifies `subjectID`, missing `groupID`, `condition`, and `sampling_rate`.

```python
def test_feature_mean_reduces_extra_dimension() -> None:
    """Mean reduces an extra dimension (run_index) and updates history."""
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

    out = cb.feature.Mean(dim="run_index").apply(data)

    assert isinstance(out, cb.Data)
    assert "run_index" not in out.data.dims
    assert out.data.shape == (2, 4)
    np.testing.assert_allclose(out.to_numpy(), arr.mean(axis=0).T)
    assert out.subjectID == "sub-01"
    assert out.groupID == "group-A"
    assert out.condition == "rest"
    assert out.sampling_rate == pytest.approx(100.0)
    assert out.history == ["Mean"]
```

## Add

Missing scenarios — new tests to add:

### `test_feature_mean_does_not_mutate_input`

```python
def test_feature_mean_does_not_mutate_input() -> None:
    """Mean.apply() leaves the input Data object unchanged."""
    arr = np.arange(24, dtype=float).reshape(3, 4, 2)
    data = cb.SignalData.from_numpy(
        arr,
        dims=["run_index", "time", "space"],
        sampling_rate=100.0,
        subjectID="sub-01",
        groupID="group-A",
        condition="rest",
    )
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.Mean(dim="run_index").apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
    assert data.subjectID == "sub-01"
    assert data.groupID == "group-A"
    assert data.condition == "rest"
    assert data.sampling_rate == pytest.approx(100.0)
```

### `test_feature_mean_preserves_all_metadata`

```python
def test_feature_mean_preserves_all_metadata() -> None:
    """Mean preserves subjectID, groupID, condition, and sampling_rate."""
    data = cb.SignalData.from_numpy(
        np.ones((5, 3)),
        dims=["time", "space"],
        sampling_rate=250.0,
        subjectID="s42",
        groupID="control",
        condition="task",
    )
    result = cb.feature.Mean(dim="time").apply(data)
    assert result.subjectID == "s42"
    assert result.groupID == "control"
    assert result.condition == "task"
    assert result.sampling_rate == pytest.approx(250.0)
```

## Action List

1. [Severity: HIGH] Add `test_feature_mean_does_not_mutate_input` to verify input Data is unchanged after apply (missing required scenario)

2. [Severity: MEDIUM] Update `test_feature_mean_reduces_extra_dimension` to also assert groupID, condition, and sampling_rate are preserved

3. [Severity: LOW] Add `test_feature_mean_preserves_all_metadata` for comprehensive metadata coverage (optional if test 2 is updated)
