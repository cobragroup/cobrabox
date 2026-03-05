# Test Review: mean

**Feature**: `src/cobrabox/features/mean.py`
**Test file**: `tests/test_feature_mean.py`
**Date**: 2025-03-05
**Verdict**: NEEDS WORK

## Coverage

```
Mean: 100% (12 statements, 0 missing)
```

Per-file coverage is at 100%.

## Summary

The existing tests cover the core happy path and error handling for the `Mean` feature. However, **critical quality scenarios are missing**: full metadata preservation verification and input immutability testing. The tests also lack explicit checks for `groupID` and `condition` preservation, which are required per the test criteria.

## Keep

Tests that are correct and complete:

- `test_feature_mean_reduces_extra_dimension` — Correctly tests dimension reduction, shape verification, history update, and basic metadata preservation (`subjectID`).
- `test_feature_mean_raises_for_unknown_dimension` — Properly tests the `ValueError` guard for missing dimensions with `match=`.
- `test_feature_mean_single_channel_timeseries_returns_single_value` — Good edge case test for single-channel, single-value output.

## Fix

None required — existing tests are correct.

## Add

Missing scenarios — new tests to add:

### `test_feature_mean_metadata_preserved`

Tests full metadata preservation including `groupID`, `condition`, and `sampling_rate`.

```python
def test_feature_mean_metadata_preserved() -> None:
    """Mean preserves subjectID, groupID, condition, and sampling_rate."""
    arr = np.arange(24, dtype=float).reshape(3, 4, 2)
    data = cb.SignalData.from_numpy(
        arr,
        dims=["run_index", "time", "space"],
        sampling_rate=100.0,
        subjectID="sub-01",
        groupID="group-A",
        condition="rest",
    )

    out = cb.feature.Mean(dim="run_index").apply(data)

    assert out.subjectID == "sub-01"
    assert out.groupID == "group-A"
    assert out.condition == "rest"
    # sampling_rate is preserved if time dim exists, otherwise None
    if "time" in out.data.dims:
        assert out.sampling_rate == pytest.approx(100.0)
```

### `test_feature_mean_does_not_mutate_input`

Tests that the input Data object is unchanged after `.apply()`.

```python
def test_feature_mean_does_not_mutate_input() -> None:
    """Mean.apply() leaves the input Data object unchanged."""
    arr = np.arange(24, dtype=float).reshape(3, 4, 2)
    data = cb.SignalData.from_numpy(
        arr,
        dims=["run_index", "time", "space"],
        sampling_rate=100.0,
        subjectID="sub-01",
    )

    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.Mean(dim="run_index").apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
    assert data.subjectID == "sub-01"
```

## Action List

1. [Severity: MEDIUM] Add `test_feature_mean_metadata_preserved` to `tests/test_feature_mean.py` — verifies `groupID`, `condition`, and `sampling_rate` preservation
2. [Severity: MEDIUM] Add `test_feature_mean_does_not_mutate_input` to `tests/test_feature_mean.py` — verifies input Data object is not modified
