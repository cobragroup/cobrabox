# Test Review: min

**Feature**: `src/cobrabox/features/min.py`
**Test file**: `tests/test_feature_min.py`
**Date**: 2026-03-04
**Verdict**: NEEDS WORK

## Summary

The Min feature tests cover 4 out of 7 required scenarios. The test suite correctly validates dimension reduction, error handling for unknown dimensions, and history tracking. However, it lacks critical tests for input immutability and has incomplete metadata preservation coverage (only checks `subjectID`, missing `groupID`, `condition`, and `sampling_rate` handling when time dimension is removed).

## Keep

Tests that are correct and complete — no changes needed:

- `test_feature_min_reduces_extra_dimension` — correctly tests reducing non-time dimensions and verifies history, output shape, and basic metadata preservation (subjectID)
- `test_feature_min_raises_for_unknown_dimension` — correctly tests the ValueError guard with match pattern
- `test_feature_min_finds_smallest_value_with_negative_numbers` — correctly validates min computation with negative values

## Fix

Tests that exist but need changes:

### `test_feature_min_reduces_extra_dimension`

Issue: Incomplete metadata preservation check

```python
# Current only checks:
assert out.subjectID == "sub-01"

# Should check all metadata fields:
assert out.subjectID == "sub-01"
assert out.groupID == "g1"  # needs to be set in test data
assert out.condition == "rest"  # needs to be set in test data
# sampling_rate behavior depends on whether time dim exists in output
if "time" in out.data.dims:
    assert out.sampling_rate == pytest.approx(100.0)
else:
    assert out.sampling_rate is None
```

### `test_feature_min_single_channel_timeseries_returns_single_value`

Issue: Line 44 assertion `out.data.dims == ("space", "time")` looks incorrect — when reducing over "time", the time dimension should be removed. This assertion passes but suggests the data structure may be unexpected. Verify this is intentional.

## Add

Missing scenarios — new tests to add:

### `test_feature_min_does_not_mutate_input`

```python
def test_feature_min_does_not_mutate_input() -> None:
    """Min.apply() leaves the input Data object unchanged."""
    arr = np.arange(24, dtype=float).reshape(3, 4, 2)
    data = cb.SignalData.from_numpy(
        arr,
        dims=["run_index", "time", "space"],
        sampling_rate=100.0,
        subjectID="s1",
        groupID="g1",
        condition="rest"
    )
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.Min(dim="run_index").apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
```

### `test_feature_min_metadata_preserved`

```python
def test_feature_min_metadata_preserved() -> None:
    """Min preserves all metadata fields correctly."""
    arr = np.arange(24, dtype=float).reshape(3, 4, 2)
    data = cb.SignalData.from_numpy(
        arr,
        dims=["run_index", "time", "space"],
        sampling_rate=250.0,
        subjectID="sub-42",
        groupID="control",
        condition="task"
    )

    # Reduce over run_index - time should still exist
    out = cb.feature.Min(dim="run_index").apply(data)
    assert out.subjectID == "sub-42"
    assert out.groupID == "control"
    assert out.condition == "task"
    assert out.sampling_rate == pytest.approx(250.0)

    # Reduce over time - sampling_rate should become None
    out2 = cb.feature.Min(dim="time").apply(data)
    assert out2.subjectID == "sub-42"
    assert out2.groupID == "control"
    assert out2.condition == "task"
    assert out2.sampling_rate is None
```

## Action List

1. [Severity: HIGH] Add `test_feature_min_does_not_mutate_input` to verify input immutability
2. [Severity: MEDIUM] Add `test_feature_min_metadata_preserved` to verify all metadata fields and sampling_rate behavior
3. [Severity: MEDIUM] Update existing tests to include full metadata in test data (groupID, condition) so metadata preservation can be properly tested
