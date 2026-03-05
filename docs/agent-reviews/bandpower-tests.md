# Test Review: bandpower

**Feature**: `src/cobrabox/features/bandpower.py`
**Test file**: `tests/test_feature_bandpower.py`
**Date**: 2025-03-05
**Verdict**: NEEDS WORK

## Coverage

```
Bandpower: 100% (45 statements, 0 missing)
```

Per-file coverage meets the ≥95% threshold.

## Summary

Test suite is comprehensive with 17 tests covering happy paths, edge cases, error handling, and parameter validation. All tests have proper docstrings and follow naming conventions. Two gaps identified: metadata preservation test is incomplete (missing groupID and condition checks), and there's no test for input mutation safety.

## Keep

Tests that are correct and complete — no changes needed:

- `test_bandpower_default_dims_and_shape` — verifies output dims and shape
- `test_bandpower_default_band_index_coords` — checks default band names in order
- `test_bandpower_custom_range_shape` — custom band spec produces correct shape
- `test_bandpower_mixed_spec_shape` — mixed True + custom range handling
- `test_bandpower_alpha_dominates_for_10hz_sine` — physics-based value validation
- `test_bandpower_true_alias_matches_explicit_range` — True alias equivalence
- `test_bandpower_all_positive_values` — power is always non-negative
- `test_bandpower_empty_bands_equals_none` — empty dict equals None behavior
- `test_bandpower_nperseg_changes_nothing_in_shape` — nperseg only affects quality
- `test_bandpower_history_appended` — history tracking works
- `test_bandpower_raises_when_sampling_rate_missing` — error when no sampling_rate
- `test_bandpower_raises_for_true_with_unknown_band` — error for unknown True band
- `test_bandpower_raises_when_nperseg_less_than_2` — __post_init__ validation
- `test_bandpower_raises_for_false_band_spec` — error for False value
- `test_bandpower_transposes_when_time_not_last` — auto-transpose behavior
- `test_bandpower_zeros_when_no_freq_bins_in_band` — zero power for out-of-range bands

## Fix

Tests that exist but need changes:

### `test_bandpower_metadata_preserved`

**Issue**: Only checks subjectID and sampling_rate; missing groupID and condition assertions per criteria.

```python
def test_bandpower_metadata_preserved() -> None:
    """subjectID, groupID, condition, and sampling_rate are carried through."""
    data = _sine_data(freq_hz=10.0, subjectID="sub-42")
    # Set groupID and condition on the data
    data = cb.SignalData.from_numpy(
        data.to_numpy(),
        dims=list(data.data.dims),
        sampling_rate=data.sampling_rate,
        subjectID="sub-42",
        groupID="group-A",
        condition="rest",
    )
    out = cb.feature.Bandpower().apply(data)

    assert out.subjectID == "sub-42"
    assert out.groupID == "group-A"
    assert out.condition == "rest"
    assert out.sampling_rate == 256.0
```

## Add

Missing scenarios — new tests to add:

### `test_bandpower_does_not_mutate_input`

```python
def test_bandpower_does_not_mutate_input() -> None:
    """Bandpower.apply() leaves the input Data object unchanged."""
    data = _sine_data(freq_hz=10.0)
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.Bandpower().apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
```

## Action List

1. [Severity: MEDIUM] Extend `test_bandpower_metadata_preserved` to assert groupID and condition preservation (`tests/test_feature_bandpower.py:151-157`)
2. [Severity: MEDIUM] Add `test_bandpower_does_not_mutate_input` to verify input Data immutability (`tests/test_feature_bandpower.py`)
