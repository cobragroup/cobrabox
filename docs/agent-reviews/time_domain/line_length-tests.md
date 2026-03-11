# Test Review: line_length

**Feature**: `src/cobrabox/features/line_length.py`
**Test file**: `tests/test_feature_line_length.py`
**Date**: 2026-03-06
**Verdict**: NEEDS WORK

## Coverage

```text
LineLength: 100% (13 statements, 0 missing)
```

## Summary

Tests are comprehensive with 100% coverage and cover all required scenarios. However, there is one MEDIUM severity issue: the test for missing time dimension uses an unnecessary `__new__` bypass when `cb.Data.from_numpy` would be cleaner and more idiomatic per the criteria.

## Keep

Tests that are correct and complete — no changes needed:

- `test_line_length_expected_values_and_history` — comprehensive happy path test with value validation, metadata preservation, and history check
- `test_line_length_single_channel_timeseries` — tests edge case of single channel input
- `test_line_length_via_chord` — tests Chord pipeline integration
- `test_line_length_does_not_mutate_input` — verifies immutability with proper snapshot comparisons
- `test_line_length_returns_data_instance` — verifies output type

## Fix

Tests that exist but need changes:

### `test_line_length_missing_time_raises`

Issue: Uses `cb.Data.__new__` bypass which is unnecessary and not idiomatic. Per criteria, use `cb.Data.from_numpy` directly.

```python
def test_line_length_missing_time_raises() -> None:
    """LineLength raises ValueError when time dimension is missing."""
    arr = np.random.default_rng(42).standard_normal((100, 10))
    data = cb.Data.from_numpy(arr, dims=["t", "space"])
    with pytest.raises(ValueError, match="time"):
        cb.feature.LineLength()(data)
```

## Add

Missing scenarios — new tests to add:

None required.

## Action List

1. [Severity: MEDIUM] Fix `test_line_length_missing_time_raises` to use `cb.Data.from_numpy` instead of `__new__` bypass (tests/test_feature_line_length.py:74-82)
