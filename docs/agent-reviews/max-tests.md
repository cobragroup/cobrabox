# Test Review: max

**Feature**: `src/cobrabox/features/max.py`
**Test file**: `tests/test_feature_max.py`
**Date**: 2025-03-05
**Verdict**: PASS

## Coverage

```text
max.py: 100% (14 statements, 0 missing)
```

## Summary

The test file for Max is well-structured and comprehensive. All required test scenarios are covered with good assertions. The feature is a simple dimension-reduction operation and the tests verify correct behavior across various scenarios.

## Keep

- `test_max_reduces_extra_dimension` — Tests dimension reduction with 3D data, verifies shape, dims, values, metadata preservation, and history
- `test_max_raises_for_unknown_dimension` — Tests ValueError for missing dimension
- `test_max_single_channel_timeseries_returns_single_value` — Tests single-channel reduction, correct shape, and history
- `test_max_does_not_mutate_input` — Verifies input Data is unchanged after apply
- `test_max_preserves_metadata` — Tests metadata preservation (subjectID, groupID, condition) and sampling_rate becomes None when time removed
- `test_max_sampling_rate_preserved_when_time_kept` — Tests sampling_rate preservation when time dimension is not reduced

## Action List

None.
