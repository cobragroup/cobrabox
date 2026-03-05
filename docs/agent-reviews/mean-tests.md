# Test Review: mean

**Feature**: `src/cobrabox/features/mean.py`
**Test file**: `tests/test_feature_mean.py`
**Date**: 2025-03-05
**Verdict**: PASS

## Coverage

```
mean.py: 100% (12 statements, 0 missing)
```

## Summary

The test file for Mean is comprehensive and well-structured. All required test scenarios are covered with proper assertions verifying shape, dims, values, metadata preservation, and error handling.

## Keep

- `test_mean_reduces_extra_dimension` — Tests dimension reduction with 3D data (run_index), verifies shape, dims, values, all metadata preserved
- `test_mean_raises_for_unknown_dimension` — Tests ValueError for missing dimension
- `test_mean_single_channel_timeseries_returns_single_value` — Tests single-channel reduction over time
- `test_mean_metadata_preserved` — Tests all metadata fields preserved (subjectID, groupID, condition, sampling_rate)
- `test_mean_does_not_mutate_input` — Verifies input Data is unchanged after apply

## Action List

None.
