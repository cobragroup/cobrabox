# Test Review: min

**Feature**: `src/cobrabox/features/min.py`
**Test file**: `tests/test_feature_min.py`
**Date**: 2025-03-05
**Verdict**: PASS

## Coverage

```text
min.py: 100% (12 statements, 0 missing)
```

## Summary

The test file for Min is comprehensive with good coverage of all required scenarios. Tests verify correct minimum computation, dimension reduction, error handling, metadata preservation, and input immutability.

## Keep

- `test_min_reduces_extra_dimension` — Tests dimension reduction with 3D data, verifies shape, dims, values, and all metadata
- `test_min_raises_for_unknown_dimension` — Tests ValueError for missing dimension
- `test_min_single_channel_timeseries_returns_single_value` — Tests single-channel reduction over time
- `test_min_finds_smallest_value_with_negative_numbers` — Tests correctness with negative values
- `test_min_does_not_mutate_input` — Verifies input Data is unchanged
- `test_min_preserves_metadata` — Tests all metadata preservation

## Action List

None.
