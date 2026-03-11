# Test Review: max

**Feature**: `src/cobrabox/features/max.py`
**Test file**: `tests/test_feature_max.py`
**Date**: 2026-03-06
**Verdict**: PASS

## Coverage

```text
max.py: 100% (14 statements, 0 missing)
```

## Summary

The test suite for `Max` is comprehensive and well-structured. All 6 required scenarios are covered, with 100% line coverage. Tests use proper naming conventions, include docstrings, and verify both happy paths and edge cases. The feature is a simple `BaseFeature[Data]` that reduces over a dimension, and the tests appropriately verify dimension removal, error handling for missing dimensions, metadata preservation, sampling_rate behavior, and input immutability.

## Keep

Tests that are correct and complete — no changes needed:

- `test_max_reduces_extra_dimension` — Correctly tests reducing a non-time dimension (run_index), verifies shape transformation, history update, and metadata preservation.
- `test_max_raises_for_unknown_dimension` — Properly tests the ValueError when a dimension doesn't exist.
- `test_max_single_channel_timeseries_returns_single_value` — Tests the common case of reducing time dimension on single-channel data.
- `test_max_does_not_mutate_input` — Thoroughly verifies input immutability by checking history, shape, and values.
- `test_max_preserves_metadata` — Tests that subjectID, groupID, condition are preserved and sampling_rate becomes None when time is removed.
- `test_max_sampling_rate_preserved_when_time_kept` — Tests that sampling_rate is preserved when the reduced dimension is not time.

## Fix

None.

## Add

None — all required scenarios are covered.

## Action List

None.
