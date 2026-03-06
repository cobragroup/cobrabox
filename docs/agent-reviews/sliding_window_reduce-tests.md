# Test Review: sliding_window_reduce

**Feature**: `src/cobrabox/features/sliding_window_reduce.py`
**Test file**: `tests/test_feature_sliding_window_reduce.py`
**Date**: 2026-03-05
**Verdict**: PASS

## Coverage

**SlidingWindowReduce**: 100% (34 statements, 0 missing)

✅ Coverage meets ≥95% requirement.

## Summary

Comprehensive test suite for SlidingWindowReduce covering all required scenarios. The tests verify correct computation, dimension handling, all aggregation functions, overlapping windows, metadata preservation, history updates, output type handling, and error cases. All tests are properly named with the feature prefix and include docstrings.

## Keep

Tests that are correct and complete — no changes needed:

- `test_sliding_window_reduce_basic_mean` — correctly computes expected windowed means using a deterministic input sequence (0-99)
- `test_sliding_window_reduce_returns_data_with_window_dim` — verifies 'time' is removed and 'window' is added with correct size
- `test_sliding_window_reduce_all_aggregations` — tests all 5 supported aggregations (mean, std, sum, min, max) with value verification
- `test_sliding_window_reduce_overlapping_windows` — verifies step_size < window_size produces correct number of windows
- `test_sliding_window_reduce_preserves_metadata` — checks subjectID, groupID, condition preserved; sampling_rate becomes None (correct for output_type=Data)
- `test_sliding_window_reduce_updates_history` — verifies history is updated with 'SlidingWindowReduce'
- `test_sliding_window_reduce_returns_data_type` — confirms output is Data (not SignalData) since time is reduced
- `test_sliding_window_reduce_other_dimension` — tests reducing over non-time dimension ('channel')
- `test_sliding_window_reduce_raises_when_window_size_less_than_one` — **post_init** validation for window_size
- `test_sliding_window_reduce_raises_when_step_size_less_than_one` — **post_init** validation for step_size
- `test_sliding_window_reduce_raises_for_invalid_agg` — **post_init** validation for agg parameter
- `test_sliding_window_reduce_raises_when_window_too_large` — runtime guard in **call** for oversized window
- `test_sliding_window_reduce_raises_for_unknown_dimension` — runtime guard for missing dimension
- `test_sliding_window_reduce_does_not_mutate_input` — verifies input Data is not modified after feature application

## Fix

No fixes needed.

## Add

No additional tests needed.

## Action List

None.
