# Test Review: sliding_window

**Feature**: `src/cobrabox/features/sliding_window.py`
**Test file**: `tests/test_feature_sliding_window.py`
**Date**: 2025-03-05
**Verdict**: PASS

## Coverage

```
sliding_window.py: 100% (26 statements, 0 missing)
```

## Summary

Excellent test coverage for the SlidingWindow SplitterFeature. The tests comprehensively cover the generator behavior, lazy evaluation, window counting, value verification, metadata preservation, parameter validation, and input immutability.

## Keep

- `test_sliding_window_yields_correct_number_of_windows` — Tests correct window count calculation
- `test_sliding_window_yields_correct_shape_and_values` — Tests each window has correct shape and values
- `test_sliding_window_each_window_is_data` — Tests each yielded item is Data instance
- `test_sliding_window_preserves_metadata` — Tests metadata preservation in each window
- `test_sliding_window_raises_when_window_too_large` — Tests window_size > signal length
- `test_sliding_window_raises_when_window_size_less_than_one` — Tests __post_init__ validation for window_size
- `test_sliding_window_raises_when_step_size_less_than_one` — Tests __post_init__ validation for step_size
- `test_sliding_window_is_lazy` — Tests generator is lazy (doesn't materialize upfront)
- `test_sliding_window_does_not_mutate_input` — Tests input Data is unchanged

## Action List

None.
