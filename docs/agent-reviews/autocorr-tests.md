# Test Review: autocorr

**Feature**: `src/cobrabox/features/autocorr.py`
**Test file**: `tests/test_feature_autocorr.py`
**Date**: 2025-03-05
**Verdict**: PASS

## Coverage

```
autocorr.py: 100% (95 statements, 0 missing)
```

## Summary

Excellent test coverage with 11 tests. Tests validate autocorrelation computation, lag parameter handling (both lag_steps and lag_ms), dimension reduction, error cases (unknown dim, both lag inputs, out-of-range lag), NaN handling, and metadata preservation. Comprehensive parameter validation testing.

## Keep

- `test_feature_autocorr_reduces_requested_dimension` — basic functionality
- `test_feature_autocorr_default_5ms_matches_explicit_steps` — lag_ms vs lag_steps
- `test_feature_autocorr_raises_for_unknown_dimension` — dimension validation
- `test_feature_autocorr_raises_when_both_lag_inputs_provided` — mutual exclusivity
- `test_feature_autocorr_raises_when_lag_out_of_range` — lag bounds checking
- `test_feature_autocorr_constant_signal_returns_nan` — edge case (zero variance)
- `test_feature_autocorr_all_nan_returns_nan` — NaN input handling
- `test_feature_autocorr_fs_non_positive_raises` — fs parameter validation
- `test_feature_autocorr_metadata_preserved` — metadata propagation
- `test_feature_autocorr_does_not_mutate_input` — input immutability
- `test_feature_autocorr_lag_ms_parameter` — lag_ms computation

## Fix

None. All tests are correct and complete.

## Add

None. All required scenarios covered.

## Action List

None.
