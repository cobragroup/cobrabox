# Test Review: granger_causality

**Feature**: `src/cobrabox/features/connectivity/granger_causality.py`
**Test file**: `tests/features/connectivity/test_feature_granger_causality.py`
**Date**: 2026-03-24
**Verdict**: PASS

## Coverage

```text
GrangerCausality: 100% (88 statements, 0 missing)
```

Per-file coverage is at 100% — exceeds the 95% threshold.

## Summary

The test suite for `GrangerCausality` and `GrangerCausalityMatrix` is comprehensive and well-structured. All 32 tests pass, covering both features' single-lag and multi-lag modes, matrix operations, metadata preservation, history tracking, and parameter validation. The tests correctly verify the Granger causality logic (positive values when y Granger-causes x, directional asymmetry) and include edge cases for small datasets and invalid inputs.

## Keep

Tests that are correct and complete — no changes needed:

- `test_granger_causality_correct_value` — verifies correct Granger causality calculation returns positive value for known causal signal
- `test_granger_causality_detects_causality` — confirms feature detects causality in synthetic data
- `test_granger_causality_directionality` — tests forward vs backward causality asymmetry
- `test_granger_causality_single_lag` — validates scalar output shape for single lag
- `test_granger_causality_multiple_lags` — verifies lag_index dimension for maxlag mode
- `test_granger_causality_lag_precedence` — confirms lag takes precedence over maxlag
- `test_granger_causality_matrix_shape` — validates 2D matrix output
- `test_granger_causality_matrix_diagonal_nan` — confirms self-causality is NaN
- `test_granger_causality_matrix_directional` — tests directional causality in matrix mode
- `test_granger_causality_matrix_default_coords` — validates default behavior uses all coords
- `test_granger_causality_matrix_multiple_lags` — verifies 3D matrix output for multi-lag mode
- `test_granger_causality_returns_positive_for_causality` — additional test with stronger causality
- `test_granger_causality_no_causality_small` — tests random (uncorrelated) data produces non-negative result
- `test_granger_causality_requires_time_dimension` — validates time dim requirement
- `test_granger_causality_requires_space_dimension` — validates space dim requirement
- `test_granger_causality_invalid_coord` — tests out-of-bounds coord handling
- `test_granger_causality_invalid_lag` / `test_granger_causality_invalid_maxlag` — validates `__post_init__` checks
- `test_granger_causality_matrix_invalid_lag` / `test_granger_causality_matrix_invalid_maxlag` — validates matrix class validation
- `test_granger_causality_matrix_empty_coords` — validates empty coords rejection
- `test_granger_causality_small_samples` — tests edge case with small time series
- `test_granger_causality_small_dataset` — validates behavior with 50-sample dataset
- `test_granger_causality_detects_known_causality` — stronger test with causal_strength=0.5
- `test_granger_causality_history_tracking` — confirms class name appended to history
- `test_granger_causality_preserves_metadata` — verifies subjectID/groupID/condition/sampling_rate
- `test_granger_causality_no_mutation` — confirms input Data unchanged
- `test_granger_causality_returns_data_instance` — validates return type
- `test_granger_causality_matrix_history_updated` — matrix class history tracking
- `test_granger_causality_matrix_preserves_metadata` — matrix class metadata preservation
- `test_granger_causality_matrix_no_mutation` — matrix class input immutability

## Fix

No tests need changes.

## Add

No missing scenarios — all required test patterns are covered.

## Action List

None.
