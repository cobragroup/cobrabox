# Test Review: PartialCorrelation

**Feature**: `src/cobrabox/features/connectivity/partial_correlation.py`
**Test file**: `tests/features/connectivity/test_feature_partial_correlation.py`
**Date**: 2026-03-24
**Verdict**: PASS

## Coverage

```text
partial_correlation.py: 100% (86 statements, 0 missing)
```

## Summary

The test suite is comprehensive and meets all required criteria. Tests cover happy paths for both `PartialCorrelation` and `PartialCorrelationMatrix`, parameter validation, edge cases (diagonal equals 1.0, symmetric matrix), metadata preservation, mutation safety, and error handling including singular correlation matrices. Coverage is 100% on the feature file.

## Keep

Tests that are correct and complete — no changes needed:

- `test_partial_correlation_returns_float` — validates output type shape, and correctness against manual computation
- `test_partial_correlation_matrix_returns_square_matrix` — checks correct output shape
- `test_partial_correlation_matrix_default_coords` — verifies default coords behavior excludes control_vars
- `test_partial_correlation_diagonal_is_one` — correctly asserts self-correlation equals 1.0
- `test_partial_correlation_matrix_diagonal_is_one` — verifies diagonal of matrix equals 1.0
- `test_partial_correlation_matrix_is_symmetric` — confirms matrix symmetry
- `test_partial_correlation_raises_empty_control_vars` — tests required validation
- `test_partial_correlation_raises_invalid_coordinate` — validates coordinate lookup error
- `test_partial_correlation_raises_invalid_control_coordinate` — tests control coord validation
- `test_partial_correlation_matrix_raises_empty_coords` — validates empty coords error
- `test_partial_correlation_preserves_history` / `test_partial_correlation_matrix_preserves_history` — verifies history appending
- `test_partial_correlation_raises_when_no_space_dim` / `test_partial_correlation_matrix_raises_when_no_space_dim` — space dimension guard tests
- `test_partial_correlation_matrix_raises_empty_control_vars` — validates empty control_vars for matrix variant
- `test_partial_correlation_matrix_raises_invalid_coord` / `test_partial_correlation_matrix_raises_invalid_control_var` — coordinate validation
- `test_partial_correlation_with_multiple_controls` — tests with multiple control variables
- `test_partial_correlation_raises_singular_matrix` — validates singular matrix handling
- `test_partial_correlation_raises_missing_time_dim` / `test_partial_correlation_matrix_raises_missing_time_dim` — time dimension guard tests
- `test_partial_correlation_metadata_preserved` / `test_partial_correlation_matrix_metadata_preserved` — all metadata fields verified
- `test_partial_correlation_sampling_rate_none` / `test_partial_correlation_matrix_sampling_rate_none` — output_type=Data behavior verified
- `test_partial_correlation_does_not_mutate_input` / `test_partial_correlation_matrix_does_not_mutate_input` — mutation safety tests

## Action List

None.
