# Test Review: partial_correlation

**Feature**: `src/cobrabox/features/partial_correlation.py`
**Test file**: `tests/test_feature_partial_correlation.py`
**Date**: 2025-03-05
**Verdict**: PASS

## Coverage

```
partial_correlation.py: 100% (84 statements, 0 missing)
```

## Summary

Excellent test coverage for PartialCorrelation and PartialCorrelationMatrix. The test file is very comprehensive with 27 tests covering both classes. All required scenarios are tested including both features, matrix computation, coordinate validation, error handling, metadata preservation, and mathematical correctness.

## Keep

- `test_partial_correlation_returns_float` — Tests basic functionality with manual verification
- `test_partial_correlation_matrix_returns_square_matrix` — Tests matrix shape
- `test_partial_correlation_diagonal_is_one` — Tests self-correlation equals 1.0
- `test_partial_correlation_raises_empty_control_vars` — Tests empty control_vars validation
- `test_partial_correlation_raises_invalid_coordinate` — Tests invalid coord_x validation
- `test_partial_correlation_raises_invalid_control_coordinate` — Tests invalid control coordinate validation
- `test_partial_correlation_matrix_raises_empty_coords` — Tests empty coords validation
- `test_partial_correlation_preserves_history` — Tests history updated
- `test_partial_correlation_matrix_preserves_history` — Tests matrix history updated
- `test_partial_correlation_raises_when_no_space_dim` — Tests missing space dimension
- `test_partial_correlation_raises_when_coord_y_not_found` — Tests coord_y not found
- `test_partial_correlation_matrix_raises_when_no_space_dim` — Tests matrix with no space
- `test_partial_correlation_matrix_raises_empty_control_vars` — Tests matrix empty controls
- `test_partial_correlation_matrix_raises_invalid_coord` — Tests matrix invalid coord
- `test_partial_correlation_matrix_raises_invalid_control_var` — Tests matrix invalid control
- `test_partial_correlation_with_multiple_controls` — Tests multiple control variables
- `test_partial_correlation_raises_singular_matrix` — Tests singular matrix error
- `test_partial_correlation_raises_missing_time_dim` — Tests missing time dimension
- `test_partial_correlation_matrix_raises_missing_time_dim` — Tests matrix missing time
- `test_partial_correlation_metadata_preserved` — Tests metadata preservation
- `test_partial_correlation_sampling_rate_none` — Tests sampling_rate is None
- `test_partial_correlation_does_not_mutate_input` — Tests input immutability
- `test_partial_correlation_matrix_metadata_preserved` — Tests matrix metadata
- `test_partial_correlation_matrix_sampling_rate_none` — Tests matrix sampling_rate
- `test_partial_correlation_matrix_does_not_mutate_input` — Tests matrix immutability
- `test_partial_correlation_matrix_diagonal_is_one` — Tests matrix diagonal
- `test_partial_correlation_matrix_is_symmetric` — Tests matrix symmetry

## Action List

None.
