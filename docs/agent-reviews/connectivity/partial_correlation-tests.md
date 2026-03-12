# Test Review: partial_correlation

**Feature**: `src/cobrabox/features/partial_correlation.py`
**Test file**: `tests/test_feature_partial_correlation.py`
**Date**: 2026-03-06
**Verdict**: PASS

## Coverage

```text
partial_correlation.py: 100% (83 statements, 0 missing)
```

## Summary

Excellent test coverage for PartialCorrelation and PartialCorrelationMatrix. The test file contains 27 tests covering both classes comprehensively. All required scenarios are tested including basic functionality, matrix computation, coordinate validation, error handling, metadata preservation, mathematical correctness, and immutability guarantees.

## Keep

All 27 tests are correct and complete:

- `test_partial_correlation_returns_float` — Basic functionality with manual verification
- `test_partial_correlation_matrix_returns_square_matrix` — Matrix shape validation
- `test_partial_correlation_diagonal_is_one` — Self-correlation equals 1.0
- `test_partial_correlation_raises_empty_control_vars` — Empty control_vars validation
- `test_partial_correlation_raises_invalid_coordinate` — Invalid coord_x validation
- `test_partial_correlation_raises_invalid_control_coordinate` — Invalid control coordinate validation
- `test_partial_correlation_matrix_raises_empty_coords` — Empty coords validation
- `test_partial_correlation_preserves_history` — History updated for PartialCorrelation
- `test_partial_correlation_matrix_preserves_history` — History updated for PartialCorrelationMatrix
- `test_partial_correlation_raises_when_no_space_dim` — Missing space dimension
- `test_partial_correlation_raises_when_coord_y_not_found` — coord_y not found validation
- `test_partial_correlation_matrix_raises_when_no_space_dim` — Matrix missing space dimension
- `test_partial_correlation_matrix_raises_empty_control_vars` — Matrix empty controls validation
- `test_partial_correlation_matrix_raises_invalid_coord` — Matrix invalid coord validation
- `test_partial_correlation_matrix_raises_invalid_control_var` — Matrix invalid control validation
- `test_partial_correlation_with_multiple_controls` — Multiple control variables
- `test_partial_correlation_raises_singular_matrix` — Singular matrix error handling
- `test_partial_correlation_raises_missing_time_dim` — Missing time dimension
- `test_partial_correlation_matrix_raises_missing_time_dim` — Matrix missing time dimension
- `test_partial_correlation_metadata_preserved` — Metadata preservation
- `test_partial_correlation_sampling_rate_none` — sampling_rate set to None
- `test_partial_correlation_does_not_mutate_input` — Input immutability
- `test_partial_correlation_matrix_metadata_preserved` — Matrix metadata preservation
- `test_partial_correlation_matrix_sampling_rate_none` — Matrix sampling_rate None
- `test_partial_correlation_matrix_does_not_mutate_input` — Matrix input immutability
- `test_partial_correlation_matrix_diagonal_is_one` — Matrix diagonal equals 1.0
- `test_partial_correlation_matrix_is_symmetric` — Matrix symmetry

## Action List

None.
