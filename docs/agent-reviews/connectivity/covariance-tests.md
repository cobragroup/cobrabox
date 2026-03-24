# Test Review: covariance

**Feature**: `src/cobrabox/features/connectivity/covariance.py`
**Test file**: `tests/features/connectivity/test_feature_covariance.py`
**Date**: 2026-03-24
**Verdict**: PASS

## Coverage

Report the per-file coverage percentage and any uncovered lines:

```text
Covariance: 100% (27 statements, 0 missing)
```

## Summary

The test file provides comprehensive coverage of the `Covariance` feature. All required scenarios are tested with well-structured tests that follow the project's naming conventions and best practices. The tests cover happy path, numerical correctness, metadata propagation, custom dimension handling, error handling, and API accessibility.

## Keep

Tests that are correct and complete — no changes needed:

- `test_covariance_output_dims_and_shape` — correctly verifies output shape and dimension naming
- `test_covariance_output_is_square` — asserts square matrix output for any n_channels
- `test_covariance_channel_coords_preserved` — properly tests coordinate preservation with custom labels
- `test_covariance_matches_numpy_cov` — exact numerical comparison with numpy's reference implementation
- `test_covariance_diagonal_equals_sample_variance` — correctly validates ddof=1 variance on diagonal
- `test_covariance_matrix_is_symmetric` — verifies symmetry with proper tolerance
- `test_covariance_diagonal_positive_for_nonzero_signals` — sanity check on variance positivity
- `test_covariance_identical_channels_diagonal_matches_off_diagonal` — edge case for identical signals
- `test_covariance_preserves_metadata_and_history` — comprehensive metadata test including sampling_rate=None for output_type=Data
- `test_covariance_history_appends_correctly` — verifies history entry is 'Covariance'
- `test_covariance_custom_dim_name` — tests non-default dimension name handling
- `test_covariance_correlates_along_non_default_dim` — validates behavior when dim is first axis
- `test_covariance_custom_dim_matches_numpy_cov` — cross-checks numerical correctness with custom dim
- `test_covariance_raises_on_3d_input` — proper ValueError for ndim!=2
- `test_covariance_raises_on_1d_input` — proper ValueError for ndim!=2
- `test_covariance_raises_when_dim_missing` — validates dim existence check
- `test_covariance_raises_when_dim_missing_includes_hint` — error message includes dim= hint
- `test_covariance_does_not_mutate_input` — verifies input immutability
- `test_covariance_accessible_via_feature_module` — API accessibility check
- `test_covariance_output_is_data_instance` — confirms Data return type

## Fix

None. All existing tests pass and are correct.

## Add

None. All required scenarios are covered.

## Action List

None.
