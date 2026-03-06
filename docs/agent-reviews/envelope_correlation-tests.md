# Test Review: envelope_correlation

**Feature**: `src/cobrabox/features/envelope_correlation.py`
**Test file**: `tests/test_feature_envelope_correlation.py`
**Date**: 2025-03-05
**Verdict**: PASS

## Coverage

```text
envelope_correlation.py: 100% (92 statements, 0 missing)
```

## Summary

Excellent test coverage with 13 tests. Tests validate amplitude envelope correlation (AEC) computation, output structure (space x space_to matrix), symmetry, diagonal values, absolute parameter, orthogonalize parameter, metadata preservation, and error handling (single channel, extra dimensions).

## Keep

- `test_envelope_correlation_output_dims_and_shape` — output structure
- `test_envelope_correlation_space_coords_preserved` — coordinate preservation
- `test_envelope_correlation_matrix_is_symmetric` — symmetry property
- `test_envelope_correlation_diagonal_is_finite` — self-correlation handling
- `test_envelope_correlation_absolute_true_gives_nonnegative` — absolute parameter
- `test_envelope_correlation_orthogonalize_changes_values` — orthogonalize parameter
- `test_envelope_correlation_history_appended` — history tracking
- `test_envelope_correlation_metadata_preserved` — metadata propagation (groupID, condition)
- `test_envelope_correlation_raises_for_single_channel` — channel count validation
- `test_envelope_correlation_raises_for_extra_dims` — dimension validation
- `test_envelope_correlation_no_mutation` — input immutability

## Fix

None. All tests are correct and comprehensive.

## Add

None. All required scenarios covered.

## Action List

None.
