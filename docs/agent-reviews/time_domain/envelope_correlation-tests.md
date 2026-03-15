# Test Review: envelope_correlation

**Feature**: `src/cobrabox/features/envelope_correlation.py`
**Test file**: `tests/test_feature_envelope_correlation.py`
**Date**: 2026-03-06
**Verdict**: PASS

## Coverage

```text
envelope_correlation.py: 100% (31 statements, 0 missing)
```

## Summary

Excellent test coverage with 13 well-structured tests. All required scenarios are covered including happy path, error handling, metadata preservation, parameter variations, and numerical properties. The tests use consistent naming conventions, include docstrings, and verify both shape/dims and coordinate preservation.

## Keep

Tests that are correct and complete — no changes needed:

- `test_envelope_correlation_output_dims_and_shape` — verifies output is Data with (space_to, space_from) dims
- `test_envelope_correlation_space_coords_preserved` — verifies space coordinates are correctly propagated to both dims
- `test_envelope_correlation_matrix_is_symmetric` — important numerical property verification
- `test_envelope_correlation_diagonal_is_finite` — verifies self-correlation values are valid
- `test_envelope_correlation_absolute_true_gives_nonnegative` — verifies absolute parameter works
- `test_envelope_correlation_values_in_valid_range` — verifies correlation values are in [-1, 1]
- `test_envelope_correlation_orthogonalize_false_same_shape` — verifies orthogonalize=False path
- `test_envelope_correlation_orthogonalize_changes_values` — verifies orthogonalization actually affects results
- `test_envelope_correlation_history_appended` — verifies history is updated correctly
- `test_envelope_correlation_metadata_preserved` — verifies subjectID, groupID, condition preserved and sampling_rate=None
- `test_envelope_correlation_no_mutation` — verifies input Data is not mutated
- `test_envelope_correlation_raises_for_single_channel` — verifies ValueError for <2 channels
- `test_envelope_correlation_raises_for_extra_dims` — verifies ValueError for extra dimensions

## Fix

None.

## Add

None.

## Action List

None.
