# Test Review: coherence

**Feature**: `src/cobrabox/features/coherence.py`
**Test file**: `tests/test_feature_coherence.py`
**Date**: 2025-03-05
**Verdict**: PASS

## Coverage

```
coherence.py: 100% (131 statements, 0 missing)
```

## Summary

Excellent test coverage with 17 tests. Tests validate magnitude-squared coherence computation, output structure (space x space_to matrix), diagonal NaN handling, symmetry, value range [0,1], extra dimension preservation (run_index), custom nperseg, and comprehensive error handling (single channel, nperseg bounds, missing space dim).

## Keep

- `test_coherence_output_dims_and_shape` — output structure
- `test_coherence_space_coords_are_preserved` — coordinate preservation
- `test_coherence_identical_channels_give_unity_coherence` — unity coherence
- `test_coherence_diagonal_is_nan` — self-coherence handling
- `test_coherence_matrix_is_symmetric` — symmetry property
- `test_coherence_values_are_in_unit_range` — value constraints
- `test_coherence_with_run_index_preserves_extra_dim` — extra dimension handling
- `test_coherence_custom_nperseg_produces_valid_output` — nperseg parameter
- `test_coherence_raises_when_fewer_than_two_channels` — channel count validation
- `test_coherence_raises_when_nperseg_exceeds_n_time` — nperseg bounds
- `test_coherence_missing_space_dim_raises` — dimension validation
- `test_coherence_default_nperseg_too_small_raises` — nperseg default validation
- `test_coherence_does_not_mutate_input` — input immutability

## Fix

None. All tests are correct and comprehensive.

## Add

None. All required scenarios covered.

## Action List

None.
