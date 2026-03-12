# Test Review: coherence

**Feature**: `src/cobrabox/features/coherence.py`
**Test file**: `tests/test_feature_coherence.py`
**Date**: 2026-03-06
**Verdict**: PASS

## Coverage

```text
src/cobrabox/features/coherence.py: 100% (61 statements, 0 missing)
```

All 61 statements in the feature are covered by the test suite.

## Summary

The test suite for `Coherence` is comprehensive and well-structured. All 17 tests pass and achieve 100% code coverage. The tests cover all required scenarios including happy path, metadata preservation, history tracking, error handling, and input immutability. Additionally, the suite includes excellent coherence-specific numerical correctness tests that verify the mathematical properties of magnitude-squared coherence (unit coherence for identical signals, symmetric matrix, NaN diagonal, values in [0, 1]).

## Keep

Tests that are correct and complete — no changes needed:

- `test_coherence_output_dims_and_shape` — verifies output structure and dimensions
- `test_coherence_space_coords_are_preserved` — checks coordinate preservation
- `test_coherence_identical_channels_give_unity_coherence` — numerical correctness (unit coherence)
- `test_coherence_diagonal_is_nan` — numerical correctness (self-coherence)
- `test_coherence_matrix_is_symmetric` — numerical correctness (symmetry)
- `test_coherence_values_are_in_unit_range` — numerical correctness (range)
- `test_coherence_preserves_metadata_and_history` — metadata, history, and sampling_rate=None
- `test_coherence_with_run_index_preserves_extra_dim` — extra dimensions handling
- `test_coherence_custom_nperseg_produces_valid_output` — custom parameter
- `test_coherence_results_depend_on_nperseg` — parameter sensitivity
- `test_coherence_raises_when_fewer_than_two_channels` — input validation
- `test_coherence_raises_when_nperseg_exceeds_n_time` — runtime validation
- `test_coherence_raises_when_nperseg_is_less_than_two` — constructor validation
- `test_coherence_accessible_via_feature_module` — API accessibility
- `test_coherence_missing_space_dim_raises` — dimension validation
- `test_coherence_default_nperseg_too_small_raises` — edge case (n_time < 2)
- `test_coherence_does_not_mutate_input` — immutability verification

## Fix

None.

## Add

None — all required scenarios are covered.

## Action List

None.
