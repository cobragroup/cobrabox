# Test Review: phase_locking_value

**Feature**: `src/cobrabox/features/connectivity/phase_locking_value.py`
**Test file**: `tests/features/connectivity/test_feature_phase_locking_value.py`
**Date**: 2025-03-24
**Verdict**: PASS

## Coverage

```text
PhaseLockingValue: 100% (58 statements, 0 missing)
```

Excellent! Both `PhaseLockingValue` and `PhaseLockingValueMatrix` classes have complete coverage.

## Summary

The test file is comprehensive and well-structured with 22 tests covering both `PhaseLockingValue` (scalar PLV between two coordinates) and `PhaseLockingValueMatrix` (pairwise PLV matrix). All required scenarios are present, plus additional algorithmic validation tests (symmetry, diagonal values, value ranges). The tests use proper `SignalData` construction for `BaseFeature[SignalData]` features and correctly verify that `sampling_rate` becomes `None` since both features set `output_type = Data`.

## Keep

Tests that are correct and complete — no changes needed:

- `test_phase_locking_value_returns_float` — validates basic happy path with manual verification against scipy's hilbert
- `test_phase_locking_value_matrix_returns_square_matrix` — validates matrix output shape and dims
- `test_phase_locking_value_matrix_default_coords` — tests default `coords=None` behavior using all space coordinates
- `test_phase_locking_value_diagonal_is_one` — algorithmic correctness: self-PLV equals 1.0
- `test_phase_locking_value_raises_invalid_coordinate` — error case for missing coord_x
- `test_phase_locking_value_matrix_raises_empty_coords` — error case for empty coords list
- `test_phase_locking_value_preserves_history` — history is updated with class name
- `test_phase_locking_value_raises_when_no_space_dim` — validates space dimension requirement
- `test_phase_locking_value_raises_when_coord_y_not_found` — error case for missing coord_y specifically
- `test_phase_locking_value_matrix_raises_when_no_space_dim` — space dimension check for matrix variant
- `test_phase_locking_value_matrix_raises_when_coord_not_found` — error case for any coord not in space
- `test_phase_locking_value_matrix_preserves_history` — history for matrix variant
- `test_phase_locking_value_metadata_preserved` — subjectID, groupID, condition preserved; sampling_rate becomes None
- `test_phase_locking_value_matrix_metadata_preserved` — same for matrix variant
- `test_phase_locking_value_does_not_mutate_input` — immutability check for scalar variant
- `test_phase_locking_value_matrix_does_not_mutate_input` — immutability check for matrix variant
- `test_phase_locking_value_matrix_symmetric` — algorithmic: PLV matrix is symmetric
- `test_phase_locking_value_in_range` — algorithmic: PLV in [0, 1]
- `test_phase_locking_value_matrix_in_range` — algorithmic: all matrix values in [0, 1]
- `test_phase_locking_value_with_string_coords` — works with string coordinate labels
- `test_phase_locking_value_matrix_diagonal_all_ones` — algorithmic: diagonal is all 1.0
- `test_phase_locking_value_via_chord` — integration test with Chord pipeline

## Fix

None. All tests follow conventions correctly.

## Add

No missing required scenarios. The test suite exceeds the minimum requirements with excellent algorithmic validation tests.

## Action List

None.
