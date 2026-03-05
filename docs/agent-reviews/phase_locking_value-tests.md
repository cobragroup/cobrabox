# Test Review: phase_locking_value

**Feature**: `src/cobrabox/features/phase_locking_value.py`
**Test file**: `tests/test_feature_phase_locking_value.py`
**Date**: 2025-03-05
**Verdict**: PASS

## Coverage

```
phase_locking_value.py: 100% (55 statements, 0 missing)
```

## Summary

Excellent test coverage for both PhaseLockingValue and PhaseLockingValueMatrix. The tests are comprehensive, covering both classes with 15 tests total. All required scenarios are tested including PLV computation, matrix generation, coordinate validation, error handling, metadata preservation, and input immutability.

## Keep

- `test_phase_locking_value_returns_float` — Tests basic PLV computation with manual verification
- `test_phase_locking_value_matrix_returns_square_matrix` — Tests matrix shape and dims
- `test_phase_locking_value_diagonal_is_one` — Tests self-PLV equals 1.0
- `test_phase_locking_value_raises_invalid_coordinate` — Tests invalid coordinate validation
- `test_phase_locking_value_matrix_raises_empty_coords` — Tests empty coords validation
- `test_phase_locking_value_preserves_history` — Tests history updated
- `test_phase_locking_value_raises_when_no_space_dim` — Tests missing space dimension
- `test_phase_locking_value_raises_when_coord_y_not_found` — Tests coord_y not found
- `test_phase_locking_value_matrix_raises_when_no_space_dim` — Tests matrix with no space
- `test_phase_locking_value_matrix_raises_when_coord_not_found` — Tests matrix invalid coord
- `test_phase_locking_value_matrix_preserves_history` — Tests matrix history
- `test_phase_locking_value_metadata_preserved` — Tests metadata preservation (subjectID, groupID, condition, sampling_rate=None)
- `test_phase_locking_value_matrix_metadata_preserved` — Tests matrix metadata
- `test_phase_locking_value_does_not_mutate_input` — Tests input immutability
- `test_phase_locking_value_matrix_does_not_mutate_input` — Tests matrix immutability

## Action List

None.
