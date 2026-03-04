# Test Review: phase_locking_value

**Feature**: `src/cobrabox/features/phase_locking_value.py`
**Test file**: `tests/test_feature_phase_locking_value.py`
**Date**: 2025-03-04
**Verdict**: NEEDS WORK

## Summary

The test file covers both `PhaseLockingValue` and `PhaseLockingValueMatrix` features with 12 tests. Core functionality is tested including happy paths, error cases, history updates, and mathematical correctness (diagonal = 1.0). However, **2 required scenarios are missing**: metadata preservation and input immutability tests. Additionally, `sampling_rate` behavior is not verified for features that remove the time dimension.

## Keep

Tests that are correct and complete — no changes needed:

- `test_phase_locking_value_returns_float` — Correctly validates output type, shape, and value accuracy using manual PLV computation.
- `test_phase_locking_value_matrix_returns_square_matrix` — Validates matrix dimensions and output type.
- `test_phase_locking_value_diagonal_is_one` — Mathematical correctness test; PLV of a signal with itself should be 1.0.
- `test_phase_locking_value_raises_invalid_coordinate` — Tests ValueError for missing coord_x.
- `test_phase_locking_value_matrix_raises_empty_coords` — Tests ValueError for empty coords list.
- `test_phase_locking_value_preserves_history` — Validates history is updated with class name.
- `test_phase_locking_value_matrix_preserves_history` — Validates history for matrix variant.
- `test_phase_locking_value_raises_when_no_space_dim` — Tests ValueError when space dimension is missing.
- `test_phase_locking_value_raises_when_coord_y_not_found` — Tests ValueError for missing coord_y (different from coord_x).
- `test_phase_locking_value_matrix_raises_when_no_space_dim` — Tests ValueError for matrix variant without space dim.
- `test_phase_locking_value_matrix_raises_when_coord_not_found` — Tests ValueError when a coord in coords is not found.

## Fix

None — all existing tests are correct.

## Add

Missing scenarios — new tests to add:

### `test_phase_locking_value_metadata_preserved`

**Issue**: No test verifies that subjectID, groupID, condition, and sampling_rate are preserved.

```python
def test_phase_locking_value_metadata_preserved() -> None:
    """PhaseLockingValue preserves subjectID, groupID, condition; sampling_rate is None."""
    data = cb.SignalData.from_numpy(
        rng.normal(size=(100, 4)),
        dims=["time", "space"],
        sampling_rate=100.0,
        subjectID="s42",
        groupID="control",
        condition="task",
    )
    result = cb.feature.PhaseLockingValue(coord_x=0, coord_y=1).apply(data)
    assert result.subjectID == "s42"
    assert result.groupID == "control"
    assert result.condition == "task"
    assert result.sampling_rate is None  # time dimension removed
```

### `test_phase_locking_value_matrix_metadata_preserved`

**Issue**: No test verifies metadata preservation for PhaseLockingValueMatrix.

```python
def test_phase_locking_value_matrix_metadata_preserved() -> None:
    """PhaseLockingValueMatrix preserves subjectID, groupID, condition; sampling_rate is None."""
    data = cb.SignalData.from_numpy(
        rng.normal(size=(100, 4)),
        dims=["time", "space"],
        sampling_rate=250.0,
        subjectID="s1",
        groupID="patient",
        condition="rest",
    )
    result = cb.feature.PhaseLockingValueMatrix(coords=[0, 1]).apply(data)
    assert result.subjectID == "s1"
    assert result.groupID == "patient"
    assert result.condition == "rest"
    assert result.sampling_rate is None  # time dimension removed
```

### `test_phase_locking_value_does_not_mutate_input`

**Issue**: No test verifies the input Data object is unchanged after calling the feature.

```python
def test_phase_locking_value_does_not_mutate_input() -> None:
    """PhaseLockingValue does not modify the input Data object."""
    data = cb.SignalData.from_numpy(
        rng.normal(size=(100, 4)),
        dims=["time", "space"],
        sampling_rate=100.0,
    )
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.PhaseLockingValue(coord_x=0, coord_y=1).apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
```

### `test_phase_locking_value_matrix_does_not_mutate_input`

**Issue**: No test verifies input immutability for PhaseLockingValueMatrix.

```python
def test_phase_locking_value_matrix_does_not_mutate_input() -> None:
    """PhaseLockingValueMatrix does not modify the input Data object."""
    data = cb.SignalData.from_numpy(
        rng.normal(size=(100, 4)),
        dims=["time", "space"],
        sampling_rate=100.0,
    )
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.PhaseLockingValueMatrix(coords=[0, 1]).apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
```

## Action List

1. [Severity: HIGH] Add `test_phase_locking_value_metadata_preserved` to verify subjectID, groupID, condition, and sampling_rate handling (`tests/test_feature_phase_locking_value.py`)
2. [Severity: HIGH] Add `test_phase_locking_value_matrix_metadata_preserved` to verify metadata preservation for matrix variant (`tests/test_feature_phase_locking_value.py`)
3. [Severity: HIGH] Add `test_phase_locking_value_does_not_mutate_input` to verify input Data is not modified (`tests/test_feature_phase_locking_value.py`)
4. [Severity: HIGH] Add `test_phase_locking_value_matrix_does_not_mutate_input` to verify input immutability for matrix variant (`tests/test_feature_phase_locking_value.py`)
