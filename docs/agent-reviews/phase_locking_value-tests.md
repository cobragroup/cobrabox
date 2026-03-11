# Test Review: phase_locking_value

**Feature**: `src/cobrabox/features/phase_locking_value.py`
**Test file**: `tests/test_feature_phase_locking_value.py`
**Date**: 2026-03-06
**Verdict**: NEEDS WORK

## Coverage

```text
phase_locking_value.py: 100% (55 statements, 0 missing)
```

## Summary

Test suite covers all required scenarios with 100% coverage. Both `PhaseLockingValue` and `PhaseLockingValueMatrix` classes are tested for happy path, error cases, history, metadata preservation, and immutability. However, some property-based tests and edge cases are missing.

## Keep

Tests that are correct and complete:

- `test_phase_locking_value_returns_float` — correct happy path test with value verification
- `test_phase_locking_value_matrix_returns_square_matrix` — correct shape verification
- `test_phase_locking_value_diagonal_is_one` — correct self-correlation property
- `test_phase_locking_value_raises_invalid_coordinate` — correct error case
- `test_phase_locking_value_matrix_raises_empty_coords` — correct validation test
- `test_phase_locking_value_preserves_history` — correct history verification
- `test_phase_locking_value_raises_when_no_space_dim` — correct dimension validation
- `test_phase_locking_value_raises_when_coord_y_not_found` — correct error case
- `test_phase_locking_value_matrix_raises_when_no_space_dim` — correct dimension validation
- `test_phase_locking_value_matrix_raises_when_coord_not_found` — correct error case
- `test_phase_locking_value_matrix_preserves_history` — correct history verification
- `test_phase_locking_value_metadata_preserved` — correct metadata test with sampling_rate=None
- `test_phase_locking_value_matrix_metadata_preserved` — correct metadata test with sampling_rate=None
- `test_phase_locking_value_does_not_mutate_input` — correct immutability test
- `test_phase_locking_value_matrix_does_not_mutate_input` — correct immutability test

## Add

Missing scenarios to add:

### `test_phase_locking_value_matrix_symmetric`

```python
def test_phase_locking_value_matrix_symmetric() -> None:
    """PLV matrix should be symmetric (PLV[i,j] == PLV[j,i])."""
    data = cb.data.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )

    result = cb.feature.PhaseLockingValueMatrix(coords=[0, 1, 2]).apply(data)
    matrix = result.to_numpy()

    np.testing.assert_allclose(matrix, matrix.T, rtol=1e-10)
```

### `test_phase_locking_value_in_range`

```python
def test_phase_locking_value_in_range() -> None:
    """PLV values should be in [0, 1]."""
    data = cb.data.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )

    result = cb.feature.PhaseLockingValue(coord_x=0, coord_y=1).apply(data)
    value = float(result.data.values)

    assert 0.0 <= value <= 1.0
```

### `test_phase_locking_value_matrix_in_range`

```python
def test_phase_locking_value_matrix_in_range() -> None:
    """All PLV matrix values should be in [0, 1]."""
    data = cb.data.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )

    result = cb.feature.PhaseLockingValueMatrix(coords=[0, 1, 2]).apply(data)
    matrix = result.to_numpy()

    assert np.all((matrix >= 0.0) & (matrix <= 1.0))
```

### `test_phase_locking_value_with_string_coords`

```python
def test_phase_locking_value_with_string_coords() -> None:
    """PhaseLockingValue works with string coordinate labels."""
    arr = rng.normal(size=(100, 3))
    data = cb.SignalData.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=100.0,
    )
    # Relabel space dimension with strings
    data = cb.Data.from_xarray(
        data.data.assign_coords(space=["ch0", "ch1", "ch2"]),
        sampling_rate=100.0,
    )

    result = cb.feature.PhaseLockingValue(coord_x="ch0", coord_y="ch1").apply(data)

    assert isinstance(result, cb.Data)
    assert result.data.shape == ()
```

### `test_phase_locking_value_matrix_diagonal_all_ones`

```python
def test_phase_locking_value_matrix_diagonal_all_ones() -> None:
    """Diagonal of PLV matrix should be all 1.0 (self-correlation)."""
    data = cb.data.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )

    result = cb.feature.PhaseLockingValueMatrix(coords=[0, 1, 2]).apply(data)
    matrix = result.to_numpy()

    np.testing.assert_allclose(np.diag(matrix), 1.0)
```

### `test_phase_locking_value_via_chord`

```python
def test_phase_locking_value_via_chord() -> None:
    """PhaseLockingValue works correctly within a Chord pipeline."""
    data = cb.data.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )

    chord = cb.Chord(
        split=cb.feature.SlidingWindow(window_size=50, step_size=25),
        pipeline=cb.feature.PhaseLockingValue(coord_x=0, coord_y=1),
        aggregate=cb.feature.MeanAggregate(),
    )
    result = chord.apply(data)

    assert isinstance(result, cb.Data)
    assert "PhaseLockingValue" in result.history
    assert "MeanAggregate" in result.history
    assert "SlidingWindow" in result.history
```

## Action List

1. [Severity: MEDIUM] Add test for PLV matrix symmetry property (`test_phase_locking_value_matrix_symmetric`)
2. [Severity: MEDIUM] Add test for PLV value range [0, 1] (`test_phase_locking_value_in_range`)
3. [Severity: MEDIUM] Add test for PLV matrix value range (`test_phase_locking_value_matrix_in_range`)
4. [Severity: MEDIUM] Add test with string coordinate labels (`test_phase_locking_value_with_string_coords`)
5. [Severity: LOW] Add test for matrix diagonal all ones (`test_phase_locking_value_matrix_diagonal_all_ones`)
6. [Severity: LOW] Add Chord composition test (`test_phase_locking_value_via_chord`)
