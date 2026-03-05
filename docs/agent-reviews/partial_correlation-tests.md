# Test Review: partial_correlation

**Feature**: `src/cobrabox/features/partial_correlation.py`
**Test file**: `tests/test_feature_partial_correlation.py`
**Date**: 2026-03-04
**Verdict**: NEEDS WORK

## Summary

The test file covers both `PartialCorrelation` and `PartialCorrelationMatrix` classes with good coverage of basic functionality and error cases. However, it is missing several required scenarios including metadata preservation, output type handling (sampling_rate), and input mutation tests. The history assertions use `in` instead of `==` for the last element.

## Keep

Tests that are correct and complete:

- `test_partial_correlation_returns_float` — verifies basic functionality and output shape
- `test_partial_correlation_matrix_returns_square_matrix` — verifies matrix output shape
- `test_partial_correlation_diagonal_is_one` — good edge case for self-correlation
- `test_partial_correlation_raises_empty_control_vars` — proper validation test
- `test_partial_correlation_raises_invalid_coordinate` — coordinate validation
- `test_partial_correlation_raises_invalid_control_coordinate` — control var validation
- `test_partial_correlation_matrix_raises_empty_coords` — empty coords validation
- `test_partial_correlation_raises_when_no_space_dim` — missing space dimension
- `test_partial_correlation_raises_when_no_time_dim` — missing time dimension
- `test_partial_correlation_matrix_raises_when_no_space_dim` — matrix missing space
- `test_partial_correlation_matrix_raises_when_no_time_dim` — matrix missing time
- `test_partial_correlation_matrix_raises_empty_control_vars` — matrix empty controls
- `test_partial_correlation_matrix_raises_invalid_coord` — matrix invalid coord
- `test_partial_correlation_matrix_raises_invalid_control_var` — matrix invalid control
- `test_partial_correlation_with_multiple_controls` — multiple control variables

## Fix

Tests that need corrections:

### `test_partial_correlation_preserves_history` (line 95)

Issue: Uses `in` instead of checking the last element

```python
# Current:
assert "PartialCorrelation" in result.history

# Should be:
assert result.history[-1] == "PartialCorrelation"
```

### `test_partial_correlation_matrix_preserves_history` (line 106)

Issue: Uses `in` instead of checking the last element

```python
# Current:
assert "PartialCorrelationMatrix" in result.history

# Should be:
assert result.history[-1] == "PartialCorrelationMatrix"
```

### `test_partial_correlation_diagonal_is_one`

Issue: Only tests PartialCorrelation, should also test PartialCorrelationMatrix diagonal
Consider adding a test for matrix diagonal or renaming to be more specific.

## Add

Missing scenarios to implement:

### `test_partial_correlation_metadata_preserved`

```python
def test_partial_correlation_metadata_preserved() -> None:
    """PartialCorrelation preserves subjectID, groupID, condition."""
    data = cb.SignalData.from_numpy(
        rng.normal(size=(100, 4)),
        dims=["time", "space"],
        sampling_rate=100.0,
        subjectID="s1",
        groupID="g1",
        condition="rest",
    )
    result = cb.feature.PartialCorrelation(coord_x=0, coord_y=1, control_vars=[2]).apply(data)
    assert result.subjectID == "s1"
    assert result.groupID == "g1"
    assert result.condition == "rest"
```

### `test_partial_correlation_sampling_rate_none`

```python
def test_partial_correlation_sampling_rate_none() -> None:
    """PartialCorrelation sets sampling_rate to None (output_type = Data)."""
    data = cb.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )
    result = cb.feature.PartialCorrelation(coord_x=0, coord_y=1, control_vars=[2]).apply(data)
    assert result.sampling_rate is None
```

### `test_partial_correlation_does_not_mutate_input`

```python
def test_partial_correlation_does_not_mutate_input() -> None:
    """PartialCorrelation does not modify input Data object."""
    data = cb.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.PartialCorrelation(coord_x=0, coord_y=1, control_vars=[2]).apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
```

### `test_partial_correlation_matrix_metadata_preserved`

```python
def test_partial_correlation_matrix_metadata_preserved() -> None:
    """PartialCorrelationMatrix preserves subjectID, groupID, condition."""
    data = cb.SignalData.from_numpy(
        rng.normal(size=(100, 4)),
        dims=["time", "space"],
        sampling_rate=100.0,
        subjectID="s1",
        groupID="g1",
        condition="rest",
    )
    result = cb.feature.PartialCorrelationMatrix(coords=[0, 1], control_vars=[2]).apply(data)
    assert result.subjectID == "s1"
    assert result.groupID == "g1"
    assert result.condition == "rest"
```

### `test_partial_correlation_matrix_sampling_rate_none`

```python
def test_partial_correlation_matrix_sampling_rate_none() -> None:
    """PartialCorrelationMatrix sets sampling_rate to None (output_type = Data)."""
    data = cb.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )
    result = cb.feature.PartialCorrelationMatrix(coords=[0, 1], control_vars=[2]).apply(data)
    assert result.sampling_rate is None
```

### `test_partial_correlation_matrix_does_not_mutate_input`

```python
def test_partial_correlation_matrix_does_not_mutate_input() -> None:
    """PartialCorrelationMatrix does not modify input Data object."""
    data = cb.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.PartialCorrelationMatrix(coords=[0, 1], control_vars=[2]).apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
```

### `test_partial_correlation_matrix_diagonal_is_one`

```python
def test_partial_correlation_matrix_diagonal_is_one() -> None:
    """PartialCorrelationMatrix diagonal equals 1.0 (self-correlation)."""
    data = cb.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )
    result = cb.feature.PartialCorrelationMatrix(coords=[0, 1, 2], control_vars=[3]).apply(data)
    matrix = result.data.values
    np.testing.assert_allclose(np.diag(matrix), 1.0)
```

### `test_partial_correlation_matrix_is_symmetric`

```python
def test_partial_correlation_matrix_is_symmetric() -> None:
    """PartialCorrelationMatrix output is symmetric."""
    data = cb.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )
    result = cb.feature.PartialCorrelationMatrix(coords=[0, 1, 2], control_vars=[3]).apply(data)
    matrix = result.data.values
    np.testing.assert_allclose(matrix, matrix.T)
```

### `test_partial_correlation_singular_matrix_raises`

```python
def test_partial_correlation_singular_matrix_raises() -> None:
    """PartialCorrelation raises ValueError when correlation matrix is singular."""
    # Create perfectly correlated data
    base = np.ones((10,))
    data = cb.SignalData.from_numpy(
        np.column_stack([base, base, base, base]),
        dims=["time", "space"],
        sampling_rate=100.0,
    )
    with pytest.raises(ValueError, match="singular"):
        cb.feature.PartialCorrelation(coord_x=0, coord_y=1, control_vars=[2]).apply(data)
```

## Action List

1. [Severity: MEDIUM] Fix history assertions to use `==` instead of `in` (lines 103, 114)
2. [Severity: HIGH] Add metadata preservation tests for both classes
3. [Severity: HIGH] Add sampling_rate=None tests for both classes (output_type handling)
4. [Severity: HIGH] Add input mutation tests for both classes
5. [Severity: MEDIUM] Add PartialCorrelationMatrix diagonal test
6. [Severity: MEDIUM] Add PartialCorrelationMatrix symmetry test
7. [Severity: MEDIUM] Add singular matrix error test
