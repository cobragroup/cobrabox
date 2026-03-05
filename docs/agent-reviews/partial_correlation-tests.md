# Test Review: partial_correlation

**Feature**: `src/cobrabox/features/partial_correlation.py`
**Test file**: `tests/test_feature_partial_correlation.py`
**Date**: 2026-03-05
**Verdict**: NEEDS WORK

## Coverage

Per-file coverage: **95%** (84 statements, 4 missing)
Missing lines:
- 40-41: LinAlgError handling in `_compute_partial_correlation`
- 93: Missing time dimension check in `PartialCorrelation`
- 171: Missing time dimension check in `PartialCorrelationMatrix`

Coverage is at the threshold but missing critical error paths. Flagged as HIGH severity.

## Summary

The test file covers both `PartialCorrelation` and `PartialCorrelationMatrix` classes with good coverage of basic functionality and error cases. However, it is missing several required scenarios including metadata preservation, output type handling (sampling_rate), input mutation tests, and the singular matrix error handling. The history assertions use `in` instead of `==` for the last element.

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
- `test_partial_correlation_raises_when_coord_y_not_found` — coord_y validation
- `test_partial_correlation_matrix_raises_when_no_space_dim` — matrix missing space
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

## Add

Missing scenarios to implement:

### `test_partial_correlation_raises_singular_matrix`

Lines 40-41: Add test for singular matrix error handling.

```python
def test_partial_correlation_raises_singular_matrix() -> None:
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

### `test_partial_correlation_raises_missing_time_dim`

Line 93: Add test for missing time dimension.

```python
def test_partial_correlation_raises_missing_time_dim() -> None:
    """PartialCorrelation raises ValueError when time dimension is missing."""
    import xarray as xr
    # Create DataArray without time dimension
    xr_data = xr.DataArray(np.ones((5, 3)), dims=["space", "channels"])
    # Bypass Data validation to test feature guard
    raw = cb.Data.__new__(cb.Data)
    object.__setattr__(raw, "_data", xr_data)
    with pytest.raises(ValueError, match="time"):
        cb.feature.PartialCorrelation(coord_x=0, coord_y=1, control_vars=[2]).apply(raw)
```

### `test_partial_correlation_matrix_raises_missing_time_dim`

Line 171: Add test for missing time dimension in matrix.

```python
def test_partial_correlation_matrix_raises_missing_time_dim() -> None:
    """PartialCorrelationMatrix raises ValueError when time dimension is missing."""
    import xarray as xr
    xr_data = xr.DataArray(np.ones((5, 3)), dims=["space", "channels"])
    raw = cb.Data.__new__(cb.Data)
    object.__setattr__(raw, "_data", xr_data)
    with pytest.raises(ValueError, match="time"):
        cb.feature.PartialCorrelationMatrix(coords=[0, 1], control_vars=[2]).apply(raw)
```

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

## Action List

1. [Severity: MEDIUM] Fix history assertions to use `==` instead of `in` (lines 103, 114)
2. [Severity: HIGH] Add singular matrix error test (lines 40-41)
3. [Severity: HIGH] Add missing time dimension test for PartialCorrelation (line 93)
4. [Severity: HIGH] Add missing time dimension test for PartialCorrelationMatrix (line 171)
5. [Severity: HIGH] Add metadata preservation tests for both classes
6. [Severity: HIGH] Add sampling_rate=None tests for both classes (output_type handling)
7. [Severity: HIGH] Add input mutation tests for both classes
8. [Severity: MEDIUM] Add PartialCorrelationMatrix diagonal test
9. [Severity: MEDIUM] Add PartialCorrelationMatrix symmetry test
