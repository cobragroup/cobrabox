# Test Review: granger_causality

**Feature**: `src/cobrabox/features/granger_causality.py`
**Test file**: `tests/test_feature_granger_causality.py`
**Date**: 2026-03-05
**Verdict**: NEEDS WORK

## Coverage

```
granger_causality: 93% (86 statements, 6 missing)
```

Missing lines:
- 187, 189: `GrangerCausalityMatrix.__post_init__` (validation of maxlag and lag)
- 206, 211, 227, 235: Multi-lag path in `GrangerCausalityMatrix.__call__`

**Coverage < 95% - HIGH severity issue**

## Summary

The test file covers basic functionality for both `GrangerCausality` and `GrangerCausalityMatrix` classes, including causality detection, directionality, and validation. However, there are critical issues with history/metadata assertions (checking wrong attributes), missing multi-lag coverage for the matrix class, and several required scenarios from the test criteria are not covered.

## Keep

Tests that are correct and complete:
- `test_granger_causality_correct_value` — basic happy path
- `test_granger_causality_detects_causality` — verifies positive GC value
- `test_granger_causality_directionality` — tests asymmetry
- `test_granger_causality_single_lag` — scalar output shape
- `test_granger_causality_multiple_lags` — lag_index dimension
- `test_granger_causality_lag_precedence` — lag takes precedence over maxlag
- `test_granger_causality_matrix_shape` — 2D output shape
- `test_granger_causality_matrix_diagonal_nan` — NaN on diagonal
- `test_granger_causality_matrix_directional` — directional causality in matrix
- `test_granger_causality_matrix_default_coords` — auto-detect coords
- `test_granger_causality_invalid_lag` — __post_init__ validation
- `test_granger_causality_invalid_maxlag` — __post_init__ validation
- `test_granger_causality_requires_time_dimension` — missing time dim
- `test_granger_causality_requires_space_dimension` — missing space dim
- `test_granger_causality_invalid_coord` — invalid coord index

## Fix

Tests that exist but need changes:

### `test_granger_causality_history_tracking`
**Issue**: Checks `result.data.attrs.get("history")` instead of `result.history`
```python
def test_granger_causality_history_tracking() -> None:
    """GrangerCausality appends class name to history."""
    data = cb.from_numpy(_create_causal_signal(n_samples=200), dims=["time", "space"])
    feature = cb.feature.GrangerCausality(coord_x=0, coord_y=1, lag=2)
    result = feature.apply(data)
    assert result.history[-1] == "GrangerCausality"
```

### `test_granger_causality_preserves_metadata`
**Issue**: Checks `result.data.attrs.get()` instead of direct properties
```python
def test_granger_causality_preserves_metadata() -> None:
    """GrangerCausality preserves subjectID, groupID, condition, sampling_rate."""
    data = cb.SignalData.from_numpy(
        _create_causal_signal(n_samples=200),
        dims=["time", "space"],
        subjectID="test_subject",
        groupID="test_group",
        condition="test_condition",
        sampling_rate=100.0,
    )
    feature = cb.feature.GrangerCausality(coord_x=0, coord_y=1, lag=2)
    result = feature.apply(data)
    assert result.subjectID == "test_subject"
    assert result.groupID == "test_group"
    assert result.condition == "test_condition"
    assert result.sampling_rate is None  # time dimension removed
```

## Add

Missing scenarios:

### `test_granger_causality_matrix_multiple_lags`
**Coverage**: Lines 206, 211, 227, 235 (multi-lag path)
```python
def test_granger_causality_matrix_multiple_lags() -> None:
    """GrangerCausalityMatrix supports maxlag for testing multiple lags."""
    data = cb.from_numpy(_create_causal_signal(n_samples=200), dims=["time", "space"])
    feature = cb.feature.GrangerCausalityMatrix(coords=[0, 1], maxlag=4)
    result = feature.apply(data)
    assert result.data.shape == (2, 2, 4)
    assert "lag_index" in result.data.dims
    assert list(result.data.coords["lag_index"].values) == [1, 2, 3, 4]
```

### `test_granger_causality_matrix_invalid_lag`
**Coverage**: Line 187
```python
def test_granger_causality_matrix_invalid_lag() -> None:
    """GrangerCausalityMatrix raises ValueError for lag < 1."""
    with pytest.raises(ValueError, match="lag must be >= 1"):
        cb.feature.GrangerCausalityMatrix(coords=[0, 1], lag=0)
```

### `test_granger_causality_matrix_invalid_maxlag`
**Coverage**: Line 189
```python
def test_granger_causality_matrix_invalid_maxlag() -> None:
    """GrangerCausalityMatrix raises ValueError for maxlag < 1."""
    with pytest.raises(ValueError, match="maxlag must be >= 1"):
        cb.feature.GrangerCausalityMatrix(coords=[0, 1], maxlag=0)
```

### `test_granger_causality_no_mutation`
```python
def test_granger_causality_no_mutation() -> None:
    """GrangerCausality does not modify input Data."""
    data = cb.SignalData.from_numpy(
        _create_causal_signal(n_samples=200),
        dims=["time", "space"],
        subjectID="s1",
        groupID="g1",
        condition="rest",
        sampling_rate=100.0,
    )
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.GrangerCausality(coord_x=0, coord_y=1, lag=2).apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
    assert data.subjectID == "s1"
    assert data.groupID == "g1"
    assert data.condition == "rest"
    assert data.sampling_rate == 100.0
```

### `test_granger_causality_returns_data_instance`
```python
def test_granger_causality_returns_data_instance() -> None:
    """GrangerCausality.apply() returns Data instance."""
    data = cb.from_numpy(_create_causal_signal(n_samples=200), dims=["time", "space"])
    result = cb.feature.GrangerCausality(coord_x=0, coord_y=1, lag=2).apply(data)
    assert isinstance(result, cb.Data)
```

### `test_granger_causality_matrix_returns_data_instance`
```python
def test_granger_causality_matrix_returns_data_instance() -> None:
    """GrangerCausalityMatrix.apply() returns Data instance."""
    data = cb.from_numpy(_create_causal_signal(n_samples=200), dims=["time", "space"])
    result = cb.feature.GrangerCausalityMatrix(coords=[0, 1], lag=2).apply(data)
    assert isinstance(result, cb.Data)
```

### `test_granger_causality_matrix_history_updated`
```python
def test_granger_causality_matrix_history_updated() -> None:
    """GrangerCausalityMatrix appends class name to history."""
    data = cb.from_numpy(_create_causal_signal(n_samples=200), dims=["time", "space"])
    result = cb.feature.GrangerCausalityMatrix(coords=[0, 1], lag=2).apply(data)
    assert result.history[-1] == "GrangerCausalityMatrix"
```

### `test_granger_causality_matrix_preserves_metadata`
```python
def test_granger_causality_matrix_preserves_metadata() -> None:
    """GrangerCausalityMatrix preserves subjectID, groupID, condition."""
    data = cb.SignalData.from_numpy(
        _create_causal_signal(n_samples=200),
        dims=["time", "space"],
        subjectID="s1",
        groupID="g1",
        condition="rest",
        sampling_rate=100.0,
    )
    result = cb.feature.GrangerCausalityMatrix(coords=[0, 1], lag=2).apply(data)
    assert result.subjectID == "s1"
    assert result.groupID == "g1"
    assert result.condition == "rest"
    assert result.sampling_rate is None  # time dimension removed
```

### `test_granger_causality_matrix_no_mutation`
```python
def test_granger_causality_matrix_no_mutation() -> None:
    """GrangerCausalityMatrix does not modify input Data."""
    data = cb.SignalData.from_numpy(
        _create_causal_signal(n_samples=200),
        dims=["time", "space"],
        subjectID="s1",
        groupID="g1",
        condition="rest",
        sampling_rate=100.0,
    )
    original_history = list(data.history)
    original_shape = data.data.shape

    _ = cb.feature.GrangerCausalityMatrix(coords=[0, 1], lag=2).apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
```

## Action List

1. [Severity: HIGH] Coverage is 93% (target ≥95%). Add tests for:
   - `test_granger_causality_matrix_multiple_lags` (covers lines 206, 211, 227, 235)
   - `test_granger_causality_matrix_invalid_lag` (covers line 187)
   - `test_granger_causality_matrix_invalid_maxlag` (covers line 189)

2. [Severity: MEDIUM] Fix `test_granger_causality_history_tracking`: change `result.data.attrs.get("history", [])` to `result.history` and assert `result.history[-1] == "GrangerCausality"`

3. [Severity: MEDIUM] Fix `test_granger_causality_preserves_metadata`: use direct property access (`result.subjectID`, etc.) instead of `result.data.attrs.get()`, and assert `sampling_rate is None` since output_type=Data removes time dimension

4. [Severity: MEDIUM] Add `test_granger_causality_no_mutation` to verify input Data is not modified

5. [Severity: MEDIUM] Add `test_granger_causality_returns_data_instance` and `test_granger_causality_matrix_returns_data_instance` to explicitly verify return type

6. [Severity: LOW] Add `test_granger_causality_matrix_history_updated` for completeness

7. [Severity: LOW] Add `test_granger_causality_matrix_preserves_metadata` for completeness

8. [Severity: LOW] Add `test_granger_causality_matrix_no_mutation` for completeness

9. [Severity: LOW] Add docstrings to tests that lack them (some tests like `test_granger_causality_correct_value`, `test_granger_causality_detects_causality` are missing docstrings)
