# Test Review: coherence

**Feature**: `src/cobrabox/features/coherence.py`
**Test file**: `tests/test_feature_coherence.py`
**Date**: 2026-03-05
**Verdict**: NEEDS WORK

## Coverage

```
Coherence: 97% (61 statements, 2 missing)
```

Missing lines:
- Line 94: `raise ValueError("data must have 'space' dimension")`
- Line 105: `raise ValueError("nperseg must be at least 2")`

Coverage is above the 95% threshold, but two edge cases are not tested.

## Summary

The Coherence feature has a solid test suite with 14 tests covering output structure, numerical correctness, metadata preservation, extra dimensions, custom nperseg, and error handling. The tests are well-organized with clear docstrings. However, two edge case error paths are not covered: missing 'space' dimension and the runtime guard for nperseg < 2 when using default value.

## Keep

Tests that are correct and complete:
- `test_coherence_output_dims_and_shape` — Correctly checks output is Data with (space, space_to) dims
- `test_coherence_space_coords_are_preserved` — Verifies coordinate preservation
- `test_coherence_identical_channels_give_unity_coherence` — Tests unity coherence for identical signals
- `test_coherence_diagonal_is_nan` — Verifies diagonal is NaN
- `test_coherence_matrix_is_symmetric` — Confirms matrix symmetry
- `test_coherence_values_are_in_unit_range` — Validates [0, 1] range
- `test_coherence_preserves_metadata_and_history` — Correctly tests metadata and history propagation
- `test_coherence_with_run_index_preserves_extra_dim` — Tests extra dimension handling
- `test_coherence_custom_nperseg_produces_valid_output` — Tests custom nperseg
- `test_coherence_results_depend_on_nperseg` — Tests different nperseg values produce different results
- `test_coherence_raises_when_fewer_than_two_channels` — Tests <2 channels error
- `test_coherence_raises_when_nperseg_exceeds_n_time` — Tests nperseg > n_time error
- `test_coherence_raises_when_nperseg_is_less_than_two` — Tests __post_init__ validation
- `test_coherence_accessible_via_feature_module` — Tests API accessibility

## Fix

No tests need fixing.

## Add

Missing edge case tests:

### `test_coherence_missing_space_dim_raises`

Tests line 94: raises ValueError when 'space' dimension is missing.

```python
def test_coherence_missing_space_dim_raises() -> None:
    """Coherence raises ValueError when data lacks 'space' dimension."""
    import xarray as xr
    arr = np.random.default_rng(100).standard_normal((100, 5))
    xr_data = xr.DataArray(arr, dims=["time", "channel"])
    raw = cb.SignalData.__new__(cb.SignalData)
    object.__setattr__(raw, "_data", xr_data)
    object.__setattr__(raw, "_sampling_rate", 100.0)
    
    with pytest.raises(ValueError, match="space"):
        cb.feature.Coherence().apply(raw)
```

### `test_coherence_default_nperseg_too_small_raises`

Tests line 105: raises ValueError when default nperseg (min(256, n_time)) is < 2.

```python
def test_coherence_default_nperseg_too_small_raises() -> None:
    """Coherence raises ValueError when n_time < 2 causes default nperseg < 2."""
    rng = np.random.default_rng(101)
    data = _make_data(rng.standard_normal((1, 3)))  # n_time=1
    
    with pytest.raises(ValueError, match="nperseg"):
        cb.feature.Coherence().apply(data)
```

### `test_coherence_does_not_mutate_input`

Tests that input Data is not modified (pattern from criteria.md).

```python
def test_coherence_does_not_mutate_input() -> None:
    """Coherence.apply() leaves the input Data object unchanged."""
    rng = np.random.default_rng(102)
    data = _make_data(rng.standard_normal((200, 3)))
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()
    
    _ = cb.feature.Coherence().apply(data)
    
    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
```

## Action List

1. [Severity: MEDIUM] Add test for missing 'space' dimension (line 94)
2. [Severity: MEDIUM] Add test for default nperseg < 2 edge case (line 105)
3. [Severity: LOW] Add test for input mutation prevention
