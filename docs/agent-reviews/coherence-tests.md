# Test Review: coherence

**Feature**: `src/cobrabox/features/coherence.py`
**Test file**: `tests/test_feature_coherence.py`
**Date**: 2025-03-04
**Verdict**: NEEDS WORK

## Summary

The coherence test suite is comprehensive with 13 tests covering happy path, numerical correctness, error handling, and metadata propagation. However, three gaps exist: missing 'space' dimension validation is untested, `__post_init__` construction-time validation for `nperseg < 2` is missing, and no test verifies the input Data is not mutated.

## Keep

Tests that are correct and complete — no changes needed:

- `test_coherence_output_dims_and_shape` — correctly asserts output dims are `(space, space_to)` and sizes match input
- `test_coherence_space_coords_are_preserved` — verifies channel coordinates propagate to both space dims
- `test_coherence_identical_channels_give_unity_coherence` — strong numerical correctness test for identical signals
- `test_coherence_diagonal_is_nan` — verifies self-coherence is NaN as designed
- `test_coherence_matrix_is_symmetric` — confirms C[i,j] == C[j,i]
- `test_coherence_values_are_in_unit_range` — checks all values in [0, 1]
- `test_coherence_preserves_metadata_and_history` — checks subjectID, groupID, condition, extra, history; correctly notes sampling_rate becomes None (output_type = Data)
- `test_coherence_with_run_index_preserves_extra_dim` — thorough extra dimension handling
- `test_coherence_custom_nperseg_produces_valid_output` — valid param variation test
- `test_coherence_results_depend_on_nperseg` — verifies different nperseg produces different estimates
- `test_coherence_raises_when_fewer_than_two_channels` — runtime guard for n_space < 2
- `test_coherence_raises_when_nperseg_exceeds_n_time` — runtime guard for seg > n_time
- `test_coherence_accessible_via_feature_module` — API accessibility check

## Fix

Tests that exist but need changes:

### `test_coherence_raises_when_nperseg_is_less_than_two`

**Issue**: Tests runtime guard (lines 104-105) but feature also has `__post_init__` validation (lines 51-52). Need to test both construction-time and runtime errors.

```python
def test_coherence_raises_when_nperseg_is_less_than_two_at_construction() -> None:
    """Coherence raises ValueError when nperseg < 2 at construction time."""
    with pytest.raises(ValueError, match="nperseg"):
        cb.feature.Coherence(nperseg=1)


def test_coherence_raises_when_nperseg_is_less_than_two_at_runtime() -> None:
    """Coherence raises ValueError when nperseg < 2 at runtime (auto-derived)."""
    data = _make_data(np.ones((2, 3)))  # n_time=2, so min(256, 2) = 2, which is valid
    # Force an invalid n_time by creating data with n_time=1
    data_invalid = _make_data(np.ones((1, 3)))
    with pytest.raises(ValueError, match="nperseg"):
        cb.feature.Coherence().apply(data_invalid)
```

## Add

Missing scenarios — new tests to add:

### `test_coherence_raises_when_missing_space_dimension`

```python
def test_coherence_raises_when_missing_space_dimension() -> None:
    """Coherence raises ValueError when 'space' dimension is missing."""
    import xarray as xr
    # Create data without space dimension
    arr = np.random.randn(100)
    xr_data = xr.DataArray(arr, dims=["time"])
    raw = cb.Data.__new__(cb.Data)
    object.__setattr__(raw, "_data", xr_data)
    with pytest.raises(ValueError, match="space"):
        cb.feature.Coherence().apply(raw)
```

### `test_coherence_does_not_mutate_input`

```python
def test_coherence_does_not_mutate_input() -> None:
    """Coherence.apply() leaves the input Data object unchanged."""
    rng = np.random.default_rng(10)
    data = _make_data(rng.standard_normal((300, 4)))
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.Coherence().apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
```

### `test_coherence_post_init_validates_nperseg`

```python
def test_coherence_post_init_validates_nperseg() -> None:
    """Coherence raises ValueError for nperseg < 2 at construction time."""
    with pytest.raises(ValueError, match="nperseg must be >= 2"):
        cb.feature.Coherence(nperseg=1)
    with pytest.raises(ValueError, match="nperseg must be >= 2"):
        cb.feature.Coherence(nperseg=0)
    with pytest.raises(ValueError, match="nperseg must be >= 2"):
        cb.feature.Coherence(nperseg=-5)
```

## Action List

1. **[Severity: HIGH]** Add test for missing 'space' dimension validation (`test_coherence_raises_when_missing_space_dimension`)

2. **[Severity: MEDIUM]** Add test for input mutation (`test_coherence_does_not_mutate_input`)

3. **[Severity: MEDIUM]** Split existing `test_coherence_raises_when_nperseg_is_less_than_two` into two tests: one for construction-time validation (`__post_init__`) and one for runtime validation, or add a new test for `__post_init__` validation
