# Test Review: reciprocal_connectivity

**Feature**: `src/cobrabox/features/reciprocal_connectivity.py`
**Test file**: `tests/test_feature_reciprocal_connectivity.py`
**Date**: 2026-03-05
**Verdict**: NEEDS WORK

## Coverage

```
ReciprocalConnectivity: 98% (62 statements, 1 missing)
Missing: line 175 (fallback when space_to not in mat.coords)
```

Coverage is excellent at 98%. The only uncovered line is the fallback path when `space_to` coords are missing (line 175), which assigns `np.arange(n_ch)` as default space values.

## Summary

The test file is comprehensive with 17 tests covering both usage paths (time-series VAR/PDC and pre-computed matrix), numerical correctness, and error validation. However, it misses two required scenarios: metadata preservation and input non-mutation. Test function naming also deviates from the project's naming convention.

## Keep

Tests that are correct and complete:

- `test_rc_is_registered` — verifies feature auto-discovery
- `test_rc_from_timeseries_output_dims` — validates output dims
- `test_rc_from_timeseries_output_shape` — validates output shape
- `test_rc_from_timeseries_space_coords_preserved` — validates coord propagation
- `test_rc_history_updated` — validates history appends class name
- `test_rc_driver_is_negative_sink_is_positive` — numerical correctness with known driver/sink
- `test_rc_normalize_changes_values_not_shape` — validates normalize parameter
- `test_rc_from_precomputed_2d_matrix` — pre-computed matrix path (2D)
- `test_rc_from_precomputed_3d_matrix_with_freq_band` — pre-computed matrix with freq averaging
- `test_rc_precomputed_space_coords_preserved` — coord propagation for matrix input
- `test_rc_unsupported_connectivity_raises` — validates connectivity parameter
- `test_rc_symmetric_matrix_raises` — validates asymmetry requirement
- `test_rc_freq_band_set_but_no_freq_dim_raises` — validates freq_band/dim consistency
- `test_rc_freq_band_outside_range_raises` — validates freq_band bounds
- `test_rc_freq_band_none_but_freq_dim_present_raises` — validates freq_band=None handling
- `test_rc_invalid_freq_band_fmin_ge_fmax_raises` — validates **post_init** constraint
- `test_rc_precomputed_missing_space_dims_raises` — validates required dims

## Fix

### Function naming convention

Issue: Tests use `test_rc_*` prefix instead of `test_reciprocal_connectivity_*` as required by criteria.md.

```python
# Current (non-compliant):
def test_rc_basic() -> None:

# Should be:
def test_reciprocal_connectivity_basic() -> None:
```

All 17 test functions should be renamed from `test_rc_*` to `test_reciprocal_connectivity_*`.

## Add

### `test_reciprocal_connectivity_metadata_preserved`

Missing required scenario per criteria.md: metadata (subjectID, groupID, condition, sampling_rate) must be preserved.

```python
def test_reciprocal_connectivity_metadata_preserved() -> None:
    """ReciprocalConnectivity preserves subjectID, groupID, condition; sampling_rate becomes None."""
    labels = ["Fz", "Cz", "Pz"]
    rng = np.random.default_rng(42)
    arr = rng.standard_normal((3, 800))
    data = _make_signal_data(
        arr,
        space=labels,
        sampling_rate=250.0,
    )
    # Set metadata via extra or direct assignment if supported
    data = data.with_attrs(
        subjectID="s42",
        groupID="control",
        condition="task"
    )
    
    out = cb.feature.ReciprocalConnectivity(freq_band=(10.0, 60.0)).apply(data)
    
    assert out.subjectID == "s42"
    assert out.groupID == "control"
    assert out.condition == "task"
    # output_type = Data removes time dimension, so sampling_rate is None
    assert out.sampling_rate is None
```

### `test_reciprocal_connectivity_does_not_mutate_input`

Missing required scenario per criteria.md: input Data must not be mutated.

```python
def test_reciprocal_connectivity_does_not_mutate_input() -> None:
    """ReciprocalConnectivity.apply() leaves the input Data object unchanged."""
    labels = ["Fz", "Cz"]
    rng = np.random.default_rng(43)
    arr = rng.standard_normal((2, 800))
    data = _make_signal_data(arr, space=labels)
    
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()
    
    _ = cb.feature.ReciprocalConnectivity(freq_band=(10.0, 60.0)).apply(data)
    
    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
```

### `test_reciprocal_connectivity_no_space_coords_fallback`

Optional: cover the missing line 175 (fallback when space_to not in coords).

```python
def test_reciprocal_connectivity_no_space_coords_fallback() -> None:
    """When space_to coords are missing, output uses integer indices."""
    mat = np.array([[0.0, 0.3], [0.7, 0.0]])
    # Create Data without space_to/space_from coordinate values
    xr_arr = xr.DataArray(mat, dims=["space_to", "space_from"])
    data = cb.Data.from_xarray(xr_arr)
    
    out = cb.feature.ReciprocalConnectivity(freq_band=None).apply(data)
    
    assert out.data.dims == ("space",)
    # Should use integer indices 0, 1 as fallback
    np.testing.assert_array_equal(out.data.coords["space"].values, [0, 1])
```

## Action List

1. [Severity: MEDIUM] Rename all test functions from `test_rc_*` to `test_reciprocal_connectivity_*` pattern (17 functions)
2. [Severity: HIGH] Add `test_reciprocal_connectivity_metadata_preserved` — missing required scenario
3. [Severity: MEDIUM] Add `test_reciprocal_connectivity_does_not_mutate_input` — missing required scenario
4. [Severity: LOW] Add `test_reciprocal_connectivity_no_space_coords_fallback` — cover line 175 to reach 100%
