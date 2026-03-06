# Test Review: partial_directed_coherence

**Feature**: `src/cobrabox/features/partial_directed_coherence.py`
**Test file**: `tests/test_feature_partial_directed_coherence.py`
**Date**: 2025-03-05
**Verdict**: NEEDS WORK

## Coverage

```text
PartialDirectedCoherence: 98% (51 statements, 1 missing)
```

Missing line 69: the 3-D input validation check in `__call__`:

```python
if xr_data.ndim != 2:
    raise ValueError(...)
```

Coverage is above the 95% threshold, but the uncovered line represents a validation path that should be tested.

## Summary

The test file has 15 tests with good coverage of the core PDC algorithm, output structure, numerical correctness (column sums to 1, values in [0, 1], directional coupling detection), and parameter validation. However, it is missing required scenarios for metadata preservation and input mutation testing.

## Keep

Tests that are correct and complete — no changes needed:

- `test_pdc_is_registered` — Feature discovery check ✓
- `test_pdc_output_dims` — Correct output dimensions ✓
- `test_pdc_output_shape` — Correct output shape with custom n_freqs ✓
- `test_pdc_space_coords_preserved` — Space coordinate propagation ✓
- `test_pdc_frequency_coord_range` — Frequency range [0, sr/2] ✓
- `test_pdc_returns_data_not_signal_data` — Output type is Data ✓
- `test_pdc_history_updated` — History tracking ✓
- `test_pdc_values_in_unit_interval` — PDC values in [0, 1] ✓
- `test_pdc_column_sums_to_one` — Normalization property ✓
- `test_pdc_directed_coupling_detected` — Directional detection ✓
- `test_pdc_fixed_var_order` — Fixed var_order parameter ✓
- `test_pdc_requires_sampling_rate` — Missing sampling_rate error ✓
- `test_pdc_requires_at_least_2_channels` — Single channel error ✓
- `test_pdc_invalid_n_freqs_raises` — n_freqs validation ✓
- `test_pdc_invalid_var_order_raises` — var_order validation ✓

## Fix

None required — all existing tests are correct.

## Add

Missing scenarios — new tests to add:

### `test_pdc_metadata_preserved`

```python
def test_pdc_metadata_preserved() -> None:
    """PDC preserves subjectID, groupID, condition; sampling_rate is None (no time dim)."""
    rng = np.random.default_rng(11)
    data = _make_data(
        rng.standard_normal((3, 400)),
        space=["Fz", "Cz", "Pz"],
    )
    # Set metadata manually since _make_data doesn't support it
    data = data._replace(subjectID="s42", groupID="control", condition="task")
    
    out = cb.feature.PartialDirectedCoherence().apply(data)
    
    assert out.subjectID == "s42"
    assert out.groupID == "control"
    assert out.condition == "task"
    assert out.sampling_rate is None  # output_type=Data removes time dim
```

### `test_pdc_does_not_mutate_input`

```python
def test_pdc_does_not_mutate_input() -> None:
    """PDC.apply() leaves the input Data object unchanged."""
    rng = np.random.default_rng(12)
    arr = rng.standard_normal((3, 400))
    data = _make_data(arr, space=["Fz", "Cz", "Pz"])
    
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()
    
    _ = cb.feature.PartialDirectedCoherence().apply(data)
    
    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
```

### `test_pdc_invalid_ndim_raises`

```python
def test_pdc_invalid_ndim_raises() -> None:
    """PDC raises ValueError for non-2-D input (3-D array)."""
    rng = np.random.default_rng(13)
    # Create 3-D data (space x time x extra)
    arr = rng.standard_normal((3, 100, 2))
    xr_arr = xr.DataArray(arr, dims=["space", "time", "trial"])
    data = cb.SignalData.from_xarray(xr_arr, sampling_rate=250.0)
    
    with pytest.raises(ValueError, match="2-D input"):
        cb.feature.PartialDirectedCoherence().apply(data)
```

## Action List

1. [Severity: MEDIUM] Add `test_pdc_metadata_preserved` to verify subjectID, groupID, condition preservation and sampling_rate=None (tests/test_feature_partial_directed_coherence.py)

2. [Severity: MEDIUM] Add `test_pdc_does_not_mutate_input` to verify input Data is not modified (tests/test_feature_partial_directed_coherence.py)

3. [Severity: MEDIUM] Add `test_pdc_invalid_ndim_raises` to cover line 69 (3-D input validation) and bring coverage to 100% (tests/test_feature_partial_directed_coherence.py)
