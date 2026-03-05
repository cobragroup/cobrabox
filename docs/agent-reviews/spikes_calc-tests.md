# Test Review: spikes_calc

**Feature**: `src/cobrabox/features/spikes_calc.py`
**Test file**: `tests/test_feature_spikes_calc.py`
**Date**: 2026-03-05
**Verdict**: NEEDS WORK

## Coverage

```
spikes_calc.py: 100% (21 statements, 0 missing)
```

## Summary

Good test coverage at 100% with 6 tests covering happy path, outliers detection, metadata preservation, scalar output shape, multivariate data, and empty data error handling. Test file follows naming conventions and all tests have docstrings and return annotations.

Missing tests for sampling_rate handling (output_type = Data removes time dimension), input mutation verification, and boundary condition testing for IQR-based spike detection.

## Keep

Tests that are correct and complete — no changes needed:
- `test_spikes_calc_clean_data_no_outliers` — correctly tests normal data returns few/no spikes
- `test_spikes_calc_with_outliers` — correctly tests extreme values are detected
- `test_spikes_calc_preserves_metadata` — verifies subjectID, groupID, condition, extra preserved
- `test_spikes_calc_returns_scalar` — verifies output shape () and scalar type
- `test_spikes_calc_multivariate_data` — tests multiple channels with different spike patterns

## Fix

Tests that exist but need changes:

### `test_spikes_calc_empty_data_raises`

Issue: Uses `np.testing.assert_raises(ValueError)` instead of `pytest.raises`. Should use `pytest.raises` with a `match` parameter for better error message validation.

```python
def test_spikes_calc_empty_data_raises() -> None:
    """SpikesCalc raises ValueError for empty input data."""
    arr = np.array([]).reshape(0, 0)
    data = cb.Data.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    with pytest.raises(ValueError, match="empty"):
        cb.feature.SpikesCalc().apply(data)
```

## Add

Missing scenarios — new tests to add:

### `test_spikes_calc_sampling_rate_none`

Since `output_type: ClassVar[type[Data]] = Data`, the time dimension is removed and `sampling_rate` should become `None`.

```python
def test_spikes_calc_sampling_rate_none(rng: np.random.Generator) -> None:
    """SpikesCalc sets sampling_rate to None when time dimension is removed."""
    arr = rng.standard_normal((50, 2))
    data = cb.Data.from_numpy(
        arr, dims=["time", "space"], sampling_rate=100.0
    )

    out = cb.feature.SpikesCalc().apply(data)

    assert out.sampling_rate is None
```

### `test_spikes_calc_does_not_mutate_input`

Verify the input Data object is unchanged after apply().

```python
def test_spikes_calc_does_not_mutate_input(rng: np.random.Generator) -> None:
    """SpikesCalc does not modify the input Data object."""
    arr = rng.standard_normal((50, 2))
    data = cb.Data.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=100.0,
        subjectID="sub-01",
        history=["PreviousFeature"],
    )

    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.SpikesCalc().apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
    assert data.subjectID == "sub-01"
```

### `test_spikes_calc_boundary_values`

Test values exactly at IQR bounds (±1.5*IQR from Q1/Q3) are NOT counted as spikes.

```python
def test_spikes_calc_boundary_values() -> None:
    """Values exactly at IQR bounds are not counted as spikes."""
    # Create data where Q1=25, Q3=75, IQR=50
    # Bounds: low = 25 - 75 = -50, high = 75 + 75 = 150
    arr = np.array([25.0, 75.0, 0.0, 100.0, -50.0, 150.0])  # Values at bounds

    data = cb.Data.from_numpy(
        arr.reshape(-1, 1), dims=["time", "space"], sampling_rate=100.0
    )

    out = cb.feature.SpikesCalc().apply(data)

    # Values exactly at bounds should NOT be spikes
    # But anything beyond should be
    assert out.to_numpy() == 0
```

## Action List

1. [Severity: MEDIUM] Fix `test_spikes_calc_empty_data_raises` — use `pytest.raises(ValueError, match="empty")` instead of `np.testing.assert_raises`

2. [Severity: MEDIUM] Add `test_spikes_calc_sampling_rate_none` to verify sampling_rate becomes None when output_type = Data removes time dimension

3. [Severity: MEDIUM] Add `test_spikes_calc_does_not_mutate_input` to verify input Data is not mutated

4. [Severity: LOW] Add `test_spikes_calc_boundary_values` to clarify spike detection boundary behavior (values at exactly ±1.5*IQR are not spikes)
