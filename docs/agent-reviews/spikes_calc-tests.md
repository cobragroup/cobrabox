# Test Review: spikes_calc

**Feature**: `src/cobrabox/features/spikes_calc.py`
**Test file**: `tests/test_feature_spikes_calc.py`
**Date**: 2026-03-04
**Verdict**: NEEDS WORK

## Summary

The test file covers basic functionality and edge cases well, but is missing a test for the `output_type = Data` behavior (verifying `sampling_rate` becomes `None`) and lacks a no-mutation test. Also uses `np.testing.assert_raises` instead of `pytest.raises`.

## Keep

Tests that are correct and complete — no changes needed:

- `test_spikes_calc_clean_data_no_outliers` — good happy path with clean data
- `test_spikes_calc_with_outliers` — tests outlier detection correctly
- `test_spikes_calc_preserves_metadata` — comprehensive metadata checks
- `test_spikes_calc_returns_scalar` — verifies scalar output shape
- `test_spikes_calc_multivariate_data` — tests multiple channels

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

```python
def test_spikes_calc_sampling_rate_none() -> None:
    """SpikesCalc sets sampling_rate to None when time dimension is removed."""
    arr = np.random.randn(100, 2)
    data = cb.Data.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)
    out = cb.feature.SpikesCalc().apply(data)
    assert out.sampling_rate is None
```

### `test_spikes_calc_does_not_mutate_input`

```python
def test_spikes_calc_does_not_mutate_input(rng: np.random.Generator) -> None:
    """SpikesCalc does not modify the input Data object."""
    arr = rng.standard_normal((50, 3))
    data = cb.Data.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.SpikesCalc().apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
```

## Action List

1. [Severity: MEDIUM] Fix `test_spikes_calc_empty_data_raises` — use `pytest.raises(ValueError, match="empty")` instead of `np.testing.assert_raises`

2. [Severity: MEDIUM] Add `test_spikes_calc_sampling_rate_none` — verify `output_type = Data` correctly sets `sampling_rate` to `None`

3. [Severity: MEDIUM] Add `test_spikes_calc_does_not_mutate_input` — verify input Data is not modified
