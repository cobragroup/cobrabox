# Test Review: lempel_ziv

**Feature**: `src/cobrabox/features/lempel_ziv.py`
**Test file**: `tests/test_feature_lempel_ziv.py`
**Date**: 2025-03-05
**Verdict**: NEEDS WORK

## Coverage

```
LempelZiv: 100% (47 statements, 0 missing)
```

Excellent coverage - all lines of the feature are tested.

## Summary

The LempelZiv test suite is comprehensive and well-written with 100% coverage. It includes 6 tests covering basic functionality, known values, complexity comparison, positivity, Chord integration, and multichannel independence. However, two required scenarios from the criteria are missing: testing for missing `time` dimension and verifying no mutation of input.

## Keep

Tests that are correct and complete:
- `test_feature_lempel_ziv_output_type_and_history` — correctly checks output type, dims, shape, metadata preservation, history, and sampling_rate=None
- `test_feature_lempel_ziv_known_value` — tests against hand-calculated expected value
- `test_feature_lempel_ziv_random_more_complex_than_periodic` — good behavioral test showing random > periodic complexity
- `test_feature_lempel_ziv_values_are_positive` — verifies output constraints
- `test_feature_lempel_ziv_via_chord` — tests Chord composition with history propagation
- `test_feature_lempel_ziv_multichannel_independent` — verifies per-channel independence

## Fix

None - all existing tests are correct.

## Add

Missing scenarios to add:

### `test_feature_lempel_ziv_missing_time_raises`

```python
def test_feature_lempel_ziv_missing_time_raises() -> None:
    """LempelZiv raises ValueError when time dimension is missing."""
    import xarray as xr
    arr = np.random.randn(10)
    xr_data = xr.DataArray(arr, dims=["space"])
    raw = cb.Data.__new__(cb.Data)
    object.__setattr__(raw, "_data", xr_data)
    with pytest.raises(ValueError, match="time"):
        cb.feature.LempelZiv().apply(raw)
```

### `test_feature_lempel_ziv_does_not_mutate_input`

```python
def test_feature_lempel_ziv_does_not_mutate_input() -> None:
    """LempelZiv does not modify the input Data object."""
    rng = np.random.default_rng(8)
    arr = rng.standard_normal((100, 3))
    data = cb.SignalData.from_numpy(
        arr, dims=["time", "space"], sampling_rate=100.0, subjectID="s1"
    )
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.LempelZiv().apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
```

## Action List

1. [Severity: MEDIUM] Add `test_feature_lempel_ziv_missing_time_raises` to verify error handling when input lacks time dimension
2. [Severity: MEDIUM] Add `test_feature_lempel_ziv_does_not_mutate_input` to verify input immutability
