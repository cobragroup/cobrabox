# Test Review: autocorr

**Feature**: `src/cobrabox/features/autocorr.py`
**Test file**: `tests/test_feature_autocorr.py`
**Date**: 2026-03-04
**Verdict**: NEEDS WORK

## Summary

The test file has good coverage of the core functionality and error cases. However, it is missing tests for:

1. `fs <= 0` validation in `__post_init__`
2. `lag_ms` parameter usage (only `lag_steps` is tested)
3. Metadata preservation beyond `subjectID` (`groupID`, `condition`)
4. `sampling_rate` becoming `None` since `output_type = Data`
5. No mutation of input Data

## Keep

Tests that are correct and complete — no changes needed:

- `test_feature_autocorr_reduces_requested_dimension` — correctly checks output shape, dims, values, type, and subjectID preservation
- `test_feature_autocorr_default_5ms_matches_explicit_steps` — validates default behavior and uses proper seeded RNG
- `test_feature_autocorr_raises_for_unknown_dimension` — validates missing dimension guard
- `test_feature_autocorr_raises_when_both_lag_inputs_provided` — validates mutual exclusivity of lag params
- `test_feature_autocorr_raises_for_lag_out_of_range` — validates runtime lag bounds check
- `test_feature_autocorr_constant_signal_returns_nan` — tests edge case (zero variance signal)
- `test_feature_autocorr_all_nan_returns_nan` — tests edge case (all-NaN input)

## Fix

None — all existing tests are correct.

## Add

Missing scenarios — new tests to add:

### `test_feature_autocorr_fs_non_positive_raises`

```python
def test_feature_autocorr_fs_non_positive_raises() -> None:
    """Autocorr raises at construction when fs is zero or negative."""
    with pytest.raises(ValueError, match="fs must be positive"):
        cb.feature.Autocorr(dim="time", fs=0.0)
    with pytest.raises(ValueError, match="fs must be positive"):
        cb.feature.Autocorr(dim="time", fs=-100.0)
```

### `test_feature_autocorr_lag_ms_computation`

```python
def test_feature_autocorr_lag_ms_computation() -> None:
    """Autocorr correctly computes lag from lag_ms parameter."""
    arr = np.random.default_rng(42).normal(size=(50, 2)).astype(float)
    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=1000.0)

    # fs=1000 Hz, lag_ms=5 → lag_steps=5
    out_ms = cb.feature.Autocorr(dim="time", fs=1000.0, lag_ms=5.0).apply(data)
    out_steps = cb.feature.Autocorr(dim="time", fs=1000.0, lag_steps=5).apply(data)

    np.testing.assert_allclose(out_ms.to_numpy(), out_steps.to_numpy(), equal_nan=True)
```

### `test_feature_autocorr_metadata_preserved`

```python
def test_feature_autocorr_metadata_preserved() -> None:
    """Autocorr preserves subjectID, groupID, condition; sampling_rate becomes None."""
    arr = np.random.default_rng(0).normal(size=(30, 2)).astype(float)
    data = cb.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=1000.0,
        subjectID="sub-01",
        groupID="control",
        condition="rest",
    )

    out = cb.feature.Autocorr(dim="time", fs=1000.0, lag_steps=1).apply(data)

    assert out.subjectID == "sub-01"
    assert out.groupID == "control"
    assert out.condition == "rest"
```

### `test_feature_autocorr_sampling_rate_none`

```python
def test_feature_autocorr_sampling_rate_none() -> None:
    """Autocorr sets sampling_rate to None when time dimension is removed."""
    arr = np.random.default_rng(0).normal(size=(30, 2)).astype(float)
    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=1000.0)

    out = cb.feature.Autocorr(dim="time", fs=1000.0, lag_steps=1).apply(data)

    assert out.sampling_rate is None
```

### `test_feature_autocorr_does_not_mutate_input`

```python
def test_feature_autocorr_does_not_mutate_input() -> None:
    """Autocorr.apply() leaves the input Data object unchanged."""
    arr = np.random.default_rng(0).normal(size=(20, 2)).astype(float)
    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=1000.0)

    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.Autocorr(dim="time", fs=1000.0, lag_steps=1).apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
```

## Action List

1. [Severity: MEDIUM] Add test for `fs <= 0` validation (`test_feature_autocorr_fs_non_positive_raises`)
2. [Severity: MEDIUM] Add test for `lag_ms` parameter (`test_feature_autocorr_lag_ms_computation`)
3. [Severity: MEDIUM] Add test for full metadata preservation (`test_feature_autocorr_metadata_preserved`)
4. [Severity: MEDIUM] Add test for `sampling_rate` becoming `None` (`test_feature_autocorr_sampling_rate_none`)
5. [Severity: MEDIUM] Add test for input mutation protection (`test_feature_autocorr_does_not_mutate_input`)
