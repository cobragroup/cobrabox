# Test Review: autocorr

**Feature**: `src/cobrabox/features/autocorr.py`
**Test file**: `tests/test_feature_autocorr.py`
**Date**: 2025-03-05
**Verdict**: NEEDS WORK

## Coverage

```
Autocorr: 98% (46 statements, 1 missing)
Missing: line 50 (fs <= 0 validation in __post_init__)
```

Coverage is at 98% — only the `fs <= 0` validation is uncovered. Target is ≥95%, so this is close but the missing line should be covered.

## Summary

Solid test suite with 7 passing tests covering happy path, dimension reduction, default lag behavior, edge cases (constant signal, all-NaN), and several error conditions. Test naming and docstrings follow conventions. Missing: input mutation guard, comprehensive metadata preservation (groupID, condition), sampling_rate becomes None verification, lag_ms parameter test, and the fs <= 0 validation.

## Keep

Tests that are correct and complete — no changes needed:

- `test_feature_autocorr_reduces_requested_dimension` — comprehensive happy path with shape, dims, values, history, and subjectID checks
- `test_feature_autocorr_default_5ms_matches_explicit_steps` — validates default lag computation
- `test_feature_autocorr_raises_for_unknown_dimension` — missing dim error case
- `test_feature_autocorr_raises_when_both_lag_inputs_provided` — __post_init__ validation
- `test_feature_autocorr_raises_for_lag_out_of_range` — runtime lag validation
- `test_feature_autocorr_constant_signal_returns_nan` — edge case (zero variance)
- `test_feature_autocorr_all_nan_returns_nan` — edge case (all NaN input)

## Fix

None — existing tests are correct.

## Add

Missing scenarios — new tests to add:

### `test_feature_autocorr_fs_non_positive_raises`

Line 50 is uncovered — test the fs validation.

```python
def test_feature_autocorr_fs_non_positive_raises() -> None:
    """Autocorr raises ValueError when fs is zero or negative."""
    with pytest.raises(ValueError, match="fs must be positive"):
        cb.feature.Autocorr(dim="time", fs=0.0)
    with pytest.raises(ValueError, match="fs must be positive"):
        cb.feature.Autocorr(dim="time", fs=-100.0)
```

### `test_feature_autocorr_metadata_preserved`

Add comprehensive metadata preservation test.

```python
def test_feature_autocorr_metadata_preserved() -> None:
    """Autocorr preserves subjectID, groupID, condition; sampling_rate becomes None."""
    arr = np.random.default_rng(42).normal(size=(20, 2)).astype(float)
    data = cb.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=1000.0,
        subjectID="sub-01",
        groupID="group-A",
        condition="task",
    )

    out = cb.feature.Autocorr(dim="time", fs=1000.0, lag_steps=1).apply(data)

    assert out.subjectID == "sub-01"
    assert out.groupID == "group-A"
    assert out.condition == "task"
    assert out.sampling_rate is None  # output_type = Data removes time dim
```

### `test_feature_autocorr_does_not_mutate_input`

Add input mutation guard.

```python
def test_feature_autocorr_does_not_mutate_input() -> None:
    """Autocorr.apply() does not modify the input Data object."""
    arr = np.arange(20, dtype=float).reshape(10, 2)
    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=1000.0, subjectID="s1")

    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.Autocorr(dim="time", fs=1000.0, lag_steps=1).apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
    assert data.subjectID == "s1"
```

### `test_feature_autocorr_lag_ms_parameter`

Test lag_ms parameter (currently only lag_steps is tested).

```python
def test_feature_autocorr_lag_ms_parameter() -> None:
    """Autocorr correctly computes lag from lag_ms parameter."""
    arr = np.random.default_rng(0).normal(size=(50, 2)).astype(float)
    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=1000.0)

    # fs=1000, lag_ms=5 -> lag = round(1000 * 5 / 1000) = 5
    out_ms = cb.feature.Autocorr(dim="time", fs=1000.0, lag_ms=5.0).apply(data)
    out_steps = cb.feature.Autocorr(dim="time", fs=1000.0, lag_steps=5).apply(data)

    np.testing.assert_allclose(out_ms.to_numpy(), out_steps.to_numpy(), equal_nan=True)
```

## Action List

1. [Severity: HIGH] Add test for `fs <= 0` validation (line 50): `test_feature_autocorr_fs_non_positive_raises`
2. [Severity: MEDIUM] Add metadata preservation test: `test_feature_autocorr_metadata_preserved`
3. [Severity: MEDIUM] Add input mutation guard: `test_feature_autocorr_does_not_mutate_input`
4. [Severity: LOW] Add lag_ms parameter test: `test_feature_autocorr_lag_ms_parameter`
