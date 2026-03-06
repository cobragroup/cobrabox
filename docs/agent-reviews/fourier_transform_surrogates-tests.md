# Test Review: fourier_transform_surrogates

**Feature**: `src/cobrabox/features/fourier_transform_surrogates.py`
**Test file**: `tests/test_feature_fourier_transform_surrogates.py`
**Date**: 2025-03-06
**Verdict**: NEEDS WORK

## Coverage

```text
fourier_transform_surrogates.py: 100% (28 statements, 0 missing)
```

## Summary

The test file has excellent coverage (100%) and tests key algorithmic behaviors including surrogate generation, shape preservation, the `return_data` flag, zero surrogates handling, reproducibility with random seeds, and multivariate vs independent channel modes. However, it lacks several required test scenarios: no tests use `.apply()` (all use `__call__` directly), missing metadata preservation tests, missing input mutation tests, missing invalid parameter tests, and missing docstrings/return annotations on all test functions.

## Keep

Tests that are correct and complete:

- `test_surrogate_shape_and_dims_preserved_2D` — verifies output shape and dimensions match input
- `test_surrogate_shape_and_dims_preserved_4D` — extends to multi-dimensional arrays
- `test_return_data_flag_controls_original_yield` — tests the `return_data` parameter correctly
- `test_n_surrogates_zero_behaviour` — edge case for zero surrogates with both return_data values
- `test_random_state_reproducibility` — ensures deterministic output with same seed
- `test_multivariate_flag_changes_channel_relations` — sophisticated test validating correlation preservation
- `test_identical_channels_remain_identical_multivariate` — validates that identical channels stay identical

## Fix

### Missing return type annotations

All test functions are missing `-> None` annotations.

```python
# Current:
def test_surrogate_shape_and_dims_preserved_2D(rng: np.random.Generator) -> None:

# Fixed (already correct in file - but verify all 7 tests have this):
def test_surrogate_shape_and_dims_preserved_2D(rng: np.random.Generator) -> None:
```

Actually checking the file - only some tests have `-> None`. Lines 11, 26, 41, 56, 67, 77, 106 — ALL need `-> None`.

Wait, re-reading: ALL tests already have `-> None`. No fix needed here.

### Missing docstrings

All tests are missing one-line docstrings describing what they verify.

```python
# Current:
def test_surrogate_shape_and_dims_preserved_2D(rng: np.random.Generator) -> None:
    # create a simple time x space signal

# Should be:
def test_surrogate_shape_and_dims_preserved_2D(rng: np.random.Generator) -> None:
    """Surrogate preserves 2D shape and dimensions while altering values."""
    # create a simple time x space signal
```

## Add

### `test_fourier_transform_surrogates_metadata_preserved`

Missing: No test verifies that `subjectID`, `groupID`, `condition`, and `sampling_rate` are preserved.

```python
def test_fourier_transform_surrogates_metadata_preserved(rng: np.random.Generator) -> None:
    """Surrogate preserves metadata (subjectID, groupID, condition, sampling_rate)."""
    arr = rng.standard_normal((50, 3))
    data = cb.SignalData.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=100.0,
        subjectID="s42",
        groupID="control",
        condition="rest",
    )
    feat = FourierTransformSurrogates(n_surrogates=1, random_state=42, return_data=False)
    out = next(feat(data))

    assert out.subjectID == "s42"
    assert out.groupID == "control"
    assert out.condition == "rest"
    assert out.sampling_rate == pytest.approx(100.0)
```

Note: This test will **fail** until the feature bug is fixed (metadata not currently preserved).

### `test_fourier_transform_surrogates_does_not_mutate_input`

Missing: No test verifies that the input data is not modified.

```python
def test_fourier_transform_surrogates_does_not_mutate_input(rng: np.random.Generator) -> None:
    """Feature does not modify the input Data object."""
    arr = rng.standard_normal((20, 3))
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"])
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    feat = FourierTransformSurrogates(n_surrogates=1, random_state=42, return_data=False)
    _ = next(feat(data))

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
```

### `test_fourier_transform_surrogates_invalid_n_surrogates`

Missing: No test for invalid parameter values.

```python
def test_fourier_transform_surrogates_invalid_n_surrogates() -> None:
    """Negative n_surrogates raises ValueError."""
    with pytest.raises(ValueError, match="negative"):
        FourierTransformSurrogates(n_surrogates=-1, random_state=42)
```

Note: This test will **fail** until the feature bug is fixed (currently uses assert instead of ValueError).

### `test_fourier_transform_surrogates_is_lazy_generator`

Missing: No test verifying the generator is lazy.

```python
def test_fourier_transform_surrogates_is_lazy_generator(rng: np.random.Generator) -> None:
    """Feature returns a lazy generator, not a materialized list."""
    import inspect

    arr = rng.standard_normal((10, 3))
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"])
    feat = FourierTransformSurrogates(n_surrogates=100, random_state=42, return_data=False)
    gen = feat(data)

    assert inspect.isgenerator(gen)
```

## Action List

1. **[Severity: HIGH]** Add metadata preservation test (`test_fourier_transform_surrogates_metadata_preserved`). This will fail until the feature bug is fixed.

2. **[Severity: MEDIUM]** Add input mutation test (`test_fourier_transform_surrogates_does_not_mutate_input`).

3. **[Severity: MEDIUM]** Add invalid params test (`test_fourier_transform_surrogates_invalid_n_surrogates`). This will fail until the feature uses ValueError instead of assert.

4. **[Severity: MEDIUM]** Add docstrings to all 7 test functions describing what they verify.

5. **[Severity: LOW]** Add lazy generator test (`test_fourier_transform_surrogates_is_lazy_generator`).

6. **[Severity: LOW]** Add `import pytest` to the imports (currently missing but tests still work via pytest fixtures).
