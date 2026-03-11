# Test Review: fourier_transform_surrogates

**Feature**: `src/cobrabox/features/fourier_transform_surrogates.py`
**Test file**: `tests/test_feature_fourier_transform_surrogates.py`
**Date**: 2026-03-06
**Verdict**: NEEDS WORK

## Coverage

```text
fourier_transform_surrogates.py: 97% (35 statements, 1 missing)
Missing line: 68
```

Coverage is below the 95% threshold. Line 68 (the `isinstance(self.n_surrogates, int)` check in `__post_init__`) is uncovered.

## Summary

The test suite covers the core functionality well, including shape preservation, metadata handling, random state reproducibility, and the multivariate flag behavior. The multivariate test is particularly thorough, verifying both correlation preservation and decorrelation. However, one validation branch is untested, preventing the file from meeting the 95% coverage requirement.

## Keep

Tests that are correct and complete:

- `test_surrogate_shape_and_dims_preserved_2D` — verifies 2D shape/dims preservation and value changes
- `test_surrogate_shape_and_dims_preserved_4D` — verifies 4D shape/dims preservation
- `test_return_data_flag_controls_original_yield` — tests return_data=True and return_data=False
- `test_n_surrogates_zero_behaviour` — tests edge case of zero surrogates
- `test_random_state_reproducibility` — verifies identical surrogates with same seed
- `test_multivariate_flag_changes_channel_relations` — excellent test for multivariate behavior
- `test_identical_channels_remain_identical_multivariate` — verifies channel identity preservation
- `test_fourier_transform_surrogates_metadata_preserved` — checks all metadata fields
- `test_fourier_transform_surrogates_does_not_mutate_input` — verifies input immutability
- `test_fourier_transform_surrogates_is_lazy_generator` — verifies generator behavior

## Fix

None — all existing tests are correct.

## Add

Missing test for non-integer n_surrogates validation:

### `test_fourier_transform_surrogates_invalid_n_surrogates_type`

```python
def test_fourier_transform_surrogates_invalid_n_surrogates_type() -> None:
    """Non-integer n_surrogates raises ValueError."""
    with pytest.raises(ValueError, match="integer"):
        FourierTransformSurrogates(n_surrogates=3.5, random_state=42)
```

## Action List

1. [Severity: HIGH] Add test for non-integer n_surrogates to reach 100% coverage (line 68)
