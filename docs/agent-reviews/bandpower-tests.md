# Test Review: bandpower

**Feature**: `src/cobrabox/features/bandpower.py`
**Test file**: `tests/test_feature_bandpower.py`
**Date**: 2025-03-04
**Verdict**: PASS

## Summary

Comprehensive test coverage for the Bandpower feature. Tests cover default bands, custom bands, mixed specifications, value correctness, error handling, and metadata preservation. The test file follows all naming conventions and includes clear docstrings. All required scenarios from the criteria are covered except for the optional "no mutation of input" test.

## Keep

Tests that are correct and complete — no changes needed:

- `test_bandpower_default_dims_and_shape` — verifies output shape (5 bands × 2 channels × 1 time) and dims
- `test_bandpower_default_band_index_coords` — verifies band_index coordinate matches default band names
- `test_bandpower_custom_range_shape` — verifies custom band specs produce correct shape
- `test_bandpower_mixed_spec_shape` — verifies mixed True + custom range specs work correctly
- `test_bandpower_alpha_dominates_for_10hz_sine` — validates that 10 Hz sine wave has highest power in alpha band
- `test_bandpower_true_alias_matches_explicit_range` — validates that `bands={'alpha': True}` equals explicit range
- `test_bandpower_all_positive_values` — verifies power values are non-negative
- `test_bandpower_empty_bands_equals_none` — validates empty dict and None produce identical results
- `test_bandpower_nperseg_changes_nothing_in_shape` — verifies nperseg only affects quality, not shape
- `test_bandpower_history_appended` — verifies 'Bandpower' is appended to history
- `test_bandpower_metadata_preserved` — verifies subjectID and sampling_rate are preserved
- `test_bandpower_raises_when_sampling_rate_missing` — validates ValueError when sampling_rate is None
- `test_bandpower_raises_for_true_with_unknown_band` — validates ValueError for unknown band name with True
- `test_bandpower_raises_when_nperseg_less_than_2` — validates ValueError for invalid nperseg
- `test_bandpower_raises_for_false_band_spec` — validates ValueError when band spec is False
- `test_bandpower_transposes_when_time_not_last` — verifies automatic transpose when time is not last dim
- `test_bandpower_zeros_when_no_freq_bins_in_band` — verifies zero power when band has no matching frequency bins

## Fix

None.

## Add

Optional enhancement:

### `test_bandpower_does_not_mutate_input`

```python
def test_bandpower_does_not_mutate_input() -> None:
    """Bandpower.apply() leaves the input Data object unchanged."""
    data = _sine_data(freq_hz=10.0)
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.Bandpower().apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
```

## Action List

1. [Severity: LOW] Optionally add `test_bandpower_does_not_mutate_input` to verify input Data is not mutated (nice-to-have, not required for 95% coverage)

None required for PASS.
