# Test Review: bandpower

**Feature**: `src/cobrabox/features/bandpower.py`
**Test file**: `tests/test_feature_bandpower.py`
**Date**: 2026-03-06
**Verdict**: NEEDS WORK

## Coverage

```text
bandpower.py: 100% (45 statements, 0 missing)
```

## Summary

The Bandpower test suite is comprehensive with 18 tests covering happy path, error handling, parameter validation, and metadata preservation. Coverage is 100%. However, there are minor style issues: two tests are missing docstrings and return type annotations.

## Keep

Tests that are correct and complete:

- `test_bandpower_default_dims_and_shape` — verifies output dims and shape
- `test_bandpower_default_band_index_coords` — verifies band names in coords
- `test_bandpower_custom_range_shape` — verifies custom band specs
- `test_bandpower_mixed_spec_shape` — verifies mixed True + custom specs
- `test_bandpower_alpha_dominates_for_10hz_sine` — verifies spectral content detection
- `test_bandpower_true_alias_matches_explicit_range` — verifies True alias behavior
- `test_bandpower_all_positive_values` — verifies power is non-negative
- `test_bandpower_empty_bands_equals_none` — verifies bands={} == bands=None
- `test_bandpower_nperseg_changes_nothing_in_shape` — verifies nperseg doesn't affect shape
- `test_bandpower_history_appended` — verifies history is updated
- `test_bandpower_metadata_preserved` — verifies metadata preservation
- `test_bandpower_does_not_mutate_input` — verifies input is not mutated
- `test_bandpower_raises_when_sampling_rate_missing` — verifies ValueError for missing sampling_rate
- `test_bandpower_raises_for_true_with_unknown_band` — verifies ValueError for unknown band name
- `test_bandpower_transposes_when_time_not_last` — verifies automatic transposition
- `test_bandpower_zeros_when_no_freq_bins_in_band` — verifies zero handling for out-of-range bands

## Fix

Tests that exist but need minor changes:

### `test_bandpower_raises_when_nperseg_less_than_2`

Issue: Missing docstring and return type annotation.

```python
def test_bandpower_raises_when_nperseg_less_than_2() -> None:
    """Bandpower raises ValueError when nperseg is less than 2."""
    with pytest.raises(ValueError, match="nperseg must be >= 2"):
        cb.feature.Bandpower(nperseg=1)
```

### `test_bandpower_raises_for_false_band_spec`

Issue: Missing docstring and return type annotation.

```python
def test_bandpower_raises_for_false_band_spec() -> None:
    """Bandpower raises ValueError when band spec is False."""
    data = _sine_data(freq_hz=10.0)
    with pytest.raises(ValueError, match="must be True"):
        cb.feature.Bandpower(bands={"alpha": False}).apply(data)
```

## Add

No missing scenarios — all required test cases are covered.

## Action List

1. [Severity: LOW] Add docstring and `-> None` to `test_bandpower_raises_when_nperseg_less_than_2` (line 215)
2. [Severity: LOW] Add docstring and `-> None` to `test_bandpower_raises_for_false_band_spec` (line 220)
