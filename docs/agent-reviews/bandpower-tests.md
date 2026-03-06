# Test Review: bandpower

**Feature**: `src/cobrabox/features/bandpower.py`
**Test file**: `tests/test_feature_bandpower.py`
**Date**: 2025-03-05
**Verdict**: PASS

## Coverage

```text
bandpower.py: 100% (119 statements, 0 missing)
```

## Summary

Excellent test coverage with 18 tests. Tests validate Welch's method PSD integration, default and custom bands, True alias handling, nperseg parameter, metadata preservation, and comprehensive error handling (missing sampling_rate, unknown band names, invalid nperseg, False band spec).

## Keep

- `test_bandpower_default_dims_and_shape` — output structure
- `test_bandpower_default_band_index_coords` — default band names
- `test_bandpower_alpha_dominates_for_10hz_sine` — frequency selectivity
- `test_bandpower_true_alias_matches_explicit_range` — True alias handling
- `test_bandpower_nperseg_changes_nothing_in_shape` — nperseg parameter
- `test_bandpower_history_appended` — history tracking
- `test_bandpower_metadata_preserved` — metadata propagation (groupID, condition)
- `test_bandpower_raises_when_sampling_rate_missing` — required field check
- `test_bandpower_raises_for_true_with_unknown_band` — unknown band validation
- `test_bandpower_raises_when_nperseg_less_than_2` — nperseg validation
- `test_bandpower_raises_for_false_band_spec` — boolean spec validation
- `test_bandpower_transposes_when_time_not_last` — transpose handling
- `test_bandpower_zeros_when_no_freq_bins_in_band` — zero power for out-of-range bands
- `test_bandpower_does_not_mutate_input` — input immutability

## Fix

None. All tests are correct and comprehensive.

## Add

None. All required scenarios covered.

## Action List

None.
