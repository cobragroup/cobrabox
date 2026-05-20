# Test Review: bandfilter

**Feature**: `src/cobrabox/features/frequency_domain/bandfilter.py`
**Test file**: `tests/features/frequency_domain/test_feature_bandfilter.py`
**Date**: 2026-03-24
**Verdict**: PASS

## Coverage

```text
Name                                        Stmts   Miss  Cover   Missing
src/cobrabox/features/bandfilter.py            36      0   100%
```

Coverage: **100%** - All statements covered.

## Summary

The `test_feature_bandfilter.py` file is comprehensive and well-structured. It contains 23 tests covering all required scenarios plus extensive numerical correctness verification. The test file properly tests the `BandFilter` feature which applies Butterworth bandpass filters and stacks results along a new `band` dimension.

## Keep

Tests that are correct and complete:

- `test_bandfilter_history_updated` — Correctly verifies history appends "BandFilter"
- `test_bandfilter_metadata_preserved` — Checks all metadata fields (subjectID, groupID, condition, sampling_rate)
- `test_bandfilter_returns_data_instance` — Verifies output type
- `test_bandfilter_does_not_mutate_input` — Validates input immutability with shape, history, and values
- `test_bandfilter_missing_sampling_rate_raises` — Tests runtime guard for missing sampling_rate
- `test_bandfilter_default_band_coords` — Verifies default EEG band names
- `test_bandfilter_default_band_coords_keep_orig` — Tests keep_orig parameter
- `test_bandfilter_custom_bands` — Tests custom band configuration
- `test_bandfilter_single_band` — Tests single band dimension handling
- `test_bandfilter_preserves_time_coords` — Validates coordinate preservation
- `test_bandfilter_preserves_space_coords_when_present` — Tests space coordinate handling
- `test_bandfilter_matches_manual_scipy_filter` — Numerical correctness against manual scipy implementation
- `test_bandfilter_output_is_not_all_zeros` — Sanity check for non-degenerate output
- `test_bandfilter_sine_concentrated_in_correct_band` — Parametric test (3 cases) verifying frequency selectivity
- `test_bandfilter_output_is_valid_data_for_further_features` — Tests chaining with other features
- `test_bandfilter_zero_order_raises` — Parameter validation (ord=0)
- `test_bandfilter_negative_order_raises` — Parameter validation (ord=-1)
- `test_bandfilter_empty_bands_raises` — Parameter validation (empty bands dict)
- `test_bandfilter_invalid_band_range_raises` — Parameter validation (low >= high)
- `test_bandfilter_negative_frequency_raises` — Parameter validation (negative frequencies)
- `test_bandfilter_band_wrong_number_of_frequencies_raises` — Parameter validation (wrong frequency count)

## Fix

None. All tests follow conventions and pass correctly.

## Add

No additional tests required. All scenarios are covered:

- Basic API and shape: 9 tests
- Coordinate preservation: 2 tests
- Numerical correctness: 3 tests (including parametrized)
- Chaining: 1 test
- Parameter validation: 6 tests

## Action List

None.
