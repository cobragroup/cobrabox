# Test Review: bandfilter

**Feature**: `src/cobrabox/features/bandfilter.py`
**Test file**: `tests/test_feature_bandfilter.py`
**Date**: 2026-03-05
**Verdict**: PASS

## Summary

The BandFilter test suite is comprehensive and well-structured. It covers all required scenarios including happy path, history tracking, metadata preservation, input immutability, parameter validation, and numerical correctness. The tests use appropriate helper functions, include thorough docstrings, and verify both shape/coordinate correctness and actual filtering behavior against reference scipy implementations.

## Keep

All existing tests are correct and complete:

- `test_bandfilter_history_updated` — Verifies 'BandFilter' appended to history
- `test_bandfilter_metadata_preserved` — Checks all metadata fields preserved
- `test_bandfilter_returns_data_instance` — Confirms apply() returns Data
- `test_bandfilter_does_not_mutate_input` — Validates input object unchanged
- `test_bandfilter_missing_sampling_rate_raises` — Tests runtime guard for missing sampling_rate
- `test_bandfilter_default_band_coords` — Validates default EEG band coordinates
- `test_bandfilter_default_band_coords_keep_orig` — Tests keep_orig parameter
- `test_bandfilter_custom_bands` — Verifies custom band dictionaries work
- `test_bandfilter_single_band` — Tests single-band edge case
- `test_bandfilter_preserves_time_coords` — Validates time coordinate preservation
- `test_bandfilter_preserves_space_coords_when_present` — Validates space coordinate preservation
- `test_bandfilter_matches_manual_scipy_filter` — Numerical correctness against manual scipy call
- `test_bandfilter_output_is_not_all_zeros` — Sanity check that filtering produces output
- `test_bandfilter_sine_concentrated_in_correct_band` — Frequency-domain validation with sinusoids
- `test_bandfilter_output_is_valid_data_for_further_features` — Chaining/integration test
- `test_bandfilter_zero_order_raises` — Parameter validation (ord=0)
- `test_bandfilter_negative_order_raises` — Parameter validation (ord=-1)
- `test_bandfilter_empty_bands_raises` — Parameter validation (empty bands dict)
- `test_bandfilter_invalid_band_range_raises` — Parameter validation (low >= high)
- `test_bandfilter_negative_frequency_raises` — Parameter validation (negative frequency)

## Fix

None.

## Add

None.

## Action List

None.
