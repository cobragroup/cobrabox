# Test Review: cordance

**Feature**: `src/cobrabox/features/frequency_domain/cordance.py`
**Test file**: `tests/features/frequency_domain/test_feature_cordance.py`
**Date**: 2025-03-24
**Verdict**: PASS

## Coverage

```text
Cordance: 100% (58 statements, 0 missing)
```

## Summary

Excellent, comprehensive test suite covering all required scenarios plus extensive semantic validation. The tests verify the Leuchter 1994 cordance algorithm implementation with multiple data scenarios, parameter combinations, and edge cases. All 31 tests pass with 100% coverage.

## Keep

All tests are well-written and should be preserved:

- `test_cordance_default_dims_and_shape` — correct shape assertion (5 bands x 4 channels)
- `test_cordance_default_band_coords` — verifies band_index coordinate matches expected names
- `test_cordance_custom_bands_shape` — single custom band produces correct shape
- `test_cordance_mixed_spec_shape` — mixed True + explicit range works correctly
- `test_cordance_values_are_finite` — sanity check for well-formed input
- `test_cordance_values_bounded` — verifies mathematical bounds [-1, 1]
- `test_cordance_concordance_positive_discordance_negative` — key semantic property
- `test_cordance_channel_with_dominant_band_highest_relative_power` — semantic correctness
- `test_cordance_agrees_with_manual_calculation` — validates against manual numpy implementation
- `test_cordance_output_concordance_only` — output='concordance' mode works
- `test_cordance_output_discordance_only` — output='discordance' mode works
- `test_cordance_combined_equals_concordance_minus_discordance` — invariant: cordance = concordance - discordance
- `test_cordance_threshold_changes_classification` — threshold parameter affects results
- `test_cordance_threshold_validation` — ValueError for invalid thresholds
- `test_cordance_nperseg_changes_nothing_in_shape` — nperseg doesn't affect output shape
- `test_cordance_history_appended` — 'Cordance' appears in history
- `test_cordance_metadata_preserved` — subjectID, groupID, condition preserved; sampling_rate=None
- `test_cordance_in_pipeline` — works in pipeline composition
- `test_cordance_identical_channels_all_concordant` — edge case: identical channels
- `test_cordance_empty_bands_equals_none` — bands={} and bands=None are equivalent
- `test_cordance_raises_when_no_space_dim` — ValueError when 'space' dim missing
- `test_cordance_raises_when_single_channel` — ValueError when only 1 channel
- `test_cordance_raises_for_true_with_unknown_band` — ValueError for unknown band name
- `test_cordance_raises_when_nperseg_less_than_2` — ValueError for invalid nperseg
- `test_cordance_raises_on_zero_signal` — ValueError for all-zero input
- `test_cordance_nan_on_zero_outputs_nan_for_silent_channels` — nan_on_zero=True works
- `test_cordance_raises_for_false_band_spec` — ValueError for bands={name: False}
- `test_cordance_true_alias_matches_explicit_range` — True and explicit range are equivalent
- `test_cordance_invalid_output_parameter` — ValueError for invalid output parameter
- `test_cordance_does_not_mutate_input` — input Data is unchanged after apply()
- `test_cordance_returns_data_instance` — returns Data instance (not raw array)

## Fix

None. All tests are correct and complete.

## Add

No missing scenarios. The test suite exceeds requirements with:

- Multiple helper functions for different data patterns (`_sine_data`, `_varied_data`, `_varied_amplitude_data`)
- Manual calculation verification test
- Output mode invariant test (combined = concordance - discordance)
- Threshold sensitivity test
- Edge cases (identical channels, empty bands, silent channels)

## Action List

None.
