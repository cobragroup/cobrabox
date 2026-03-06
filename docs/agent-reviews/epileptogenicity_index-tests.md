# Test Review: epileptogenicity_index

**Feature**: `src/cobrabox/features/epileptogenicity_index.py`
**Test file**: `tests/test_feature_epileptogenicity_index.py`
**Date**: 2026-03-06
**Verdict**: PASS

## Coverage

```text
epileptogenicity_index.py: 100% (79 statements, 0 missing)
```

Per-file coverage requirement met.

## Summary

Comprehensive test suite with 19 passing tests covering all required scenarios. The tests use a sophisticated synthetic signal generator (`_gamma_onset_signal`) that produces theta-background with gamma bursts to trigger the Page-Hinkley detection algorithm. All edge cases are well-covered including: no-discharge channels, flat signals, multi-channel ordering, and parameter sensitivity. Error handling tests use mock `_FakeData` classes to bypass SignalData construction guards.

## Keep

Tests that are correct and complete:

- `test_epileptogenicity_index_dims` — verifies output has exactly (space,) dimensions
- `test_epileptogenicity_index_output_shape` — validates shape matches channel count
- `test_epileptogenicity_index_space_coords_preserved` — confirms coordinate labels survive
- `test_epileptogenicity_index_values_in_unit_interval` — asserts [0, 1] normalization
- `test_epileptogenicity_index_max_is_one` — confirms max value equals 1 after normalization
- `test_epileptogenicity_index_early_channel_scores_higher` — validates core EI logic (temporal ordering)
- `test_epileptogenicity_index_no_discharge_channel_near_zero` — tests Page-Hinkley "no alarm" path
- `test_epileptogenicity_index_flat_signal_all_zero` — tests degenerate input (constant signal)
- `test_epileptogenicity_index_three_channels_ordering` — validates monotonic EI ordering
- `test_epileptogenicity_index_history_appended` — confirms history tracking
- `test_epileptogenicity_index_metadata_preserved` — validates subjectID, groupID, condition preservation
- `test_epileptogenicity_index_window_duration_accepted_and_shape_unchanged` — parameter sensitivity test
- `test_epileptogenicity_index_very_high_threshold_suppresses_detection` — tests threshold=1e9 edge case
- `test_epileptogenicity_index_raises_without_time_dim` — validates dimension guard
- `test_epileptogenicity_index_raises_without_sampling_rate` — validates sampling_rate requirement
- `test_epileptogenicity_index_raises_with_extra_dims` — validates exact dimension requirement
- `test_epileptogenicity_index_raises_when_signal_shorter_than_window` — validates window size guard
- `test_epileptogenicity_index_returns_data_instance` — confirms output type
- `test_epileptogenicity_index_does_not_mutate_input` — validates immutability contract

## Fix

None. All tests pass and meet criteria.

## Add

None. All required scenarios are covered.

## Action List

None.
