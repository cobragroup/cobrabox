# Test Review: amplitude_variation

**Feature**: `src/cobrabox/features/amplitude_variation.py`
**Test file**: `tests/test_feature_amplitude_variation.py`
**Date**: 2025-03-05
**Verdict**: PASS

## Coverage

```text
amplitude_variation.py: 100% (13 statements, 0 missing)
```

## Summary

Good test coverage with 7 tests. Tests validate standard deviation computation, constant signal handling, single channel support, Chord integration, metadata preservation, and input immutability. All required scenarios are covered for this simple feature.

## Keep

- `test_feature_amp_var_expected_values_and_history` — correct values and history
- `test_feature_amp_var_constant_signal_is_zero` — edge case (zero variance)
- `test_feature_amp_var_single_channel` — single channel support
- `test_feature_amp_var_via_chord` — Chord pipeline integration
- `test_feature_amp_var_no_mutation` — input immutability
- `test_amp_var_metadata_preserved` — metadata propagation (groupID, condition)
- `test_amp_var_missing_time_raises` — error handling

## Fix

None. All tests are correct.

## Add

None. All required scenarios covered for a parameter-less feature.

## Action List

None.
