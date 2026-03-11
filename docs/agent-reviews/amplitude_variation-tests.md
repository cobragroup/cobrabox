# Test Review: amplitude_variation

**Feature**: `src/cobrabox/features/amplitude_variation.py`
**Test file**: `tests/test_feature_amplitude_variation.py`
**Date**: 2026-03-06
**Verdict**: NEEDS WORK

## Coverage

```text
amplitude_variation.py: 100% (11 statements, 0 missing)
```

## Summary

Test coverage is excellent at 100%, and the test file covers most required scenarios. However, there is one HIGH severity issue: `test_amp_var_missing_time_raises` uses an internal `__new__` bypass that violates the criteria for `BaseFeature[SignalData]` features. According to the criteria, features inheriting from `BaseFeature[SignalData]` should skip the "missing time" test entirely because `SignalData` enforces the time dimension at construction time. Additionally, some test names don't follow the `test_feature_<name>_<scenario>` convention consistently.

## Keep

Tests that are correct and complete:

- `test_feature_amp_var_expected_values_and_history` — Comprehensive happy path with value verification, history check, output type, and sampling_rate=None
- `test_feature_amp_var_constant_signal_is_zero` — Edge case for constant input
- `test_feature_amp_var_single_channel` — Edge case for single-channel data
- `test_feature_amp_var_via_chord` — Integration test with Chord composition
- `test_feature_amp_var_no_mutation` — Input immutability check
- `test_amp_var_metadata_preserved` — Metadata preservation (though name should be fixed)

## Fix

### `test_amp_var_missing_time_raises`

**Issue**: This test uses `cb.Data.__new__(cb.Data)` and `object.__setattr__` to bypass validation, which is explicitly discouraged in the criteria. For `BaseFeature[SignalData]` features, this scenario should be skipped entirely.

**Action**: Remove this test entirely. `AmplitudeVariation` inherits from `BaseFeature[SignalData]`, and `SignalData` enforces the time dimension at construction time, making this test invalid per criteria.

### Test naming convention

**Issue**: Two tests don't follow the `test_feature_<name>_<scenario>` pattern:

- `test_amp_var_metadata_preserved` → should be `test_feature_amp_var_metadata_preserved`
- `test_amp_var_missing_time_raises` → should be removed (see above)

**Action**: Rename `test_amp_var_metadata_preserved` to `test_feature_amp_var_metadata_preserved`.

## Add

No additional tests required. All required scenarios are covered except the invalid "missing time" test which should be removed.

## Action List

1. [Severity: HIGH] Remove `test_amp_var_missing_time_raises` — this test uses discouraged internal bypass and is invalid for `BaseFeature[SignalData]` features (line 106-118)
2. [Severity: MEDIUM] Rename `test_amp_var_metadata_preserved` to `test_feature_amp_var_metadata_preserved` for naming consistency (line 88)
