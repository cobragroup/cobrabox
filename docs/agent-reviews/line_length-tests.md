# Test Review: line_length

**Feature**: `src/cobrabox/features/line_length.py`
**Test file**: `tests/test_feature_line_length.py`
**Date**: 2025-03-05
**Verdict**: PASS

## Coverage

```
line_length.py: 100% (37 statements, 0 missing)
```

## Summary

Good test coverage with 6 tests. Tests validate line length computation (sum of absolute differences), correct expected values, single channel support, Chord pipeline integration, error handling (missing time dim), and input immutability. All required scenarios covered for this simple, parameter-less feature.

## Keep

- `test_line_length_expected_values_and_history` — correct values and history
- `test_line_length_single_channel_timeseries` — single channel support
- `test_line_length_via_chord` — Chord pipeline integration
- `test_line_length_missing_time_raises` — dimension validation
- `test_line_length_does_not_mutate_input` — input immutability
- `test_line_length_returns_data_instance` — return type

## Fix

None. All tests are correct.

## Add

None. All required scenarios covered for a parameter-less feature.

## Action List

None.
