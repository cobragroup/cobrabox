# Test Review: lempel_ziv

**Feature**: `src/cobrabox/features/lempel_ziv.py`
**Test file**: `tests/test_feature_lempel_ziv.py`
**Date**: 2025-03-05
**Verdict**: PASS

## Coverage

```text
lempel_ziv.py: 100% (47 statements, 0 missing)
```

## Summary

Excellent test coverage with 8 tests. Tests validate the Lempel-Ziv complexity computation, known values, comparison between random and periodic signals, Chord pipeline integration, and input validation. All required scenarios are covered.

## Keep

- `test_feature_lempel_ziv_output_type_and_history` — output structure and history
- `test_feature_lempel_ziv_known_value` — hand-verified expected values
- `test_feature_lempel_ziv_random_more_complex_than_periodic` — algorithmic correctness
- `test_feature_lempel_ziv_values_are_positive` — output value constraints
- `test_feature_lempel_ziv_via_chord` — Chord pipeline integration
- `test_feature_lempel_ziv_multichannel_independent` — per-channel independence
- `test_feature_lempel_ziv_missing_time_raises` — error handling
- `test_feature_lempel_ziv_does_not_mutate_input` — input immutability

## Fix

None. All tests are correct and complete.

## Add

None. All required scenarios covered.

## Action List

None.
