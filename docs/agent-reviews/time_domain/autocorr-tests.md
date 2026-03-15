# Test Review: autocorr

**Feature**: `src/cobrabox/features/autocorr.py`
**Test file**: `tests/test_feature_autocorr.py`
**Date**: 2025-03-06
**Verdict**: PASS

## Coverage

```text
src/cobrabox/features/autocorr.py: 100% (46 statements, 0 missing)
```

Full coverage achieved. All lines in the feature file are exercised by the test suite.

## Summary

The test file for `Autocorr` is comprehensive and well-structured. It covers all
required scenarios including happy path, parameter validation, edge cases with
NaN and constant signals, metadata preservation, and immutability. The tests
use a helper function `_expected_autocorr` to validate numerical correctness
against a reference implementation. All 11 tests pass and meet project conventions.

## Keep

Tests that are correct and complete — no changes needed:

- `test_feature_autocorr_reduces_requested_dimension` — Comprehensive happy path test
  that verifies dimension reduction, shape, numerical correctness via helper, and
  history/metadata preservation.

- `test_feature_autocorr_default_5ms_matches_explicit_steps` — Validates default
  lag calculation matches explicit 5ms lag_steps.

- `test_feature_autocorr_raises_for_unknown_dimension` — Runtime validation for
  missing dimension.

- `test_feature_autocorr_raises_when_both_lag_inputs_provided` — `__post_init__`
  validation for mutually exclusive parameters.

- `test_feature_autocorr_raises_for_lag_out_of_range` — Runtime validation for
  invalid lag values.

- `test_feature_autocorr_constant_signal_returns_nan` — Edge case for zero-energy
  signal after demeaning.

- `test_feature_autocorr_all_nan_returns_nan` — Edge case for all-NaN input.

- `test_feature_autocorr_fs_non_positive_raises` — `__post_init__` validation for
  invalid sampling frequency.

- `test_feature_autocorr_metadata_preserved` — Verifies all metadata fields are
  preserved and sampling_rate becomes None.

- `test_feature_autocorr_does_not_mutate_input` — Immutability check for history,
  shape, values, and metadata.

- `test_feature_autocorr_lag_ms_parameter` — Validates lag_ms parameter correctly
  computes lag in milliseconds.

## Fix

None.

## Add

No additional tests required. All required scenarios from criteria.md are covered:

- Happy path with shape/dims assertions
- History updated (PascalCase class name)
- Metadata preserved (subjectID, groupID, condition, sampling_rate)
- Output type handling (sampling_rate becomes None when time dim removed)
- Invalid params (`__post_init__` and runtime validation)
- Output type is Data (isinstance checks)
- No mutation of input

## Action List

None.
