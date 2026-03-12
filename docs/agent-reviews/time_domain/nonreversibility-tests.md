# Test Review: nonreversibility

**Feature**: `src/cobrabox/features/nonreversibility.py`
**Test file**: `tests/test_feature_nonreversibility.py`
**Date**: 2026-03-11
**Verdict**: PASS

## Coverage

```text
src/cobrabox/features/nonreversibility.py           42      0   100%
```

Excellent coverage — all lines are exercised by the test suite.

## Summary

The test file has been updated to cover all previously uncovered code paths and now meets the 95% project threshold (100% coverage). The missing‑space‑dimension guard is now exercised via `test_nonreversibility_raises_on_missing_space_dim`, the spectral‑radius rescaling branch is covered by `test_nonreversibility_spectral_radius_rescaling`, and the zero‑denominator fallback is covered by `test_nonreversibility_zero_denominator_returns_zero`. The test suite also includes the required no‑mutation test and a public‑API verification test. All tests follow naming conventions, carry `-> None` return annotations, and have appropriate docstrings.

## Keep

All tests are correct and complete — no changes needed.

- `test_nonreversibility_output_shape_dims_and_metadata` — thorough combined test covering shape, dims, all four metadata fields, history, and `sampling_rate is None`.
- `test_nonreversibility_scalar_is_nonnegative` — asserts `[0, 1)` bounds with a different seed.
- `test_nonreversibility_raises_on_single_channel` — matches error message precisely.
- `test_nonreversibility_raises_on_too_few_timepoints` — covers the `_compute_dc_norm` guard.
- `test_nonreversibility_raises_on_missing_space_dim` — exercises the missing‑`space`‑dimension guard.
- `test_nonreversibility_spectral_radius_rescaling` — forces the spectral‑radius rescaling branch.
- `test_nonreversibility_zero_denominator_returns_zero` — covers the zero‑denominator fallback.
- `test_nonreversibility_does_not_mutate_input` — verifies input immutability.
- `test_nonreversibility_public_api` — confirms auto‑discovery registration via `cb.feature.Nonreversibility`.

## Fix

None.

## Add

None — all required scenarios are covered.

## Action List

None.