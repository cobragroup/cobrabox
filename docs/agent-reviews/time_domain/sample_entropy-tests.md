# Test Review: sample_entropy

**Feature**: `src/cobrabox/features/sample_entropy.py`
**Test file**: `tests/test_feature_sample_entropy.py`
**Date**: 2026-03-06
**Verdict**: PASS

## Coverage

```text
sample_entropy.py: 100% (42 statements, 0 missing)
```

## Summary

The test suite for `SampleEntropy` is comprehensive and well-structured. All 13 tests pass and achieve 100% line coverage. The tests cover the happy path with known-value validation against a reference implementation, multiple logarithm bases (2, e, 10), multi-dimensional data, parameter validation, edge cases (no matches returns NaN), and all required infrastructure tests (history, metadata, immutability).

A helper function `_naive_sampen` provides a clean reference implementation for validating correctness. The tests use appropriate tolerances and seeded data for reproducibility.

## Keep

Tests that are correct and complete:

- `test_sampen_known_value` — Validates against reference implementation with log_base=2 (default)
- `test_sampen_natural_log` — Validates with natural logarithm (log_base=np.e)
- `test_sampen_base_10` — Validates with base-10 logarithm
- `test_sampen_raises_on_short_series` — Runtime guard for series length <= m
- `test_sampen_raises_on_invalid_log_base` — __post_init__ validation for log_base (0, 1, negative)
- `test_sampen_multi_dim_preserves_other_dims` — Correctly tests multi-channel data with space dimension
- `test_sampen_invalid_m` — __post_init__ validation for m < 1
- `test_sampen_no_matches_returns_nan` — Edge case: returns NaN when no template matches
- `test_sampen_history_updated` — History contains "SampleEntropy"
- `test_sampen_metadata_preserved` — subjectID, groupID, condition preserved; sampling_rate becomes None
- `test_sampen_does_not_mutate_input` — Input Data unchanged after apply()
- `test_sampen_returns_data_instance` — Returns cb.Data instance
- `test_sampen_default_r_uses_std` — Validates default tolerance (0.2 * std)

## Fix

Minor naming convention issues:

### Function naming

Issue: Test functions use `test_sampen_*` prefix instead of `test_sample_entropy_*` per convention.

```python
# Current
def test_sampen_known_value() -> None:

def test_sampen_history_updated() -> None:

# Recommended
def test_sample_entropy_known_value() -> None:

def test_sample_entropy_history_updated() -> None:
```

## Add

No additional tests required. All required scenarios are covered:

- Happy path (known values)
- History updated
- Metadata preserved
- Output type handling (sampling_rate=None when time removed)
- Invalid params (m < 1, invalid log_base, short series)
- Output type is Data
- No mutation of input
- Edge case (no matches → NaN)
- Multiple log bases (2, e, 10)
- Multi-dimensional data
- Default parameter (r=None uses 0.2*std)

## Action List

1. [Severity: LOW] Rename test functions from `test_sampen_*` to `test_sample_entropy_*` for consistency with project naming conventions (tests/test_feature_sample_entropy.py)
