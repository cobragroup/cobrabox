# Test Review: min

**Feature**: `src/cobrabox/features/min.py`
**Test file**: `tests/test_feature_min.py`
**Date**: 2026-03-06
**Verdict**: PASS

## Coverage

```text
Min: 100% (12 statements, 0 missing)
```

## Summary

The test file provides comprehensive coverage of the `Min` feature with 6 well-structured tests. All required scenarios are present, including happy path, error handling, metadata preservation, history tracking, and immutability verification. Coverage is 100%.

## Keep

Tests that are correct and complete:

- `test_min_reduces_extra_dimension` — Tests reducing a non-time dimension (run_index), verifies shape transformation, metadata preservation, and history update. Comprehensive happy path test.
- `test_min_raises_for_unknown_dimension` — Validates that an unknown dimension raises ValueError with a clear message. Good error case coverage.
- `test_min_single_channel_timeseries_returns_single_value` — Tests edge case of single-channel data with time reduction, verifies correct output shape and value.
- `test_min_finds_smallest_value_with_negative_numbers` — Tests correctness with negative values to ensure true minimum is found (not just smallest absolute value).
- `test_min_does_not_mutate_input` — Verifies immutability by checking history, shape, and values remain unchanged after apply().
- `test_min_preserves_metadata` — Explicitly tests all metadata fields (subjectID, groupID, condition, sampling_rate) are preserved.

## Action List

None.
