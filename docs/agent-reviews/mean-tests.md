# Test Review: mean

**Feature**: `src/cobrabox/features/mean.py`
**Test file**: `tests/test_feature_mean.py`
**Date**: 2026-03-06
**Verdict**: PASS

## Coverage

```text
mean.py: 100% (12 statements, 0 missing)
```

Full coverage achieved.

## Summary

The test file provides comprehensive coverage of the `Mean` feature with 5 test
functions covering all required scenarios. Tests correctly verify dimension
reduction, error handling, metadata preservation, and input immutability.

## Keep

All existing tests are correct and complete:

- `test_mean_reduces_extra_dimension` — Verifies mean reduces "run_index"
  dimension, correct output shape, history update, and metadata preservation.

- `test_mean_raises_for_unknown_dimension` — Confirms clear ValueError when
  specified dimension does not exist in data.

- `test_mean_single_channel_timeseries_returns_single_value` — Tests mean over
  "time" on single-channel signal returns scalar value.

- `test_mean_metadata_preserved` — Dedicated test for metadata preservation
  (subjectID, groupID, condition, sampling_rate).

- `test_mean_does_not_mutate_input` — Verifies input Data object is unchanged
  after `.apply()` call.

## Action List

None.
