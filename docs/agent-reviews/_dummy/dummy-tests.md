# Test Review: dummy

**Feature**: `src/cobrabox/features/dummy.py`
**Test file**: `tests/test_feature_dummy.py`
**Date**: 2025-03-05
**Verdict**: PASS

## Coverage

```text
dummy.py: 100% (19 statements, 0 missing)
```

## Summary

Good test coverage for Dummy feature. The tests appropriately cover this negative reference feature with tests for basic functionality, missing dimension errors, and input immutability. Since Dummy is a negative reference (as noted in its docstring), the test coverage is appropriate for its purpose.

## Keep

- `test_dummy_preserves_data_and_metadata` — Tests data values preserved and sampling_rate propagated (note: other metadata is intentionally not preserved as this is a negative reference)
- `test_dummy_missing_time_raises` — Tests ValueError when time dimension missing
- `test_dummy_missing_space_raises` — Tests ValueError when space dimension missing
- `test_dummy_does_not_mutate_input` — Tests input Data is unchanged
- `test_dummy_optional_arg` — Tests optional_arg parameter accepted

## Action List

None.
