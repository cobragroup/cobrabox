# Feature Review: mean_aggregate

**File**: `src/cobrabox/features/mean_aggregate.py`
**Date**: 2025-03-05
**Verdict**: PASS

## Summary

Well-implemented `AggregatorFeature` that averages across windows in a stream. Properly handles the responsibility of building history manually (required for `AggregatorFeature` subclasses). Clean implementation with good documentation and validation.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

- ✅ `from __future__ import annotations` present (line 1)
- ✅ `@dataclass` decorator with `AggregatorFeature` inheritance (lines 12-13)
- ✅ No `output_type` needed (AggregatorFeature returns Data by contract)
- ✅ `__call__` signature correct for AggregatorFeature: `def __call__(self, data: Data, stream: Iterator[Data]) -> Data` (line 38)
- ✅ Class name `MeanAggregate` matches filename `mean_aggregate.py`

## Docstring

Complete Google-style docstring with all required sections:

- ✅ One-line summary (line 14)
- ✅ Extended description explains window stacking behavior (lines 16-17)
- ✅ `Args:` section correctly notes no configuration parameters (lines 19-20)
- ✅ `Returns:` section describes output shape and metadata preservation (lines 22-27)
- ✅ `Example:` section shows Chord pipeline usage (lines 29-35)

## Typing

- ✅ `Iterator` imported from `collections.abc` (line 3)
- ✅ `__call__` return type `Data` correctly annotated (line 38)
- ✅ No bare `Any` types
- ✅ Parameter types for `data` and `stream` correctly specified

## Safety & Style

- ✅ No `print()` statements
- ✅ Input validation for empty stream (lines 40-41)
- ✅ Manually builds history (lines 45, 52) — correct for AggregatorFeature
- ✅ No mutation of input `data` — creates new `Data` object (lines 46-54)
- ✅ Preserves all metadata from original data (subjectID, groupID, condition, sampling_rate, extra)
- ✅ Line length within 100 characters

## Action List

None.
