# Feature Review: mean_aggregate

**File**: `src/cobrabox/features/mean_aggregate.py`
**Date**: 2026-03-06
**Verdict**: PASS

## Summary

A clean, well-implemented `AggregatorFeature` that folds a stream of windowed data back into a single averaged result. The implementation correctly handles history propagation, metadata preservation, and empty stream validation. No issues found.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

- ✅ `from __future__ import annotations` present at line 1
- ✅ `@dataclass` decorator with correct `AggregatorFeature` base class (lines 12-13)
- ✅ No `output_type` needed — this is correct since `AggregatorFeature` returns `Data`
- ✅ Class name `MeanAggregate` matches filename (`mean_aggregate.py`)
- ✅ `__call__` signature matches `AggregatorFeature` contract (line 38)
- ✅ No redundant `apply()` override — correctly inherits from base
- ✅ Clean imports in proper order (stdlib → third-party → internal)

## Docstring

Comprehensive Google-style docstring with all required sections:

- ✅ One-line summary at line 14
- ✅ Extended description explaining stacking and reduction (lines 16-17)
- ✅ `Args:` section correctly documents no configuration parameters (lines 19-20)
- ✅ `Returns:` section details shape, metadata preservation, and history (lines 22-27)
- ✅ `Example:` section shows chord usage pattern (lines 29-35)

**Suggestion**: Add a `Raises:` section documenting the `ValueError` at lines 40-41.

## Typing

- ✅ No dataclass fields to type (aggregator takes no config)
- ✅ `__call__` has correct return type annotation `-> Data` (line 38)
- ✅ Parameter types are explicit: `data: Data`, `stream: Iterator[Data]`
- ✅ No bare `Any` types

## Safety & Style

- ✅ No `print()` statements
- ✅ Input validation raises `ValueError` with clear message for empty streams (lines 40-41)
- ✅ No mutation of input `data` — creates new `Data` instance (lines 46-54)
- ✅ History correctly propagated: original data history + per-window history + "MeanAggregate" (lines 45, 52)
- ✅ Metadata preserved: `subjectID`, `groupID`, `condition`, `sampling_rate`, `extra` (lines 48-51, 53)
- ✅ Uses `join="override"` to avoid coordinate conflicts during concat (line 42)

## Action List

1. [Severity: LOW] Add a `Raises:` section to the docstring documenting the `ValueError` raised when an empty stream is received (line 40-41).
