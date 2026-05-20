# Feature Review: MeanAggregate

**File**: `src/cobrabox/features/windowing/mean_aggregate.py`
**Date**: 2026-03-24
**Verdict**: PASS

## Summary

MeanAggregate is a well-written aggregator feature that correctly implements the `AggregatorFeature` interface. It averages values across a stream of windowed Data objects, properly propagates per-window history, and maintains immutability by creating new Data instances. The code follows all project conventions and includes appropriate validation.

## Ruff

### `uvx ruff check`
Clean — no issues found.

### `uvx ruff format --check`
Clean — no formatting issues.

## Signature & Structure

**Line 1**: Correctly imports `from __future__ import annotations` as the first import.

**Line 12**: Uses `@dataclass` decorator appropriately.

**Line 13**: Correctly inherits from `AggregatorFeature` (no type parameter needed for aggregators).

**Line 38**: `__call__` signature correctly implements the `AggregatorFeature` contract: `def __call__(self, data: Data, stream: Iterator[Data]) -> Data`. The parameter name `stream` is appropriate and matches the intent.

**No issues**: Does not implement `apply()` (correctly inherited), has no loose helper functions, imports are well-ordered (stdlib, third-party, internal relative imports).

## Docstring

Complete Google-style docstring with all required sections:

- **One-line summary** (line 14): Clear verb phrase describing the operation.
- **Extended description** (lines 16-17): Explains the implementation approach (stacking and reducing).
- **Args** (lines 19-20): Correctly documents that this aggregator takes no configuration parameters.
- **Returns** (lines 22-27): Comprehensive description including shape, metadata preservation, and history handling.
- **Example** (lines 29-35): Shows typical usage within a `Chord` pipeline.

No `Raises` section in docstring, though the code does raise `ValueError` for empty streams. This is a minor omission but acceptable given the clear inline comment on line 41.

No `References` section needed — this is a self-evident statistical operation without a specific literature basis.

## Typing

All type annotations are present and correct:

- **Line 38**: `data: Data` parameter typed correctly.
- **Line 38**: `stream: Iterator[Data]` parameter typed correctly.
- **Line 38**: Return type `-> Data` matches the `AggregatorFeature` contract.

No bare `Any` types. No `Literal` needed (no string option fields).

## Safety & Style

- **No print statements**: Clean.
- **Input validation** (line 40-41): Properly validates that the stream is not empty and raises a clear `ValueError` with message "MeanAggregate received an empty stream".
- **No mutation**: The feature correctly creates a new `Data` instance (lines 46-54) rather than modifying input objects. All metadata from the original `data` argument is preserved.
- **History handling** (lines 45, 52): Correctly propagates per-window pipeline history and appends "MeanAggregate" to the result history.

## Action List

None.
