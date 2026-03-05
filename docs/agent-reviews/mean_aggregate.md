# Feature Review: mean_aggregate

**File**: `src/cobrabox/features/mean_aggregate.py`
**Date**: 2026-03-04
**Verdict**: PASS

## Summary

A well-structured `AggregatorFeature` that folds a stream of windowed Data objects back into a single Data by averaging. The implementation correctly propagates history from the pipeline, validates the input stream is non-empty, and preserves all metadata including sampling_rate. Ruff is clean and all criteria are satisfied.

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

1 file already formatted

## Signature & Structure

Clean and compliant. Uses `@dataclass` decorator (line 12) and inherits `AggregatorFeature` (line 13). The `__call__` signature matches the base class contract: `def __call__(self, data: Data, stream: Iterator[Data]) -> Data` (line 35). Class name `MeanAggregate` matches the filename `mean_aggregate.py`. Imports are in correct order with `from __future__ import annotations` first.

## Docstring

Complete Google-style docstring with all required sections:

- One-line summary: "Aggregate a stream of per-window Data by averaging across windows." (line 14)
- Extended description explaining the stacking and reduction approach (lines 16-17)
- Returns section describing shape, metadata preservation, and history propagation (lines 19-24)
- Example section showing Chord pipeline usage (lines 26-32)

No Args section is needed since there are no dataclass fields.

## Typing

All type annotations are correct:

- `data: Data` parameter typed (line 35)
- `stream: Iterator[Data]` parameter typed (line 35)
- Return type `-> Data` explicit (line 35)
- No bare `Any` types

## Safety & Style

- No `print()` statements
- Input validation: raises `ValueError` with clear message if stream is empty (lines 37-38)
- No mutation of input `data`: creates new `Data` object with merged history (lines 43-51)
- History propagation correctly combines original data history, per-window pipeline history, and the aggregator itself (lines 42, 49)
- `sampling_rate` is correctly preserved in the Data constructor (line 48)
- Uses `xr.concat` with `join="override"` to handle potential coordinate conflicts during stacking (line 39)

## Action List

None.
