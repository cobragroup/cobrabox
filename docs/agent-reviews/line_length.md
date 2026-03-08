# Feature Review: line_length

**File**: `src/cobrabox/features/line_length.py`
**Date**: 2026-03-05
**Verdict**: PASS

## Summary

The `LineLength` feature is exemplary and serves as the positive reference in the codebase. It correctly inherits from `BaseFeature[SignalData]`, has complete type annotations, a thorough Google-style docstring with all required sections (including Args with "None" since there are no dataclass fields), and follows all safety conventions. Ruff passes cleanly. No issues found.

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

1 file already formatted

## Signature & Structure

Line 1: `from __future__ import annotations` present and correctly placed.

Line 12-13: Correctly decorated with `@dataclass` and inherits `BaseFeature[SignalData]` (appropriate for a time-series feature).

Line 32: Correctly sets `output_type: ClassVar[type[Data]] = Data` since this feature removes the time dimension.

Line 34: `__call__` signature is correct: `def __call__(self, data: SignalData) -> xr.DataArray`.

No `apply()` method is implemented (correctly inherited from `BaseFeature`).

Imports are in the correct order and only import what is used.

## Docstring

Lines 14-30: Complete Google-style docstring with all required sections:

- **One-line summary** (line 14): "Compute line length over the time dimension."
- **Extended description** (lines 16-17): Explains what line length measures.
- **Args** (lines 19-20): Present with "None" since there are no dataclass fields.
- **Returns** (lines 22-26): Thorough description of output shape and dimensions.
- **Example** (lines 28-29): Working snippet showing typical usage.

## Typing

All types are correctly specified:

- `output_type: ClassVar[type[Data]] = Data` (line 32)
- `def __call__(self, data: SignalData) -> xr.DataArray` (line 34)

No bare `Any` types are used.

## Safety & Style

No `print()` statements found.

No input validation required—`SignalData` already enforces the `time` dimension at construction time.

No mutation of input `data`—the feature correctly works on `data.data` and returns a new array.

Line length is within limits (enforced by ruff).

## Action List

None.
