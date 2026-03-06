# Feature Review: max

**File**: `src/cobrabox/features/max.py`
**Date**: 2025-03-05
**Verdict**: PASS

## Summary

Clean, well-structured feature that computes the maximum value across a specified dimension. Follows all conventions with proper docstring, type annotations, input validation, and correctly sets `output_type` since this feature removes the reduced dimension.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

- ✅ `from __future__ import annotations` present (line 1)
- ✅ `@dataclass` decorator with `BaseFeature[Data]` inheritance (lines 12-13)
- ✅ Correct `output_type: ClassVar[type[Data]] = Data` set (line 28) — appropriate since the feature reduces over a dimension
- ✅ `__call__` signature correct: `def __call__(self, data: Data) -> xr.DataArray` (line 32)
- ✅ Class name `Max` matches filename `max.py`

## Docstring

Complete Google-style docstring with all required sections:

- ✅ One-line summary (line 14)
- ✅ `Args:` section documents the `dim` field (lines 16-17)
- ✅ `Returns:` section describes output shape and values (lines 19-22)
- ✅ `Example:` section shows `.apply()` usage (lines 24-25)

## Typing

- ✅ Field `dim: str` properly typed (line 30)
- ✅ `__call__` return type `xr.DataArray` correctly annotated (line 32)
- ✅ `ClassVar` type annotation for `output_type` (line 28)
- ✅ No bare `Any` types

## Safety & Style

- ✅ No `print()` statements
- ✅ Input validation checks if dimension exists (lines 33-34)
- ✅ No mutation of input `data` — operates on `data.data` and returns new array
- ✅ Line length within 100 characters

## Action List

None.
