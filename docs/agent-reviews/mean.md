# Feature Review: mean

**File**: `src/cobrabox/features/mean.py`
**Date**: 2026-03-04
**Verdict**: PASS

## Summary

Clean, well-structured feature that computes the arithmetic mean across a specified dimension. Uses `BaseFeature[Data]` appropriately since it works with any dimension (not time-specific). Includes proper input validation, complete Google-style docstring with all required sections, and correct typing throughout.

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

1 file already formatted

## Signature & Structure

- ✅ `from __future__ import annotations` at line 1
- ✅ `@dataclass` decorator at line 11
- ✅ Inherits `BaseFeature[Data]` (line 12) — correct for dimension-agnostic feature
- ✅ Class name `Mean` matches filename `mean.py`
- ✅ `__call__` signature correct: `def __call__(self, data: Data) -> xr.DataArray:` (line 29)
- ✅ No `apply()` override — uses inherited method
- ✅ Clean imports: dataclass, xarray, internal modules only

## Docstring

Complete Google-style docstring with all required sections:

- ✅ One-line summary (line 13)
- ✅ `Args:` section documenting `dim` field (lines 15-16)
- ✅ `Returns:` section describing output shape and values (lines 18-21)
- ✅ `Example:` section with `.apply()` usage (lines 23-24)

## Typing

- ✅ Field `dim: str` properly typed (line 27)
- ✅ `__call__` return type `xr.DataArray` explicit (line 29)
- ✅ No bare `Any` usage

## Safety & Style

- ✅ No `print()` statements
- ✅ Input validation at lines 30-31: raises `ValueError` with clear message if dimension not found
- ✅ No mutation of input data — operates on `data.data` and returns new array
- ✅ Line 31 at 98 characters, within 100 limit

## Action List

None.
