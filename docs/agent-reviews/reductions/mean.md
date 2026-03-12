# Feature Review: mean

**File**: `src/cobrabox/features/mean.py`
**Date**: 2026-03-06
**Verdict**: NEEDS WORK

## Summary

Clean, minimal implementation of a generic mean-reduction feature. The code is well-structured, properly typed, and passes all ruff checks. Only issue is a missing `Raises:` section in the docstring for the `ValueError` raised at line 30.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

All structural requirements met:

- Line 1: `from __future__ import annotations` present
- Line 11: `@dataclass` decorator applied
- Line 12: Correctly inherits `BaseFeature[Data]` (generic, works with any dimension)
- Line 27: Single field `dim: str` properly typed
- Line 29: `__call__` signature correct: `def __call__(self, data: Data) -> xr.DataArray`
- No custom `apply()` method (correctly uses inherited implementation)
- Class name `Mean` matches filename `mean.py`
- Import order follows convention

## Docstring

Good Google-style docstring with most required sections:

- ✅ One-line summary at line 13
- ✅ `Args:` section documents the `dim` field
- ✅ `Returns:` section describes output shape and values
- ✅ `Example:` section shows usage with `.apply()`
- ❌ Missing `Raises:` section (feature raises `ValueError` at line 30-31)

The `ValueError` raised when the specified dimension is not found should be documented.

## Typing

Fully typed:

- Line 27: `dim: str` field annotation
- Line 29: `data: Data` parameter and `-> xr.DataArray` return type
- No bare `Any` types

## Safety & Style

- ✅ No `print()` statements
- ✅ Input validation at lines 30-31: checks if `dim` exists in data dimensions
- ✅ No mutation of input data (operates on `data.data.mean()`)
- ✅ Line length within 100 characters

## Action List

1. [Severity: MEDIUM] Add `Raises:` section to docstring documenting `ValueError` when `dim` is not found in data dimensions (line 30).
