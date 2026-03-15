# Feature Review: max

**File**: `src/cobrabox/features/max.py`
**Date**: 2025-03-06
**Verdict**: NEEDS WORK

## Summary

The `Max` feature is well-structured and follows the dataclass pattern correctly. It properly reduces a dimension using xarray's max operation and validates the input dimension exists. The implementation is clean with no ruff errors. The only issue is a missing `Raises:` section in the docstring to document the `ValueError` that is raised when the dimension is not found.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

All structural requirements met:

- Line 1: `from __future__ import annotations` present
- Line 12: `@dataclass` decorator applied
- Line 13: Inherits `BaseFeature[Data]` (appropriate for generic dimension reduction)
- Line 28: `output_type: ClassVar[type[Data]] = Data` correctly set since this removes a dimension
- Line 30: Field `dim: str` properly typed
- Line 32: `__call__` signature correct: `def __call__(self, data: Data) -> xr.DataArray:`
- No `apply()` override (correctly inherited)
- Imports ordered correctly (stdlib, third-party, internal)

## Docstring

Google-style docstring with most sections present:

- ✅ One-line summary (line 14)
- ✅ `Args:` section documents `dim` parameter (lines 16-17)
- ✅ `Returns:` section describes output shape and values (lines 19-22)
- ✅ `Example:` section shows usage (lines 24-25)
- ❌ Missing `Raises:` section — the feature raises `ValueError` on line 34 when the dimension is not found

## Typing

Fully typed:

- Line 30: `dim: str` has type annotation
- Line 32: Return type `xr.DataArray` declared
- Line 28: `output_type` classvar properly typed with `ClassVar[type[Data]]`
- No bare `Any` types

## Safety & Style

- ✅ No `print()` statements
- ✅ Input validation present on lines 33-34: checks if `dim` exists in data dimensions and raises clear `ValueError`
- ✅ No mutation of input `data` — operates on `data.data` and returns new array
- Line 35 within 100 character limit

## Action List

1. [Severity: MEDIUM] Add a `Raises:` section to the docstring documenting the `ValueError` raised when the specified dimension is not found in the data. Insert after the `Returns:` section around line 22.
