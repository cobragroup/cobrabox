# Feature Review: Mean

**File**: `src/cobrabox/features/reductions/mean.py`
**Date**: 2025-03-24
**Verdict**: PASS

## Summary

A clean, well-structured reduction feature that computes the mean across a specified dimension. The implementation follows all cobrabox conventions: proper dataclass structure, complete Google-style docstring with all required sections, input validation for the dimension parameter, and correct typing throughout. The feature correctly inherits from `BaseFeature[Data]` (not `SignalData`) since it operates generically on any dimension, making it suitable for non-time-series reductions as well.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

- **Line 1**: `from __future__ import annotations` present.
- **Line 11**: `@dataclass` decorator correctly applied.
- **Line 12**: Inherits `BaseFeature[Data]` — appropriate for a dimension-agnostic reduction feature (does not require time dimension).
- **Line 12**: Class name `Mean` matches filename `mean.py` in PascalCase.
- **Line 30**: Single dataclass field `dim: str` properly typed.
- **Line 32**: `__call__` signature correct: `def __call__(self, data: Data) -> xr.DataArray`.
- No `apply()` override — correctly inherits from base class.
- No loose helper functions — the implementation is self-contained.
- Import order follows convention: `__future__`, stdlib, third-party, internal.

## Docstring

Complete Google-style docstring with all required sections:

- **One-line summary** (line 13): Clear verb phrase describing the operation.
- **Args** (lines 15-16): Documents the `dim` field with type and description.
- **Returns** (lines 18-21): Describes output shape (input shape minus reduced dimension) and value semantics.
- **Raises** (lines 23-24): Documents `ValueError` for missing dimension.
- **Example** (lines 26-27): Shows typical usage with `.apply()`.

The docstring is accurate and follows the project's style conventions (using double backticks for code literals).

## Typing

- **Line 30**: Field `dim: str` has explicit type annotation.
- **Line 32**: `__call__` has complete type annotations for parameter and return type.
- No bare `Any` types.
- Type parameter `BaseFeature[Data]` correctly constrains the input type.

## Safety & Style

- No `print()` statements.
- **Lines 33-34**: Proper input validation — checks if `self.dim` exists in `data.data.dims` and raises `ValueError` with a clear message including available dimensions.
- No mutation of input `data` — operates on `data.data` and returns new array.
- Line length within 100 characters (longest line is 90 chars).

## Action List

None.
