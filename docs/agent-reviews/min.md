# Feature Review: min

**File**: `src/cobrabox/features/min.py`
**Date**: 2026-03-05
**Verdict**: PASS

## Summary

A clean, well-structured generic feature that computes the minimum value across a specified dimension. The implementation follows all cobrabox conventions: proper `@dataclass` + `BaseFeature[Data]` inheritance, complete Google-style docstring with all required sections, typed fields and return types, input validation for the dimension parameter, and no mutation of input data. Ruff checks pass cleanly.

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

1 file already formatted

## Signature & Structure

Line 11: Correct `@dataclass` decorator applied.
Line 12: Properly inherits `BaseFeature[Data]` — appropriate choice since this is a dimension-agnostic reduction feature (works on any dimension, not just time).
Line 12: Class name `Min` matches filename `min.py` in PascalCase.
Line 29: `__call__` signature correctly takes `data: Data` and returns `xr.DataArray`.
Lines 1-8: Import order follows convention (`__future__`, stdlib, third-party, internal).
No `apply()` override — correctly inherits from `BaseFeature`.

## Docstring

Lines 13-25: Complete Google-style docstring with all required sections:

- One-line summary (line 13) clearly states purpose
- `Args:` section (lines 15-16) documents the `dim` field with type and description
- `Returns:` section (lines 18-21) describes shape change and value semantics
- `Example:` section (lines 23-24) shows correct `.apply()` usage pattern

## Typing

Line 27: Field `dim: str` has explicit type annotation.
Line 29: `__call__` has full type signature `def __call__(self, data: Data) -> xr.DataArray`.
No bare `Any` types present.

## Safety & Style

Line 30-31: Input validation checks that `dim` exists in data dimensions and raises `ValueError` with a clear message if not.
Line 32: Returns result of `data.data.min()` without mutating input — follows immutability convention.
No `print()` statements present.
Line 31: Line length is 100 characters, within the limit.

## Action List

None.
