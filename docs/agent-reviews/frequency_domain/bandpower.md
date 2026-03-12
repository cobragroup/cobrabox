# Feature Review: bandpower

**File**: `src/cobrabox/features/bandpower.py`
**Date**: 2026-03-06
**Verdict**: NEEDS WORK

## Summary

A well-structured feature implementing Welch's method for band power computation. Clean code with proper validation, good docstring coverage, and correct typing. The main gap is a missing `Raises:` section that documents the three `ValueError` conditions in the implementation.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

Excellent. The feature correctly:

- Uses `from __future__ import annotations` (line 1)
- Decorated with `@dataclass` and inherits `BaseFeature[SignalData]` (lines 21-22)
- Correctly omits `output_type` (preserves container type)
- Class name `Bandpower` matches filename `bandpower.py`
- `__call__` signature is correct: `def __call__(self, data: SignalData) -> xr.DataArray` (line 65)
- Does not implement `apply()` — uses inherited method
- Imports are clean and properly ordered (stdlib, third-party, internal)

## Docstring

Comprehensive Google-style docstring with all required sections except one:

- ✅ One-line summary (line 23)
- ✅ Extended description explaining algorithm (lines 25-28)
- ✅ `Args:` section documenting both fields (`bands`, `nperseg`)
- ✅ `Returns:` section with shape, dimensions, and units
- ✅ `Example:` section with three usage examples
- ❌ Missing `Raises:` section

The implementation raises `ValueError` in three places that should be documented:

1. `__post_init__` (lines 62-63): when `nperseg < 2`
2. `__call__` (lines 68-72): when `sampling_rate` is `None`
3. Band resolution logic (lines 82-92): unknown band names or `False` values

## Typing

All type annotations are correct:

- Fields properly typed: `bands: dict[str, list[float] | bool] | None` and `nperseg: int | None` (lines 58-59)
- `__call__` return type: `xr.DataArray` (line 65)
- `__post_init__` return type: `None` (line 61)
- No bare `Any` types

## Safety & Style

Excellent:

- No `print()` statements
- Proper input validation in `__post_init__` for `nperseg`
- Proper validation in `__call__` for missing `sampling_rate`
- No mutation of input `data` — works on copy and returns new DataArray
- Clean use of numpy and xarray operations

## Action List

1. [Severity: MEDIUM] Add `Raises:` section to docstring documenting the three `ValueError` conditions:
   - When `sampling_rate` is `None` (lines 68-72)
   - When `nperseg < 2` (lines 62-63)
   - When band name is unknown or band spec is `False` (lines 82-92)
