# Feature Review: autocorr

**File**: `src/cobrabox/features/autocorr.py`
**Date**: 2026-03-06
**Verdict**: PASS

## Summary

The Autocorr feature is well-implemented with clean code structure, proper type annotations, and comprehensive input validation. The feature correctly uses `BaseFeature[Data]` since it operates on any dimension (not requiring a time dimension specifically). The algorithm follows the MATLAB reference implementation with appropriate NaN handling. The only minor issue is a missing `Raises:` section in the docstring.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

All structural requirements are met:

- ✅ `from __future__ import annotations` at line 1
- ✅ `@dataclass` decorator with `BaseFeature[Data]` inheritance (line 14)
- ✅ Class name `Autocorr` matches filename `autocorr.py`
- ✅ `output_type: ClassVar[type[Data]] = Data` correctly set (line 44) — necessary since the dimension is removed
- ✅ `__call__` signature correct: `(self, data: Data) -> xr.DataArray` (line 74)
- ✅ No `apply()` override (inherits from base)
- ✅ Clean imports in correct order

The choice of `BaseFeature[Data]` is appropriate — the feature operates on any user-specified dimension, not just time series data.

## Docstring

Good coverage with Google-style formatting:

- ✅ One-line summary describes the computation
- ✅ Extended description includes MATLAB reference (lines 18-19)
- ✅ Args section documents all four fields (dim, fs, lag_steps, lag_ms)
- ✅ Returns section describes the output shape and meaning (lines 30-33)
- ✅ Example shows proper `.apply()` usage (line 36)
- ❌ **Missing Raises section** — the feature validates multiple conditions but does not document them

**Recommendation**: Add a Raises section documenting:

- `ValueError`: If both `lag_steps` and `lag_ms` are specified
- `ValueError`: If `fs` is not positive
- `ValueError`: If `dim` is not found in data dimensions
- `ValueError`: If lag is out of valid range (1 to n-1)

## Typing

Excellent type coverage:

- ✅ All dataclass fields typed (lines 39-42)
- ✅ `__call__` return type: `xr.DataArray` (line 74)
- ✅ `@staticmethod` helper has return type: `-> float` (line 53)
- ✅ Local variable `lag_ms_value` explicitly typed as `float` (line 84)
- ✅ No bare `Any` types

## Safety & Style

- ✅ No `print()` statements
- ✅ Input validation in `__post_init__` (lines 46-50) validates mutual exclusivity of lag parameters and fs > 0
- ✅ Input validation in `__call__` (lines 77-78, 89-90) validates dimension existence and lag range
- ✅ No mutation of input data — operates on `data.data` and returns new array via `xr.apply_ufunc`
- ✅ Proper NaN handling in `_acf_numpy` (lines 57-63)
- ✅ Division by zero protection (line 68-69)

## Action List

1. [Severity: MEDIUM] Add `Raises:` section to docstring documenting all ValueError conditions raised in `__post_init__` (lines 47-50) and `__call__` (lines 78, 90).
