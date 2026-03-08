# Feature Review: partial_correlation

**File**: `src/cobrabox/features/partial_correlation.py`
**Date**: 2026-03-05
**Verdict**: PASS

## Summary

This file contains two well-implemented feature classes (`PartialCorrelation` and `PartialCorrelationMatrix`) that compute partial correlation coefficients while controlling for confounding variables. The code is clean, properly typed, and follows all established conventions. Both features inherit from `BaseFeature[SignalData]`, correctly specify `output_type = Data` (since they remove the time dimension), and include comprehensive input validation. Ruff passes cleanly with no issues.

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

1 file already formatted

## Signature & Structure

Both classes correctly follow the pattern:

- ✅ `@dataclass` decorator present
- ✅ Inherit `BaseFeature[SignalData]` (line 54, 133)
- ✅ `output_type: ClassVar[type[Data]] = Data` set correctly (line 78, 157)
- ✅ Class names are PascalCase matching filename
- ✅ `__call__` signature correct: `def __call__(self, data: SignalData) -> xr.DataArray`
- ✅ No `apply()` override (correctly inherited)
- ✅ `from __future__ import annotations` is first import (line 1)
- ✅ Clean import order: stdlib → third-party → internal

## Docstring

Both classes have complete Google-style docstrings:

- ✅ One-line summary present and descriptive
- ✅ Extended description explains the algorithm
- ✅ Args section documents all dataclass fields
- ✅ Returns section describes output shape and dimensions
- ✅ Raises section documents ValueError conditions
- ✅ Example section shows `.apply()` usage

## Typing

- ✅ All dataclass fields typed: `coord_x: str | int`, `coord_y: str | int`, `control_vars: list[str] | list[int]`
- ✅ `__call__` return type: `xr.DataArray`
- ✅ Helper function `_compute_partial_correlation` fully typed
- ✅ No bare `Any` types

## Safety & Style

- ✅ No print statements
- ✅ Input validation present and thorough:
  - Space dimension existence check (lines 88-89, 166-167)
  - Time dimension check (lines 92-93, 170-171)
  - Coordinate existence in space dimension (lines 97-104, 181-189)
  - Non-empty control_vars validation (lines 106-107, 176-177)
- ✅ Input data is never mutated
- ✅ Proper error handling with `np.linalg.LinAlgError` catch and re-raise as `ValueError` with helpful message (lines 38-46)
- ✅ Handles edge case: returns 1.0 when x and y are allclose (line 31-32)

## Action List

None.
