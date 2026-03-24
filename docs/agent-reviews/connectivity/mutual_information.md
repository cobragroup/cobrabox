# Feature Review: MutualInformation

**File**: `src/cobrabox/features/connectivity/mutual_information.py`
**Date**: 2026-03-24
**Verdict**: PASS

## Summary

Excellent implementation of mutual information computation. The feature is well-structured with comprehensive docstring including mathematical background, proper type annotations, thorough input validation, and follows all cobrabox conventions. The code handles both 2D and higher-dimensional data with proper dimension inference.

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

1 file already formatted

## Signature & Structure

- Line 1: Correctly has `from __future__ import annotations`
- Line 13: Uses `@dataclass` decorator
- Line 14: Inherits `BaseFeature[SignalData]` — correct since MI operates on time-series
- Line 76: Sets `output_type: ClassVar[type[Data] | None] = Data` — correct since output removes time dimension
- Line 89: `__call__` signature correctly takes `data: SignalData` and returns `xr.DataArray`
- Lines 109-154: All helper methods are instance methods (no loose module-level functions)
- Imports are in correct order: future, stdlib, third-party, internal

## Docstring

Excellent Google-style docstring with all required sections:

- **One-line summary** (line 16-17): Clear and descriptive
- **Extended description** (lines 19-26): Explains MI formula and binning strategies
- **Args** (lines 28-37): All 5 fields documented with types and descriptions
- **Returns** (lines 39-42): Describes output dimensions ("space_from", "space_to")
- **Raises** (lines 44-50): Lists 6 ValueError conditions
- **References** (lines 52-54): Cites Shannon 1948 (the original source)
- **Example** (lines 56-66): Working doctest with shape verification

## Typing

- All fields typed: `dim: str`, `other_dim: str | None`, `bins: int | None`, `equiprobable_bins: bool`, `log_base: float`
- `__call__` return type: `xr.DataArray` (implicitly included in union)
- No bare `Any` types
- Line 76: Proper `ClassVar[type[Data] | None]` for output_type

## Safety & Style

- No print statements found
- Input validation in `__post_init__` (lines 78-87): Validates bins is positive int, dim is str, other_dim is str or None
- Input validation in `__call__` (lines 90-103): Checks dimensions exist in data, validates other_dim for >2D
- No mutation of input `data` — creates new xr.DataArray at line 151-154
- Line length within 100 characters

## Action List

None.
