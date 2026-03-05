# Feature Review: sliding_window_reduce

**File**: `src/cobrabox/features/sliding_window_reduce.py`
**Date**: 2026-03-05
**Verdict**: PASS

## Summary

SlidingWindowReduce is a well-implemented feature that combines sliding window operations with aggregation into a single step. The implementation is clean, follows all cobrabox conventions, and uses xarray's `rolling()` API efficiently. The feature correctly outputs `Data` (not `SignalData`) since the time dimension is reduced, and includes proper input validation for all parameters.

## Ruff

### `uvx ruff check`
Clean — no issues found.

### `uvx ruff format --check`
Clean — no formatting issues.

## Signature & Structure

**Line 1**: ✅ Has `from __future__ import annotations` as first import.

**Lines 13-14**: ✅ Correctly decorated with `@dataclass` and inherits `BaseFeature[SignalData]` (time-series feature that operates on the time dimension).

**Line 49**: ✅ Properly sets `output_type: ClassVar[type[Data]] = Data` since the feature removes the time dimension.

**Line 44-47**: ✅ All fields properly typed with `field()` defaults.

**Line 60**: ✅ Correct `__call__` signature: `def __call__(self, data: SignalData) -> xr.DataArray:`

**Lines 51-58**: ✅ Has `__post_init__` validation for all numeric constraints.

**Imports (lines 1-10)**: ✅ Clean import structure following convention:
1. `from __future__ import annotations`
2. stdlib (`dataclasses`, `typing`)
3. third-party (`numpy`, `xarray`)
4. internal (`..base_feature`, `..data`)

## Docstring

**Lines 14-42**: ✅ Complete Google-style docstring with all required sections:
- One-line summary (line 15)
- Extended description (lines 17-21)
- `Args:` section documenting all four fields (lines 23-28)
- `Returns:` section describing output shape and dimensions (lines 30-33)
- `Example:` section with working code snippet (lines 35-41)

## Typing

- ✅ All fields have explicit type annotations
- ✅ `__call__` return type is `xr.DataArray` (correct for `BaseFeature`)
- ✅ Uses `Literal` type for `agg` parameter to restrict valid values
- ✅ No bare `Any` types

## Safety & Style

- ✅ No `print()` statements
- ✅ Input validation in `__post_init__` for `window_size`, `step_size`, and `agg`
- ✅ Input validation in `__call__` for dimension existence and window size bounds
- ✅ No mutation of input `data` — works on `data.data` and returns new arrays
- ✅ Uses xarray's built-in `rolling()` and aggregation methods (no manual loops)
- ✅ Line length within 100 characters

## Action List

None.
