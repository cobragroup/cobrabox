# Feature Review: sliding_window_reduce

**File**: `src/cobrabox/features/sliding_window_reduce.py`
**Date**: 2026-03-05
**Verdict**: PASS

## Summary

The `SlidingWindowReduce` feature is well-written and compliant with all cobrabox standards. It correctly implements a `BaseFeature[SignalData]` that combines sliding window creation with aggregation into a single feature. The implementation uses xarray's rolling window for efficient computation, includes comprehensive input validation, and follows all documentation and typing requirements.

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

1 file already formatted

## Signature & Structure

The feature follows all structural requirements:

- ✅ `from __future__ import annotations` on line 1
- ✅ `@dataclass` decorator applied correctly (line 12)
- ✅ Inherits from `BaseFeature[SignalData]` (line 13) - appropriate since it requires the time dimension
- ✅ Class name `SlidingWindowReduce` matches filename (PascalCase conversion)
- ✅ `output_type: ClassVar[type[Data]] = Data` correctly set (line 48) since the feature removes the time dimension
- ✅ `__call__` signature is correct: `def __call__(self, data: SignalData) -> xr.DataArray:` (line 59)
- ✅ `apply()` is inherited from base class, not reimplemented
- ✅ Imports are in correct order (stdlib, third-party, internal) and only import what's used

## Docstring

Excellent Google-style docstring with all required sections:

- ✅ One-line summary: "Sliding window with automatic per-window reduction."
- ✅ Extended description explaining the purpose and benefit (combines windowing + aggregation without needing a Chord)
- ✅ Args section documents all four dataclass fields with types and constraints:
  - `window_size`: Number of samples per window. Must be >= 1.
  - `step_size`: Step between window starts in samples. Must be >= 1.
  - `dim`: Name of the dimension to window over and reduce (default: "time").
  - `agg`: Aggregation function to apply to each window (mean, std, sum, min, max).
- ✅ Returns section describes the output shape and dimension changes
- ✅ Example section shows practical usage with `.apply()` and explains the resulting dimensions

## Typing

All typing requirements are met:

- ✅ All dataclass fields have explicit type annotations:
  - `window_size: int`
  - `step_size: int`
  - `dim: str`
  - `agg: Literal["mean", "std", "sum", "min", "max"]`
- ✅ `__call__` return type is `xr.DataArray` (correct for `BaseFeature`)
- ✅ `output_type` is properly typed as `ClassVar[type[Data]]`
- ✅ No bare `Any` types

## Safety & Style

Strong safety practices throughout:

- ✅ No `print()` statements
- ✅ Validation in `__post_init__` (lines 50-57):
  - `window_size >= 1`
  - `step_size >= 1`
  - `agg` is one of valid options
- ✅ Validation in `__call__` (lines 62-69):
  - Checks that `dim` exists in data dimensions
  - Checks that `window_size <= n_dim` (prevents windows larger than data)
- ✅ No mutation of input `data` - works on `data.data` and returns new array
- ✅ Clean use of xarray's rolling window API for efficient computation

The implementation efficiently uses xarray's rolling construct (line 72) with the aggregation function accessed via `getattr` (line 73), then selects valid window positions (line 78) and renames the dimension (line 81).

## Action List

None.
