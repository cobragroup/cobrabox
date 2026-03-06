# Feature Review: sliding_window_reduce

**File**: `src/cobrabox/features/sliding_window_reduce.py`
**Date**: 2026-03-06
**Verdict**: PASS

## Summary

Well-implemented feature that combines sliding window creation with aggregation into a single operation. Code is clean, properly typed, and follows all structural conventions. The only minor issue is a missing `Raises:` section in the docstring.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

All structural requirements met:

- Line 1: `from __future__ import annotations` present
- Line 12: `@dataclass` decorator applied
- Line 13: Correctly inherits `BaseFeature[SignalData]` (requires time dimension)
- Line 48: `output_type: ClassVar[type[Data]] = Data` correctly declared — feature removes time dimension
- Lines 43-46: All four fields properly declared as dataclass fields
- Line 59: `__call__` signature correct: `def __call__(self, data: SignalData) -> xr.DataArray:`
- No `apply()` override — correctly inherits from base class

## Docstring

Google-style docstring with most required sections present:

- One-line summary (line 14)
- Extended description explaining purpose (lines 16-20)
- `Args:` section for all four fields (lines 22-27)
- `Returns:` section describing output shape (lines 29-40)
- `Example:` section with working code (lines 34-40)
- Missing `Raises:` section

The feature raises `ValueError` in `__post_init__` (lines 51-57) for invalid parameter values and in `__call__` (lines 62-69) for dimension errors. These should be documented.

## Typing

Excellent typing coverage:

- Line 43: `window_size: int`
- Line 44: `step_size: int`
- Line 45: `dim: str`
- Line 46: `agg: Literal["mean", "std", "sum", "min", "max"]` — excellent use of Literal for constrained strings
- Line 48: `output_type: ClassVar[type[Data]]`
- Line 50: `__post_init__(self) -> None`
- Line 59: `__call__(self, data: SignalData) -> xr.DataArray`

No bare `Any` types found.

## Safety & Style

Clean implementation with proper safety measures:

- No `print()` statements
- `__post_init__` validates `window_size >= 1`, `step_size >= 1`, and `agg` in valid set
- `__call__` validates `dim` exists in data and `window_size <= n_dim`
- No mutation of input `data` — works on `data.data` and returns new array
- Line 81: Returns `indexed.rename({self.dim: "window"})` — creates new array, no mutation

## Action List

1. [Severity: LOW] Add `Raises:` section to docstring documenting the three ValueError conditions:
   - Invalid `window_size` or `step_size` in `__post_init__` (lines 51-54)
   - Invalid `agg` value in `__post_init__` (lines 55-57)
   - Missing dimension or window too large in `__call__` (lines 62-69)
