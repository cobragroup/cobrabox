# Feature Review: coherence

**File**: `src/cobrabox/features/coherence.py`
**Date**: 2026-03-04
**Verdict**: PASS

## Summary

The `Coherence` feature is a high-quality implementation that correctly computes
magnitude-squared coherence between all pairwise channel combinations using Welch's
method. It follows all cobrabox conventions: proper dataclass structure, comprehensive
docstring with all required sections, full type annotations, and robust input validation.
The code is clean, well-structured, and ready for production use.

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

1 file already formatted

## Signature & Structure

- ✅ `from __future__ import annotations` present (line 1)
- ✅ `@dataclass` decorator applied (line 14)
- ✅ Correct base class `BaseFeature[SignalData]` (line 15) — appropriate since it requires time dimension
- ✅ `output_type: ClassVar[type[Data]] = Data` correctly set (line 46) — coherence removes time dimension
- ✅ Class name `Coherence` matches filename
- ✅ `__call__` signature correct: `def __call__(self, data: SignalData) -> xr.DataArray:` (line 90)
- ✅ Does not implement `apply()` — uses inherited implementation
- ✅ Import order correct: **future**, stdlib, third-party, internal
- ✅ No unused imports

## Docstring

Excellent Google-style docstring with all required sections:

- ✅ One-line summary: "Compute magnitude-squared coherence for all pairwise channel combinations."
- ✅ Extended description explains Welch's method, 50% overlap, Hann window, symmetry handling
- ✅ Args section documents `nperseg` parameter with types and constraints (lines 28-30)
- ✅ Returns section comprehensively describes output shape, dimensions, and values (lines 38-43)
- ✅ Example section shows typical `.apply()` usage (lines 32-36)

## Typing

- ✅ Field `nperseg: int | None` properly typed (line 48)
- ✅ `__call__` return type `xr.DataArray` explicit (line 90)
- ✅ No bare `Any` types

## Safety & Style

- ✅ No `print()` statements
- ✅ Input validation in `__call__` (lines 93-109):
  - Checks 'space' dimension exists
  - Validates at least 2 spatial channels
  - Validates nperseg constraints
- ✅ `__post_init__` validates `nperseg >= 2` (lines 50-52)
- ✅ No mutation of input `data` — works on `data.data` and returns new array
- ✅ Uses numpy vectorization appropriately in `_mean_squared_coherence`

## Action List

None.
