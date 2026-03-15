# Feature Review: phase_locking_value

**File**: `src/cobrabox/features/phase_locking_value.py`
**Date**: 2026-03-05
**Verdict**: PASS

## Summary

Both `PhaseLockingValue` and `PhaseLockingValueMatrix` are well-structured, compliant features. They correctly inherit from `BaseFeature[SignalData]`, set `output_type = Data` (scalar/matrix output removes time dimension), include complete Google-style docstrings with all required sections, and validate inputs thoroughly. The helper function `_compute_plv` is cleanly factored out at module level and shared by both classes. No issues found.

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

1 file already formatted

## Signature & Structure

**PhaseLockingValue (lines 33-81):**

- ✅ `@dataclass` decorator present (line 33)
- ✅ Correctly inherits `BaseFeature[SignalData]` (line 34)
- ✅ `output_type: ClassVar[type[Data]] = Data` correctly set (line 55) — scalar output removes time dimension
- ✅ `__call__` signature correct: `def __call__(self, data: SignalData) -> xr.DataArray` (line 60)
- ✅ Uses inherited `apply()` — no override

**PhaseLockingValueMatrix (lines 84-139):**

- ✅ `@dataclass` decorator present (line 84)
- ✅ Correctly inherits `BaseFeature[SignalData]` (line 85)
- ✅ `output_type: ClassVar[type[Data]] = Data` correctly set (line 106) — matrix output removes time dimension
- ✅ `__call__` signature correct: `def __call__(self, data: SignalData) -> xr.DataArray` (line 110)

**Helper function `_compute_plv` (lines 14-30):**

- ✅ Cleanly separated as module-level private function
- ✅ Shared by both classes (DRY principle followed)
- ✅ Has its own docstring with Args/Returns

**Imports (lines 1-11):**

- ✅ `from __future__ import annotations` first (line 1)
- ✅ Correct order: stdlib → third-party → internal
- ✅ All imports used

## Docstring

**PhaseLockingValue:**

- ✅ One-line summary: "Compute phase locking value (PLV) between two coordinates."
- ✅ Extended description explains space/time dimensions and PLV meaning
- ✅ `Args:` section documents `coord_x` and `coord_y`
- ✅ `Returns:` describes scalar DataArray
- ✅ `Raises:` documents ValueError cases (correctly omits time dimension claim)
- ✅ `Example:` shows `.apply()` usage

**PhaseLockingValueMatrix:**

- ✅ One-line summary: "Compute pairwise phase locking value matrix for multiple coordinates."
- ✅ Extended description explains pairwise computation
- ✅ `Args:` section documents `coords`
- ✅ `Returns:` describes DataArray with dims `(coord_i, coord_j)`
- ✅ `Raises:` documents both ValueError cases
- ✅ `Example:` shows `.apply()` usage

## Typing

**PhaseLockingValue:**

- ✅ Fields typed: `coord_x: str | int`, `coord_y: str | int` (lines 57-58)
- ✅ `__call__` return type: `xr.DataArray` (line 60)
- ✅ No bare `Any`

**PhaseLockingValueMatrix:**

- ✅ Fields typed: `coords: list[str] | list[int]` (line 108)
- ✅ `__call__` return type: `xr.DataArray` (line 110)
- ✅ No bare `Any`

## Safety & Style

**PhaseLockingValue:**

- ✅ No `print()` statements
- ✅ Input validation (lines 64-76):
  - Checks `space` dimension exists
  - Validates both coordinates exist in space dimension
- ✅ No mutation of input `data` — works on `data.data` and returns new array

**PhaseLockingValueMatrix:**

- ✅ No `print()` statements
- ✅ Input validation (lines 114-124):
  - Checks `space` dimension exists
  - Validates `coords` is not empty
  - Validates all coordinates exist in space dimension
- ✅ No mutation of input `data`

**Algorithm considerations:**

- Line 30: `_compute_plv` uses `hilbert()` from scipy.signal — appropriate for phase extraction
- Line 27-28: Early return of 1.0 when x and y are identical (optimization)
- Lines 127-133: Matrix version uses nested loops; O(n²) is acceptable for typical space dimension sizes

## Action List

None.
