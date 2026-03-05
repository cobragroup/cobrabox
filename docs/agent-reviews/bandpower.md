# Feature Review: Bandpower

**File**: `src/cobrabox/features/bandpower.py`
**Date**: 2025-03-05
**Verdict**: PASS

## Summary

Excellent, production-quality feature. Bandpower is a well-implemented spectral analysis feature using Welch's method. The code is clean, properly typed, thoroughly documented, and follows all cobrabox conventions. It includes thoughtful features like default frequency bands, flexible band specification syntax, and robust input validation.

## Ruff

### `uvx ruff check`
All checks passed!

### `uvx ruff format --check`
1 file already formatted

## Signature & Structure

- **Line 1**: `from __future__ import annotations` present ✓
- **Line 21**: `@dataclass` decorator applied ✓
- **Line 22**: Correctly inherits `BaseFeature[SignalData]` (time-series operation) ✓
- Class name `Bandpower` matches filename `bandpower.py` ✓
- **Line 65**: `__call__` signature correct: `def __call__(self, data: SignalData) -> xr.DataArray` ✓
- No `apply()` override — correctly inherits from `BaseFeature` ✓
- Imports are minimal and properly ordered (stdlib → third-party → internal) ✓

## Docstring

Comprehensive Google-style docstring with all required sections:

- **One-line summary**: Line 23 — clear verb phrase describing the operation
- **Extended description**: Lines 25-28 — explains Welch's method and the integration approach
- **Args section**: Lines 30-44 — both `bands` and `nperseg` documented with types and behavior
  - Excellent documentation of the flexible `bands` parameter (default bands, custom ranges, True for defaults)
  - Lists all five default bands with frequency ranges
- **Returns section**: Lines 51-55 — describes output shape, dimensions, units, and coordinates
- **Example section**: Lines 46-49 — three examples covering default usage, custom bands, and parameter tuning

## Typing

- **Line 58**: `bands: dict[str, list[float] | bool] | None` — precise type for flexible band specification
- **Line 59**: `nperseg: int | None` — optional integer parameter
- **Line 65**: Return type `xr.DataArray` explicitly annotated ✓
- No bare `Any` types ✓
- `__post_init__` return type annotated ✓

## Safety & Style

- **No print statements** ✓
- **Input validation**:
  - **Lines 62-63**: `__post_init__` validates `nperseg >= 2` with clear error message ✓
  - **Lines 68-72**: Validates `sampling_rate` is present (required for frequency calculations) ✓
  - **Lines 79-94**: Comprehensive band specification validation:
    - Handles `True` for default bands with helpful error for unknown band names
    - Rejects `False` with clear message
    - Converts list specs to tuples properly
- **No mutation**: Works on `data.data` (line 66), creates new `xr.DataArray` (line 115), never modifies input ✓
- **Line length**: All lines under 100 characters ✓

## Notable Implementation Details

1. **Default bands dictionary** (lines 12-18): Cleanly defined as module constant `_DEFAULTS`
2. **Flexible band API**: Supports `{"alpha": True}` for defaults and `{"custom": [10, 20]}` for ranges
3. **Welch optimization**: Transposes time to last axis (lines 97-98) for efficient scipy operation
4. **Edge case handling**: Gracefully handles frequency bands with no matching bins (lines 107-108)
5. **Coordinate preservation**: Preserves non-time coordinates from input (lines 116-118)

## Action List

None.
