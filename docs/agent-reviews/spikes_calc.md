# Feature Review: spikes_calc

**File**: `src/cobrabox/features/spikes_calc.py`
**Date**: 2026-03-04
**Verdict**: PASS

## Summary

The SpikesCalc feature implements IQR-based outlier detection and returns a (1, 1) array with the spike count. All review issues have been fixed. The feature now has complete docstring sections and input validation for empty data.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

Line 12: `@dataclass` decorator correctly applied.
Line 13: Inherits `BaseFeature[Data]` — appropriate for a dimension-agnostic feature.
Line 30: `__call__` signature is correct: `def __call__(self, data: Data) -> xr.DataArray`.
Line 50: Returns `xr.DataArray` with dims `["time", "space"]` as documented.

Structure is compliant. The class does not implement `apply()` — correctly inherits from BaseFeature.

## Docstring

The docstring at lines 14-28 now includes:

- One-line summary (line 14)
- Extended description (lines 16-17)
- Args section documenting no parameters (lines 19-20)
- Returns section describing output shape and contents (lines 22-24)
- Example section showing usage (lines 26-27)

All required sections are present and complete.

## Typing

All typing is correct:

- No dataclass fields to type (feature takes no parameters).
- `__call__` has explicit parameter type `Data` and return type `xr.DataArray`.
- Uses `data.data.values` for type-safe access to underlying numpy array.
- No bare `Any` types.

## Safety & Style

- No print statements: Clean.
- Input validation: Added check for empty data at line 33-34, raises ValueError with clear message.
- No mutation: The feature correctly works on a copy and returns a new array without modifying input.

## Action List

None.
