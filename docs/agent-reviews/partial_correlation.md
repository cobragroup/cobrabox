# Feature Review: partial_correlation

**File**: `src/cobrabox/features/partial_correlation.py`
**Date**: 2025-03-04
**Verdict**: PASS

## Summary

The `PartialCorrelation` and `PartialCorrelationMatrix` features are well-migrated to the class-based pattern. Both classes properly inherit `BaseFeature[SignalData]`, have complete docstrings, and include the helper method `_compute_partial_correlation` as a class method. The implementation correctly computes partial correlations controlling for specified variables.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

**Line 1**: ✅ `from __future__ import annotations` present as first import.

**Lines 12, 99**: ✅ Both classes use `@dataclass` decorator.

**Lines 13, 100**: ✅ Both correctly inherit `BaseFeature[SignalData]`.

**Lines 13, 100**: ✅ Class names `PartialCorrelation` and `PartialCorrelationMatrix` match filename.

**Lines 35, 122**: ✅ `__call__` signatures correct:

- `data: SignalData` as argument
- Return type `xr.DataArray`
- No `apply()` implementation (inherited)

**Lines 1-9**: ✅ Imports in correct order with no unused imports.

## Docstring

**Lines 14-33** (`PartialCorrelation`): ✅ Complete docstring:

- One-line summary (line 14)
- Extended description (lines 16-17)
- `Args:` section for all three fields (lines 19-22)
- `Returns:` section (lines 24-25)
- `Raises:` section (lines 27-29)
- `Example:` section with `.apply()` usage (lines 31-33)

**Lines 101-122** (`PartialCorrelationMatrix`): ✅ Complete docstring with same sections.

## Typing

**Lines 35-37** (`PartialCorrelation`): ✅ All fields typed:

- `coord_x: str`
- `coord_y: str`
- `control_vars: list[str]`

**Lines 122-123** (`PartialCorrelationMatrix`): ✅ All fields typed:

- `coords: list[str]`
- `control_vars: list[str]`

**Lines 39-42, 126-129**: ✅ Helper method `_compute_partial_correlation` properly typed with return type `float`.

No bare `Any` types.

## Safety & Style

**Lines 39-65, 126-152**: ✅ Helper method `_compute_partial_correlation` properly implemented as a class method with `self` parameter.

**Lines 67-97, 154-194**: ✅ Input validation present in both classes:

- Validates presence of `space` dimension
- Validates presence of `time` dimension
- Validates coordinates exist in space dimension
- Validates `control_vars` is not empty

**Lines 67-97, 154-194**: ✅ No mutation of input `data`.

**Lines 39-65**: ✅ No `print()` statements.

## Action List

None.
