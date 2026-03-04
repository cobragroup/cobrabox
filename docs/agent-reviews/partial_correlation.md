# Feature Review: partial_correlation

**File**: `src/cobrabox/features/partial_correlation.py`
**Date**: 2026-03-04
**Verdict**: PASS

## Summary

Both `PartialCorrelation` and `PartialCorrelationMatrix` features are well-implemented with complete docstrings, proper typing, and comprehensive input validation. The code follows all structural conventions and passes ruff checks. Both classes correctly declare `output_type = Data` since they produce correlation outputs without meaningful time dimensions.

## Ruff

### `uvx ruff check`
Clean — no issues found.

### `uvx ruff format --check`
Clean — no formatting issues.

## Signature & Structure

Both classes are correctly structured:
- `from __future__ import annotations` present at line 1
- `@dataclass` decorator with `BaseFeature[SignalData]` inheritance (lines 13-14, 122-123)
- `output_type: ClassVar[type[Data]] = Data` set appropriately on both classes (lines 38, 147)
- Class names match filename convention
- `__call__(self, data: SignalData) -> xr.DataArray` signatures correct (lines 74, 182)
- No `apply()` override — inherits correctly from base
- Clean imports in proper order

## Docstring

Both features have complete Google-style docstrings:
- One-line summary describing purpose
- Extended description explaining behavior
- Args section documenting all fields (`coord_x`, `coord_y`, `control_vars` for `PartialCorrelation`; `coords`, `control_vars` for `PartialCorrelationMatrix`)
- Returns section describing output shape and type
- Raises section documenting `ValueError` conditions
- Example section showing `.apply()` usage (lines 32-35, 141-144)

## Typing

All type annotations are present and correct:
- All dataclass fields have explicit types (`str`, `list[str]`)
- `__call__` return type is `xr.DataArray`
- No bare `Any` types
- Helper method `_compute_partial_correlation` fully typed with `np.ndarray`, `list[np.ndarray]`, and `float` annotations

## Safety & Style

- No `print()` statements
- Comprehensive input validation in `__call__`:
  - Checks for `space` dimension existence (lines 78-79, 186-187)
  - Checks for `time` dimension (lines 82-83, 190-191)
  - Validates coordinate existence in space dimension (lines 87-103, 201-209)
  - Validates non-empty `control_vars` and `coords` (lines 96-97, 193-197)
- No mutation of input `data` — works on `data.data` and returns new arrays

## Action List

None.
