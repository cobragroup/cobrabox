# Feature Review: phase_locking_value

**File**: `src/cobrabox/features/connectivity/phase_locking_value.py`
**Date**: 2025-03-24
**Verdict**: PASS

## Summary

The PhaseLockingValue and PhaseLockingValueMatrix features are well-implemented with proper structure, comprehensive docstrings, and good type annotations. Both classes correctly inherit from `BaseFeature[SignalData]`, set `output_type = Data` (since they reduce time dimension), and include all required docstring sections. Ruff passes cleanly. The only issue is a module-level helper function that should be moved inside the class as a staticmethod per project conventions.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

Both `PhaseLockingValue` (lines 33-81) and `PhaseLockingValueMatrix` (lines 84-147) are correctly structured:

- ✅ `from __future__ import annotations` at line 1
- ✅ `@dataclass` decorator on both classes
- ✅ Correct base class: `BaseFeature[SignalData]` (line 34, 85)
- ✅ Correct `output_type: ClassVar[type[Data]] = Data` (lines 55, 110) — required since output removes time dimension
- ✅ PascalCase class names match filename
- ✅ `__call__` signature: `def __call__(self, data: SignalData) -> xr.DataArray` (lines 60, 114)
- ✅ `apply()` not implemented (correctly inherited)
- ✅ No `_is_cobrabox_feature` marker (correctly inherited from base)

**Issue (line 14-30)**: The helper function `_compute_plv()` is a module-level private function only used by these classes. Per project conventions, loose helpers that are only called by the feature class should be moved inside as `@staticmethod` methods to keep the module's public surface clean.

## Docstring

Both classes have complete Google-style docstrings with all required sections:

- ✅ One-line summary
- ✅ Extended description explaining the algorithm
- ✅ `Args:` section documenting all dataclass fields
- ✅ `Returns:` section describing output shape and dimensions
- ✅ `Raises:` section listing ValueError conditions
- ✅ `Example:` section with working usage

**Suggestion**: Consider adding a `References:` section citing the original PLV paper (Lachaux et al., 1999, *Human Brain Mapping*).

## Typing

- ✅ All fields typed: `coord_x: str | int`, `coord_y: str | int`, `coords: list[str] | list[int] | None = None`
- ✅ `__call__` return type: `-> xr.DataArray`
- ✅ No bare `Any` types
- ✅ Proper use of `ClassVar` for `output_type`

## Safety & Style

- ✅ No `print()` statements
- ✅ Input validation present: checks for `space` dimension (lines 64-65, 118-119) and valid coordinates (lines 69-76, 130-132)
- ✅ No mutation of input `data` — works on `data.data` and returns new arrays
- ✅ Line length within 100 characters

## Action List

1. [Severity: MEDIUM] Move `_compute_plv()` from module-level (lines 14-30) to a `@staticmethod` inside `PhaseLockingValue` class. Update the call in `PhaseLockingValueMatrix` (line 141) to use `PhaseLockingValue._compute_plv()` or duplicate it as a staticmethod in both classes.

2. [Severity: LOW] Add `References:` section to both class docstrings citing: Lachaux, J.P., et al. (1999). Measuring phase synchrony in brain signals. *Human Brain Mapping*, 8(4), 194-208.
