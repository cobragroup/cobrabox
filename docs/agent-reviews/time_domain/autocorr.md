# Feature Review: Autocorr

**File**: `src/cobrabox/features/time_domain/autocorr.py`
**Date**: 2025-03-24
**Verdict**: PASS

## Summary

Clean, well-structured feature implementing normalized autocorrelation. Follows the dataclass pattern correctly with proper validation, NaN handling, and dimension-agnostic design. Ruff is clean. Minor docstring gap: missing `Raises:` section for documented exceptions.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

Line 1: `from __future__ import annotations` present.

Line 13-14: Correct `@dataclass` + `BaseFeature[Data]` inheritance. The feature is dimension-agnostic (user specifies `dim`), so `BaseFeature[Data]` is appropriate.

Line 44: `output_type: ClassVar[type[Data]] = Data` is correctly set — the feature removes the `dim` dimension entirely, returning a reduced array.

Line 74: `__call__` signature correct: `def __call__(self, data: Data) -> xr.DataArray`.

Lines 52-72: Helper `_acf_numpy` is a `@staticmethod` inside the class — follows the "no loose helpers" rule.

Imports are ordered correctly: `__future__`, stdlib, third-party, internal.

## Docstring

Line 15: Good one-line summary.

Lines 17-20: Extended description includes MATLAB reference for provenance.

Lines 21-22: Clear mutual exclusivity note for `lag_steps`/`lag_ms`.

Lines 24-28: `Args:` section documents all four dataclass fields.

Lines 30-33: `Returns:` section describes shape and semantics.

Line 36: `Example:` section shows `.apply()` usage.

**Missing**: `Raises:` section. The feature raises `ValueError` in `__post_init__` (lines 47-50) for mutually exclusive parameters and invalid `fs`, plus in `__call__` (lines 78, 90) for missing dimension and invalid lag. These should be documented.

**Optional**: `References:` section could cite autocorrelation literature, but this is acceptable to omit for a standard statistical measure.

## Typing

Lines 39-42: All four fields have explicit type annotations.

Line 74: `__call__` has explicit return type `xr.DataArray`.

Line 84: `lag_ms_value` has explicit `float` annotation (though the cast is slightly redundant given the annotation, it is harmless).

No bare `Any` types.

## Safety & Style

No `print()` statements.

Lines 47-50: `__post_init__` validates mutual exclusivity of `lag_steps`/`lag_ms` and that `fs > 0`.

Line 78: Validates that `dim` exists in data.

Lines 88-90: Validates lag is within valid bounds (1 to n-1).

Lines 57-63: Handles NaN values properly in `_acf_numpy` by returning NaN for all-NaN inputs and using `nanmean`/`nan_to_num` for partial NaN handling.

Line 68-69: Guards against division by zero when autocorrelation at lag 0 is zero.

No mutation of input `data` — works on `data.data` and returns new array.

## Action List

1. [Severity: MEDIUM] Add `Raises:` section to docstring documenting the three `ValueError` conditions (lines 47-50, 78, 90).
