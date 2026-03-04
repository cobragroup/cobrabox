# Feature Review: dummy

**File**: `src/cobrabox/features/dummy.py`
**Date**: 2026-03-04
**Verdict**: NEEDS WORK

## Summary

The Dummy feature has been significantly improved since the previous review. It now has `from __future__ import annotations`, a complete docstring with all required sections, and input validation for required dimensions. However, it still has typing issues: missing type parameter on `BaseFeature` inheritance and incorrect return type annotation on `__call__`. The dataclass fields also lack validation in `__post_init__`.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

Line 10: Inherits `BaseFeature` without type parameter. Should be `BaseFeature[Data]` or `BaseFeature[SignalData]` depending on whether time dimension is required. Since the `__call__` validates for "time" dimension (lines 36-37), this should likely be `BaseFeature[SignalData]`.

Line 35: `__call__` return type is `-> Data` but per criteria should be `-> xr.DataArray | Data` to match the base class contract.

Line 35: `data` parameter is typed as `Data` which is consistent with the untyped base class, but should be `SignalData` if the type parameter is fixed.

The class correctly uses `@dataclass` decorator and does not implement `apply()` (inherits from base). Class name matches filename.

## Docstring

Complete Google-style docstring with all required sections:

- One-line summary (line 11)
- Extended description (lines 13-17)
- Args section (lines 19-21)
- Returns section (lines 23-26)
- Example section (lines 28-29)

The docstring appropriately warns this is a "negative reference" and explicitly tells users "Do not use this as a template for new features."

## Typing

Lines 32-33: All fields are properly typed (`mandatory_arg: int`, `optional_arg: int = 0`).

Line 35: Return type annotation is `-> Data` but should be `-> xr.DataArray | Data` per the base class contract.

Missing import for `SignalData` if the type parameter is to be corrected (currently only imports `Data`).

## Safety & Style

Lines 36-39: Has input validation for required dimensions ("time" and "space"), raising `ValueError` with clear messages. Good.

Lines 40-42: Correctly does not mutate input `data`. Returns a new `Data` object via `Data.from_numpy()`.

No `print()` statements found.

Missing `__post_init__` validation for dataclass fields. The docstring describes `mandatory_arg` as "Required integer argument" but there's no validation that it's actually provided or valid.

## Action List

1. [Severity: MEDIUM] Add type parameter to base class: Change `class Dummy(BaseFeature):` to `class Dummy(BaseFeature[SignalData]):` (line 10) and add `SignalData` to imports.

2. [Severity: MEDIUM] Fix `__call__` return type annotation from `-> Data` to `-> xr.DataArray | Data` (line 35).

3. [Severity: MEDIUM] Update `__call__` parameter type from `data: Data` to `data: SignalData` (line 35) to match the fixed base class type parameter.

4. [Severity: LOW] Add `__post_init__` method to validate `mandatory_arg` if there are constraints (e.g., must be positive, non-zero, etc.). If truly any integer is valid, document that explicitly.
