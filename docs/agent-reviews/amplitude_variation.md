# Feature Review: amplitude_variation

**File**: `src/cobrabox/features/amplitude_variation.py`
**Date**: 2025-03-05
**Verdict**: NEEDS WORK

## Summary

The feature implements amplitude variation (standard deviation over time) correctly and follows most conventions. However, there's a critical naming mismatch: the class is named `AmpVar` but the file is `amplitude_variation.py`. Per convention, the class name must be `AmplitudeVariation` to match the filename. All other aspects are clean—ruff passes, docstring is complete, typing is correct, and the implementation properly uses `SignalData` with `output_type=Data` since the time dimension is removed.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

Line 13: Class is named `AmpVar` but filename is `amplitude_variation.py`. Convention requires PascalCase matching filename: `AmplitudeVariation`.

All other structural elements are correct:

- `from __future__ import annotations` present (line 1)
- `@dataclass` decorator present (line 12)
- Correct base class `BaseFeature[SignalData]` (line 13)
- `output_type: ClassVar[type[Data]] = Data` correctly set (line 32) since time dimension is removed
- `__call__` signature correct: `def __call__(self, data: SignalData) -> xr.DataArray:` (line 34)
- No `apply()` override — correctly inherited
- Imports in correct order, all used

## Docstring

Complete Google-style docstring with all required sections:

- One-line summary (line 14)
- Extended description (lines 16-17)
- `Args:` section present, correctly states "None" since feature has no fields (lines 19-20)
- `Returns:` section describes shape and units (lines 22-26)
- `Example:` section shows `.apply()` usage (lines 28-29)

## Typing

All typing is correct:

- No dataclass fields to type (feature is parameterless)
- `__call__` has correct return type `xr.DataArray` (line 34)
- `output_type` has correct `ClassVar[type[Data]]` annotation (line 32)
- No bare `Any` types

## Safety & Style

- No `print()` statements
- No input validation needed — `SignalData` enforces time dimension at construction
- No mutation of input data — returns new `xr.DataArray` from `.std()` operation

## Action List

1. [Severity: HIGH] Rename class from `AmpVar` to `AmplitudeVariation` to match filename (line 13).
